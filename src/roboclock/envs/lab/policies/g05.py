"""Galaxea G0.5 (RoboDojo sim checkpoint) over XPolicyLab, driving the dual-X5 arms.

The checkpoint ``OpenGalaxea/g05-robodojo`` (``g05_robodojo_fm_only_checkpoint.tar``) is served
by XPolicyLab's ``policy/G05`` adapter. It reads one observation (three cameras, the 14-value
RoboDojo joint state) and answers with RoboDojo ``joint`` actions: the model predicts 32 steps
by flow matching, and the adapter returns the first 16 (``deploy.yml`` ``action_steps: 16``).
XPolicyLab's ``policy/G05/deploy.py`` executes all 16 before it asks again, so this client does
the same.

Timing. Each chunk entry is one RoboDojo simulator step, 1/25 s (``env_cfg/arx_x5.yml``
``collect_freq: 25``; one ``take_action`` runs ``collect_interval`` physics steps). Run the
evaluator at that rate (``roboclock eval --policy-hz 25``): one act() is one chunk entry, and the
evaluator ramps it over two 50 Hz control steps. The client raises an error at any other rate.
``deploy.yml``'s ``frequency: 30`` is only the adapter's fallback for an observation without
``additional_info.frequency``; RoboDojo's observation manager always sends its ``collect_freq``
(25), so this client sends 25 as well. The value is G0.5's control-frequency input, which only
its AR action tokenizer reads; it does not change the replay rate.

State. No observation history and no server-side memory: the checkpoint has
``num_obs_steps: 1`` and the adapter's ``ROBODOJO_G05_OBS_OFFSETS`` default is ``0``, so
deploy.py's update_obs calls between chunk entries only overwrite the observation the next
get_action reads. The client therefore sends one observation per request, with the ``infer``
message (update_obs + get_action under the server's model lock), and several episodes may share
one server. The gripper state is RoboDojo's: the last commanded opening clipped to [0, 1], and
before the first command the measured opening.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL     ws://host:port of the G0.5 XPolicyLab server
  XPOLICYLAB_COOKIE  code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG     optional JSONL file receiving one row per request
  G05_REPLAN         chunk entries executed per request (default 16, deploy.py's setting)
  G05_FRAME_DIR      optional directory receiving the head JPEG of every request
  XPOLICYLAB_PROMPT_TIME  1: send "<instruction> Time: <sim_time:.1f> s", the prompt of the
                     train45 fine-tune (docs/five-domain-training.md rule 4); default off

See docs/g05.md.
"""

import json
import os
import time
import uuid

import numpy as np

from roboclock.envs.lab.policies.xpolicylab import (
    ARM_DIMS,
    ARM_ORDER,
    CAMERAS,
    HIGH,
    LOW,
    SIDES,
    encode_jpeg,
    lab_action,
    normalise_gripper,
    policy_instruction,
    prompt_time_enabled,
    robodojo_vector,
)
from roboclock.envs.lab.policies.xpolicylab import connection as xpolicylab_connection

# policy/G05/deploy.yml action_steps: the adapter returns 16 of the model's 32 steps.
ACTION_STEPS = 16
# RoboDojo env_cfg/arx_x5.yml collect_freq: one chunk entry per 1/25 s, sent with every
# observation as additional_info.frequency (env/observation_manager/obs_manager.py).
POLICY_HZ = 25.0
HEAD_KEY = dict(CAMERAS)["cam_head"]
# Arm joints of a packed 14-value RoboDojo vector.
ROBODOJO_ARM = np.r_[0:6, 7:13]


def sim_time(observation):
    return float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])


class G05Policy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.replan = int(os.environ.get("G05_REPLAN", ACTION_STEPS))
        if not 0 < self.replan <= ACTION_STEPS:
            raise ValueError(f"G05_REPLAN must be in 1..{ACTION_STEPS}")
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.frame_dir = os.environ.get("G05_FRAME_DIR")
        self.prompt_time = prompt_time_enabled()
        self.warned_cameras = set()

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.inferences = 0
        self.frame = -1
        self.queue = []
        self.previous_time = None
        self.rate_checked = False
        # Read by the evaluator after act(): whether it queried the server, and the server's
        # own compute time for that query (None: nothing to charge).
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        self.gripper = np.clip(normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]]), 0.0, 1.0)
        # deploy.py starts every episode with model.reset().
        self.connection.request("reset", {"trial_id": self.trial_id}, trial_id=self.trial_id)

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = sim_time(observation)
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"G0.5 expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
                    "run roboclock eval with --policy-hz 25"
                )
            self.rate_checked = np.isfinite(step)
        self.previous_time = now

    def act(self, observation):
        self.frame += 1
        self.check_rate(observation)
        self.queried = not self.queue
        if self.queried:
            self.queue = self.infer(observation)
        else:
            self.last_inference_latency = None
        command, target = self.queue.pop(0)
        # RoboDojo reports the commanded opening after clipping it to [0, 1].
        self.gripper = np.clip(command[[6, 13]], 0.0, 1.0)
        return target

    def observation(self, observation):
        """The request RoboDojo's eval client sends for one frame (env/observation_manager)."""
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        head = observation[HEAD_KEY]
        vision = {}
        for camera, key in CAMERAS:
            image = head if key == HEAD_KEY else observation.get(key)
            if image is None:
                # G0.5 needs all three views; a black frame keeps the request valid.
                if camera not in self.warned_cameras:
                    print(f"[g05] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(head)
            vision[camera] = {"color": encode_jpeg(image)}
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            state[f"{side}_arm_joint_state"] = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_ee_joint_state"] = np.asarray(self.gripper[arm : arm + 1], dtype=np.float32)
        return {
            "instruction": policy_instruction(observation, self.prompt_time),
            "env_idx": 0,
            "data_format_version": "v1.0",
            "additional_info": {"frequency": POLICY_HZ},
            "vision": vision,
            "state": state,
        }

    def infer(self, observation):
        # Packing reads the camera images, which renders them (roboclock.envs.observation);
        # build the request first so round_trip_ms is transport plus server time only.
        request = self.observation(observation)
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            path = os.path.join(self.frame_dir, f"head_{self.frame:05d}.jpg")
            with open(path, "wb") as stream:
                stream.write(request["vision"]["cam_head"]["color"])
        start = time.perf_counter()
        reply = self.connection.request(
            "infer", {"observation": request}, trial_id=self.trial_id, step=self.inferences
        )
        round_trip = time.perf_counter() - start
        self.inferences += 1
        server_ms = reply.get("latency_ms")
        self.last_inference_latency = None if server_ms is None else float(server_ms) / 1000.0
        actions = reply["actions"]
        if len(actions) != ACTION_STEPS:
            raise ValueError(f"G0.5 returned {len(actions)} actions, expected {ACTION_STEPS}")
        commands = np.stack([robodojo_vector(action) for action in actions[: self.replan]])
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            joints = np.asarray(observation["joint_positions"], dtype=np.float32)
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": sim_time(observation),
                "server_ms": server_ms,
                "round_trip_ms": round_trip * 1000.0,
                "state_gripper": [round(float(g), 3) for g in self.gripper],
                # Arm joints in RoboDojo order: measured, and the chunk's first and last command.
                "state_arm": [round(float(v), 3) for v in joints[ARM_DIMS]],
                "first_command": [round(float(v), 3) for v in commands[0][ROBODOJO_ARM]],
                "last_command": [round(float(v), 3) for v in commands[-1][ROBODOJO_ARM]],
                "gripper_commands": [[round(float(c[6]), 3), round(float(c[13]), 3)] for c in commands],
                # Gripper targets below 0 m are RoboDojo's squeeze command, not an error.
                "clipped_arm_targets": int(clipped[:, ARM_DIMS].sum()),
                "start_jump_rad": float(np.abs(arm_path[0] - joints[ARM_DIMS]).max()),
                "chunk_travel_rad": float(np.abs(np.diff(arm_path, axis=0)).sum(axis=0).max()),
            }
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return list(zip(commands, np.clip(targets, LOW, HIGH)))


def make():
    return G05Policy()
