"""OpenWAM-alpha (RoboDojo sim checkpoint) over XPolicyLab, driving the dual-X5 arms.

The checkpoint ``OpenWAM/OpenWAM-Alpha-Sim-RoboDojo`` is a world-action model: a Wan2.2-TI2V-5B
video DiT and an ActionDiT joined by self-attention, conditioned on one composite frame (head
camera on top, the two wrist cameras below), the instruction and the current end-effector
state. XPolicyLab's ``policy/OpenWAM`` adapter serves it. Each request answers with 32 steps,
1/25 s apart, of absolute RoboDojo ``ee`` actions per arm: the link6 pose
``[x, y, z, qw, qx, qy, qz]`` in RoboDojo's environment frame plus a gripper opening in [0, 1]
(0 closed). The client replays the whole chunk before it asks again, as the adapter's
``deploy.py`` does with ``replan_steps: null``.

What the server does, so the client does not: it maps both link6 poses into each arm's base
frame (the dual_x5 bases at (-0.30, -0.45, 0.765) and (0.30, -0.45, 0.765), yawed +90 deg),
packs them as EEF20 (xyz, the first two rotation-matrix columns, gripper per arm),
normalises and scatters into its 80-D action space, composes and resizes the three views,
and maps the answer back to RoboDojo-frame poses. The model keeps no history: a request
depends only on the observation it carries, so concurrent episodes may share one server (the
``infer`` message runs update_obs and get_action under one lock).

What the client does: it reports each arm's link6 pose in RoboDojo's environment frame from
the measured joints (``wallx.link6_pose``), the last commanded gripper opening (RoboDojo's
state semantics, clipped to [0, 1]), the three RGB views as XPL-RGB1 JPEGs, and solves the
returned poses to joints with the X5 kinematics, chained along the chunk from the measured
joints (``wallx.solve_chunk``). Images are read, and so rendered, only on request frames.

Run the evaluator at RoboDojo's policy rate (``roboclock eval --policy-hz 25``): one act() is
one chunk entry. The client raises an error at any other rate.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL     ws://host:port of the OpenWAM XPolicyLab server
  XPOLICYLAB_COOKIE  code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG     optional JSONL file receiving one row per request
  OPENWAM_REPLAN     actions executed per chunk (default: the whole 32-step chunk)
  OPENWAM_FRAME_DIR  optional directory receiving the head JPEG of every request

See docs/openwam.md.
"""

import json
import os
import time
import uuid

import numpy as np

from roboclock.envs.lab.policies.wallx import link6_pose, solve_chunk
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
)
from roboclock.envs.lab.policies.xpolicylab import connection as xpolicylab_connection

# config.yaml dataloader.num_frames 33: the current state plus 32 future states.
ACTION_CHUNK = 32
POLICY_HZ = 25.0
HEAD_KEY = dict(CAMERAS)["cam_head"]


def sim_time(observation):
    return float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])


class OpenWAMPolicy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        replan = int(os.environ.get("OPENWAM_REPLAN", "0") or 0)
        self.replan = replan or ACTION_CHUNK
        if not 0 < self.replan <= ACTION_CHUNK:
            raise ValueError(f"OPENWAM_REPLAN must be in 1..{ACTION_CHUNK}")
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.prompt_time = prompt_time_enabled()
        self.frame_dir = os.environ.get("OPENWAM_FRAME_DIR")
        self.warned_cameras = set()

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.inferences = 0
        self.frame = -1
        self.queue = []
        self.previous_time = None
        self.rate_checked = False
        # Read by the evaluator after act().
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        # The adapter rejects openings outside [0, 1], and RoboDojo clips its commands.
        measured = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        self.gripper = np.clip(measured, 0.0, 1.0)
        self.connection.request("reset", {"trial_id": self.trial_id}, trial_id=self.trial_id)

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = sim_time(observation)
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"OpenWAM expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
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
        """The RoboDojo observation the adapter's update_obs reads (reading images renders)."""
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        vision = {}
        for camera, key in CAMERAS:
            image = observation.get(key)
            if image is None:
                if camera not in self.warned_cameras:
                    print(f"[openwam] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(observation[HEAD_KEY])
            vision[camera] = {"color": encode_jpeg(image)}
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            arm_joints = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_arm_joint_state"] = arm_joints
            state[f"{side}_ee_joint_state"] = np.asarray(self.gripper[arm : arm + 1], dtype=np.float32)
            state[f"{side}_ee_pose"] = link6_pose(side, arm_joints)
        return {
            "instruction": policy_instruction(observation, self.prompt_time),
            "env_idx": 0,
            "vision": vision,
            "state": state,
        }

    def request(self, obs):
        """One XPolicyLab ``infer`` (update_obs + get_action under the server's model lock).
        Returns the action dicts and the server latency in seconds."""
        reply = self.connection.request(
            "infer", {"observation": obs}, trial_id=self.trial_id, step=self.inferences
        )
        latency = reply.get("latency_ms")
        return reply["actions"], None if latency is None else float(latency) / 1000.0

    def infer(self, observation):
        request = self.observation(observation)
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            path = os.path.join(self.frame_dir, f"head_{max(self.frame, 0):05d}.jpg")
            with open(path, "wb") as stream:
                stream.write(request["vision"]["cam_head"]["color"])
        start = time.perf_counter()
        actions, latency = self.request(request)
        round_trip = time.perf_counter() - start
        self.inferences += 1
        self.last_inference_latency = latency
        if len(actions) != ACTION_CHUNK:
            raise ValueError(f"OpenWAM returned {len(actions)} actions, expected {ACTION_CHUNK}")
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        commands, ik_error = solve_chunk(
            actions[: self.replan], [joints[8 * arm : 8 * arm + 6] for arm in ARM_ORDER]
        )
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            measured = joints[ARM_DIMS]
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]

            def pose(value):
                return [round(float(v), 4) for v in np.asarray(value).reshape(-1)]

            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": sim_time(observation),
                "server_ms": None if latency is None else latency * 1000.0,
                "round_trip_ms": round_trip * 1000.0,
                "state_pose": {side: pose(request["state"][f"{side}_ee_pose"]) for side in SIDES},
                "state_gripper": [round(float(g), 3) for g in self.gripper],
                "first_target_pose": {side: pose(actions[0][f"{side}_ee_pose"]) for side in SIDES},
                "last_target_pose": {
                    side: pose(actions[self.replan - 1][f"{side}_ee_pose"]) for side in SIDES
                },
                "gripper_commands": [
                    [round(float(np.asarray(a[f"{side}_ee_joint_state"]).reshape(-1)[0]), 3) for side in SIDES]
                    for a in actions[: self.replan]
                ],
                "ik_error_m": ik_error,
                "clipped_arm_targets": int(clipped[:, ARM_DIMS].sum()),
                "start_jump_rad": float(np.abs(arm_path[0] - measured).max()),
                "chunk_travel_rad": float(np.abs(np.diff(arm_path, axis=0)).sum(axis=0).max()),
            }
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return list(zip(commands, np.clip(targets, LOW, HIGH)))


def make():
    return OpenWAMPolicy()
