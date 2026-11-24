"""InternW0-Delta (RoboDojo checkpoint) over XPolicyLab, driving the dual-X5 arms.

The checkpoint ``InternRobotics/InternW0-Delta-RoboDojo`` (``robodojo.pt``) is a world-action
model: a Wan2.2-TI2V-5B video expert and an action DiT, conditioned by RynnBrain on the three
views. XPolicyLab's ``policy/InternW0_delta`` adapter serves it. It predicts 32 joint actions,
1/25 s apart, in RoboDojo's ``arx_x5`` layout ([left j1-6, left gripper, right j1-6, right
gripper], grippers in [0, 1]) and returns the first 10 (``replan_steps``).

The session lives on the server. The adapter keeps one session per ``env_idx`` with the
episode's memory: the first frame (anchor), the frame 32 steps before each request (recent),
their joint states, and a step counter. A session holds the 10 actions it returned as pending
until each is acknowledged by an observation; it refuses a new request while any is pending.
``reset`` clears every session of the server. So the client reproduces RoboDojo's request
cadence exactly (``policy/InternW0_delta/deploy.py``, ``eval_one_episode_batch``, which
RoboDojo runs because ``deploy.yml`` sets ``eval_batch: true``):

- ``reset`` at the start of the episode;
- ``update_obs_batch`` with the observation of every step: after the first, each one
  acknowledges the action just executed and records the frame and state of the new step;
- ``get_action_batch`` for ``env_idx`` 0 whenever the 10 actions are used up, i.e. at steps
  0, 10, 20, ... right after that step's ``update_obs_batch``.

One server serves one episode at a time: run one server per concurrent episode.

Frames. The server reads the images of only some steps: the current frame at each request
(steps 0, 10, 20, ...) and the recent frame 32 steps before a request (steps 8, 18, 28, ...).
It records every step's frame but uses no other. The client renders those steps' cameras and
sends the last rendered images on the others, which the server records but never reads; the
joint state is always the current one. ``INTERNW0_RENDER_ALL=1`` renders every step instead.

State. Joint positions as measured; each gripper as the last commanded opening clipped to
[0, 1] (RoboDojo's state semantics), the measured opening before the first command.

Run the evaluator at RoboDojo's policy rate (``roboclock eval --policy-hz 25``): one act() is one
RoboDojo step and one session step. The client raises an error at any other rate.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL       ws://host:port of the InternW0_delta XPolicyLab server
  XPOLICYLAB_COOKIE    code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG       optional JSONL file receiving one row per request
  INTERNW0_RENDER_ALL  1 renders the cameras on every step (slower, same requests)
  INTERNW0_FRAME_DIR   optional directory receiving the head JPEG of every request

See docs/internw0.md.
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

# policy/InternW0_delta/deploy.yml: 32-step horizon, 10 actions executed per request, 25 Hz.
ACTION_HORIZON = 32
REPLAN_STEPS = 10
POLICY_HZ = 25.0
ENV_IDX = 0
HEAD_KEY = dict(CAMERAS)["cam_head"]


def sim_time(observation):
    return float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])


def frame_needed(step, horizon=ACTION_HORIZON, replan=REPLAN_STEPS):
    """Whether the server reads the images of session step ``step``.

    Requests come at multiples of ``replan``; each reads its current frame, the episode's
    first frame (step 0) and the frame ``horizon`` steps earlier
    (``RobotWinPolicySession._build_memory_inputs``, one recent frame).
    """
    return step % replan == 0 or (step + horizon) % replan == 0


class InternW0Policy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.render_all = os.environ.get("INTERNW0_RENDER_ALL") == "1"
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.prompt_time = prompt_time_enabled()
        self.frame_dir = os.environ.get("INTERNW0_FRAME_DIR")
        self.warned_cameras = set()

    def call(self, func_name, obs=None):
        """One XPolicyLab ``call`` (as WsModelClient.call). Returns (result, server s)."""
        payload = {"func_name": func_name}
        if obs is not None:
            payload["obs"] = obs
        reply = self.connection.request(
            "call", payload, trial_id=self.trial_id, step=self.inferences
        )
        latency = reply.get("latency_ms")
        return reply.get("result"), None if latency is None else float(latency) / 1000.0

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.inferences = 0
        self.step = -1
        self.queue = []
        self.vision = None
        self.rendered_frames = 0
        self.previous_time = None
        self.rate_checked = False
        # Read by the evaluator after act().
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        # RoboDojo's eval client resets the policy at every episode (WsModelClient sends the
        # reset message); the adapter clears the sessions of every env_idx.
        self.connection.request("reset", {"trial_id": self.trial_id}, trial_id=self.trial_id)

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = sim_time(observation)
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"InternW0 expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
                    "run roboclock eval with --policy-hz 25"
                )
            self.rate_checked = np.isfinite(step)
        self.previous_time = now

    def render(self, observation):
        """XPL-RGB1 JPEGs of the three views (reading them renders the cameras)."""
        head = observation[HEAD_KEY]
        vision = {}
        for camera, key in CAMERAS:
            image = head if key == HEAD_KEY else observation.get(key)
            if image is None:
                # The adapter needs all three views; a black frame keeps the request valid.
                if camera not in self.warned_cameras:
                    print(f"[internw0] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(head)
            vision[camera] = {"color": encode_jpeg(image)}
        self.rendered_frames += 1
        return vision

    def observation(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            state[f"{side}_arm_joint_state"] = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_ee_joint_state"] = np.asarray(
                self.gripper[arm : arm + 1], dtype=np.float32
            )
        return {
            "instruction": policy_instruction(observation, self.prompt_time),
            "env_idx": ENV_IDX,
            "vision": self.vision,
            "state": state,
        }

    def act(self, observation):
        self.step += 1
        self.check_rate(observation)
        due = not self.queue
        if due or self.render_all or self.vision is None or frame_needed(self.step):
            self.vision = self.render(observation)
        request = self.observation(observation)
        start = time.perf_counter()
        _, update_latency = self.call("update_obs_batch", [request])
        self.queried = due
        if due:
            self.queue = self.infer(observation, request, start, update_latency)
        else:
            self.last_inference_latency = None
        command, target = self.queue.pop(0)
        # RoboDojo reports the commanded opening after clipping it to [0, 1].
        self.gripper = np.clip(command[[6, 13]], 0.0, 1.0)
        return target

    def infer(self, observation, request, start, update_latency):
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            name = os.path.join(self.frame_dir, f"head_{self.step:05d}.jpg")
            with open(name, "wb") as stream:
                stream.write(request["vision"]["cam_head"]["color"])
        result, action_latency = self.call("get_action_batch", [ENV_IDX])
        round_trip = time.perf_counter() - start
        self.inferences += 1
        latencies = [value for value in (update_latency, action_latency) if value is not None]
        self.last_inference_latency = sum(latencies) if latencies else None
        if len(result) != 1:
            raise ValueError(f"InternW0 returned {len(result)} chunks for one env_idx")
        actions = result[0]
        if len(actions) != REPLAN_STEPS:
            raise ValueError(f"InternW0 returned {len(actions)} actions, expected {REPLAN_STEPS}")
        commands = np.stack([robodojo_vector(action) for action in actions])
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            joints = np.asarray(observation["joint_positions"], dtype=np.float32)
            measured = joints[ARM_DIMS]
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "step": self.step,
                "sim_time": sim_time(observation),
                "prompt": policy_instruction(observation, True) if self.prompt_time else None,
                "server_ms": None if action_latency is None else action_latency * 1000.0,
                "update_ms": None if update_latency is None else update_latency * 1000.0,
                "round_trip_ms": round_trip * 1000.0,
                "rendered_frames": self.rendered_frames,
                "state_gripper": [round(float(v), 3) for v in self.gripper],
                "gripper_commands": [[round(float(v), 3) for v in c[[6, 13]]] for c in commands],
                "clipped_arm_targets": int(clipped[:, ARM_DIMS].sum()),
                # Lab joint indices (0-15) clipped somewhere in the chunk.
                "clipped_joints": [int(i) for i in ARM_DIMS if clipped[:, i].any()],
                "start_jump_rad": float(np.abs(arm_path[0] - measured).max()),
                "chunk_travel_rad": float(np.abs(np.diff(arm_path, axis=0)).sum(axis=0).max()),
            }
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        return list(zip(commands, np.clip(targets, LOW, HIGH)))


def make():
    return InternW0Policy()
