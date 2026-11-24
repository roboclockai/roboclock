"""X-VLA (RoboDojo checkpoint ``X_VLA/RoboDojo-sim-arx_x5-ee-0``) over XPolicyLab, driving the dual-X5 arms.

The checkpoint is served by XPolicyLab's ``policy/X_VLA`` adapter. Each request returns 30
steps of RoboDojo end-effector actions: per arm, the link6 pose ``[x, y, z, qw, qx, qy, qz]``
in RoboDojo's environment frame and a [0, 1] gripper opening. The client replays the whole
chunk before it asks again, as ``policy/X_VLA/deploy.py`` does.

What the adapter and its training data fix (see docs/xvla.md):
  - The model reads one image, ``cam_head`` (RGB, resized to 224x224 by the processor), the
    instruction, and a 20-value proprio vector the server builds from ``state``:
    ``[xyz, rot6d, gripper] x (left, right)`` with absolute link6 poses. rot6d is the first two
    columns of the rotation matrix, flattened row-major; the action space zeroes the gripper
    channels of the proprio, so only the poses matter. The domain id is deploy.yml's 0.
  - Training targets are the *state* trajectory of the next 30 recorded frames (absolute, no
    deltas): entry k is RoboDojo's state k+1 frames ahead, i.e. its 25 Hz action k. The
    server decodes rot6d to a quaternion and passes the sigmoid gripper through unchanged.
  - The adapter keeps no history: ``update_obs`` overwrites one observation slot and
    ``get_action`` reads it. The ``infer`` message runs both under the server's model lock,
    so concurrent episodes can share one server.

Frames. Run the evaluator at RoboDojo's policy rate (``roboclock eval --policy-hz 25``): one
act() is one RoboDojo simulator step and one chunk entry. The client raises at any other
rate. The head camera is read (and so rendered) only on request frames.

Frames of reference. As for Wall-X (``roboclock.envs.lab.policies.wallx``): the client reports
each arm's link6 pose in RoboDojo's environment frame from the measured joints, and solves the
returned poses to joints step by step along the chunk from the measured joints.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL     ws://host:port of the X_VLA XPolicyLab server
  XPOLICYLAB_COOKIE  code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG     optional JSONL file receiving one row per request
  XVLA_REPLAN        actions executed per chunk (default 30, the whole chunk, as deploy.py)
  XVLA_FRAME_DIR     optional directory receiving the head JPEG of every request
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

# X_VLA RoboDojo-sim-arx_x5-ee-0 config.json num_actions; deploy.py replays the whole chunk.
ACTION_CHUNK = 30
POLICY_HZ = 25.0
# The adapter reads only the head camera (model.encode_obs: cam_high / cam_head / ...).
HEAD_KEY = dict(CAMERAS)["cam_head"]


def sim_time(observation):
    return float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])


def quat_to_rotate6d(quat_wxyz):
    """The adapter's (and its training data's) rot6d: rotation-matrix columns 0-1, row-major."""
    from scipy.spatial.transform import Rotation

    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat([x, y, z, w]).as_matrix()[:, :2].reshape(6)


def proprio(state):
    """The 20-value vector the server builds from a request's state (model.build_xvla_proprio)."""
    parts = []
    for side in SIDES:
        pose = np.asarray(state[f"{side}_ee_pose"], dtype=float)
        grip = np.asarray(state[f"{side}_ee_joint_state"], dtype=float)[-1:]
        parts += [pose[:3], quat_to_rotate6d(pose[3:]), grip]
    return np.concatenate(parts).astype(np.float32)


class XVLAPolicy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.replan = int(os.environ.get("XVLA_REPLAN", ACTION_CHUNK))
        if not 0 < self.replan <= ACTION_CHUNK:
            raise ValueError(f"XVLA_REPLAN must be in 1..{ACTION_CHUNK}")
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.prompt_time = prompt_time_enabled()
        self.frame_dir = os.environ.get("XVLA_FRAME_DIR")

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
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        # deploy.py: model_client.call(func_name="reset"). The adapter's reset clears its one
        # observation slot; every infer refills it, so this is harmless on a shared server.
        self.connection.request("call", {"func_name": "reset"}, trial_id=self.trial_id)

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = sim_time(observation)
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"X-VLA expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
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

    def observation(self, observation, head_jpeg):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            arm_joints = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_arm_joint_state"] = arm_joints
            state[f"{side}_ee_joint_state"] = np.asarray(self.gripper[arm : arm + 1], dtype=np.float32)
            state[f"{side}_ee_pose"] = link6_pose(side, arm_joints)
        return {
            "instruction": policy_instruction(observation, self.prompt_time),
            "env_idx": 0,
            "vision": {"cam_head": {"color": head_jpeg}},
            "state": state,
        }

    def infer(self, observation):
        # Reading the image renders the cameras; do it before timing the request.
        head_jpeg = encode_jpeg(observation[HEAD_KEY])
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            with open(os.path.join(self.frame_dir, f"head_{self.frame:05d}.jpg"), "wb") as stream:
                stream.write(head_jpeg)
        request = self.observation(observation, head_jpeg)
        start = time.perf_counter()
        # The infer message is the adapter's update_obs + get_action under one model lock.
        reply = self.connection.request(
            "infer", {"observation": request}, trial_id=self.trial_id, step=self.inferences
        )
        round_trip = time.perf_counter() - start
        self.inferences += 1
        latency = reply.get("latency_ms")
        self.last_inference_latency = None if latency is None else float(latency) / 1000.0
        actions = reply["actions"]
        if len(actions) != ACTION_CHUNK:
            raise ValueError(f"X-VLA returned {len(actions)} actions, expected {ACTION_CHUNK}")
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        commands, ik_error = solve_chunk(
            actions[: self.replan], [joints[8 * arm : 8 * arm + 6] for arm in ARM_ORDER]
        )
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            measured = joints[ARM_DIMS]
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": sim_time(observation),
                "server_ms": None if latency is None else float(latency),
                "round_trip_ms": round_trip * 1000.0,
                "state_pose": {
                    side: [round(float(v), 4) for v in request["state"][f"{side}_ee_pose"]] for side in SIDES
                },
                "first_target_pose": {
                    side: [round(float(v), 4) for v in np.asarray(actions[0][f"{side}_ee_pose"]).reshape(-1)]
                    for side in SIDES
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
    return XVLAPolicy()
