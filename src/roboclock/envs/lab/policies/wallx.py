"""Wall-X (WALL-OSS 1.0 lite, RoboDojo checkpoint) over XPolicyLab, driving the dual-X5 arms.

The checkpoint ``x-square-robot/wall-oss-1.0-lite-robodojo`` answers with 32-step chunks of
native RoboDojo end-effector actions (``<side>_ee_pose`` [x, y, z, qw, qx, qy, qz] of link6
plus a [0, 1] gripper opening). The client replays the first 16 of them before it asks again,
as XPolicyLab's ``policy/WallX/deploy.py`` does. The checkpoint also keeps a keyframe memory
whose bookkeeping lives on the client: every request carries the episode's memory state
(``kfmem_client_state_v1``) and raw head-camera frames, and the reply returns the updated state.
This module drives that state with XPolicyLab's own ``policy/WallX/memory_client.py``, so the
request layout is the reference client's.

Frames. Run the evaluator at RoboDojo's policy rate (``roboclock eval --policy-hz 25``): one
act() is one RoboDojo simulator step and one memory frame. The server reads only some frames
(the current one at each request and, every ``select_stride`` frames, ``num_candidates``
frames spread over the selection window); its ``ring_plan`` and the selection rule are
deterministic, so the client renders exactly those frames and skips the rest. A frame the
server would need but the client skipped is an error, never a substitute. Set
``WALLX_RECORD_ALL_FRAMES=1`` to render every frame instead.

Frames of reference. RoboDojo reports and commands link6 in its environment frame, where the
dual_x5 bases sit at (-0.30, -0.45, 0.765) (left) and (0.30, -0.45, 0.765) (right), both yawed
+90 deg (``env_cfg/robot/dual_x5.yml``). The checkpoint reads that absolute pose as its state,
so the client reports each arm's link6 pose in the same frame: the roboclock rig keeps RoboDojo's
geometry relative to the arms (``roboclock.envs.robodojo_rig``), so the RoboDojo base pose is a
constant per arm. Returned poses go back through the inverse transform and are solved to
joints with the X5 kinematics, step by step along the chunk from the measured joints.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL           ws://host:port of the Wall-X XPolicyLab server
  XPOLICYLAB_COOKIE        code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG           optional JSONL file receiving one row per request
  WALLX_REPLAN             actions executed per chunk (default 16, the published setting)
  WALLX_TASK_ID            task id sent to the server (default roboclock); it only selects
                           per-task memory options, none of which match roboclock tasks
  WALLX_RECORD_ALL_FRAMES  1 renders every frame for the memory (slower, same requests)
  WALLX_FRAME_DIR          optional directory receiving the head JPEG of every request

XPolicyLab's ``policy/WallX/memory_client.py`` must be importable: put the RoboDojo checkout
on PYTHONPATH (it provides ``XPolicyLab``). See docs/wallx.md.
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
)
from roboclock.envs.lab.policies.xpolicylab import connection as xpolicylab_connection

# RoboDojo env_cfg/robot/dual_x5.yml: base positions in the environment frame, both yawed
# +90 deg about z (quaternion wxyz [0.707, 0, 0, 0.707]).
ROBODOJO_BASES = {"left": (-0.30, -0.45, 0.765), "right": (0.30, -0.45, 0.765)}
ROBODOJO_BASE_YAW_DEG = 90.0
# policy/WallX/deploy.yml: the checkpoint predicts 32 steps, the simulator replays 16.
ACTION_CHUNK = 32
REPLAN_STEPS = 16
POLICY_HZ = 25.0
# The memory frames come from cam_head.
HEAD_KEY = dict(CAMERAS)["cam_head"]

_KINEMATICS = []


def memory_client():
    try:
        from XPolicyLab.policy.WallX import memory_client as module
    except ImportError as error:
        raise ImportError(
            "Wall-X needs XPolicyLab's policy/WallX/memory_client.py: put the RoboDojo "
            "checkout (with XPolicyLab/policy/WallX) on PYTHONPATH"
        ) from error
    return module


def kinematics():
    """X5 kinematics in the arm's own base frame, built once (it parses the URDF)."""
    if not _KINEMATICS:
        from roboclock.policies.x5_kinematics import X5Kinematics

        _KINEMATICS.append(X5Kinematics())
    return _KINEMATICS[0]


def robodojo_base(side):
    """4x4 pose of a RoboDojo dual_x5 arm base in the RoboDojo environment frame."""
    angle = np.radians(ROBODOJO_BASE_YAW_DEG)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    pose[:3, 3] = ROBODOJO_BASES[side]
    return pose


def link6_pose(side, joints):
    """RoboDojo's ``<side>_ee_pose`` (link6, [x, y, z, qw, qx, qy, qz]) in its environment frame.

    The lab chain ends at grasp_center, a fixed transform past link6 (URDF ``tool_frame``).
    """
    from scipy.spatial.transform import Rotation

    model = kinematics()
    pose = robodojo_base(side) @ model.tool_pose(np.asarray(joints, dtype=float)[:6])
    pose = pose @ np.linalg.inv(model.tool)
    x, y, z, w = Rotation.from_matrix(pose[:3, :3]).as_quat()
    return np.array([*pose[:3, 3], w, x, y, z], dtype=np.float32)


def solve_chunk(actions, arm_joints):
    """RoboDojo end-effector actions as packed 14-value joint vectors (RoboDojo arm order).

    Each step's link6 target is solved from the previous step's solution, starting at the
    measured joints of each arm (``arm_joints``, left then right). Returns the vectors and the
    worst remaining position error in metres (an unreachable target stops at the closest pose).
    """
    from roboclock.policies.x5_kinematics import pose_matrix

    model = kinematics()
    to_base = {side: np.linalg.inv(robodojo_base(side)) for side in SIDES}
    current = [np.asarray(joints, dtype=float)[:6] for joints in arm_joints]
    rows, worst = [], 0.0
    for action in actions:
        row = []
        for index, side in enumerate(SIDES):
            pose = np.asarray(action[f"{side}_ee_pose"], dtype=float).reshape(7)
            target = to_base[side] @ pose_matrix(pose[:3], pose[3:7]) @ model.tool
            current[index], error = model.solve(target, current[index], orientation_weight=1.0)
            worst = max(worst, float(error))
            opening = np.asarray(action[f"{side}_ee_joint_state"], dtype=np.float32).reshape(-1)
            row += [*current[index], float(opening[0])]
        rows.append(row)
    return np.asarray(rows, dtype=np.float32), worst


def jpeg_roundtrip(image):
    """The JPEG the server receives and the pixels it decodes from it (XPL-RGB1 is RGB)."""
    import io

    from PIL import Image

    data = encode_jpeg(image)
    return data, np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def planned_frames(plan, now, last_select, replan, needed_ring_frames):
    """Frames after ``now`` that requests at now + replan, now + 2 replan, ... will read.

    Mirrors the server: a request at t reads ``needed_ring_frames(plan, t, last_select)``,
    and a selection that is due at t sets last_select to t. Requests far enough ahead that
    they cannot read a frame before the next request are left out.
    """
    wanted = set()
    request = now
    while True:
        request += replan
        if request - int(plan["span"]) > now + replan:
            return wanted
        wanted.update(f for f in needed_ring_frames(plan, request, last_select) if f > now)
        if plan["selection"] and request - last_select >= int(plan["select_stride"]):
            last_select = request


class WallXPolicy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.memory_module = memory_client()
        self.replan = int(os.environ.get("WALLX_REPLAN", REPLAN_STEPS))
        if not 0 < self.replan <= ACTION_CHUNK:
            raise ValueError(f"WALLX_REPLAN must be in 1..{ACTION_CHUNK}")
        self.task_id = os.environ.get("WALLX_TASK_ID", "roboclock")
        self.record_all = os.environ.get("WALLX_RECORD_ALL_FRAMES") == "1"
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        # Optional: keep the head image of every request (one per chunk) for inspection.
        self.frame_dir = os.environ.get("WALLX_FRAME_DIR")
        self.warned_cameras = set()

    def call(self, func_name, obs=None):
        """One XPolicyLab ``call`` (WallX's infer returns the memory state next to the actions,
        which the plain ``infer`` message drops). Returns (result, server latency in s)."""
        payload = {"func_name": func_name}
        if obs is not None:
            payload["obs"] = obs
        reply = self.connection.request("call", payload, trial_id=self.trial_id, step=self.inferences)
        latency = reply.get("latency_ms")
        return reply.get("result"), None if latency is None else float(latency) / 1000.0

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        self.inferences = 0
        self.frame = -1
        self.since_request = self.replan  # request on the first step
        self.queue = []
        self.wanted = set()
        self.rendered_frames = 0
        self.previous_time = None
        self.rate_checked = False
        # Read by the evaluator after act().
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        key = f"roboclock-{self.trial_id[:12]}"
        self.memory = self.memory_module.KFMemClientState(key)
        initial, _ = self.call("reset")
        self.memory.reset(initial)

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"Wall-X expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
                    "run roboclock eval with --policy-hz 25"
                )
            self.rate_checked = np.isfinite(step)
        self.previous_time = now

    def act(self, observation):
        self.frame += 1
        due = self.since_request >= self.replan
        self.check_rate(observation)
        head_jpeg = None
        if due or self.record_all or self.frame in self.wanted:
            head_jpeg, pixels = jpeg_roundtrip(observation[HEAD_KEY])
            self.memory.record_frame(self.frame, pixels)
            self.rendered_frames += 1
        self.queried = due
        if due:
            self.queue = self.infer(observation, head_jpeg)
            self.since_request = 0
        else:
            self.last_inference_latency = None
        command, target = self.queue.pop(0)
        # RoboDojo reports the commanded opening after clipping it to [0, 1].
        self.gripper = np.clip(command[[6, 13]], 0.0, 1.0)
        self.since_request += 1
        return target

    def observation(self, observation, head_jpeg):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        vision = {"cam_head": {"color": head_jpeg}}
        for camera, key in CAMERAS:
            if camera == "cam_head":
                continue
            image = observation.get(key)
            if image is None:
                if camera not in self.warned_cameras:
                    print(f"[wallx] observation has no {key!r}; sending black {camera}")
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
            "instruction": observation["instruction"],
            "env_idx": 0,
            "vision": vision,
            "state": state,
            "task_id": self.task_id,
            "sim_frame": self.frame,
            "kfmem_client_state_v1": self.memory.request_state(self.frame),
            "kfmem_history": self.memory.history(self.frame),
        }

    def infer(self, observation, head_jpeg):
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            with open(os.path.join(self.frame_dir, f"head_{self.frame:05d}.jpg"), "wb") as stream:
                stream.write(head_jpeg)
        request = self.observation(observation, head_jpeg)
        memory_before = list(self.memory.mem_frame_ids)
        start = time.perf_counter()
        result, latency = self.call("infer", request)
        round_trip = time.perf_counter() - start
        self.inferences += 1
        self.last_inference_latency = latency
        actions = result["actions"]
        if len(actions) != ACTION_CHUNK:
            raise ValueError(f"Wall-X returned {len(actions)} actions, expected {ACTION_CHUNK}")
        self.memory.accept_result(result["kfmem_state"])
        plan = self.memory.ring_plan
        if plan is not None and not self.record_all:
            self.wanted = planned_frames(
                plan, self.frame, self.memory.last_sel_frame, self.replan,
                self.memory_module.needed_ring_frames,
            )
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        commands, ik_error = solve_chunk(
            actions[: self.replan], [joints[8 * arm : 8 * arm + 6] for arm in ARM_ORDER]
        )
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            measured = joints[ARM_DIMS]
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            poses = {side: request["state"][f"{side}_ee_pose"] for side in SIDES}
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0]),
                "server_ms": None if latency is None else latency * 1000.0,
                "round_trip_ms": round_trip * 1000.0,
                "ring_frames": len(request["kfmem_client_state_v1"]["ring"]),
                "memory_frames": memory_before,
                "memory_after": list(self.memory.mem_frame_ids),
                "last_sel_frame": self.memory.last_sel_frame,
                "rendered_frames": self.rendered_frames,
                "state_pose": {side: [round(float(v), 4) for v in pose] for side, pose in poses.items()},
                "first_target_pose": {
                    side: [round(float(v), 4) for v in np.asarray(actions[0][f"{side}_ee_pose"]).reshape(-1)]
                    for side in SIDES
                },
                "gripper_commands": [
                    [round(float(a[f"{side}_ee_joint_state"][0]), 3) for side in SIDES]
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
    return WallXPolicy()
