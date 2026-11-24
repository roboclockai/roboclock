"""VPP2 (Video Prediction Policy 2, RoboDojo joint Video + Action2B 100k) over XPolicyLab.

The checkpoint (``Haodong082399/VPP2``, bundle ``joint2b_s100000``) is a Wan2.1-I2V-14B video
model with a 2B action expert. It answers with 32-step chunks of absolute RoboDojo EE16 actions:
per arm, the link6 pose ``[x, y, z, qw, qx, qy, qz]`` in RoboDojo's environment frame and a
[0, 1] gripper opening, left arm first. The server returns the first 24 (``replan_steps``) and
the client replays them before it asks again, as XPolicyLab's ``policy/VPP2/deploy.py`` does.

Observation history lives on the server. Its adapter keeps one history per client id
(``_vpp2_client_id``) and records one observation per ``update_obs`` call, so the step index
of a frame is the number of observations recorded before it. Like ``deploy.py``, the client
calls ``update_obs`` on every RoboDojo step and ``get_action`` on every 24th. A request at step
q is conditioned on the episode's first frame (the anchor) and the frames at q, q-25, ...,
q-175 (steps <= 0 read the anchor), all composed into the 240x416 T-shaped view
(``vpp2.policy.RoboDojoVPP2Policy._build_history_condition``).

Frames. Run the evaluator at RoboDojo's rate (``roboclock eval --policy-hz 25``): one act() is
one RoboDojo step and one recorded observation. With replan 24 and stride 25 the server reads
only steps s with s % 24 in {0, 17, ..., 23} (s + 25 j is a request step for some j in 0..7),
a third of all steps. The client renders those and, on the other steps, repeats the last
rendered JPEGs so that the server's step count stays right; the server never reads those
images. Set ``VPP2_RECORD_ALL_FRAMES=1`` to render every step instead.

State and actions use the RoboDojo frame as ``wallx`` does: the link6 pose of each arm from the
measured joints (``wallx.link6_pose``), and returned poses solved to joints along the chunk
(``wallx.solve_chunk``). The checkpoint z-scores the state quaternion component by component,
so its sign matters too: the client sends RoboDojo's sign, the one PhysX gets by composing the
joint rotations (``robodojo_ee_pose``). The gripper state is the last commanded opening, as
RoboDojo reports.

Configured by environment (custom policy factories take no arguments):
  XPOLICYLAB_URL          ws://host:port of the VPP2 XPolicyLab server
  XPOLICYLAB_COOKIE       code-server-session cookie when the URL goes through code-server
  XPOLICYLAB_LOG          optional JSONL file receiving one row per request
  VPP2_RECORD_ALL_FRAMES  1 renders every step (slower, same requests otherwise)
  VPP2_FRAME_DIR          optional directory receiving the head JPEG of every request

See docs/vpp2.md.
"""

import json
import os
import socket
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
    POLICY_HZ,
    SIDES,
    encode_jpeg,
    lab_action,
    normalise_gripper,
)
from roboclock.envs.lab.policies.xpolicylab import connection as xpolicylab_connection

# policy/VPP2/deploy.yml and the bundle manifest: 32 predicted actions, 24 executed, a history of
# 8 frames 25 native steps apart plus the episode anchor.
ACTION_HORIZON = 32
REPLAN_STEPS = 24
HISTORY_FRAMES = 8
HISTORY_STRIDE = 25
CLIENT_ID_FIELD = "_vpp2_client_id"


def _quaternion_product(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


_CHAIN = []


def chain_quaternion(joints):
    """link6 orientation (wxyz) in RoboDojo's frame as PhysX composes it: the base yaw, then
    each joint's fixed origin and its rotation [cos(q/2), sin(q/2) axis] in turn.

    Its sign is a continuous function of the joints, and with the global sign set so that joint
    zero gives RoboDojo's recorded [0.707, 0, 0, 0.707] it matches every recorded ``ee_pose``
    quaternion of the replay demonstrations; a quaternion taken from the rotation matrix has
    the opposite sign on about a quarter of those frames.
    """
    from scipy.spatial.transform import Rotation

    from roboclock.envs.lab.policies.wallx import ROBODOJO_BASE_YAW_DEG, kinematics

    model = kinematics()
    if not _CHAIN:
        half = np.radians(ROBODOJO_BASE_YAW_DEG) / 2.0
        fixed = [np.roll(Rotation.from_matrix(origin[:3, :3]).as_quat(), 1) for origin in model.origins]
        _CHAIN.extend([np.array([np.cos(half), 0.0, 0.0, np.sin(half)]), fixed, 1.0])
        _CHAIN[2] = 1.0 if chain_quaternion(np.zeros(6))[0] > 0 else -1.0
    base, fixed, sign = _CHAIN
    quaternion = base
    for origin, axis, angle in zip(fixed, model.axes, np.asarray(joints, dtype=float)[:6]):
        quaternion = _quaternion_product(quaternion, origin)
        quaternion = _quaternion_product(quaternion, np.r_[np.cos(angle / 2.0), np.sin(angle / 2.0) * axis])
    return sign * quaternion


def robodojo_ee_pose(side, joints):
    """RoboDojo's ``<side>_ee_pose`` with RoboDojo's quaternion sign.

    VPP2 reads the state quaternion component by component (z-scored), so q and -q are
    different inputs; ``wallx.link6_pose`` gives the right rotation but not always its sign.
    """
    pose = link6_pose(side, joints)
    if float(np.dot(pose[3:], chain_quaternion(joints))) < 0.0:
        pose[3:] = -pose[3:]
    return pose


def history_steps(request_step):
    """Steps a request at ``request_step`` reads: the anchor, then the 8 history frames.

    Mirrors ``RoboDojoVPP2Policy._build_history_condition`` for a history with every step
    recorded: offsets -(7 - i) * 25, and steps <= 0 read the anchor (step 0).
    """
    steps = [0]
    for index in range(HISTORY_FRAMES):
        requested = request_step - (HISTORY_FRAMES - 1 - index) * HISTORY_STRIDE
        steps.append(max(requested, 0))
    return steps


def frame_needed(step, replan=REPLAN_STEPS):
    """Whether a later (or this) request reads the image recorded at ``step``."""
    return step == 0 or any((step + j * HISTORY_STRIDE) % replan == 0 for j in range(HISTORY_FRAMES))


class VPP2Policy:
    def __init__(self):
        self.connection = xpolicylab_connection()
        self.replan = REPLAN_STEPS
        self.record_all = os.environ.get("VPP2_RECORD_ALL_FRAMES") == "1"
        self.log_path = os.environ.get("XPOLICYLAB_LOG")
        self.frame_dir = os.environ.get("VPP2_FRAME_DIR")
        self.warned_cameras = set()

    def call(self, func_name, obs=None):
        """One XPolicyLab ``call``, as deploy.py makes them. Returns (result, server s)."""
        payload = {"func_name": func_name}
        if obs is not None:
            payload["obs"] = obs
        reply = self.connection.request("call", payload, trial_id=self.trial_id, step=self.frame)
        latency = reply.get("latency_ms")
        return reply.get("result"), None if latency is None else float(latency) / 1000.0

    def reset(self, observation):
        self.trial_id = uuid.uuid4().hex
        # One history per client id on the server: a fresh id per episode.
        self.client_id = f"roboclock-{socket.gethostname()}:{os.getpid()}:{self.trial_id[:8]}"
        self.frame = -1
        self.inferences = 0
        self.queue = []
        self.jpegs = None
        self.rendered_frames = 0
        self.rendered_steps = set()
        self.previous_time = None
        self.rate_checked = False
        # Read by the evaluator after act().
        self.queried = True
        self.last_inference_latency = None
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        # RoboDojo reports the last commanded opening; before any command, the measured one.
        self.gripper = normalise_gripper(joints[[8 * arm + 6 for arm in ARM_ORDER]])
        self.call("reset")
        self.call("reset_client", {CLIENT_ID_FIELD: self.client_id})

    def check_rate(self, observation):
        """One act() must be one 25 Hz RoboDojo step: run the evaluator with --policy-hz 25."""
        now = float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0])
        if self.previous_time is not None and not self.queried and not self.rate_checked:
            step = now - self.previous_time
            if np.isfinite(step) and abs(step - 1.0 / POLICY_HZ) > 0.25 / POLICY_HZ:
                raise RuntimeError(
                    f"VPP2 expects one act() per 1/{POLICY_HZ:g} s (got {step:.4f} s); "
                    "run roboclock eval with --policy-hz 25"
                )
            self.rate_checked = np.isfinite(step)
        self.previous_time = now

    def act(self, observation):
        self.frame += 1
        due = self.frame % self.replan == 0
        self.check_rate(observation)
        # Packing reads the camera images, which renders them; build the request first so the
        # logged round trip is transport plus server time only.
        request = self.observation(observation)
        start = time.perf_counter()
        _, observe_latency = self.call("update_obs", request)
        observe_time = time.perf_counter() - start
        self.queried = due
        if due:
            self.queue = self.infer(observation, observe_latency, observe_time)
        else:
            self.last_inference_latency = None
        command, target = self.queue.pop(0)
        # RoboDojo reports the commanded opening after clipping it to [0, 1].
        self.gripper = np.clip(command[[6, 13]], 0.0, 1.0)
        return target

    def images(self, observation):
        """JPEGs of the three views; rendered only on steps the server will read."""
        if self.jpegs is not None and not (self.record_all or frame_needed(self.frame, self.replan)):
            return self.jpegs
        head = observation[CAMERAS[0][1]]
        jpegs = {}
        for camera, key in CAMERAS:
            image = head if camera == "cam_head" else observation.get(key)
            if image is None:
                if camera not in self.warned_cameras:
                    print(f"[vpp2] observation has no {key!r}; sending black {camera}")
                    self.warned_cameras.add(camera)
                image = np.zeros_like(head)
            jpegs[camera] = encode_jpeg(image)
        self.jpegs = jpegs
        self.rendered_frames += 1
        self.rendered_steps.add(self.frame)
        return jpegs

    def observation(self, observation):
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        state = {}
        for arm, (side, lab_arm) in enumerate(zip(SIDES, ARM_ORDER)):
            arm_joints = joints[8 * lab_arm : 8 * lab_arm + 6]
            state[f"{side}_arm_joint_state"] = arm_joints
            state[f"{side}_ee_joint_state"] = np.asarray(self.gripper[arm : arm + 1], dtype=np.float32)
            state[f"{side}_ee_pose"] = robodojo_ee_pose(side, arm_joints)
        return {
            "instruction": observation["instruction"],
            "env_idx": 0,
            "vision": {camera: {"color": data} for camera, data in self.images(observation).items()},
            "state": state,
            CLIENT_ID_FIELD: self.client_id,
        }

    def infer(self, observation, observe_latency, observe_time):
        if self.frame_dir:
            os.makedirs(self.frame_dir, exist_ok=True)
            with open(os.path.join(self.frame_dir, f"head_{self.frame:05d}.jpg"), "wb") as stream:
                stream.write(self.jpegs["cam_head"])
        missing = [step for step in history_steps(self.frame) if step not in self.rendered_steps]
        if missing:
            raise RuntimeError(f"VPP2 request at step {self.frame} reads unrendered steps {missing}")
        start = time.perf_counter()
        actions, latency = self.call("get_action", {CLIENT_ID_FIELD: self.client_id})
        round_trip = time.perf_counter() - start + observe_time
        self.inferences += 1
        if latency is not None and observe_latency is not None:
            latency += observe_latency
        self.last_inference_latency = latency
        if len(actions) != self.replan:
            raise ValueError(f"VPP2 returned {len(actions)} actions, expected {self.replan}")
        joints = np.asarray(observation["joint_positions"], dtype=np.float32)
        commands, ik_error = solve_chunk(actions, [joints[8 * arm : 8 * arm + 6] for arm in ARM_ORDER])
        targets = lab_action(commands)
        clipped = (targets < LOW) | (targets > HIGH)
        if self.log_path:
            measured = joints[ARM_DIMS]
            arm_path = np.clip(targets, LOW, HIGH)[:, ARM_DIMS]
            row = {
                "trial_id": self.trial_id,
                "inference": self.inferences - 1,
                "frame": self.frame,
                "sim_time": float(np.asarray(observation.get("sim_time", [np.nan])).reshape(-1)[0]),
                "server_ms": None if latency is None else latency * 1000.0,
                "round_trip_ms": round_trip * 1000.0,
                "rendered_frames": self.rendered_frames,
                "history_steps": history_steps(self.frame),
                "state_pose": {
                    side: [round(float(v), 4) for v in robodojo_ee_pose(side, joints[8 * arm : 8 * arm + 6])]
                    for side, arm in zip(SIDES, ARM_ORDER)
                },
                "state_gripper": [round(float(v), 3) for v in self.gripper],
                "first_target_pose": {
                    side: [round(float(v), 4) for v in np.asarray(actions[0][f"{side}_ee_pose"]).reshape(-1)]
                    for side in SIDES
                },
                "last_target_pose": {
                    side: [round(float(v), 4) for v in np.asarray(actions[-1][f"{side}_ee_pose"]).reshape(-1)]
                    for side in SIDES
                },
                "gripper_commands": [
                    [round(float(np.asarray(a[f"{side}_ee_joint_state"]).reshape(-1)[0]), 3) for side in SIDES]
                    for a in actions
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
    return VPP2Policy()
