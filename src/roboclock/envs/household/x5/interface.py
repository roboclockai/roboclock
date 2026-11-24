"""Agent-facing Dual-X5 household action/observation conventions (no simulator imports).

The household arms are the X5 chains built in ``x5/sim/physics.py``: six revolute joints with
the arx5-sdk limits (identical to ``envs/lab/robot.py`` JOINT_LOW/HIGH[:6]) and two prismatic
jaws per arm, ``jawP`` (native range [-JAW_TRAVEL, 0]) and ``jawN`` ([0, JAW_TRAVEL]), both 0
when open.

Actions and ``joint_positions`` use the shared 16-value layout ``left/joint1..8`` then
``right/joint1..8``. Joints 1-6 are radians. Joints 7 and 8 are the two fingers in the lab
convention: per-finger opening in metres from the closed stop (0 = closed, positive = open).
For the household jaws that is ``JAW_TRAVEL + q(jawP)`` and ``JAW_TRAVEL - q(jawN)``.

Finger limits differ from the lab robot: the household jaw travels 30.14 mm per side
(lab: 44 mm), and a target below 0 (down to -0.03 m) is allowed. Such a target lies beyond the
closed stop, so the position drive (800 N/m, 20 N limit) clamps a thin part with up to its force
limit; the privileged oracle uses this ("squeeze", at most 25 mm). Targets above the 30.14 mm
travel up to the lab's 0.044 m only press the jaw against its open stop. The ``gripper_width``
observation is bounded by the real opening, ``MAX_GRIPPER_WIDTH`` = 60.28 mm.
"""

import numpy as np

from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW, ROBOT_ID

SIDES = ("Left", "Right")
JOINT_ORDER = tuple(f"{side}/joint{i}" for side in ("left", "right") for i in range(1, 9))
ARM_DOFS = tuple(f"joint{i}" for i in range(1, 7))
JAW_DOFS = ("jawP_joint", "jawN_joint")
DOF_ORDER = ARM_DOFS + JAW_DOFS  # per-arm action order, as articulation DOF names

# Mirrors x5/sim/physics.py (which imports pxr and cannot be imported without Isaac Sim).
ARM_LOWER = (-3.14, -0.05, -0.1, -1.6, -1.57, -2.0)
ARM_UPPER = (2.618, 3.5, 3.2, 1.55, 1.57, 2.0)
JAW_TRAVEL = 0.03014
MAX_SQUEEZE = 0.03
FINGER_LOW, FINGER_HIGH = -MAX_SQUEEZE, float(JOINT_HIGH[6])
READY_Q = (0.0, 0.94, 1.34, -1.40, 0.0, 0.0)  # x5/scene_lib.py READY_Q

ARM_LOW = np.asarray(JOINT_LOW[:6], dtype=np.float32)
ARM_HIGH = np.asarray(JOINT_HIGH[:6], dtype=np.float32)
assert np.allclose(ARM_LOW, ARM_LOWER) and np.allclose(ARM_HIGH, ARM_UPPER)
ACTION_LOW = np.tile(np.r_[ARM_LOW, FINGER_LOW, FINGER_LOW].astype(np.float32), 2)
ACTION_HIGH = np.tile(np.r_[ARM_HIGH, FINGER_HIGH, FINGER_HIGH].astype(np.float32), 2)
# The widest the jaws open: both at their open stop, 60.3 mm. It bounds the `gripper_width`
# observation, so a policy reads the real range from the observation space. Until 2026-11-10 it
# was the lab robot's 0.088, which made an agent believe 88 mm and grasp a 56 mm glass with 2 mm
# to spare per side (household A2, every run).
MAX_GRIPPER_WIDTH = 2 * JAW_TRAVEL


def native_to_action(native):
    """One arm's 8 native DOF values (joint1..6, jawP, jawN) to the action convention."""
    native = np.asarray(native, dtype=np.float64)
    out = native.copy()
    out[6] = JAW_TRAVEL + native[6]
    out[7] = JAW_TRAVEL - native[7]
    return out


def action_to_native(action):
    """One arm's 8 action values to native DOF targets (inverse of ``native_to_action``)."""
    action = np.asarray(action, dtype=np.float64)
    out = action.copy()
    out[6] = action[6] - JAW_TRAVEL
    out[7] = JAW_TRAVEL - action[7]
    return out


def ready_action(arm_joints=READY_Q):
    """The 16 targets applied at reset: ready pose (or `arm_joints`), jaws open (native 0)."""
    arm = native_to_action([*arm_joints, 0.0, 0.0])
    return np.tile(arm, 2)


def start_joints(config):
    """Arm joints 1-6 at reset: READY_Q, or all 0 under robot.start_pose: zero."""
    from roboclock.envs.start_pose import ARM_DOF, zero_start

    return (0.0,) * ARM_DOF if zero_start(config) else READY_Q


def robot_configuration(config=None):
    from roboclock.envs.start_pose import start_pose

    config = config or {}
    return {
        "start_pose": start_pose(config),
        "start_joints_rad": list(start_joints(config)),
        "model": ROBOT_ID,
        "joint_order": list(JOINT_ORDER),
        "arm_joint_limits_rad": [list(ARM_LOWER), list(ARM_UPPER)],
        "finger_convention": "per-finger opening from the closed stop in metres (lab convention); "
        "native jawP = opening - 0.03014, native jawN = 0.03014 - opening",
        "finger_action_limits_m": [FINGER_LOW, FINGER_HIGH],
        "finger_travel_m": JAW_TRAVEL,
        "finger_note": "below 0: target past the closed stop, the 20 N jaw drive clamps the part; "
        "above 0.03014: presses against the open stop",
        "ready_joints_rad": list(READY_Q),
        "drives": "PhysX position drives: arm stiffness 3000, damping 90, SDK effort limits "
        "[30, 40, 30, 15, 10, 10] N m; jaws stiffness 800 N/m, damping 8, 20 N per side",
        "mounts": "bases at x=-0.06 / 0.66 m, y=-0.255 m on the island counter, yaw +90 deg",
    }
