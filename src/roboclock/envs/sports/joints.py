"""The shared 16-dim Dual ARX X5 joint layout and the sports X5A native articulation.

The benchmark action is ``[left joint1..joint8, right joint1..joint8]``: six arm
radians and two finger meters per arm, bounded by ``roboclock.envs.lab.robot``
``JOINT_LOW``/``JOINT_HIGH``. The sports scenes load the manufacturer X5A pair
(``common/assets/dual_arx_x5``) as one articulation whose DOFs are named
``left_joint1`` ... ``right_joint8`` and ordered by PhysX
(``left_joint1, right_joint1, left_joint2, ...``). The mapping is a pure
permutation by name: same joint, same unit, same sign and zero.

Native X5A limits differ slightly from the laboratory X5 limits that bound the
action space (radians / meters):

=========  ====================  ===================
joint      native X5A            benchmark action
=========  ====================  ===================
joint1     [-2.618, 3.1416]      [-3.14, 2.618]
joint2     [0, 3.665]            [-0.05, 3.50]
joint3     [0, 3.1416]           [-0.10, 3.20]
joint4     [-1.571, 1.571]       [-1.60, 1.55]
joint5     [-1.571, 1.571]       [-1.57, 1.57]
joint6     [-2.094, 2.094]       [-2.0, 2.0]
joint7/8   [0, 0.044]            [0, 0.044]
=========  ====================  ===================

Targets inside the action bounds but outside the native range are clipped by the
native drive (``RobotRig.apply``). Native ranges outside the action bounds
(joint1 > 2.618, joint2 > 3.50, joint4 > 1.55, abs(joint6) > 2.0) cannot be
commanded. None of the ten recorded oracle episodes commands them; their
measured positions only leave the bounds through finger noise below 1e-5 m.
Observed joint positions are therefore clipped to the action bounds, so every
observation is a valid hold action.

The oracles command each arm joint a bounded step from its measured position, and
the drive closes only part of that step before the next command. The arm speed a
limit reaches therefore depends on the control period: the X5 reaches 62% of it at
30 Hz, 44% at 50 Hz and 38% at 60 Hz (measured with ``limit * dt`` steps).
``joint_step_speed`` restates a limit tuned at another rate so the arm keeps its speed.
"""

import math

import numpy as np

SIDES = ("left", "right")
JOINT_ORDER = tuple(f"{side}/joint{i}" for side in SIDES for i in range(1, 9))
NATIVE_NAMES = tuple(f"{side}_joint{i}" for side in SIDES for i in range(1, 9))


# X5 arm drive of ``RobotRig``: damping 65 over stiffness 1800 (s).
ARM_DRIVE_TIME_CONSTANT = 65 / 1800


def joint_step_speed(speed, tuned_hz, control_dt):
    """Step rate limit that reaches at ``control_dt`` the arm speed ``speed`` reached at ``tuned_hz``."""

    def closed(period):
        return 1 - math.exp(-period / ARM_DRIVE_TIME_CONSTANT)

    return speed * closed(1 / tuned_hz) / closed(control_dt)


def action_bounds():
    from roboclock.envs.lab.robot import JOINT_HIGH, JOINT_LOW

    return (
        np.tile(JOINT_LOW, 2).astype(np.float32),
        np.tile(JOINT_HIGH, 2).astype(np.float32),
    )


class JointMap:
    """Permutation between a native X5A DOF vector and the benchmark layout."""

    def __init__(self, dof_names):
        names = list(dof_names)
        missing = [name for name in NATIVE_NAMES if name not in names]
        if missing:
            raise ValueError(f"Native articulation lacks X5 joints: {', '.join(missing)}")
        self.native_names = names
        self.index = np.array([names.index(name) for name in NATIVE_NAMES], dtype=int)
        self.low, self.high = action_bounds()

    def to_unified(self, native):
        return np.asarray(native, dtype=np.float64)[self.index]

    def to_native(self, unified, template):
        """Scatter benchmark targets into a copy of a full native command vector.

        The template's dtype is kept: native rig commands are float32 arrays, so a
        float32 action round-trips exactly.
        """
        native = np.array(template, copy=True)
        native[self.index] = np.asarray(unified, dtype=native.dtype)
        return native

    def observe(self, native):
        """Measured positions in benchmark order, clipped to the action bounds."""
        return np.clip(self.to_unified(native), self.low, self.high).astype(np.float32)

    def describe(self):
        return {
            unified: self.native_names[i] for unified, i in zip(JOINT_ORDER, self.index.tolist())
        }
