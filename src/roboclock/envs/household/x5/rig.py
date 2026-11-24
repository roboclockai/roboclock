"""The Dual-X5 rig: arm base mounts and the head camera rigidly placed relative to them.

Common rig = the RoboDojo-aligned rig (``roboclock.envs.robodojo_rig``, docs/robodojo-rig-spec.md),
the single source of every value below: both arm bases side by side on the island counter, same
height, same forward coordinate, both facing +y (yaw +90 deg), 0.60 m centre to centre. The rig
frame's origin is the base midpoint on the mount plane, (0.30, -0.255, 0.8704), yawed 90 deg.
The head camera (``/World/AgentCamera``, the policy's ``rgb``) sits at (0.04, 0, 0.543) m in the
rig frame, pitched 60 deg down, with the Gemini 345Lg intrinsics; one pose and lens for every task.

Only tasks listed in ``RIG_TASKS`` use it; any other task (C1 for now) keeps the earlier 0.72 m
layout (bases at x = -0.06 / 0.66 m) and its own per-task camera until it is added there.
``use(task_id)`` switches the shared ``BASES`` mapping in place (``scene_lib``, ``kinematics``, ``sim/physics`` and
``sim/servo`` all read this one dict), so call it before building a task's scene, physics overlay
or servos. Pure Python: importable without OpenUSD or Isaac Sim.
"""

from roboclock.envs import robodojo_rig as RD

COUNTER_TOP = .8584  # island counter top (m)
MOUNT_THICKNESS = .012
MOUNT_TOP = COUNTER_TOP + MOUNT_THICKNESS  # arm base_link origin height
BASE_YAW_DEG = 90.  # both arms face +y (the back wall)

SEPARATION = RD.ARM_SEPARATION_M  # m, centre to centre along x
CENTER = (.30, -.255)  # midpoint of the two bases (x, y)
HEAD_FORWARD = RD.HEAD_IN_RIG_M[0]  # m ahead (+y) of the base midpoint
HEAD_HEIGHT = RD.HEAD_IN_RIG_M[2]  # m above the top of the base mounts
HEAD_PITCH_DEG = RD.HEAD_PITCH_DOWN_DEG  # below horizontal
HEAD_FOCAL_MM = RD.HEAD_CAMERA['focal_length_mm']
HEAD_APERTURE_MM = (RD.HEAD_CAMERA['horizontal_aperture_mm'], RD.HEAD_CAMERA['vertical_aperture_mm'])

COMMON = {'Left': (CENTER[0] - SEPARATION / 2, CENTER[1]), 'Right': (CENTER[0] + SEPARATION / 2, CENTER[1])}
LEGACY = {'Left': (-.06, -.255), 'Right': (.66, -.255)}  # 0.72 m apart
# C2 joined 2026-11-03 (screened for this rig); C1 joined with a smaller, turned microwave (R2.2; R3 has no microwave).
RIG_TASKS = frozenset({'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3',
                       'ALARM'})  # ALARM: the alarm-clock promo scene (alarm_x5), not a benchmark task

BASES = dict(LEGACY)  # the active layout, mutated in place by use(); legacy until a task is chosen
_active = {'task': None}


def bases_for(task_id):
    return dict(COMMON if task_id in RIG_TASKS else LEGACY)


def on_common_rig(task_id):
    return task_id in RIG_TASKS


def use(task_id):
    """Make ``task_id``'s base layout the active one; returns the layout."""
    BASES.clear()
    BASES.update(bases_for(task_id))
    _active['task'] = task_id
    return dict(BASES)


def active_task():
    return _active['task']


def rig_origin(bases=None):
    """Rig frame origin (base midpoint on the mount plane, world m) and yaw (deg)."""
    b = bases or COMMON
    return ((b['Left'][0] + b['Right'][0]) / 2, (b['Left'][1] + b['Right'][1]) / 2, MOUNT_TOP), BASE_YAW_DEG


def head_camera_matrix(bases=None):
    """4x4 world pose of the head camera (USD camera axes) from ``robodojo_rig``."""
    return RD.head_camera_world(*rig_origin(bases))


def head_camera(bases=None):
    """Head camera on the rig: dict(eye, look, matrix, focal, h_ap, v_ap, clip), world metres."""
    m = head_camera_matrix(bases)
    eye = m[:3, 3]
    look = eye - m[:3, 2]  # USD cameras look along -Z
    return dict(eye=[round(float(v), 6) for v in eye], look=[round(float(v), 6) for v in look],
                matrix=m.tolist(), focal=HEAD_FOCAL_MM, h_ap=HEAD_APERTURE_MM[0], v_ap=HEAD_APERTURE_MM[1],
                clip=list(RD.HEAD_CAMERA['clipping_range_m']), resolution=list(RD.HEAD_CAMERA['resolution']))
