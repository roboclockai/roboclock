"""B2 coach display: one rigid screen group placed inside the RoboDojo head camera's view.

The screen (case, face, live cue labels and the response-window bar) is authored
in its design frame: centred at ``DESIGN_ORIGIN`` with the text facing -y. The
group ``PATH`` maps that frame into the ring: scaled by ``SCALE``, turned by
``yaw_deg()`` about +z and tilted back by ``pitch_deg()`` so that it faces the
rig's head camera, centred at ``CENTER``. The RoboDojo head camera
(``roboclock.envs.robodojo_rig``: rig frame (0.04, 0, 0.543) m, 60 deg down) sees
nothing higher than about 20 deg below its horizon, and the pads and gloves fill the
upper image, so the screen is a floor monitor in front of the robot plinth, between
and well below the arms' guard and strike paths, tilted up toward the head camera.
``screen_corners`` gives the case corners in world coordinates for the checks.

The screen and its post are visual geometry (no collider, no mass).
"""

import math

import numpy as np

from roboclock.envs import robodojo_rig

PATH = "/World/CoachDisplay"
# Design frame of the screen: case centre, text facing -y.
DESIGN_ORIGIN = (0.65, 0.61, 1.525)
# Case (equipment panel + bezel) half extents in the design frame.
HALF_WIDTH, HALF_HEIGHT, DEPTH = 0.4625, 0.2275, 0.024
# Placement in the ring (B2 rig: base midpoint (0.05, 0, 0.92), yaw 0).
RIG_ORIGIN = (0.05, 0.0, 0.92)
CENTER = (0.40, 0.0, 0.50)
SCALE = 0.6
STAND_RADIUS = 0.014


def head_eye():
    return robodojo_rig.head_camera_world(RIG_ORIGIN, 0.0)[:3, 3]


def yaw_deg():
    """Turn about +z that points the text's horizontal normal at the head camera."""
    d = head_eye() - np.asarray(CENTER)
    # Design normal -y turned by yaw: (sin yaw, -cos yaw).
    return math.degrees(math.atan2(d[0], -d[1]))


def pitch_deg():
    """Backward tilt that points the text normal up at the head camera."""
    d = head_eye() - np.asarray(CENTER)
    return math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))


def _rotation():
    yaw, pitch = math.radians(yaw_deg()), math.radians(-pitch_deg())
    rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    return rz @ rx


def to_world(points):
    """Design-frame points to world coordinates."""
    local = np.atleast_2d(np.asarray(points, float)) - DESIGN_ORIGIN
    return np.asarray(CENTER) + SCALE * local @ _rotation().T


def facing():
    """Unit direction the screen text faces (design -y)."""
    return _rotation() @ np.array([0.0, -1.0, 0.0])


def screen_corners():
    """World corners of the screen case front (bezel included)."""
    x, y, z = DESIGN_ORIGIN
    return to_world(
        [
            [x + dx * HALF_WIDTH, y, z + dz * HALF_HEIGHT]
            for dx, dz in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )


def stand():
    """(x, y) of the post behind the screen's lower edge, and that edge's height z."""
    corners = screen_corners()
    bottom = float(corners[:, 2].min())
    back = facing().copy()
    back[2] = 0
    back /= np.linalg.norm(back)
    xy = np.asarray(CENTER[:2]) - back[:2] * (SCALE * DEPTH + STAND_RADIUS + 0.03)
    return xy, bottom


def frame(b):
    """The screen group's Xform; children are authored in design coordinates."""
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xform.Define(b.stage, PATH)
    if not xf.GetOrderedXformOps():
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, CENTER)))
        xf.AddRotateZOp().Set(float(yaw_deg()))
        xf.AddRotateXOp().Set(float(-pitch_deg()))
        xf.AddScaleOp().Set(Gf.Vec3f(float(SCALE)))
        xf.AddTranslateOp(opSuffix="design").Set(Gf.Vec3d(*(-float(v) for v in DESIGN_ORIGIN)))
        xf.GetPrim().SetCustomDataByKey(
            "benchmark:provenance",
            "B2 coach screen facing the RoboDojo head camera; visual only",
        )
    return PATH
