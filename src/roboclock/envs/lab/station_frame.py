"""Seeded rigid placement of the A1-A6 workstations relative to the robot rig.

Every A1-A6 station is authored in a fixed *nominal* frame (the constants in liquid_scene,
wet_bench, titration_visuals and bath_scene). Per episode a layout.ApparatusFrame maps that
nominal frame to the world: a rotation by `yaw` about `anchor`, then a shift by `offset` (xy
only; heights are unchanged). The rig itself stays where configs/benchmark.yaml puts it, so
the station moves relative to the robot.

One source of truth: the frame is drawn once by the task's layout builder (liquid_layout,
bath_layout) and stored as `layout.apparatus`, the field the B/C tasks use too (it is written
to the episode metadata as `apparatus_placement`). Everything that depends on the station
reads it from there (`station_of`):

- scene builders author the station's prims under roots placed with `place` / `author_xform`,
  so the nominal coordinates written by the builders and by their per-frame updates stay local;
- rigid bodies and joints (world-frame APIs) use `frame.point` / `quat`;
- layout objects/targets, scoring geometry, clutter keep-outs, cameras and the oracles use
  `frame.point` on the nominal constants.

The draw uses its own SeedSequence([seed, STREAMS[task_id]]), so it depends on the episode
seed only (same layout at every difficulty) and leaves every other seeded draw unchanged.
"""

import numpy as np

from roboclock.envs.lab.layout import ApparatusFrame, apparatus_frame

# Dedicated seed streams (distinct from layout.py's 101 and the clutter's 202).
STREAMS = {"A1": 7311, "A2": 7312, "A3": 7313, "A4": 7314, "A5": 7315, "A6": 7316}
# Uniform draw ranges: (x low, x high) m, (y low, y high) m, yaw half-range deg. A1-A4 are not
# yawed: their valves' angle readings, the A3 collector rail (a world-x prismatic joint) and the
# oracles' wrist-yaw commands are all defined in world axes. The A5/A6 baths are static and
# their samples are cylinders, so a small yaw needs only the rotated bath centres.
# A3: at most 1 cm toward the rig. The fourth fraction tube is drawn under the outlet with the
# collector grip 0.30 m nearer the rig than at the start, a few cm in front of the working
# arm's base; with the station 2.6 cm nearer (seed 3, 2026-11-20) the arm touched a fixture in
# every difficulty and on hard jammed short of the waste position (fraction timeout).
RANGES = {
    "A1": ((-0.03, 0.03), (-0.03, 0.03), 0.0),
    "A2": ((-0.03, 0.03), (-0.03, 0.03), 0.0),
    "A3": ((-0.01, 0.03), (-0.03, 0.03), 0.0),
    "A4": ((-0.03, 0.03), (-0.03, 0.03), 0.0),
    "A5": ((-0.03, 0.03), (-0.03, 0.03), 6.0),
    "A6": ((-0.03, 0.03), (-0.03, 0.03), 6.0),
}
# Yaw anchor (nominal xy): the midpoint between the two baths.
ANCHORS = {"A5": (0.48, 0.01), "A6": (0.48, 0.01)}


def draw_station_frame(task):
    """The episode's station placement (episode seed only; independent of difficulty)."""
    if task.task_id not in RANGES:
        return ApparatusFrame()
    stream = np.random.SeedSequence([int(task.spec.seed), STREAMS[task.task_id]])
    (x_low, x_high), (y_low, y_high), yaw = RANGES[task.task_id]
    draw = np.random.default_rng(stream).uniform(-1.0, 1.0, 3)
    return ApparatusFrame(
        np.array(
            [
                (x_low + x_high) / 2 + (x_high - x_low) / 2 * draw[0],
                (y_low + y_high) / 2 + (y_high - y_low) / 2 * draw[1],
            ]
        ),
        float(np.radians(yaw * draw[2])),
        np.array(ANCHORS.get(task.task_id, (0.0, 0.0)), dtype=float),
    )


def station_of(holder):
    """The station placement of a layout, or of a task/backend's layout (identity if none)."""
    return apparatus_frame(getattr(holder, "layout", holder))


def place(frame, xformable):
    """Prepend `frame`'s placement to a UsdGeom.Xformable whose children (or own later ops)
    are authored in nominal coordinates. The ops are written even for the identity."""
    from pxr import Gf

    xformable.AddTranslateOp(opSuffix="apparatusPlace").Set(
        Gf.Vec3d(*map(float, frame.anchor + frame.offset), 0.0)
    )
    xformable.AddRotateZOp(opSuffix="apparatusYaw").Set(float(np.degrees(frame.yaw)))
    xformable.AddTranslateOp(opSuffix="apparatusAnchor").Set(
        Gf.Vec3d(*map(float, -frame.anchor), 0.0)
    )
    return xformable


def root(frame, stage, path):
    """Define the Xform `path` placed by `frame`; build nominal children under it."""
    from pxr import UsdGeom

    return place(frame, UsdGeom.Xform.Define(stage, path))


def quat(frame, q=(1.0, 0.0, 0.0, 0.0)):
    """World orientation (w, x, y, z) of nominal orientation `q`: the frame's yaw times `q`."""
    w, x, y, z = q
    c, s = np.cos(frame.yaw / 2), np.sin(frame.yaw / 2)
    return np.array([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], dtype=float)
