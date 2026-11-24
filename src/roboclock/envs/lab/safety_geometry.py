"""Meter-scale joint geometry shared by layout, rigid bodies and diagnostic control."""

import numpy as np

# B5's ultra-low freezer was designed with its door facing +y, centred at DESIGN_XY.
# It stands at FREEZER_XY, turned FREEZER_YAW about +z: the door faces -x, toward the
# rig, centred in front of it (rig origin y = 0.16). Every freezer coordinate in
# safety_scene / safety_instruments is written in the design frame and mapped here.
# FREEZER_XY / FREEZER_YAW and every point below are NOMINAL: each seed places the whole
# apparatus by layout.apparatus (layout.ApparatusFrame), passed here as `frame`. Scene prims
# stay nominal under /World/FaultRig, which carries that frame; world consumers (layout,
# joints, oracle, cues, clutter) pass the frame.
DESIGN_XY = np.array([0.465, -0.24])
FREEZER_XY = np.array([0.755, 0.16])
FREEZER_YAW = np.pi / 2


def freezer_point(point):
    """World point of a point given in the freezer's design frame (z unchanged)."""
    point = np.asarray(point, dtype=float)
    xy = rotated(np.r_[point[:2] - DESIGN_XY, 0.0], "Z", FREEZER_YAW)[:2] + FREEZER_XY
    return np.r_[xy, point[2:]]


def freezer_footprint(frame=None):
    """(centre, half extents) in world xy of the freezer and its door swing toward the rig.

    With a placement `frame` the extents grow to the axis-aligned box of the yawed footprint.
    """
    centre, half = np.array([FREEZER_XY[0] - 0.08, FREEZER_XY[1]]), np.array([0.33, 0.20])
    if frame is None:
        return centre, half
    c, s = abs(np.cos(frame.yaw)), abs(np.sin(frame.yaw))
    return frame.point(centre), np.array([c * half[0] + s * half[1], s * half[0] + c * half[1]])


def mechanism(task_id, table_z=0.75, frame=None):
    """World joint geometry of the B5-B7 actuator, placed by the layout's apparatus `frame`."""
    from roboclock.envs.lab.layout import IDENTITY_FRAME

    frame = frame or IDENTITY_FRAME
    if task_id == "B5":
        # The door is hinged on the rig's left (+y) so its handle is on the side of the
        # task arm (rig right). Opening swings the handle toward -x: a rotation about -z
        # (turn = -1); `angle` and `open_angle` stay positive when open.
        return dict(
            axis="Z",
            pivot=frame.point(freezer_point([0.61, -0.06, table_z + 0.45])),
            offset=frame.vector(rotated(np.array([-0.235, 0.055, 0.0]), "Z", FREEZER_YAW)),
            yaw0=FREEZER_YAW + frame.yaw,
            turn=-1.0,
            angle=0.0,
            open_angle=np.deg2rad(24),
            mass=1.0,
        )
    return dict(
        axis="Z",
        pivot=frame.point([0.40, 0.16, table_z + 0.15]),
        offset=np.zeros(3),
        yaw0=frame.yaw,
        turn=1.0,
        angle=0.0,
        mass=0.06,
    )


def rotated(vector, axis, angle):
    c, s = np.cos(angle), np.sin(angle)
    matrix = (
        np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        if axis == "X"
        else np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    )
    return matrix @ vector


def body_yaw(spec, angle):
    """World yaw of a z-hinged actuator body at the given (positive-open) angle."""
    return spec.get("yaw0", 0.0) + spec.get("turn", 1.0) * angle


def hinge_angle(spec, world_yaw):
    """Opening angle of a z-hinged actuator from its body's measured world yaw."""
    error = (world_yaw - spec.get("yaw0", 0.0) + np.pi) % (2 * np.pi) - np.pi
    return spec.get("turn", 1.0) * error


def grip_point(spec, angle):
    """World position of the actuator body origin (B5: the door handle) at an angle."""
    return spec["pivot"] + rotated(spec["offset"], spec["axis"], spec.get("turn", 1.0) * angle)


def body_hinge_offset(spec):
    """Hinge point in the actuator body's own frame (the joint's localPos1)."""
    return rotated(-spec["offset"], spec["axis"], -spec.get("yaw0", 0.0))


def joint_frames(spec):
    """(localRot0 in world, localRot1 in the body), each wxyz, for the actuator joint.

    Both carry the body's closed yaw; turn = -1 adds a half turn about x, which makes the
    joint's positive direction a rotation about world -z.
    """
    from scipy.spatial.transform import Rotation

    flip = Rotation.from_euler("x", np.pi if spec.get("turn", 1.0) < 0 else 0.0)
    rot0 = Rotation.from_euler("z", spec.get("yaw0", 0.0)) * flip
    return rot0.as_quat()[[3, 0, 1, 2]], flip.as_quat()[[3, 0, 1, 2]]


def actuator_pose(task_id, table_z, frame=None):
    spec = mechanism(task_id, table_z, frame)
    if spec["axis"] == "Z":
        yaw = body_yaw(spec, spec["angle"])
        q = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    else:
        q = np.array([np.cos(spec["angle"] / 2), np.sin(spec["angle"] / 2), 0.0, 0.0])
    return grip_point(spec, spec["angle"]), q
