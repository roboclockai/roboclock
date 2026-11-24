"""The 14 world-frame dimensions `move_eef` names: per arm x, y, z, three angles and the gripper.

This is RoboDojo's surface (XPolicyLab policy/GPT_6_Astra_Direct_EEF/pose.py). A position is
the tool point, the point between the jaws where an object is held. The orientation is three
angles in degrees measured from a straight-down grasp, not absolute roll/pitch/yaw: a
straight-down hand is the commonest orientation, and as absolute extrinsic angles it would sit
on the gimbal lock. At 0, 0, 0 the fingers point at world -z and the jaws close along world
+x. The angles are extrinsic rotations applied to that reference: pitch about world +x, then
roll about world +y, then yaw about world +z.
"""

import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.policies.vision_llm import SIDES
from roboclock.policies.vision_llm_primitives.geometry import top_rotation

REFERENCE = top_rotation(0.0)  # tool frame (z along the fingers, y across the jaws) of the zero orientation
POSITION_AXES = ("x", "y", "z")
ANGLE_AXES = ("pitch_deg", "roll_deg", "yaw_deg")
# The canonical Euler domain: the middle angle within +-90 covers every orientation once.
ANGLE_BOUNDS = {"pitch_deg": (-180.0, 180.0), "roll_deg": (-90.0, 90.0), "yaw_deg": (-180.0, 180.0)}
GRIPPER_BOUNDS = (0.0, 1.0)
REACH = 0.80  # m from its base a tool point may be named; the bounds only reject gross errors


def labels():
    """The 14 dimension names, left arm first, in the order the tool and the state give them."""
    return tuple(f"{side}_{axis}" for side in SIDES for axis in (*POSITION_AXES, *ANGLE_AXES, "gripper"))


def bounds(label, bases):
    """[low, high] of one dimension; `bases` are the two arms' base positions."""
    side, _, axis = label.partition("_")
    if axis == "gripper":
        return GRIPPER_BOUNDS
    if axis in ANGLE_BOUNDS:
        return ANGLE_BOUNDS[axis]
    centre = float(bases[SIDES.index(side)][POSITION_AXES.index(axis)])
    return (round(centre - REACH, 2), round(centre + REACH, 2))


def bounds_text(bases):
    return ", ".join(f"{label}: [{low:g}, {high:g}]" for label in labels() for low, high in [bounds(label, bases)])


def angles_to_rotation(pitch_deg, roll_deg, yaw_deg):
    """Tool rotation matrix of one angle triple."""
    turn = Rotation.from_euler("xyz", [pitch_deg, roll_deg, yaw_deg], degrees=True)  # extrinsic x, y, z
    return turn.as_matrix() @ REFERENCE


def rotation_to_angles(rotation):
    """(pitch_deg, roll_deg, yaw_deg) of a tool rotation matrix. At roll +-90 pitch and yaw are
    coupled; yaw is then 0 and pitch carries the rest, so the triple still round-trips."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's gimbal-lock notice
        pitch, roll, yaw = Rotation.from_matrix(np.asarray(rotation) @ REFERENCE.T).as_euler("xyz", degrees=True)
    return float(pitch) + 0.0, float(roll) + 0.0, float(yaw) + 0.0


def pose_to_values(pose):
    """The six named values of a 4x4 world tool pose."""
    pitch, roll, yaw = rotation_to_angles(pose[:3, :3])
    x, y, z = (float(value) for value in pose[:3, 3])
    return {"x": x, "y": y, "z": z, "pitch_deg": pitch, "roll_deg": roll, "yaw_deg": yaw}


def values_to_pose(values):
    """The 4x4 world tool pose of six named values."""
    pose = np.eye(4)
    pose[:3, :3] = angles_to_rotation(values["pitch_deg"], values["roll_deg"], values["yaw_deg"])
    pose[:3, 3] = [values["x"], values["y"], values["z"]]
    return pose


def angle_between(one, other):
    """Angle in degrees between two rotation matrices."""
    return float(np.degrees(Rotation.from_matrix(np.asarray(one) @ np.asarray(other).T).magnitude()))
