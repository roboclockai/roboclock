"""The RoboDojo dual ARX X5 rig of every hospital task (no simulator imports).

Every value relative to the rig comes from the shared spec `roboclock.envs.robodojo_rig`
(docs/robodojo-rig-spec.md): two X5 bases side by side 0.60 m apart at one height and one forward
coordinate, both facing the rig's +x; the head camera (the policy's `rgb`, RoboDojo `cam_head`) at
(0.04, 0, 0.543) m in the rig frame, pitched 60 degrees down, with the Gemini 345Lg intrinsics; one
D435 wrist camera rigid on each arm's `link6`. This module keeps no copies of those values.

hospital places the rig per task (`robot.rig.default` / `robot.rig.tasks.<ID>`: `origin` = the
midpoint of the two bases on the mount plane, optional `yaw` in radians). Internal arm 0, the arm
single-arm tasks drive (`/World/ARXLeft` despite its name), stands on the rig's RIGHT (-y at yaw
0); internal arm 1 (`/World/ARXRight`) on its LEFT. The spec's left wrist camera
(`left_wrist_rgb`) therefore rides arm 1 and `right_wrist_rgb` rides arm 0. The env interface
(LabEnv.step / get_obs) lists the rig's LEFT arm first: action and joint_positions entries 0-7
drive internal arm 1, entries 8-15 internal arm 0 (roboclock.envs.lab.robot.EXTERNAL_ARMS).
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboclock.envs import robodojo_rig as spec

# Internal arm index (lab ROOTS order) -> physical spec side. Arm 0 is the rig's right-hand arm.
# (The env interface is in spec order instead: left first, roboclock.envs.lab.robot.EXTERNAL_ARMS.)
ARM_SIDES = ("right", "left")
WRIST_KEYS = {"right": "right_wrist_rgb", "left": "left_wrist_rgb"}
WRIST_DEPTH_KEYS = {"right": "right_wrist_depth", "left": "left_wrist_depth"}
RIG_KEYS = {"default", "tasks"}  # anything else would be a local copy of a spec value
X5A_URDF = (
    Path(__file__).resolve().parents[4]
    / "assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf"
)


def rig_pose(config, task_id):
    """(origin, yaw radians) of the rig for one task."""
    rig = config["robot"]["rig"]
    pose = {**rig.get("default", {}), **rig.get("tasks", {}).get(task_id, {})}
    return np.asarray(pose["origin"], dtype=float), float(pose.get("yaw", 0.0))


def rig_mounts(config, task_id):
    """Internal arm-0 (rig right) and arm-1 (rig left) base positions and their common yaw
    (radians). Internal order: the env interface lists arm 1 (rig left) first."""
    origin, yaw = rig_pose(config, task_id)
    left, right = spec.base_positions(origin, math.degrees(yaw))
    return [np.asarray(right, dtype=float), np.asarray(left, dtype=float)], yaw


def head_camera_world(config, task_id):
    """4x4 world pose (USD camera axes) of the head camera."""
    origin, yaw = rig_pose(config, task_id)
    return spec.head_camera_world(origin, math.degrees(yaw))


def head_camera_pose(config, task_id):
    """World position, a look-at point 1 m along the optical axis and the pitch (radians)."""
    pose = head_camera_world(config, task_id)
    position = pose[:3, 3].copy()
    return position, position - pose[:3, 2], math.radians(spec.HEAD_PITCH_DOWN_DEG)


def dial_target(backend, fallback):
    """Point a task's countdown dial turns to: the rig's head camera (the policy's `rgb`), or
    `fallback` (the task's third-person camera) for configs without `robot.rig`."""
    if "rig" not in backend.config.get("robot", {}):
        return np.asarray(fallback, dtype=float)
    task_id = getattr(getattr(backend, "task", None), "task_id", None)
    task_id = task_id or getattr(getattr(backend, "layout", None), "task_id", None)
    if task_id is None:  # a bare authoring test double: no task to place the rig for
        return np.asarray(fallback, dtype=float)
    return head_camera_pose(backend.config, task_id)[0]


def head_pixels(config, task_id, points, conservative=False):
    """Pinhole projection of world points into the 640x480 head image: (N, 2) pixels (u right,
    v down) and the depth along the optical axis. Uses the square-pixel intrinsics the renderer
    produces (`robodojo_rig.intrinsics`, 96.0 x 79.6 deg). `conservative=True` projects with
    the aperture model instead (`square_pixels=False`: fy from the vertical aperture, 71.0 deg,
    what `get_intrinsics_matrix()` reports), which is stricter vertically: a point inside that
    image is also inside the rendered one."""
    pose = head_camera_world(config, task_id)
    k = spec.intrinsics(spec.HEAD_CAMERA, square_pixels=not conservative)
    d = np.atleast_2d(np.asarray(points, dtype=float)) - pose[:3, 3]
    x, y, z = (d @ pose[:3, i] for i in range(3))  # USD camera axes: look along -Z, image up +Y
    depth = -z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k[0, 2] + k[0, 0] * x / depth
        v = k[1, 2] - k[1, 1] * y / depth
    return np.stack([u, v], axis=1), depth


def check_rig(config, task_ids):
    """Config-level check: the rig block only places the rig, and every task's placement gives
    the spec rig (bases, head camera) through `robodojo_rig.check_rig`. Also confirms the lab
    robot builder (`lab.robot.mount_positions`, which creates the arms) agrees."""
    from roboclock.envs.lab.robot import mount_positions

    robot = config["robot"]
    rig = robot.get("rig")
    if rig is None:
        raise ValueError("robot.rig (the common dual-X5 rig) is required")
    if set(robot) & {"mount_positions", "task_mount_positions", "task_mount_yaws"}:
        raise ValueError("per-arm mounts are replaced by robot.rig")
    local = set(rig) - RIG_KEYS
    if local:
        raise ValueError(
            f"robot.rig keeps only the placement (default/tasks); {sorted(local)} are "
            "robodojo_rig values and must not be copied into the config"
        )
    unknown = set(rig.get("tasks", {})) - set(task_ids)
    if unknown:
        raise ValueError(f"robot.rig.tasks has unknown task ids: {sorted(unknown)}")
    for task_id in task_ids:
        (right, left), yaw = rig_mounts(config, task_id)
        frames = [spec.rig_to_world(p, math.degrees(yaw)) for p in (left, right)]
        try:
            spec.check_rig(*frames, head_camera_world(config, task_id))
        except ValueError as error:
            raise ValueError(f"{task_id}: {error}") from None
        built = np.asarray(mount_positions(config, task_id), dtype=float)
        if not np.allclose(built, [right, left], atol=1e-9):
            raise ValueError(f"{task_id}: lab.robot.mount_positions disagrees with the rig spec")


# --- measured checks at reset -------------------------------------------------------------------


def _x5a_chain():
    """(origin 4x4, axis) of X5A.urdf joint1..joint6, base_link -> link6."""
    joints = {j.get("name"): j for j in ET.parse(X5A_URDF).getroot().iter("joint")}
    chain, link = [], "base_link"
    for index in range(1, 7):
        joint = joints[f"joint{index}"]
        if joint.find("parent").get("link") != link:
            raise ValueError(f"X5A.urdf joint{index} does not continue the chain at {link}")
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        rpy = [float(v) for v in origin.get("rpy").split()]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        chain.append((spec.transform(xyz, spec.urdf_rpy(*rpy)), axis))
        link = joint.find("child").get("link")
    if link != spec.WRIST_PARENT_LINK:
        raise ValueError(f"X5A.urdf chain ends at {link}")
    return chain


def x5a_link6_in_base(joints):
    """X5A.urdf forward kinematics: 4x4 `link6` pose in `base_link` for six joint angles."""
    pose = np.eye(4)
    for (origin, axis), angle in zip(_x5a_chain(), joints[:6]):
        a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        rotation = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
        pose = pose @ origin @ spec.transform(np.zeros(3), rotation)
    return pose


def pose_error(a, b):
    """(position error m, rotation error deg) between two 4x4 poses."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(math.acos(cosine))


def check_measured(task_id, bases, head, link6, joints, wrists, *, tol_m=1e-3, tol_deg=0.1):
    """Spec check on poses measured from the stage at reset; raises on any deviation.

    `bases`, `link6`, `joints`: per arm (arm 0 = rig right). `head`: the head camera's measured
    world pose, or None when sim.no_render built no camera (then the head mount frame is passed).
    `wrists`: per arm the measured wrist camera world pose, or None without cameras."""
    left, right = bases[1], bases[0]
    result = {"task_id": task_id, **spec.check_rig(left, right, head, tolerance_m=tol_m, tolerance_deg=tol_deg)}
    result["bases"] = {s: np.asarray(bases[i])[:3, 3].round(6).tolist() for i, s in enumerate(ARM_SIDES)}
    result["head_position"] = np.asarray(head)[:3, 3].round(6).tolist()
    result["head_orientation_wxyz"] = spec.quat_wxyz(np.asarray(head)[:3, :3]).round(6).tolist()
    result["link6_vs_x5a_urdf"], result["wrist_cameras"] = {}, {}
    for i, side in enumerate(ARM_SIDES):
        expected = np.asarray(bases[i]) @ x5a_link6_in_base(joints[i])
        dp, dr = pose_error(link6[i], expected)
        result["link6_vs_x5a_urdf"][side] = {"position_m": dp, "rotation_deg": dr}
        if dp > tol_m or dr > tol_deg:
            raise ValueError(f"{task_id}: {side} link6 is {dp:.4f} m / {dr:.3f} deg from X5A.urdf")
        if wrists[i] is None:
            result["wrist_cameras"][side] = None
            continue
        dp, dr = pose_error(wrists[i], np.asarray(link6[i]) @ spec.wrist_camera_in_link6())
        result["wrist_cameras"][side] = {"position_m": dp, "rotation_deg": dr}
        if dp > tol_m or dr > tol_deg:
            raise ValueError(f"{task_id}: {side} wrist camera is {dp:.4f} m / {dr:.3f} deg off link6 spec")
    result["tolerance"] = {"position_m": tol_m, "rotation_deg": tol_deg}
    return result
