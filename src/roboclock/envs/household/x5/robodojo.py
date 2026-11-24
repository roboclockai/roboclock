"""The RoboDojo rig spec on a household Isaac stage: renderer, policy cameras, per-reset rig check.

Every value comes from ``roboclock.envs.robodojo_rig`` (docs/robodojo-rig-spec.md); nothing is
copied here. Used by the roboclock environment (``x5/episode.py``) and the recorder
(``scripts/household/run_x5_episode.py``), so both build the same cameras and run the same check.

- Head camera ``/World/AgentCamera`` (``cam_head`` -> ``rgb``): authored in the initial scene at the
  rig pose (``x5/rig.py``); here it gets RoboDojo's Camera API calls (focal length, both apertures
  with ``maintain_square_pixels=False``, clipping range).
- Wrist cameras (``cam_left_wrist`` / ``cam_right_wrist``): one USD camera per arm, a child of the
  arm's ``link6`` body at ``robodojo_rig.wrist_camera_in_link6()``, so it moves rigidly with the
  link. They are added to the opened simulation stage (not the initial scene, whose hash stays).
- ``rig_check``: measures both base frames (physics), the head camera (stage), each wrist camera
  relative to its measured ``link6`` (stage + physics), runs ``robodojo_rig.check_rig`` and the
  1 mm / 0.1 deg wrist check, and compares the household ``link6`` with X5A.urdf's ``link6``.

Isaac Sim only (import after SimulationApp starts), except the numpy helpers.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboclock.envs import robodojo_rig as RD

REPO = Path(__file__).resolve().parents[5]
X5A_URDF = REPO / "assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf"
HOUSEHOLD_URDF = Path(__file__).resolve().parents[1] / "assets/x5/robot/arx5-sdk/models/X5.urdf"
HEAD_PATH = "/World/AgentCamera"
WRIST_PRIM = "cam_wrist"  # child of <arm>/link6
SIDES = ("Left", "Right")  # Left = the rig's left (+y in the rig frame, world -x here)
VIEWS = {"head": "cam_head", "Left": "cam_left_wrist", "Right": "cam_right_wrist"}
RGB_KEYS = {view: RD.OBSERVATION_KEYS[name] for view, name in VIEWS.items()}
DEPTH_KEYS = {view: RD.DEPTH_KEYS[name] for view, name in VIEWS.items()}
INTRINSICS_NOTE = (
    "camera_intrinsics is what Isaac's get_intrinsics_matrix() reports (fy from the vertical "
    "aperture). RTX ignores the vertical aperture and renders square pixels, so the image "
    "follows camera_intrinsics_rendered (robodojo_rig.intrinsics: fy = fx, principal point at "
    "the centre). Use the rendered matrix for projection and depth unprojection."
)
AA_NAMES = {0: "Off", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}
LINK6_TOLERANCE = (1e-3, 0.1)  # m, deg (the spec's wrist tolerance)


def image_shape(view):
    width, height = (RD.HEAD_CAMERA if view == "head" else RD.WRIST_CAMERA)["resolution"]
    return height, width


def blank_observations(depth=False):
    """Zero images of the rendered shapes, for ``sim.no_render`` runs."""
    obs = {RGB_KEYS[v]: np.zeros((*image_shape(v), 3), np.uint8) for v in VIEWS}
    if depth:
        obs.update({DEPTH_KEYS[v]: np.zeros(image_shape(v), np.float32) for v in VIEWS})
    return obs


# ------------------------------------------------------------------------------ renderer
def apply_renderer():
    """RTX Real-Time + DLAA (Isaac Lab's call) + the quality preset; then force RENDER_MODE."""
    import carb
    import omni.replicator.core as rep

    rep.settings.set_render_rtx_realtime(antialiasing=RD.ANTIALIASING)
    settings = carb.settings.get_settings()
    # Kit 110's set_render_rtx_realtime selects RealTimePathTracing; the spec's mode is RaytracedLighting.
    settings.set("/rtx/rendermode", RD.RENDER_MODE)
    for key, value in RD.RENDER_SETTINGS.items():
        settings.set(key, value)
    return renderer_readback()


def renderer_readback():
    import carb

    settings = carb.settings.get_settings()
    op = settings.get("/rtx/post/aa/op")
    name = AA_NAMES.get(op, str(op))
    mismatched = {
        key: {"set": value, "read": settings.get(key)}
        for key, value in RD.RENDER_SETTINGS.items()
        if settings.get(key) != value
    }
    return {
        "rendermode": settings.get("/rtx/rendermode"),
        "rendermode_ok": settings.get("/rtx/rendermode") == RD.RENDER_MODE,
        "render_settings_ok": not mismatched,
        "render_settings_mismatched": mismatched,
        "antialiasing": name,
        "aa_op": op,
        "dlaa_in_effect": name == RD.ANTIALIASING,
        "fallback": None if name == RD.ANTIALIASING else name,
        "dlss_exec_mode": settings.get("/rtx/post/dlss/execMode"),
        "render_settings": {key: settings.get(key) for key in RD.RENDER_SETTINGS},
    }


# ------------------------------------------------------------------------------- cameras
def add_wrist_cameras(stage, arms_root):
    """Author one camera per arm under its link6 at the RoboDojo mount; returns side -> prim path."""
    from pxr import Gf, UsdGeom

    local = RD.wrist_camera_in_link6()
    paths = {}
    for side in SIDES:
        link = f"{arms_root}/{side}/{RD.WRIST_PARENT_LINK}"
        if not stage.GetPrimAtPath(link).IsValid():
            raise RuntimeError(f"no {link} prim for the {side} wrist camera")
        path = f"{link}/{WRIST_PRIM}"
        camera = UsdGeom.Camera.Define(stage, path)
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(local.T.tolist()))
        paths[side] = path
    return paths


def make_cameras(wrist_paths, depth=False):
    """isaacsim Camera objects (before world.reset): view -> Camera."""
    from isaacsim.sensors.camera import Camera

    cams = {"head": Camera(HEAD_PATH, name="cam_head", resolution=tuple(RD.HEAD_CAMERA["resolution"]))}
    for side, path in wrist_paths.items():
        cams[side] = Camera(path, name=VIEWS[side], resolution=tuple(RD.WRIST_CAMERA["resolution"]))
    return cams


def configure_cameras(cams, depth=False):
    """After world.reset: initialize and apply RoboDojo's intrinsics calls; returns reported intrinsics."""
    for view, cam in cams.items():
        spec = RD.HEAD_CAMERA if view == "head" else RD.WRIST_CAMERA
        cam.initialize()
        cam.set_focal_length(spec["focal_length_mm"])
        cam.set_horizontal_aperture(spec["horizontal_aperture_mm"], False)
        cam.set_vertical_aperture(spec["vertical_aperture_mm"], False)
        cam.set_clipping_range(*spec["clipping_range_m"])
        if depth:
            cam.add_distance_to_image_plane_to_frame()
    return intrinsics_report(cams)


# Measured on a rendered B1 head frame (2026-11-03): the mount plates' front edge (y -0.195, z 0.8704)
# appears at v 392, where square pixels put it (fy = fx = 288.1), not at v 418 (fy 336.5 from the
# vertical aperture, which get_intrinsics_matrix() reports). RTX keeps square pixels: the rendered
# head image spans 96.0 x 79.6 deg (robodojo_rig.intrinsics()). Results record the reported
# matrices (camera_intrinsics) and the rendered ones (camera_intrinsics_rendered); visibility.py
# projects with the rendered model.
RENDERED_PIXEL_MODEL = "square pixels (fy = fx); RTX fits the horizontal aperture"


def intrinsics_report(cams):
    out = {"renderer_pixel_model": RENDERED_PIXEL_MODEL}
    for view, cam in cams.items():
        k = np.asarray(cam.get_intrinsics_matrix(), dtype=float)
        out[VIEWS[view]] = {
            "resolution": list(cam.get_resolution()),
            "intrinsics_matrix": np.round(k, 4).tolist(),
            "fov_deg": [
                round(math.degrees(2 * math.atan(k[0, 2] / k[0, 0])), 3),
                round(math.degrees(2 * math.atan(k[1, 2] / k[1, 1])), 3),
            ],
        }
    return out


def rendered_intrinsics_report(views=VIEWS):
    """The square-pixel matrices RTX renders (fy = fx, principal point at the centre), per view."""
    out = {"renderer_pixel_model": RENDERED_PIXEL_MODEL, "note": INTRINSICS_NOTE}
    for view in views:
        spec = RD.HEAD_CAMERA if view == "head" else RD.WRIST_CAMERA
        out[VIEWS[view]] = {
            "resolution": list(spec["resolution"]),
            "intrinsics_matrix": np.round(RD.intrinsics(spec), 4).tolist(),
            "fov_deg": [round(v, 3) for v in RD.field_of_view_deg(spec)],
        }
    return out


def frame_rgb(cam):
    rgba = cam.get_rgba()
    if rgba is None or np.asarray(rgba).ndim != 3 or np.asarray(rgba).size == 0:
        return None
    return np.asarray(rgba)[:, :, :3].astype(np.uint8).copy()


# ----------------------------------------------------------------------------- rig check
def _world(prim):
    from pxr import Usd, UsdGeom

    return np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def link_pose(robot, name):
    """Measured 4x4 world pose of one articulation link (physics view)."""
    from scipy.spatial.transform import Rotation as R

    view = robot._articulation_view
    index = list(view.body_names).index(name)
    t = np.asarray(view._physics_view.get_link_transforms(), dtype=float)[0, index]
    pose = np.eye(4)
    pose[:3, :3] = R.from_quat(t[3:7]).as_matrix()  # x, y, z, qx, qy, qz, qw
    pose[:3, 3] = t[:3]
    return pose


def _chain(urdf):
    rows = []
    for joint in ET.parse(urdf).getroot().iter("joint"):
        if joint.get("type") != "revolute" or len(rows) == 6:
            continue
        origin = joint.find("origin")
        rows.append(
            (
                RD.transform(
                    np.fromstring(origin.get("xyz"), sep=" "),
                    RD.urdf_rpy(*np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")),
                ),
                np.fromstring(joint.find("axis").get("xyz"), sep=" "),
            )
        )
    return rows


def _axis_angle(axis, angle):
    a = axis / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def link6_in_base(q, urdf=X5A_URDF):
    """URDF forward kinematics: link6 in base_link for six joint angles."""
    pose = np.eye(4)
    for (origin, axis), angle in zip(_chain(urdf), q):
        pose = pose @ origin @ RD.transform(np.zeros(3), _axis_angle(axis, angle))
    return pose


def pose_error(a, b):
    """(position m, angle deg) between two 4x4 poses."""
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), math.degrees(math.acos(cosine))


def rig_check(stage, robots, wrist_paths=None, *, raise_on_error=True):
    """Measure the rig from the stage and physics; ``robodojo_rig.check_rig`` + wrist + link6 checks.

    Without cameras (``sim.no_render``) the head camera prim of the scene is still measured (it is
    authored in the initial scene); the wrist mounts are checked only when the cameras exist.
    """
    bases = {side: link_pose(robots[side], "base_link") for side in SIDES}
    head = _world(stage.GetPrimAtPath(HEAD_PATH))
    errors = []
    try:
        frame = RD.check_rig(bases["Left"], bases["Right"], head)
    except ValueError as exc:
        frame = None
        errors.append(f"check_rig: {exc}")
    expected_head = None if frame is None else RD.head_camera_world(frame["midpoint"], frame["yaw_deg"])
    out = {
        "rig_frame": frame,
        "bases_world_m": {side: np.round(pose[:3, 3], 6).tolist() for side, pose in bases.items()},
        "head_camera_world_m": np.round(head[:3, 3], 6).tolist(),
        "head_camera_error": None if expected_head is None else dict(zip(("m", "deg"), pose_error(head, expected_head))),
        "wrist": {},
        "link6_vs_x5a": {},
    }
    wrist_local = RD.wrist_camera_in_link6()
    for side in SIDES:
        robot = robots[side]
        link6 = link_pose(robot, RD.WRIST_PARENT_LINK)
        names = list(robot.dof_names)
        positions = np.asarray(robot.get_joint_positions(), dtype=float)
        q = [positions[names.index(f"joint{i}")] for i in range(1, 7)]
        measured = np.linalg.inv(bases[side]) @ link6
        dp, da = pose_error(measured, link6_in_base(q, X5A_URDF))
        hp, ha = pose_error(measured, link6_in_base(q, HOUSEHOLD_URDF))
        out["link6_vs_x5a"][side] = {
            "position_error_m": round(dp, 6),
            "angle_error_deg": round(da, 4),
            "matches": dp <= LINK6_TOLERANCE[0] and da <= LINK6_TOLERANCE[1],
            "household_urdf_error_m": round(hp, 6),  # sanity: the simulated arm follows its own URDF
        }
        if wrist_paths:
            prim = stage.GetPrimAtPath(wrist_paths[side])
            parent = str(prim.GetParent().GetPath())
            local = np.linalg.inv(_world(prim.GetParent())) @ _world(prim)
            camera = link6 @ local  # the camera relative to the measured link6
            wp, wa = pose_error(camera, link6 @ wrist_local)
            ok = wp <= 1e-3 and wa <= 0.1 and parent.endswith("/" + RD.WRIST_PARENT_LINK)
            out["wrist"][side] = {
                "prim": wrist_paths[side],
                "parent": parent,
                "camera_world_m": np.round(camera[:3, 3], 6).tolist(),
                "position_error_m": round(wp, 7),
                "angle_error_deg": round(wa, 5),
                "ok": ok,
            }
            if not ok:
                errors.append(f"{side} wrist camera {wp * 1000:.2f} mm / {wa:.3f} deg from the spec mount")
        if hp > 1e-3:
            errors.append(f"{side} link6 {hp * 1000:.2f} mm from its own URDF")
    out["errors"] = errors
    out["ok"] = not errors
    if errors and raise_on_error:
        raise RuntimeError("RoboDojo rig check failed: " + "; ".join(errors))
    return out


def json_ready(value):
    """Round-trip numpy scalars and arrays to JSON types."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
