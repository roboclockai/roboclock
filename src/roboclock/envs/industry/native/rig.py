"""Industry adapter for the shared RoboDojo rig spec (``roboclock.envs.robodojo_rig``).

Every value - arm separation, head-camera pose and intrinsics, wrist-camera mount and
intrinsics, physics rate, renderer - comes from ``robodojo_rig``; this module keeps no
copies. It only maps the spec onto the industry scenes:

* A task places the rig with ``mounts(center)``: ``center`` is the base midpoint on the
  mount plane (the spec's rig origin) and the rig yaw is DualARX's default mount yaw of
  90 deg, so the arms face world +Y. A mount is the world position of an ARX
  ``base_link`` origin (the DualARX root translate).
* ``attach_cameras`` builds the policy cameras - the head camera (``scene.camera``) and
  one wrist camera rigid on each arm's ``link6`` - with RoboDojo's camera API calls, plus
  a recording-only third-person overview camera. ``configure(no_render=True)`` creates no
  cameras at all.
* ``measure`` checks the built rig against the spec at reset (``robodojo_rig.check_rig``
  on the measured bases and head camera; wrist cameras against ``wrist_camera_in_link6``
  applied to the measured ``link6``).
"""

import math

import numpy as np

from roboclock.envs import robodojo_rig as spec

# DualARX's default mount orientation turns the arm's +X to world +Y.
DEFAULT_YAW_DEG = 90.0
OVERVIEW_RESOLUTION = (1280, 960)
_SETTINGS = dict(no_render=False, overview=False)


def configure(*, no_render=None, overview=None):
    """Process-wide camera switches, set by the runner/environment before a scene is built."""
    if no_render is not None:
        _SETTINGS["no_render"] = bool(no_render)
    if overview is not None:
        _SETTINGS["overview"] = bool(overview)
    return dict(_SETTINGS)


# ----------------------------------------------------------------------------- geometry
def mounts(center, yaw_deg=DEFAULT_YAW_DEG):
    """Left/right base mounts for a rig origin ``center`` (base midpoint on the mount plane)."""
    left, right = spec.base_positions(np.asarray(center, float), yaw_deg)
    return dict(left=np.round(left, 6).tolist(), right=np.round(right, 6).tolist())


def mounts_keeping(side, position, yaw_deg=DEFAULT_YAW_DEG):
    """Rig with one arm's base kept at ``position`` and the other placed beside it."""
    offset = np.asarray(spec.LEFT_BASE_IN_RIG_M if side == "left" else spec.RIGHT_BASE_IN_RIG_M)
    rotation = spec.rig_to_world((0, 0, 0), yaw_deg)[:3, :3]
    return mounts(np.asarray(position, float) - rotation @ offset, yaw_deg)


def center(rig_mounts):
    return (np.asarray(rig_mounts["left"], float) + np.asarray(rig_mounts["right"], float)) / 2


def check(rig_mounts, yaw_deg=DEFAULT_YAW_DEG, tolerance=1e-6):
    """Assert that the mounts are the spec's two bases for some rig origin at ``yaw_deg``."""
    expected = mounts(center(rig_mounts), yaw_deg)
    for side in ("left", "right"):
        assert np.allclose(rig_mounts[side], expected[side], atol=tolerance), (side, rig_mounts, expected)
    return rig_mounts


def head_pose(rig_mounts, yaw_deg=DEFAULT_YAW_DEG):
    """4x4 world pose of the head camera (USD camera axes) for these mounts."""
    return spec.head_camera_world(center(check(rig_mounts, yaw_deg)), yaw_deg)


def project_head(rig_mounts, points, *, yaw_deg=DEFAULT_YAW_DEG):
    """Pixel coordinates (u, v) of world points in the rendered head image, or None behind
    the camera. Uses ``robodojo_rig.intrinsics()`` (RTX square pixels: fx = fy = 288.13)."""
    pose = head_pose(rig_mounts, yaw_deg)
    K = spec.intrinsics(spec.HEAD_CAMERA)
    out = []
    for p in np.atleast_2d(np.asarray(points, float)):
        local = pose[:3, :3].T @ (p - pose[:3, 3])  # USD camera axes: looks along -Z, +Y up
        if local[2] >= 0:
            out.append(None)
            continue
        u = K[0, 2] + K[0, 0] * local[0] / -local[2]
        v = K[1, 2] - K[1, 1] * local[1] / -local[2]
        out.append((float(u), float(v)))
    return out


def inside_head_image(rig_mounts, points, margin_px=10, **kwargs):
    """True if every point projects at least ``margin_px`` inside the 640x480 head image."""
    width, height = spec.HEAD_CAMERA["resolution"]
    for uv in project_head(rig_mounts, points, **kwargs):
        if uv is None or not (margin_px <= uv[0] <= width - margin_px and margin_px <= uv[1] <= height - margin_px):
            return False
    return True


# ----------------------------------------------------------------------------- cameras
def _set_pose(prim, matrix):
    """Local translate + orient of a camera prim (reusing ops the Camera class authored)."""
    from pxr import Gf, UsdGeom

    q = [float(v) for v in spec.quat_wxyz(matrix[:3, :3])]
    t = [float(v) for v in matrix[:3, 3]]
    xform = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xform.GetOrderedXformOps()}
    translate = ops.get("xformOp:translate") or xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    orient = ops.get("xformOp:orient") or xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    double = translate.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    translate.Set(Gf.Vec3d(*t) if double else Gf.Vec3f(*t))
    double = orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble
    orient.Set(Gf.Quatd(q[0], q[1], q[2], q[3]) if double else Gf.Quatf(q[0], q[1], q[2], q[3]))
    xform.SetXformOpOrder([translate, orient])


def apply_intrinsics(camera, template):
    """RoboDojo's camera API calls: focal length and both apertures without square-pixel sync."""
    camera.set_focal_length(template["focal_length_mm"])
    camera.set_horizontal_aperture(template["horizontal_aperture_mm"], False)
    camera.set_vertical_aperture(template["vertical_aperture_mm"], False)
    camera.set_clipping_range(*template["clipping_range_m"])


def attach_cameras(scene, rig_mounts, task, *, yaw_deg=DEFAULT_YAW_DEG):
    """Build the policy cameras (and the recording overview) for a constructed scene.

    Sets ``scene.camera`` (head, ``rgb``), ``scene.wrist_cameras`` (``left``/``right``),
    ``scene.overview_camera`` and ``scene.rig`` (``describe`` metadata). With
    ``configure(no_render=True)`` no camera is created and the three attributes are None/{}.
    """
    from isaacsim.sensors.camera import Camera

    from .overview import OVERVIEW

    scene.rig_mounts = check(rig_mounts, yaw_deg)
    scene.rig_yaw_deg = yaw_deg
    scene.camera, scene.wrist_cameras, scene.overview_camera = None, {}, None
    if not _SETTINGS["no_render"]:
        stage = scene.world.stage
        head = scene.world.scene.add(Camera("/World/RigHeadCamera", name="rig_head", resolution=tuple(spec.HEAD_CAMERA["resolution"])))
        _set_pose(head.prim, head_pose(rig_mounts, yaw_deg))
        head.initialize()
        apply_intrinsics(head, spec.HEAD_CAMERA)
        scene.camera = head
        link6 = scene.dual.manifest["relative_body_paths"][spec.WRIST_PARENT_LINK]
        for side in ("left", "right"):
            path = f"/World/ARX/{side}/{link6}/RigWristCamera"
            camera = scene.world.scene.add(Camera(path, name=f"rig_{side}_wrist", resolution=tuple(spec.WRIST_CAMERA["resolution"])))
            _set_pose(camera.prim, spec.wrist_camera_in_link6())
            camera.initialize()
            apply_intrinsics(camera, spec.WRIST_CAMERA)
            scene.wrist_cameras[side] = camera
        if _SETTINGS["overview"]:
            view = OVERVIEW[task]
            camera = scene.world.scene.add(Camera("/World/RigOverviewCamera", name="rig_overview", resolution=OVERVIEW_RESOLUTION))
            from .plating_physics import camera_pose

            camera_pose(camera, view["eye"], view["target"])
            camera.initialize()
            camera.set_focal_length(view["focal_m"])
            camera.set_horizontal_aperture(0.036, False)
            camera.set_vertical_aperture(0.036 * OVERVIEW_RESOLUTION[1] / OVERVIEW_RESOLUTION[0], False)
            camera.set_clipping_range(0.01, 30.0)
            scene.overview_camera = camera
        del stage
    scene.rig = describe(scene)
    return scene.rig


def reapply_intrinsics(scene):
    """Camera.initialize() after destroy() re-syncs square pixels; restore the spec apertures."""
    if scene.camera is not None:
        apply_intrinsics(scene.camera, spec.HEAD_CAMERA)
    for camera in scene.wrist_cameras.values():
        apply_intrinsics(camera, spec.WRIST_CAMERA)


def policy_cameras(scene):
    """(observation key, camera) for the head and both wrists, in spec order."""
    if scene.camera is None:
        return []
    return [
        (spec.OBSERVATION_KEYS["cam_head"], scene.camera),
        (spec.OBSERVATION_KEYS["cam_left_wrist"], scene.wrist_cameras["left"]),
        (spec.OBSERVATION_KEYS["cam_right_wrist"], scene.wrist_cameras["right"]),
    ]


# ----------------------------------------------------------------------------- checks
def _pose(position, rotation):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rotation, position
    return m


def _angle_deg(a, b):
    return math.degrees(math.acos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def measure(scene):
    """Measured rig against the spec: bases and head (check_rig), wrist cameras, link6."""
    from pxr import UsdGeom

    dual = scene.dual
    bases = {}
    for side in ("left", "right"):
        p, R = dual.body_world(side, "base_link")
        # DualARX turns the arm's +X to the mount yaw; base_link +X is the arm's forward.
        bases[side] = _pose(p, R)
    result = dict(bases_m={s: bases[s][:3, 3].round(6).tolist() for s in bases})
    if scene.camera is not None:
        head = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(scene.camera.prim)).T
        result["head_measured"] = True
    else:
        head = head_pose(scene.rig_mounts, scene.rig_yaw_deg)
        result["head_measured"] = False  # no camera in no-render runs; spec pose used
    result["check_rig"] = spec.check_rig(bases["left"], bases["right"], head)
    result["head_position_m"] = head[:3, 3].round(6).tolist()
    wrists = {}
    cache = UsdGeom.XformCache()
    for side, camera in scene.wrist_cameras.items():
        p, R = dual.body_world(side, spec.WRIST_PARENT_LINK)
        expected = _pose(p, R) @ spec.wrist_camera_in_link6()
        link6_usd = np.array(cache.GetLocalToWorldTransform(camera.prim.GetParent())).T
        actual = np.array(cache.GetLocalToWorldTransform(camera.prim)).T
        local = np.linalg.inv(link6_usd) @ actual
        mount = spec.wrist_camera_in_link6()
        wrists[side] = dict(
            mount_error_m=float(np.linalg.norm(local[:3, 3] - mount[:3, 3])),
            mount_error_deg=_angle_deg(local[:3, :3], mount[:3, :3]),
            world_error_m=float(np.linalg.norm(actual[:3, 3] - expected[:3, 3])),
            world_error_deg=_angle_deg(actual[:3, :3], expected[:3, :3]),
        )
        if wrists[side]["mount_error_m"] > 1e-3 or wrists[side]["mount_error_deg"] > 0.1:
            raise ValueError(f"{side} wrist camera off its link6 mount: {wrists[side]}")
    result["wrist_cameras"] = wrists
    return result


def describe(scene):
    """``robodojo_rig.describe()`` plus this task's placement and camera readback."""
    data = dict(spec=spec.describe(), mounts_m=scene.rig_mounts, rig_center_m=center(scene.rig_mounts).round(6).tolist(),
                rig_yaw_deg=scene.rig_yaw_deg, cameras_created=scene.camera is not None)
    if scene.camera is not None:
        data["intrinsics_reported"] = {
            key: np.asarray(camera.get_intrinsics_matrix()).round(4).tolist() for key, camera in policy_cameras(scene)
        }
        data["intrinsics_rendered"] = rendered_intrinsics()
    return data


def rendered_intrinsics():
    """The spec's intrinsics of the rendered images (``robodojo_rig.intrinsics()``, square
    pixels). ``intrinsics_reported`` keeps what ``get_intrinsics_matrix()`` returns; for the
    head camera its fy (336.46) comes from the vertical aperture and does not match the image."""
    return {
        key: spec.intrinsics(template).round(4).tolist()
        for key, template in (("rgb", spec.HEAD_CAMERA), ("left_wrist_rgb", spec.WRIST_CAMERA),
                              ("right_wrist_rgb", spec.WRIST_CAMERA))
    }


# ----------------------------------------------------------------------------- renderer
def apply_renderer():
    """RaytracedLighting + DLAA + Isaac Lab's quality preset (spec). Returns the readback."""
    import carb
    import omni.replicator.core as rep

    settings = carb.settings.get_settings()
    # Isaac Lab's call. In Kit 110 (Isaac Sim 6) it also switches /rtx/rendermode to
    # RealTimePathTracing, so the spec's RaytracedLighting is set after it.
    rep.settings.set_render_rtx_realtime(antialiasing=spec.ANTIALIASING)
    settings.set("/rtx/rendermode", spec.RENDER_MODE)
    for key, value in spec.RENDER_SETTINGS.items():
        settings.set(key, value)
    return renderer_readback()


AA_NAMES = {0: "None", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}


def renderer_readback():
    import carb

    settings = carb.settings.get_settings()
    aa = settings.get("/rtx/post/aa/op")
    return {
        "rtx_rendermode": settings.get("/rtx/rendermode"),
        "rtx_post_aa_op": aa,
        "antialiasing": AA_NAMES.get(aa, aa),
        "dlss_exec_mode": settings.get("/rtx/post/dlss/execMode"),
        "spec_render_mode": spec.RENDER_MODE,
        "spec_antialiasing": spec.ANTIALIASING,
        "quality_preset_applied": {k: settings.get(k) for k in spec.RENDER_SETTINGS},
    }
