"""RoboDojo-aligned dual ARX X5 rig: the camera, rate and render spec shared by every domain.

Every roboclock domain (industry, hospital, sports, lab, household) builds its policy
cameras, physics rate and renderer from these values, so a policy trained on RoboDojo sees
the same embodiment geometry everywhere. The module is simulator-free (numpy only).

Sources, read 2026-11-03:

- RoboDojo f57e163 (github.com/RoboDojo-Benchmark/RoboDojo)
  - ``env_cfg/robot/dual_x5.yml``: X5 bases at (-0.30, -0.45, 0.765) and (0.30, -0.45, 0.765),
    both yawed +90 deg (facing world +y).
  - ``env_cfg/camera/camera_config.yml``: ``cam_head`` at (0, -0.41, 1.308), euler XYZ (30, 0, 0) deg.
  - ``env_cfg/camera/template.py``: GEMINI_345LG (head) and D435 (wrist) intrinsics.
  - ``env_cfg/sim/sim_config.yml``: dt 0.004 s, render_interval 10.
  - ``env_cfg/arx_x5.yml``: observation collect_freq 25 Hz (one policy action per 10 physics steps).
  - ``env/environment/base_env.py``: Isaac Lab rendering_mode "quality", antialiasing "DLAA",
    translucency, reflections and global illumination on, dlss_mode 2.
- RoboDojo assets, HF dataset RoboDojo-Benchmark/RoboDojo, ``Assets/Robots/x5``
  - ``robot_config.yml``: ``cam_wrist`` (type d435) on link ``camera``, pos (0, 0, 0.001),
    euler XYZ (0, -80, -90) deg.
  - ``X5A.urdf``: fixed joints link6 -> camera_base -> camera. These match
    ``assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf`` in this repo.
- Isaac Lab afca7b0 (RoboDojo's submodule)
  - ``apps/rendering_modes/quality.kit``: the preset carb settings in ``RENDER_SETTINGS``.
  - ``SimulationContext._apply_render_settings_from_cfg`` applies antialiasing with
    ``omni.replicator.core.settings.set_render_rtx_realtime(antialiasing=...)``. RoboDojo
    installs Isaac Sim 5.1.0, where RTX Real-Time 2.0 is opt-in, so its render mode is
    RaytracedLighting. In Isaac Sim 6.0.1 the same call switches to RealTimePathTracing
    (measured by sports, 2026-11-04): call it for the antialiasing, then set
    ``/rtx/rendermode`` to ``RENDER_MODE`` and read it back.

Conventions:

- Rig frame: origin at the midpoint of the two bases on the mount plane. +x is the direction
  both arms face, +y is the rig's left, +z is up.
- Camera orientations use USD camera axes: the camera looks along its -Z, and image up is +Y.
- Isaac euler angles follow RoboDojo ``utils/rotations.euler_angles_to_quat`` (intrinsic XYZ).
  URDF rpy is fixed-axis roll, pitch, yaw.

Where RoboDojo pins the rig in one world pose for every task, roboclock keeps per-task rig
placement: each task may place the rig (origin and yaw) in its own scene. Everything
relative to the rig is fixed here.
"""

import math

import numpy as np

# Rates (Hz). Physics runs at exactly 250 Hz in every task, not a multiple of it.
PHYSICS_HZ = 250
POLICY_HZ = 25  # observation and policy action rate (RoboDojo collect_freq)
RENDER_HZ = 25  # render_interval 10 at 250 Hz; roboclock recordings use this rate
# RoboDojo renders every observation at 25 Hz. roboclock renders a policy observation only when
# the policy reads it (roboclock.envs.observation): once per model query, at that query's
# observation instant, so a chunked policy costs one render per chunk.
POLICY_RENDER = "on_read"
VIDEO_FPS = 25

# Rig geometry (m).
ARM_SEPARATION_M = 0.60
LEFT_BASE_IN_RIG_M = (0.0, ARM_SEPARATION_M / 2, 0.0)
RIGHT_BASE_IN_RIG_M = (0.0, -ARM_SEPARATION_M / 2, 0.0)
HEAD_IN_RIG_M = (0.04, 0.0, 0.543)  # 1.308 - 0.765 = 0.543 above the mount plane
HEAD_PITCH_DOWN_DEG = 60.0  # euler X 30 deg on a camera that starts looking straight down
HEAD_ROLL_DEG = 0.0

# Camera names: RoboDojo / XPolicyLab view name -> roboclock observation key.
OBSERVATION_KEYS = {
    "cam_head": "rgb",
    "cam_left_wrist": "left_wrist_rgb",
    "cam_right_wrist": "right_wrist_rgb",
}
DEPTH_KEYS = {
    "cam_head": "depth",
    "cam_left_wrist": "left_wrist_depth",
    "cam_right_wrist": "right_wrist_depth",
}

# Intrinsics, in the units RoboDojo passes to isaacsim.sensors.camera.Camera
# (focal length and apertures in mm; only their ratios set the field of view).
# RoboDojo calls set_horizontal_aperture(h, False) and set_vertical_aperture(v, False).
HEAD_CAMERA = {
    "model": "Gemini_345Lg",
    "resolution": (640, 480),
    "focal_length_mm": 10.0,
    "horizontal_aperture_mm": 22.212,
    "vertical_aperture_mm": 14.266,
    "clipping_range_m": (0.005, 10.0),
}
WRIST_CAMERA = {
    "model": "D435",
    "resolution": (640, 480),
    "focal_length_mm": 13.0,
    "horizontal_aperture_mm": 20.955,
    "vertical_aperture_mm": 15.71625,
    "clipping_range_m": (0.0001, 10.0),
}

# Wrist mount chain from X5A.urdf, then RoboDojo's camera pose on link "camera".
WRIST_PARENT_LINK = "link6"
_WRIST_CHAIN = (
    ((0.057, 0.0, 0.0), "rpy", (0.0, 0.0, -3.1416)),  # hand_to_camera_mount: link6 -> camera_base
    ((-0.0275, 0.0, 0.05), "rpy", (0.0, 0.3491, math.pi)),  # camera_joint: camera_base -> camera
    ((0.0, 0.0, 0.001), "euler_xyz_deg", (0.0, -80.0, -90.0)),  # robot_config.yml cam_wrist
)

# Renderer: RTX Real-Time with DLAA and Isaac Lab's "quality" preset.
RENDER_MODE = "RaytracedLighting"
ANTIALIASING = "DLAA"
RENDER_SETTINGS = {
    "/rtx/translucency/enabled": True,
    "/rtx/reflections/enabled": True,
    "/rtx/reflections/denoiser/enabled": True,
    "/rtx/directLighting/sampledLighting/enabled": True,
    "/rtx/sceneDb/ambientLightIntensity": 1.0,
    "/rtx/shadows/enabled": True,
    "/rtx/indirectDiffuse/enabled": True,
    "/rtx/indirectDiffuse/denoiser/enabled": True,
    "/rtx/ambientOcclusion/enabled": True,
    "/rtx/ambientOcclusion/denoiserMode": 0,
    "/rtx/raytracing/subpixel/mode": 1,
    "/rtx/raytracing/cached/enabled": True,
    "/rtx-transient/dlssg/enabled": False,
    "/rtx-transient/dldenoiser/enabled": True,
    "/rtx/post/dlss/execMode": 2,
    "/rtx/pathtracing/maxSamplesPerLaunch": 1000000,
    "/rtx/viewTile/limit": 1000000,
}


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def urdf_rpy(roll, pitch, yaw):
    """URDF fixed-axis roll, pitch, yaw (rad) as a rotation matrix."""
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def euler_xyz(x, y, z, degrees=True):
    """Intrinsic XYZ euler angles, as RoboDojo's euler_angles_to_quat (extrinsic=False)."""
    if degrees:
        x, y, z = (math.radians(v) for v in (x, y, z))
    return _rx(x) @ _ry(y) @ _rz(z)


def transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = position
    return matrix


def quat_wxyz(rotation):
    """Unit quaternion (w, x, y, z) of a rotation matrix, w >= 0."""
    m = np.asarray(rotation, dtype=float)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def look_rotation(look, up):
    """USD camera rotation (columns X, Y, Z) that looks along ``look`` with image up near ``up``."""
    z = -np.asarray(look, dtype=float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def head_camera_in_rig():
    """4x4 pose of the head camera (USD axes) in the rig frame."""
    pitch = math.radians(HEAD_PITCH_DOWN_DEG)
    look = np.array([math.cos(pitch), 0.0, -math.sin(pitch)])
    up = np.array([math.sin(pitch), 0.0, math.cos(pitch)])
    return transform(HEAD_IN_RIG_M, look_rotation(look, up))


def wrist_camera_in_link6():
    """4x4 pose of a wrist camera (USD axes) in the X5 ``link6`` frame."""
    pose = np.eye(4)
    for position, kind, angles in _WRIST_CHAIN:
        rotation = urdf_rpy(*angles) if kind == "rpy" else euler_xyz(*angles)
        pose = pose @ transform(position, rotation)
    return pose


def rig_to_world(origin, yaw_deg):
    """4x4 rig frame in the world: ``origin`` is the base midpoint on the mount plane."""
    return transform(origin, _rz(math.radians(yaw_deg)))


def base_positions(origin, yaw_deg):
    """World positions of the (left, right) base mounts."""
    rig = rig_to_world(origin, yaw_deg)
    return tuple((rig @ np.r_[p, 1.0])[:3] for p in (LEFT_BASE_IN_RIG_M, RIGHT_BASE_IN_RIG_M))


def head_camera_world(origin, yaw_deg):
    """4x4 world pose of the head camera (USD axes) for a rig at ``origin``, ``yaw_deg``."""
    return rig_to_world(origin, yaw_deg) @ head_camera_in_rig()


# RTX renders square pixels: the vertical aperture does not change the image. Measured in
# Isaac Sim 6.0.1 (industry Task 45, 2026-11-04): a lamp imaged at (398.0, 69.6) px against
# (398.4, 70.4) px predicted with square pixels and (398.4, 42.0) px with both apertures.
# Camera.get_intrinsics_matrix() still reports fy from the vertical aperture (336.46 px for
# the head camera), which does not describe the rendered image. For the head camera the
# rendered field of view is 96.0 x 79.6 deg; the D435 apertures are 4:3, so the two
# models agree for the wrist cameras.
RENDERER_SQUARE_PIXELS = True


def field_of_view_deg(camera, square_pixels=RENDERER_SQUARE_PIXELS):
    """(horizontal, vertical) field of view in degrees of the rendered image.

    With ``square_pixels=False`` the vertical value comes from the vertical aperture instead.
    """
    width, height = camera["resolution"]
    f, horizontal = camera["focal_length_mm"], camera["horizontal_aperture_mm"]
    vertical = horizontal * height / width if square_pixels else camera["vertical_aperture_mm"]
    return tuple(math.degrees(2 * math.atan(a / (2 * f))) for a in (horizontal, vertical))


def intrinsics(camera, square_pixels=RENDERER_SQUARE_PIXELS):
    """3x3 pinhole matrix of the rendered image (fy = fx under square pixels)."""
    width, height = camera["resolution"]
    f = camera["focal_length_mm"]
    fx = f / camera["horizontal_aperture_mm"] * width
    fy = fx if square_pixels else f / camera["vertical_aperture_mm"] * height
    return np.array([[fx, 0.0, width / 2], [0.0, fy, height / 2], [0.0, 0.0, 1.0]])


def check_rig(left_base, right_base, head_pose, *, tolerance_m=1e-3, tolerance_deg=0.1):
    """Raise if measured world poses deviate from the spec.

    ``left_base`` and ``right_base`` are 4x4 world poses of the two base mounts, with +x
    the arm's forward direction. ``head_pose`` is the 4x4 world pose of the head camera
    in USD axes.
    """
    left, right = np.asarray(left_base), np.asarray(right_base)
    forward = left[:3, 0]
    if np.degrees(np.arccos(np.clip(forward @ right[:3, 0], -1, 1))) > tolerance_deg:
        raise ValueError("bases face different directions")
    if abs(forward[2]) > math.sin(math.radians(tolerance_deg)):
        raise ValueError("bases are not level")
    delta = left[:3, 3] - right[:3, 3]
    if abs(delta[2]) > tolerance_m:
        raise ValueError(f"bases at different heights ({delta[2]:+.4f} m)")
    if abs(delta @ forward) > tolerance_m:
        raise ValueError(f"bases offset front to back ({delta @ forward:+.4f} m)")
    if abs(np.linalg.norm(delta) - ARM_SEPARATION_M) > tolerance_m:
        raise ValueError(f"bases {np.linalg.norm(delta):.4f} m apart, expected {ARM_SEPARATION_M}")
    leftward = np.cross([0.0, 0.0, 1.0], forward)
    if delta @ leftward <= 0:
        raise ValueError("left base is not on the rig's left")
    midpoint = (left[:3, 3] + right[:3, 3]) / 2
    yaw = math.degrees(math.atan2(forward[1], forward[0]))
    expected = head_camera_world(midpoint, yaw)
    head = np.asarray(head_pose)
    error = np.linalg.norm(head[:3, 3] - expected[:3, 3])
    if error > tolerance_m:
        raise ValueError(f"head camera {error:.4f} m from the spec pose")
    cosine = np.clip((np.trace(expected[:3, :3].T @ head[:3, :3]) - 1) / 2, -1, 1)
    if math.degrees(math.acos(cosine)) > tolerance_deg:
        raise ValueError(f"head camera turned {math.degrees(math.acos(cosine)):.3f} deg from the spec")
    return {"midpoint": midpoint.tolist(), "yaw_deg": yaw}


def describe():
    """JSON-ready summary for result metadata."""
    wrist = wrist_camera_in_link6()
    return {
        "source": "RoboDojo f57e163 dual_x5 / camera_config / X5A.urdf",
        "arm_separation_m": ARM_SEPARATION_M,
        "head_in_rig_m": list(HEAD_IN_RIG_M),
        "head_pitch_down_deg": HEAD_PITCH_DOWN_DEG,
        "head_camera": {
            **HEAD_CAMERA,
            "fov_deg": field_of_view_deg(HEAD_CAMERA),
            "rendered_intrinsics": intrinsics(HEAD_CAMERA).round(4).tolist(),
        },
        "wrist_camera": {
            **WRIST_CAMERA,
            "fov_deg": field_of_view_deg(WRIST_CAMERA),
            "rendered_intrinsics": intrinsics(WRIST_CAMERA).round(4).tolist(),
            "parent_link": WRIST_PARENT_LINK,
            "position_in_link6_m": wrist[:3, 3].round(5).tolist(),
            "orientation_in_link6_wxyz": quat_wxyz(wrist[:3, :3]).round(5).tolist(),
        },
        "physics_hz": PHYSICS_HZ,
        "policy_hz": POLICY_HZ,
        "video_fps": VIDEO_FPS,
        "render_mode": RENDER_MODE,
        "antialiasing": ANTIALIASING,
    }
