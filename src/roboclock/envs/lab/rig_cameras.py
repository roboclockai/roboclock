"""RoboDojo policy cameras, renderer and measured rig check for the lab's dual-X5 rig.

Every value comes from roboclock.envs.robodojo_rig (docs/robodojo-rig-spec.md):

- head camera (`rgb`, RoboDojo `cam_head`): rigid at HEAD_IN_RIG_M, pitched 60 deg down;
- wrist cameras (`left_wrist_rgb` / `right_wrist_rgb`): children of each arm's `link6` at
  wrist_camera_in_link6(), so they move with the measured link without per-tick updates.
  `left` is the arm on the rig's left (+y), which is lab side RIG_LEFT_ARM (/World/ARXRight);
- renderer: RENDER_MODE + DLAA + RENDER_SETTINGS, read back after the first frame.

With `sim.no_render: true` no camera is created, nothing is rendered and observations
carry zero images of the same shapes. Imported only after SimulationApp starts, except
for the simulator-free helpers (``zero_images``, ``link6_in_base``).
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboclock.envs import robodojo_rig
from roboclock.envs.lab.robot import RIG_LEFT_ARM, RIG_RIGHT_ARM, ROOTS, rig_pose

HEAD_PATH = "/World/RigHeadCamera"
WRIST_PRIM = "RigWristCamera"
# (observation key, depth key, lab arm index) per RoboDojo camera name.
VIEWS = {
    "cam_head": ("rgb", "depth", None),
    "cam_left_wrist": ("left_wrist_rgb", "left_wrist_depth", RIG_LEFT_ARM),
    "cam_right_wrist": ("right_wrist_rgb", "right_wrist_depth", RIG_RIGHT_ARM),
}
INTRINSICS_NOTE = (
    "camera_intrinsics is what Isaac's get_intrinsics_matrix() reports (fy from the vertical "
    "aperture). RTX ignores the vertical aperture and renders square pixels, so the image "
    "follows camera_intrinsics_rendered (robodojo_rig.intrinsics: fy = fx, principal point at "
    "the centre). Use the rendered matrix for projection and depth unprojection."
)
AA_NAMES = {0: "Off", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}
X5A_URDF = (
    Path(__file__).resolve().parents[4]
    / "assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf"
)


def camera_spec(name):
    return robodojo_rig.HEAD_CAMERA if name == "cam_head" else robodojo_rig.WRIST_CAMERA


def zero_images(depth=False):
    """Observation images of the spec shapes (sim.no_render)."""
    images = {}
    for name, (rgb_key, depth_key, _) in VIEWS.items():
        width, height = camera_spec(name)["resolution"]
        images[rgb_key] = np.zeros((height, width, 3), dtype=np.uint8)
        if depth:
            images[depth_key] = np.zeros((height, width), dtype=np.float32)
    return images


def depth_image(camera, name):
    """Distance-to-image-plane image of a camera with the depth annotator attached.

    Isaac Sim 6 leaves ``get_current_frame()`` without the annotator's data, so read it like
    ``get_rgba()`` does, through ``get_depth()``.
    """
    distances = camera.get_depth()
    if distances is None or np.asarray(distances).ndim != 2:
        raise RuntimeError(f"{name} has no depth frame. Was depth enabled at construction?")
    return np.asarray(distances, dtype=np.float32).copy()


def observation_spaces(depth=False):
    import gymnasium as gym

    spaces = {}
    for name, (rgb_key, depth_key, _) in VIEWS.items():
        width, height = camera_spec(name)["resolution"]
        spaces[rgb_key] = gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)
        if depth:
            spaces[depth_key] = gym.spaces.Box(0, np.inf, (height, width), dtype=np.float32)
    return spaces


def _urdf_chain(path, tip="link6"):
    """Fixed joint origins and axes from base_link to ``tip`` of a URDF."""
    joints = {}
    for joint in ET.parse(path).getroot().iter("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints[joint.find("child").get("link")] = (
            joint.find("parent").get("link"),
            np.array([float(v) for v in origin.get("xyz").split()]),
            np.array([float(v) for v in origin.get("rpy").split()]),
            None if axis is None else np.array([float(v) for v in axis.get("xyz").split()]),
            joint.get("type"),
        )
    chain, link = [], tip
    while link in joints:
        chain.append(joints[link])
        link = joints[link][0]
    return chain[::-1]


def link6_in_base(joints, urdf=X5A_URDF):
    """4x4 pose of X5A.urdf ``link6`` in ``base_link`` for six arm joint angles (rad)."""
    pose = np.eye(4)
    angles = iter(joints)
    for _, xyz, rpy, axis, kind in _urdf_chain(urdf):
        pose = pose @ robodojo_rig.transform(xyz, robodojo_rig.urdf_rpy(*rpy))
        if kind == "revolute":
            angle = next(angles)
            k = axis / np.linalg.norm(axis)
            skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            rotation = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew
            pose = pose @ robodojo_rig.transform(np.zeros(3), rotation)
    return pose


def _pose_error(measured, expected):
    distance = float(np.linalg.norm(measured[:3, 3] - expected[:3, 3]))
    cosine = np.clip((np.trace(expected[:3, :3].T @ measured[:3, :3]) - 1) / 2, -1, 1)
    return distance, math.degrees(math.acos(cosine))


def _quat_matrix(quat):
    from roboclock.physics.geometry import rotation_matrix

    return rotation_matrix(np.asarray(quat, dtype=float))


def _world_matrix(prim):
    """4x4 world pose of a USD prim (column-vector convention)."""
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(matrix, dtype=float).T


def _set_local(prim, pose):
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(pose, dtype=float).T.tolist()))


class RigCameras:
    """Head + wrist policy cameras, renderer settings and the per-reset rig check."""

    def __init__(self, backend):
        self.backend = backend
        self.no_render = bool(backend.config["sim"].get("no_render", False))
        self.cameras = {}
        self.frames = None
        self.frame_time = None
        self.renderer = None
        self.intrinsics = {}  # as get_intrinsics_matrix() reports them
        self.rendered_intrinsics = {}  # square pixels, what RTX renders
        self.rig_check = None
        if self.no_render:
            return
        from isaacsim.sensors.camera import Camera
        from pxr import UsdGeom

        stage = backend.stage
        origin, yaw = rig_pose(backend.config, backend.task.task_id)
        head = robodojo_rig.head_camera_world(origin, math.degrees(yaw))
        UsdGeom.Camera.Define(stage, HEAD_PATH)
        _set_local(stage.GetPrimAtPath(HEAD_PATH), head)
        paths = {"cam_head": HEAD_PATH}
        for name, (_, _, side) in VIEWS.items():
            if side is None:
                continue
            path = f"{ROOTS[side]}/link6/{WRIST_PRIM}"
            UsdGeom.Camera.Define(stage, path)
            _set_local(stage.GetPrimAtPath(path), robodojo_rig.wrist_camera_in_link6())
            paths[name] = path
        for name, path in paths.items():
            self.cameras[name] = Camera(
                prim_path=path,
                name=f"robodojo_{name}",
                resolution=tuple(camera_spec(name)["resolution"]),
            )

    def initialize(self, depth=None):
        """After World.reset: render products, RoboDojo intrinsics, renderer settings.

        Depth annotators are attached only when the environment serves depth (--depth).
        """
        if self.no_render:
            return
        if depth is None:
            depth = bool(getattr(self.backend, "serve_depth", False))
        for name, camera in self.cameras.items():
            spec = camera_spec(name)
            camera.initialize()
            if depth:
                camera.add_distance_to_image_plane_to_frame()
            # RoboDojo's calls: focal length, then both apertures without square-pixel sync.
            camera.set_focal_length(spec["focal_length_mm"])
            camera.set_horizontal_aperture(spec["horizontal_aperture_mm"], False)
            camera.set_vertical_aperture(spec["vertical_aperture_mm"], False)
            camera.set_clipping_range(*spec["clipping_range_m"])
            self.intrinsics[name] = np.asarray(camera.get_intrinsics_matrix(), dtype=float).tolist()
            self.rendered_intrinsics[name] = robodojo_rig.intrinsics(spec).tolist()
        apply_renderer()

    def warm_up(self):
        """Render until every policy camera has a nonblank frame (RTX startup)."""
        if self.no_render:
            return
        for _ in range(20):
            self.backend.capture_camera(1)
            if all(
                (rgba := camera.get_rgba()) is not None and rgba.size and np.any(rgba[..., :3])
                for camera in self.cameras.values()
            ):
                break
        else:
            raise RuntimeError("A RoboDojo policy camera produced no RGB frame")
        self.renderer = renderer_state()
        self.frames = None

    def read(self, time, depth=False):
        """Policy images of the scene at `time`: rendered now, or the frames already
        rendered at this instant. Observations render only when a policy reads them
        (roboclock.envs.observation), so this runs once per model query."""
        if self.no_render:
            return zero_images(depth)
        if self.frames is None or not math.isclose(time, self.frame_time, abs_tol=1e-9):
            self.backend.capture_camera(1)
            self.frames = self.grab(depth)
            self.frame_time = time
        elif depth and "depth" not in self.frames:
            self.frames = self.grab(depth)
        return {key: value.copy() for key, value in self.frames.items()}

    def grab(self, depth=False):
        """Copy the current render products (no render call)."""
        images = {}
        for name, (rgb_key, depth_key, _) in VIEWS.items():
            camera = self.cameras[name]
            rgba = camera.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError(f"{name} has no RGB frame. Check RTX rendering.")
            images[rgb_key] = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
            if depth:
                images[depth_key] = depth_image(camera, name)
        return images

    def measure(self):
        """Measure the rig from the stage and check it against the spec (raises on error).

        Bases: the articulation root of each arm. Head and wrist cameras: their camera
        prims. link6: the physics pose of each arm's end-effector link, checked against
        X5A.urdf forward kinematics at the measured joint angles, then used as the parent
        frame for the wrist camera check (1 mm / 0.1 deg).
        """
        b = self.backend
        robots = (b.robot, b.secondary_robot)
        bases = []
        for robot in robots:
            position, quat = robot.get_world_pose()
            bases.append(robodojo_rig.transform(position, _quat_matrix(quat)))
        if self.no_render:
            origin, yaw = rig_pose(b.config, b.task.task_id)
            head = robodojo_rig.head_camera_world(origin, math.degrees(yaw))
            head_source = "spec (sim.no_render: no camera prims)"
        else:
            head = _world_matrix(self.cameras["cam_head"].prim)
            head_source = "measured camera prim"
        summary = robodojo_rig.check_rig(bases[RIG_LEFT_ARM], bases[RIG_RIGHT_ARM], head)
        wrists, link6 = {}, {}
        for name, (_, _, side) in VIEWS.items():
            if side is None:
                continue
            robot = robots[side]
            position, quat = robot.end_effector.get_world_pose()
            measured_link6 = robodojo_rig.transform(position, _quat_matrix(quat))
            urdf_link6 = bases[side] @ link6_in_base(robot.get_joint_positions()[:6])
            distance, angle = _pose_error(measured_link6, urdf_link6)
            link6[ROOTS[side]] = {"error_m": distance, "error_deg": angle}
            if distance > 1e-3 or angle > 0.1:
                raise ValueError(
                    f"{ROOTS[side]}/link6 differs from X5A.urdf link6 by {distance:.4f} m / {angle:.3f} deg"
                )
            if self.no_render:
                continue
            camera = _world_matrix(self.cameras[name].prim)
            distance, angle = _pose_error(camera, measured_link6 @ robodojo_rig.wrist_camera_in_link6())
            wrists[name] = {
                "link": f"{ROOTS[side]}/link6",
                "position_m": camera[:3, 3].round(5).tolist(),
                "error_m": distance,
                "error_deg": angle,
            }
            if distance > 1e-3 or angle > 0.1:
                raise ValueError(f"{name} is {distance:.4f} m / {angle:.3f} deg off its link6 mount")
        left, right = bases[RIG_LEFT_ARM][:3, 3], bases[RIG_RIGHT_ARM][:3, 3]
        self.rig_check = {
            "passed": True,
            "tolerance": {"m": 1e-3, "deg": 0.1},
            "rig_midpoint_m": summary["midpoint"],
            "rig_yaw_deg": summary["yaw_deg"],
            "left_base_m": left.round(5).tolist(),
            "right_base_m": right.round(5).tolist(),
            "left_arm_prim": ROOTS[RIG_LEFT_ARM],
            "right_arm_prim": ROOTS[RIG_RIGHT_ARM],
            "arm_separation_m": float(np.linalg.norm(left - right)),
            "head_camera_m": head[:3, 3].round(5).tolist(),
            "head_camera_source": head_source,
            "wrist_cameras": wrists or "not created (sim.no_render)",
            "link6_vs_x5a_urdf": link6,
        }
        return self.rig_check

    def wrist_poses(self):
        """World pose (position, wxyz) of each wrist camera prim, USD camera axes."""
        poses = {}
        for name, (_, _, side) in VIEWS.items():
            if side is None or name not in self.cameras:
                continue
            matrix = _world_matrix(self.cameras[name].prim)
            poses[name] = {
                "position": matrix[:3, 3].round(5).tolist(),
                "orientation_wxyz": robodojo_rig.quat_wxyz(matrix[:3, :3]).round(5).tolist(),
            }
        return poses

    def metadata(self):
        return {
            "robodojo_rig": robodojo_rig.describe(),
            "rig_check": self.rig_check,
            "renderer": self.renderer,
            "camera_intrinsics": self.intrinsics,
            "camera_intrinsics_rendered": self.rendered_intrinsics,
            "camera_intrinsics_note": INTRINSICS_NOTE,
            "camera_prims": {name: camera.prim_path for name, camera in self.cameras.items()},
            "no_render": self.no_render,
        }

    def close(self):
        for camera in self.cameras.values():
            camera.destroy()
        self.cameras = {}
        self.frames = None


def apply_renderer():
    """RTX Real-Time (RaytracedLighting) with DLAA and Isaac Lab's quality preset."""
    import carb
    import omni.replicator.core as rep

    rep.settings.set_render_rtx_realtime(antialiasing=robodojo_rig.ANTIALIASING)
    settings = carb.settings.get_settings()
    # set_render_rtx_realtime selects Isaac 6's RealTimePathTracing; the spec mode wins.
    settings.set("/rtx/rendermode", robodojo_rig.RENDER_MODE)
    settings.set("/rtx-transient/dldenoiser/enabled", True)
    for key, value in robodojo_rig.RENDER_SETTINGS.items():
        settings.set(key, value)


def renderer_state():
    """Effective renderer settings read back after a rendered frame."""
    import carb

    settings = carb.settings.get_settings()
    aa = settings.get("/rtx/post/aa/op")
    state = {
        "rendermode": settings.get("/rtx/rendermode"),
        "antialiasing_op": aa,
        "antialiasing": AA_NAMES.get(aa, str(aa)),
        "render_settings": {key: settings.get(key) for key in robodojo_rig.RENDER_SETTINGS},
    }
    state["matches_spec"] = (
        state["rendermode"] == robodojo_rig.RENDER_MODE
        and state["antialiasing"] == robodojo_rig.ANTIALIASING
    )
    if state["antialiasing"] != robodojo_rig.ANTIALIASING:
        state["antialiasing_fallback"] = state["antialiasing"]
    return state
