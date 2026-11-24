"""RoboDojo rig cameras of every sports task: head camera, two wrist cameras, rig check.

All ten tasks use one mechanical rig, the RoboDojo dual ARX X5
(``roboclock.envs.robodojo_rig``, the single source of every value here): the two
X5A bases bolted side by side on the crossbeam of the extrusion fixture
(``arx_mount.build_mount``), same height, forward coordinate and heading, 0.60 m
apart. Only the rig as a whole is placed per task: ``embodiment.position`` is the
base midpoint on the mount plane and ``embodiment.yaw`` (radians) its heading.

Cameras (RoboDojo ``camera_config.yml`` / ``robot_config.yml``):

- ``cam_head`` -> ``rgb``: Gemini 345Lg, 640x480, rigid on the rig at (0.04, 0, 0.543) m
  in the rig frame, 60 deg down.
- ``cam_left_wrist`` -> ``left_wrist_rgb`` and ``cam_right_wrist`` -> ``right_wrist_rgb``:
  D435, 640x480, child prims of each arm's ``link6`` at ``wrist_camera_in_link6()``, so
  the renderer moves them with the measured wrist.

The cameras are built with RoboDojo's calls (``set_focal_length``,
``set_horizontal_aperture(h, False)``, ``set_vertical_aperture(v, False)``). The result records both the
intrinsics Isaac's camera API reports (``get_intrinsics_matrix()``, per camera
``intrinsics_matrix``) and the square-pixel matrix RTX actually renders
(``camera_intrinsics_rendered``, ``robodojo_rig.intrinsics()``): RTX ignores the
head camera's vertical aperture, so its rendered vertical field of view is 79.6 deg, not
the reported 71.0 deg.

``measure``/``check`` run at every reset, also under ``sim.no_render`` (no cameras
exist then and only the arm frames are checked): both base frames against
``robodojo_rig.check_rig``, the measured ``link6`` of each arm against forward
kinematics of RoboDojo's X5A.urdf at the measured joint angles, and each wrist
camera against ``link6 @ wrist_camera_in_link6()`` (1 mm / 0.1 deg).
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from roboclock.envs import robodojo_rig as spec

INTRINSICS_NOTE = (
    "camera_intrinsics is what Isaac's get_intrinsics_matrix() reports (fy from the vertical "
    "aperture). RTX ignores the vertical aperture and renders square pixels, so the image "
    "follows camera_intrinsics_rendered (robodojo_rig.intrinsics: fy = fx, principal point at "
    "the centre). Use the rendered matrix for projection and depth unprojection."
)
HEAD_PATH = "/World/HeadCamera"
SENSOR_PATH = HEAD_PATH + "/Sensor"
WRIST_NAME = "RoboDojoWristCamera"
SIDES = ("left", "right")
WRIST_VIEWS = {"cam_left_wrist": "left", "cam_right_wrist": "right"}
VIEWS = ("cam_head", *WRIST_VIEWS)
TOLERANCE_M, TOLERANCE_DEG = 1e-3, 0.1
# RoboDojo's X5A.urdf (identical copy shipped with the industry ARX X5 assets).
X5A_URDF = (
    Path(__file__).resolve().parents[4]
    / "assets/industry/runtime/arx-x5/robot-v1/source/ARX-X5/X5A.urdf"
)
X5_CHAIN = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def observation_shapes(depth=False):
    """Observation key -> (height, width[, 3]) of every rig camera image."""
    shapes = {}
    for view, key in spec.OBSERVATION_KEYS.items():
        width, height = _model(view)["resolution"]
        shapes[key] = (height, width, 3)
        if depth:
            shapes[spec.DEPTH_KEYS[view]] = (height, width)
    return shapes


def blank_images(depth=False):
    """Zero images of the observation shapes (``sim.no_render``)."""
    return {
        key: np.zeros(shape, dtype=np.uint8 if len(shape) == 3 else np.float32)
        for key, shape in observation_shapes(depth).items()
    }


def _model(view):
    return spec.HEAD_CAMERA if view == "cam_head" else spec.WRIST_CAMERA


# -- rig placement ------------------------------------------------------------------
def rig_placement(config):
    """(origin, yaw in degrees) of the task's rig: base midpoint on the mount plane."""
    embodiment = config["embodiment"]
    return (
        np.asarray(embodiment["position"], dtype=float),
        math.degrees(float(embodiment.get("yaw", 0.0) or 0.0)),
    )


def head_world(config):
    """4x4 world pose (USD camera axes) of the head camera for the task's placement."""
    return spec.head_camera_world(*rig_placement(config))


# -- X5A.urdf forward kinematics ----------------------------------------------------
def x5a_chain(path=X5A_URDF):
    """[(fixed origin 4x4, joint axis)] of joint1..joint6 in X5A.urdf (base_link -> link6)."""
    joints = {j.get("name"): j for j in ET.parse(path).getroot().iter("joint")}
    chain = []
    for name in X5_CHAIN:
        joint = joints[name]
        origin = joint.find("origin")
        xyz = [float(v) for v in origin.get("xyz").split()]
        rpy = [float(v) for v in origin.get("rpy").split()]
        axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
        chain.append((spec.transform(xyz, spec.urdf_rpy(*rpy)), axis / np.linalg.norm(axis)))
    return chain


def _axis_angle(axis, angle):
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def link6_in_base(q, chain=None):
    """4x4 pose of ``link6`` in ``base_link`` at arm joint angles ``q`` (X5A.urdf)."""
    pose = np.eye(4)
    for (origin, axis), angle in zip(chain or x5a_chain(), q):
        pose = pose @ origin @ spec.transform((0, 0, 0), _axis_angle(axis, float(angle)))
    return pose


# -- measurement --------------------------------------------------------------------
def _gf_pose(matrix):
    """USD (row-vector) world matrix -> 4x4 column pose with unit rotation axes."""
    m = np.asarray(matrix, dtype=float).T
    rotation = m[:3, :3] / np.linalg.norm(m[:3, :3], axis=0)
    return spec.transform(m[:3, 3], rotation)


def _pose_pq(position, quaternion_wxyz):
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return spec.transform(np.asarray(position, float), rotation)


def pose_error(a, b):
    """(position error m, rotation error deg) between two 4x4 poses."""
    a, b = np.asarray(a), np.asarray(b)
    cosine = np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1)
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(math.degrees(math.acos(cosine)))


def _prim(stage, root, name):
    from pxr import Usd

    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetName() == name:
            return prim
    raise ValueError(f"No {name} under {root}")


def wrist_path(rig, side):
    return rig.rigid[f"{side}_link6"] + "/" + WRIST_NAME


def measure(rig, config):
    """Measured world frames of the rig (USD + PhysX); cameras only if they exist."""
    from pxr import Usd, UsdGeom

    stage = rig.stage
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    names = list(rig.robot.dof_names)
    q = np.asarray(rig.robot.get_joint_positions(), dtype=float).reshape(-1)
    measured = {"bases": {}, "link6_physx": {}, "link6_usd": {}, "joints": {}, "cameras": {}}
    for side in SIDES:
        base = _prim(stage, rig.path, side + "_base_link")
        measured["bases"][side] = _gf_pose(cache.GetLocalToWorldTransform(base))
        measured["link6_physx"][side] = _pose_pq(*rig.tools[side].get_world_pose())
        link6 = stage.GetPrimAtPath(rig.rigid[f"{side}_link6"])
        measured["link6_usd"][side] = _gf_pose(cache.GetLocalToWorldTransform(link6))
        measured["joints"][side] = [q[names.index(f"{side}_{j}")] for j in X5_CHAIN]
        wrist = stage.GetPrimAtPath(wrist_path(rig, side))
        if wrist:
            measured["cameras"][f"cam_{side}_wrist"] = _gf_pose(cache.GetLocalToWorldTransform(wrist))
    head = stage.GetPrimAtPath(SENSOR_PATH)
    if head:
        measured["cameras"]["cam_head"] = _gf_pose(cache.GetLocalToWorldTransform(head))
    return measured


def check(measured, config, *, cameras=True):
    """Raise unless the measured rig is the RoboDojo rig; returns the JSON-ready check.

    ``cameras=False`` (``sim.no_render``): no camera prims exist; the arm frames are
    still checked and the spec head pose stands in for ``check_rig``'s head argument.
    """
    bases = measured["bases"]
    head = measured["cameras"].get("cam_head")
    if cameras and head is None:
        raise ValueError("Sports rig has no head camera prim")
    expected_head = head_world(config)
    rig_check = spec.check_rig(
        bases["left"], bases["right"], head if head is not None else expected_head,
        tolerance_m=TOLERANCE_M, tolerance_deg=TOLERANCE_DEG,
    )
    origin, yaw = rig_placement(config)
    placement = pose_error(spec.rig_to_world(rig_check["midpoint"], rig_check["yaw_deg"]),
                           spec.rig_to_world(origin, yaw))
    if placement[0] > TOLERANCE_M or placement[1] > TOLERANCE_DEG:
        raise ValueError(f"rig placed {placement[0]:.4f} m / {placement[1]:.3f} deg off embodiment.position/yaw")
    chain = x5a_chain()
    wrist_local = spec.wrist_camera_in_link6()
    arms = {}
    for side in SIDES:
        base = bases[side]
        link6 = measured["link6_physx"][side]
        fk = base @ link6_in_base(measured["joints"][side], chain)
        urdf = pose_error(link6, fk)
        if urdf[0] > TOLERANCE_M or urdf[1] > TOLERANCE_DEG:
            raise ValueError(
                f"{side} link6 {urdf[0] * 1000:.2f} mm / {urdf[1]:.3f} deg from X5A.urdf forward kinematics"
            )
        arm = dict(link6_vs_x5a_urdf_m_deg=list(urdf))
        view = f"cam_{side}_wrist"
        camera = measured["cameras"].get(view)
        if cameras and camera is None:
            raise ValueError(f"Sports rig has no {view} prim")
        if camera is not None:
            # The camera prim is a child of link6: the same USD link6 frame carries it.
            error = pose_error(camera, measured["link6_usd"][side] @ wrist_local)
            if error[0] > TOLERANCE_M or error[1] > TOLERANCE_DEG:
                raise ValueError(f"{view} {error[0] * 1000:.2f} mm / {error[1]:.3f} deg from the spec mount")
            arm[f"{view}_vs_spec_m_deg"] = list(error)
            arm[f"{view}_position"] = camera[:3, 3].round(5).tolist()
        arms[side] = arm
    result = dict(
        check_rig=rig_check,
        placement=dict(position=origin.tolist(), yaw_deg=yaw),
        left_base=bases["left"][:3, 3].round(5).tolist(),
        right_base=bases["right"][:3, 3].round(5).tolist(),
        separation_m=float(np.linalg.norm(bases["left"][:3, 3] - bases["right"][:3, 3])),
        arms=arms,
        tolerance=dict(m=TOLERANCE_M, deg=TOLERANCE_DEG),
        cameras_measured=sorted(measured["cameras"]),
    )
    if head is not None:
        result["head_vs_spec_m_deg"] = list(pose_error(head, expected_head))
        result["head_position"] = head[:3, 3].round(5).tolist()
    return result


def measure_and_check(rig, config, *, cameras=True):
    return check(measure(rig, config), config, cameras=cameras)


# -- renderer -----------------------------------------------------------------------
AA_NAMES = {0: "Off", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}


def apply_renderer():
    """RTX Real-Time (RaytracedLighting) with DLAA and Isaac Lab's ``quality`` preset."""
    import carb

    settings = carb.settings.get_settings()
    try:
        import omni.replicator.core as rep

        # Isaac Lab's call; on Isaac Sim 6 it also selects RTX Real-Time 2.0, so the
        # spec's render mode is set explicitly afterwards.
        rep.settings.set_render_rtx_realtime(antialiasing=spec.ANTIALIASING)
    except Exception:  # replicator unavailable: set the same keys directly
        settings.set("/rtx/post/aa/op", 4)
    settings.set("/rtx/rendermode", spec.RENDER_MODE)
    settings.set("/rtx-defaults/post/aa/op", 4)
    for key, value in spec.RENDER_SETTINGS.items():
        settings.set(key, value)
    return renderer_state()


def renderer_state():
    """Effective renderer settings read back from carb (not the requested values)."""
    import carb

    settings = carb.settings.get_settings()
    aa = settings.get("/rtx/post/aa/op")
    mode = settings.get("/rtx/rendermode")
    state = dict(
        render_mode=mode,
        antialiasing_op=aa,
        antialiasing=AA_NAMES.get(aa, str(aa)),
        settings={key: settings.get(key) for key in spec.RENDER_SETTINGS},
    )
    if state["antialiasing"] != spec.ANTIALIASING:
        state["antialiasing_fallback"] = state["antialiasing"]
    mismatched = [k for k, v in spec.RENDER_SETTINGS.items() if state["settings"][k] != v]
    if mode != spec.RENDER_MODE:
        mismatched.append("/rtx/rendermode")
    state["matches_spec"] = not mismatched and "antialiasing_fallback" not in state
    state["mismatched"] = mismatched
    return state


# -- cameras ------------------------------------------------------------------------
def _build_camera(path, view):
    from isaacsim.sensors.camera import Camera

    model = _model(view)
    camera = Camera(path, resolution=tuple(model["resolution"]))
    camera.initialize()
    # RoboDojo's calls; apertures without the square-pixel correction.
    camera.set_focal_length(model["focal_length_mm"])
    camera.set_horizontal_aperture(model["horizontal_aperture_mm"], False)
    camera.set_vertical_aperture(model["vertical_aperture_mm"], False)
    camera.set_clipping_range(*model["clipping_range_m"])
    return camera


def _hardware(stage, config):
    """Visual post and housing of the head camera; no collider or mass."""
    from pxr import Gf, UsdGeom, UsdShade

    from roboclock.simulators.isaac.embodiments.arx_appearance import material

    origin, yaw = rig_placement(config)
    root = UsdGeom.Xform.Define(stage, HEAD_PATH + "/Hardware")
    root.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    root.AddRotateZOp().Set(float(yaw))
    root.GetPrim().SetCustomDataByKey(
        "benchmark:provenance", "Authored head-camera post and housing; visual only"
    )
    alloy = material(stage, HEAD_PATH + "/Alloy", (0.31, 0.34, 0.37), 0.3, 0.8)
    dark = material(stage, HEAD_PATH + "/Coating", (0.045, 0.055, 0.067), 0.4, 0.25)
    pitch = spec.HEAD_PITCH_DOWN_DEG
    hx, _, hz = spec.HEAD_IN_RIG_M

    def box(name, center, size, mat, tilt=0.0):
        cube = UsdGeom.Cube.Define(stage, HEAD_PATH + "/Hardware/" + name)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
        if tilt:
            cube.AddRotateYOp().Set(float(tilt))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    # Post well behind the optical centre (outside the head view), then an arm
    # forward to a housing that sits behind and above the lens.
    top = hz + 0.03
    box("Post", [-0.11, 0, (top - 0.01) / 2], [0.03, 0.03, top + 0.01], alloy)
    box("Arm", [(-0.11 + hx) / 2 - 0.01, 0, top], [hx + 0.11, 0.03, 0.012], alloy)
    look = np.array([math.cos(math.radians(pitch)), 0, -math.sin(math.radians(pitch))])
    box("Housing", np.array([hx, 0, hz]) - look * 0.02, [0.025, 0.09, 0.025], dark, pitch)


class RigCameras:
    """The three RoboDojo policy cameras of one backend stage."""

    def __init__(self, backend, *, hardware=True):
        self.backend = backend
        rig, config, stage = backend.rig, backend.config, backend.stage
        self.stage = stage
        if hardware and not stage.GetPrimAtPath(HEAD_PATH + "/Hardware"):
            _hardware(stage, config)
        self.paths = {"cam_head": SENSOR_PATH}
        self.paths.update({view: wrist_path(rig, side) for view, side in WRIST_VIEWS.items()})
        self.cameras = {view: _build_camera(self.paths[view], view) for view in VIEWS}
        head = head_world(config)
        self.cameras["cam_head"].set_world_pose(
            head[:3, 3], spec.quat_wxyz(head[:3, :3]), camera_axes="usd"
        )
        wrist = spec.wrist_camera_in_link6()
        for view in WRIST_VIEWS:
            self.cameras[view].set_local_pose(
                wrist[:3, 3], spec.quat_wxyz(wrist[:3, :3]), camera_axes="usd"
            )
        self.depth = False
        self.measured = measure_and_check(rig, config)

    def enable_depth(self):
        for camera in self.cameras.values():
            camera.add_distance_to_image_plane_to_frame()
        self.depth = True

    def rgb(self, view):
        rgba = self.cameras[view].get_rgba()
        if rgba is None or np.asarray(rgba).ndim != 3 or not np.asarray(rgba).size:
            raise RuntimeError(f"{view} has no RGB frame; check RTX rendering")
        return np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()

    def depth_image(self, view):
        raw = self.cameras[view].get_depth(device="cpu")
        if raw is None:
            raise RuntimeError(f"{view} has no fresh depth frame")
        depth = np.asarray(raw, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return depth.copy()

    def images(self, depth=False):
        """Observation key -> image of every camera (already rendered)."""
        images = {spec.OBSERVATION_KEYS[v]: self.rgb(v) for v in VIEWS}
        if depth:
            images.update({spec.DEPTH_KEYS[v]: self.depth_image(v) for v in VIEWS})
        for key, shape in observation_shapes(depth).items():
            if images[key].shape != shape:
                raise RuntimeError(f"{key} has shape {images[key].shape}, expected {shape}")
        return images

    def intrinsics(self):
        """What the camera API reports after construction (focal, apertures, K).

        ``intrinsics_matrix`` is the reported K; the image follows ``rendered_intrinsics()``.
        """
        report = {}
        for view, camera in self.cameras.items():
            report[view] = dict(
                prim=self.paths[view],
                resolution=list(camera.get_resolution()),
                focal_length=float(camera.get_focal_length()),
                horizontal_aperture=float(camera.get_horizontal_aperture()),
                vertical_aperture=float(camera.get_vertical_aperture()),
                clipping_range=list(camera.get_clipping_range()),
                intrinsics_matrix=np.asarray(camera.get_intrinsics_matrix()).round(4).tolist(),
            )
        return report

    def describe(self):
        return dict(
            rendered=True,
            cameras=self.intrinsics(),
            camera_intrinsics_rendered=rendered_intrinsics(),
            camera_intrinsics_note=INTRINSICS_NOTE,
            rig_check=self.measured,
        )


def rendered_intrinsics():
    """Square-pixel K that RTX renders, per RoboDojo view name."""
    return {
        view: spec.intrinsics(spec.HEAD_CAMERA if view == "cam_head" else spec.WRIST_CAMERA)
        .round(4)
        .tolist()
        for view in spec.OBSERVATION_KEYS
    }


def describe(rig_check, cameras=None, renderer=None):
    """Result-JSON block: the shared spec, the measured rig check, cameras and renderer.

    ``cameras[view]["intrinsics_matrix"]`` is the reported K; ``camera_intrinsics_rendered``
    is the rendered one.
    """
    rendered = None if cameras is None else {v: k for v, k in rendered_intrinsics().items() if v in cameras}
    return dict(
        robodojo_rig=spec.describe(),
        rig_check=rig_check,
        cameras=cameras,
        camera_intrinsics_rendered=rendered,
        camera_intrinsics_note=None if cameras is None else INTRINSICS_NOTE,
        rendered=cameras is not None,
        renderer=renderer,
    )


def ensure(backend):
    """Install the rig cameras once per backend stage (before the first render)."""
    cams = getattr(backend, "rig_cameras", None)
    if cams is None or cams.stage is not backend.stage:
        cams = backend.rig_cameras = RigCameras(backend)
    return cams


def disable_updates(camera):
    """Stop rendering a camera's render product (presentation camera during eval)."""
    product = getattr(camera, "_render_product", None)
    texture = getattr(product, "hydra_texture", None)
    if texture is None:
        return False
    texture.set_updates_enabled(False)
    return True


class RigVideoEncoder:
    """Write head/left_wrist/right_wrist (640x480) beside the overview, frame-synchronized.

    The wrapped encoder's caller has rendered the frame (``b.frame()`` captures every
    render product), so all four images belong to the same simulation instant.
    """

    FILES = {"cam_head": "head", "cam_left_wrist": "left_wrist", "cam_right_wrist": "right_wrist"}

    def __init__(self, encoder, cams, output):
        from roboclock.simulators.isaac.video import VideoEncoder

        self.encoder, self.cams = encoder, cams
        self.output = Path(output)
        self.fps = spec.VIDEO_FPS
        self.encoders = {
            view: VideoEncoder(self.output / f"{name}.mp4", _model(view)["resolution"], self.fps)
            for view, name in self.FILES.items()
        }
        self.frames = 0
        self.first, self.last = {}, {}
        self.timestamps = []

    def __getattr__(self, name):
        return getattr(self.encoder, name)

    def write(self, rgb):
        before = float(self.cams.backend.world.current_time)
        self.encoder.write(rgb)
        for view, encoder in self.encoders.items():
            frame = self.cams.rgb(view)
            encoder.write(frame)
            self.first.setdefault(view, frame)
            self.last[view] = frame
        if float(self.cams.backend.world.current_time) != before:
            raise RuntimeError("Recording advanced physics time")
        self.timestamps.append(before)
        self.frames += 1

    def close(self):
        import json

        from PIL import Image

        result = self.encoder.close()
        for view, encoder in self.encoders.items():
            encoder.close()
            name = self.FILES[view]
            if view in self.first:
                Image.fromarray(self.first[view]).save(self.output / f"{name}_initial.png")
                Image.fromarray(self.last[view]).save(self.output / f"{name}_final.png")
        (self.output / "rig_views.json").write_text(
            json.dumps(
                dict(
                    frames=self.frames,
                    fps=self.fps,
                    files={self.FILES[v]: f"{self.FILES[v]}.mp4" for v in self.encoders},
                    synchronized=True,
                    physics_advanced_during_capture=False,
                    simulation_timestamps=self.timestamps,
                    **self.cams.describe(),
                ),
                indent=2,
            )
        )
        return result


def attach(encoder, backend, output):
    """Record the rig cameras beside ``encoder``; without a video nothing is installed."""
    if encoder is None:
        return None
    return RigVideoEncoder(encoder, ensure(backend), output)
