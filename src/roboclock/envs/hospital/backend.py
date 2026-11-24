"""Isaac Sim hospital scene: built entirely from the layout, with no per-task branching.

The robot is the laboratory's dual ARX X5 (roboclock.envs.lab.robot); single-arm hospital tasks
drive internal arm 0 (/World/ARXLeft despite its name: the rig's RIGHT arm) and arm 1 (the rig's
left arm) holds its home pose. The backend works in this internal arm order (`apply_action`,
`joint_positions`, `observation`); HospitalEnv/LabEnv present the rig's left arm first.

Imported only after SimulationApp starts. State measurement, camera capture, action application
and teardown are inherited from the laboratory backend. Objects are never attached or teleported
to manufacture success; held objects are released by removing a fixed support joint.

The bedside A1-A4 path has real Isaac Sim 6.0.1 rendering/contact diagnostics.
"""

import math
from pathlib import Path

import numpy as np
import yaml

from roboclock.envs.hospital.appearance import HospitalAppearance
from roboclock.envs.hospital.clutter import sample_clutter
from roboclock.envs.hospital.layout import FLOOR, WALL, make_layout
from roboclock.envs.lab.backend import IsaacBackend
from roboclock.envs.lab.rig_cameras import depth_image
from roboclock.physics.contacts import GraspTracker
from roboclock.resources import resolve_asset

VESSELS = ("med_bottle", "saline_bottle")  # drawn as glass vials with caps
INDICATOR_SIZE = (0.050, 0.070, 0.100)


def _apply_intrinsics(camera, template):
    """RoboDojo's camera API calls (robodojo_rig): focal length, then both apertures without
    forcing square pixels, then the clipping range."""
    camera.set_focal_length(template["focal_length_mm"])
    camera.set_horizontal_aperture(template["horizontal_aperture_mm"], False)
    camera.set_vertical_aperture(template["vertical_aperture_mm"], False)
    camera.set_clipping_range(*template["clipping_range_m"])


def apply_render_spec(read_only=False):
    """robodojo_rig renderer: RTX Real-Time with DLAA (Isaac Lab's set_render_rtx_realtime) and the
    `quality` preset, then /rtx/rendermode = RENDER_MODE. Returns what the renderer reports back,
    including the antialiasing actually in effect (a fallback if DLAA is unavailable)."""
    import carb

    from roboclock.envs import robodojo_rig as spec

    settings = carb.settings.get_settings()
    if not read_only:
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime(antialiasing=spec.ANTIALIASING)
        settings.set("/rtx/rendermode", spec.RENDER_MODE)
        for key, value in spec.RENDER_SETTINGS.items():
            settings.set(key, value)
    names = {0: "Off", 1: "TAA", 2: "FXAA", 3: "DLSS", 4: "DLAA"}
    op = settings.get("/rtx/post/aa/op")
    antialiasing = names.get(op, str(op))
    return {
        "rendermode": settings.get("/rtx/rendermode"),
        "antialiasing": antialiasing,
        "aa_op": op,
        "antialiasing_fallback": None if antialiasing == spec.ANTIALIASING else antialiasing,
        "render_settings": {key: settings.get(key) for key in spec.RENDER_SETTINGS},
    }


class HospitalBackend(IsaacBackend):
    # hospital builds its own robodojo_rig policy cameras (head + wrists, `render_frame`) and
    # renders them from HospitalEnv.get_obs; the laboratory's camera path stays off here.
    robodojo_cameras = False

    @property
    def no_render(self):
        """sim.no_render: no cameras, no render calls, zero images of the same shapes."""
        return bool(self.config["sim"].get("no_render", False))

    def __init__(self, runtime, config, asset_catalog=None):
        path = (
            resolve_asset("env://catalog.yaml", environment="hospital")
            if asset_catalog is None
            else asset_catalog
        )
        with open(Path(path), encoding="utf-8") as stream:
            catalog = yaml.safe_load(stream)
        for name, spec in catalog.items():
            if spec.get("usd_path"):
                raise ValueError(f"{name}: custom USD assets are not supported for hospital yet")
        super().__init__(runtime, config, path)
        self.catalog = catalog
        self.indicators = {}
        self.zone_prims = {}
        self.timer_bars = {}

    # --- scene construction --------------------------------------------------------------

    def reset(self, task):
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import euler_angles_to_quat, gf_quat_to_np_array
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom

        self.close()
        World.clear_instance()
        create_new_stage()
        self.objects, self.target_parts, self.supports = {}, {}, {}
        self.indicators, self.zone_prims, self.kinematic_targets = {}, {}, {}
        self.timer_bars = {}
        self.layout = make_layout(task, self.config, self.catalog)
        task.bind(self.layout, self.config["sim"]["placement_dwell"])
        self.task = task
        task.recovery_dwell = self.config["physics"]["recovery_dwell"]
        task.catch_dwell = self.config["physics"]["catch_dwell"]
        dt = self.config["sim"]["physics_dt"]
        self.world = World(stage_units_in_meters=1.0, physics_dt=dt, rendering_dt=dt)
        self.world.get_physics_context().enable_ccd(True)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()
        ground_scale = float(self.config["visuals"].get("ground_light_scale", 1.0))
        if ground_scale != 1.0:  # the default ground plane's SphereLight (see benchmark.yaml)
            from pxr import Usd, UsdLux

            for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/defaultGroundPlane")):
                if prim.HasAPI(UsdLux.LightAPI):
                    attr = UsdLux.LightAPI(prim).GetIntensityAttr()
                    attr.Set(float(attr.Get()) * ground_scale)
        from roboclock.simulators.isaac.rendering import configure_renderer

        configure_renderer(self.config)
        apply_render_spec()
        film_iso = self.config["tasks"].get(task.task_id, {}).get("film_iso")
        if film_iso is not None:  # per-task exposure, e.g. a white counter vs a dim bedside
            import carb

            carb.settings.get_settings().set("/rtx/post/tonemap/filmIso", float(film_iso))
        scene = self.config["scene"]
        self.appearance = HospitalAppearance(self.stage, self.world, self.config)
        physics = self.config["physics"]
        self.grasps = GraspTracker(
            physics["minimum_grasp_time"],
            physics["contact_force_threshold"],
            physics["relative_hold_speed"],
        )
        self._cached_state = None
        self._previous_tool = None
        self._previous_object_positions = {}
        self._release_speeds = {}
        # A ward task may replace the generic bench with real furniture (counter, cart top):
        # info["work_surface"] = {"center", "size"} sizes the invisible collider to match it.
        surface = self.layout.info.get("work_surface") or {
            "center": scene["table_center"],
            "size": scene["table_size"],
        }
        table = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array(surface["center"]),
                scale=np.array(surface["size"]),
            )
        )
        UsdGeom.Imageable(table.prim).MakeInvisible()
        if not self.layout.info.get("hide_bench"):
            self.appearance.rounded(
                "/World/TableVisual", scene["table_center"], scene["table_size"], "bench_top", 0.006
            )
        from roboclock.envs.lab.robot import create_robots

        create_robots(self)
        self.appearance.texture_robot_mounts()
        self.appearance.lights(self.layout)
        self.appearance.workbench(self.layout)
        for name, path in getattr(self.appearance, "callouts", {}).items():
            self.zone_prims[name] = self.stage.GetPrimAtPath(path)
        self.clutter_specs = sample_clutter(task, self.layout, self.config)
        from roboclock.envs.lab.clutter import build_clutter

        self.clutter = build_clutter(self) if self.clutter_specs else {}
        self.clutter_robot_contact_count = 0
        self._last_clutter_contact = False

        self.fixture_bodies = {}
        for fixture in self.layout.fixtures:
            self._fixture(fixture)
        from roboclock.envs.hospital.drawer import author_drawer

        for drawer in self.layout.drawers:
            author_drawer(self.stage, self.appearance, drawer, self._collision_settings)
        for indicator in self.layout.indicators:
            self._indicator(indicator, PreviewSurface, VisualCuboid)
        for name, target in self.layout.targets.items():
            self._target(name, target)
        for index, tag in enumerate(self.layout.tags):
            if self.config["visuals"]["labels"]:
                self.appearance.label(
                    f"/World/Tags/Tag_{index}",
                    tag.text,
                    tag.position,
                    list(tag.size),
                    (0.1, 0.35, 0.5),
                )
        for name, obj in self.layout.objects.items():
            material = PhysicsMaterial(
                f"/World/Materials/{name}",
                static_friction=obj.friction,
                dynamic_friction=obj.friction,
                restitution=0.0,
            )
            self.objects[name] = self._object(obj, material)
            if name in self.layout.held:
                self._support(name, obj)

        from roboclock.envs.hospital.bedside import module_for

        bedside = module_for(task)
        if bedside is not None:
            bedside.author(self)
        if "spill" in self.layout.info:  # B3 hazardous drug spill: models, hidden glass pieces
            from roboclock.envs.hospital import spill_scene

            spill_scene.author(self)
        # Every countdown dial is built facing +x (the old third-person views); on the RoboDojo rig
        # each one turns to the head camera so the policy's `rgb` reads its face, not its back.
        # Ward modules that turn their own dial group (/World/Dials) aim it at the head camera
        # themselves (rig.dial_target).
        dials = self.stage.GetPrimAtPath("/World/Dials")
        own = dials.IsValid() and bool(UsdGeom.Xformable(dials).GetOrderedXformOps())
        if not own and (self.layout.info.get("face_dials") or "rig" in self.config.get("robot", {})):
            self._face_dials_to_head(task.task_id)

        self.surface_paths = {}
        for name, parts in self.target_parts.items():
            self.surface_paths[name] = parts[0][0].prim_path
            for body, _ in parts[1:]:
                self.surface_paths[f"{name}:{body.prim_path.rsplit('/', 1)[-1]}"] = body.prim_path
        for name, body in self.fixture_bodies.items():
            self.surface_paths[f"fixture:{name}"] = body.prim_path
        self.surface_paths["ground"] = "/World/defaultGroundPlane"
        self.surface_paths.update({name: body.prim_path for name, body in self.clutter.items()})
        arm_filter = (
            ["/World/Table"]
            + [body.prim_path for parts in self.target_parts.values() for body, _ in parts]
            + [body.prim_path for body in self.fixture_bodies.values()]
            + [
                path
                for path in ("/World/ARXMount0", "/World/ARXMount1")
                if self.stage.GetPrimAtPath(path).IsValid()
            ]
            + [body.prim_path for body in self.clutter.values()]
        )
        self.contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[f"/World/Objects/{name}" for name in self.objects],
                name="sample_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    [
                        "/World/ARXLeft/link7",
                        "/World/ARXLeft/link8",
                        "/World/ARXLeft/link6",
                        "/World/Table",
                    ]
                    + list(self.surface_paths.values())
                    for _ in self.objects
                ],
                max_contact_count=4096,
            )
        )
        self.arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[f"/World/ARXLeft/link{i}" for i in range(1, 7)],
                name="arm_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[arm_filter for _ in range(6)],
                max_contact_count=4096,
            )
        )
        self.secondary_sample_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[f"/World/Objects/{name}" for name in self.objects],
                name="secondary_sample_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    ["/World/ARXRight/link7", "/World/ARXRight/link8", "/World/ARXRight/link6"]
                    for _ in self.objects
                ],
                max_contact_count=4096,
            )
        )
        from roboclock.envs.lab.dual_arm import build_collision_monitor

        self.secondary_contacts = build_collision_monitor(
            self,
            [body.prim_path for parts in self.target_parts.values() for body, _ in parts]
            + [body.prim_path for body in self.fixture_bodies.values()],
        )
        # Policy cameras (robodojo_rig): the rig's head camera (`rgb`) and one wrist camera on each
        # arm's link6 (`right_wrist_rgb` on arm 0, `left_wrist_rgb` on arm 1). The third-person
        # task camera (bedside modules aim it in `measure`) exists only when a recording asks for
        # it (sim.task_camera). sim.no_render builds no camera at all.
        from roboclock.envs import robodojo_rig as spec
        from roboclock.envs.hospital.rig import head_camera_world
        from roboclock.envs.lab.robot import ROOTS

        head_pose = head_camera_world(self.config, task.task_id)
        head_position = head_pose[:3, 3].copy()
        head_orientation = spec.quat_wxyz(head_pose[:3, :3])
        self.appearance_head_mount(head_position)
        # The head camera's rigid frame, authored even without cameras so the reset check always
        # measures the head pose from the stage.
        frame = UsdGeom.Xform.Define(self.stage, "/World/Rig/HeadFrame")
        frame.AddTransformOp().Set(Gf.Matrix4d(*head_pose.T.reshape(-1).tolist()))
        self.camera = None
        self.wrist_cameras = [None, None]
        wrist_pose = spec.wrist_camera_in_link6()
        if not self.no_render:
            self.camera = Camera(
                prim_path="/World/HeadCamera",
                position=head_position,
                orientation=head_orientation,
                resolution=tuple(spec.HEAD_CAMERA["resolution"]),
            )
            for side, root in enumerate(ROOTS):
                self.wrist_cameras[side] = Camera(
                    prim_path=f"{root}/{spec.WRIST_PARENT_LINK}/RigWristCamera",
                    translation=wrist_pose[:3, 3].copy(),
                    orientation=spec.quat_wxyz(wrist_pose[:3, :3]),
                    resolution=tuple(spec.WRIST_CAMERA["resolution"]),
                )
        self.task_camera = None
        if self.config["sim"].get("task_camera") and not self.no_render:
            camera_position = (
                np.array(
                    scene.get("bedside_camera_position", scene["camera_position"])
                    if bedside
                    else scene["camera_position"]
                )
                + self.layout.camera_jitter
            )
            camera_target = (
                scene.get("bedside_camera_look_at", scene["camera_look_at"])
                if bedside
                else scene["camera_look_at"]
            )
            orientation = self._look_at(camera_position, np.asarray(camera_target, dtype=float))
            resolution = self.config["sim"].get(
                "task_camera_resolution", self.config["sim"]["camera_resolution"]
            )
            self.task_camera = Camera(
                prim_path="/World/Camera",
                position=camera_position,
                orientation=orientation,
                resolution=tuple(resolution),
            )
        self.world.reset()
        from roboclock.envs.lab.robot import initialize_robots

        initialize_robots(self)
        if bedside is not None:
            self._prepare_bedside_posture()
        if self.camera is not None:
            depth = bool(getattr(self, "want_depth", False))
            self.camera.initialize()
            self.camera.set_world_pose(head_position, head_orientation, camera_axes="usd")
            _apply_intrinsics(self.camera, spec.HEAD_CAMERA)
            if depth:
                self.camera.add_distance_to_image_plane_to_frame()
            for camera in self.wrist_cameras:
                camera.initialize()
                camera.set_local_pose(
                    wrist_pose[:3, 3].copy(), spec.quat_wxyz(wrist_pose[:3, :3]), camera_axes="usd"
                )
                _apply_intrinsics(camera, spec.WRIST_CAMERA)
                if depth:
                    camera.add_distance_to_image_plane_to_frame()
        if self.task_camera is not None:
            self.task_camera.initialize()
            self.task_camera.set_world_pose(camera_position, orientation, camera_axes="usd")
            self.task_camera.set_focal_length(scene["camera_focal_length"] / 1000)
            self.task_camera.set_horizontal_aperture(scene["camera_horizontal_aperture"] / 1000)
            self.task_camera.set_clipping_range(0.01, 10.0)
        for _ in range(round(self.config["sim"]["settle_seconds"] / dt)):
            self.world.step(render=False)
        apply_render_spec()  # again after scene authoring, before the first render
        for _ in range(0 if self.no_render else 5):
            self.world.render()
            rgba = self.camera.get_rgba()
            if rgba is not None and rgba.size:
                break
        self.render_state = apply_render_spec(read_only=True)
        views = (
            ("rgb", self.camera, spec.HEAD_CAMERA),
            ("right_wrist_rgb", self.wrist_cameras[0], spec.WRIST_CAMERA),
            ("left_wrist_rgb", self.wrist_cameras[1], spec.WRIST_CAMERA),
        )
        # Reported: what get_intrinsics_matrix() returns (fy from the vertical aperture).
        self.camera_intrinsics = {
            key: None if camera is None else np.asarray(camera.get_intrinsics_matrix()).tolist()
            for key, camera, _ in views
        }
        # Rendered: RTX ignores the vertical aperture and keeps square pixels (fy = fx).
        self.camera_intrinsics_rendered = {
            key: None if camera is None else spec.intrinsics(template).tolist()
            for key, camera, template in views
        }
        self.rig_check = self._measure_rig(task.task_id)
        # Setup and settling never advance the benchmark's episode clock.
        self.last_collision = False
        self.collision_count = 0
        self._cached_state = self._measure_state(0.0)
        task.previous = self.state()
        self.home_orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
        self.frame_time = None
        self.render_frame()
        return self.observation()

    def capture_camera(self, subframes=1):
        if not getattr(self, "no_render", False):
            super().capture_camera(subframes)

    def render_frame(self, subframes=1):
        """One render of every policy camera at the current instant (`images` calls it when a
        policy reads an observation; recordings call it per video frame). sim.no_render: zero
        images of the same shapes and no render call."""
        from roboclock.envs import robodojo_rig as spec

        depth = bool(getattr(self, "want_depth", False))
        cameras = {
            "rgb": (self.camera, spec.HEAD_CAMERA, "depth"),
            "right_wrist_rgb": (self.wrist_cameras[0], spec.WRIST_CAMERA, "right_wrist_depth"),
            "left_wrist_rgb": (self.wrist_cameras[1], spec.WRIST_CAMERA, "left_wrist_depth"),
        }
        frames = {}
        if self.no_render:
            for key, (_, camera, depth_key) in cameras.items():
                width, height = camera["resolution"]
                frames[key] = np.zeros((height, width, 3), dtype=np.uint8)
                if depth:
                    frames[depth_key] = np.zeros((height, width), dtype=np.float32)
        else:
            self.capture_camera(subframes)
            for key, (camera, _, depth_key) in cameras.items():
                rgba = camera.get_rgba()
                if rgba is None or rgba.size == 0:
                    raise RuntimeError(f"{key}: camera has no RGB frame")
                frames[key] = np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()
                if depth:
                    frames[depth_key] = depth_image(camera, key)
        self.frames = frames
        self.frame_time = self.task.time
        return frames

    def images(self, depth=False):
        """Policy camera images of the current instant: the frames `render_frame` made at
        this simulation time, or a new render."""
        if self.frame_time is None or not math.isclose(self.frame_time, self.task.time, abs_tol=1e-9):
            self.render_frame()
        result = {key: value.copy() for key, value in self.frames.items() if "depth" not in key}
        if depth:
            if "depth" not in self.frames:
                raise RuntimeError("depth observations need the env built with depth=True")
            result.update({k: v.copy() for k, v in self.frames.items() if "depth" in k})
        return result

    def observation(self, depth=False, images=True):
        """Dual-arm proprioception, preceded by the policy cameras unless `images` is false."""
        position, orientation = self._tool_pose()
        pos2, quat2 = self._tool_pose(self.secondary_robot)
        q = self.joint_positions()
        result = self.images(depth) if images else {}
        result.update(
            joint_positions=np.asarray(q, dtype=np.float32),
            ee_pose=np.asarray([*position, *orientation, *pos2, *quat2], dtype=np.float32),
            gripper_width=np.asarray(
                [np.clip(q[6] + q[7], 0, 0.088), np.clip(q[14] + q[15], 0, 0.088)],
                dtype=np.float32,
            ),
            instruction=self.task.gen_instruction(),
            sim_time=np.array([self.task.time], dtype=np.float32),
        )
        return result

    def _measure_rig(self, task_id):
        """robodojo_rig requirement 2, measured from the stage at every reset: both base frames
        and the head camera (`check_rig`), each link6 against X5A.urdf forward kinematics, and
        each wrist camera against `wrist_camera_in_link6()` on the measured link6."""
        from pxr import Usd, UsdGeom

        from roboclock.envs.hospital.rig import check_measured
        from roboclock.physics.geometry import rotation_matrix

        def pose(position, quat):
            matrix = np.eye(4)
            matrix[:3, :3] = rotation_matrix(np.asarray(quat, dtype=float))
            matrix[:3, 3] = np.asarray(position, dtype=float)
            return matrix

        def stage_pose(path):
            prim = self.stage.GetPrimAtPath(path)
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            return np.array(matrix, dtype=float).T

        robots = (self.robot, self.secondary_robot)
        bases = [pose(*robot.get_world_pose()) for robot in robots]
        link6 = [pose(*robot.end_effector.get_world_pose()) for robot in robots]
        joints = [np.asarray(robot.get_joint_positions(), dtype=float) for robot in robots]
        if self.camera is not None:
            head = pose(*self.camera.get_world_pose(camera_axes="usd"))
            wrists = [link6[i] @ pose(*camera.get_local_pose(camera_axes="usd")) for i, camera in enumerate(self.wrist_cameras)]
        else:
            head = stage_pose("/World/Rig/HeadFrame")
            wrists = [None, None]
        result = check_measured(task_id, bases, head, link6, joints, wrists)
        result["head_measured_from"] = "/World/HeadCamera" if self.camera is not None else "/World/Rig/HeadFrame"
        return result

    def _face_dials_to_head(self, task_id):
        """Turn each countdown dial (built facing +x) and its label about its own vertical axis to
        face the rig's head camera, so the policy's `rgb` reads it."""
        import math

        from pxr import Gf, UsdGeom

        from roboclock.envs.hospital.rig import head_camera_pose

        head, _, _ = head_camera_pose(self.config, task_id)
        for indicator in self.layout.indicators:
            if getattr(indicator, "style", "dial") != "dial":
                continue
            x, y = (float(v) for v in indicator.position[:2])
            yaw = math.degrees(math.atan2(float(head[1]) - y, float(head[0]) - x))
            prefix = f"/World/Dials/{indicator.name}_"
            paths = [
                str(p.GetPath())
                for p in self.stage.GetPrimAtPath("/World/Dials").GetChildren()
                if str(p.GetPath()).startswith(prefix)
            ] + [f"/World/Labels/indicator_{indicator.name}"]
            for path in paths:
                prim = self.stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                xform = UsdGeom.Xformable(prim)
                order = xform.GetOrderedXformOps()
                pivot = xform.AddTranslateOp(opSuffix="facePivot")
                pivot.Set(Gf.Vec3d(x, y, 0.0))
                turn = xform.AddRotateZOp(opSuffix="face")
                turn.Set(yaw)
                back = xform.AddTranslateOp(opSuffix="faceUnpivot")
                back.Set(Gf.Vec3d(-x, -y, 0.0))
                xform.SetXformOpOrder([pivot, turn, back, *order])

    def close(self):
        for camera in [*getattr(self, "wrist_cameras", []), getattr(self, "task_camera", None)]:
            if camera is not None:
                camera.destroy()
        self.wrist_cameras, self.task_camera = [None, None], None
        super().close()

    @staticmethod
    def _look_at(position, target):
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from pxr import Gf

        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*np.asarray(position, dtype=float).tolist()),
            Gf.Vec3d(*np.asarray(target, dtype=float).tolist()),
            Gf.Vec3d(0, 0, 1),
        )
        return gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())

    def appearance_head_mount(self, head_position):
        """Visual rig hardware (no colliders): a bridge plate joining the two X5 columns, a mast
        and a boom carrying the head camera body, all outside the head camera's view."""
        from roboclock.envs.hospital.rig import rig_mounts

        mounts, yaw = rig_mounts(self.config, self.task.task_id)
        mounts = [np.asarray(m, dtype=float) for m in mounts]
        center = 0.5 * (mounts[0] + mounts[1])
        span = float(np.linalg.norm(mounts[0][:2] - mounts[1][:2]))
        z = float(center[2])
        a = self.appearance
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        coat = a.textured("rig_coat", "plastic_matte", color=(0.86, 0.88, 0.90))
        dark = a.material("rig_camera_body", color=[0.05, 0.05, 0.06], roughness=0.5)

        def box(path, position, size, material):
            from pxr import UsdGeom

            mesh = a.rounded(path, position, size, material, 0.004)
            if abs(yaw) > 1e-9:
                UsdGeom.Xformable(mesh).AddRotateZOp().Set(float(np.degrees(yaw)))

        # The 60-degree head camera sees slightly behind itself at its lower image edge, so the
        # hardware stands behind the bases' line: a slim 20 mm bridge plate just under the mount
        # plane (it clears what stands between the bases, e.g. the B2 AED lid at 1.01 m), a mast
        # 0.14 m behind the midpoint and a boom above the lens carrying the camera body.
        back = center - 0.14 * forward
        bridge = center - 0.115 * forward  # its front edge 0.06 m behind the bases' line
        box("/World/Rig/Bridge", [bridge[0], bridge[1], z - 0.025], [0.11, span + 0.12, 0.02], coat)
        top = float(head_position[2]) + 0.045
        box("/World/Rig/Mast", [back[0], back[1], 0.5 * (z + top)], [0.035, 0.035, top - z], coat)
        reach = float(np.dot(np.asarray(head_position[:2]) - back[:2], forward[:2]))
        boom = back + 0.5 * reach * forward
        box("/World/Rig/Boom", [boom[0], boom[1], top - 0.0125], [reach, 0.03, 0.025], coat)
        body = np.asarray(head_position, dtype=float) - 0.035 * forward
        box("/World/Rig/HeadCamera", body.tolist(), [0.05, 0.10, 0.035], dark)

    def _fixture(self, fixture):
        path = f"/World/Fixtures/{fixture.name}"
        if fixture.material == "drape":
            material = self.appearance.material("drape", color=[0.12, 0.36, 0.55], roughness=0.85)
        elif fixture.material == "polymer":
            material = "sterile_blue"  # sterile wrap under instruments, supplies and samples
        else:
            material = fixture.material
        if fixture.collision:
            body = self.appearance.cube(
                path, fixture.position, fixture.size, material, collision=True
            )
            self._collision_settings(body.prim)
            if not fixture.visible:
                self._invisible(body.prim)
            self.fixture_bodies[fixture.name] = body
        else:
            self.appearance.rounded(path, fixture.position, fixture.size, material, 0.0005)
        if fixture.label and self.config["visuals"]["labels"]:
            size = fixture.size
            self.appearance.label(
                f"/World/Labels/fixture_{fixture.name}",
                fixture.label,
                [
                    fixture.position[0],
                    fixture.position[1] - size[1] / 2 - 0.0005,
                    fixture.position[2] + 0.001,
                ],
                [min(0.8 * size[0], 0.12), min(0.04, max(size[2] * 2, 0.02))],
            )

    def _beacon(self, indicator):
        """A flashing emergency lamp on a short post: a lit dome whose color the task changes, plus a real light."""
        from isaacsim.core.api.materials import OmniPBR
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        center = indicator.position
        material = OmniPBR(f"/World/Materials/{indicator.name}", color=np.array([0.25, 0.03, 0.03]))
        shader = material.shaders_list[0]
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.25, 0.03, 0.03)
        )
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(400.0)
        self.appearance.cylinder(
            f"/World/Indicators/{indicator.name}_Base", center, 0.045, 0.030, "stainless"
        )
        self.appearance.cylinder(
            f"/World/Indicators/{indicator.name}_Post",
            center + np.array([0.0, 0.0, 0.025]),
            0.018,
            0.030,
            "rubber",
        )
        self.appearance.sphere(
            f"/World/Indicators/{indicator.name}_Dome",
            center + np.array([0.0, 0.0, 0.065]),
            0.052,
            material,
        )
        glow = UsdLux.SphereLight.Define(self.stage, f"/World/Indicators/{indicator.name}_Glow")
        glow.CreateRadiusAttr(0.05)
        glow.CreateIntensityAttr(self.config["visuals"].get("beacon_intensity", 600.0) * 6)
        glow.CreateColorAttr(Gf.Vec3f(0.25, 0.03, 0.03))
        UsdGeom.Xformable(glow).AddTranslateOp().Set(
            Gf.Vec3d(*[float(v) for v in center + np.array([0.0, 0.0, 0.10])])
        )
        self.indicators[indicator.name] = (material, shader, glow)

    def _indicator(self, indicator, PreviewSurface, VisualCuboid):
        if getattr(indicator, "style", "dial") == "beacon":
            return self._beacon(indicator)
        """Housing, a glowing face panel and a beacon on top, so the cue reads from afar."""
        from isaacsim.core.api.materials import OmniPBR
        from pxr import Gf, Sdf

        width, depth, height = INDICATOR_SIZE
        center = indicator.position + np.array([0.0, 0.0, 0.005])
        housing = self.appearance.cube(
            f"/World/Equipment/{indicator.name}_Housing",
            center,
            INDICATOR_SIZE,
            "cabinet_white",
            collision=True,
        )
        self.fixture_bodies[f"indicator_{indicator.name}"] = housing
        material = OmniPBR(f"/World/Materials/{indicator.name}", color=np.array([0.1, 0.3, 1.0]))
        shader = material.shaders_list[0]
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.1, 0.3, 1.0)
        )
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(120.0)
        from roboclock.envs.hospital.timers import Dial

        dial_center = center + np.array([0.0, 0.0, height / 2 + 0.078])
        self.timer_bars[indicator.name] = Dial(
            self.stage,
            self.appearance,
            indicator.name,
            dial_center,
            hub_material=material,
            scale=1.6,
        )
        from pxr import UsdGeom, UsdLux

        # A real light at the beacon, so the status color spills onto the bench around it.
        glow = UsdLux.SphereLight.Define(self.stage, f"/World/Indicators/{indicator.name}_Glow")
        glow.CreateRadiusAttr(0.02)
        glow.CreateIntensityAttr(self.config["visuals"].get("beacon_intensity", 600.0))
        glow.CreateColorAttr(Gf.Vec3f(0.1, 0.3, 1.0))
        UsdGeom.Xformable(glow).AddTranslateOp().Set(
            Gf.Vec3d(*[float(v) for v in center + np.array([0.0, 0.0, height / 2 + 0.06])])
        )
        self.indicators[indicator.name] = (material, shader, glow)
        if self.config["visuals"]["labels"]:
            label = self.appearance.label(
                f"/World/Labels/indicator_{indicator.name}",
                indicator.label,
                center + np.array([width / 2 + 0.0015, 0.0, -0.030]),
                [0.050, 0.020],
            )
            label.AddRotateZOp().Set(90.0)

    def _target(self, name, target):
        from isaacsim.core.api.objects import FixedCuboid

        visual = self.appearance.material(
            f"Target_{name}",
            color=np.array([0.80, 0.82, 0.84]),
            roughness=0.28,
            metallic=0.85,
        )
        cx, cy = target.position[:2]
        hx, hy = target.half_extents[:2]
        floor_top, rim = target.support_z, target.rim_z
        height = rim - floor_top
        mid = floor_top + height / 2
        parts = [
            ("floor", [cx, cy, floor_top - FLOOR / 2], [2 * hx, 2 * hy, FLOOR]),
            ("left", [cx - hx - WALL / 2, cy, mid], [WALL, 2 * hy + 2 * WALL, height]),
            ("right", [cx + hx + WALL / 2, cy, mid], [WALL, 2 * hy + 2 * WALL, height]),
            ("front", [cx, cy - hy - WALL / 2, mid], [2 * hx, WALL, height]),
            ("back", [cx, cy + hy + WALL / 2, mid], [2 * hx, WALL, height]),
        ]
        root = f"/World/Targets/{name}"
        self.target_parts[name] = []
        for part, position, scale in parts:
            body = self.world.scene.add(
                FixedCuboid(
                    prim_path=f"{root}/{part}",
                    name=f"{name}_{part}",
                    position=np.array(position),
                    scale=np.array(scale),
                    visual_material=visual,
                )
            )
            self._collision_settings(body.prim)
            self._invisible(body.prim)
            self.appearance.rounded(f"{root}/{part}_Visual", position, scale, visual, 0.0008)
            self.target_parts[name].append((body, np.array(position)))
        text = self.layout.labels.get(name)
        if text and self.config["visuals"]["labels"]:
            self.appearance.label(
                f"/World/Labels/{name}",
                text,
                [cx, cy - hy - WALL - 0.0005, floor_top + height * 0.5],
                [min(2 * hx, 0.11), min(height * 0.8, 0.03)],
            )

    @staticmethod
    def _invisible(prim):
        from pxr import UsdGeom

        UsdGeom.Imageable(prim).MakeInvisible()

    def _object(self, obj, material):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        spec = self.catalog[obj.asset]
        path = f"/World/Objects/{obj.name}"
        look = self.appearance
        prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        colored = look.material(f"Item_{obj.name}", color=obj.color, roughness=0.35)
        labels = self.config["visuals"]["labels"]
        if spec["proxy"] == "cylinder":
            radius, height = spec["radius"], spec["height"]
            collider = look.cylinder_child(
                path, "Body", radius, height, 0.0, look.material("polymer"), True
            )
            self._invisible(collider)
            if obj.asset == "pill_vial":
                look.pill_vial_visual(
                    path,
                    radius,
                    height,
                    int(obj.label.split()[-1]),
                    obj.color,
                    seed=int(obj.label.split()[-1]),
                )
            elif obj.asset in VESSELS:
                look.vial(path, radius, height, colored)
            else:
                # Open cup or can: opaque wall with a lighter rim band.
                look.cylinder_child(path, "Wall", radius, height, 0.0, colored)
                look.cylinder_child(
                    path, "Rim", radius + 0.0004, 0.004, height / 2 - 0.002, look.material("paper")
                )
            if labels and obj.label and obj.asset != "pill_vial":
                look.label(
                    path + "/Identity",
                    obj.label,
                    [0, 0, -0.004],
                    [min(2.2 * radius, 0.05), min(height * 0.5, 0.03)],
                    obj.color,
                    cylindrical_radius=radius + 0.0005,
                )
        elif spec["proxy"] == "pouch":
            self._pouch(path, spec)
        else:
            size = np.array(spec["size"])
            shape = UsdGeom.Cube.Define(self.stage, path + "/Body")
            shape.CreateSizeAttr(1.0)
            shape.AddScaleOp().Set(Gf.Vec3f(*size.tolist()))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            self._invisible(shape.GetPrim())
            self._box_visual(obj, size, colored)
            if labels and obj.label:
                look.label(
                    path + "/Identity",
                    obj.label,
                    [0, -size[1] / 2 - 0.0005, 0],
                    [min(0.8 * size[0], 0.05), min(0.8 * size[2], 0.03)],
                    obj.color,
                )
        colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(obj.mass)
        body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        body_api.CreateEnableCCDAttr(True)
        body_api.CreateSolverPositionIterationCountAttr(
            self.config["physics"]["solver_position_iterations"]
        )
        body_api.CreateSolverVelocityIterationCountAttr(
            self.config["physics"]["solver_velocity_iterations"]
        )
        for collider in colliders:
            self._collision_settings(collider)
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                material.material, materialPurpose="physics"
            )
        body = SingleRigidPrim(
            prim_path=path,
            name=obj.name,
            position=obj.position,
            orientation=obj.orientation,
            mass=obj.mass,
            scale=np.array(spec["scale"]),
        )
        return self.world.scene.add(body)

    def _pouch(self, path, spec):
        """Body plus a four-bar eyelet as separate convex colliders on one rigid body."""
        from pxr import Gf, UsdGeom, UsdPhysics

        bx, by, bz = spec["size"]
        eye = spec["eyelet"]
        width, height, bar, thick = eye["width"], eye["height"], eye["bar"], eye["thickness"]
        parts = (
            ("Body", [0, 0, 0], [bx, by, bz]),
            ("EyeL", [-(width / 2 - bar / 2), 0, bz / 2 + height / 2], [bar, thick, height]),
            ("EyeR", [width / 2 - bar / 2, 0, bz / 2 + height / 2], [bar, thick, height]),
            ("EyeTop", [0, 0, bz / 2 + height - bar / 2], [width, thick, bar]),
        )
        for name, offset, size in parts:
            shape = UsdGeom.Cube.Define(self.stage, f"{path}/{name}")
            shape.CreateSizeAttr(1.0)
            shape.AddTranslateOp().Set(Gf.Vec3d(*offset))
            shape.AddScaleOp().Set(Gf.Vec3f(*size))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            self._invisible(shape.GetPrim())
        self.appearance.blood_bag_visual(path, spec["size"], eye)

    def _box_visual(self, obj, size, colored):
        """Recognizable visuals inside the box collision envelope (meters, rigid-root frame)."""
        path, look = f"/World/Objects/{obj.name}", self.appearance
        steel, paper = look.material("steel"), look.material("paper")
        sx, sy, sz = size
        if obj.asset in (
            "scalpel",
            "forceps",
            "scissors",
            "clamp",
            "retractor",
        ) and look.model_visual(path, f"tool_{obj.asset}", size):
            pass  # a real instrument model sits on the collision box
        elif obj.asset == "syringe" and look.model_visual(path, "syringe_b", size):
            pass
        elif obj.asset == "instrument" and look.model_visual(
            path, ("tool_scissors", "tool_clamp", "tool_forceps")[sum(map(ord, obj.name)) % 3], size
        ):
            pass
        elif obj.asset in ("scalpel", "forceps", "scissors", "clamp", "retractor"):
            look.surgical_tool(path, obj.asset, size)
        elif obj.asset in ("instrument", "syringe"):
            # Shaft along x with a wider head at +x (spoon/tool) or a flange and plunger (syringe).
            look.rounded(
                path + "/Shaft",
                [-sx * 0.1, 0, 0],
                [sx * 0.75, sy * 0.5, sz * 0.5],
                steel if obj.asset == "instrument" else paper,
                0.0015,
            )
            look.rounded(
                path + "/Head",
                [sx * 0.38, 0, 0],
                [sx * 0.24, sy, sz],
                steel if obj.asset == "instrument" else colored,
                0.003,
            )
            look.rounded(
                path + "/Grip", [-sx * 0.42, 0, 0], [sx * 0.16, sy * 0.9, sz * 0.9], colored, 0.002
            )
        elif obj.asset == "aed_pad":
            # Foam backing, a gel face, a colored stripe matching its chest mark, and a connector tab.
            stripe = (1.0, 0.5, 0.1) if obj.label == "1" else (0.2, 0.45, 0.9)
            look.rounded(path + "/Foam", [0, 0, -sz * 0.15], [sx, sy, sz * 0.7], colored, 0.003)
            look.rounded(
                path + "/Gel",
                [0, 0, sz * 0.2],
                [sx * 0.78, sy * 0.80, sz * 0.25],
                look.material("pad_gel", color=(0.62, 0.66, 0.70), roughness=0.35),
                0.002,
            )
            look.rounded(
                path + "/Stripe",
                [-sx * 0.40, 0, sz * 0.36],
                [sx * 0.07, sy * 0.9, 0.0012],
                look.material(f"pad_stripe_{obj.label}", color=stripe, roughness=0.5),
                0.0004,
            )
            look.rounded(
                path + "/Connector",
                [sx * 0.5 + 0.006, 0, -sz * 0.1],
                [0.014, 0.022, sz * 0.5],
                look.material("pad_connector", color=(0.95, 0.80, 0.15), roughness=0.5),
                0.002,
            )
        elif obj.asset == "fire_blanket":
            # A folded glass-fiber blanket: stacked layers with a stitched edge and visible fold lines.
            line = look.material("blanket_fold", color=(0.52, 0.50, 0.44), roughness=0.9)
            layers = 5
            for k in range(layers):
                look.rounded(
                    path + f"/Layer_{k}",
                    [0, 0, -sz / 2 + sz * (k + 0.5) / layers],
                    [
                        sx * (1.0 - 0.012 * (k % 2)),
                        sy * (1.0 + 0.012 * (k % 2)),
                        sz / layers * 0.96,
                    ],
                    colored,
                    0.004,
                )
            look.rounded(
                path + "/Stitch", [0, 0, sz / 2 + 0.0004], [sx * 0.9, 0.004, 0.0012], line, 0.0004
            )
        elif obj.asset in ("aed_tab", "blanket_tab"):
            look.rounded(path + "/Tab", [0, 0, 0], list(size), colored, 0.004)
            grip = look.material(
                "tab_grip" if obj.asset == "aed_tab" else "tab_grip_black",
                color=(0.05, 0.45, 0.20) if obj.asset == "aed_tab" else (0.15, 0.15, 0.16),
                roughness=0.6,
            )
            for k in range(3):
                look.rounded(
                    path + f"/Rib_{k}",
                    [(k - 1) * sx * 0.25, 0, sz / 2],
                    [0.004, sy * 0.85, 0.002],
                    grip,
                    0.0008,
                )
        elif obj.asset == "blood_bag":
            look.rounded(path + "/Pouch", [0, 0, 0], list(size), colored, 0.008)
            look.rounded(
                path + "/Port", [sx * 0.5 + 0.004, 0, 0], [0.008, 0.012, 0.010], paper, 0.002
            )
            look.rounded(
                path + "/Seam", [0, 0, sz / 2], [sx * 0.9, sy * 0.9, 0.0008], paper, 0.0004
            )
        else:
            look.rounded(path + "/Shell", [0, 0, 0], list(size), colored, 0.002)
            look.rounded(
                path + "/Band", [0, 0, 0], [sx + 0.0006, sy + 0.0006, sz * 0.22], paper, 0.0006
            )
        if obj.asset == "syringe":
            # The dose: a column of liquid inside the barrel, hidden until the syringe has filled on the dock.
            liquid = self.appearance.cylinder(
                path + "/Liquid",
                [-sx * 0.12, 0, 0],
                min(sy, sz) * 0.30,
                sx * 0.45,
                look.material(
                    "dose_liquid", color=(0.95, 0.65, 0.10), roughness=0.2, emissive=True
                ),
                axis="X",
            )
            if (
                "dose_prep" in self.layout.info
            ):  # filled on the dock first; a prepared dose shows it already
                self._invisible(liquid.GetPrim())
            self.zone_prims[f"liquid_{obj.name.split('_')[-1]}"] = liquid.GetPrim()

    # --- runtime commands ----------------------------------------------------------------

    def command(self, command):
        if command.kind == "indicator":
            entry = self.indicators.get(command.name)
            if entry is not None:
                from pxr import Gf

                color = Gf.Vec3f(*[float(v) for v in command.value])
                shader = entry[1]
                shader.GetInput("diffuse_color_constant").Set(color)
                shader.GetInput("emissive_color").Set(color)
                entry[2].GetColorAttr().Set(color)
                dial = self.timer_bars.get(command.name)
                if dial is not None:
                    dial.state(tuple(float(v) for v in command.value))
        elif command.kind == "timer":
            bar = self.timer_bars.get(command.name)
            if bar is not None:
                bar.set(*command.value)
        elif command.kind == "release":
            self.supports[command.name].CreateJointEnabledAttr(False)
            self.objects[command.name].set_linear_velocity(np.array(command.value))
            self._release_speeds[command.name] = float(
                np.linalg.norm(self.objects[command.name].get_linear_velocity()[:2])
            )
        elif command.kind == "spin":
            self.objects[command.name].set_angular_velocity(np.array(command.value))
        elif command.kind == "spill":
            from roboclock.envs.hospital import spill_scene

            spill_scene.command(self, command)
        elif command.kind == "show":
            from pxr import UsdGeom

            prim = self.zone_prims[command.name]
            if command.value:
                UsdGeom.Imageable(prim).MakeVisible()
            else:
                UsdGeom.Imageable(prim).MakeInvisible()
        else:
            raise ValueError(f"Unknown scene command: {command.kind}")

    def _prepare_bedside_posture(self):
        """Initialize the prescribed ready pose before the episode clock starts."""
        from isaacsim.core.utils.rotations import euler_angles_to_quat

        from roboclock.envs.lab.robot import DOF, JOINT_HIGH, JOINT_LOW, make_solver
        from roboclock.envs.start_pose import zero_start

        info = self.layout.info
        if "initial_ee_position" not in info:
            return
        if zero_start(self.config):
            # robot.start_pose: zero. The arm stays at joint 0 (lab.robot.home_joints); only
            # the task's start gripper width applies.
            q = self.robot.get_joint_positions().copy()
            q[6:] = info.get("initial_gripper_width", 0.04)
            self.robot.set_joint_positions(q)
            self.robot.set_joint_velocities(np.zeros(DOF))
            self.initial_hold_joints = np.concatenate([q, self.secondary_robot.get_joint_positions()])
            self.apply_action(self.initial_hold_joints)
            return
        solver = make_solver(self.robot)
        action, solved = solver.compute_inverse_kinematics(
            target_position=np.asarray(info["initial_ee_position"]),
            target_orientation=euler_angles_to_quat(
                np.array([0.0, np.pi, info.get("initial_ee_yaw", 0.0)])
            ),
        )
        if not solved:
            raise RuntimeError(f"{self.task.task_id}: prescribed bedside ready pose is unreachable")
        q = self.robot.get_joint_positions().copy()
        indices = (
            action.joint_indices
            if action.joint_indices is not None
            else range(len(action.joint_positions))
        )
        for i, value in zip(indices, action.joint_positions):
            if value is not None:
                q[i] = value
        q[6:] = info.get("initial_gripper_width", 0.04)
        if np.any(q < JOINT_LOW - 0.001) or np.any(q > JOINT_HIGH + 0.001):
            raise RuntimeError(f"{self.task.task_id}: ready pose exceeds robot joint limits")
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros(DOF))
        self.initial_hold_joints = np.concatenate([q, self.secondary_robot.get_joint_positions()])
        self.apply_action(self.initial_hold_joints)

    def _measure_state(self, elapsed):
        from dataclasses import replace

        from roboclock.envs.hospital.bedside import module_for

        state = super()._measure_state(elapsed)
        bedside = module_for(self.task)
        if bedside is None:
            return state
        return replace(state, controls={**state.controls, **bedside.measure(self, state)})
