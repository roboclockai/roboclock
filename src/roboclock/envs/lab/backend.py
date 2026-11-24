"""Isaac Sim 5.1 scene, PhysX dynamics, camera, and measured manipulation state.

Imported only after SimulationApp starts. Actions command the robot articulation.
C1's optional expert demonstration adds an explicitly recorded physical grasp
constraint after contact; ordinary benchmark control has no grasp attachment.
"""

from pathlib import Path

import numpy as np
import yaml

from roboclock.envs.lab.appearance import LabAppearance
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.physics.contacts import GraspTracker
from roboclock.resources import resolve_asset, resolve_asset_reference


class IsaacBackend:
    environment_id = "lab"
    layout_factory = staticmethod(make_layout)
    appearance_factory = LabAppearance

    def lab_task(self, *ids):
        """Keep laboratory fixture behavior scoped to its domain."""
        return self.environment_id == "lab" and self.task.task_id in ids

    def build_domain_fixtures(self):
        """Optional domain fixtures, built before contact views are initialized."""

    def indicator_position(self, task):
        from roboclock.envs.lab.layout import indicator_position

        return indicator_position(task)

    def __init__(self, runtime, config, asset_catalog=None):
        self.runtime = runtime
        self.task = None
        self.config = config
        catalog_path = (
            resolve_asset("env://catalog.yaml", environment=self.environment_id)
            if asset_catalog is None
            else asset_catalog
        )
        catalog_path = Path(catalog_path).resolve()
        with open(catalog_path, encoding="utf-8") as stream:
            self.catalog = yaml.safe_load(stream)
        for spec in self.catalog.values():
            if spec["usd_path"]:
                spec["usd_path"] = resolve_asset_reference(
                    spec["usd_path"], environment=self.environment_id, base_dir=catalog_path.parent
                )
        self.world = None
        self.camera = None
        self.rig_cameras = None
        self.secondary_robot = None
        self.inter_arm_contacts = None
        self.objects = {}
        self.target_parts = {}
        self.supports = {}
        self.clutter = {}
        self.clutter_specs = []
        self.last_collision = False
        self.collision_count = 0
        self._release_speeds = {}

    @property
    def robodojo_cameras(self):
        """Explicit opt-in (`sim.robodojo_cameras: true`, set by the lab config) to the
        RoboDojo head/wrist policy cameras (rig_cameras.py), 25 Hz rendering and the
        measured rig check. Configs without the key (household) keep the legacy single
        observation camera; a subclass may also pin a class attribute (hospital: False).
        """
        return bool(self.config["sim"].get("robodojo_cameras", False))

    @property
    def no_render(self):
        return bool(self.config["sim"].get("no_render", False))

    def overview_enabled(self):
        """The third-person camera: the observation camera of legacy (non-RoboDojo)
        backends, otherwise an overview that exists only for recordings."""
        if self.no_render:
            return False
        return not self.robodojo_cameras or bool(self.config["sim"].get("overview_camera", False))

    def create_overview_camera(self, position, look_at):
        """Third-person camera (`<task>.mp4`, or legacy `rgb`), before World.reset; None if disabled."""
        if not self.overview_enabled():
            return None, None
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from isaacsim.sensors.camera import Camera
        from pxr import Gf

        # USD cameras look along local -Z; SetLookAt gives the world-to-camera
        # matrix. Inverting it avoids the transposed lookat_to_quatf convention.
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*np.asarray(position, dtype=float).tolist()),
            Gf.Vec3d(*np.asarray(look_at, dtype=float).tolist()),
            Gf.Vec3d(0, 0, 1),
        )
        orientation = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
        resolution = (
            self.config["sim"].get("overview_resolution", [1280, 960])
            if self.robodojo_cameras
            else self.config["sim"]["camera_resolution"]
        )
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.asarray(position, dtype=float),
            orientation=orientation,
            resolution=tuple(resolution),
        )
        return np.asarray(position, dtype=float), orientation

    def create_rig_cameras(self):
        """RoboDojo policy cameras for opted-in backends, before World.reset."""
        self.rig_cameras = None
        if self.robodojo_cameras:
            from roboclock.envs.lab.rig_cameras import RigCameras

            self.rig_cameras = RigCameras(self)

    def initialize_overview_camera(self, position, orientation, focal_length_mm, clipping_far):
        if self.camera is not None:
            self.camera.initialize()
            if not self.robodojo_cameras:
                self.camera.add_distance_to_image_plane_to_frame()  # legacy `depth`
            self.camera.set_world_pose(position, orientation, camera_axes="usd")
            self.camera.set_focal_length(focal_length_mm / 1000)
            self.camera.set_horizontal_aperture(
                self.config["scene"]["camera_horizontal_aperture"] / 1000
            )
            self.camera.set_clipping_range(0.01, clipping_far)
        if self.rig_cameras is not None:
            self.rig_cameras.initialize()

    def finish_camera_setup(self):
        """After settling: RoboDojo renderer read-back and the measured rig check, or the
        legacy camera warm-up."""
        if self.rig_cameras is None:
            if self.camera is not None:
                for _ in range(5):
                    self.world.render()
                    rgba = self.camera.get_rgba()
                    if rgba is not None and rgba.size:
                        break
            return
        self.rig_cameras.warm_up()
        if self.camera is not None:
            for _ in range(5):
                rgba = self.camera.get_rgba()
                if rgba is not None and rgba.size and np.any(rgba[..., :3]):
                    break
                self.capture_camera(1)
        self.rig_cameras.measure()

    def reset(self, task):
        if getattr(task, "liquid_process", False):
            from roboclock.envs.lab.liquid_scene import reset_liquid

            return reset_liquid(self, task)
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        self.close()
        World.clear_instance()
        create_new_stage()
        self.objects, self.target_parts, self.supports = {}, {}, {}
        self.layout = self.layout_factory(task, self.config, self.catalog)
        task.bind(self.layout, self.config["sim"]["placement_dwell"])
        self.task = task
        task.recovery_dwell = self.config["physics"]["recovery_dwell"]
        task.catch_dwell = self.config["physics"]["catch_dwell"]
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config["sim"]["physics_dt"],
            rendering_dt=self.config["sim"]["physics_dt"],
        )
        self.world.get_physics_context().enable_ccd(True)
        self.world.scene.add_default_ground_plane()
        self.stage = get_current_stage()
        from roboclock.simulators.isaac.rendering import configure_renderer

        hotplate = getattr(task, "laboratory_v2", False)
        throughput_room = self.lab_task("C1", "C2", "C3") and self.config["scene"].get(
            "throughput_background_v2", True
        )
        equipment = (
            getattr(task, "power_task", False)
            or getattr(task, "tube_task", False)
            or getattr(task, "safety_task", False)
            or getattr(task, "bath_task", False)
        )
        appearance_config = self.config
        if hotplate or throughput_room:
            from roboclock.envs.lab.liquid_scene import ASSET_ROOT

            appearance_config = dict(self.config)
            appearance_config["visuals"] = dict(
                self.config["visuals"], **self.config.get("liquid_visuals", {})
            )
            appearance_config["visuals"]["hdr_path"] = str(
                ASSET_ROOT / "mountain/alps_field_2k.hdr"
            )
        configure_renderer(appearance_config)
        scene = self.config["scene"]
        self.appearance = self.appearance_factory(self.stage, self.world, appearance_config)
        if hotplate or throughput_room:
            glass = self.appearance.glass()
            glass.shaders_list[0].GetInput("depth").Set(0.0015)
            glass.shaders_list[0].GetInput("glass_color").Set(Gf.Vec3f(0.99, 0.995, 1.0))
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
        self.kinematic_targets = {}
        self.fixture_offsets = {}
        table_physics = None
        if self.lab_task("B2") and not equipment:
            from pxr import PhysxSchema

            table_physics = PhysicsMaterial(
                "/World/Materials/SlipSurface",
                static_friction=0.025,
                dynamic_friction=0.025,
                restitution=0.0,
            )
            PhysxSchema.PhysxMaterialAPI.Apply(table_physics.prim).CreateFrictionCombineModeAttr(
                "min"
            )
        table = self.world.scene.add(
            FixedCuboid(
                prim_path="/World/Table",
                name="table",
                position=np.array([0.45, 0, 0.72] if hotplate else scene["table_center"]),
                scale=np.array([1.3, 1.3, 0.06] if hotplate else scene["table_size"]),
                physics_material=table_physics,
            )
        )
        UsdGeom.Imageable(table.prim).MakeInvisible()
        self.appearance.rounded(
            "/World/TableVisual",
            [0.45, 0, 0.72] if hotplate else scene["table_center"],
            [1.3, 1.3, 0.06] if hotplate else scene["table_size"],
            "worktop",
            0.006,
        )
        from roboclock.envs.lab.robot import create_robots

        create_robots(self)
        self.appearance.lights(self.layout)
        from roboclock.envs.lab.clutter import build_clutter, sample_clutter

        if hotplate:
            from roboclock.envs.lab.liquid_scene import LiquidScene

            self.hotplate_setting = LiquidScene(self, setting_only=True)
        else:
            self.appearance.workbench(self.layout)
            if throughput_room:
                from pxr import Usd

                from roboclock.envs.lab.liquid_scene import LiquidScene

                # The old room contains decorations only. Hide its meshes while
                # retaining all foreground geometry, rigid bodies and contacts.
                for prim in Usd.PrimRange(self.stage.GetPrimAtPath("/World/Room")):
                    if prim.IsA(UsdGeom.Gprim):
                        UsdGeom.Imageable(prim).MakeInvisible()
                self.throughput_setting = LiquidScene(self, background_only=True)
            self.clutter_specs = sample_clutter(task, self.layout, self.config)
            self.clutter = build_clutter(self) if self.clutter_specs else {}
        self.clutter_robot_contact_count = 0
        self._last_clutter_contact = False
        self.indicator_material = PreviewSurface(
            "/World/Materials/Indicator", color=np.array([0.1, 0.3, 1.0]), roughness=0.20
        )
        # Safety tasks expose real equipment instruments, not a generic STATE marker.
        if not getattr(task, "safety_task", False):
            indicator = self.indicator_position(task)
            self.appearance.cube(
                "/World/Equipment/IndicatorHousing",
                indicator,
                [0.065, 0.040, 0.090],
                "ceramic",
                collision=True,
            )
            self.world.scene.add(
                VisualCuboid(
                    prim_path="/World/Indicator",
                    name="indicator",
                    position=indicator + np.array([0.0, 0.021, 0.017]),
                    scale=np.array([0.043, 0.002, 0.023]),
                    visual_material=self.indicator_material,
                )
            )
            if self.config["visuals"]["labels"]:
                self.appearance.label(
                    "/World/Equipment/IndicatorLabel",
                    "STATE",
                    indicator + np.array([0.0, 0.021, -0.020]),
                    [0.053, 0.020],
                )
        if self.lab_task("A1"):
            pos = self.layout.heating_zone.position.copy()
            pos[2] = self.layout.table_height + 0.001
            self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/HeatingZone",
                    name="heater",
                    position=pos,
                    scale=np.array([0.11, 0.11, 0.002]),
                    visual_material=self.appearance.material("ceramic"),
                )
            )
        if getattr(task, "hotplate_task", False):
            from roboclock.envs.lab.hotplate_scene import HotplateScene

            self.hotplate_scene = HotplateScene(self)
        if getattr(task, "rack_organization", False):
            from roboclock.envs.lab.rack_organization import build_racks, configure_recording_logging

            build_racks(self)
            # Contact queries on the racks' triangle meshes make PhysX warn
            # "getMaterialFromInternalFaceIndex received 0xFFFFffff" ~1e6 times per
            # C1 episode (~300 MB of log). Log level only: physics is unchanged.
            configure_recording_logging()
        for name, target in self.layout.targets.items():
            if not equipment:
                self._target(name, target)
        if self.lab_task("A4"):
            marker = self.layout.targets["destination"].position.copy()
            marker[1] += 0.08
            marker[2] = self.layout.table_height + 0.001
            self.world.scene.add(
                __import__("isaacsim.core.api.objects", fromlist=["VisualCuboid"]).VisualCuboid(
                    prim_path="/World/AlignmentReference",
                    name="alignment_reference",
                    position=marker,
                    scale=np.array([0.005, 0.035, 0.002]),
                    color=np.array([0.1, 1.0, 0.2]),
                )
            )
        if self.lab_task("B2", "B3") and not equipment:
            holder = self.layout.objects["sample_0"].position.copy()
            holder[1] += 0.12
            self.world.scene.add(
                FixedCuboid(
                    prim_path="/World/Holder",
                    name="sample_holder",
                    position=holder,
                    scale=np.array([0.07, 0.012, 0.09]),
                    color=np.array([0.35, 0.38, 0.42]),
                )
            )
        if self.lab_task("B3") and not equipment:
            source = self.layout.objects["sample_0"].position
            height = source[2] - self.layout.table_height
            self.appearance.cube(
                "/World/Equipment/LauncherColumn",
                [source[0], source[1] + 0.12, self.layout.table_height + height / 2],
                [0.025, 0.025, height],
                "steel",
            )
            self.appearance.cube(
                "/World/Equipment/LauncherFoot",
                [source[0], source[1] + 0.12, self.layout.table_height + 0.004],
                [0.08, 0.08, 0.008],
                "rubber",
            )
        for name, obj in self.layout.objects.items():
            material = PhysicsMaterial(
                f"/World/Materials/{name}",
                static_friction=obj.friction,
                dynamic_friction=obj.friction,
                restitution=0.0,
            )
            self.objects[name] = self._object(obj, material)
            if (self.lab_task("B2", "B3") and not equipment) or (
                getattr(task, "continuous_arrivals", False) and name not in task.active
            ):
                self._support(name, obj)
                if getattr(task, "continuous_arrivals", False):
                    self.objects[name].set_visibility(False)
                    self._set_collision(name, False)
        if getattr(task, "safety_task", False):
            from roboclock.envs.lab.safety_scene import SafetyEquipmentScene

            self.reactive_scene = SafetyEquipmentScene(self)
        elif getattr(task, "bath_task", False):
            from roboclock.envs.lab.bath_scene import BathScene

            self.reactive_scene = BathScene(self)
        elif equipment:
            from roboclock.envs.lab.reactive_scene import ReactiveEquipmentScene

            self.reactive_scene = ReactiveEquipmentScene(self)
        self.build_domain_fixtures()
        self.surface_paths = {
            name: self.kinematic_targets[name].prim_path
            if name in self.kinematic_targets
            else parts[0][0].prim_path
            for name, parts in self.target_parts.items()
        }
        for name, parts in self.target_parts.items():
            if name not in self.kinematic_targets:
                for body, _ in parts[1:]:
                    self.surface_paths[f"{name}:{body.prim_path.rsplit('/', 1)[-1]}"] = (
                        body.prim_path
                    )
        if self.lab_task("B2", "B3") and not equipment:
            self.surface_paths["holder"] = "/World/Holder"
        self.surface_paths["ground"] = "/World/defaultGroundPlane"
        self.surface_paths.update({name: body.prim_path for name, body in self.clutter.items()})
        if self.lab_task("A1"):
            self.surface_paths["heater"] = "/World/HeatingZone"
        if getattr(task, "hotplate_task", False):
            self.surface_paths.update(self.hotplate_scene.surfaces)
        if equipment:
            self.surface_paths.update(self.reactive_scene.surfaces)
        accessory_paths = (
            [
                str(p.GetPath())
                for p in self.stage.Traverse()
                if str(p.GetPath()).startswith(
                    "/World/Accessories/" if hotplate else "/World/TubeRacks/"
                )
                and p.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if hotplate or getattr(task, "rack_organization", False)
            else []
        )
        # Pairwise contact forces distinguish bilateral grasp from proximity.
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
                contact_filter_prim_paths_expr=[
                    ["/World/Table"]
                    + [body.prim_path for parts in self.target_parts.values() for body, _ in parts]
                    + (
                        list(self.hotplate_scene.surfaces.values())
                        if getattr(task, "hotplate_task", False)
                        else []
                    )
                    + (list(self.reactive_scene.surfaces.values()) if equipment else [])
                    + accessory_paths
                    + [
                        path
                        for path in ("/World/ARXMount0", "/World/ARXMount1")
                        if self.stage.GetPrimAtPath(path).IsValid()
                    ]
                    + [body.prim_path for body in self.clutter.values()]
                    for _ in range(6)
                ],
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
        if self.secondary_robot is not None:
            from roboclock.envs.lab.dual_arm import build_collision_monitor

            self.secondary_contacts = build_collision_monitor(
                self,
                accessory_paths
                + [body.prim_path for parts in self.target_parts.values() for body, _ in parts]
                + (
                    list(self.hotplate_scene.surfaces.values())
                    if getattr(task, "hotplate_task", False)
                    else []
                ),
            )
        camera_position = np.array(scene["camera_position"]) + self.layout.camera_jitter
        camera_look_at = scene["camera_look_at"]
        if hotplate:
            camera_position = np.array(scene.get("liquid_camera_position", [1.45, 2.3, 2.05]))
            camera_look_at = scene.get("liquid_camera_look_at", [0.25, -0.45, 1.15])
        if throughput_room:
            camera_position = (
                np.array(scene.get("throughput_camera_position", [1.45, 2.3, 2.05]))
                + self.layout.camera_jitter
            )
            camera_look_at = scene.get("throughput_camera_look_at", [0.25, -0.45, 1.15])
        if getattr(task, "safety_task", False) and task.task_id == "B7":
            camera_position = np.array([2.20, 4.40, 2.50])
            camera_look_at = [0.48, 0.20, 0.65]
        self.camera = None
        camera_position, orientation = self.create_overview_camera(camera_position, camera_look_at)
        self.create_rig_cameras()
        self.world.reset()
        if self.kinematic_targets:
            import omni.physics.tensors as tensors

            self.kinematic_sim = tensors.create_simulation_view("numpy")
            self.kinematic_sim.set_subspace_roots("/")
            self.kinematic_views = {
                name: self.kinematic_sim.create_rigid_body_view(body.prim_path)
                for name, body in self.kinematic_targets.items()
            }
        from roboclock.envs.lab.robot import initialize_robots

        initialize_robots(self)
        if getattr(task, "safety_task", False):
            self.contacts.disable_gravities()
        self.initialize_overview_camera(
            camera_position,
            orientation,
            (
                16.0
                if getattr(task, "safety_task", False) and task.task_id == "B7"
                else scene.get("liquid_camera_focal_length", 29.0)
            )
            if hotplate
            else scene.get("throughput_camera_focal_length", 29.0)
            if throughput_room
            else scene["camera_focal_length"],
            30.0 if hotplate or throughput_room else 10.0,
        )
        settle_seconds = self.config["sim"]["settle_seconds"]
        if getattr(task, "safety_task", False):
            settle_seconds = max(settle_seconds, 1.5)
        for _ in range(round(settle_seconds / self.config["sim"]["physics_dt"])):
            self.world.step(render=False)
        self.finish_camera_setup()
        # Setup/settling never advances the benchmark's episode clock.
        self.last_collision = False
        self.collision_count = 0
        self._cached_state = self._measure_state(0.0)
        task.previous = self.state()
        self.home_orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
        return self.observation()

    def _collision_settings(self, prim):
        from pxr import PhysxSchema

        config = self.config["physics"]
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(config["contact_offset"])
        api.CreateRestOffsetAttr(config["rest_offset"])

    def _object(self, obj, material):
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        spec = self.catalog[obj.asset]
        path = f"/World/Objects/{obj.name}"
        appearance = self.appearance
        if spec["usd_path"]:
            prim = add_reference_to_stage(spec["usd_path"], path)
            if not prim or not prim.IsValid() or not prim.GetChildren():
                raise RuntimeError(f"USD did not load at {path}: {spec['usd_path']}")
            colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
            nested_bodies = [
                p for p in Usd.PrimRange(prim) if p != prim and p.HasAPI(UsdPhysics.RigidBodyAPI)
            ]
            if not colliders or nested_bodies:
                raise ValueError(
                    "Sample USD must have authored collision shapes under a single rigid root; materials and collision approximations are preserved"
                )
        else:
            prim = UsdGeom.Xform.Define(self.stage, path).GetPrim()
            colored = appearance.material(f"Sample_{obj.name}", color=obj.color, roughness=0.32)
            radius, height = spec["radius"], spec["height"]
            if getattr(self.task, "bath_task", False):
                from roboclock.envs.lab.bath_scene import sample_visual

                sample_visual(appearance, path, self.task.task_id)
            elif obj.asset == "safety_actuator":
                from roboclock.envs.lab.safety_scene import actuator_geometry

                actuator_geometry(appearance, path, self.task.task_id)
            elif obj.asset == "knife_switch":
                from roboclock.envs.lab.knife_switch_scene import handle_geometry

                handle_geometry(appearance, path)
            elif getattr(self.task, "rack_organization", False):
                from roboclock.envs.lab.rack_organization import tube_visual

                tube_visual(appearance, path, obj)
            elif obj.asset == "test_tube":
                from roboclock.envs.lab.reactive_scene import test_tube_visual

                test_tube_visual(appearance, path, radius, height)
            elif obj.asset == "beaker":
                from roboclock.envs.lab.hotplate_scene import beaker_visual

                collider = appearance.cylinder_child(
                    path, "Body", radius, height, 0.0, appearance.material("polymer"), True
                )
                UsdGeom.Imageable(collider).MakeInvisible()
                beaker_visual(appearance, path, radius, height)
            elif obj.asset in ("container", "bottle"):
                # The verified cylindrical collision envelope remains conservative;
                # a separate hollow glass mesh supplies shoulders and a real wall.
                collider = appearance.cylinder_child(
                    path, "Body", radius, height, 0.0, appearance.material("polymer"), True
                )
                UsdGeom.Imageable(collider).MakeInvisible()
                appearance.vial(path, radius, height, colored)
                if self.config["visuals"]["labels"]:
                    appearance.label(
                        path + "/Identity",
                        f"S{int(obj.name.split('_')[-1]) + 1:02d}",
                        [0, 0, -0.008],
                        [0.042, 0.026],
                        obj.color,
                        cylindrical_radius=radius + 0.00035,
                    )
            elif spec["proxy"] == "sphere":
                shape = UsdGeom.Sphere.Define(self.stage, path + "/Body")
                shape.CreateRadiusAttr(radius)
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(colored.material)
            elif spec["proxy"] == "cuboid":
                shape = UsdGeom.Cube.Define(self.stage, path + "/Body")
                shape.CreateSizeAttr(1.0)
                shape.AddScaleOp().Set(Gf.Vec3f(2 * radius, 2 * radius, height))
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(colored.material)
                UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
                appearance.rounded(
                    path + "/Shell", [0, 0, 0], [2 * radius, 2 * radius, height], colored, 0.0015
                )
                # Direction arrow is visual only, in meters under the unscaled rigid root.
                for label, center, size in (
                    ("axis", [0, 0, height / 2 + 0.0002], [0.032, 0.003, 0.0004]),
                    ("axis_end", [0.013, 0, height / 2 + 0.0002], [0.005, 0.008, 0.0004]),
                ):
                    cue = UsdGeom.Cube.Define(self.stage, path + "/" + label)
                    cue.CreateSizeAttr(1.0)
                    cue.AddTranslateOp().Set(Gf.Vec3d(*center))
                    cue.AddScaleOp().Set(Gf.Vec3f(*size))
                    UsdShade.MaterialBindingAPI.Apply(cue.GetPrim()).Bind(
                        appearance.material("paper").material
                    )
                if self.config["visuals"]["labels"]:
                    appearance.label(
                        path + "/Identity",
                        f"S{int(obj.name.split('_')[-1]) + 1:02d}",
                        [0, radius + 0.0002, 0],
                        [0.034, 0.025],
                        obj.color,
                    )
            elif obj.asset == "probe":
                appearance.cylinder_child(
                    path, "Shaft", radius, height, 0, appearance.material("steel"), True
                )
                appearance.cylinder_child(
                    path, "Handle", radius + 0.0002, 0.030, height / 2 - 0.016, colored
                )
            else:
                raise ValueError(f"Unknown proxy: {spec['proxy']}")
            colliders = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI)]
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(obj.mass)
        physics = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        physics.CreateEnableCCDAttr(True)
        physics.CreateSolverPositionIterationCountAttr(
            self.config["physics"]["solver_position_iterations"]
        )
        physics.CreateSolverVelocityIterationCountAttr(
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

    def _target(self, name, target):
        if getattr(self.task, "rack_organization", False):
            from roboclock.envs.lab.rack_organization import build_slot

            return build_slot(self, name, target)
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        from isaacsim.core.prims import SingleXFormPrim
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        samples = [
            obj for key, obj in self.layout.objects.items() if self.layout.assignments[key] == name
        ]
        color = (
            samples[0].color if self.task.category == "throughput" else np.array([0.12, 0.45, 0.35])
        )
        visual = self.appearance.material(
            f"Target_{name}", color=color * 0.55 + 0.2, roughness=0.32
        )
        cx, cy, _ = target.position
        hx, hy, _ = target.half_extents
        z = self.layout.table_height
        height = target.rim_z - z
        parts = [("floor", [cx, cy, z + 0.002], [2 * hx, 2 * hy, 0.004])]
        if target.kind != "marker":
            parts += [
                ("left", [cx - hx - 0.003, cy, z + height / 2], [0.006, 2 * hy + 0.012, height]),
                ("right", [cx + hx + 0.003, cy, z + height / 2], [0.006, 2 * hy + 0.012, height]),
                ("front", [cx, cy - hy - 0.003, z + height / 2], [2 * hx, 0.006, height]),
                ("back", [cx, cy + hy + 0.003, z + height / 2], [2 * hx, 0.006, height]),
            ]
        root = f"/World/Targets/{name}"
        moving = self.lab_task("A4")
        if moving:
            prim = UsdGeom.Xform.Define(self.stage, root).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(True)
        self.target_parts[name] = []
        for part, position, scale in parts:
            if moving:
                shape = UsdGeom.Cube.Define(self.stage, root + "/" + part)
                shape.CreateSizeAttr(1.0)
                shape.AddTranslateOp().Set(
                    Gf.Vec3d(*[float(v) for v in np.array(position) - [cx, cy, z]])
                )
                shape.AddScaleOp().Set(Gf.Vec3f(*scale))
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(visual.material)
                self._collision_settings(shape.GetPrim())
                UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
                self.appearance.rounded(
                    root + "/" + part + "_Visual",
                    np.array(position) - [cx, cy, z],
                    scale,
                    visual,
                    0.0008,
                )
            else:
                body = self.world.scene.add(
                    FixedCuboid(
                        prim_path=root + "/" + part,
                        name=name + "_" + part,
                        position=np.array(position),
                        scale=np.array(scale),
                        visual_material=visual,
                    )
                )
                self._collision_settings(body.prim)
                UsdGeom.Imageable(body.prim).MakeInvisible()
                self.appearance.rounded(
                    root + "/" + part + "_Visual", position, scale, visual, 0.0008
                )
                self.target_parts[name].append((body, np.array(position)))
        if moving:
            offset = self.task.amplitude * np.sin(self.task.spec.phase)
            self.task.fixture_offset = offset
            body = self.world.scene.add(
                SingleXFormPrim(
                    prim_path=root,
                    name=name + "_kinematic",
                    position=np.array([cx + offset, cy, z]),
                )
            )
            self.kinematic_targets[name] = body
            self.fixture_offsets[name] = offset
            self.target_parts[name] = [(body, np.array([cx, cy, z]))]
        if target.kind == "marker":
            for part, pos, size in (
                ("axis", [cx, cy, z + 0.0043], [0.035, 0.003, 0.0004]),
                ("axis_end", [cx + 0.015, cy, z + 0.0043], [0.005, 0.008, 0.0004]),
            ):
                self.world.scene.add(
                    VisualCuboid(
                        prim_path=root + "/" + part,
                        name=name + "_" + part,
                        position=np.array(pos),
                        scale=np.array(size),
                        visual_material=self.appearance.material("paper"),
                    )
                )
        if self.config["visuals"]["labels"]:
            # Nameplates are visual meshes with no collider. Markers use a flat
            # identity sticker beside the arrow; moving labels follow the rigid root.
            text = (
                f"S{int(name.split('_')[-1]) + 1:02d}"
                if name.startswith("target_")
                else "SAFE"
                if getattr(self.task, "hotplate_task", False)
                else name.replace("destination", "RACK").replace("bin_", "BIN ")
            )
            position = [cx, cy + hy + 0.0065, z + height * 0.6]
            size = [min(2 * hx, 0.10), min(height * 0.55, 0.027)]
            path = f"/World/Labels/{name}"
            if moving:
                path = root + "/Label"
                position = [0.0, hy + 0.0065, height * 0.6]
            elif target.kind == "marker":
                position = [cx, cy - 0.012, target.support_z + 0.0005]
                size = [0.022, 0.009]
            label = self.appearance.label(
                path,
                text,
                position,
                size,
                color,
            )
            if target.kind == "marker":
                label.AddRotateXOp().Set(-90.0)

    def _support(self, name, obj):
        from pxr import Gf, Sdf, UsdPhysics

        joint = UsdPhysics.FixedJoint.Define(self.stage, f"/World/Supports/{name}")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/Objects/{name}")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(x) for x in obj.position]))
        joint.CreateLocalRot0Attr(
            Gf.Quatf(float(obj.orientation[0]), Gf.Vec3f(*[float(x) for x in obj.orientation[1:]]))
        )
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        self.supports[name] = joint

    def _set_collision(self, name, enabled):
        from pxr import Usd, UsdPhysics

        for prim in Usd.PrimRange(self.objects[name].prim):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(enabled)

    def command(self, command):
        if command.kind == "liquid_visual":
            self.liquid_scene.update(command.value)
        elif command.kind == "hotplate_visual":
            self.hotplate_scene.update(command.value)
        elif command.kind == "bath_visual":
            self.reactive_scene.update(command.value)
        elif command.kind == "safety_visual":
            self.reactive_scene.update(command.value)
        elif command.kind == "safety_fault_release":
            self.reactive_scene.release_fault()
        elif command.kind == "equipment_visual":
            self.reactive_scene.update(command.value)
        elif command.kind == "disturb_tube":
            self.reactive_scene.disturb_tube()
        elif command.kind == "indicator":
            self.indicator_material.set_color(np.array(command.value))
        elif command.kind == "target_color":
            self.appearance.materials[f"Target_{command.name}"].set_color(np.array(command.value))
        elif command.kind == "move_target":
            initial = self.target_parts[command.name][0][1]
            pose = np.array(
                [[*(initial + np.array([command.value, 0.0, 0.0])), 0.0, 0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            self.kinematic_views[command.name].set_kinematic_targets(
                pose, np.array([0], dtype=np.uint32)
            )
            self.fixture_offsets[command.name] = command.value
        elif command.kind in ("release", "activate"):
            self.supports[command.name].CreateJointEnabledAttr(False)
            if command.kind == "activate":
                self.objects[command.name].set_visibility(True)
                self._set_collision(command.name, True)
            else:
                self.objects[command.name].set_linear_velocity(np.array(command.value))
                # Record the velocity actually assigned to PhysX before its contact solve.
                self._release_speeds[command.name] = float(
                    np.linalg.norm(self.objects[command.name].get_linear_velocity()[:2])
                )
        else:
            raise ValueError(f"Unknown scene command: {command.kind}")

    def apply_action(self, joints):
        """16 joint targets in internal side order: side 0 (/World/ARXLeft, the rig's RIGHT arm)
        first. `joint_positions` / `observation` use the same order; LabEnv converts to and
        from the interface order (rig left first)."""
        from isaacsim.core.utils.types import ArticulationAction

        joints = np.asarray(joints)
        if joints.shape != (16,) or not np.all(np.isfinite(joints)):
            raise ValueError("Dual ARX X5 requires 16 finite joint targets")
        self.robot.apply_action(ArticulationAction(joint_positions=joints[:8]))
        if self.secondary_robot is not None:
            if joints.shape != (16,):
                raise ValueError("Dual ARX X5 requires 16 joint targets")
            self.secondary_robot.apply_action(ArticulationAction(joint_positions=joints[8:]))

    def joint_positions(self):
        q = self.robot.get_joint_positions()
        return (
            q
            if self.secondary_robot is None
            else np.concatenate([q, self.secondary_robot.get_joint_positions()])
        )

    def tick(self, render=False):
        self.world.step(render=render)
        inter_arm = getattr(self, "inter_arm_contacts", None)
        inter_arm_collision = bool(
            inter_arm is not None
            and np.any(
                np.linalg.norm(
                    inter_arm.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"]),
                    axis=-1,
                )
                > 0.5
            )
        )
        if getattr(self.task, "liquid_process", False):
            from roboclock.envs.lab.liquid_scene import measure_liquid

            forces = self.arm_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
            colliding = inter_arm_collision or bool(np.any(np.linalg.norm(forces, axis=-1) > 0.5))
            secondary = self.secondary_contacts.get_contact_force_matrix(
                dt=self.config["sim"]["physics_dt"]
            )
            colliding |= bool(np.any(np.linalg.norm(secondary, axis=-1) > 0.5))
            if colliding and not self.last_collision:
                self.collision_count += 1
            self.last_collision = colliding
            self._cached_state = measure_liquid(self, self.config["sim"]["physics_dt"])
            return
        forces = self.arm_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        colliding = inter_arm_collision or bool(np.any(np.linalg.norm(forces, axis=-1) > 0.5))
        secondary_clutter_contact = False
        if colliding and not self.last_collision:
            self.collision_count += 1
        if self.secondary_robot is not None:
            secondary_forces = self.secondary_contacts.get_contact_force_matrix(
                dt=self.config["sim"]["physics_dt"]
            )
            colliding |= bool(np.any(np.linalg.norm(secondary_forces, axis=-1) > 0.5))
            indices = getattr(self, "secondary_clutter_filter_indices", [])
            secondary_clutter_contact = bool(
                indices and np.any(np.linalg.norm(secondary_forces[:, indices], axis=-1) > 0.5)
            )
            if (
                colliding
                and not self.last_collision
                and not inter_arm_collision
                and not np.any(np.linalg.norm(forces, axis=-1) > 0.5)
            ):
                self.collision_count += 1
        clutter_contact = secondary_clutter_contact or bool(
            self.clutter and np.any(np.linalg.norm(forces[:, -len(self.clutter) :], axis=-1) > 0.5)
        )
        if clutter_contact and not self._last_clutter_contact:
            self.clutter_robot_contact_count += 1
        self._last_clutter_contact = clutter_contact
        self.last_collision = colliding
        self._cached_state = self._measure_state(self.config["sim"]["physics_dt"])

    def state(self):
        return self._cached_state

    def _measure_state(self, elapsed):
        dt = self.config["sim"]["physics_dt"]
        forces = self.contacts.get_contact_force_matrix(dt=dt)
        # View ordering need not match dictionary order. Bind by actual prim path.
        rows = {path.rsplit("/", 1)[-1]: i for i, path in enumerate(self.contacts.prim_paths)}
        ee, ee_orientation = self._tool_pose()
        tool_velocity = (
            np.zeros(3)
            if self._previous_tool is None or elapsed == 0
            else (ee - self._previous_tool) / elapsed
        )
        self._previous_tool = ee.copy()
        q = self.robot.get_joint_positions()
        q2 = self.secondary_robot.get_joint_positions()
        ee2, _ = self._tool_pose(self.secondary_robot)
        previous2 = getattr(self, "_previous_secondary_tool", None)
        velocity2 = (
            np.zeros(3) if previous2 is None or elapsed == 0 else (ee2 - previous2) / elapsed
        )
        self._previous_secondary_tool = ee2.copy()
        # Measured from joint positions: PhysX reports about 0.04 rad/s for an arm that
        # holds its pose against gravity, which read as motion on every tick.
        arm_joints = np.stack([q[:6], q2[:6]])
        previous_joints = getattr(self, "_previous_arm_joints", None)
        robot_moving = bool(
            previous_joints is not None
            and elapsed > 0
            and np.any(np.linalg.norm(arm_joints - previous_joints, axis=1) > 0.03 * elapsed)
        )
        self._previous_arm_joints = arm_joints.copy()
        forces2 = self.secondary_sample_contacts.get_contact_force_matrix(dt=dt)
        rows2 = {
            path.rsplit("/", 1)[-1]: i
            for i, path in enumerate(self.secondary_sample_contacts.prim_paths)
        }
        self.grasp_owners = {}
        objects = {}
        for name, body in self.objects.items():
            position, orientation = body.get_world_pose()
            row = rows[name]
            contacts = np.linalg.norm(forces[row], axis=-1) > 0.05
            velocity = body.get_linear_velocity().copy()
            previous_position = self._previous_object_positions.get(name)
            measured_velocity = (
                velocity
                if previous_position is None or elapsed == 0
                else (position - previous_position) / elapsed
            )
            self._previous_object_positions[name] = position.copy()
            radius, height = self.layout.dimensions[name]
            grasped, relative_speed = self.grasps.update(
                name,
                forces[row, :2],
                q[6] + q[7],
                position,
                ee,
                measured_velocity,
                tool_velocity,
                radius,
                height,
                elapsed,
            )
            grasped2, speed2 = self.grasps.update(
                ("right", name),
                forces2[rows2[name], :2],
                q2[6] + q2[7],
                position,
                ee2,
                measured_velocity,
                velocity2,
                radius,
                height,
                elapsed,
            )
            if grasped or grasped2:
                self.grasp_owners[name] = 0 if grasped else 1
            if grasped2:
                relative_speed = speed2
            hand_contact = bool(
                np.any(contacts[:3]) or np.any(np.linalg.norm(forces2[rows2[name]], axis=-1) > 0.05)
            )
            surfaces = frozenset(key for i, key in enumerate(self.surface_paths) if contacts[4 + i])
            objects[name] = ObjectState(
                position.copy(),
                orientation.copy(),
                velocity,
                grasped or grasped2,
                hand_contact,
                bool(contacts[3]),
                surfaces,
                relative_speed,
                self._release_speeds.pop(name, 0.0),
            )
        return PhysicalState(
            objects,
            ee,
            bool(q[6] + q[7] > 0.065),
            robot_moving,
            ee_orientation.copy(),
            self.reactive_scene.measure(objects["sample_0"], elapsed)
            if (
                getattr(self.task, "power_task", False)
                or getattr(self.task, "tube_task", False)
                or getattr(self.task, "safety_task", False)
                or getattr(self.task, "bath_task", False)
            )
            else {},
            ee2.copy(),
        )

    def capture_camera(self, subframes=1):
        from roboclock.simulators.isaac.rendering import capture_camera

        capture_camera(subframes)

    def images(self, depth=False):
        """Policy camera images of the current scene, rendered at most once per instant."""
        if self.robodojo_cameras:
            return self.rig_cameras.read(self.task.time, depth=depth)
        if self.camera is None and self.no_render:
            width, height = self.config["sim"]["camera_resolution"]
            images = {"rgb": np.zeros((height, width, 3), dtype=np.uint8)}
            if depth:
                images["depth"] = np.zeros((height, width), dtype=np.float32)
        else:
            self.capture_camera()
            rgba = self.camera.get_rgba()
            if rgba is None or rgba.size == 0:
                raise RuntimeError(
                    "Camera has no RGB frame. Check RTX rendering and camera initialization."
                )
            images = {"rgb": np.asarray(rgba[:, :, :3], dtype=np.uint8).copy()}
            if depth:
                from roboclock.envs.lab.rig_cameras import depth_image

                images["depth"] = depth_image(self.camera, "camera")
        return images

    def observation(self, depth=False, images=True):
        """Proprioception, preceded by the camera images unless `images` is false."""
        position, orientation = self._tool_pose()
        q = self.joint_positions()
        result = {
            **(self.images(depth) if images else {}),
            "joint_positions": np.asarray(q, dtype=np.float32),
            "ee_pose": np.asarray([*position, *orientation], dtype=np.float32),
            "gripper_width": np.array([np.clip(q[6] + q[7], 0.0, 0.088)], dtype=np.float32),
            "instruction": self.task.gen_instruction(),
            "sim_time": np.array([self.task.time], dtype=np.float32),
        }
        if self.secondary_robot is not None:
            pos2, quat2 = self._tool_pose(self.secondary_robot)
            result["ee_pose"] = np.asarray(
                [*position, *orientation, *pos2, *quat2], dtype=np.float32
            )
            result["gripper_width"] = np.asarray(
                [np.clip(q[6] + q[7], 0, 0.088), np.clip(q[14] + q[15], 0, 0.088)], dtype=np.float32
            )
        return result

    def _tool_pose(self, robot=None):
        from roboclock.envs.lab.robot import tool_pose

        return tool_pose(self.robot if robot is None else robot)

    def close(self):
        if self.camera is not None:
            self.camera.destroy()
            self.camera = None
        if getattr(self, "rig_cameras", None) is not None:
            self.rig_cameras.close()
            self.rig_cameras = None
        if self.world is not None:
            self.world.stop()
            if getattr(self, "kinematic_sim", None) is not None:
                self.kinematic_views.clear()
                self.kinematic_sim.invalidate()
                self.kinematic_sim = None
            self.world.clear()
            self.world = None
            self.objects.clear()
            self.clutter.clear()
            self.target_parts.clear()
            self.supports.clear()
            self.contacts = None
            self.arm_contacts = None
            self.secondary_contacts = None
            self.inter_arm_contacts = None
            self.secondary_sample_contacts = None
            self.reactive_scene = None
            self.hotplate_scene = None
            self.liquid_scene = None
            self._previous_secondary_tool = None
            self.robot = None
            self.secondary_robot = None
            self.appearance = None
            self.indicator_material = None
            self.stage = None
            import gc

            gc.collect()

