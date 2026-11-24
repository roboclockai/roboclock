"""Isaac Sim sports tabletop: shared robot, physical controls and dynamic props."""

import json
from pathlib import Path

import numpy as np

from roboclock.physics.contacts import GraspTracker
from roboclock.physics.geometry import rotation_matrix
from roboclock.simulators.isaac.rig_view import RigView


class SportsBackend(RigView):
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.world = self.camera = None
        self.catalog = json.loads((Path(__file__).parent / "assets/catalog.json").read_text())

    def reset(self, task):
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
        from isaacsim.core.api.objects import FixedCuboid, VisualCuboid
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, UsdGeom, UsdLux, UsdShade

        from roboclock.envs.lab.appearance import LabAppearance
        from roboclock.resources import resolve_asset
        from roboclock.simulators.isaac.rendering import configure_renderer

        self.close()
        World.clear_instance()
        create_new_stage()
        self.task = task
        self.layout = task.layout
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config["sim"]["physics_dt"],
            rendering_dt=self.config["sim"]["physics_dt"],
        )
        self.world.get_physics_context().enable_ccd(True)
        self.stage = get_current_stage()
        configure_renderer(self.config)
        self.world.scene.add_default_ground_plane()
        h = self.config["scene"]["table_height"]
        self.world.scene.add(
            FixedCuboid(
                "/World/Table",
                name="table",
                position=np.array([0.43, 0, h - 0.025]),
                scale=np.array([1.05, 1.0, 0.05]),
                color=np.array([0.13, 0.17, 0.21]),
            )
        )
        # Reuse the laboratory's checked UV meshes and MDL texture library.
        # Collision geometry stays on the original table and ground plane.
        appearance = LabAppearance(self.stage, self.world, self.config)
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/Table")).MakeInvisible()
        UsdGeom.Imageable(self.stage.GetPrimAtPath("/World/defaultGroundPlane")).MakeInvisible()
        appearance.rounded(
            "/World/TableSurface", [0.43, 0, h - 0.025], [1.05, 1.0, 0.05], "worktop", radius=0.004
        )
        appearance.rounded("/World/FloorSurface", [0.4, 0, -0.021], [6, 6, 0.04], "floor")
        appearance.rounded("/World/Backdrop", [0.45, 1.7, 1.4], [5, 0.08, 2.8], "wall")
        for x in (0.05, 0.80):
            for y in (-0.38, 0.38):
                self.world.scene.add(
                    FixedCuboid(
                        f"/World/Leg_{x}_{y}".replace(".", "_").replace("-", "n"),
                        name=f"leg_{x}_{y}",
                        position=np.array([x, y, h / 2 - 0.02]),
                        scale=np.array([0.055, 0.055, h - 0.04]),
                        color=np.array([0.13, 0.15, 0.18]),
                    )
                )
                leg_path = f"/World/Leg_{x}_{y}".replace(".", "_").replace("-", "n")
                UsdShade.MaterialBindingAPI.Apply(self.stage.GetPrimAtPath(leg_path)).Bind(
                    appearance.material("steel").material
                )
        from roboclock.simulators.isaac.embodiments.single import spawn

        spawn(self)  # Dual ARX X5 (config["embodiment"]); sets robot, finger/hand/arm paths
        self.material = PhysicsMaterial(
            "/World/Materials/Contact",
            static_friction=self.layout.friction,
            dynamic_friction=self.layout.friction,
            restitution=0.0,
        )
        self.surface_paths = ["/World/Table"]
        for fixture in self.layout.fixtures:
            name = fixture["name"].replace("-", "n").replace(".", "_")
            if fixture["name"] in ("goal_mark", "mole_board", "spot_deck"):
                cls = VisualCuboid
            else:
                cls = FixedCuboid
            body = self.world.scene.add(
                cls(
                    f"/World/Fixtures/{name}",
                    name=name,
                    position=np.array(fixture["position"]),
                    scale=np.array(fixture["size"]),
                    color=np.array(fixture["color"]),
                )
            )
            if cls is FixedCuboid:
                self.surface_paths.append(body.prim_path)
        self.objects = {}
        for name, spec in self.layout.bodies.items():
            self.objects[name] = self.make_body(spec)
        self.controls = {}
        self.drives = {}
        self.control_materials = {}
        self.rest = {}
        for name, spec in self.layout.controls.items():
            self.make_control(spec)
        from roboclock.envs.sports.common.equipment import build_equipment

        build_equipment(self)
        self.customize_scene()
        for name, p in self.layout.targets.items():
            if name == "hopper":
                continue
            self.world.scene.add(
                VisualCuboid(
                    f"/World/Targets/{name}",
                    name="target_" + name,
                    position=np.array([p[0], p[1], h + 0.0005]),
                    scale=np.array([0.042, 0.042, 0.001]),
                    color=np.array([0.14, 0.45, 0.40]),
                )
            )
            self.label(
                f"/World/TargetLabels/{name}",
                str(int(name.split("_")[1]) + 1)
                if name.startswith(("slot_", "pin_"))
                else name.upper(),
                [p[0], p[1], h + 0.0011],
                [0.025, 0.020],
                True,
            )
        self.display_material = PreviewSurface(
            "/World/Materials/Signal", color=np.array([0.1, 0.4, 0.9])
        )
        signal = self.world.scene.add(
            VisualCuboid(
                "/World/Signal",
                name="signal",
                position=np.array([0.50, 0.32, h + 0.028]),
                scale=np.array([0.40, 0.075, 0.045]),
                visual_material=self.display_material,
            )
        )
        if task.task_id not in {"36", "39"}:
            from pxr import UsdGeom

            UsdGeom.Imageable(signal.prim).MakeInvisible()
        self.card_materials = []
        if task.task_id == "39":
            for i in range(5):
                material = PreviewSurface(
                    f"/World/Materials/Card{i}", color=np.array([0.5, 0.5, 0.5])
                )
                self.card_materials.append(material)
                self.world.scene.add(
                    VisualCuboid(
                        f"/World/CardTiles/Tile{i}",
                        name=f"card_tile_{i}",
                        position=np.array([0.50 + (i - 2) * 0.065, 0.32, h + 0.054]),
                        scale=np.array([0.052, 0.054, 0.005]),
                        visual_material=material,
                    )
                )
        self.label(
            "/World/Title",
            f"SPORTS {task.task_id}  {task.name.replace('_', ' ').upper()}",
            [0.47, 0.42, h + 0.001],
            [0.52, 0.045],
            False,
        )
        self._display_text = None
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lights/Dome")
        dome.CreateTextureFileAttr(str(resolve_asset(self.config["visuals"]["hdr"])))
        dome.CreateIntensityAttr(self.layout.light_intensity)
        for i, (p, power) in enumerate((([0.3, -0.5, 2.0], 1000), ([0.8, 0.5, 1.8], 700))):
            light = UsdLux.RectLight.Define(self.stage, f"/World/Lights/Area{i}")
            light.AddTranslateOp().Set(Gf.Vec3d(*p))
            light.CreateWidthAttr(0.8)
            light.CreateHeightAttr(0.6)
            light.CreateIntensityAttr(power)
        paths = [b.prim_path for b in self.objects.values()] + [
            b.prim_path for b in self.controls.values()
        ]
        self.contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=paths,
                name="sports_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    self.finger_paths + [self.hand_path] + self.surface_paths for _ in paths
                ],
                max_contact_count=16384,
            )
        )
        self.arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=self.arm_paths,
                name="sports_arm_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[self.surface_paths + paths for _ in self.arm_paths],
                max_contact_count=16384,
            )
        )
        cp = np.array(self.config["scene"]["camera_position"])
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*cp), Gf.Vec3d(*self.config["scene"]["camera_look_at"]), Gf.Vec3d(0, 0, 1)
        )
        cq = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
        self.camera = Camera(
            "/World/Camera",
            position=cp,
            orientation=cq,
            resolution=tuple(self.config["sim"]["camera_resolution"]),
        )
        self.world.reset()
        self.initialize_robot(task)
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.set_world_pose(cp, cq, camera_axes="usd")
        self.camera.set_focal_length(0.022)
        self.camera.set_horizontal_aperture(0.036)
        self.camera.set_clipping_range(0.01, 10.0)
        if task.task_id == "34":
            for drive in self.drives.values():
                drive.GetTargetPositionAttr().Set(-0.080)
        for _ in range(self.config["sim"]["settle_steps"]):
            self.world.step(render=False)
        self.grasps = GraspTracker(
            minimum_time=0.04, force_threshold=self.config["physics"]["contact_force"]
        )
        self.previous_tool = None
        self.previous_positions = {}
        self.previous_orientations = {}
        self.collision_count = 0
        self.last_collision = False
        self._state = self.measure(0.0)
        self.update_cues()
        self.world.render()

    def initialize_robot(self, task):
        from roboclock.simulators.isaac.embodiments.single import initialize

        initialize(self)

    def customize_scene(self):
        """Optional versioned scene extension, before PhysX views initialize."""

    def collider(self, path, kind, position, dimensions, color, collision=True):
        from isaacsim.core.api.materials import PreviewSurface
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        if kind == "sphere":
            shape = UsdGeom.Sphere.Define(self.stage, path)
            shape.CreateRadiusAttr(dimensions[0])
        elif kind == "cube":
            shape = UsdGeom.Cube.Define(self.stage, path)
            shape.CreateSizeAttr(1.0)
            shape.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        else:
            shape = UsdGeom.Cylinder.Define(self.stage, path)
            shape.CreateRadiusAttr(dimensions[0])
            shape.CreateHeightAttr(dimensions[1])
            shape.CreateAxisAttr("Z")
        translation = shape.AddTranslateOp()
        translation.Set(Gf.Vec3d(*map(float, position)))
        # USD composes ordered ops: translation must precede local scaling.
        ops = shape.GetOrderedXformOps()
        shape.SetXformOpOrder([translation] + [op for op in ops if op != translation])
        if collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            api = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
            api.CreateContactOffsetAttr(0.001)
            api.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(
                self.material.material, materialPurpose="physics"
            )
        mat = PreviewSurface(
            path + "Material", color=np.array(color), roughness=0.42, metallic=0.15
        )
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat.material)
        return shape, mat

    def imported_visual(self, path, asset, size, color):
        from isaacsim.core.api.materials import PreviewSurface
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

        prim = add_reference_to_stage(self.catalog[asset]["url"], path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"NVIDIA asset failed to load: {asset}")
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.RigidBodyAPI):
                child.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        bounds = (
            UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            .ComputeUntransformedBound(prim)
            .ComputeAlignedRange()
        )
        low = np.array(bounds.GetMin())
        high = np.array(bounds.GetMax())
        extent = high - low
        if not np.all(np.isfinite(extent)) or np.any(extent <= 0):
            raise RuntimeError(f"Invalid asset bounds: {asset}")
        scale = np.asarray(size) / extent
        center = (low + high) / 2
        xf.AddTranslateOp(opSuffix="sportsNormalize").Set(Gf.Vec3d(*map(float, -center * scale)))
        xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="sportsNormalize").Set(
            Gf.Vec3d(*map(float, scale))
        )
        material = PreviewSurface(path + "Tint", color=np.asarray(color), roughness=0.42)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )

    def make_body(self, spec):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

        path = f"/World/Objects/{spec.name}"
        UsdGeom.Xform.Define(self.stage, path)
        if spec.shape == "dumbbell":
            from isaacsim.core.utils.stage import add_reference_to_stage

            # Reuse the actual simulation-ready USD, original mesh collisions,
            # material, texture and 2.27 kg mass. No replacement visual primitive.
            source = Path(__file__).parent / "assets" / self.catalog["dumbbell"]["local_path"]
            add_reference_to_stage(str(source.resolve()), path)
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(path)):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    collision.CreateContactOffsetAttr(0.001)
                    collision.CreateRestOffsetAttr(0.0)
        else:
            kind = "sphere" if spec.shape == "sphere" else "cylinder"
            shape, _ = self.collider(
                path + "/Collision", kind, [0, 0, 0], [spec.radius, spec.height], spec.color
            )
            if spec.shape in ("sphere", "puck", "cylinder"):
                UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
                asset = "sphere" if spec.shape == "sphere" else "cylinder"
                self.imported_visual(
                    path + "/Visual",
                    asset,
                    [2 * spec.radius, 2 * spec.radius, spec.height],
                    spec.color,
                )
            if spec.name.startswith("pin_"):
                from roboclock.envs.sports.common.equipment import annulus

                for z in (0.015, 0.022):
                    annulus(
                        self,
                        path + f"/Stripe{int(z * 1000)}",
                        [0, 0, z],
                        spec.radius - 0.0002,
                        spec.radius + 0.0003,
                        0.003,
                        [0.75, 0.025, 0.025],
                    )
            if spec.shape == "cup":
                # Hollow visual shell with conservative convex grasp envelope.
                from roboclock.envs.sports.common.equipment import annulus

                UsdGeom.Imageable(shape.GetPrim()).MakeInvisible()
                annulus(
                    self,
                    path + "/Shell",
                    [0, 0, 0],
                    spec.radius - 0.002,
                    spec.radius,
                    spec.height,
                    spec.color,
                )
                self.collider(
                    path + "/InnerFloor",
                    "cylinder",
                    [0, 0, -spec.height / 2 + 0.0015],
                    [spec.radius - 0.002, 0.003],
                    spec.color,
                    False,
                )
                # Upper rim retains the original grasp envelope.
                annulus(
                    self,
                    path + "/Rim",
                    [0, 0, spec.height / 2 + 0.001],
                    spec.radius - 0.002,
                    spec.radius + 0.001,
                    0.003,
                    spec.color,
                )

        body = self.world.scene.add(
            SingleRigidPrim(path, name=spec.name, position=np.array(spec.position), mass=spec.mass)
        )
        api = PhysxSchema.PhysxRigidBodyAPI.Apply(body.prim)
        api.CreateEnableCCDAttr(True)
        api.CreateSolverPositionIterationCountAttr(16)
        api.CreateSolverVelocityIterationCountAttr(4)
        if spec.label and spec.shape not in ("dumbbell", "cup"):
            self.label(
                path + "/Identity",
                spec.label,
                [0, 0, spec.height / 2 + 0.0004],
                [0.018, 0.018],
                False,
            )
        return body

    def make_control(self, spec):
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        h = self.config["scene"]["table_height"]
        if spec.kind != "mole":
            self.world.scene.add(
                FixedCuboid(
                    f"/World/Bases/{spec.name}",
                    name="base_" + spec.name,
                    position=np.array([*spec.position[:2], h + 0.016]),
                    scale=np.array([spec.radius * 2 + 0.02, spec.radius * 2 + 0.02, 0.032]),
                    color=np.array([0.12, 0.15, 0.20]),
                )
            )
            self.surface_paths.append(f"/World/Bases/{spec.name}")
        path = f"/World/Controls/{spec.name}"
        UsdGeom.Xform.Define(self.stage, path)
        _, material = self.collider(
            path + "/Cap",
            "cylinder",
            [0, 0, 0],
            [spec.radius, spec.height],
            spec.color,
            collision=spec.kind != "pull",
        )
        if spec.kind == "pull":
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/Cap")).MakeInvisible()
            self.collider(
                path + "/Stem", "cylinder", [0, 0, -0.003], [0.009, 0.014], [0.15, 0.18, 0.21]
            )
            self.collider(path + "/Handle", "cylinder", [0, 0, 0.006], [0.014, 0.008], spec.color)
        body = self.world.scene.add(
            SingleRigidPrim(path, name=spec.name, position=np.array(spec.position), mass=0.06)
        )
        PhysxSchema.PhysxRigidBodyAPI.Apply(body.prim).CreateDisableGravityAttr(True)
        joint_path = f"/World/Joints/{spec.name}"
        if spec.kind == "twist":
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, joint_path)
            joint.CreateLowerLimitAttr(-5.0)
            joint.CreateUpperLimitAttr(100.0)
            mode = "angular"
        else:
            joint = UsdPhysics.PrismaticJoint.Define(self.stage, joint_path)
            mode = "linear"
            joint.CreateLowerLimitAttr(-spec.travel if spec.kind in ("press", "mole") else 0.0)
            joint.CreateUpperLimitAttr(spec.travel if spec.kind == "pull" else 0.0)
        joint.CreateBody1Rel().SetTargets([path])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, spec.position)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), mode)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(
            0.0005 if spec.kind == "twist" else 120.0 if spec.kind == "press" else 30.0
        )
        drive.CreateDampingAttr(0.0002 if spec.kind == "twist" else 5.0)
        drive.CreateTargetPositionAttr(0.0)
        self.controls[spec.name] = body
        self.drives[spec.name] = drive
        self.rest[spec.name] = np.array(spec.position)
        self.control_materials[spec.name] = material
        if spec.kind == "mole":
            # Round head, eyes and nose travel with the real prismatic rigid body.
            from pxr import UsdGeom

            UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/Cap")).MakeInvisible()
            _, material = self.collider(
                path + "/Head", "sphere", [0, 0, 0], [spec.radius], spec.color, False
            )
            self.control_materials[spec.name] = material
            for side in (-1, 1):
                self.collider(
                    path + f"/Eye{side}".replace("-", "n"),
                    "sphere",
                    [side * 0.008, -0.020, 0.010],
                    [0.004],
                    [0.95, 0.95, 0.86],
                    False,
                )
                self.collider(
                    path + f"/Pupil{side}".replace("-", "n"),
                    "sphere",
                    [side * 0.008, -0.023, 0.010],
                    [0.0018],
                    [0.01, 0.015, 0.02],
                    False,
                )
            self.collider(
                path + "/Nose", "sphere", [0, -0.023, 0.002], [0.004], [0.25, 0.09, 0.075], False
            )
            return
        if spec.name == "bell":
            UsdGeom.Imageable(self.stage.GetPrimAtPath(path + "/Cap")).MakeInvisible()
            dome, _ = self.collider(
                path + "/Dome", "sphere", [0, 0, -0.006], [spec.radius], [0.72, 0.49, 0.12], False
            )
            dome.AddScaleOp().Set(Gf.Vec3f(1, 1, 0.43))
            self.collider(
                path + "/Striker",
                "cylinder",
                [0, 0, 0.004],
                [0.007, 0.004],
                [0.8, 0.82, 0.85],
                False,
            )
            return
        if spec.kind == "twist":
            self.collider(
                path + "/Pointer",
                "cube",
                [0, 0.006, spec.height / 2 + 0.0006],
                [0.003, 0.013, 0.001],
                [0.96, 0.96, 0.88],
                False,
            )
        if spec.kind == "pull":
            from roboclock.envs.sports.common.equipment import annulus

            annulus(
                self,
                path + "/GripRim",
                [0, 0, spec.height / 2 - 0.002],
                0.010,
                0.014,
                0.004,
                [0.08, 0.12, 0.17],
            )
        if spec.kind in {"pull", "twist"}:
            self.label(
                path + "/Identity",
                "PULL" if spec.kind == "pull" else "TURN",
                [0, 0, spec.height / 2 + 0.0003],
                [0.013, 0.008],
                False,
            )
            return
        self.label(
            path + "/Label",
            str(int(spec.name[-1]) + 1) if spec.name.startswith("pad_") else spec.name.upper(),
            [0, 0, spec.height / 2 + 0.0003],
            [spec.radius * 0.7, spec.radius * 0.55]
            if spec.name.startswith("pad_")
            else [spec.radius * 1.7, spec.radius * 0.8],
            False if spec.name.startswith("pad_") else True,
        )

    def update_cues(self):
        from roboclock.envs.sports.common.equipment import update_equipment

        update_equipment(self)
        if self.task.task_id not in {"36", "39"}:
            return
        text, color = self.task.display()
        if text != self._display_text:
            self.display_material.set_color(np.array(color))
            h = self.config["scene"]["table_height"]
            if self.task.task_id == "39":
                from roboclock.envs.sports.common.layout import COLORS

                for i, color_id in enumerate(self.task.events[self.task.card_index]["order"]):
                    self.card_materials[i].set_color(np.array(COLORS[color_id]))
                self.label(
                    "/World/Display",
                    f"CARD {self.task.card_index + 1}",
                    [0.5, 0.38, h + 0.001],
                    [0.16, 0.028],
                    False,
                )
            else:
                self.label("/World/Display", text, [0.50, 0.32, h + 0.055], [0.36, 0.055], False)
            self._display_text = text

    def tick(self, render=False):
        task = self.task
        if task.task_id == "34":
            for name, drive in self.drives.items():
                up = any(
                    e["control"] == name and task.mole_active(i) for i, e in enumerate(task.events)
                )
                drive.GetTargetPositionAttr().Set(0.0 if up else -self.layout.controls[name].travel)
        while task.commands:
            command, name = task.commands.pop(0)
            if command == "fire":
                # A machine launch impulse changes velocity through PhysX; position is never reset.
                self.objects[name].set_linear_velocity(np.array([0.0, 0.6, 2.4]))
        self.world.step(render=render)
        forces = self.arm_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        collision = bool(np.any(np.linalg.norm(forces, axis=-1) > 0.5))
        if collision and not self.last_collision:
            self.collision_count += 1
        self.last_collision = collision
        self._state = self.measure(self.config["sim"]["physics_dt"])

    def measure(self, dt):
        forces = self.contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        rows = {p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.contacts.prim_paths)}
        ee, q = self.tool_pose()
        v = np.zeros(3) if self.previous_tool is None or dt == 0 else (ee - self.previous_tool) / dt
        self.previous_tool = ee.copy()
        from roboclock.simulators.isaac.embodiments.single import width

        gripper_width = width(self)
        threshold = self.config["physics"]["contact_force"]
        objects = {}
        controls = {}
        for name, body in self.objects.items():
            p, r = body.get_world_pose()
            f = forces[rows[name]]
            spec = self.layout.bodies[name]
            velocity = body.get_linear_velocity().copy()
            prev = self.previous_positions.get(name)
            measured = velocity if prev is None or dt == 0 else (p - prev) / dt
            self.previous_positions[name] = p.copy()
            from roboclock.physics.geometry import angular_velocity_from_poses

            previous_orientation = self.previous_orientations.get(name)
            measured_angular = (
                body.get_angular_velocity().copy()
                if previous_orientation is None or dt == 0
                else angular_velocity_from_poses(previous_orientation, r, dt)
            )
            self.previous_orientations[name] = r.copy()
            grasped, _ = self.grasps.update(
                name, f[:2], gripper_width, p, ee, measured, v, spec.radius, spec.height, dt
            )
            objects[name] = {
                "position": p.copy(),
                "orientation": r.copy(),
                "velocity": measured,
                "solver_velocity": velocity,
                "angular_velocity": measured_angular,
                "solver_angular_velocity": body.get_angular_velocity().copy(),
                "grasped": grasped,
                "robot_contact": bool(np.any(np.linalg.norm(f[:3], axis=-1) > threshold)),
                "supported": bool(np.any(np.linalg.norm(f[3:], axis=-1) > threshold)),
                "contact_force": float(np.linalg.norm(f[:3])),
                "finger_contact_forces": f[:2].copy(),
                "gripper_width": gripper_width,
            }
        for name, body in self.controls.items():
            p, r = body.get_world_pose()
            spec = self.layout.controls[name]
            f = forces[rows[name]]
            contact = bool(np.any(np.linalg.norm(f[:3], axis=-1) > threshold))
            if spec.kind == "twist":
                rotation = rotation_matrix(r)
                travel = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
            elif spec.kind == "pull":
                travel = float(p[2] - self.rest[name][2])
            else:
                travel = float(self.rest[name][2] - p[2])
            activated = travel >= spec.threshold and contact
            if spec.kind == "mole":
                up = any(
                    e["control"] == name and self.task.mole_active(i)
                    for i, e in enumerate(self.task.events)
                )
                exposed = p[2] + spec.radius >= self.config["scene"]["table_height"] + 0.112
                activated = up and exposed and 0.003 <= travel <= 0.035 and contact
            controls[name] = {
                "position": p.copy(),
                "orientation": r.copy(),
                "travel": travel,
                "exposed": bool(exposed) if spec.kind == "mole" else True,
                "contact": contact,
                "activated": activated,
                "bilateral": bool(np.all(np.linalg.norm(f[:2], axis=-1) > threshold)),
            }
        return {
            "objects": objects,
            "controls": controls,
            "tool_position": ee,
            "tool_orientation": q,
            "tool_velocity": v,
        }
