"""Chess scene (real cylindrical pieces, spring-return clock buttons) for sports A1 on the Dual ARX X5."""

import numpy as np

from roboclock.envs.chess.assets import add_model
from roboclock.physics.contacts import GraspTracker
from roboclock.simulators.isaac.rig_view import RigView


class ChessBackend(RigView):
    def __init__(self, runtime, config):
        self.runtime, self.config = runtime, config
        self.world = self.camera = None

    def reset(self, task):
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial, PreviewSurface
        from isaacsim.core.api.objects import DynamicCylinder, FixedCuboid, VisualCuboid
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.rotations import gf_quat_to_np_array
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage
        from isaacsim.sensors.camera import Camera
        from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from roboclock.resources import resolve_asset
        from roboclock.simulators.isaac.rendering import configure_renderer

        self.close()
        World.clear_instance()
        create_new_stage()
        self.task = task
        c, scene = self.config, self.config["scene"]
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=c["sim"]["physics_dt"],
            rendering_dt=c["sim"]["physics_dt"],
        )
        self.world.get_physics_context().enable_ccd(True)
        self.stage = get_current_stage()
        configure_renderer(c)
        self.world.scene.add_default_ground_plane()
        h = scene["table_height"]
        self.world.scene.add(
            FixedCuboid(
                "/World/Table",
                name="table",
                position=np.array([0.4, 0, h - 0.025]),
                scale=np.array([1.05, 0.95, 0.05]),
                color=np.array([0.13, 0.17, 0.19]),
            )
        )
        x, y = scene["board_center"]
        size = scene["square_size"] * 8
        board = self.world.scene.add(
            FixedCuboid(
                "/World/BoardSupport",
                name="board_support",
                position=np.array([x, y, h + scene["board_height"] / 2]),
                scale=np.array([size + 0.06, size + 0.06, scene["board_height"]]),
            )
        )
        UsdGeom.Imageable(board.prim).MakeInvisible()
        board_visual = add_model(
            self.stage,
            "/World/Board",
            "board.glb",
            [x, y, h],
            [scene["square_size"] / 0.05] * 2 + [scene["board_height"] / 0.019],
        )
        # The source corner is ivory; rotate it so a1 is dark and h1 is light.
        board_visual.AddRotateZOp().Set(90.0)
        clock = np.array(scene["clock_position"])
        add_model(self.stage, "/World/ClockVisual", "clock.glb", clock, skip_buttons=True)
        self.world.scene.add(
            FixedCuboid(
                "/World/ClockHousing",
                name="clock_housing",
                position=clock + [0, 0, 0.030],
                scale=np.array([0.16, 0.058, 0.060]),
                visible=False,
            )
        )
        capture = self.world.scene.add(
            FixedCuboid(
                "/World/CaptureTray",
                name="capture_tray",
                position=np.array([0.35, 0.31, h + 0.003]),
                scale=np.array([0.095, 0.095, 0.006]),
                color=np.array([0.16, 0.24, 0.26]),
            )
        )
        from roboclock.simulators.isaac.embodiments.single import spawn

        spawn(self)  # Dual ARX X5 (config["embodiment"]); sets robot, finger/hand/arm paths
        material = PhysicsMaterial(
            "/World/Materials/Grip", static_friction=1.0, dynamic_friction=0.8, restitution=0.0
        )
        self.objects = {}
        self.grasps = GraspTracker(minimum_time=0.04, force_threshold=c["physics"]["contact_force"])
        for name, symbol in task.initial_symbols.items():
            white = symbol.isupper()
            body = self.world.scene.add(
                DynamicCylinder(
                    "/World/Pieces/" + name,
                    name=name,
                    position=task.position(name[6:]),
                    radius=scene["piece_radius"],
                    height=scene["piece_height"],
                    mass=scene["piece_mass"],
                    color=np.array([0.88, 0.82, 0.65] if white else [0.055, 0.07, 0.085]),
                    physics_material=material,
                )
            )
            api = PhysxSchema.PhysxCollisionAPI.Apply(body.prim)
            api.CreateContactOffsetAttr(0.001)
            api.CreateRestOffsetAttr(0.0)
            # Isaac's primitive material defaults to overriding descendants.
            # Let each top-face identity label keep its own textured material.
            binding = UsdShade.MaterialBindingAPI(body.prim).GetDirectBindingRel()
            UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
                binding, UsdShade.Tokens.weakerThanDescendants
            )
            self.objects[name] = body
            self.label(
                body.prim_path + "/Identity",
                symbol.upper(),
                [0, 0, scene["piece_height"] / 2 + 0.0002],
                [0.017, 0.017],
                white,
            )
        self.buttons, self.button_drives, self.button_rest = {}, {}, {}
        for name, offset in (("robot", -0.046), ("opponent", 0.046)):
            pos = clock + [offset, 0, 0.075]
            body = self.world.scene.add(
                DynamicCylinder(
                    f"/World/Buttons/{name}",
                    name=f"button_{name}",
                    position=pos,
                    radius=0.016,
                    height=0.008,
                    mass=0.04,
                    color=np.array([0.20, 0.75, 0.40] if name == "robot" else [0.90, 0.40, 0.18]),
                )
            )
            PhysxSchema.PhysxRigidBodyAPI.Apply(body.prim).CreateDisableGravityAttr(True)
            button_collision = PhysxSchema.PhysxCollisionAPI.Apply(body.prim)
            button_collision.CreateContactOffsetAttr(0.001)
            button_collision.CreateRestOffsetAttr(0.0)
            joint = UsdPhysics.PrismaticJoint.Define(self.stage, f"/World/Joints/{name}")
            joint.CreateBody1Rel().SetTargets([body.prim_path])
            joint.CreateAxisAttr("Z")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos)))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
            joint.CreateLowerLimitAttr(-c["physics"]["button_travel"])
            joint.CreateUpperLimitAttr(0.0)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(180.0 if name == "robot" else 2000.0)
            drive.CreateDampingAttr(8.0)
            drive.CreateTargetPositionAttr(0.0)
            self.buttons[name], self.button_drives[name], self.button_rest[name] = body, drive, pos
        self.indicator = PreviewSurface(
            "/World/Materials/Turn", color=np.array([0.9, 0.25, 0.08]), roughness=0.4
        )
        self.world.scene.add(
            VisualCuboid(
                "/World/TurnIndicator",
                name="turn_indicator",
                position=clock + [0, -0.032, 0.052],
                scale=np.array([0.10, 0.002, 0.012]),
                visual_material=self.indicator,
            )
        )
        self.label(
            "/World/ClockLabel",
            "YOUR CLOCK  |  OPPONENT",
            clock + [0, -0.038, 0.002],
            [0.23, 0.028],
            True,
        )
        for f in range(8):
            p = task.position(chr(97 + f) + "1")
            p[0] -= 0.039
            p[2] = h + scene["board_height"] + 0.0003
            self.label(f"/World/File{f}", chr(65 + f), p, [0.017, 0.017], True)
        for rank in range(1, 9):
            p = task.position("h" + str(rank))
            p[1] -= 0.039
            p[2] = h + scene["board_height"] + 0.0003
            self.label(f"/World/Rank{rank}", str(rank), p, [0.017, 0.017], True)
        self.customize_scene()
        dome = UsdLux.DomeLight.Define(self.stage, "/World/Lighting/Dome")
        dome.CreateTextureFileAttr(str(resolve_asset("shared://hdr/studio_small_09_2k.hdr")))
        dome.CreateIntensityAttr(700)
        for i, (pos, intensity) in enumerate((([0.3, -0.5, 2.0], 1300), ([0.8, 0.5, 1.8], 900))):
            light = UsdLux.RectLight.Define(self.stage, f"/World/Lighting/Area{i}")
            light.AddTranslateOp().Set(Gf.Vec3d(*pos))
            light.CreateWidthAttr(0.8)
            light.CreateHeightAttr(0.6)
            light.CreateIntensityAttr(intensity)
        self.names = list(self.objects) + ["robot", "opponent"]
        self.contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[b.prim_path for b in self.objects.values()]
                + [b.prim_path for b in self.buttons.values()],
                name="chess_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    self.finger_paths
                    + [
                        self.hand_path,
                        board.prim_path,
                        capture.prim_path,
                    ]
                    for _ in self.names
                ],
                max_contact_count=8192,
            )
        )
        self.arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=self.arm_paths,
                name="chess_arm_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    ["/World/Table", "/World/BoardSupport", "/World/ClockHousing"]
                    + [body.prim_path for body in self.objects.values()]
                    for _ in self.arm_paths
                ],
                max_contact_count=8192,
            )
        )
        cp = np.array(scene["camera_position"])
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*cp), Gf.Vec3d(*scene["camera_look_at"]), Gf.Vec3d(0, 0, 1)
        )
        orientation = gf_quat_to_np_array(view.GetInverse().ExtractRotationQuat())
        self.camera = Camera(
            "/World/Camera",
            position=cp,
            orientation=orientation,
            resolution=tuple(c["sim"]["camera_resolution"]),
        )
        self.world.reset()
        self.initialize_robot()
        self.camera.initialize()
        self.camera.add_distance_to_image_plane_to_frame()
        self.camera.set_world_pose(cp, orientation, camera_axes="usd")
        self.camera.set_focal_length(0.022)
        self.camera.set_horizontal_aperture(0.036)
        self.camera.set_clipping_range(0.01, 10.0)
        for _ in range(c["sim"]["settle_steps"]):
            self.rig.apply()
            self.world.step(render=False)
        self.previous_tool = None
        self.last_arm_collision = False
        self.collision_count = 0
        self._state = self.measure(0.0)
        self.world.render()

    def initialize_robot(self):
        from roboclock.simulators.isaac.embodiments.single import initialize

        initialize(self)

    def customize_scene(self):
        """Optional versioned scene extension, before PhysX views initialize."""

    def tick(self, render=False):
        if self.task.time >= self.task.spec.opponent_press_time:
            self.button_drives["opponent"].GetTargetPositionAttr().Set(
                -self.config["physics"]["button_travel"]
            )
        self.world.step(render=render)
        arm_forces = self.arm_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        collision = bool(np.any(np.linalg.norm(arm_forces, axis=-1) > 0.5))
        if collision and not self.last_arm_collision:
            self.collision_count += 1
        self.last_arm_collision = collision
        self._state = self.measure(self.config["sim"]["physics_dt"])
        if self._state["opponent_pressed"]:
            self.indicator.set_color(np.array([0.08, 0.85, 0.32]))

    def measure(self, dt):
        threshold = self.config["physics"]["contact_force"]
        forces = self.contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        rows = {p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.contacts.prim_paths)}
        ee, orientation = self.tool_pose()
        tool_v = (
            np.zeros(3) if self.previous_tool is None or dt == 0 else (ee - self.previous_tool) / dt
        )
        self.previous_tool = ee.copy()
        from roboclock.simulators.isaac.embodiments.single import width

        aperture = width(self)
        objects = {}
        for name, body in self.objects.items():
            p, r = body.get_world_pose()
            v = body.get_linear_velocity()
            f = forces[rows[name]]
            grasped, _ = self.grasps.update(
                name,
                f[:2],
                aperture,
                p,
                ee,
                v,
                tool_v,
                self.config["scene"]["piece_radius"],
                self.config["scene"]["piece_height"],
                dt,
            )
            support_index = 4 if self.task.expected[name] == "capture" else 3
            objects[name] = {
                "position": p.copy(),
                "orientation": r.copy(),
                "velocity": v.copy(),
                "angular_velocity": body.get_angular_velocity().copy(),
                "grasped": grasped,
                "robot_contact": bool(np.any(np.linalg.norm(f[:3], axis=-1) > threshold)),
                "supported": bool(np.linalg.norm(f[support_index]) > threshold),
            }
        pressed = {}
        opponent_robot_contact = False
        travel = {}
        for name, body in self.buttons.items():
            p, _ = body.get_world_pose()
            travel[name] = float(self.button_rest[name][2] - p[2])
            contact = bool(np.any(np.linalg.norm(forces[rows[name], :3], axis=-1) > threshold))
            if name == "opponent":
                opponent_robot_contact = contact
            pressed[name] = travel[name] >= self.config["physics"]["button_threshold"] and (
                contact or name == "opponent"
            )
        return {
            "objects": objects,
            "tool_position": ee,
            "tool_orientation": orientation,
            "robot_pressed": pressed["robot"],
            "opponent_pressed": pressed["opponent"],
            "button_travel": travel,
            "opponent_robot_contact": opponent_robot_contact,
        }
