"""Two native Dual ARX X5 platforms (white and black) with independent finger/contact measurements."""

import numpy as np

from roboclock.envs.sports.arcade.chess_backend import RoomChessBackend
from roboclock.physics.contacts import GraspTracker


class DuelBackend(RoomChessBackend):
    def customize_scene(self):
        super().customize_scene()
        from isaacsim.core.prims import RigidPrim
        from pxr import Gf, UsdGeom

        from roboclock.envs.lab.appearance import LabAppearance
        from roboclock.envs.sports.arcade.scenes import imported

        h = self.config["scene"]["table_height"]
        from roboclock.simulators.isaac.embodiments.rig import RobotRig

        self.opponent_rig = RobotRig(
            self.world,
            self.stage,
            self.config["embodiment"]["model"],
            [
                0.96 - self.config["embodiment"]["position"][0],
                0,
                self.config["embodiment"]["position"][2],
            ],
            yaw=np.pi,
            path="/World/Opponent",
        )
        self.opponent_rig.x5_finger_force = self.config["embodiment"].get("finger_force_n", 30)
        self.opponent = self.opponent_rig.robot
        self.opponent_active_side = "left"
        black_fingers = self.opponent_rig.fingers[self.opponent_active_side]
        black_hand = self.opponent_rig.tools[self.opponent_active_side].prim_path
        black_arms = self.opponent_rig.arm_paths
        self.black_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=[o.prim_path for o in self.objects.values()]
                + [b.prim_path for b in self.buttons.values()],
                name="black_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    black_fingers
                    + [
                        black_hand,
                        "/World/BoardSupport",
                        "/World/CaptureTray",
                    ]
                    for _ in range(len(self.objects) + 2)
                ],
                max_contact_count=8192,
            )
        )
        self.black_arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=black_arms,
                name="black_arm_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    ["/World/Table", "/World/BoardSupport", "/World/ClockHousing"]
                    + [o.prim_path for o in self.objects.values()]
                    for _ in black_arms
                ],
                max_contact_count=8192,
            )
        )
        self.black_grasps = GraspTracker(
            minimum_time=0.04, force_threshold=self.config["physics"]["contact_force"]
        )
        self.previous_black = None
        self.black_collision_count = 0
        self.black_last_collision = False
        self.inter_arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=self.arm_paths + [self.hand_path] + self.finger_paths,
                name="inter_arm_contacts",
                track_contact_forces=True,
                reset_xform_properties=False,
                contact_filter_prim_paths_expr=[
                    black_arms + [black_hand] + black_fingers
                    for _ in self.arm_paths + [self.hand_path] + self.finger_paths
                ],
                max_contact_count=8192,
            )
        )
        self.inter_arm_collision_steps = 0
        self.button_drives["opponent"].GetStiffnessAttr().Set(180)
        table_width = 1.18
        table = UsdGeom.Xformable(self.stage.GetPrimAtPath("/World/Table"))
        for op in table.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(0.48, 0, h - 0.025))
            if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                op.Set(Gf.Vec3d(table_width, 1.05, 0.05))
        self.stage.RemovePrim("/World/Room/PlayingTable")
        imported(
            self.stage,
            "side_table_01",
            "/World/Room/PlayingTable",
            [0.48, 0, 0],
            [table_width, 1.05, 0.75],
        )
        self.appearance = LabAppearance(self.stage, self.world, self.config)
        # The wider two-platform view needs a complete room envelope.
        self.appearance.material("duel_wall", color=[0.15, 0.19, 0.19], roughness=0.85)
        self.appearance.rounded(
            "/World/Room/RightWall", [3.2, 0, 1.6], [0.08, 3.8, 3.2], "duel_wall"
        )
        self.appearance.material("duel_case", color=[0.018, 0.025, 0.035], roughness=0.65)
        self.appearance.material("duel_screen", color=[0.025, 0.045, 0.065], roughness=0.85)
        self.appearance.material(
            "duel_brass", color=[0.52, 0.32, 0.12], roughness=0.30, metallic=0.78
        )
        # Dedicated names preserve the existing furniture materials.
        self.appearance.rounded(
            "/World/Duel/TurnPanel/Case",
            [0.48, 0.655, 1.39],
            [0.73, 0.045, 0.28],
            "duel_case",
            0.012,
        )
        self.appearance.rounded(
            "/World/Duel/TurnPanel/Face",
            [0.48, 0.629, 1.39],
            [0.70, 0.008, 0.25],
            "duel_screen",
            0.006,
        )
        for i, z in enumerate([1.273, 1.507]):
            self.appearance.rounded(
                f"/World/Duel/TurnPanel/Trim{i}",
                [0.48, 0.622, z],
                [0.68, 0.004, 0.003],
                "duel_brass",
            )
        self.label(
            "/World/ClockLabel",
            "WHITE  |  BLACK",
            np.asarray(self.config["scene"]["clock_position"]) + [0, -0.038, 0.002],
            [0.23, 0.028],
            True,
        )

    def reset(self, task):
        super().reset(task)
        from scipy.spatial.transform import Rotation

        rotation = Rotation.from_euler("xyz", [0, np.pi, np.pi]).as_matrix()
        targets = {
            "left": (np.array([0.98, -0.48, 1.12]), rotation, 0.04),
            "right": (np.array([0.66, 0.27, 1.06]), rotation, 0.04),
        }
        from roboclock.simulators.isaac.embodiments.views import observation_pose

        c = self.config["embodiment"]
        targets["left"] = (np.array([0.66, -0.27, 1.06]), rotation, 0.025)
        targets["right"] = observation_pose(
            [0.96-c["position"][0], 0, c["position"][2]], np.pi, side="right"
        )
        self.opponent_rig.initialize(targets)
        self.world.reset()
        self.rig.apply()
        self.opponent_rig.apply()
        for _ in range(120):
            self.rig.apply()
            self.opponent_rig.apply()
            self.world.step(render=False)
        self.previous_tool = self.previous_black = None
        self.states = self.measure_both(0)

    def black_tool(self):
        return self.opponent_rig.pose(self.opponent_active_side)

    def measure_both(self, dt):
        white = self.measure(dt)
        ee, qtool = self.black_tool()
        velocity = (
            np.zeros(3)
            if self.previous_black is None or dt == 0
            else (ee - self.previous_black) / dt
        )
        self.previous_black = ee.copy()
        forces = self.black_contacts.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
        rows = {p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.black_contacts.prim_paths)}
        threshold = self.config["physics"]["contact_force"]
        objects = {}
        for name, o in white["objects"].items():
            f = forces[rows[name]]
            grasped, _ = self.black_grasps.update(
                name,
                f[:2],
                self.opponent_rig.width(self.opponent_active_side),
                o["position"],
                ee,
                o["velocity"],
                velocity,
                self.config["scene"]["piece_radius"],
                self.config["scene"]["piece_height"],
                dt,
            )
            black_contact = bool(np.any(np.linalg.norm(f[:3], axis=-1) > threshold))
            white["objects"][name]["robot_contact"] |= black_contact
            objects[name] = {**o, "grasped": grasped}
        pressed = {}
        for name in self.buttons:
            pressed[name] = bool(
                white["button_travel"][name] >= self.config["physics"]["button_threshold"]
                and np.any(np.linalg.norm(forces[rows[name], :3], axis=-1) > threshold)
            )
        black = dict(
            objects=objects,
            tool_position=ee,
            tool_orientation=qtool,
            robot_pressed=pressed["opponent"],
            opponent_pressed=white["robot_pressed"],
            button_travel=white["button_travel"],
            opponent_robot_contact=bool(
                np.any(np.linalg.norm(forces[rows["robot"], :3], axis=-1) > threshold)
            ),
        )
        white["opponent_pressed"] = pressed["opponent"]
        return dict(white=white, black=black)

    def tick(self, render=False):
        # Both spring targets stay at rest: only robot contact can press either clock.
        self.world.step(render=render)
        self.states = self.measure_both(self.config["sim"]["physics_dt"])
        self._state = self.states["white"]
        inter_force = self.inter_arm_contacts.get_contact_force_matrix(
            dt=self.config["sim"]["physics_dt"]
        )
        inter_collision = np.any(np.linalg.norm(inter_force, axis=-1) > 0.5)
        if inter_collision and self.inter_arm_collision_steps == 0:
            pairs = np.argwhere(np.linalg.norm(inter_force, axis=-1) > 0.5)
            print("FIRST_INTER_ARM_CONTACT", self.task.time, pairs.tolist(), flush=True)
        self.inter_arm_collision_steps += int(inter_collision)
        for side, view in [("white", self.arm_contacts), ("black", self.black_arm_contacts)]:
            f = view.get_contact_force_matrix(dt=self.config["sim"]["physics_dt"])
            collision = bool(np.any(np.linalg.norm(f, axis=-1) > 0.5))
            if side == "white":
                self.collision_count += int(collision and not self.last_arm_collision)
                self.last_arm_collision = collision
            else:
                self.black_collision_count += int(collision and not self.black_last_collision)
                self.black_last_collision = collision
