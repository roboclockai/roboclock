"""Task-compatible native replacement backend; temporal contracts stay unchanged."""

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.sports._shared.workflow.backend import Backend as MeasurementBackend
from roboclock.simulators.isaac.embodiments.rig import RobotRig


class ReplacementBackend(MeasurementBackend):
    def __init__(self, runtime, config):
        super().__init__(runtime, config)
        if config["workflow"].get("fixed_service") or config["workflow"].get("fixed_freeze"):
            self.mobile = False
        if self.mobile:
            raise ValueError("Mobile tasks are excluded from this migration")
        self.is_bench = config["workflow"].get("bench_press", False)
        self.single_arm = self.kind in ("A3", "C1")
        self.is_racking = config["workflow"].get("pool_racking", False)
        if self.is_racking:
            self.single_arm = False
        self.human_effort = None
        self.mount = config["workflow"].get("bench_mount", 0)
        self.tcp_offset = 0.17 if self.kind == "B2" else 0.075

    def reset(self):
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.utils.stage import create_new_stage, get_current_stage

        from roboclock.simulators.isaac.rendering import configure_renderer

        World.clear_instance()
        create_new_stage()
        self.world = World(
            stage_units_in_meters=1,
            physics_dt=self.config["sim"]["physics_dt"],
            rendering_dt=self.config["sim"]["physics_dt"],
        )
        self.stage = get_current_stage()
        configure_renderer(self.config)
        self.world.get_physics_context().enable_ccd(True)
        self.world.scene.add_default_ground_plane()
        self.material = PhysicsMaterial(
            "/World/Materials/Contact", static_friction=1, dynamic_friction=0.85, restitution=0.08
        )
        model = self.config["embodiment"]["model"]
        position = self.config["embodiment"]["position"]
        self.rig = RobotRig(
            self.world,
            self.stage,
            model,
            position,
            yaw=self.config["embodiment"].get("yaw", 0),
            boxing=self.kind == "B2",
        )
        if self.kind in ("B4", "C3"):
            self.rig.x5_finger_force = 5.0
        if self.kind == "B4":
            from pxr import PhysxSchema, Usd, UsdPhysics
            self.rig.x5_finger_force = 18.0
            for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.rig.path)):
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
                    api.CreateSolverPositionIterationCountAttr(64)
                    api.CreateSolverVelocityIterationCountAttr(32)
        self.rig.synchronized_joint_steps = self.is_bench or self.kind == "C1"
        if self.kind == "C1":
            self.rig.x5_finger_force = self.config.get("pool", {}).get("finger_force_n", 4.0)
        self.robot, self.robot_path = self.rig.robot, self.rig.path
        self.arm_path = self.robot_path
        self.kin, self.tool_links = self.rig.kin, self.rig.tools
        self.tool_paths = {s: tool.prim_path for s, tool in self.tool_links.items()}
        self.active_side = self.config["embodiment"].get("active_side", "right")
        self.pool_exchange = self.kind == "C1" and self.config.get("pool", {}).get("dual_cue_exchange", False)
        sides = (["right", "left"] if self.pool_exchange
                 else [self.active_side] if self.single_arm else ["left", "right"])
        self.fingers = [p for s in sides for p in self.rig.fingers[s]]
        if self.kind == "B4":
            from roboclock.envs.sports.freeze.scene import build_scene
        elif self.kind == "C3":
            from roboclock.envs.sports.ball_service.scene import build_scene
        elif self.kind in ("A3", "B2", "C2"):
            from roboclock.envs.sports.arena.scenes import build_scene
        else:
            from roboclock.envs.sports._shared.workflow.scenes import build_scene
        build_scene(self)
        self.support_paths = getattr(self, "support_paths", ["/World/Table"])
        self.targets = list(self.objects) + list(self.pads)
        paths = [o.prim_path for o in self.objects.values()] + [
            p.prim_path for p in self.pads.values()
        ]
        self.filter_paths = (
            self.fingers + self.support_paths + [o.prim_path for o in self.objects.values()]
        )
        if self.single_arm and not self.pool_exchange:
            self.filter_paths.insert(2, self.tool_paths[self.active_side])
        if self.kind == "B2":
            self.filter_paths = [self.tool_paths[s] for s in ("left", "right")]
        self.contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=paths,
                name="contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[self.filter_paths for _ in paths],
                max_contact_count=32768,
            )
        )
        pool_coop = self.is_racking or (self.kind == "C1" and self.config.get("pool", {}).get("spot_clearance", False))
        arm_paths = self.rig.arm_paths
        self.arm_contacts = self.world.scene.add(
            RigidPrim(
                prim_paths_expr=arm_paths,
                name="arm_contacts",
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=[
                    ["/World/Table"] + (self.support_paths if self.is_bench or self.kind in ("B4", "C3") or pool_coop else []) + paths
                    + ([p for p in arm_paths if ("/left_" in p) != ("/left_" in arm)] if self.kind in ("B4", "C3") or pool_coop else [])
                    for arm in arm_paths
                ],
                max_contact_count=32768,
            )
        )
        self.world.reset()
        targets = {}
        for side, y in (("left", 0.25), ("right", -0.25)):
            if side not in self.rig.sides:
                continue
            p = np.array([0.32, y, 1.02])
            r = Rotation.from_euler("y", 180, degrees=True).as_matrix()
            grip = 0.038
            if self.kind == "C1":
                p = np.array(
                    [0.45 if side == "left" else 0.65, -0.45 if side == "left" else -0.25, 1.08]
                )
            if self.kind == "C1" and side != self.active_side:
                p = np.array([0.3, 0.3, 1.1])
            if self.kind == "C1" and side == self.active_side and not self.is_racking:
                # Choose the grasp-side IK branch at reset, above the actual cue.
                p = np.asarray(self.cue_home) + [getattr(self, "cue_grip_x", 0.), 0, 0.20]
                yaw = self.config.get("pool", {}).get("grasp_yaw_by_side", {}).get(side, 0.)
                r = Rotation.from_euler("z", yaw, degrees=True).as_matrix() @ r
            if self.kind == "B2":
                p = np.array([0.35, np.sign(y) * 0.23, 1.06])
                r = Rotation.from_euler("y", 90, degrees=True).as_matrix()
                grip = 0
            if self.is_bench:
                from roboclock.envs.sports.bench_press.oracle import grip_rotation

                c = self.config["workflow"]
                p = np.array(c["standby_position"]) + [0, np.sign(y) * c["bench_grip_y"], 0]
                r, grip = grip_rotation(side), c["standby_grip"]
            if self.is_racking:
                p = np.asarray(self.config["rack"]["home"][side])
                r = Rotation.from_euler("y", self.config["rack"]["initial_pitch"], degrees=True).as_matrix()
                grip = self.config["rack"]["open_grip"]
            targets[side] = p, r, grip
        if self.single_arm:
            from .views import observation_pose

            targets["left"] = observation_pose(position, self.config["embodiment"].get("yaw", 0))
        self.rig.initialize(
            targets, zero_arms=self.config["embodiment"].get("start_pose") == "zero"
        )
        self.world.reset()
        if self.kind == "B4":
            # World reset recreates the PhysX articulation from USD drive limits.
            # Reapply this task's bounded efforts after that reset, and log them.
            controller = self.robot.get_articulation_controller()
            gains, damping = controller.get_gains()
            efforts = controller.get_max_efforts().copy()
            self.reset_drive_settings = dict(gains=gains.copy(), damping=damping.copy(), efforts=efforts.copy())
            for ids in self.rig.arm_ids.values():
                gains[ids], damping[ids] = 1800., 65.
                efforts[ids] = [27, 27, 27, 7, 7, 7]
            for ids in self.rig.grip_ids.values():
                gains[ids], damping[ids] = 1800., 35.
                efforts[ids] = self.config["freeze"]["finger_force"]
            controller.set_gains(kps=gains, kds=damping)
            controller.set_max_efforts(efforts)
            self.rig.kps = gains.copy()
            self.actual_drive_limits = controller.get_max_efforts().copy()
            self.actual_drive_gains = controller.get_gains()
        if self.kind == "C1":
            # Reset restores the USD gains/efforts, invalidating initialize().
            # Reapply the intended controller gains and native force limits.
            controller = self.robot.get_articulation_controller()
            gains, damping = controller.get_gains()
            efforts = controller.get_max_efforts().copy()
            self.reset_drive_settings = dict(gains=gains.copy(), damping=damping.copy(), efforts=efforts.copy())
            for ids in self.rig.arm_ids.values():
                gains[ids], damping[ids], efforts[ids] = 1800., 65., [27, 27, 27, 7, 7, 7]
            for ids in self.rig.grip_ids.values():
                gains[ids], damping[ids], efforts[ids] = 1800., 35., self.rig.x5_finger_force
            controller.set_gains(kps=gains, kds=damping)
            controller.set_max_efforts(efforts)
            self.rig.kps = gains.copy()
            self.actual_drive_limits = controller.get_max_efforts().copy()
            self.actual_drive_gains = controller.get_gains()
        self.rig.apply()
        self.names = self.rig.names
        self.command = self.rig.command
        for _ in range(120):
            self.rig.apply()
            self.world.step(render=False)
        if self.kind == "C1":
            for side in self.rig.sides:
                hp, hq = self.rig.pose(side)
                self.hand_targets[side] = (hp.copy(), Rotation.from_quat(np.roll(hq, -1)).as_matrix())
            self.hand_targets["main"] = self.hand_targets[self.active_side]
        self.contact_rows = {
            p.rsplit("/", 1)[-1]: i for i, p in enumerate(self.contacts.prim_paths)
        }
        self.camera.initialize()
        self.camera.set_world_pose(self.camera_position, self.camera_orientation, camera_axes="usd")
        self.camera.set_focal_length(0.032 if self.kind == "B2" else self.config["visuals"].get("focal_length", 0.022))
        self.camera.set_horizontal_aperture(0.036)
        self.start_sim = float(self.world.current_time)
        self.time = 0
        if self.is_bench:
            from roboclock.envs.sports.bench_press.process import HumanEffort

            self.human_effort = HumanEffort(self)
        self.last_state = self.measure()
        print(
            "EMBODIMENT", self.config["embodiment"], "INITIAL_IK", self.rig.initial_ik, flush=True
        )

    def hand_pose(self, side="main"):
        return self.rig.pose(self.active_side if side == "main" else side)

    def tool_pose(self):
        return self.hand_pose()

    def move(self, side, target, rotation=None, grip=0.038, speed=0.25):
        requested_side = side
        side = self.active_side if side == "main" else side
        if rotation is None:
            rotation = Rotation.from_euler("y", 180, degrees=True).as_matrix()
        dt = self.config["sim"]["physics_dt"] * self.config["sim"]["control_decimation"]
        angle = 0.0
        if self.kind == "C1":
            # Preserve the pool controller's bounded tool rotation and angular
            # convergence signal on the native rig.
            previous = self.hand_targets.get(
                requested_side, (None, Rotation.from_euler("y", 180, degrees=True).as_matrix())
            )[1]
            delta = Rotation.from_matrix(rotation @ previous.T).as_rotvec()
            angle = np.linalg.norm(delta)
            rotation = Rotation.from_rotvec(
                delta * min(1.0, 0.6 * dt / max(angle, 1e-9))
            ).as_matrix() @ previous
        mover = self.rig.move
        if self.kind == "C1":
            from functools import partial

            from roboclock.envs.sports._shared.precision.pool_x5 import move_cue_arm

            mover = partial(move_cue_arm, self.rig)
        error, _ = mover(
            side,
            target,
            rotation,
            grip,
            dt,
            speed,
            self.config["workflow"].get("bench_joint_speed", 1.7) if self.is_bench
            else self.config.get("basketball", {}).get("joint_speed", 1.7) if self.kind == "A3"
            else self.config.get("freeze", {}).get("joint_speed", 1.7) if self.kind == "B4"
            else self.config.get("pool", {}).get("stroke_joint_speed" if getattr(self, "cue_stroking", False) else "joint_speed", 1.7) if self.kind == "C1"
            else self.config.get("service", {}).get("joint_speed", 1.7),
        )
        self.hand_targets[side] = (np.array(target), rotation)
        self.hand_targets[requested_side] = (np.array(target), rotation)
        return max(error, 0.1 * angle)

    def tick(self):
        if self.human_effort is not None:
            self.human_effort.apply()
        self.rig.command = self.command
        self.rig.apply()
        self.world.step(render=False)
        self.time = float(self.world.current_time) - self.start_sim
        self.last_state = self.measure()
        return self.last_state

    def measure(self):
        result = super().measure()
        if getattr(self, "pool_exchange", False):
            result["active_cue_arm"] = self.active_side
            if self.active_side == "left":
                for obj in result["objects"].values():
                    obj["forces"][:4] = obj["forces"][[2, 3, 0, 1]]
        if self.kind == "C1" and self.config.get("pool", {}).get("spot_clearance"):
            for side in self.rig.sides:
                hp, hq = self.rig.pose(side)
                result["hands"][side] = dict(position=hp.copy(), orientation=hq.copy())
        if self.is_bench:
            result["finger_gaps"] = np.array([self.rig.width(s) for s in ("left", "right")])
            if self.human_effort is not None:
                result["human_actuator"] = self.human_effort.state()
        return result

    def frame(self):
        if self.is_bench:
            self.athlete.update(
                self.last_state["objects"]["bar"]["position"],
                self.human_effort is not None and self.human_effort.released,
            )
        return super().frame()
