"""Native Dual ARX X5 articulation (the only robot in roboclock); no invisible proxy joints."""

import numpy as np
from scipy.spatial.transform import Rotation

from .assets import asset_path
from .kinematics import USDKinematics


def joint_gravity_forces(generalized, joint_count):
    """PhysX floating articulations prepend a six-component base wrench."""
    forces = np.asarray(generalized).reshape(-1)
    if len(forces) == joint_count + 6:
        return forces[6:]
    if len(forces) != joint_count:
        raise ValueError(
            f"Unexpected generalized gravity shape: {forces.shape}, joints={joint_count}"
        )
    return forces


class RobotRig:
    def __init__(self, world, stage, model, position, yaw=0, path="/World/Robot", boxing=False):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

        if model != "dual_arx_x5":
            raise ValueError(f"Unsupported robot {model!r}: roboclock uses only dual_arx_x5")
        self.model, self.path = model, path
        self.stage = stage
        prim = add_reference_to_stage(str(asset_path(model)), path)
        from .arx_mount import build_mount, close_room

        build_mount(world, stage, path + "Mount", position, yaw)
        close_room(stage)
        from .variants import configure_variant

        self.sides = configure_variant(stage, prim, model)
        transform = UsdGeom.Xformable(prim)
        transform.ClearXformOpOrder()
        transform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
        transform.AddRotateZOp().Set(float(np.rad2deg(yaw)))
        native = list(Usd.PrimRange(prim))
        self.rigid = {
            p.GetName(): str(p.GetPath()) for p in native if p.HasAPI(UsdPhysics.RigidBodyAPI)
        }
        self.joints = {p.GetName(): str(p.GetPath()) for p in native if p.IsA(UsdPhysics.Joint)}
        # The fixed-base manufacturer models already contain a world constraint.
        # A mobile-shaped variant is anchored here rather than driven kinematically.
        anchors = [
            UsdPhysics.Joint(p)
            for p in native
            if p.IsA(UsdPhysics.Joint) and not UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
        ]
        if not anchors:
            root = self.rigid.get("base_link")
            if root is None:
                raise ValueError("No identifiable base body for anchoring")
            anchor = UsdPhysics.FixedJoint.Define(stage, path + "/BenchmarkBaseAnchor")
            anchor.GetBody1Rel().SetTargets([root])
            anchors = [anchor]
        cache = UsdGeom.XformCache()
        for anchor in anchors:
            child = stage.GetPrimAtPath(anchor.GetBody1Rel().GetTargets()[0])
            # Keep the source anchor's local child frame and compose its world pose.
            child_matrix = cache.GetLocalToWorldTransform(child)
            local = Gf.Matrix4d().SetRotate(Gf.Rotation(anchor.GetLocalRot1Attr().Get()))
            local.SetTranslateOnly(Gf.Vec3d(anchor.GetLocalPos1Attr().Get()))
            matrix = local * child_matrix
            anchor.GetLocalPos0Attr().Set(Gf.Vec3f(matrix.ExtractTranslation()))
            q = matrix.ExtractRotationQuat()
            anchor.GetLocalRot0Attr().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
        for p in native:
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(False)
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                api = PhysxSchema.PhysxArticulationAPI.Apply(p)
                api.CreateEnabledSelfCollisionsAttr(True)
                api.CreateSolverPositionIterationCountAttr(32)
                api.CreateSolverVelocityIterationCountAttr(8)
        # Bodies connected through fixed joints form one mechanical assembly.
        # Do not resolve contacts inside that assembly (including adapter plates).
        groups = {p: {p} for p in self.rigid.values()}
        for prim_joint in native:
            if prim_joint.IsA(UsdPhysics.FixedJoint):
                joint = UsdPhysics.Joint(prim_joint)
                a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if a and b and str(a[0]) in groups and str(b[0]) in groups:
                    merged = groups[str(a[0])] | groups[str(b[0])]
                    for member in merged:
                        groups[member] = merged
        filtered = {p: set(group) - {p} for p, group in groups.items()}
        self.collision_filters = {p: sorted(others) for p, others in filtered.items() if others}
        for p, others in self.collision_filters.items():
            UsdPhysics.FilteredPairsAPI.Apply(
                stage.GetPrimAtPath(p)
            ).CreateFilteredPairsRel().SetTargets(others)
        self.robot = world.scene.add(SingleArticulation(path, name=path.rsplit("/", 1)[-1]))
        self.kin, self.tools, self.fingers, self.arm_names, self.grip_names = {}, {}, {}, {}, {}
        self.offsets, self.arm_paths = {}, []
        self.targets = {}
        # Native X5 advances along local +X; canonical grasp frame advances +Z.
        self.tcp_rotation = Rotation.from_euler("y", 90, degrees=True).as_matrix()
        for side in self.sides:
            arms = [f"{side}_joint{i}" for i in range(1, 7)]
            tool = f"{side}_link6"
            fingers = [f"{side}_link{i}" for i in (7, 8)]
            grips = [f"{side}_joint{i}" for i in (7, 8)]
            offset = [0.17 if boxing else 0.14, 0, 0]
            self.arm_names[side], self.grip_names[side] = arms, grips
            self.offsets[side] = np.asarray(offset)
            self.fingers[side] = [self.rigid[n] for n in fingers]
            self.kin[side] = USDKinematics(
                stage, [self.joints[n] for n in arms], self.rigid[tool], offset, self.tcp_rotation
            )
            self.tools[side] = world.scene.add(
                SingleRigidPrim(
                    self.rigid[tool],
                    name=path.rsplit("/", 1)[-1] + "_" + side,
                    reset_xform_properties=False,
                )
            )
            self.arm_paths.extend(
                str(
                    UsdPhysics.Joint(stage.GetPrimAtPath(self.joints[n]))
                    .GetBody1Rel()
                    .GetTargets()[0]
                )
                for n in arms[:-1]
            )

    def initialize(self, targets, zero_arms=False):
        """Initialize poses only at reset, then use bounded articulation drives.

        ``zero_arms`` (robot.start_pose: zero, roboclock.envs.start_pose): the arm joints of
        every side in ``targets`` start at 0 rad instead of the IK solution for its pose; the
        gripper still takes its target width. Cartesian moves then start from the measured
        tool pose, and ``initial_ik`` holds None for those sides.
        """
        self.names = list(self.robot.dof_names)
        self.arm_ids = {
            s: [self.names.index(n) for n in names] for s, names in self.arm_names.items()
        }
        self.grip_ids = {
            s: [self.names.index(n) for n in names if n in self.names]
            for s, names in self.grip_names.items()
        }
        self.command = self.robot.get_joint_positions().copy()
        self.initial_ik = {}
        for side, (p, r, grip) in targets.items():
            if side not in self.sides:
                continue
            if zero_arms:
                self.targets.pop(side, None)
                self.initial_ik[side] = None
                self.command[self.arm_ids[side]] = 0.0
            else:
                self.targets[side] = np.asarray(p).copy()
                q, error = self.kin[side].solve(np.asarray(p), r, multistart=True)
                self.initial_ik[side] = error
                self.command[self.arm_ids[side]] = q
            self.grip(side, grip)
        # Retain full native DOF ordering, including coupled gripper joints.
        gains = np.full(len(self.names), 5000.0)
        damping = np.full(len(self.names), 100.0)
        efforts = np.full(len(self.names), 87.0)
        for ids in self.arm_ids.values():
            gains[ids], damping[ids] = 1800, 65
            efforts[ids] = [27, 27, 27, 7, 7, 7]
        for ids in self.grip_ids.values():
            gains[ids], damping[ids], efforts[ids] = 1800, 35, getattr(self, "x5_finger_force", 30)
        self.actuated_names = [name for name, effort in zip(self.names, efforts) if effort > 0]
        self.kps = gains.copy()
        controller = self.robot.get_articulation_controller()
        controller.set_gains(kps=gains, kds=damping)
        controller.set_max_efforts(efforts)
        self.robot.set_joints_default_state(positions=self.command.copy())
        self.robot.set_joint_positions(self.command)
        self.apply()

    def set_arm_command(self, side, joints):
        """Update one arm while keeping the parked arm at its commanded pose."""
        ids = self.arm_ids[side]
        self.command[ids] = np.asarray(joints)[ids]

    def grip(self, side, half_width):
        # Manufacturer finger tips meet at native zero (RoboTwin added 0.8 mm).
        self.command[self.grip_ids[side]] = np.clip(half_width, 0, 0.044)

    def pose(self, side):
        p, q = self.tools[side].get_world_pose()
        r = Rotation.from_quat(np.roll(q, -1))
        grasp_q = Rotation.from_matrix(r.as_matrix() @ self.tcp_rotation).as_quat()
        return p + r.apply(self.offsets[side]), np.roll(grasp_q, 1)

    def width(self, side):
        joints = self.robot.get_joint_positions()
        return float(np.sum(joints[self.grip_ids[side]]))

    def move(self, side, position, rotation, grip, dt, speed=0.25, joint_speed=1.7):
        p, _ = self.pose(side)
        previous = self.targets.get(side, p)
        delta = np.asarray(position) - previous
        desired = previous + delta * min(1, speed * dt / max(1e-9, np.linalg.norm(delta)))
        self.targets[side] = desired.copy()
        actual = self.robot.get_joint_positions()
        ids = self.arm_ids[side]
        q, error = self.kin[side].solve(desired, rotation, seed=actual[ids])
        delta = q - actual[ids]
        if getattr(self, "synchronized_joint_steps", False):
            delta *= min(1.0, joint_speed * dt / max(1e-9, np.max(np.abs(delta))))
        else:
            delta = np.clip(delta, -joint_speed * dt, joint_speed * dt)
        self.command[ids] = actual[ids] + delta
        self.grip(side, grip)
        return float(np.linalg.norm(np.asarray(position) - p)), error

    def apply(self):
        from isaacsim.core.utils.types import ArticulationAction

        # Model-based gravity feed-forward through the position drives. Native
        # force limits still bound the total drive torque; object gravity remains.
        gravity = joint_gravity_forces(
            np.asarray(
                self.robot._articulation_view._physics_view.get_gravity_compensation_forces()
            )[0],
            len(self.names),
        )
        target = self.command.copy()
        ids = [i for indices in self.arm_ids.values() for i in indices]
        target[ids] += gravity[ids] / self.kps[ids]
        limits = self.robot.dof_properties
        target = np.clip(target, limits["lower"], limits["upper"])
        self.robot.apply_action(ArticulationAction(joint_positions=target))
