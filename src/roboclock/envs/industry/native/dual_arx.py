"""Complete, fixed-mount Dual ARX-X5 candidate with bounded native drives.

Two complete source articulations, without rescaling/replacing grippers.
No product attachment, task state or task-success predicate lives here.

Merged from the four recorded variants. ``mount_yaw_deg`` (Tasks 46-49) and the
link-transform cache (``native_readback_cache_enabled``, Tasks 47/48) are opt-in and
leave the other tasks unchanged. ``gripper_velocity_cap`` sets the slider joints'
PhysX max velocity to 95% of the declared gripper speed: Tasks 46-50 were recorded
with it, Tasks 41-45 without (their sliders kept the imported limit), so it changes
gripper dynamics and stays a per-task setting.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .paths import portable_json_loads


class DualARX:

    def __init__(self, world, manifest, *, mounts=None, mount_yaw_deg=None, gripper_velocity_cap=True):
        from isaacsim.core.prims import SingleArticulation
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        self.world = world
        self.manifest = portable_json_loads(Path(manifest).read_text())
        self.mounts = mounts or dict(left=[-0.43, 0.08, 1.02], right=[0, 0.08, 1.02])
        self.rotation = Rotation.from_euler('z', np.pi / 2).as_matrix()
        self.rotations = {side: Rotation.from_euler('z', mount_yaw_deg[side], degrees=True).as_matrix() if mount_yaw_deg is not None else self.rotation.copy() for side in ('left', 'right')}
        self.robots = {}
        self.commands = {}
        self.names = {}
        self.linear_velocity_overrides = []
        for side in ('left', 'right'):
            path = '/World/ARX/' + side
            root = UsdGeom.Xform.Define(world.stage, path)
            root.GetPrim().GetReferences().AddReference(self.manifest['root_usd'])
            root.AddTranslateOp().Set(Gf.Vec3d(*self.mounts[side]))
            if mount_yaw_deg is None:
                root.AddOrientOp().Set(Gf.Quatf(float(np.sqrt(0.5)), 0, 0, float(np.sqrt(0.5))))
            else:
                quat = Rotation.from_matrix(self.rotations[side]).as_quat()
                root.AddOrientOp().Set(Gf.Quatf(float(quat[3]), *map(float, quat[:3])))
            for prim in Usd.PrimRange(root.GetPrim()):
                if gripper_velocity_cap and prim.IsA(UsdPhysics.PrismaticJoint):
                    cap = float(self.manifest['profile']['gripper_speed_m_s'])
                    configured = cap * 0.95
                    native = PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr()
                    imported = native.Get()
                    native.Set(configured)
                    self.linear_velocity_overrides.append(dict(path=str(prim.GetPath()), imported=imported, declared_m_s=cap, configured_m_s=configured))
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0)
            base = world.stage.GetPrimAtPath(path + '/' + self.manifest['relative_body_paths']['base_link'])
            if not base:
                raise RuntimeError('Missing complete ARX base')
            frame = UsdGeom.XformCache().GetLocalToWorldTransform(base)
            anchor = UsdPhysics.FixedJoint.Define(world.stage, path + '/WorldMount')
            anchor.CreateBody1Rel().SetTargets([base.GetPath()])
            anchor.CreateLocalPos0Attr(Gf.Vec3f(*frame.ExtractTranslation()))
            anchor.CreateLocalRot0Attr(Gf.Quatf(frame.ExtractRotationQuat()))
            anchor.CreateLocalPos1Attr(Gf.Vec3f(0))
            anchor.CreateLocalRot1Attr(Gf.Quatf(1))
            UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
            props = PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim())
            props.CreateEnabledSelfCollisionsAttr(True)
            props.CreateSolverPositionIterationCountAttr(64)
            props.CreateSolverVelocityIterationCountAttr(16)
            robot = SingleArticulation(path, name='arx_' + side)
            self.robots[side] = world.scene.add(robot)

    def initialize(self):
        """Call after world.reset; initial conditions only, never mid-episode."""
        for side, robot in self.robots.items():
            names = list(robot.dof_names)
            if set(names) != {f'joint{i}' for i in range(1, 9)}:
                raise RuntimeError('Unexpected ARX articulation DOFs: ' + str(names))
            self.names[side] = names
            q = {f'joint{i + 1}': v for i, v in enumerate([1.0 if side == 'left' else -0.8, 1.3789, 1.242, -1.4339, 0, 1.0 if side == 'left' else -0.8, 0.04, 0.04])}
            target = np.asarray([q[n] for n in names])
            robot.set_joints_default_state(positions=target, velocities=np.zeros(8))
            robot.set_joint_positions(target)
            robot.set_joint_velocities(np.zeros(8))
            self.commands[side] = target.copy()
            kp = np.asarray([700 if int(n[5:]) <= 6 else 1500 for n in names])
            kd = np.asarray([70 if int(n[5:]) <= 6 else 80 for n in names])
            robot.get_articulation_controller().set_gains(kps=kp, kds=kd)

    def apply(self):
        from isaacsim.core.utils.types import ArticulationAction
        for side, robot in self.robots.items():
            robot.apply_action(ArticulationAction(joint_positions=self.commands[side]))

    def jaw_travel(self, side, travel):
        if not 0 <= travel <= 0.044:
            raise ValueError('ARX native slider travel outside source bounds')
        for name in ('joint7', 'joint8'):
            self.commands[side][self.names[side].index(name)] = travel

    def body_world(self, side, name):
        robot = self.robots[side]
        i = robot._articulation_view.body_names.index(name)
        now = float(self.world.current_time)
        if not getattr(self, 'native_readback_cache_enabled', False):
            transforms = np.asarray(robot._articulation_view._physics_view.get_link_transforms())
        else:
            if getattr(self, '_link_cache_time', None) != now:
                self._link_cache_time, self._link_cache = (now, {})
            if side not in self._link_cache:
                self._link_cache[side] = np.asarray(robot._articulation_view._physics_view.get_link_transforms()).copy()
            transforms = self._link_cache[side]
            checks = getattr(self, '_link_cache_checked', {})
            stamp = int(now)
            if checks.get(side) != stamp:
                fresh = np.asarray(robot._articulation_view._physics_view.get_link_transforms())
                assert np.array_equal(fresh, transforms), 'Native link pose changed without clock advancement'
                checks[side] = stamp
                self._link_cache_checked = checks
        pose = transforms[0, i]
        return (pose[:3].copy(), Rotation.from_quat(pose[3:7]).as_matrix())

    def tool_world(self, side):
        p, R = self.body_world(side, 'link6')
        return (p + R @ np.asarray(self.manifest['tool']['offset_m']), R)

    def observation(self, side):
        robot = self.robots[side]
        p, R = self.tool_world(side)
        jaw1, _ = self.body_world(side, 'link7')
        jaw2, _ = self.body_world(side, 'link8')
        return dict(joint_names=self.names[side], q=robot.get_joint_positions().tolist(), qd=robot.get_joint_velocities().tolist(), tool_position=p.tolist(), tool_rotation=R.tolist(), jaw_body_distance_m=float(np.linalg.norm(jaw1 - jaw2)))
