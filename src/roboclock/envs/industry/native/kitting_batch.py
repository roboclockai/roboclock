"""Two fixed-cell kits using original imported products and bounded native drives.

Both arms independently pack their case. Native retention and support are required.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs import robodojo_rig
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ

from . import rig
from .arx_cell_task import (
    ArmMotion,
    ContactBook,
    NativeReadbackAudit,
    add_mounts,
    collide,
    remove_preview,
)
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2'))
# Common rig (rig.py). Recorded: left [-0.64, 0.2, 1.1] packed kit 0 at x=-0.28 from
# sources at x=-0.72/-0.56/-0.40; right [0.64, 0.2, 1.1] packed kit 1 at x=+0.28 (mirror).
# Two 0.545 m kit cases do not fit between bases 0.60 m apart, so the roles swap and each
# kit group (case + its three feed trays) moves 0.94 m: kit 0 is now packed by the RIGHT
# arm at x=+0.66 from sources at x=+0.22/+0.38/+0.54, kit 1 by the LEFT arm at x=-0.66.
# Each arm then sees exactly the recorded relative geometry of the arm it replaces.
MOUNTS = rig.mounts([0.0, 0.2, 1.1])
KIT_X = [0.66, -0.66]
KIT_SIDES = ['right', 'left']
NAMES = ['bearing', 'bracket', 'pin']

def resolve_kitting(seed=0, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)]
    d = dict(task_id=47, embodiment='dual_arx_x5', seed=seed, source_xy_jitter_m=streams[0].uniform(-0.001, 0.001, (3, 2)).tolist() if layout else np.zeros((3, 2)).tolist(), start_s=float(streams[1].uniform(1, 2)) if timing else 1.0, mass_scale=float(streams[2].uniform(0.9, 1.1)) if physical else 1.0, budget_s=90.0, physics_hz=robodojo_rig.PHYSICS_HZ, runtime_joint_velocity_limit_rad_s=0.73, kit_floor_m=1.0475, placement_stability=dict(dwell_s=0.3, position_span_m=0.0002, orientation_from_first_deg=0.2, scope='Continuous measured native pose and original insert-floor support after full release. Raw constraint-step velocities remain diagnostics.'), kit_bounds_xy_m=[[-0.18, 0.11], [0.18, 0.33]], required_delivered_kits=2, bounds=dict(source_xy_jitter_m=[-0.001, 0.001], start_s=[1, 2], mass_scale=[0.9, 1.1]), randomization=dict(layout=layout, timing=timing, physical=physical), scope='Manipulation-only pilot: one supported three-component kit. Mobile base not selected; full delivery/throughput NOT validated.', controller='Privileged native state/contact bounded joint-drive oracle')
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d

class KittingARXScene:
    title = 'TWO MAINTENANCE KITS / FIXED WORKCELL'

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .layout_review_revisions import build_layout_review
        from .remaining_environments import TRAY
        from .v2.geometry import body_envelope_points
        self.base = build_layout_review(47, output, (1280, 800))
        c = self.base.cell
        self.world, self.stage = (c.world, c.stage)
        build_two_kit_layout(c)
        self.world.set_simulation_dt(physics_dt=1 / resolved['physics_hz'], rendering_dt=1 / CONTROL_HZ)
        self.stage.RemovePrim('/World/PlannedMount')
        for side in ['left', 'right']:
            self.stage.RemovePrim('/World/Signs/Mount_' + side)
        self.resolved, self.events, self.credited = (resolved, {}, set())
        self.physical_completed = self.success = False
        self.manipulation_success = False
        self.cue_state = 'PACKING'
        self.parts, self.envelopes = ({}, {})
        material = PhysicsMaterial('/World/KitContact', static_friction=0.95, dynamic_friction=0.85, restitution=0)
        masses = [0.35, 0.07, 0.02]
        heights = [0.02228106, 0.05, 0.016]
        self.targets = {}
        for i, name in enumerate(NAMES):
            for j in range(2):
                path = f'/World/KitProducts/{name}{j}'
                x = resolved['source_x_m'][j][i]
                source_y = resolved['source_y_m'][i]
                tray_name = f'Feed/Kit{j}_{name}'
                c.asset(tray_name, TRAY, [x, source_y, 0.905], [0.145, 0.24, 0.035], material='steel')
                c.sign(f'Feed_{j}_{name}Label', name.upper(), [x, source_y - 0.132, 0.87], [0.14, 0.035])
                p = np.array([x, source_y, 0.89785 + heights[i] / 2])
                p[:2] += resolved['source_xy_jitter_m'][i]
                file = ROOT / ('task46/prepared/bearing.usd' if i == 0 else f'task47/prepared/{name}.usda')
                root = UsdGeom.Xform.Define(self.stage, path)
                root.GetPrim().GetReferences().AddReference(str(file))
                UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
                mass = masses[i] * resolved['mass_scale']
                UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
                api = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
                api.CreateEnableCCDAttr(True)
                api.CreateAngularDampingAttr(2.0)
                api.CreateSolverPositionIterationCountAttr(64)
                api.CreateSolverVelocityIterationCountAttr(16)
                PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0)
                for mesh in Usd.PrimRange(root.GetPrim()):
                    if mesh.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(mesh)
                        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('sdf' if name == 'bracket' else 'convexHull')
                        if name == 'bracket':
                            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh).CreateSdfResolutionAttr(1024)
                        else:
                            hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh)
                            hull.CreateHullVertexLimitAttr(255)
                            hull.CreateMinThicknessAttr(0.0001)
                        col = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                        col.CreateContactOffsetAttr(0.0003)
                        col.CreateRestOffsetAttr(0)
                        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material.material, materialPurpose='physics')
                b = self.world.scene.add(SingleRigidPrim(path, name=f'kit_{name}{j}', position=p, mass=mass, orientation=np.array([np.sqrt(0.5), np.sqrt(0.5), 0, 0]) if name == 'bearing' else None))
                self.parts[name, j] = b
                self.envelopes[name, j] = body_envelope_points(self.stage, root.GetPrim())
                self.targets[name, j] = np.array([KIT_X[j] + (1 if j == 0 else -1) * (0.0 if name == 'bracket' else -0.11 + i * 0.11), 0.22, resolved['kit_floor_m'] + heights[i] / 2])
        self.asset_manifest = dict(task_id=47, original_assets='Original prepared NVIDIA bearing, CC-BY3.0 FreeCAD NEMA bracket and uniformly4x pin; see original asset manifest', source_manifest=portable_json_loads((ROOT / 'task47/asset-manifest.json').read_text()) if (ROOT / 'task47/asset-manifest.json').exists() else {}, adaptations='Approved dedicated-case layout; original arm pedestals raised120mm to reach raised insert, center bracket stocked at a collision-clear front-left position inside its original tray. Bearings stocked/packed flat by initial90degree orientation. Original products/robot geometry and scales unchanged', scope=resolved['scope'], approved_layout=self.base.metadata, case_support='Original RoboLab VOMP case_a01 static triangles, ordinary internal riser and original removable insert. Separate original lift-off lid parked; no closure or mobile-delivery claim.', static_colliders_pending=True)
        self.static_colliders = collide(self.stage, ['/World/KittingBench', '/World/Feed/', '/World/Kit', '/World/LidRest'], material)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS)
        for prim in self.stage.Traverse():
            if str(prim.GetPath()).startswith('/World/ARX/') and prim.IsA(UsdPhysics.RevoluteJoint):
                PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(float(np.rad2deg(resolved['runtime_joint_velocity_limit_rad_s'])))
        add_mounts(c, MOUNTS)
        self.static_colliders += collide(self.stage, ['/World/ARXMounts/'], material)
        for side in ['left', 'right']:
            for p in Usd.PrimRange(self.stage.GetPrimAtPath('/World/ARX/' + side)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
        self.asset_manifest['static_colliders'] = self.static_colliders
        self.asset_manifest.pop('static_colliders_pending', None)
        self.contacts = ContactBook(self.world, [b.prim_path for b in self.parts.values()], record_points=True)
        self.readback = NativeReadbackAudit(self.world, self.parts.values(), 'kit_native_readback')
        self.readback_checks = self.readback.checks
        for p in self.stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(p).CreateEnableStabilizationAttr(False)
        self.world.reset()
        self.dual.initialize()
        for b in self.parts.values():
            b.initialize()
        self.readback.initialize()
        for data in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            data.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=47)

    @property
    def detail(self):
        return f'Packed components: {len(self.credited)}/6'

    @staticmethod
    def pose(b):
        p, q = b.get_world_pose()
        return (np.asarray(p), Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix())

    def contained(self, name, j):
        b = self.parts[name, j]
        p, R = self.pose(b)
        pts = self.envelopes[name, j] @ R.T + p
        return bool(np.all(pts[:, 0] >= KIT_X[j] - 0.18) and np.all(pts[:, 0] <= KIT_X[j] + 0.18) and np.all(pts[:, 1] >= 0.11) and np.all(pts[:, 1] <= 0.33) and (pts[:, 2].min() >= self.resolved['kit_floor_m'] - 0.001) and (pts[:, 2].max() < 1.12))

    def observe(self):
        self.readback.check(float(self.world.current_time))
        parts = {}
        for (name, j), b in self.parts.items():
            p, R = self.pose(b)
            parts[f'{name}{j}'] = dict(position=p.tolist(), rotation=R.tolist(), linear_velocity=b.get_linear_velocity().tolist(), angular_velocity=b.get_angular_velocity().tolist())
            if j in [0, 1]:
                parts[f'{name}{j}'].update(whole_contained=self.contained(name, j), bounds_min_m=(self.envelopes[name, j] @ R.T + p).min(0).tolist(), bounds_max_m=(self.envelopes[name, j] @ R.T + p).max(0).tolist(), kit_touching=any((v['active'] and b.prim_path in v['actors'] and any((a.startswith(f'/World/Kit{j}/Insert/') for a in v['actors'])) for v in self.contacts.active.values())), kit_support_N=self.contacts.force(b.prim_path, f'/World/Kit{j}/Insert'), finger_forces_N={side: [self.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]] for side in ['left', 'right']})
        return dict(time_s=float(self.world.current_time), tool_position=self.dual.tool_world('left')[0].tolist(), parts=parts, credited=sorted(self.credited), cue_state=self.cue_state, unexpected_contacts=self.contacts.unexpected.copy())

    def final_valid(self, o):
        return bool(len(self.credited) == 6 and all((self.contained(n, j) for n in NAMES for j in range(2))) and (o['time_s'] <= self.resolved['budget_s'] + 0.02))

class KitArmWorker:

    def __init__(self, scene, mode, side, kit):
        self.side, self.kit = (side, kit)
        self.s, self.mode = (scene, mode)
        self.motions = {side: ArmMotion(scene.dual, side) for side in [self.side]}
        for motion in self.motions.values():
            motion.position_gain = 1.5
            motion.orientation_feedback = True
        self.phase, self.failure = ('park', None)
        self.complete = False
        self.entered, self.index, self.since = (0.0, int(scene.resolved.get('probe_component_index', 0)), None)
        self.travel = 0.044
        self.settle_samples = []

    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.events[f'kit{self.kit}_{self.index}_{phase}_s'] = self.entered
        print('PHASE', self.index, phase, self.entered, flush=True)

    def step(self, o):
        if self.complete or self.failure or self.mode in ['absent', 'passive']:
            return
        s, t = (self.s, o['time_s'])
        if t < s.resolved['start_s']:
            return
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if self.phase == 'park':
                done = []
                for side, x in [(self.side, MOUNTS[self.side][0])]:
                    done.append(self.motions[side].move([x, 0.39, 1.17], 0.12))
                    s.dual.jaw_travel(side, 0.044)
                if all(done):
                    self.transition('approach')
                return
            name = NAMES[self.index]
            side = self.side
            m = self.motions[side]
            b = s.parts[name, self.kit]
            target_R = Rotation.from_euler('z', s.resolved['bracket_tool_yaw_deg'][side] if name == 'bracket' else 90, degrees=True).as_matrix() @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
            actual_R = s.dual.tool_world(side)[1]
            delta_R = Rotation.from_matrix(target_R @ m.R.T).as_rotvec()
            if Rotation.from_matrix(m.R @ actual_R.T).magnitude() < 0.05:
                angle = np.linalg.norm(delta_R)
                m.R = Rotation.from_rotvec(delta_R * min(1.0, s.resolved['tool_rotation_rate_rad_s'] / CONTROL_HZ / max(angle, 1e-12))).as_matrix() @ m.R
            p, R = s.pose(b)
            local_grip = np.asarray(s.resolved['bracket_tool_offset_m']) if name == 'bracket' else np.array([0, 0, 0.011])
            grip = p + (local_grip if name == 'bearing' else R @ local_grip)
            opened = dict(bearing=0.044, bracket=0.035, pin=0.02)[name]
            threshold = s.resolved['grasp_minimum_per_jaw_N'][name]
            desired_kp = s.resolved['gripper_stiffness_N_m'][name]
            if getattr(self, 'active_gripper_kp', None) != desired_kp:
                robot = s.dual.robots[side]
                original = np.asarray(robot._articulation_view._physics_view.get_dof_max_forces())[0].copy()
                kp = np.array([700 if int(n[5:]) <= 6 else desired_kp for n in s.dual.names[side]])
                kd = np.array([70 if int(n[5:]) <= 6 else 80 for n in s.dual.names[side]])
                robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
                actual = np.asarray(robot._articulation_view._physics_view.get_dof_max_forces())[0].copy()
                assert np.array_equal(original, actual)
                assert all((abs(actual[s.dual.names[side].index(f'joint{f}')] - 20.0) < 1e-06 for f in [7, 8]))
                s.events[f'kit{self.kit}_{name}_gripper_gains'] = dict(time_s=t, stiffness_N_m=desired_kp, damping_N_s_m=80, native_max_efforts=actual.tolist(), original_effort_caps_unchanged=True)
                self.active_gripper_kp = desired_kp
            if self.phase in ['lift', 'clear_feed', 'raise_case', 'transfer', 'place']:
                forces = [s.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + s.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]]
                part_state = o['parts'][name + str(self.kit)]
                placed_support = self.phase == 'place' and part_state['whole_contained'] and (part_state['kit_support_N'] > 0.005 or part_state['kit_touching'])
                if min(forces) < threshold and (not placed_support):
                    self.travel = max(0.0, self.travel - 0.008 / CONTROL_HZ)
                retained = min(forces) > 0.1 or placed_support
                self.lost_since = None if retained else t if getattr(self, 'lost_since', None) is None else self.lost_since
                if self.lost_since is not None and t - self.lost_since > 0.2:
                    raise RuntimeError('Measured bilateral grasp lost during loaded motion')
            if self.phase == 'approach':
                self.travel = opened
                arrived = m.move(grip + [0, 0, 0.055], 0.12)
                if arrived and Rotation.from_matrix(target_R @ s.dual.tool_world(side)[1].T).magnitude() < 0.015:
                    self.transition('lower')
            elif self.phase == 'lower':
                if m.move(grip, 0.045):
                    self.close_goal = grip.copy()
                    self.transition('close')
            elif self.phase == 'close':
                if name == 'bracket':
                    self.close_goal = grip.copy()
                m.move(self.close_goal, 0.015 if name == 'bracket' else 0.025)
                self.travel = max(0, self.travel - 0.008 / CONTROL_HZ)
                forces = [s.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + s.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]]
                both = min(forces) > threshold
                if name == 'bracket':
                    axis = s.dual.tool_world(side)[1] @ np.array([0.0, 1.0, 0.0])
                    lateral = []
                    for finger in [7, 8]:
                        actor = '/World/ARX/' + side + '/' + s.dual.manifest['relative_body_paths'][f'link{finger}']
                        sample = s.contacts.contact_samples.get(' | '.join(sorted([b.prim_path, actor])), {})
                        lateral.append(sum((point['normal_force_N'] * abs(float(np.dot(point['normal'], axis))) for point in sample.get('points', []) if abs(float(np.dot(point['normal'], axis))) >= s.resolved['bracket_lateral_normal_cos_min'])))
                    both = both and min(lateral) > threshold
                self.since = (self.since if self.since is not None else t) if both else None
                if self.since is not None and t - self.since > 0.12:
                    tool, toolR = s.dual.tool_world(side)
                    self.offset = toolR.T @ (p - tool)
                    self.grasp_source_p = p.copy()
                    self.lost_since = None
                    s.events[f'kit{self.kit}_{name}_native_grasp'] = dict(time_s=t, product_position=p.tolist(), tool_position=tool.tolist(), tool_rotation=toolR.tolist(), contacts={str(f): s.contacts.contact_samples.get(' | '.join(sorted([b.prim_path, '/World/ARX/' + side + '/' + s.dual.manifest['relative_body_paths'][f'link{f}']]))) for f in [7, 8]})
                    self.lift = tool + [0, 0, 0.045 if name == 'bracket' else 0.075]
                    self.transition('lift')
            elif self.phase == 'lift':
                if m.move(self.lift, 0.08):
                    required = 0.045 if name == 'bracket' else 0.075
                    if p[2] - self.grasp_source_p[2] < required - 0.004:
                        raise RuntimeError('Measured product did not lift with the hand')
                    if self.mode == 'grasp':
                        s.events[f'kit{self.kit}_{name}_probe_lift_completed_s'] = t
                        self.complete = True
                        return
                    self.feed_clear = np.array([self.lift[0], 0.5, 1.08])
                    self.case_clear = np.array([self.lift[0], 0.5, 1.18])
                    self.transition('clear_feed')
            elif self.phase == 'clear_feed':
                if m.move(self.feed_clear, 0.09):
                    self.transition('raise_case')
            elif self.phase == 'raise_case':
                if m.move(self.case_clear, 0.07):
                    self.transition('transfer')
            else:
                goal = s.targets[name, self.kit] - m.R @ self.offset
                if self.phase == 'transfer':
                    if m.move(goal + [0, 0, 0.095], 0.12):
                        self.transition('place')
                elif self.phase == 'place':
                    if m.move(goal + [0, 0, 0.002], 0.045):
                        self.transition('release')
                elif self.phase == 'release':
                    m.move(goal + [0, 0, 0.002], 0.04)
                    self.travel = min(opened, self.travel + 0.035 / CONTROL_HZ)
                    if self.travel >= opened:
                        self.transition('clear_release')
                elif self.phase == 'clear_release':
                    if m.move(goal + [0, 0, 0.055], 0.08):
                        self.since = None
                        self.settle_samples = []
                        self.transition('settle')
                elif self.phase == 'settle':
                    m.move(goal + [0, 0, 0.055], 0.04)
                    part = o['parts'][name + str(self.kit)]
                    valid = part['whole_contained'] and s.resolved['kit_floor_m'] - 0.001 <= part['bounds_min_m'][2] <= s.resolved['kit_floor_m'] + 0.002 and (part['kit_support_N'] > 0.005 or part['kit_touching']) and (max((f for pair in part['finger_forces_N'].values() for f in pair)) < 0.01)
                    self.settle_samples = self.settle_samples + [(t, p.copy(), R.copy())] if valid else []
                    gate = s.resolved['placement_stability']
                    while len(self.settle_samples) > 1 and self.settle_samples[1][0] <= t - gate['dwell_s']:
                        self.settle_samples.pop(0)
                    stable = None
                    if self.settle_samples and t - self.settle_samples[0][0] >= gate['dwell_s'] - 1e-06:
                        span = float(np.linalg.norm(np.ptp([x[1] for x in self.settle_samples], axis=0)))
                        angle = float(Rotation.from_matrix(np.array([x[2] for x in self.settle_samples]) @ self.settle_samples[0][2].T).magnitude().max())
                        if span <= gate['position_span_m'] and angle <= np.deg2rad(gate['orientation_from_first_deg']):
                            stable = dict(time_s=t, samples=len(self.settle_samples), position_span_m=span, orientation_from_first_rad=angle, raw_velocity_gate=False)
                    if stable:
                        s.events[f'kit{self.kit}_{self.index}_released_insert_stability'] = stable
                        s.credited.add((self.kit, name))
                        self.transition('withdraw')
                elif self.phase == 'withdraw':
                    if m.move([MOUNTS[side][0], 0.39, 1.17], 0.12):
                        self.index += 1
                        if self.index == 3:
                            self.complete = True
                            s.events[f'kit{self.kit}_complete_s'] = t
                            self.transition('complete')
                        else:
                            self.since = None
                            self.transition('approach')
            s.dual.jaw_travel(side, self.travel)
            if t - self.entered > 22:
                raise RuntimeError(f'{name}: timeout {self.phase}')
        except RuntimeError as exc:
            self.failure = f'{self.side}:{self.index}: {exc}'
            print('FAILURE', self.failure, flush=True)

def build_two_kit_layout(c):
    """Reset-only reuse of the original imported cases and inserts, no new products."""
    from pxr import Gf, Sdf, UsdGeom
    layer = c.stage.GetRootLayer()
    for j, x in enumerate(KIT_X):
        parent = UsdGeom.Xform.Define(c.stage, f'/World/Kit{j}')
        parent.AddTranslateOp().Set(Gf.Vec3d(x, 0, 0))
        for name in ['Case', 'Insert', 'InternalRiser']:
            Sdf.CopySpec(layer, '/World/Kit/' + name, layer, f'/World/Kit{j}/' + name)
        c.sign(f'Kit{j}ID', f'KIT {j + 1}', [x, 0.035, 1.005], [0.4, 0.055])
    for name in ['Case', 'Insert']:
        c.world.scene.remove_object(name, registry_only=True)
    for name in ['Case', 'Insert', 'InternalRiser']:
        c.stage.RemovePrim('/World/Kit/' + name)
    for path in ['KitID', 'Slot_BEARING', 'Slot_BRACKET', 'Slot_PIN']:
        c.stage.RemovePrim('/World/Signs/' + path)
    for name in NAMES:
        remove_preview(c, '/World/Feed/' + name)
        c.stage.RemovePrim('/World/Signs/Feed_' + name + 'Label')
        for j in range(3):
            remove_preview(c, f'/World/Parts/{name}{j}')

class KittingBatchController:

    def __init__(self, scene, mode):
        self.s = scene
        self.arms = [KitArmWorker(scene, mode, side, kit) for kit, side in enumerate(KIT_SIDES)]
        self.phase, self.failure, self.complete = ('parallel:park', None, False)

    def step(self, o):
        if self.failure or self.complete:
            return
        if o['time_s'] > self.s.resolved['budget_s']:
            self.failure = f"Two complete kits not finished within{self.s.resolved['budget_s']:g}s"
            return
        for arm in self.arms:
            arm.step(o)
        self.phase = ' | '.join((f'{arm.side}:{arm.index}:{arm.phase}' for arm in self.arms))
        self.failure = next((arm.failure for arm in self.arms if arm.failure), None)
        if all((arm.complete for arm in self.arms)) and (not self.failure):
            self.complete = True
            self.s.success = self.s.physical_completed = self.s.manipulation_success = self.s.final_valid(o)
            self.s.events['two_complete_kits_s'] = o['time_s']
            self.phase = 'complete'
resolve_one_kit = resolve_kitting

def resolve_kitting_batch(seed=0, *, layout=False, timing=False, physical=False):
    d = resolve_one_kit(seed, layout=layout, timing=timing, physical=physical)
    d.pop('sha256', None)
    d.update(required_completed_kits=2, required_delivered_kits=0, mobile_base_required=False, kit_centers_x_m=KIT_X, fixed_mounts_m=MOUNTS, source_x_m=[[round(KIT_X[j] + (-1 if j == 0 else 1) * (a - 0.28), 6) for a in [0.72, 0.56, 0.4]] for j in range(2)], kit_sides=KIT_SIDES, source_y_m=[0.68, 0.68, 0.6], grasp_minimum_per_jaw_N=dict(bearing=2.4, bracket=3.0, pin=0.2), gripper_stiffness_N_m=dict(bearing=1500, bracket=1500, pin=1500), bracket_tool_offset_m=[0.0235, 0, 0.012], bracket_tool_yaw_deg=dict(left=180, right=0), tool_rotation_rate_rad_s=0.4, bracket_lateral_normal_cos_min=0.8, probe_component_index=0, scope='Two fixed-workcell kits, both arms independently pack one bearing, bracket and pin into original case inserts. No mobile delivery required, per review, October22.')
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d
