"""C3 (task 49): weigh three machined parts on a bench scale and sort them by weight.

Since 2026-11-19 the task has no camera inspection. Each part is set on the scale between the
arms, released and read (``weigh_scale``); one of at least ``weigh_scale.MIN_G`` goes to a
PASS tray, a lighter one to a REJECT tray. Either arm's tray of the right result counts.

The scripted oracle (``WeighSortController``) moves every Cartesian segment along a
minimum-jerk profile and every joint transit along a minimum-jerk joint interpolation, and
lets both arms work at once: only the scale is exclusive, and an arm uses it only while the
other arm's tool is out of the central corridor.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig, weigh_scale
from .arx_cell_task import ArmMotion, ContactBook, add_mounts, collide, remove_preview
from .layout_review_revisions import RECEIVER_SIZE_49, RECEIVERS_49
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2'))
MOUNTS = rig.check(dict(left=[-0.3, 0.32, 1.2], right=[0.3, 0.32, 1.2]))
SORT_DWELL_S = 0.3
# Part centre height when it rests on the feed rails or the scale pan.
REST_Z = 0.94885


def resolve_weigh_sort(seed=0, *, layout=False, timing=False, physical=False):
    rs = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(4)]
    jitter = rs[0].uniform(-0.001, 0.001, (3, 2)).tolist() if layout else np.zeros((3, 2)).tolist()
    start_s = float(rs[1].uniform(1, 2)) if timing else 1.0
    mass_scale = float(rs[2].uniform(0.9, 1.1)) if physical else 1.0
    # Child 3 drew the old symbolic QC result; a REJECT draw is now an underweight part, so the
    # PASS/REJECT split of every seed is unchanged.
    results = rs[3].choice(['PASS', 'REJECT'], 3).tolist()
    underweight = [r == 'REJECT' for r in results]
    d = dict(
        task_id=49, embodiment='dual_arx_x5', seed=seed,
        part_xy_jitter_m=jitter, start_s=start_s, mass_scale=mass_scale,
        bounds=dict(part_xy_jitter_m=[-0.001, 0.001], start_s=[1, 2], mass_scale=[0.9, 1.1]),
        randomization=dict(layout=layout, timing=timing, physical=physical),
        finger_torsional_patch_radius_m=0.0, finger_min_torsional_patch_radius_m=0.0,
        budget_s=120.0, arm_command_speed_rad_s=0.7, runtime_joint_velocity_limit_rad_s=0.7,
        runtime_finger_velocity_limit_m_s=0.02, required_parts=3,
        underweight=underweight,
        part_mass_kg=[(weigh_scale.UNDERWEIGHT_KG if u else weigh_scale.NOMINAL_KG) * mass_scale for u in underweight],
        sort_results=results,
        weighing=dict(
            pan_xy_m=list(weigh_scale.PAN_XY), pan_top_z_m=weigh_scale.PAN_TOP_Z, min_g=weigh_scale.MIN_G,
            nominal_kg=weigh_scale.NOMINAL_KG, underweight_kg=weigh_scale.UNDERWEIGHT_KG,
            dwell_s=weigh_scale.WEIGH_DWELL_S, display_update_s=weigh_scale.UPDATE_S,
            sampling='SeedSequence child 3, independent per part: REJECT draw = underweight (70 % mass).',
            scope='Reading = measured native pan contact normal force / g; the display shows the 0.4 s mean once '
                  'the load stays within 2 g over 0.4 s (refreshed every 0.2 s), else its last stable value. '
                  'Weighed once resting alone, released and at rest on the pan for dwell_s. PASS at '
                  'min_g or more, else REJECT.'),
        source_positions_xy_m=[[-0.28, 0.73], [0, 0.73], [0.12, 0.73]], source_yaw_deg=0,
        input_tray_dimensions_m=[0.64, 0.24, 0.035],
        receivers_by_arm_xy_m=RECEIVERS_49, receiver_size_m=RECEIVER_SIZE_49,
        receiver_inner_half_xy_m=[0.075, 0.09], receiver_slot_offsets_y_m=[-0.04, 0.04],
        deposit_tool_yaw_deg=dict(left=90, right=90),
        arm_assignments=dict(left=[0], right=[2, 1]),
        joint_transit=dict(height_m=1.06, loaded_joint_speed_rad_s=0.7, loaded_linear_speed_m_s=0.22,
                           loaded_angular_speed_rad_s=0.9, empty_joint_speed_rad_s=0.7,
                           empty_linear_speed_m_s=0.3, empty_angular_speed_rad_s=1.5,
                           sampled_original_part_floor_m=1.0, sampled_empty_tool_floor_m=0.99,
                           profile='minimum jerk'),
        final_clearance_m=0.10,
        interlocks='Only the scale is exclusive. An arm claims it while the other arm\'s tool is '
                   'outside |x| <= 0.2 m and not weighing. Neighbouring feed nests (< 0.3 m apart) '
                   'are not entered by both arms at once. Original inter-arm collisions enabled.',
        controller='Privileged native-state bounded original joint drives; minimum-jerk Cartesian '
                   'segments and joint transits.',
        completion_event='Third weighed part fully released, supported and at rest in a tray of '
                         'its weight result for 0.3 s.',
    )
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d


def min_jerk(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (10 - 15 * u + 6 * u * u)


class WeighSortScene:
    title = 'WEIGHING / SORTING'

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .layout_review_revisions import build_layout_review
        from .remaining_environments import TRAY
        from .v2.geometry import body_envelope_points
        self.base = build_layout_review(49, output, (1280, 800))
        c = self.base.cell
        self.world, self.stage = c.world, c.stage
        self.stage.RemovePrim('/World/PlannedMount')
        for side in ['left', 'right']:
            self.stage.RemovePrim('/World/Signs/Mount_' + side)
        self.resolved, self.events = resolved, {}
        self.scale_owner = None
        self.arm_phase, self.arm_part = {}, {}
        self.weigh = weigh_scale.WeighTracker()
        self.sort_since, self.sorted = {}, set()
        self.physical_completed = self.success = False
        self.parts, self.envelopes = {}, {}
        self.asset_manifest = portable_json_loads((ROOT / 'task49/prepared-v3/asset-manifest.json').read_text())
        material = PhysicsMaterial('/World/InspectionContact', static_friction=0.95, dynamic_friction=0.85, restitution=0)
        remove_preview(c, '/World/Input/Tray')
        c.asset('Input/Tray', TRAY, [0, 0.73, 0.905], resolved['input_tray_dimensions_m'], material='steel')
        self.stage.RemovePrim('/World/Signs/Input_TrayLabel')
        c.sign('Input_TrayLabel', 'MACHINED / INPUT', [0, 0.598, 0.87], [0.4, 0.045])
        if self.stage.GetPrimAtPath('/World/Inspection/Part'):
            remove_preview(c, '/World/Inspection/Part')
        self.stage.RemovePrim('/World/Inspection/Nest')
        for i in range(3):
            remove_preview(c, f'/World/Input/Part{i}')
            path = f'/World/InspectionProducts/Part{i}'
            root = UsdGeom.Xform.Define(self.stage, path)
            root.GetPrim().GetReferences().AddReference(str(ROOT / 'task49/prepared-v3/port_block.usda'))
            contact = UsdGeom.Xform.Define(self.stage, path + '/ContactEnvelope')
            contact.GetPrim().GetReferences().AddReference(str(ROOT / 'task49/contact-v1/contact.usda'))
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            mass = resolved['part_mass_kg'][i]
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(mass)
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(root.GetPrim())
            api.CreateEnableCCDAttr(True)
            api.CreateAngularDampingAttr(2.0)
            api.CreateSolverPositionIterationCountAttr(64)
            api.CreateSolverVelocityIterationCountAttr(16)
            PhysxSchema.PhysxContactReportAPI.Apply(root.GetPrim()).CreateThresholdAttr(0)
            for mesh in Usd.PrimRange(root.GetPrim()):
                if mesh.IsA(UsdGeom.Mesh):
                    if not str(mesh.GetPath()).startswith(path + '/ContactEnvelope/'):
                        if mesh.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(mesh).CreateCollisionEnabledAttr(False)
                        continue
                    UsdPhysics.CollisionAPI.Apply(mesh)
                    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexHull')
                    a = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh)
                    a.CreateHullVertexLimitAttr(255)
                    a.CreateMinThicknessAttr(0.0001)
                    col = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                    col.CreateContactOffsetAttr(0.0003)
                    col.CreateRestOffsetAttr(0)
                    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material.material, materialPurpose='physics')
            p = np.r_[resolved['source_positions_xy_m'][i], REST_Z]
            p[:2] += resolved['part_xy_jitter_m'][i]
            for j, dx in enumerate([-0.024, 0.024]):
                c.box(f'FeedNests/Part{i}Rail{j}', [p[0] + dx, p[1], 0.91285], [0.006, 0.04, 0.03], 'steel')
            b = self.world.scene.add(SingleRigidPrim(path, name=f'inspect_part{i}', position=p, mass=mass, orientation=np.array([1.0, 0, 0, 0])))
            self.parts[i] = b
            self.envelopes[i] = body_envelope_points(self.stage, root.GetPrim())
        self.receivers = {side: {key: np.asarray(xy) for key, xy in trays.items()} for side, trays in resolved['receivers_by_arm_xy_m'].items()}
        for side, trays in self.receivers.items():
            for key in trays:
                assert self.stage.GetPrimAtPath(f'/World/Output/{key}_{side}'), (side, key)
        weigh_scale.build(c)
        self.static_colliders = collide(self.stage, ['/World/InspectionBench', '/World/Input/Tray', '/World/Output/', '/World/FeedNests/', '/World/Scale/Base', '/World/Scale/Pan', '/World/Scale/Housing'], material)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS)
        self.velocity_limit_overrides = []
        for joint in self.stage.Traverse():
            if str(joint.GetPath()).startswith('/World/ARX/') and joint.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                original = float(api.GetMaxJointVelocityAttr().Get())
                assert abs(original - np.rad2deg(0.75)) < 0.001
                limit = np.rad2deg(resolved['runtime_joint_velocity_limit_rad_s'])
                assert limit < original
                api.CreateMaxJointVelocityAttr(float(limit))
                self.velocity_limit_overrides.append(dict(joint=str(joint.GetPath()), imported_deg_s=original, configured_deg_s=float(limit)))
            elif str(joint.GetPath()).startswith('/World/ARX/') and joint.IsA(UsdPhysics.PrismaticJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                imported = float(api.GetMaxJointVelocityAttr().Get())
                limit = resolved['runtime_finger_velocity_limit_m_s']
                assert 0 < limit < imported <= 0.04
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(dict(joint=str(joint.GetPath()), imported_m_s=imported, configured_m_s=limit))
        add_mounts(c, MOUNTS, base_z=0.8875)
        self.static_colliders += collide(self.stage, ['/World/ARXMounts/'], material)
        for side in ['left', 'right']:
            for p in Usd.PrimRange(self.stage.GetPrimAtPath('/World/ARX/' + side)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
                    if any((str(p.GetPath()).startswith('/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{j}'] + '/') for j in [7, 8])):
                        contact = PhysxSchema.PhysxCollisionAPI.Apply(p)
                        contact.CreateTorsionalPatchRadiusAttr(resolved['finger_torsional_patch_radius_m'])
                        contact.CreateMinTorsionalPatchRadiusAttr(resolved['finger_min_torsional_patch_radius_m'])
        self.contacts = ContactBook(self.world, [b.prim_path for b in self.parts.values()])
        self.scale = weigh_scale.WeighScale(self, output)
        self.world.reset()
        self.dual.initialize()
        self.scale.reset_display()
        for b in self.parts.values():
            b.initialize()
        for data in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            data.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=49)
        self.asset_manifest.update(
            native_adaptations='Original product visuals on 30 mm flange-support rails; opposed X-face grasp. '
                               'Bench scale between the arms; a labeled PASS and REJECT receiver beside each arm. '
                               'Sorting by measured weight only (no camera inspection since 2026-11-19).',
            native_collision_derivation=portable_json_loads((ROOT / 'task49/contact-v1/contact.json').read_text()),
            approved_layout=self.base.metadata, static_colliders=self.static_colliders,
            velocity_limit_overrides=self.velocity_limit_overrides, weighing=resolved['weighing'])

    @property
    def detail(self):
        return f"Weighed and sorted: {len(self.sorted)}/3 | Budget {self.resolved['budget_s']:g}s"

    @staticmethod
    def pose(b):
        p, q = b.get_world_pose()
        return np.asarray(p), Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()

    def contained(self, index):
        p, R = self.pose(self.parts[index])
        pts = self.envelopes[index] @ R.T + p
        half = self.resolved['receiver_inner_half_xy_m']
        result = self.resolved['sort_results'][index]
        return any(bool(np.all(abs(pts[:, :2] - trays[result]) <= half) and pts[:, 2].min() >= 0.89765 and pts[:, 2].max() < 0.98) for trays in self.receivers.values())

    def receiver_touching(self, index):
        result = self.resolved['sort_results'][index]
        return any(self.contacts.touching(self.parts[index].prim_path, f'/World/Output/{result}_{side}/Offset/Asset/Mesh_0') for side in self.receivers)

    def observe(self):
        parts = {}
        rel = self.dual.manifest['relative_body_paths']
        for i, b in self.parts.items():
            p, R = self.pose(b)
            by_arm = {side: [self.contacts.force(b.prim_path, f'/World/ARX/{side}/' + rel[f'link{j}']) for j in [7, 8]] for side in ['left', 'right']}
            parts[str(i)] = dict(
                position=p.tolist(), rotation=R.tolist(), linear_velocity=b.get_linear_velocity().tolist(),
                angular_velocity=b.get_angular_velocity().tolist(), contained=self.contained(i),
                support_N=self.contacts.force(b.prim_path, '/World/Output/'),
                scale_N=self.contacts.force(b.prim_path, '/World/Scale/Pan'),
                support_touching=self.receiver_touching(i), finger_forces_N=by_arm['left'], finger_forces_by_arm=by_arm)
        tools = {side: self.dual.tool_world(side) for side in ['left', 'right']}
        return dict(
            time_s=float(self.world.current_time), tool_position=tools['left'][0].tolist(), parts=parts,
            scale_reading_g=self.scale.grams, weighed={str(k): v for k, v in self.weigh.weighed.items()},
            sorted=sorted(self.sorted), resource_owners=dict(scale=self.scale_owner),
            tools_world={side: dict(position=p.tolist(), rotation=R.tolist()) for side, (p, R) in tools.items()},
            unexpected_contacts=self.contacts.unexpected.copy())

    @staticmethod
    def at_rest_released(part):
        return (np.linalg.norm(part['linear_velocity']) < 0.012 and np.linalg.norm(part['angular_velocity']) < 0.1
                and max(f for forces in part['finger_forces_by_arm'].values() for f in forces) < 0.01)

    def process_step(self, o):
        """The scale (one load sample per control tick, display, weighed credit) and the sort
        credit the oracle's completion uses (the scoring monitor keeps its own)."""
        t = o['time_s']
        self.scale.update(t, sum(part['scale_N'] for part in o['parts'].values()))
        o['scale_reading_g'] = self.scale.grams
        for index, grams in self.weigh.update(o):
            self.events[f'{index}_weighed'] = dict(time_s=t, reading_g=grams, mass_g=1000 * self.resolved['part_mass_kg'][index], result=self.resolved['sort_results'][index])
        for index in range(3):
            part = o['parts'][str(index)]
            good = index in self.weigh.weighed and part['contained'] and part['support_N'] > 0.01 and self.at_rest_released(part)
            if index in self.sorted or not good:
                self.sort_since.pop(index, None)
                continue
            start = self.sort_since.setdefault(index, t)
            if t - start > SORT_DWELL_S:
                self.sorted.add(index)
                self.events[f'{index}_sorted_s'] = t

    def final_valid(self, o):
        return len(self.sorted) == 3 and all(
            self.contained(i) and i in self.weigh.weighed
            and (o['parts'][str(i)]['support_N'] > 0.01 or self.receiver_touching(i))
            and self.at_rest_released(o['parts'][str(i)]) for i in range(3)
        ) and self.events.get('completed_s', 999) <= self.resolved['budget_s']


class WeighArm:
    """One arm's scripted pick -> weigh -> re-grasp -> sort cycle over its assigned parts."""

    SCALE_PHASES = ('wait_scale_hold', 'scale_', 'regrip_')
    SOURCE_PHASES = ('approach', 'lower', 'close', 'lift')

    def __init__(self, scene, mode, side):
        self.s, self.mode, self.side = scene, mode, side
        self.other = 'right' if side == 'left' else 'left'
        self.sign = -1 if side == 'left' else 1
        self.m = ArmMotion(scene.dual, side)
        self.m.position_gain = 1.5
        self.m.command_speed_rad_s = scene.resolved['arm_command_speed_rad_s']
        self.m.orientation_feedback = True
        self.deposit_R = Rotation.from_euler('z', scene.resolved['deposit_tool_yaw_deg'][side], degrees=True).as_matrix() @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.pick_R = scene.dual.rotation @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.m.R = scene.dual.tool_world(side)[1].copy()
        self.indices = list(scene.resolved['arm_assignments'][side])
        if mode == 'grasp' and side == 'right':
            self.indices = []
        self.ordinal = 0
        self.index = self.indices[0] if self.indices else -1
        self.hold = np.array([-0.34 if side == 'left' else 0.34, 0.44, 1.22])
        self.phase, self.failure, self.complete = 'park', None, False
        self.entered = 0.0
        self.travel, self.since = 0.024, None
        self.regrip = False
        self.glide_state = None
        self.joint_path_kind = None
        self.transport_lost_since = None
        self.last_empty_start = None
        scene.arm_phase[side] = 'park'

    # ------------------------------------------------------------------ motion helpers
    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.arm_phase[self.side] = phase
        self.s.arm_part[self.side] = self.index
        self.s.events[f'{self.index}_{phase}_s'] = self.entered
        self.glide_state = None
        print('PHASE', self.index, phase, self.entered, flush=True)

    def glide(self, goal, speed, R=None):
        """Minimum-jerk Cartesian segment from the current command to ``goal`` (and from the
        current orientation to ``R``); True once the command has arrived and the tool follows."""
        m = self.m
        goal = np.asarray(goal, float)
        if self.glide_state is not None and np.linalg.norm(self.glide_state['goal'] - goal) < 0.003:
            # The same segment: a goal read from a part's pose jitters by micrometres.
            self.glide_state['goal'] = goal
        if self.glide_state is None or not np.array_equal(self.glide_state['goal'], goal):
            R0 = m.R.copy()
            angle = 0.0 if R is None else float(Rotation.from_matrix(R @ R0.T).magnitude())
            duration = max(1.875 * np.linalg.norm(goal - m.command) / speed, 1.875 * angle / 1.0, 0.2)
            self.glide_state = dict(start=m.command.copy(), goal=goal, R0=R0,
                                    delta=None if R is None else Rotation.from_matrix(R @ R0.T).as_rotvec(),
                                    duration=duration, u=0.0)
        g = self.glide_state
        actual = self.s.dual.tool_world(self.side)[0]
        if np.linalg.norm(actual - m.command) < 0.012:
            g['u'] = min(1.0, g['u'] + 1 / (CONTROL_HZ * g['duration']))
        k = min_jerk(g['u'])
        target = g['start'] + (g['goal'] - g['start']) * k
        if g['delta'] is not None:
            m.R = Rotation.from_rotvec(g['delta'] * k).as_matrix() @ g['R0']
        arrived = m.move(target, 10.0)
        return g['u'] >= 1 and arrived

    def settle_command(self):
        """Restart the Cartesian command from the measured tool (after a gripper change)."""
        self.m.command = self.s.dual.tool_world(self.side)[0].copy()
        self.m.correction[:] = 0
        self.m.orientation_correction[:] = 0
        self.glide_state = None

    def begin_joint_path(self, kind, t, *, goal=None, rotation=None, joint_goal=None, loaded=False):
        s, d, m = self.s, self.s.dual, self.m
        self.joint_path_kind = kind
        self.joint_from = d.commands[self.side][m.ids].copy()
        bp, bR = d.body_world(self.side, 'base_link')
        profile = d.manifest['profile']
        if joint_goal is None:
            joint_goal, error = m.kin.solve(bR.T @ (np.asarray(goal) - bp), bR.T @ rotation, dict(zip(d.names[self.side], d.commands[self.side])), profile['arm_lower_rad'], profile['arm_upper_rad'], restarts=4)
            if error > 0.001:
                raise RuntimeError('Bounded joint transit endpoint unreachable: ' + str(error))
        self.joint_goal = np.asarray(joint_goal).copy()
        if not (np.all(self.joint_goal >= profile['arm_lower_rad']) and np.all(self.joint_goal <= profile['arm_upper_rad'])):
            raise RuntimeError('Joint transit exceeds original bounds')
        points, rotations, minimum = [], [], float('inf')
        for u in np.linspace(0, 1, 81):
            q = self.joint_from * (1 - u) + self.joint_goal * u
            T = m.kin.forward(dict(zip(m.kin.arm_names, q)))
            pos, Rtool = bp + bR @ T[:3, 3], bR @ T[:3, :3]
            points.append(pos + Rtool @ self.offset if loaded else pos)
            rotations.append(Rtool)
            if loaded:
                v = s.envelopes[self.index] @ (Rtool @ self.grasp_rotation).T + points[-1]
                minimum = min(minimum, float(v[:, 2].min()))
            else:
                minimum = min(minimum, float(pos[2]))
        settings = s.resolved['joint_transit']
        floor = settings['sampled_original_part_floor_m' if loaded else 'sampled_empty_tool_floor_m']
        v = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max() * 80)
        W = np.asarray(rotations)
        w = float(Rotation.from_matrix(W[1:] @ np.swapaxes(W[:-1], 1, 2)).magnitude().max() * 80)
        prefix = 'loaded' if loaded else 'empty'
        # Minimum jerk peaks at 1.875x the mean rate.
        self.joint_duration = 1.875 * max(float(abs(self.joint_goal - self.joint_from).max()) / settings[prefix + '_joint_speed_rad_s'], v / settings[prefix + '_linear_speed_m_s'], w / settings[prefix + '_angular_speed_rad_s'], 0.1)
        self.joint_progress = 0.0
        self.joint_terminal_feedback = False
        Tend = m.kin.forward(dict(zip(m.kin.arm_names, self.joint_goal)))
        self.joint_tool_goal = bp + bR @ Tend[:3, 3]
        self.joint_tool_rotation = bR @ Tend[:3, :3]
        s.events[f'{kind}_joint_path'] = dict(time_s=t, q_start=self.joint_from.tolist(), q_goal=self.joint_goal.tolist(), planned_duration_s=self.joint_duration, minimum_sampled_z_m=minimum)
        return minimum >= floor

    def advance_joint_path(self):
        d = self.s.dual
        if self.joint_progress >= 1:
            if not self.joint_terminal_feedback:
                self.settle_command()
                self.m.R = self.joint_tool_rotation.copy()
                self.joint_terminal_feedback = True
            return self.glide(self.joint_tool_goal, 0.05, self.joint_tool_rotation)
        actual = np.asarray(d.robots[self.side].get_joint_positions())[self.m.ids]
        previous = d.commands[self.side][self.m.ids]
        if float(abs(actual - previous).max()) < 0.05:
            self.joint_progress = min(1.0, self.joint_progress + 1 / (CONTROL_HZ * self.joint_duration))
        k = min_jerk(self.joint_progress)
        d.commands[self.side][self.m.ids] = self.joint_from * (1 - k) + self.joint_goal * k
        return False

    # ------------------------------------------------------------------ interlocks
    def other_clear(self):
        """The other arm is out of the central corridor and not weighing."""
        x = self.s.dual.tool_world(self.other)[0][0]
        return abs(x) > 0.2 and not self.s.arm_phase.get(self.other, 'park').startswith(self.SCALE_PHASES)

    def source_free(self, index):
        """No other arm works in a feed nest less than 0.3 m from this one."""
        phase = self.s.arm_phase.get(self.other, 'park')
        other = self.s.arm_part.get(self.other)
        if not phase.startswith(self.SOURCE_PHASES) or other is None or other < 0:
            return True
        xs = self.s.resolved['source_positions_xy_m']
        return abs(xs[index][0] - xs[other][0]) >= 0.3

    # ------------------------------------------------------------------ cycle
    def step(self, o):
        if self.complete or self.failure or self.mode in ['absent', 'passive']:
            return
        s, t, m = self.s, o['time_s'], self.m
        if t < s.resolved['start_s']:
            return
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            self.cycle(o, t)
            s.dual.jaw_travel(self.side, self.travel)
            if t - self.entered > 18:
                raise RuntimeError(f'Part{self.index}: timeout {self.phase}')
        except RuntimeError as exc:
            self.failure = str(exc)
            print('FAILURE', self.failure, flush=True)

    def cycle(self, o, t):
        s, m = self.s, self.m
        if self.phase == 'park':
            if self.glide(self.hold, 0.2, self.pick_R):
                if self.indices:
                    self.transition('wait_source')
                else:
                    self.complete = True
            return
        if self.phase == 'complete':
            return
        b = s.parts[self.index]
        p, R = s.pose(b)
        grip = p + R @ np.array([0, 0, -0.002])
        if self.phase == 'wait_source':
            self.entered = t
            if self.source_free(self.index):
                self.transition('approach')
        elif self.phase == 'approach':
            if self.glide(grip + [0, 0, 0.06], 0.3, self.pick_R):
                self.transition('lower')
        elif self.phase in ('lower', 'regrip_lower'):
            arrived = self.glide(grip, 0.1 if self.regrip else 0.08)
            tool, tool_R = s.dual.tool_world(self.side)
            precise = arrived and np.linalg.norm(tool - grip) < 0.001 and Rotation.from_matrix(m.R @ tool_R.T).magnitude() < 0.01
            self.since = (self.since if self.since is not None else t) if precise else None
            if self.since is not None and t - self.since > 0.08:
                self.close_goal = grip.copy()
                self.since = None
                self.transition('regrip_close' if self.regrip else 'close')
        elif self.phase in ('close', 'regrip_close'):
            m.move(self.close_goal, 0.025)
            forces = o['parts'][str(self.index)]['finger_forces_by_arm'][self.side]
            good = min(forces) > 4.0
            if not good:
                self.travel = max(0, self.travel - 0.014 / CONTROL_HZ)
            self.since = (self.since if self.since is not None else t) if good else None
            if self.since is not None and t - self.since > 0.1:
                tool, toolR = s.dual.tool_world(self.side)
                self.offset = toolR.T @ (p - tool)
                self.grasp_rotation = toolR.T @ R
                self.lift_goal = tool + [0, 0, 0.07]
                self.grip_travel = self.travel
                self.since = None
                self.transition('regrip_lift' if self.regrip else 'lift')
        elif self.phase in ('lift', 'regrip_lift'):
            if self.glide(self.lift_goal, 0.2):
                if self.mode == 'grasp':
                    self.complete = True
                elif self.regrip:
                    self.regrip = False
                    self.transition('transfer')
                else:
                    self.transition('wait_scale')
        elif self.phase == 'wait_scale':
            m.move(self.lift_goal, 0.05)
            self.entered = t
            if s.scale_owner is None and self.other_clear():
                s.scale_owner = self.side
                s.events[f'{self.index}_scale_claimed_s'] = t
                tool = s.dual.tool_world(self.side)[0]
                above = np.r_[weigh_scale.PAN_XY, REST_Z + 0.05]
                self.scale_above = tool + (above - p)
                self.scale_place = self.scale_above - [0, 0, 0.047]
                self.transition('scale_move')
        elif self.phase == 'scale_move':
            if self.glide(self.scale_above, 0.3):
                self.transition('scale_place')
        elif self.phase == 'scale_place':
            if self.glide(self.scale_place, 0.09):
                self.transition('scale_release')
        elif self.phase == 'scale_release':
            m.move(self.scale_place, 0.02)
            # 8 mm past the grip: clear of the part, and a short close for the re-grasp.
            opened = min(0.024, self.grip_travel + 0.008)
            self.travel = min(opened, self.travel + 0.02 / CONTROL_HZ)
            if self.travel >= opened:
                self.settle_command()
                self.since = None
                self.transition('scale_read')
        elif self.phase == 'scale_read':
            # Straight up, clear of the part, until the reading counts; then straight back down.
            arrived = self.glide(self.scale_place + [0, 0, 0.05], 0.12)
            if arrived and f'{self.index}_weighed' in s.events:
                self.since = self.since if self.since is not None else t
                if t - self.since > 0.3:
                    self.since = None
                    self.regrip = True
                    self.transition('regrip_lower')
        elif self.phase == 'transfer':
            result = s.resolved['sort_results'][self.index]
            matching = [i for i in self.indices if s.resolved['sort_results'][i] == result]
            self.slot_xy = s.receivers[self.side][result] + [0, s.resolved['receiver_slot_offsets_y_m'][matching.index(self.index)]]
            if self.joint_path_kind != f'{self.index}_loaded':
                high = np.r_[self.slot_xy, s.resolved['joint_transit']['height_m']]
                self.begin_joint_path(f'{self.index}_loaded', t, goal=high - self.deposit_R @ self.offset, rotation=self.deposit_R, loaded=True)
            if s.scale_owner == self.side and abs(s.dual.tool_world(self.side)[0][0]) > 0.2:
                s.scale_owner = None
                s.events[f'{self.index}_scale_released_s'] = t
            retained = min(o['parts'][str(self.index)]['finger_forces_by_arm'][self.side]) > 0.2 and np.linalg.norm(p - s.dual.tool_world(self.side)[0]) < 0.02
            self.transport_lost_since = None if retained else t if self.transport_lost_since is None else self.transport_lost_since
            if self.transport_lost_since is not None and t - self.transport_lost_since > 0.2:
                raise RuntimeError('Native part grasp lost during transit')
            if self.advance_joint_path():
                if s.scale_owner == self.side:
                    s.scale_owner = None
                    s.events[f'{self.index}_scale_released_s'] = t
                self.place_goal = np.r_[self.slot_xy, 0.91885] - self.deposit_R @ self.offset
                self.transition('place')
        elif self.phase == 'place':
            if self.glide(self.place_goal + [0, 0, 0.003], 0.12, self.deposit_R):
                self.transition('release')
        elif self.phase == 'release':
            self.travel = min(0.024, self.travel + 0.02 / CONTROL_HZ)
            if self.travel >= 0.024:
                self.settle_command()
                self.transition('clear_release')
        elif self.phase == 'clear_release':
            if self.glide(self.place_goal + [0, 0, s.resolved['final_clearance_m']], 0.2):
                self.ordinal += 1
                if self.ordinal == len(self.indices):
                    self.transition('withdraw')
                else:
                    self.index = self.indices[self.ordinal]
                    self.transition('wait_next')
        elif self.phase == 'wait_next':
            self.entered = t
            if self.source_free(self.index):
                b = s.parts[self.index]
                p, R = s.pose(b)
                above = p + R @ np.array([0, 0, -0.002]) + [0, 0, 0.06]
                if not self.begin_joint_path(f'{self.index}_empty_reposition', t, goal=above, rotation=self.pick_R):
                    raise RuntimeError('Reposition joint transit dips below the tool floor')
                self.transition('reposition')
        elif self.phase == 'reposition':
            if self.advance_joint_path():
                self.m.R = self.pick_R.copy()
                self.transition('approach')
        elif self.phase == 'withdraw':
            end = np.array([0.45 * self.sign, 0.48, 1.12])
            if self.glide(end, 0.18, self.pick_R):
                self.complete = True
                self.transition('complete')


class WeighSortController:
    """Both arms' scripted cycles; the scene's process_step credits weighing and sorting."""

    def __init__(self, scene, mode):
        self.s, self.mode = scene, mode
        self.arms = [WeighArm(scene, mode, side) for side in ['left', 'right']]
        self.phase, self.failure, self.complete = 'park', None, False

    def step(self, observation):
        if self.complete or self.failure or self.mode in ['absent', 'passive']:
            return
        for arm in self.arms:
            arm.step(observation)
            if arm.failure:
                self.failure = f'{arm.side}: {arm.failure}'
                return
        self.phase = ' / '.join(f'{arm.side}:{arm.phase}' for arm in self.arms)
        if len(self.s.sorted) == 3 and all(arm.complete for arm in self.arms):
            self.s.events.setdefault('completed_s', max(self.s.events[f'{i}_sorted_s'] for i in range(3)))
            self.s.physical_completed = True
            self.s.success = self.s.final_valid(observation)
            self.complete = True
            self.phase = 'complete'
        elif all(arm.complete for arm in self.arms) and len(self.s.sorted) < 3:
            # Wait (bounded by the episode) for the last released part's sort credit.
            pass
