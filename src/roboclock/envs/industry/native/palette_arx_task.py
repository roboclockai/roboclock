"""Native Dual ARX four-color assembly; imported products, no attachments.

Feed fixtures raise thin pans above the original trays for jaw clearance.
Only released, supported, fully contained products can earn throughput credit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs import robodojo_rig
from roboclock.envs.rates import CONTROL_HZ, PHYSICS_HZ

from . import rig
from .arx_cell_task import ArmMotion, ContactBook, add_mounts, collide, remove_preview
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2/task50'))
# Recorded mounts already form the common rig (0.60 m apart, same height and depth).
MOUNTS = rig.check(dict(left=[-0.3, 0.22, 0.98], right=[0.3, 0.22, 0.98]))

# Seeded floor-plane offset of every feeder pan and its post (layout stream).
PAN_XY_JITTER_M = 0.002  # was 0.0008 (and off) before 2026-11-20


def resolve_palette(seed=0, *, layout=False, timing=False, physical=False):
    rs = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)]
    d = dict(task_id=50, embodiment='dual_arx_x5', seed=seed, pan_xy_jitter_m=rs[0].uniform(-PAN_XY_JITTER_M, PAN_XY_JITTER_M, (4, 6, 2)).tolist() if layout else np.zeros((4, 6, 2)).tolist(), start_s=float(rs[1].uniform(1, 2)) if timing else 1.0, mass_scale=float(rs[2].uniform(0.9, 1.1)) if physical else 1.0, budget_s=90.0, physics_hz=robodojo_rig.PHYSICS_HZ, solver_velocity_iterations=1, enable_stabilization=False, maximum_pan_face_tilt_deg=5.0, runtime_joint_velocity_limit_rad_s=0.73, runtime_finger_velocity_limit_m_s=0.038, carrier_handling_rotation_rate_rad_s=0.5, carrier_close_command_rate_m_s=0.002, delivery_payload='original_case_and_four_free_pans', case_grip=dict(tool_offset_m=[0, -0.053314, 0.0215], original_front_band_y_m=[-0.0602, -0.046428], original_front_band_height_m=0.01156218, jaw_axis='Case localY; downward approach pinches the original front lip above the empty recess floor, clear of all four pans', bilateral_force_target_N=12.0, opening_per_slider_m=0.012, output_withdraw_clearance_m=0.14, finger_position_stiffness_N_m=4000.0, finger_damping_N_s_m=80.0, effort_cap_N=20.0, alignment=dict(position_m=0.0003, orientation_rad=0.003, dwell_s=0.1), supported_unload=dict(native_output_support_N=0.6, support_dwell_s=0.15, opening_rate_m_s=0.002, released_slider_travel_m=0.009, scope='Seat original case on the output tray before gradual jaw unloading; freeze the tool waypoint during release. No free drop of the loaded case.')), carrier_control='Stationary unscored assembly nest; direct original case-front-lip native grasp, measured4mm/0.03rad payload endpoint gates and original bounded joint drives. No product attachments.', carrier_yaw_deg=[90, -90], pan_release_alignment=dict(position_m=0.0005, orientation_rad=0.005, dwell_s=0.1, floor_clearance_m=0.008), pan_release_stability=dict(dwell_s=0.2, position_span_m=0.0002, orientation_from_first_deg=0.2, scope='Continuous actual pan and case poses, original slot containment/support and no hand contact. Raw contact-step velocities remain logged diagnostics.'), carrier_output_stability=dict(dwell_s=0.5, position_span_m=0.0005, orientation_from_first_deg=0.3, scope='Consecutive actual native poses of original case and all4 pans with native output support and no hand contacts; assembly nest must remain stationary. Raw velocities retained separately.'), required_palettes=2, randomization=dict(layout=layout, timing=timing, physical=physical), bounds=dict(pan_xy_jitter_m=[-PAN_XY_JITTER_M, PAN_XY_JITTER_M], start_s=[1, 2], mass_scale=[0.9, 1.1]), independence='Independent SeedSequence layout/temporal/physical streams', controller='Privileged-state bounded original joint-drive oracle', scoring='Two four-color palettes, unique correct pans fully within original pockets, released native support and output tray delivery within90s', assumptions='Existing source four-pocket case/pans,25mm feeder posts and stationary original tray/35mm assembly supports. Original products unchanged; direct case-front-lip grip and release into lower front output tray. Free case/pans, no product attachments. Candidate masses/friction, not calibrated.')
    # robodojo_rig head view (60 deg down): the output trays stood behind the bases at
    # (+/-0.6, 0.015, 0.76), outside the head image. They now sit on the bench top in front
    # of each nest, and the loaded case moves straight forward to them.
    d.update(output_tray_xy_m=[[-0.5, 0.69], [0.5, 0.69]], output_tray_floor_m=0.8974, output_tray_size_m=[0.26, 0.22, 0.025],
             case_transfer_xy_m=[[[-0.5, 0.43], [-0.5, 0.69]], [[0.5, 0.43], [0.5, 0.69]]])
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d

class PaletteARXScene:
    title = 'FOUR-COLOR PALETTE / ASSEMBLY + OUTPUT'

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import RigidPrim, SingleRigidPrim
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .remaining_environments import TABLE, build_remaining_environment
        self.base = build_remaining_environment(50, output, resolution=(1280, 800))
        c = self.base.cell
        self.world, self.stage = (c.world, c.stage)
        self.world.set_simulation_dt(physics_dt=1 / resolved.get('physics_hz', PHYSICS_HZ), rendering_dt=1 / CONTROL_HZ)
        stage = self.stage
        self.resolved, self.events = (resolved, {})
        self.asset_manifest = portable_json_loads((ROOT / 'asset-manifest.json').read_text())
        self.delivered = set()
        self.transport_verified = set()
        self.manipulation_success = False
        self.credited = set()
        self.physical_completed = False
        self.success = False
        self.active_path = None
        self.last_sample = None
        material = PhysicsMaterial('/World/PaletteContact', static_friction=0.95, dynamic_friction=0.85, restitution=0)
        carrier_metadata = portable_json_loads((ROOT / 'direct-case-v1/fixture.json').read_text())
        original_case_mass_properties = portable_json_loads((ROOT / 'direct-case-v1/original_case_mass_properties.json').read_text())
        self.case_initial_z = 0.9335 + carrier_metadata['case_support_rise_m']
        self.carrier_partitions = []
        self.pan_partitions = {}
        self._native_cache = None
        self._bulk_checked = False
        self._last_bulk_check_t = None
        self.readback_checks = []
        carrier_visual = c.appearance.material('carrier_abs', color=[0.1, 0.13, 0.15], roughness=0.6, metallic=0)

        def body(path, file, p, mass, case=False, carrier=False):
            x = UsdGeom.Xform.Define(stage, path)
            x.GetPrim().GetReferences().AddReference(str(file))
            if case:
                shell = UsdGeom.Xform.Define(stage, path + '/ContactShell')
                shell.GetPrim().GetReferences().AddReference(str(ROOT / 'direct-case-v1/case_collision.usda'))
            UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            if carrier:
                UsdPhysics.RigidBodyAPI(x.GetPrim()).CreateKinematicEnabledAttr(True)
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass * resolved['mass_scale'])
            if case:
                previous = original_case_mass_properties[path]
                tensor = np.asarray(previous['inertia'])[0].reshape(3, 3)
                moments, axes = np.linalg.eigh(tensor)
                assert np.all(moments > 0)
                if np.linalg.det(axes) < 0:
                    axes[:, -1] *= -1
                q = Rotation.from_matrix(axes).as_quat()
                props = UsdPhysics.MassAPI(x.GetPrim())
                props.CreateCenterOfMassAttr(Gf.Vec3f(*previous['center_of_mass_m'][0]))
                props.CreateDiagonalInertiaAttr(Gf.Vec3f(*moments * resolved['mass_scale']))
                props.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), *map(float, q[:3])))
            if not case and (not carrier):
                from .v2.convex_partition import partition_convex
                original = body_envelope_points(stage, x.GetPrim())
                pieces, partition = partition_convex(original)
                self.pan_partitions[file.name] = partition
                for mesh in Usd.PrimRange(x.GetPrim()):
                    if mesh.IsA(UsdGeom.Mesh):
                        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(False)
                for i, (vertices, faces) in enumerate(pieces):
                    mesh = UsdGeom.Mesh.Define(stage, path + f'/PanContactVolumes/Piece{i}')
                    mesh.CreatePointsAttr(vertices.tolist())
                    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
                    mesh.CreateVisibilityAttr('invisible')
                    prim = mesh.GetPrim()
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                    hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
                    hull.CreateHullVertexLimitAttr(64)
                    hull.CreateMinThicknessAttr(0.0001)
                    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    col.CreateContactOffsetAttr(0.0003)
                    col.CreateRestOffsetAttr(0)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose='physics')
            if carrier:
                inertia = carrier_metadata['mass_properties']
                UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateDiagonalInertiaAttr(Gf.Vec3f(*np.array(inertia['diagonal_inertia_kg_m2']) * resolved['mass_scale']))
            ph = PhysxSchema.PhysxRigidBodyAPI.Apply(x.GetPrim())
            ph.CreateEnableCCDAttr(True)
            ph.CreateAngularDampingAttr(2.0)
            ph.CreateSolverPositionIterationCountAttr(64)
            ph.CreateSolverVelocityIterationCountAttr(resolved['solver_velocity_iterations'])
            PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0)
            for mesh in Usd.PrimRange(x.GetPrim()):
                if mesh.IsA(UsdGeom.Gprim):
                    if carrier and mesh.GetName() == 'Tray':
                        UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(False)
                        continue
                    if case and (not str(mesh.GetPath()).startswith(path + '/ContactShell/')):
                        continue
                    UsdPhysics.CollisionAPI.Apply(mesh)
                    if mesh.IsA(UsdGeom.Mesh):
                        approximation = 'convexHull'
                        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(approximation)
                        if approximation == 'sdf':
                            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh).CreateSdfResolutionAttr(1024)
                    if mesh.IsA(UsdGeom.Mesh) and (case or carrier):
                        hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(mesh)
                        hull.CreateMinThicknessAttr(0.0001)
                        hull.CreateHullVertexLimitAttr(255 if carrier else 64)
                    api = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                    api.CreateContactOffsetAttr(0.0003)
                    api.CreateRestOffsetAttr(0)
                    UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material.material, materialPurpose='physics')
                    if carrier:
                        UsdShade.MaterialBindingAPI.Apply(mesh).Bind(carrier_visual.material)
            if carrier:
                from .v2.convex_partition import partition_convex
                visual = UsdGeom.Mesh(stage.GetPrimAtPath(path + '/Tray'))
                pieces, partition = partition_convex(np.array(visual.GetPointsAttr().Get(), float))
                partition['scope'] = 'Unused original tray cavity conservatively filled by exact outer hull; scored case is supported35mm above original interior floor, above the rim. Original visuals/mass/inertia preserved.'
                self.carrier_partitions.append(partition)
                for i, (vertices, faces) in enumerate(pieces):
                    mesh = UsdGeom.Mesh.Define(stage, path + f'/TrayContactVolumes/Piece{i}')
                    mesh.CreatePointsAttr(vertices.tolist())
                    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
                    mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
                    mesh.CreateVisibilityAttr('invisible')
                    prim = mesh.GetPrim()
                    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexHull')
                    hull = PhysxSchema.PhysxConvexHullCollisionAPI.Apply(prim)
                    hull.CreateHullVertexLimitAttr(64)
                    hull.CreateMinThicknessAttr(0.0001)
                    col = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    col.CreateContactOffsetAttr(0.0003)
                    col.CreateRestOffsetAttr(0)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material.material, materialPurpose='physics')
            return self.world.scene.add(SingleRigidPrim(path, name=path.replace('/', '_'), position=np.asarray(p, float), mass=mass * resolved['mass_scale']))
        from .v2.geometry import body_envelope_points
        self.case_support_since = {}
        self.case_envelopes = {}
        self.carrier_envelopes = {}
        self.pan_envelopes = {}
        self.pan = {}
        self.case = {}
        self.carrier = {}
        self.pan_initial = {}
        self.product_paths = []
        self.feed_centers = [[0.0, y] for y in [0.42, 0.51, 0.6, 0.69]]
        for color in range(4):
            remove_preview(c, f'/World/Feed/Color{color}')
            xy = np.asarray(self.feed_centers[color])
            c.asset(f'Feed/Color{color}', Path(asset_path('/opt/industry/assets/v2/task46/prepared/tray.usd')), [*xy, 0.9], [0.56, 0.08, 0.025], material='steel')
            label = f'Feed_Color{color}Label'
            stage.RemovePrim('/World/Signs/' + label)
            c.sign(label, ['CHAMPAGNE', 'ROSE', 'COPPER', 'DEEP BROWN'][color], [xy[0], xy[1] - 0.047, 0.87], [0.24, 0.025])
            for j in range(6):
                remove_preview(c, f'/World/Feed/Pan{color}_{j}')
                stock = xy + [[-0.23, -0.09, 0.23, -0.03, 0.03, 0.09][j], 0.0] + np.asarray(resolved['pan_xy_jitter_m'][color][j])
                pos = np.r_[stock, 0.92117]
                if mode in ['seat', 'delivery'] and j in [0, 2]:
                    slot = self.asset_manifest['slots'][color]
                    pos = np.array([-0.5 if j == 0 else 0.5, 0.41009356, self.case_initial_z]) + np.r_[slot['slot_xy_m'], slot['resting_pan_center_z_m'] + 0.006]
                path = f'/World/Pans/Pan{color}_{j}'
                self.pan[color, j] = body(path, ROOT / 'prepared-v2' / f'pan_{color}.usd', pos, 0.01)
                self.pan_initial[color, j] = pos
                self.product_paths.append(path)
                c.pipe(f'FeederPosts/Color{color}_{j}', [*stock, 0.89489], [*stock, 0.91989], 0.004, 'steel')
        for path in ['/World/Assembly/Case', '/World/Cases/Empty0', '/World/Cases/Empty1', '/World/Empty/Cases', '/World/Output/Tray']:
            remove_preview(c, path)
        stage.RemovePrim('/World/Signs/Recipe')
        c.sign('Recipe', 'RECIPE 1:CHAMPAGNE 2:ROSE 3:COPPER 4:DEEP BROWN', [0, 1.16, 1.29], [1.75, 0.11])
        for name in ['Empty_CasesLabel', 'Output_TrayLabel']:
            stage.RemovePrim('/World/Signs/' + name)
        stage.RemovePrim('/World/Nest/Base')
        self.output_roots = {}
        self.output_rotations = {}
        self.output_floors = {}
        for cycle, x in enumerate([-0.5, 0.5]):
            side = 'left' if cycle == 0 else 'right'
            c.box('Nest/' + side, [x, 0.43, 0.90680455], [0.17, 0.26, 0.0386091], 'frame')
            self.carrier[cycle] = body(f'/World/Carriers/Carrier{cycle}', ROOT / 'direct-case-v1/fixture.usda', [x, 0.43, 0.9386091], carrier_metadata['proposed_mass_kg'], carrier=True)
            self.case[cycle] = body(f'/World/Palettes/Case{cycle}', ROOT / 'prepared-v2/case.usd', [x, 0.41009356, self.case_initial_z], 0.1, True)
            outx, outy = resolved['output_tray_xy_m'][cycle]
            floor = resolved['output_tray_floor_m']
            self.output_rotations[cycle] = Rotation.from_euler('z', resolved['carrier_yaw_deg'][cycle], degrees=True).as_matrix()
            # On the bench top (0.89 m); same tray asset, floor 7.4 mm above its base as before.
            c.asset('Output/' + side + 'Tray', Path(asset_path('/opt/industry/assets/v2/task46/prepared/tray.usd')), [outx, outy, floor - 0.0074 + resolved['output_tray_size_m'][2] / 2], resolved['output_tray_size_m'], material='steel', yaw=np.deg2rad(resolved['carrier_yaw_deg'][cycle]))
            self.output_roots[cycle] = np.array([outx, outy, floor])
            self.output_floors[cycle] = floor
            self.case_envelopes[cycle] = body_envelope_points(stage, stage.GetPrimAtPath(self.case[cycle].prim_path))
            self.carrier_envelopes[cycle] = body_envelope_points(stage, stage.GetPrimAtPath(self.carrier[cycle].prim_path))
            c.sign(side + 'OutputLabel', 'FINISHED / OUTPUT', [outx, outy + 0.125, 0.905], [0.22, 0.03])
        for key, b in self.pan.items():
            self.pan_envelopes[key] = body_envelope_points(stage, stage.GetPrimAtPath(b.prim_path))
        self.product_paths += ['/World/Palettes/Case0', '/World/Palettes/Case1']
        self.static_colliders = collide(stage, ['/World/AssemblyBench', '/World/Nest/', '/World/Feed/Color', '/World/Empty/Cases', '/World/Output/', '/World/FeederPosts/'], material)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS)
        self.velocity_limit_overrides = []
        for joint in stage.Traverse():
            if str(joint.GetPath()).startswith('/World/ARX/') and joint.IsA(UsdPhysics.PrismaticJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                imported = float(api.GetMaxJointVelocityAttr().Get())
                source = float(joint.GetAttribute('urdf:limit:velocity').Get())
                assert abs(source - 0.04) < 1e-06
                limit = resolved['runtime_finger_velocity_limit_m_s']
                assert 0 < limit < source
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(dict(joint=str(joint.GetPath()), kind='linear', imported_stage_units_s=imported, source_urdf_m_s=source, configured_m_s=limit, reason='Importer incorrectly converted prismatic0.04m/s to2.29183 as if angular. Enforce source linear limit with numerical headroom.'))
            if str(joint.GetPath()).startswith('/World/ARX/') and joint.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(joint)
                original = float(api.GetMaxJointVelocityAttr().Get())
                assert abs(original - np.rad2deg(0.75)) < 0.001
                limit = float(np.rad2deg(resolved['runtime_joint_velocity_limit_rad_s']))
                assert limit < original
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_limit_overrides.append(dict(joint=str(joint.GetPath()), imported_deg_s=original, configured_deg_s=limit))
        add_mounts(c, MOUNTS, base_z=0.8875)
        self.static_colliders += collide(stage, ['/World/ARXMounts/'], material)
        for side in ['left', 'right']:
            for p in Usd.PrimRange(stage.GetPrimAtPath('/World/ARX/' + side)):
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
        self.contacts = ContactBook(self.world, self.product_paths, watched_prefixes=('/World/Carriers/',), record_points=True)
        self.bulk_readback = self.world.scene.add(RigidPrim(prim_paths_expr=[b.prim_path for b in [*self.pan.values(), *self.case.values(), *self.carrier.values()]], name='palette_readback_only', reset_xform_properties=False, prepare_contact_sensors=False, disable_stablization=False))
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(p).CreateEnableStabilizationAttr(resolved['enable_stabilization'])
            if str(p.GetPath()).startswith('/World/ARX/') and p.HasAPI(PhysxSchema.PhysxArticulationAPI):
                PhysxSchema.PhysxArticulationAPI(p).CreateSolverVelocityIterationCountAttr(resolved['solver_velocity_iterations'])
        self.world.reset()
        self.dual.initialize()
        self.initial_joint_positions = {}
        for side in ['left', 'right']:
            motion = ArmMotion(self.dual, side)
            bp, bR = self.dual.body_world(side, 'base_link')
            park = np.array([-0.56 if side == 'left' else 0.56, 0.36, 1.1])
            profile = self.dual.manifest['profile']
            q, error = motion.kin.solve(bR.T @ (park - bp), bR.T @ motion.R, dict(zip(self.dual.names[side], self.dual.commands[side])), profile['arm_lower_rad'], profile['arm_upper_rad'], restarts=8)
            if error > 0.0001:
                raise RuntimeError('Initial safe park is unreachable')
            initial = self.dual.commands[side].copy()
            initial[motion.ids] = q
            robot = self.dual.robots[side]
            robot.set_joints_default_state(positions=initial, velocities=np.zeros(8))
            robot.set_joint_positions(initial)
            robot.set_joint_velocities(np.zeros(8))
            self.dual.commands[side] = initial.copy()
            self.initial_joint_positions[side] = initial.tolist()
        for b in [*self.pan.values(), *self.case.values(), *self.carrier.values()]:
            b.initialize()
        self.bulk_readback.initialize()
        self._readback_paths = list(self.bulk_readback.prim_paths)
        assert set(self._readback_paths) == {b.prim_path for b in [*self.pan.values(), *self.case.values(), *self.carrier.values()]}
        self.native_carrier_mass_properties = {}
        for k, b in self.carrier.items():
            center, orientation = b.get_com()
            self.native_carrier_mass_properties[str(k)] = dict(mass_kg=float(b.get_mass()), center_of_mass_m=center.tolist(), principal_axes_wxyz=orientation.tolist(), inertia=b._rigid_prim_view.get_inertias().tolist())
        self.native_payload_mass_properties = {}
        for b in [*self.case.values(), *self.pan.values()]:
            center, orientation = b.get_com()
            self.native_payload_mass_properties[b.prim_path] = dict(mass_kg=float(b.get_mass()), center_of_mass_m=center.tolist(), principal_axes_wxyz=orientation.tolist(), inertia=b._rigid_prim_view.get_inertias().tolist())
            if b.prim_path in original_case_mass_properties:
                previous = original_case_mass_properties[b.prim_path]
                actual = self.native_payload_mass_properties[b.prim_path]
                assert np.allclose(actual['center_of_mass_m'], previous['center_of_mass_m'], atol=1e-07, rtol=0)
                assert np.allclose(actual['inertia'], np.asarray(previous['inertia']) * resolved['mass_scale'], atol=1e-08, rtol=1e-05)
        for m in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            m.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=50)
        self.cue_state = 'PRELOADED TRANSPORT COMPONENT' if mode == 'delivery' else 'ASSEMBLY'
        self.carrier_metadata = carrier_metadata
        self.asset_manifest['native_adaptations'] = dict(feeder_posts='8mm diameter25mm tall; original pans unchanged', feed_row='Four transverse six-pan rows; left picks atX-.23, right atX+.23. Disjoint approach corridors allow parallel fetches; native inter-arm collision checks remain enabled. Two side assembly nests/outputs. Original products unchanged', case_collision='Original case surface patches and four unchanged original-outline pocket floor volumes; front lip is its actual source-material convex solid rather than thin contact skins. Original surface/visual/pocket dimensions unchanged. Native validation required.', case_collision_derivation=portable_json_loads((ROOT / 'direct-case-v1/case_collision.json').read_text()), original_case_mass_properties_preserved=original_case_mass_properties, case_mass_policy='Preserve previous native case mass,COM and full local inertia tensor explicitly while refining only the front-lip contact volume. Original candidate properties are not hardware calibrated. Independent mass scaling multiplies both mass and inertia; COM unchanged.', carrier=carrier_metadata, carrier_scope='Stationary unscored original-tray assembly nests retain unchanged35mm support pads and side/rear guides. Carrying handles/frame and front guide removed. Original free case is grasped directly at its original empty front lip and delivered without its nest; no attachments or post-reset product state writes. Native validation required.', velocity_limit_overrides=self.velocity_limit_overrides, initial_joint_positions=self.initial_joint_positions, native_carrier_mass_properties=self.native_carrier_mass_properties, native_payload_mass_properties=self.native_payload_mass_properties, physics_configuration=dict(configured_gravity=self.world.get_physics_context().get_gravity(), actual_physics_dt_s=self.world.get_physics_dt(), solver_position_iterations=64, solver_velocity_iterations=resolved['solver_velocity_iterations'], enable_stabilization=resolved['enable_stabilization'], scope='Candidate solver configuration, not a calibrated contact model; native support, penetration, pose windows and all8DOF caps remain independently checked.'), carrier_outer_hull_partitions=self.carrier_partitions, pan_original_hull_partitions=self.pan_partitions, operating_layout=dict(mounts=MOUNTS, case_centers_x_m=[-0.5, 0.5], feeder_selected_x_m=[-0.23, 0.23], output_roots_m={str(k): v.tolist() for k, v in self.output_roots.items()}, side_translation_from_first_probe_m=0.08, policy='Arm/feed/nest/output coordinates retain the reviewed80mm outward translation and160mm added inter-arm spacing. Direct-case approach replaces handle transport; original products unchanged. Native validation required.'), native_readback='One read-only28-body pose/velocity batch at each60Hz control instant; assert live physics handle and cross-check all individual native getters every1s. No body state writes. Physics rate/solver settings logged separately; this cross-check is distinct from whole-process replay.')

    @property
    def detail(self):
        return f"Complete palettes: {len(self.delivered)}/2 | Four colors in recipe order | Budget{self.resolved['budget_s']:g}s"

    def pose(self, b):
        if self._native_cache is not None:
            row = self._native_cache[b.prim_path]
            return (row[0], row[1])
        p, q = b.get_world_pose()
        return (np.asarray(p), Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix())

    def contained(self, cycle, color, body):
        cp, cR = self.pose(self.case[cycle])
        p, R = self.pose(body)
        key = next((k for k, b in self.pan.items() if b is body))
        points = (self.pan_envelopes[key] @ R.T + p - cp) @ cR
        slot = self.asset_manifest['slots'][color]
        bounds = np.asarray(slot['pocket_bounds_xy_m'])
        return bool(np.dot(R[:, 2], cR[:, 2]) >= np.cos(np.deg2rad(self.resolved['maximum_pan_face_tilt_deg'])) and np.all(points[:, :2] >= bounds[0] - 0.0002) and np.all(points[:, :2] <= bounds[1] + 0.0002) and (points[:, 2].min() >= slot['floor_z_m'] - 0.0002) and (points[:, 2].max() <= 0.01156218 + 0.0002))

    def observe(self):
        t = float(self.world.current_time)
        assert self.bulk_readback.is_physics_handle_valid(), 'Native product view unavailable'
        p, q = self.bulk_readback.get_world_poses()
        p = np.asarray(p).reshape(-1, 3)
        q = np.asarray(q).reshape(-1, 4)
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        linear = np.asarray(self.bulk_readback.get_linear_velocities()).reshape(-1, 3)
        angular = np.asarray(self.bulk_readback.get_angular_velocities()).reshape(-1, 3)
        self._native_cache = {path: (p[i], rotations[i], linear[i], angular[i]) for i, path in enumerate(self._readback_paths)}
        if self._last_bulk_check_t is None or t - self._last_bulk_check_t >= 1 - 1e-06:
            max_pose_error = max_velocity_error = 0.0
            for b in [*self.pan.values(), *self.case.values(), *self.carrier.values()]:
                bp, bq = b.get_world_pose()
                br = Rotation.from_quat(np.r_[bq[1:], bq[0]]).as_matrix()
                cached = self._native_cache[b.prim_path]
                max_pose_error = max(max_pose_error, float(abs(bp - cached[0]).max()), float(abs(br - cached[1]).max()))
                assert np.allclose(bp, cached[0], rtol=0, atol=1e-07) and np.allclose(br, cached[1], rtol=0, atol=1e-07)
                bv, bw = (b.get_linear_velocity(), b.get_angular_velocity())
                max_velocity_error = max(max_velocity_error, float(abs(bv - cached[2]).max()), float(abs(bw - cached[3]).max()))
                assert np.allclose(bv, cached[2], rtol=0, atol=1e-07) and np.allclose(bw, cached[3], rtol=0, atol=1e-07)
            self._bulk_checked = True
            self._last_bulk_check_t = t
            self.readback_checks.append(dict(time_s=t, bodies=28, max_pose_error=max_pose_error, max_velocity_error=max_velocity_error))
        pans = {}
        cases = {}
        carriers = {}
        for key, b in self.pan.items():
            p, R = self.pose(b)
            path = b.prim_path
            pan = dict(position=p.tolist(), rotation=R.tolist(), linear_velocity=self._native_cache[b.prim_path][2].tolist(), angular_velocity=self._native_cache[b.prim_path][3].tolist(), finger_forces_N=[self.contacts.force(path, f'/World/ARX/{side}') for side in ['left', 'right']])
            pans[f'{key[0]}_{key[1]}'] = pan
            # Per case the pan is at (within a case's half-diagonal of its centre): whether it
            # sits in its colour's pocket there. The judge credits any pan of the colour.
            pan['cases'] = {}
            for cycle, case in self.case.items():
                if np.linalg.norm(p[:2] - self._native_cache[case.prim_path][0][:2]) < 0.12:
                    pan['cases'][str(cycle)] = dict(contained=self.contained(cycle, key[0], b), support_N=self.contacts.force(path, case.prim_path), touching=self.contacts.touching(path, case.prim_path))
            if key[1] in [0, 2]:
                cycle = 0 if key[1] == 0 else 1
                pan['slot_contained'] = self.contained(cycle, key[0], b)
                pan['case_support_N'] = self.contacts.force(path, self.case[cycle].prim_path)
                pan['case_touching'] = self.contacts.touching(path, self.case[cycle].prim_path)
        for k, b in self.case.items():
            p, R = self.pose(b)
            cases[str(k)] = dict(position=p.tolist(), rotation=R.tolist(), linear_velocity=self._native_cache[b.prim_path][2].tolist(), angular_velocity=self._native_cache[b.prim_path][3].tolist(), carrier_support_N=self.contacts.force(b.prim_path, self.carrier[k].prim_path), carrier_touching=self.contacts.touching(b.prim_path, self.carrier[k].prim_path), hand_forces_N=[self.contacts.force(b.prim_path, '/World/ARX/' + side) for side in ['left', 'right']], output_support_N=self.contacts.force(b.prim_path, '/World/Output/'), output_touching=self.contacts.touching(b.prim_path, '/World/Output/' + ('left' if k == 0 else 'right') + 'Tray/Offset/Asset/Mesh_0'), finger_forces_N={side: [self.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{j}']) for j in [7, 8]] for side in ['left', 'right']})
        for k, b in self.carrier.items():
            p, R = self.pose(b)
            carriers[str(k)] = dict(position=p.tolist(), rotation=R.tolist(), linear_velocity=self._native_cache[b.prim_path][2].tolist(), angular_velocity=self._native_cache[b.prim_path][3].tolist(), output_support_N=self.contacts.force(b.prim_path, '/World/Output/'), output_touching=self.contacts.touching(b.prim_path, '/World/Output/' + ('left' if k == 0 else 'right') + 'Tray/Offset/Asset/Mesh_0'), nest_support_N=self.contacts.force(b.prim_path, '/World/Nest/'), finger_forces_N={side: [self.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]] for side in ['left', 'right']}, finger_on_handle={side: [any((v['active'] and 'HandlingStud' in key and ('/World/ARX/' + side + '/' in key) and ('/' + self.dual.manifest['relative_body_paths'][f'link{j}'] + '/' in key) for key, v in self.contacts.active.items())) for j in [7, 8]] for side in ['left', 'right']})
        side = getattr(self, 'active_side', 'left')
        return dict(time_s=t, tool_position=self.dual.tool_world(side)[0].tolist(), pans=pans, cases=cases, carriers=carriers, tools_world={side: dict(position=self.dual.tool_world(side)[0].tolist(), rotation=self.dual.tool_world(side)[1].tolist()) for side in ['left', 'right']}, native_case_contact_samples={key: value for key, value in self.contacts.contact_samples.items() if '/World/Palettes/Case' in key and t - value['time_s'] <= 1 / CONTROL_HZ + 1e-06}, delivered=sorted(self.delivered), transport_verified=sorted(self.transport_verified), credited=sorted(self.credited), cue_state=self.cue_state, unexpected_contacts=self.contacts.unexpected.copy())

    def final_valid(self, o):
        return len(self.delivered) >= 2 and max(self.events.get('delivery0_s', 999), self.events.get('delivery1_s', 999)) <= self.resolved['budget_s'] and all((self.payload_native_valid(o, cycle) and all((self.contained(cycle, color, self.pan[color, 0 if cycle == 0 else 2]) for color in range(4))) for cycle in [0, 1]))

    def case_lip_contact_valid(self, cycle, side):
        """Require real contacts on both original lip walls, clear of the recess floor."""
        body = self.case[cycle]
        p, R = self.pose(body)
        valid = []
        for finger in [7, 8]:
            path = '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{finger}']
            sample = self.contacts.contact_samples.get(' | '.join(sorted([path, body.prim_path])))
            points = [] if sample is None else sample['points']
            valid.append(any((point['normal_force_N'] > 0.05 and abs(((np.asarray(point['position_m']) - p) @ R)[0]) < 0.012 and (-0.061 < ((np.asarray(point['position_m']) - p) @ R)[1] < -0.045) and (0.0075 < ((np.asarray(point['position_m']) - p) @ R)[2] < 0.012) and (abs((np.asarray(point['normal']) @ R)[1]) > 0.65) for point in points)))
        return all(valid)

    def payload_native_valid(self, o, cycle, pans=None):
        """`pans`: {colour: pan index} of the pans that fill this case (the judge's; any pan of
        a colour may). Default: the pans the recorded controller uses."""
        if pans is not None:
            seats = [o['pans'][f'{color}_{j}'].get('cases', {}).get(str(cycle)) for color, j in pans.items()]
            if len(pans) != 4 or any((seat is None or not seat['contained'] or not (seat['support_N'] > 0.0001 or seat['touching']) for seat in seats)) or any((max(o['pans'][f'{color}_{j}']['finger_forces_N']) >= 0.01 for color, j in pans.items())):
                return False
            return self.case_delivered(o, cycle)
        return self.case_delivered(o, cycle) and all((self.contained(cycle, color, self.pan[color, 0 if cycle == 0 else 2]) and (o['pans'][f'{color}_{(0 if cycle == 0 else 2)}']['case_support_N'] > 0.0001 or o['pans'][f'{color}_{(0 if cycle == 0 else 2)}']['case_touching']) and (max(o['pans'][f'{color}_{(0 if cycle == 0 else 2)}']['finger_forces_N']) < 0.01) for color in range(4)))

    def case_delivered(self, o, cycle):
        """The case rests released on its output tray, clear of its nest, which has not moved."""
        case = o['cases'][str(cycle)]
        fixture = o['carriers'][str(cycle)]
        fixture_origin = np.array([-0.5 if cycle == 0 else 0.5, 0.43, 0.9386091])
        return bool(self.output_contained(cycle) and (case['output_support_N'] > 0.005 or case['output_touching']) and (case['carrier_support_N'] < 0.01) and (not case['carrier_touching']) and (max(case['hand_forces_N']) < 0.01) and (np.linalg.norm(np.asarray(fixture['position']) - fixture_origin) < 1e-05) and (Rotation.from_matrix(np.asarray(fixture['rotation'])).magnitude() < 1e-05))

    def output_contained(self, cycle):
        p, R = self.pose(self.case[cycle])
        points = self.case_envelopes[cycle] @ R.T + p
        local = (points[:, :2] - self.output_roots[cycle][:2]) @ self.output_rotations[cycle][:2, :2]
        floor = self.output_floors[cycle]
        return bool(np.all(abs(local) <= [0.135, 0.19]) and points[:, 2].min() >= floor - 0.0002 and (points[:, 2].min() <= floor + 0.003) and (points[:, 2].max() < floor + np.ptp(self.case_envelopes[cycle][:, 2]) + 0.005) and (Rotation.from_matrix(R).magnitude() < np.deg2rad(1)))

    def case_in_carrier(self, cycle):
        p, R = self.pose(self.case[cycle])
        vp, vR = self.pose(self.carrier[cycle])
        points = (self.case_envelopes[cycle] @ R.T + p - vp) @ vR
        return bool(np.all(abs(points[:, :2]) <= [0.056, 0.125]) and points[:, 2].min() >= self.carrier_metadata['carrier_floor_local_z_m'] + self.carrier_metadata['case_support_rise_m'] - 0.0002 and (points[:, 2].min() <= self.carrier_metadata['carrier_floor_local_z_m'] + self.carrier_metadata['case_support_rise_m'] + 0.002) and (points[:, 2].max() < self.carrier_metadata['carrier_floor_local_z_m'] + self.carrier_metadata['case_support_rise_m'] + 0.14))

class PaletteArmOperation:
    """One native arm using a dedicated side of the transverse feeders."""

    def __init__(self, scene, side):
        self.s, self.side = (scene, side)
        self.cycle = 0 if side == 'left' else 1
        self.motion = ArmMotion(scene.dual, side)
        self.motion.position_gain = 1.5
        self.motion.orientation_feedback = True
        self.colors = [0, 2, 1, 3] if side == 'left' else [1, 3, 0, 2]
        self.index = 0
        self.phase = 'park'
        self.entered = 0.0
        self.travel = 0.03
        self.since = None
        self.complete = False
        self.park = np.array([-0.56 if side == 'left' else 0.56, 0.36, 1.1])
        self.kind = 'pan'
        self.b = None
        self.lost_since = None
        self.output_window = []
        self.pan_window = []

    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.events[f'{self.side}_{self.index}_{phase}_s'] = self.entered
        print('PHASE', self.side, self.index, phase, self.entered, flush=True)

    def start(self):
        if self.index < 4:
            self.kind = 'pan'
            self.color = self.colors[self.index]
            self.b = self.s.pan[self.color, 0 if self.side == 'left' else 2]
            self.R = self.motion.R
            self.travel = 0.016
            self.grip_offset = np.array([0, 0, 0.011])
        else:
            self.kind = 'case'
            self.b = self.s.case[self.cycle]
            self.R = self.motion.R.copy()
            self.handle_R = Rotation.from_euler('z', 180 if self.side == 'left' else 0, degrees=True).as_matrix() @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
            self.handle_R = self.s.pose(self.b)[1] @ self.handle_R
            self.travel = self.s.resolved['case_grip']['opening_per_slider_m']
            self.grip_offset = np.asarray(self.s.resolved['case_grip']['tool_offset_m'])
            robot = self.s.dual.robots[self.side]
            original_efforts = np.asarray(robot._articulation_view._physics_view.get_dof_max_forces())[0].copy()
            kp = np.array([700 if int(name[5:]) <= 6 else self.s.resolved['case_grip']['finger_position_stiffness_N_m'] for name in self.s.dual.names[self.side]])
            kd = np.array([70 if int(name[5:]) <= 6 else self.s.resolved['case_grip']['finger_damping_N_s_m'] for name in self.s.dual.names[self.side]])
            robot.get_articulation_controller().set_gains(kps=kp, kds=kd)
            native_efforts = np.asarray(robot._articulation_view._physics_view.get_dof_max_forces())[0].copy()
            assert np.array_equal(original_efforts, native_efforts)
            for finger in [7, 8]:
                assert abs(native_efforts[self.s.dual.names[self.side].index(f'joint{finger}')] - 20) < 1e-06
            self.s.events[f'{self.side}_direct_case_drive_gains'] = dict(time_s=float(self.s.world.current_time), stiffness_N_m=self.s.resolved['case_grip']['finger_position_stiffness_N_m'], damping_N_s_m=self.s.resolved['case_grip']['finger_damping_N_s_m'], original_effort_cap_N=20.0, native_effort_caps_by_joint=dict(zip(self.s.dual.names[self.side], native_efforts.tolist())), scope='Position-drive gain change for the narrow original front lip only; original slider travel, speed and effort caps unchanged. Pan assembly keeps prior gains.')
            p, R = self.s.pose(self.b)
            self.handle_safe = np.array([-0.5 if self.side == 'left' else 0.5, 0.38, 1.03])
        self.since = None
        self.lost_since = None
        self.pan_window.clear()
        self.transition(('source_front' if self.index == 0 else 'source_lane') if self.kind == 'pan' else 'handle_prepose')

    def move_case(self, target_root, target_R, speed):
        actual_tool_R = self.s.dual.tool_world(self.side)[1]
        desired_R = target_R @ self.grasp_rotation.T
        delta = Rotation.from_matrix(desired_R @ self.R.T).as_rotvec()
        if Rotation.from_matrix(self.R @ actual_tool_R.T).magnitude() < 0.05:
            self.R = Rotation.from_rotvec(delta * min(1, self.s.resolved.get('carrier_handling_rotation_rate_rad_s', 0.4) / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12))).as_matrix() @ self.R
        arrived = self.motion.move(np.asarray(target_root) - self.R @ self.offset, speed, self.R)
        p, R = self.s.pose(self.b)
        return arrived and np.linalg.norm(delta) < 0.005 and (np.linalg.norm(p - target_root) < 0.004) and (Rotation.from_matrix(target_R @ R.T).magnitude() < 0.03)

    def step(self, o):
        s = self.s
        t = o['time_s']
        side = self.side
        m = self.motion
        if self.complete:
            return
        if self.phase == 'park':
            if m.move(self.park, 0.12):
                self.start()
            s.dual.jaw_travel(side, 0.03)
            return
        p, bodyR = s.pose(self.b)
        grip = p + (self.grip_offset if self.kind == 'pan' else bodyR @ self.grip_offset)
        if self.kind == 'case' and self.phase in ['lift', 'case_balance', 'case_front', 'case_clear', 'place']:
            forces = o['cases'][str(self.cycle)]['finger_forces_N'][side]
            if min(forces) < s.resolved['case_grip']['bilateral_force_target_N']:
                self.travel = max(0, self.travel - s.resolved['carrier_close_command_rate_m_s'] / CONTROL_HZ)
        if self.phase in ['lift', 'case_balance', 'case_front', 'case_clear', 'transfer', 'place', 'clear_front', 'clear_side']:
            tool, toolR = s.dual.tool_world(side)
            held = np.linalg.norm(p - (tool + toolR @ self.offset)) < 0.004 if self.kind == 'pan' else np.linalg.norm(p + bodyR @ self.held_point - tool) < 0.004
            self.lost_since = None if held else self.lost_since if self.lost_since is not None else t
            if self.lost_since is not None and t - self.lost_since > 0.2:
                raise RuntimeError(f'{side} lost native free {self.kind} grasp')
        if self.phase == 'source_front':
            if m.move([self.park[0], 0.38, 1.03], 0.12, self.R):
                self.transition('source_lane')
        elif self.phase == 'source_lane':
            if m.move([p[0], 0.38, 1.03], 0.14, self.R):
                self.transition('approach')
        elif self.phase == 'handle_prepose':
            if m.move(self.handle_safe, 0.12, self.R):
                self.transition('handle_rotate')
        elif self.phase == 'handle_rotate':
            delta = Rotation.from_matrix(self.handle_R @ self.R.T).as_rotvec()
            if Rotation.from_matrix(self.R @ s.dual.tool_world(side)[1].T).magnitude() < 0.05:
                self.R = Rotation.from_rotvec(delta * min(1, s.resolved.get('carrier_handling_rotation_rate_rad_s', 0.4) / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12))).as_matrix() @ self.R
            if m.move(self.handle_safe, 0.035, self.R) and np.linalg.norm(delta) < 0.005:
                self.transition('case_pregrasp')
        elif self.phase == 'case_pregrasp':
            if m.move(grip + [0, 0, 0.025], 0.1, self.R):
                self.transition('case_lower')
        elif self.phase == 'case_lower':
            arrived = m.move(grip, 0.025, self.R)
            gate = s.resolved['case_grip']['alignment']
            tool, toolR = s.dual.tool_world(side)
            aligned = arrived and np.linalg.norm(tool - grip) <= gate['position_m'] and (Rotation.from_matrix(self.R @ toolR.T).magnitude() <= gate['orientation_rad'])
            self.since = (self.since if self.since is not None else t) if aligned else None
            if self.since is not None and t - self.since >= gate['dwell_s']:
                self.close_goal = grip.copy()
                self.since = None
                self.transition('close')
        elif self.phase == 'approach':
            clearance = 0.025 if self.kind == 'pan' else 0.075
            if m.move(grip + [0, 0, clearance], 0.14, self.R):
                self.transition('lower')
        elif self.phase == 'lower':
            if m.move(grip, 0.05, self.R):
                self.close_goal = grip.copy()
                self.transition('close')
        elif self.phase == 'close':
            m.move(self.close_goal, 0.025, self.R)
            forces = [s.contacts.force(self.b.prim_path, '/World/ARX/' + side + '/' + s.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]]
            both = min(forces) > (0.2 if self.kind == 'pan' else s.resolved['case_grip']['bilateral_force_target_N'])
            if self.kind == 'case':
                both = both and s.case_lip_contact_valid(self.cycle, side)
            if not both:
                close_rate = 0.004 if self.kind == 'pan' else s.resolved.get('carrier_close_command_rate_m_s', 0.006)
                self.travel = max(0, self.travel - close_rate / CONTROL_HZ)
            self.since = (self.since if self.since is not None else t) if both else None
            if self.since is not None and t - self.since > 0.1:
                tool, toolR = s.dual.tool_world(side)
                self.offset = toolR.T @ (p - tool)
                self.grasp_rotation = toolR.T @ bodyR
                if self.kind == 'case':
                    self.held_point = bodyR.T @ (tool - p)
                    if np.linalg.norm(self.held_point - self.grip_offset) > 0.015:
                        raise RuntimeError('Native contact is not near intended original case-front-lip grasp')
                    s.events[f'case{self.cycle}_direct_grasp'] = dict(time_s=t, bilateral_forces_N=forces, original_case_contact_points={key: value for key, value in s.contacts.contact_samples.items() if self.b.prim_path in key and '/World/ARX/' + side + '/' in key})
                self.lift_start = tool.copy()
                self.transition('lift')
        elif self.phase == 'lift':
            lift = 0.025 if self.kind == 'pan' else 0.04
            if m.move(self.lift_start + [0, 0, lift], 0.075, self.R):
                self.since = None
                self.transition('clear_front' if self.kind == 'pan' else 'case_balance')
        elif self.phase == 'case_balance':
            arrived = m.move(self.lift_start + [0, 0, 0.04], 0.04, self.R)
            valid = arrived and all((s.contained(self.cycle, c, s.pan[c, 0 if side == 'left' else 2]) for c in range(4))) and (o['cases'][str(self.cycle)]['carrier_support_N'] < 0.01) and (not o['cases'][str(self.cycle)]['carrier_touching'])
            self.since = (self.since if self.since is not None else t) if valid else None
            if self.since is not None and t - self.since > 0.5:
                tool, toolR = s.dual.tool_world(side)
                self.offset = toolR.T @ (p - tool)
                self.grasp_rotation = toolR.T @ bodyR
                self.case_high_root = p.copy()
                if s.mode == 'grasp':
                    self.complete = True
                    return
                self.transition('case_front')
        elif self.phase == 'case_front':
            goal = np.r_[s.resolved['case_transfer_xy_m'][self.cycle][0], self.case_high_root[2]]
            if self.move_case(goal, np.eye(3), 0.1):
                self.transition('case_clear')
        elif self.phase == 'case_clear':
            goal = np.r_[s.resolved['case_transfer_xy_m'][self.cycle][1], self.case_high_root[2]]
            if self.move_case(goal, np.eye(3), 0.1):
                self.transition('place')
        elif self.phase == 'clear_front':
            tool = s.dual.tool_world(side)[0]
            goal = np.array([self.lift_start[0], 0.38, 1.03])
            if m.move(goal, 0.14, self.R):
                tool, toolR = s.dual.tool_world(side)
                self.grasp_rotation = toolR.T @ bodyR
                self.offset = toolR.T @ (p - tool)
                if s.mode == 'grasp':
                    self.complete = True
                    return
                self.transition('transfer')
        elif self.phase == 'clear_side':
            goal = np.array([-0.5 if side == 'left' else 0.5, 0.38, 1.03])
            if m.move(goal, 0.14, self.R):
                s.source_owner = None
                if s.mode == 'grasp':
                    self.complete = True
                    return
                self.transition('transfer')
        elif self.phase in ['transfer', 'place', 'release', 'clear_release', 'settle', 'retreat', 'retreat_front', 'case_unload']:
            if self.kind == 'pan':
                cp, cR = s.pose(s.case[self.cycle])
                slot = s.asset_manifest['slots'][self.color]
                target = cp + cR @ np.r_[slot['slot_xy_m'], slot['resting_pan_center_z_m']]
                if self.phase in ['transfer', 'place']:
                    self.R = cR @ self.grasp_rotation.T
            else:
                target = s.output_roots[self.cycle]
            goal = target - self.R @ self.offset
            if self.phase in ['release', 'clear_release', 'settle', 'retreat', 'retreat_front', 'case_unload']:
                goal = self.release_goal
            if self.phase == 'transfer':
                clearance = 0.03 if self.kind == 'pan' else 0.055
                if m.move(goal + [0, 0, clearance], 0.14, self.R):
                    self.transition('place')
            elif self.phase == 'place':
                if self.kind == 'case':
                    arrived = self.move_case(target, np.eye(3), 0.025)
                    native = o['cases'][str(self.cycle)]
                    gate = s.resolved['case_grip']['supported_unload']
                    supported = arrived and native['output_support_N'] >= gate['native_output_support_N'] and s.output_contained(self.cycle) and all((s.contained(self.cycle, c, s.pan[c, 0 if side == 'left' else 2]) for c in range(4)))
                    self.since = (self.since if self.since is not None else t) if supported else None
                    if self.since is not None and t - self.since >= gate['support_dwell_s']:
                        self.release_goal = s.dual.tool_world(side)[0].copy()
                        s.events[f'case{self.cycle}_native_support_before_release'] = dict(time_s=t, output_support_N=native['output_support_N'], case_position=native['position'], case_rotation=native['rotation'], frozen_tool_position=self.release_goal.tolist())
                        self.transition('case_unload')
                    s.dual.jaw_travel(side, self.travel)
                    if t - self.entered > 18:
                        raise RuntimeError(f'{side} case output placement timed out')
                    return
                release_height = s.resolved['pan_release_alignment']['floor_clearance_m'] if self.kind == 'pan' else 0.002
                waypoint = goal + [0, 0, release_height]
                arrived = m.move(waypoint, 0.045, self.R)
                if self.kind == 'pan':
                    gate = s.resolved['pan_release_alignment']
                    tool, actual_R = s.dual.tool_world(side)
                    aligned = arrived and np.linalg.norm(tool - waypoint) < gate['position_m'] and (Rotation.from_matrix(self.R @ actual_R.T).magnitude() < gate['orientation_rad']) and (tool[2] - 0.0133 > (cp + cR @ np.r_[slot['slot_xy_m'], 0.01156218])[2] + 0.002)
                    self.since = (self.since if self.since is not None else t) if aligned else None
                    arrived = self.since is not None and t - self.since >= gate['dwell_s']
                if arrived:
                    self.release_goal = goal.copy()
                    self.transition('release')
            elif self.phase == 'case_unload':
                m.move(goal, 0.015, self.R)
                gate = s.resolved['case_grip']['supported_unload']
                self.travel = min(gate['released_slider_travel_m'], self.travel + gate['opening_rate_m_s'] / CONTROL_HZ)
                native = o['cases'][str(self.cycle)]
                released = self.travel >= gate['released_slider_travel_m'] and max(native['hand_forces_N']) < 0.01 and (native['output_support_N'] > 0.005) and s.output_contained(self.cycle)
                self.since = (self.since if self.since is not None else t) if released else None
                if self.since is not None and t - self.since >= gate['support_dwell_s']:
                    self.since = None
                    self.transition('clear_release')
            elif self.phase == 'release':
                release_height = s.resolved['pan_release_alignment']['floor_clearance_m'] if self.kind == 'pan' else 0.002
                m.move(goal + [0, 0, release_height], 0.04, self.R)
                release_travel = 0.016 if self.kind == 'pan' else s.resolved['case_grip']['opening_per_slider_m']
                self.travel = min(release_travel, self.travel + 0.04 / CONTROL_HZ)
                if self.travel >= release_travel:
                    self.since = None
                    self.transition('clear_release')
            elif self.phase == 'clear_release':
                if m.move(goal + [0, 0, 0.018], 0.07, self.R):
                    self.since = None
                    self.transition('settle')
            elif self.phase == 'settle':
                m.move(goal + [0, 0, 0.018], 0.04, self.R)
                support = s.contacts.force(self.b.prim_path, s.case[self.cycle].prim_path if self.kind == 'pan' else '/World/Output/')
                touching = s.contacts.touching(self.b.prim_path, s.case[self.cycle].prim_path) if self.kind == 'pan' else support > 0.005
                valid = support > 0.005 or touching
                if self.kind == 'pan':
                    pan = o['pans'][f'{self.color}_{(0 if side == "left" else 2)}']
                    valid = valid and s.contained(self.cycle, self.color, self.b) and (max(pan['finger_forces_N']) < 0.01)
                    gate = s.resolved['pan_release_stability']
                    if valid:
                        cp, cr = s.pose(s.case[self.cycle])
                        self.pan_window.append((t, np.array([p, cp]), np.array([bodyR, cr])))
                    else:
                        self.pan_window.clear()
                    dwell = gate['dwell_s']
                    while len(self.pan_window) > 1 and self.pan_window[1][0] <= t - dwell:
                        self.pan_window.pop(0)
                    if len(self.pan_window) >= 2 and t - self.pan_window[0][0] >= dwell:
                        positions = np.array([v[1] for v in self.pan_window])
                        rotations = np.array([v[2] for v in self.pan_window])
                        span = float(np.linalg.norm(np.ptp(positions, axis=0), axis=1).max())
                        angle = float(Rotation.from_matrix((rotations @ rotations[0].transpose(0, 2, 1)).reshape(-1, 3, 3)).magnitude().max())
                        if span <= gate['position_span_m'] and angle <= np.deg2rad(gate['orientation_from_first_deg']):
                            s.events[f'pan{self.cycle}_{self.color}_native_window'] = dict(time_s=t, samples=len(self.pan_window), position_span_m=span, orientation_from_first_rad=angle, raw_velocity_gate=False)
                            s.credited.add(f'{self.cycle}_{self.color}')
                            self.transition('retreat')
                    s.dual.jaw_travel(side, self.travel)
                    if t - self.entered > 18:
                        raise RuntimeError(f'{side} operation{self.index}: timeout {self.phase}')
                    return
                else:
                    valid = s.payload_native_valid(o, self.cycle) and (s.mode == 'delivery' or all((f'{self.cycle}_{c}' in s.credited for c in range(4))))
                    bodies = [o['cases'][str(self.cycle)]]
                    bodies += [o['pans'][f'{c}_{(0 if self.cycle == 0 else 2)}'] for c in range(4)]
                    if valid:
                        self.output_window.append((t, np.array([b['position'] for b in bodies]), np.array([b['rotation'] for b in bodies])))
                    else:
                        self.output_window.clear()
                    dwell = s.resolved['carrier_output_stability']['dwell_s']
                    while len(self.output_window) > 1 and self.output_window[1][0] <= t - dwell:
                        self.output_window.pop(0)
                    if len(self.output_window) >= 2 and t - self.output_window[0][0] >= dwell:
                        positions = np.array([v[1] for v in self.output_window])
                        rotations = np.array([v[2] for v in self.output_window])
                        span = float(np.linalg.norm(np.ptp(positions, axis=0), axis=1).max())
                        angular = float(Rotation.from_matrix((rotations @ rotations[0].transpose(0, 2, 1)).reshape(-1, 3, 3)).magnitude().max())
                        stable = span <= s.resolved['carrier_output_stability']['position_span_m'] and angular <= np.deg2rad(s.resolved['carrier_output_stability']['orientation_from_first_deg'])
                        if stable:
                            s.events[f'output{self.cycle}_native_window'] = dict(time_s=t, samples=len(self.output_window), position_span_m=span, orientation_from_first_rad=angular, raw_velocity_gate=False)
                            s.transport_verified.add(self.cycle)
                            if s.mode == 'delivery':
                                s.events[f'component_delivery{self.cycle}_s'] = t
                            else:
                                s.delivered.add(self.cycle)
                                s.events[f'delivery{self.cycle}_s'] = t
                            self.transition('retreat')
                    s.dual.jaw_travel(side, self.travel)
                    if t - self.entered > 18:
                        raise RuntimeError(f'{side} operation{self.index}: timeout {self.phase}')
                    return
            elif self.phase == 'retreat':
                clearance = 0.035 if self.kind == 'pan' else s.resolved['case_grip']['output_withdraw_clearance_m']
                if m.move(goal + [0, 0, clearance], 0.12, self.R):
                    if self.kind == 'case':
                        self.complete = True
                        self.transition('complete')
                    else:
                        self.transition('retreat_front')
            elif self.phase == 'retreat_front':
                if m.move(np.array([goal[0], 0.38, goal[2] + 0.035]), 0.14, self.R):
                    self.index += 1
                    if self.index == 5:
                        self.complete = True
                        self.transition('complete')
                    else:
                        self.start()
        s.dual.jaw_travel(side, self.travel)
        if t - self.entered > 18:
            raise RuntimeError(f'{side} operation{self.index}: timeout {self.phase}')

class PaletteARXController:

    def __init__(self, scene, mode):
        self.s, self.mode = (scene, mode)
        scene.source_owner = None
        scene.mode = mode
        self.arms = [PaletteArmOperation(scene, side) for side in ['left', 'right']]
        if mode == 'delivery':
            for arm in self.arms:
                arm.index = 4
        self.phase = 'assembly'
        self.failure = None
        self.complete = False

    def step(self, o):
        if self.complete or self.failure or self.mode in ['passive', 'absent', 'seat']:
            return
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if o['time_s'] < self.s.resolved['start_s']:
                return
            for arm in self.arms:
                arm.step(o)
            self.phase = ' / '.join((a.side + ':' + a.phase for a in self.arms))
            if all((a.complete for a in self.arms)):
                self.complete = True
                self.s.physical_completed = len(self.s.delivered) == 2
                self.s.success = self.s.final_valid(o)
                self.s.manipulation_success = len(self.s.transport_verified) == 2 and all((self.s.payload_native_valid(o, c) for c in [0, 1]))
                self.s.cue_state = 'DELIVERED' if self.s.success else 'TRANSPORT COMPONENT VERIFIED' if self.s.manipulation_success else 'GRASP PROBE'
            budget = self.s.resolved['budget_s']
            if o['time_s'] > budget and (not self.complete):
                raise RuntimeError(f'{budget:g}s throughput budget expired')
        except RuntimeError as exc:
            self.failure = str(exc)
            print('FAILURE', self.failure, flush=True)
