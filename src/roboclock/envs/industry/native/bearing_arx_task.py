"""Approved factory46: native gravity return, post-entry arrest and recovery.

Original scored bearing/chute/tray, unchanged complete Dual ARX X5. Gate is
an upstream release fixture; no contact stop inside the scored safety area.
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
    StatusDisplay,
    add_mounts,
    collide,
    remove_preview,
)
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2/task46'))
# Common rig (rig.py). Recorded: left [0.65, 0.59, 0.98], right [1.05, 0.59, 0.98] (0.40 m).
# Centered on the rework tray's x and 0.03 m further from the lane: the head camera sees
# the capture zone, the lane and the tray's far half. The right brake's vertical withdraw
# (downward tool, joint6 within +/-2 rad) bounds how far back the rig can go (y=0.50 was
# unreachable at brake_withdraw); the left outer-race pickup stays inside the URDF reach.
MOUNTS = rig.mounts([0.85, 0.56, 0.98])
BEARING = '/World/NativeBearing'

def resolve_bearing(seed=0, *, layout=False, timing=False, physical=False):
    rs = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(3)]
    d = dict(task_id=46, embodiment='dual_arx_x5', seed=seed, release_x_m=float(rs[0].uniform(0.815, 0.825)) if layout else 0.82, lane_y_m=float(rs[0].uniform(0.908, 0.912)) if layout else 0.91, gate_release_s=float(rs[1].uniform(7, 9)) if timing else 8.0, mass_scale=float(rs[2].uniform(0.9, 1.1)) if physical else 1.0, dynamic_friction=float(rs[2].uniform(0.23, 0.27)) if physical else 0.25, bounds=dict(release_x_m=[0.815, 0.825], lane_y_m=[0.908, 0.912], gate_release_s=[7, 9], mass_scale=[0.9, 1.1], dynamic_friction=[0.23, 0.27]), randomization=dict(layout=layout, timing=timing, physical=physical), zone_x_m=[0.9, 1.2], response_deadline_s=3.0, physics_hz=robodojo_rig.PHYSICS_HZ, bearing_collision_method='sdf', bearing_sdf_resolution=1024, capture_above_reference_center_m=0.042, capture_close_rate_m_s=0.028, capture_minimum_command_m=0.0002, brake_yaw_deg=0.0, recovery_keep_pickup_orientation=True, recovery_output_pitch_deg=75.0, arrest_policy='Post-entry contact-stable stop in any supported orientation within3s; at least one original right fingertip mechanically brakes against native lane support. Right-arm braking remains continuous while left-arm outer-race grasp establishes strong bilateral pickup before brake withdrawal, free lift and recovery.', recovery_preparation='two_arm_continuous_brake_outer_race_handoff', handoff=dict(pickup_pitch_deg=30.0, approach_from_bore_deg=-12.15, approach_yaw_max_deg=72.0, pickup_above_center_m=0.025, bore_horizontal_offset_m=0.006, right_park_yaw_deg=60.0, policy='Side entry above intact chute guides; continuous right brake until measured strong left outer-race grasp, narrow vertical brake withdrawal then clear park.'), stop_minimum_each_finger_N=0.2, tracked_regrasp=dict(preopen_travel_m=0.01, open_travel_m=0.016, clear_height_m=0.05, grip_height_m=0.042, prediction_s=0.05, maximum_predict_speed_m_s=0.4, motion_command_speed_m_s=0.25, alignment_m=0.002, minimum_pickup_finger_N=10.0), capture_geometry='Original fingertip reach and42mm reference height target the original outer race. Actual failedSDF trace/FK has submicrometer tool agreement and shows fingers remain axially outside the part during interception;28mm/s closure is below the38mm/s native cap and compensates observed closing delay. Native contact/penetration checks remain required.', arrest_dwell_s=0.15, tray_retention_stability=dict(dwell_s=0.5, position_span_m=0.0002, orientation_from_first_deg=0.5, scope='Free released bearing remains on the actual tray floor, inside its complete usable region and without hand contact throughout a measured0.5s pose window. Actual pose variation is evaluated directly, separately from the unchanged arrest/lift windows; raw constraint-step velocities remain logged.'), minimum_arrest_finger_N=10.0, holding_close_rate_m_s=0.004, recovery_orientation_rate_rad_s=0.15, preparation_orientation_rate_rad_s=0.3, recovery_output_yaw_deg=105.0, recovery_target_xy_m=[0.87, 0.67], recovery_tray_center_m=[0.85, 0.65, 0.9075], recovery_tray_yaw_deg=0.0, recovery_tray_size_m=[0.23, 0.29, 0.04], recovery_tray_inner_bounds_xy_m=[[0.75225, 0.54], [0.94775, 0.76]], recovery_control='Continue bounded closing toward source20N native effort cap during loaded motion, instead of freezing a contact-onset position target. Original0.04m/s and all geometry/deadlines retained.', arrest_linear_speed_m_s=0.02, arrest_angular_speed_rad_s=0.2, arrest_pose_stability=dict(position_span_m=0.0002, orientation_from_first_deg=0.2, scope='Continuous measured pose window with native lane support and at least one real braking fingertip contact; separate strong bilateral left-hand pickup remains mandatory. Raw contact-step velocities are logged diagnostics, not zero-motion claims.'), retained_grasp=dict(center_to_tool_bound_m=0.05, minimum_each_finger_N=0.2, loss_dwell_s=0.2, provenance='Original35mm bearing radius + original12.57mm fingertip reach + contact/edge clearance. A round bearing may pivot/settle along the original fingers; native bilateral contacts and whole-part lift/retention remain required.'), slope_deg=8.2, runtime_joint_velocity_limit_rad_s=0.73, runtime_finger_velocity_limit_m_s=0.038, machine_outfeed=dict(enabled=True, source_x_m=float(rs[0].uniform(-1.705, -1.695)) if layout else -1.7, belt_start_x_m=-1.82, junction_x_m=0.78, surface_speed_m_s=float(rs[2].uniform(0.33, 0.37)) if physical else 0.35, arrival_dwell_s=0.5, scope='Original free bearing transports from the supplied left machine opening on a native conveyor, settles at the native metering gate, then rolls into the unchanged safety/capture zone. No product writes after reset.'), assumptions='Candidate0.35kg bearing/friction and original8.2deg gravity lane. Native lifting upstream gate sets release time; no product pose/velocity writes after reset. Camera/quality limits are demo settings, not certified safety limits.', scoring='Measured leading-edge zone entry to native stable arrest wholly inside safety zone within3s; separate released whole-bearing tray retention.')
    d['bounds']['machine_outfeed_source_x_m'] = [-1.705, -1.695]
    # 250 Hz (robodojo_rig): at 0.3 mm contact offsets the released bearing jittered on the
    # rework tray and never held the 0.5 s retention window; 2 mm offsets with a 0.3 m/s
    # depenetration cap keep the lane roll, arrest and tray rest stable.
    d.update(bearing_contact_offset_m=0.002, static_contact_offset_m=0.002, bearing_max_depenetration_m_s=0.3)
    # robodojo_rig head view (60 deg down): the tray's near 5.5 cm fell below the image, so
    # the reused tray is 0.235 m deep (was 0.29 m) with its far edge unchanged at 0.795 m.
    d.update(recovery_tray_center_m=[0.85, 0.6775, 0.9075], recovery_tray_size_m=[0.23, 0.235, 0.04],
             recovery_tray_inner_bounds_xy_m=[[0.75225, 0.595], [0.94775, 0.76]])
    # 250 Hz: standing on its rim on the flat tray floor, the released bearing kept rocking and
    # rolling (0.05-0.2 mm per control tick, 0.1-0.3 rad/s) and broke the 0.2 mm / 0.5 deg
    # retention window again after it first held. A rim cradle at the release point (two
    # 1.5 mm rails, inner edges 8 mm either side of the rim's lowest point) holds it; the rim
    # rests on both edges 0.6 mm above the floor (inside the 2 mm floor band).
    d.update(recovery_cradle=dict(center_xy_m=[0.87, 0.672], inner_half_gap_m=0.008, rail_width_m=0.006,
                                  rail_height_m=0.0015, rail_length_m=0.07))
    d['bounds']['machine_outfeed_surface_speed_m_s'] = [0.33, 0.37]
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d

class BearingARXScene:
    title = 'BEARING RETURN / ARREST + RECOVER'

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .factory_revisions import build_factory_revision
        from .remaining_environments import TABLE, TRAY
        from .v2.geometry import body_envelope_points
        self.base = build_factory_revision(46, output, (1280, 800))
        c = self.base.cell
        self.world, self.stage = (c.world, c.stage)
        remove_preview(c, '/World/Recovery/Bench')
        c.asset('Recovery/Bench', TABLE, [0.65, 0.55, 0.43], [1.3, 0.55, 0.86])
        self.stage.RemovePrim('/World/Recovery/Top')
        c.box('Recovery/Top', [0.65, 0.55, 0.875], [1.3, 0.55, 0.025], 'steel')
        remove_preview(c, '/World/Recovery/ReworkTray')
        c.asset('Recovery/ReworkTray', TRAY, resolved['recovery_tray_center_m'], resolved['recovery_tray_size_m'], material='steel', yaw=np.deg2rad(resolved['recovery_tray_yaw_deg']))
        cradle = resolved.get('recovery_cradle')
        if cradle:
            # Two low steel rails across the rolling direction (x) on the tray floor: the
            # released bearing stands on its rim and rests on their inner edges instead of
            # rolling freely on the flat floor. Part of the tray (counted as tray contact).
            floor = 0.9075 + -0.0057221898 / 0.028 * 0.04
            width, height, length = cradle['rail_width_m'], cradle['rail_height_m'], cradle['rail_length_m']
            for name, sign in (('Near', -1), ('Far', 1)):
                x = cradle['center_xy_m'][0] + sign * (cradle['inner_half_gap_m'] + width / 2)
                c.box('Recovery/ReworkTrayCradle/' + name, [x, cradle['center_xy_m'][1], floor + height / 2], [width, length, height], 'steel')
        self.stage.RemovePrim('/World/Signs/Rework')
        c.sign('Rework', 'RECOVERED / REWORK', [0.85, 0.552, 0.945], [0.42, 0.07])
        self.world.set_simulation_dt(physics_dt=1 / resolved['physics_hz'], rendering_dt=1 / CONTROL_HZ)
        stage = self.stage
        self.resolved, self.events = (resolved, {})
        self.success = self.physical_completed = self.manipulation_success = False
        self.tray_retention_valid = False
        self.cue_state = 'UPSTREAM HOLD'
        self.entry_at = None
        self.first_hand_at = None
        for i in range(3):
            remove_preview(c, f'/World/Queue/Bearing{i}')
        stage.RemovePrim('/World/Safety/EndMarker')
        c.box('Safety/EndMarker', [1.22, 0.91, 0.985 - (1.22 - 0.7) * np.tan(np.deg2rad(8.2)) + 0.0008], [0.008, 0.22, 0.0016], 'red')
        material = PhysicsMaterial('/World/BearingContact', static_friction=0.35, dynamic_friction=resolved['dynamic_friction'], restitution=0)
        root = UsdGeom.Xform.Define(stage, BEARING).GetPrim()
        file = ROOT / 'prepared/bearing.usd'
        root.GetReferences().AddReference(str(file))
        UsdPhysics.RigidBodyAPI.Apply(root)
        mass = 0.35 * resolved['mass_scale']
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass)
        ph = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        ph.CreateEnableCCDAttr(True)
        ph.CreateAngularDampingAttr(2.0)
        ph.CreateLinearDampingAttr(0.0)
        # 250 Hz contact stability (robodojo_rig physics rate); optional, declared in the spec.
        if resolved.get('bearing_max_depenetration_m_s') is not None:
            ph.CreateMaxDepenetrationVelocityAttr(resolved['bearing_max_depenetration_m_s'])
        ph.CreateSolverPositionIterationCountAttr(64)
        ph.CreateSolverVelocityIterationCountAttr(16)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        for mesh in Usd.PrimRange(root):
            if mesh.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(mesh).CreateCollisionEnabledAttr(True)
                method = resolved.get('bearing_collision_method', 'convexDecomposition')
                UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr(method)
                if method == 'sdf':
                    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(mesh).CreateSdfResolutionAttr(resolved['bearing_sdf_resolution'])
                else:
                    assert method == 'convexDecomposition'
                    decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(mesh)
                    decomp.CreateMaxConvexHullsAttr(128)
                    decomp.CreateVoxelResolutionAttr(2000000)
                    decomp.CreateErrorPercentageAttr(1)
                col = PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                col.CreateContactOffsetAttr(resolved.get('bearing_contact_offset_m', 0.0003))
                col.CreateRestOffsetAttr(0)
                UsdShade.MaterialBindingAPI.Apply(mesh).Bind(material.material, materialPurpose='physics')
        self.outfeed = None
        if resolved.get('machine_outfeed', {}).get('enabled', False):
            from .bearing_outfeed import BearingOutfeed
            self.outfeed = BearingOutfeed(self, c, material)
            self.initial = np.array([resolved['machine_outfeed']['source_x_m'], resolved['lane_y_m'], self.outfeed.top + 0.035 + 0.002])
        else:
            self.initial = np.array([resolved['release_x_m'], resolved['lane_y_m'], self.surface_z(resolved['release_x_m']) + 0.035 + 0.002])
        self.bearing = self.world.scene.add(SingleRigidPrim(BEARING, name='native_bearing', position=self.initial, mass=mass))
        self.envelope = body_envelope_points(stage, root)
        self.gate_position = np.array([resolved['release_x_m'] + 0.039, resolved['lane_y_m'], self.surface_z(resolved['release_x_m']) + 0.037])
        assembly = UsdGeom.Xform.Define(stage, '/World/HoldingGate')
        assembly.AddTranslateOp().Set(Gf.Vec3d(*self.gate_position))
        for name, m in [('base', 2.0), ('plate', 0.15)]:
            prim = UsdGeom.Xform.Define(stage, '/World/HoldingGate/' + name).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(m)
        plate = UsdGeom.Cube.Define(stage, '/World/HoldingGate/plate/ContactPlate')
        plate.CreateSizeAttr(1.0)
        plate.AddScaleOp().Set(Gf.Vec3f(0.008, 0.075, 0.08))
        UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(c.appearance.material('yellow').material)
        UsdShade.MaterialBindingAPI.Apply(plate.GetPrim()).Bind(material.material, materialPurpose='physics')
        rod = UsdGeom.Cylinder.Define(stage, '/World/HoldingGate/plate/PneumaticRod')
        rod.CreateAxisAttr('Z')
        rod.CreateRadiusAttr(0.006)
        rod.CreateHeightAttr(0.3)
        rod.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.19))
        UsdShade.MaterialBindingAPI.Apply(rod.GetPrim()).Bind(c.appearance.material('steel').material)
        gx = float(self.gate_position[0])
        bottom = self.surface_z(gx) - 0.02
        for i, y in enumerate([1.1]):
            c.box(f'HoldingGateHousing/Post{i}', [gx, y, (bottom + 1.27) / 2], [0.025, 0.025, 1.27 - bottom], 'frame')
        c.box('HoldingGateHousing/Bridge', [gx, 1.01, 1.27], [0.045, 0.235, 0.035], 'frame')
        c.pipe('HoldingGateHousing/PneumaticCylinder', [gx, 0.91, 1.25], [gx, 0.91, 1.51], 0.018, 'steel')
        anchor = UsdPhysics.FixedJoint.Define(stage, '/World/HoldingGate/Anchor')
        anchor.CreateBody1Rel().SetTargets(['/World/HoldingGate/base'])
        anchor.CreateLocalPos0Attr(Gf.Vec3f(*self.gate_position))
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/HoldingGate/LiftJoint')
        joint.CreateBody0Rel().SetTargets(['/World/HoldingGate/base'])
        joint.CreateBody1Rel().SetTargets(['/World/HoldingGate/plate'])
        joint.CreateAxisAttr('Z')
        joint.CreateLowerLimitAttr(0)
        joint.CreateUpperLimitAttr(0.15)
        joint.CreateCollisionEnabledAttr(False)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateTypeAttr('force')
        drive.CreateStiffnessAttr(8000)
        drive.CreateDampingAttr(400)
        drive.CreateMaxForceAttr(500)
        self.gate = self.world.scene.add(SingleArticulation('/World/HoldingGate', name='upstream_gate'))
        self.static_colliders = collide(stage, ['/World/Return/GravityLane', '/World/Return/Deck', '/World/Return/Guide', '/World/Recovery/Top', '/World/Recovery/ReworkTray', '/World/Recovery/ReworkTrayCradle', '/World/Collection/KLTTote', '/World/HoldingGateHousing/', *(['/World/MachineOutfeed/', '/World/Production/CNCLathe', '/World/RecoverySignal/Post', '/World/RecoverySignal/Housing'] if self.outfeed is not None else [])], material)
        offset = resolved.get('static_contact_offset_m')
        if offset is not None:
            for prim in stage.Traverse():
                if str(prim.GetPath()) in self.static_colliders or any(str(prim.GetPath()).startswith(c + '/') for c in self.static_colliders):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        PhysxSchema.PhysxCollisionAPI.Apply(prim).CreateContactOffsetAttr(offset)
        if resolved.get('scene_stabilization') is not None:
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    PhysxSchema.PhysxSceneAPI.Apply(prim).CreateEnableStabilizationAttr(bool(resolved['scene_stabilization']))
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS)
        pads = PhysicsMaterial('/World/BearingGripperPads', static_friction=0.95, dynamic_friction=0.85, restitution=0)
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/ARX/') and prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(pads.material, materialPurpose='physics')
                if '/link7/' in str(prim.GetPath()) or '/link8/' in str(prim.GetPath()):
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    contact.CreateContactOffsetAttr(0.0003)
                    contact.CreateRestOffsetAttr(0)
        self.velocity_overrides = []
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/ARX/') and prim.IsA(UsdPhysics.PrismaticJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(prim)
                imported = float(api.GetMaxJointVelocityAttr().Get())
                source = float(prim.GetAttribute('urdf:limit:velocity').Get())
                assert abs(source - 0.04) < 1e-06
                limit = resolved['runtime_finger_velocity_limit_m_s']
                assert 0 < limit < source
                api.CreateMaxJointVelocityAttr(limit)
                self.velocity_overrides.append(dict(joint=str(prim.GetPath()), kind='linear', imported_stage_units_s=imported, source_urdf_m_s=source, configured_m_s=limit, reason='Correct importer angular conversion on prismatic velocity; preserve source0.04m/s cap.'))
            if str(prim.GetPath()).startswith('/World/ARX/') and prim.IsA(UsdPhysics.RevoluteJoint):
                api = PhysxSchema.PhysxJointAPI.Apply(prim)
                old = float(api.GetMaxJointVelocityAttr().Get())
                assert abs(old - np.rad2deg(0.75)) < 0.001
                api.CreateMaxJointVelocityAttr(float(np.rad2deg(resolved['runtime_joint_velocity_limit_rad_s'])))
                self.velocity_overrides.append(dict(joint=str(prim.GetPath()), imported_deg_s=old, configured_rad_s=resolved['runtime_joint_velocity_limit_rad_s']))
        add_mounts(c, MOUNTS, base_z=0.8875)
        self.static_colliders += collide(stage, ['/World/ARXMounts/'], material)
        self.contacts = ContactBook(self.world, [BEARING], record_points=True)
        self.readback = NativeReadbackAudit(self.world, [self.bearing], 'bearing_native_readback')
        self.readback_checks = self.readback.checks
        c.sign('BearingHMI', 'UPSTREAM HOLD', [0.68, 1.13, 1.35], [0.92, 0.1])
        self.display = StatusDisplay(c, 'BearingHMI', {'hold': 'UPSTREAM HOLD', 'rolling': 'ROLLING / ARREST <=3 s', 'arrested': 'ARREST VERIFIED / RECOVER', 'complete': 'RECOVERED / REWORK'})
        self.world.reset()
        self.dual.initialize()
        self.initial_joint_positions = {}
        initial_left = [-1.3432699312104535, 1.4510008377136596, 0.4793951227809272, 0.24721799732654384, -0.7158796459393496, -0.5261140044078181]
        for side in ['left', 'right']:
            motion = ArmMotion(self.dual, side)
            bp, bR = self.dual.body_world(side, 'base_link')
            initial = self.dual.commands[side].copy()
            if side == 'left':
                q = np.asarray(initial_left)
            else:
                goal = np.array([1.04, resolved['lane_y_m'], self.surface_z(1.04) + 0.077])
                target = Rotation.from_euler('y', 90, degrees=True).as_matrix()
                profile = self.dual.manifest['profile']
                q, error = motion.kin.solve(bR.T @ (goal - bp), bR.T @ target, dict(zip(self.dual.names[side], initial)), profile['arm_lower_rad'], profile['arm_upper_rad'])
                assert error < 0.0001
            initial[motion.ids] = q
            for name in ['joint7', 'joint8']:
                initial[self.dual.names[side].index(name)] = 0.04 if side == 'left' else 0.02
            robot = self.dual.robots[side]
            robot.set_joints_default_state(positions=initial, velocities=np.zeros(8))
            robot.set_joint_positions(initial)
            robot.set_joint_velocities(np.zeros(8))
            self.dual.commands[side] = initial.copy()
            self.initial_joint_positions[side] = initial.tolist()
        self.active_side = 'right'
        self._tool_cache = None
        self._last_tool_readback_s = None
        self.bearing.initialize()
        self.readback.initialize()
        self.gate.initialize()
        self.native_collision_queries = None
        if resolved.get('bearing_collision_method') == 'sdf':
            import carb
            import omni.physx
            query = omni.physx.get_physx_scene_query_interface()
            position, quat = self.bearing.get_world_pose()
            rotation = Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix()
            rays = []
            for x, z in [(0, 0), (0.03, 0), (-0.03, 0), (0, 0.03), (0, -0.03)]:
                hits = []

                def hit(value):
                    if str(value.rigid_body) == BEARING:
                        hits.append(dict(distance_m=float(value.distance)))
                    return True
                origin = np.asarray(position) + rotation @ np.array([x, -0.08, z])
                direction = rotation @ np.array([0, 1, 0])
                query.raycast_all(carb.Float3(*map(float, origin)), carb.Float3(*map(float, direction)), 0.16, hit, True)
                rays.append(dict(local_xz_m=[x, z], native_bearing_hits=hits))
            assert not rays[0]['native_bearing_hits']
            assert all((r['native_bearing_hits'] for r in rays[1:]))
            self.native_collision_queries = dict(rays=rays, bore_clear_and_four_outer_race_hits=True, tensor_view_status='CPU SdfShapeView not implemented by installed PhysX110.1.13; no native SDF tensor-count claim', scope='Actual native physics rays test open bore and original outer-race geometry; authored SDF and successful rolling/contact tests are reported separately')
        self.gate.get_articulation_controller().set_gains(kps=np.array([8000.0]), kds=np.array([400.0]))
        for m in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            m.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=46)
        self.tray_floor = 0.9075 + -0.0057221898 / 0.028 * 0.04
        self.asset_manifest = dict(task_id=46, environment=self.base.metadata, source_bearing=str(file), source_bearing_sha256=hashlib.sha256(file.read_bytes()).hexdigest(), physics_hz=resolved['physics_hz'], source_manifest=portable_json_loads((ROOT / 'prepared/manifest.json').read_text()), zone_x_m=resolved['zone_x_m'], release_position_m=self.initial.tolist(), upstream_gate_position_m=self.gate_position.tolist(), machine_outfeed=self.outfeed.metadata if self.outfeed else None, gate_housing='Ordinary rear-supported cantilever pneumatic stopper, collision enabled; front manipulation corridor clear. Native gate/product release mechanics unchanged.', tray_floor_z_m=self.tray_floor, recovery_tray_geometry=dict(source=str(TRAY), source_sha256=hashlib.sha256(TRAY.read_bytes()).hexdigest(), size_m=resolved['recovery_tray_size_m'], center_m=resolved['recovery_tray_center_m'], yaw_deg=resolved['recovery_tray_yaw_deg'], original_inner_half_m=[0.17, 0.11]), recovery_layout_adaptation='Original reusable tray referenced at230x290mm between unchanged full bolted feet/plates,10mm side clearance; its usableX narrows to195.5mm. Existing bench shifted150mm inY to support complete tray. Avoid loaded wrist flip; scored bearing, grippers, native release/whole-body support/retention and3s clock unchanged.', static_colliders=self.static_colliders, velocity_overrides=self.velocity_overrides, initial_joint_positions=self.initial_joint_positions, mounts=MOUNTS, assumptions=resolved['assumptions'], bearing_collision=dict(method=resolved.get('bearing_collision_method', 'convexDecomposition'), sdf_resolution=resolved.get('bearing_sdf_resolution'), source_geometry_unchanged=True, mesh_inspection='Original source mesh has4 watertight, consistently wound connected components; no geometry remeshing or hole filling authored.', scope='Actual cooking, rolling, grasp, cap and penetration validation remain separate from authored collision metadata.'), native_collision_queries=self.native_collision_queries, contact_materials='Candidate steel lane/bearing static0.35/dynamic0.25, gripper pads0.95/0.85; default average combination, nominal dynamic hand-bearing0.55. Not calibrated.', contact_offsets_m=dict(bearing=0.0003, fingers=0.0003, rest=0), scope='New Dual ARX native development; earlier robots/static approval are not a physics pass.')

    def process_step(self, o):
        """Machine outfeed (conveyor, QC lamp, review camera) and the upstream gate.

        Recorded inside the controller's step; they are the machine's own schedule,
        so the scene runs them on each control tick for every policy.
        """
        from isaacsim.core.utils.types import ArticulationAction

        t = o['time_s']
        if self.outfeed is not None:
            self.outfeed.step(o)
        release = t >= self.resolved['gate_release_s'] and (self.outfeed is None or self.outfeed.ready)
        self.gate.apply_action(ArticulationAction(joint_positions=np.array([0.15 if release else 0.0])))
        if release:
            self.events.setdefault('gate_open_command_s', t)
        if release and o['gate_lift_m'] > 0.10:
            self.events.setdefault('gate_clear_s', t)

    def surface_z(self, x):
        return 0.985 - (x - 0.7) * np.tan(np.deg2rad(self.resolved['slope_deg']))

    @property
    def detail(self):
        response = self.events.get('response_s')
        return 'Arrest deadline3s | Recovery to rework tray' if response is None else f'Arrest response {response:.3f}s /3s | Native contact recovery'

    def observe(self):
        self.readback.check(float(self.world.current_time))
        t = float(self.world.current_time)
        if self._last_tool_readback_s is None or t - self._last_tool_readback_s >= 1 / CONTROL_HZ - 1e-06:
            self._tool_cache = {}
            for side in ['left', 'right']:
                tp, tr = self.dual.tool_world(side)
                self._tool_cache[side] = dict(tool_position=tp.tolist(), tool_rotation=tr.tolist())
            self._last_tool_readback_s = t
        p, q = self.bearing.get_world_pose()
        R = Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix()
        points = self.envelope @ R.T + p
        forces = {side: [self.contacts.force(BEARING, '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{i}']) for i in [7, 8]] for side in ['left', 'right']}
        if max((v for pair in forces.values() for v in pair)) > 0.05 and self.first_hand_at is None:
            self.first_hand_at = t
            self.events['first_hand_contact_s'] = t
        if self.entry_at is None and points[:, 0].max() >= self.resolved['zone_x_m'][0]:
            self.entry_at = t
            self.events['zone_entry_s'] = t
            self.cue_state = 'ROLLING / ARREST <=3 s'
            self.display.set('rolling')
        return dict(time_s=t, tool_position=self._tool_cache[self.active_side]['tool_position'], active_side=self.active_side, tools_world=self._tool_cache, tool_readback_time_s=self._last_tool_readback_s, bearing_position=p.tolist(), bearing_rotation=R.tolist(), bounds_min_m=points.min(0).tolist(), bounds_max_m=points.max(0).tolist(), linear_velocity=self.bearing.get_linear_velocity().tolist(), angular_velocity=self.bearing.get_angular_velocity().tolist(), finger_forces_N=forces, native_finger_contact_samples={key: value for key, value in self.contacts.contact_samples.items() if self.contacts.interval.get(key, 0) > 0}, tray_support_N=self.contacts.force(BEARING, '/World/Recovery/ReworkTray'), tray_touching=any((v['active'] and BEARING in v['actors'] and any((a.startswith('/World/Recovery/ReworkTray') for a in v['actors'])) for v in self.contacts.active.values())), lane_support_N=self.contacts.force(BEARING, '/World/Return/GravityLane'), lane_touching=any((v['active'] and BEARING in v['actors'] and any((a.startswith('/World/Return/GravityLane') for a in v['actors'])) for v in self.contacts.active.values())), gate_support_N=self.contacts.force(BEARING, '/World/HoldingGate/plate'), gate_lift_m=float(self.gate.get_joint_positions()[0]), zone_entry_s=self.entry_at, first_hand_contact_s=self.first_hand_at, cue_state=self.cue_state, indicator_state=self.outfeed.state if self.outfeed else None, machine_outfeed_ready=self.outfeed.ready if self.outfeed else None, review_camera_eye=self.outfeed.camera_eye.tolist() if self.outfeed and self.outfeed.camera_eye is not None else None, review_camera_target=self.outfeed.camera_target.tolist() if self.outfeed and self.outfeed.camera_target is not None else None, unexpected_contacts=self.contacts.unexpected.copy())

    def tray_valid(self, o):
        return bool(np.all(np.array(o['bounds_min_m'][:2]) >= self.resolved['recovery_tray_inner_bounds_xy_m'][0]) and np.all(np.array(o['bounds_max_m'][:2]) <= self.resolved['recovery_tray_inner_bounds_xy_m'][1]) and (self.tray_floor - 0.001 <= o['bounds_min_m'][2] <= self.tray_floor + 0.002) and (o['tray_support_N'] > 0.2 or o['tray_touching']) and (max((v for pair in o['finger_forces_N'].values() for v in pair)) < 0.01))

    def final_valid(self, o):
        return bool(self.physical_completed and self.events.get('response_s', 999) <= self.resolved['response_deadline_s'] and self.tray_valid(o) and self.tray_retention_valid)


from .bearing_handoff import BearingARXController  # noqa: E402,F401  (re-export kept at its recorded position)
