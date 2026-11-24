"""Native three-side inspection/sorting with the approved original D455.

Two opposite side bands and the underside are measured using actual poses, optical
frustum and native occlusion rays. PASS/REJECT is a symbolic process recipe,
not a trained image defect detector or calibrated commercial sensor result.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig, weigh_scale
from .arx_cell_task import (
    ArmMotion,
    ContactBook,
    StatusDisplay,
    add_mounts,
    collide,
    remove_preview,
)
from .layout_review_revisions import RECEIVER_SIZE_49, RECEIVERS_49
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2'))
# Recorded mounts already form the common rig (0.60 m apart, same height and depth).
# The recorded shared PASS (y=0.41) / REJECT (y=0.12) receivers sat at or behind the head
# camera (y=0.36). Every receiver position the head camera sees (y >= ~0.52 at tray
# height) and that clears the input tray lies beside one arm, so each arm now has its
# own PASS and REJECT receiver on its side (layout_review_revisions.RECEIVERS_49) and a
# part counts in either side's tray of its result.
MOUNTS = rig.check(dict(left=[-0.3, 0.32, 1.2], right=[0.3, 0.32, 1.2]))

def resolve_inspection(seed=0, *, layout=False, timing=False, physical=False):
    # Child 4 (weighing) was added on 2026-11-19; children 0-3 are unchanged by spawn(5).
    rs = [np.random.default_rng(x) for x in np.random.SeedSequence(seed).spawn(5)]
    d = dict(task_id=49, embodiment='dual_arx_x5', seed=seed, part_xy_jitter_m=rs[0].uniform(-0.001, 0.001, (3, 2)).tolist() if layout else np.zeros((3, 2)).tolist(), start_s=float(rs[1].uniform(1, 2)) if timing else 1.0, mass_scale=float(rs[2].uniform(0.9, 1.1)) if physical else 1.0, finger_torsional_patch_radius_m=0.0, finger_min_torsional_patch_radius_m=0.0, finger_contact_assumption='Original zero torsional-radius contact model. The finite-tip candidate in nominal-v7 did not prevent tipping and is preserved as a failed probe. Original finger geometry and actuator caps unchanged.', budget_s=150.0, arm_command_speed_rad_s=0.7, runtime_joint_velocity_limit_rad_s=0.7, runtime_finger_velocity_limit_m_s=0.02, required_parts=3, inspection_dwell_s=0.3, inspection_stability=dict(roi_span_px=0.25, position_span_m=0.0002, orientation_from_first_deg=0.2, basis='Measured consecutive native poses/source-ROI projections over the complete acquisition dwell. Raw constraint-step velocities remain logged separately.'), process_results=rs[3].choice(['PASS', 'REJECT'], 3).tolist(), qc_sampling='Independent SeedSequence child3, per-part symbolic QC. Available after three actual camera acquisitions; no relation to arm identity.', source_positions_xy_m=[[-0.28, 0.73], [0, 0.73], [0.12, 0.73]], source_yaw_deg=0, receivers_by_arm_xy_m=RECEIVERS_49, receiver_size_m=RECEIVER_SIZE_49, receiver_inner_half_xy_m=[0.075, 0.09], receiver_slot_offsets_y_m=[-0.04, 0.04], deposit_tool_yaw_deg=dict(left=90, right=90), bounds=dict(part_xy_jitter_m=[-0.001, 0.001], start_s=[1, 2], mass_scale=[0.9, 1.1]), randomization=dict(layout=layout, timing=timing, physical=physical), inspection_faces=['upper_side', 'underside', 'upper_reverse'], inspection_views=[dict(position_m=[0, 0.62, 1.3], rpy_deg=[25, 0, 0]), dict(position_m=[0, 0.7, 1.31], approach_position_m=[0, 0.7, 1.29], rpy_deg=[75, 0, 0]), dict(position_m=[0.37, 0.48, 1.3], rpy_deg=[15, 0, -117])], reverse_view_by_arm=dict(left=dict(position_m=[-0.37, 0.48, 1.3], rpy_deg=[15, 0, 117]), right=dict(position_m=[0.37, 0.48, 1.3], rpy_deg=[15, 0, -117])), arm_assignments=dict(left=[0], right=[2, 1]), joint_transit=dict(height_m=1.06, loaded_joint_speed_rad_s=0.7, loaded_linear_speed_m_s=0.18, loaded_angular_speed_rad_s=0.8, empty_joint_speed_rad_s=0.7, empty_linear_speed_m_s=0.3, empty_angular_speed_rad_s=1.5, sampled_original_part_floor_m=1.0, scope='Actual bounded joint drives through original URDF configurations; sampled source-part swept height and original limits checked before motion. Native collisions and retained-grasp checks remain enabled. No product state writes.'), final_clearance_m=0.12, completion_event='Third unique fully released, floor-supported, stable correctly sorted part, after all9 actual camera acquisitions. End-effector clearing precedes credit; return to home is not an extra reviewer deadline.', reverse_path=dict(outward_arc_m=0.1, waypoint_fraction=0.025, scope='Coordinated measured-body translation and shortest orientation interpolation.100mm outward sine arc avoids the original-arm unreachable underside/reverse transition. All native collisions remain enabled.'), third_pick_interlock='While the left arm owns the camera, wait for measured part0 X<-0.20m in its reverse-view corridor before right arm enters central source1. Original inter-arm collisions remain enabled.', interlocks='Separate exclusive source and camera owners. Each arm sorts into the PASS/REJECT receivers on its own side; either side\'s receiver of the right result is correct. Original inter-arm collisions remain enabled.', controller='Privileged native-state bounded original joint drives', quality_scope='Two opposite actual lateral CAD bands and the actual underside; finite visual-triangle/native-ray geometric visibility proxy. Symbolic process labels, not learned defect classification.')
    d['source_layout_revision'] = 'User identifies visible22–24s crowding: left source moves outward toX=-0.28m. Same imported tray scaled to640mm width, still on original bench. Original arms/products/receivers/sensor/limits/quality/90s preserved; native validation pending.'
    d['visual_interarm_review_margin_m'] = 0.05
    d['finger_operating_revision'] = 'Lower native operating limit from30 to20mm/s after measured41.76mm/s opening transient in v1. Original40mm/s hardware acceptance limit and all quality/throughput criteria unchanged.'
    d['camera_approach_speed_m_s'] = 0.17
    # Weighing (2026-11-19): exactly one part per batch of 3 is a short shot that weighs 70 %
    # of nominal; it looks the same. A part is PASS only with a green QC lamp and at least
    # MIN_G on the bench scale, else REJECT.
    light = int(rs[4].integers(3))
    d['underweight'] = [i == light for i in range(3)]
    d['part_mass_kg'] = [(weigh_scale.UNDERWEIGHT_KG if u else weigh_scale.NOMINAL_KG) * d['mass_scale'] for u in d['underweight']]
    d['sort_results'] = ['REJECT' if u or qc == 'REJECT' else 'PASS' for u, qc in zip(d['underweight'], d['process_results'])]
    d['weighing'] = dict(pan_xy_m=list(weigh_scale.PAN_XY), pan_top_z_m=weigh_scale.PAN_TOP_Z, min_g=weigh_scale.MIN_G, nominal_kg=weigh_scale.NOMINAL_KG, underweight_kg=weigh_scale.UNDERWEIGHT_KG, dwell_s=weigh_scale.WEIGH_DWELL_S, display_update_s=weigh_scale.UPDATE_S, sampling='SeedSequence child4: exactly one underweight part per batch.', scope='Reading = measured native pan contact normal force / g, mean over each display window. Weighed once resting alone, released and at rest on the pan for dwell_s. Sorting needs all three faces and the weighing; REJECT if the lamp is red or the part is under min_g.')
    d['input_tray_dimensions_m'] = [0.64,0.24,0.035]
    d['camera_wait_clearance'] = dict(left_tool_offset_m=[-0.04,-0.02,0], right_reverse_clear_x_m=0.26, loaded_wait_speed_m_s=0.12, scope='Non-scanning left arm waits outward until measured right product is in its reverse corridor; all sensor acquisition/resource/quality conditions unchanged.')
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d

# When the inspection camera sees a face: its normal within normal_error_deg of the line to the
# lens, distance_m from it, image_margin_px inside the image. The instruction states the first
# two (tasks.TASK_RULES): a policy cannot read them off the scene.
VIEW_QUALITY = dict(image_margin_px=8, normal_error_deg=35, distance_m=[0.15, 0.65])


class InspectionARXScene:
    title = 'THREE-SIDE INSPECTION / SORTING'

    def __init__(self, manifest, output, resolved, mode):
        import omni.physx
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from isaacsim.sensors.camera import Camera
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .layout_review_revisions import build_layout_review
        from .remaining_environments import TRAY
        from .v2.geometry import body_envelope_points
        self.base = build_layout_review(49, output, (1280, 800))
        c = self.base.cell
        self.world, self.stage = (c.world, c.stage)
        self.stage.RemovePrim('/World/PlannedMount')
        for side in ['left', 'right']:
            self.stage.RemovePrim('/World/Signs/Mount_' + side)
        self.resolved, self.events = (resolved, {})
        self.source_owner = self.camera_owner = self.deposit_owner = None
        self.weigh = weigh_scale.WeighTracker()
        self.scale_owner = None
        self.arm_phase = {}
        self.inspected, self.sorted = ({}, set())
        self.physical_completed = self.success = False
        self.cue_state = 'INPUT'
        self.parts, self.envelopes = ({}, {})
        self.asset_manifest = portable_json_loads((ROOT / 'task49/prepared-v3/asset-manifest.json').read_text())
        raw = np.load(ROOT / 'task49/prepared-v3/visual_triangles.npz')
        self.triangles = raw['vertices'][raw['faces']]
        self.regions = {}
        for face, normal, axis in [('upper_front', [1, 0, 0], 0), ('upper_side', [0, 1, 0], 1), ('upper_back', [-1, 0, 0], 0)]:
            sign = -1 if face == 'upper_back' else 1
            fixed = sign * 0.01875
            pts, boundary = ([], [])
            other = 1 - axis
            for lateral, z in [(-0.008, 0.007), (-0.004, 0.013), (0, 0.01), (0.004, 0.013), (0.008, 0.007)]:
                point = np.zeros(3)
                point[axis] = fixed
                point[other] = lateral
                point[2] = z
                pts.append(point.tolist())
            for lateral in [-0.0105, 0.0105]:
                for z in [0.004, 0.021]:
                    point = np.zeros(3)
                    point[axis] = fixed
                    point[other] = lateral
                    point[2] = z
                    boundary.append(point.tolist())
            self.regions[face] = dict(normal_local=normal, sample_points_local_m=pts, boundary_points_local_m=boundary)
        side = self.regions['upper_side']
        reverse = portable_json_loads(json.dumps(side))
        reverse['normal_local'] = [0, -1, 0]
        for key in ['sample_points_local_m', 'boundary_points_local_m']:
            for point in reverse[key]:
                point[1] *= -1
        bottom = self.triangles[np.max(abs(self.triangles[:, :, 2] + 0.021), axis=1) < 1e-06]
        areas = np.linalg.norm(np.cross(bottom[:, 1] - bottom[:, 0], bottom[:, 2] - bottom[:, 0]), axis=1) / 2
        underside = dict(normal_local=[0, 0, -1], sample_points_local_m=bottom[np.argsort(areas)[-7:]].mean(1).tolist(), boundary_points_local_m=np.unique(bottom.reshape(-1, 3), axis=0).tolist())
        self.regions = dict(upper_side=side, underside=underside, upper_reverse=reverse)
        material = PhysicsMaterial('/World/InspectionContact', static_friction=0.95, dynamic_friction=0.85, restitution=0)
        remove_preview(c, '/World/Input/Tray')
        c.asset('Input/Tray', TRAY, [0, 0.73, 0.905], resolved.get('input_tray_dimensions_m', [0.64, 0.24, 0.035]), material='steel')
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
            p = np.r_[resolved['source_positions_xy_m'][i], 0.94885]
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
        self.display = StatusDisplay(c, 'Recipe', dict(INPUT='LOAD / THREE-SIDE CAMERA INSPECTION', FACE1='FACE1 / SIDE A', FACE2='FACE2 / UNDERSIDE', FACE3='FACE3 / OPPOSITE SIDE B', PASS='THREE FACES VERIFIED / PASS', REJECT='THREE FACES VERIFIED / REJECT', COMPLETE='THREE PARTS INSPECTED / SORTED'))
        weigh_scale.build(c)
        self.static_colliders = collide(self.stage, ['/World/InspectionBench', '/World/Input/Tray', '/World/Output/', '/World/Camera/', '/World/FeedNests/', '/World/Scale/Base', '/World/Scale/Pan', '/World/Scale/Housing'], material)
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
                self.velocity_limit_overrides.append(dict(joint=str(joint.GetPath()), imported_deg_s=original, configured_deg_s=float(limit), scope='Conservative operating limiter; source asset unmodified and0.75rad/s benchmark cap unchanged'))
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
        cameras = [p for p in Usd.PrimRange(self.stage.GetPrimAtPath('/World/Camera/RealSenseD455')) if p.IsA(UsdGeom.Camera)]
        rgb = next((p for p in cameras if 'color' in str(p.GetPath()).lower()), None)
        if rgb is None:
            raise RuntimeError('Original D455 RGB optical camera missing: ' + str([str(p.GetPath()) for p in cameras]))
        self.inspection_camera = self.world.scene.add(Camera(str(rgb.GetPath()), name='native_d455_rgb', resolution=(640, 480)))
        self.scale = weigh_scale.WeighScale(self, output)
        self.world.reset()
        self.dual.initialize()
        self.scale.reset_display()
        for b in self.parts.values():
            b.initialize()
        for data in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            data.clear()
        if not rig.configure()['no_render']:
            # The D455 renders only for evidence images; visibility uses its geometry and intrinsics.
            self.inspection_camera.initialize()
        frame = UsdGeom.XformCache().GetLocalToWorldTransform(rgb)
        matrix = np.array(frame)
        self.camera_calibration = dict(eye_m=matrix[3, :3].tolist(), rotation_world=matrix[:3, :3].T.tolist(), intrinsics=self.inspection_camera.get_intrinsics_matrix().tolist(), resolution=[640, 480], source='Original D455 RGB optical prim; ideal simulated geometry, not calibrated physical device')
        self.query = omni.physx.get_physx_scene_query_interface()
        self.quality = dict(VIEW_QUALITY)
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=49)
        self.asset_manifest.update(native_adaptations='Original product visuals; source on30mm flange-support rails. Opposed X-face grasp, inspect Y/-Y/base. Approved centered input; a labeled PASS and REJECT receiver beside each arm, inside the rig head view (common rig). Source products keep their original orientation and verified local-X grip for inspection. Each arm deposits into its own side with a 90 deg downward tool yaw. Reverse camera view corridor moves toX+/-0.37/Y0.48 within the original optical frustum. Independent per-part QC sampling; left handles part0, right2/1 regardless of result. Exclusive source/camera owners; per-arm receivers. No classification or full randomization-matrix claim.', native_collision_derivation=portable_json_loads((ROOT / 'task49/contact-v1/contact.json').read_text()), camera_calibration=self.camera_calibration, approved_layout=self.base.metadata, static_colliders=self.static_colliders, qc_sampling=resolved['qc_sampling'], velocity_limit_overrides=self.velocity_limit_overrides, inspected_regions_arx=self.regions)

    @property
    def detail(self):
        return f"Inspected and sorted: {len(self.sorted)}/3 | Three measured face views | Budget{self.resolved['budget_s']:g}s"

    @staticmethod
    def pose(b):
        p, q = b.get_world_pose()
        return (np.asarray(p), Rotation.from_quat(np.r_[q[1:], q[0]]).as_matrix())

    def visible(self, index, face):
        import carb

        from .v2.inspection_visibility import visible_region
        b = self.parts[index]
        p, R = self.pose(b)

        def blocked(origin, direction, length):
            hits = []

            def hit(value):
                if str(value.rigid_body) != b.prim_path and (not str(value.rigid_body).startswith('/World/Camera/RealSenseD455/')):
                    hits.append(str(value.rigid_body))
                return True
            self.query.raycast_all(carb.Float3(*map(float, origin)), carb.Float3(*map(float, direction)), float(max(0, length - 0.001)), hit, True)
            return bool(hits)
        return visible_region(p, R, self.regions[face], self.triangles, self.camera_calibration, self.quality, blocked)

    def contained(self, index):
        p, R = self.pose(self.parts[index])
        pts = self.envelopes[index] @ R.T + p
        half = self.resolved['receiver_inner_half_xy_m']
        result = self.resolved['sort_results'][index]
        return any(bool(np.all(abs(pts[:, :2] - trays[result]) <= half) and pts[:, 2].min() >= 0.89765 and (pts[:, 2].max() < 0.98)) for trays in self.receivers.values())

    def receiver_meshes(self, index):
        result = self.resolved['sort_results'][index]
        return [f'/World/Output/{result}_{side}/Offset/Asset/Mesh_0' for side in self.receivers]

    def receiver_touching(self, index):
        return any(self.contacts.touching(self.parts[index].prim_path, mesh) for mesh in self.receiver_meshes(index))

    def observe(self):
        parts = {}
        for i, b in self.parts.items():
            p, R = self.pose(b)
            parts[str(i)] = dict(position=p.tolist(), rotation=R.tolist(), linear_velocity=b.get_linear_velocity().tolist(), angular_velocity=b.get_angular_velocity().tolist(), contained=self.contained(i), support_N=self.contacts.force(b.prim_path, '/World/Output/'), scale_N=self.contacts.force(b.prim_path, '/World/Scale/Pan'), support_touching=self.receiver_touching(i), finger_forces_N=[self.contacts.force(b.prim_path, '/World/ARX/left/' + self.dual.manifest['relative_body_paths'][f'link{j}']) for j in [7, 8]], finger_forces_by_arm={side: [self.contacts.force(b.prim_path, '/World/ARX/' + side + '/' + self.dual.manifest['relative_body_paths'][f'link{j}']) for j in [7, 8]] for side in ['left', 'right']})
        return dict(time_s=float(self.world.current_time), tool_position=self.dual.tool_world('left')[0].tolist(), parts=parts, inspected={k: list(v) for k, v in self.inspected.items()}, sorted=sorted(self.sorted), scale_reading_g=self.scale.grams, weighed={str(k): v for k, v in self.weigh.weighed.items()}, cue_state=self.cue_state, resource_owners=dict(source=self.source_owner, camera=self.camera_owner, deposit=self.deposit_owner), tools_world={side: dict(position=self.dual.tool_world(side)[0].tolist(), rotation=self.dual.tool_world(side)[1].tolist()) for side in ['left', 'right']}, unexpected_contacts=self.contacts.unexpected.copy())

    def process_step(self, o):
        """The scale: one load sample per control tick, display refresh, weighed credit."""
        load = sum(part['scale_N'] for part in o['parts'].values())
        self.scale.update(o['time_s'], load)
        o['scale_reading_g'] = self.scale.grams
        for index, grams in self.weigh.update(o):
            self.events[f'{index}_weighed'] = dict(time_s=o['time_s'], reading_g=grams, mass_g=1000 * self.resolved['part_mass_kg'][index], underweight=grams is not None and grams < weigh_scale.MIN_G)

    def final_valid(self, o):
        return len(self.sorted) == 3 and all((self.contained(i) and i in self.weigh.weighed and set(self.inspected.get(i, [])) == set(self.resolved['inspection_faces']) and (o['parts'][str(i)]['support_N'] > 0.01 or self.receiver_touching(i)) and (np.linalg.norm(self.parts[i].get_linear_velocity()) < 0.012) and (np.linalg.norm(self.parts[i].get_angular_velocity()) < 0.1) and (max((v for forces in o['parts'][str(i)]['finger_forces_by_arm'].values() for v in forces)) < 0.01) for i in range(3))) and (self.events.get('completed_s', 999) <= self.resolved['budget_s'])

class InspectionArmOperation:

    def __init__(self, scene, mode, side):
        self.s, self.mode, self.side = (scene, mode, side)
        self.m = ArmMotion(scene.dual, side)
        self.m.position_gain = 1.5
        self.m.command_speed_rad_s = scene.resolved['arm_command_speed_rad_s']
        self.m.orientation_feedback = True
        # Downward deposit tool; 90 deg yaw keeps joint6 well inside +/-2 rad at the side receivers.
        self.deposit_R = Rotation.from_euler('z', scene.resolved['deposit_tool_yaw_deg'][side], degrees=True).as_matrix() @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.pick_R = scene.dual.rotation @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        self.m.R = scene.dual.tool_world(side)[1].copy()
        self.indices = list(scene.resolved['arm_assignments'][side])
        if mode == 'grasp' and side == 'right':
            self.indices = []
        self.ordinal = 0
        self.hold = np.array([-0.34 if side == 'left' else 0.34, 0.44, 1.22])
        self.phase, self.failure, self.complete = ('park', None, False)
        self.entered, self.index, self.face_index = (0.0, self.indices[0] if self.indices else -1, 0)
        self.travel, self.since = (0.024, None)
        self.view_window = []
        self.camera_joint_hold = False
        self.view_stability = None
        self.reverse_path_start = None
        self.joint_path_kind = None
        self.park_joints = None
        self.transport_lost_since = None
        self.last_empty_start = None
        self.regrip = False

    # Phases in which an arm works in the centre (camera views, scale, re-grasp on the scale).
    CENTRAL = ('view_move', 'inspect0', 'inspect1', 'camera_setup', 'scale_', 'wait_camera')

    def other_clear(self):
        """The other arm is out of the central corridor, so this arm may use the scale."""
        other = 'right' if self.side == 'left' else 'left'
        x = self.s.dual.tool_world(other)[0][0]
        return abs(x) > 0.2 and not self.s.arm_phase.get(other, 'park').startswith(self.CENTRAL)

    def transition(self, phase):
        self.phase = phase
        self.s.arm_phase[self.side] = phase
        self.entered = float(self.s.world.current_time)
        self.s.events[f'{self.index}_{phase}_s'] = self.entered
        if phase == 'view_move2':
            self.reverse_path_start = None
        if phase.startswith('inspect'):
            self.view_window = []
            self.camera_joint_hold = False
        print('PHASE', self.index, phase, self.entered, flush=True)

    def rotate_command(self, target):
        delta = Rotation.from_matrix(target @ self.view_R.T).as_rotvec()
        angle = np.linalg.norm(delta)
        self.view_R = Rotation.from_rotvec(delta * min(1, 0.65 / CONTROL_HZ / max(angle, 1e-12))).as_matrix() @ self.view_R
        return angle < 0.01

    def begin_joint_path(self, kind, t, *, goal=None, rotation=None, joint_goal=None, loaded=False):
        s, d, m = (self.s, self.s.dual, self.m)
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
        points, rotations, minimum = ([], [], float('inf'))
        for u in np.linspace(0, 1, 81):
            q = self.joint_from * (1 - u) + self.joint_goal * u
            T = m.kin.forward(dict(zip(m.kin.arm_names, q)))
            pos, Rtool = (bp + bR @ T[:3, 3], bR @ T[:3, :3])
            points.append(pos + Rtool @ self.offset if loaded else pos)
            rotations.append(Rtool)
            if loaded:
                v = s.envelopes[self.index] @ (Rtool @ self.grasp_rotation).T + points[-1]
                minimum = min(minimum, float(v[:, 2].min()))
        settings = s.resolved['joint_transit']
        if loaded and minimum < settings['sampled_original_part_floor_m']:
            raise RuntimeError('Joint transit sampled original part too near fixtures')
        v = float(np.linalg.norm(np.diff(points, axis=0), axis=1).max() * 80)
        W = np.asarray(rotations)
        w = float(Rotation.from_matrix(W[1:] @ np.swapaxes(W[:-1], 1, 2)).magnitude().max() * 80)
        prefix = 'loaded' if loaded else 'empty'
        self.joint_duration = max(float(abs(self.joint_goal - self.joint_from).max()) / settings[prefix + '_joint_speed_rad_s'], v / settings[prefix + '_linear_speed_m_s'], w / settings[prefix + '_angular_speed_rad_s'], 0.1)
        self.joint_progress = 0.0
        self.joint_terminal_feedback = False
        Tend = m.kin.forward(dict(zip(m.kin.arm_names, self.joint_goal)))
        self.joint_tool_goal = bp + bR @ Tend[:3, 3]
        self.joint_tool_rotation = bR @ Tend[:3, :3]
        s.events[f'{kind}_joint_path'] = dict(time_s=t, source='Original bounded URDF; actual native drives', q_start=self.joint_from.tolist(), q_goal=self.joint_goal.tolist(), planned_duration_s=self.joint_duration, minimum_sampled_original_part_z_m=minimum if loaded else None)

    def advance_joint_path(self):
        d = self.s.dual
        if self.joint_progress >= 1:
            if self.joint_path_kind.endswith(('_empty_fold', '_empty_swing')):
                actual = np.asarray(d.robots[self.side].get_joint_positions())[self.m.ids]
                return bool(float(abs(actual - self.joint_goal).max()) < 0.02)
            if not self.joint_terminal_feedback:
                self.m.command = d.tool_world(self.side)[0].copy()
                self.m.correction[:] = 0
                self.m.orientation_correction[:] = 0
                self.joint_terminal_feedback = True
            return self.m.move(self.joint_tool_goal, 0.04, self.joint_tool_rotation)
        actual = np.asarray(d.robots[self.side].get_joint_positions())[self.m.ids]
        previous = d.commands[self.side][self.m.ids]
        if float(abs(actual - previous).max()) < 0.05:
            self.joint_progress = min(1.0, self.joint_progress + 1 / (CONTROL_HZ * self.joint_duration))
        command = self.joint_from * (1 - self.joint_progress) + self.joint_goal * self.joint_progress
        d.commands[self.side][self.m.ids] = command
        p, R = d.tool_world(self.side)
        arrived = self.joint_progress >= 1 and float(abs(actual - self.joint_goal).max()) < 0.03 and (np.linalg.norm(p - self.joint_tool_goal) < 0.006) and (Rotation.from_matrix(self.joint_tool_rotation @ R.T).magnitude() < 0.04)
        if arrived:
            self.m.command = p.copy()
            self.m.correction[:] = 0
            self.m.orientation_correction[:] = 0
        return bool(arrived)

    def stable_camera_view(self, t, p, R, visibility, arrived, forces):
        minimum_force = 0.15 * self.s.resolved['mass_scale'] * 9.81 / (2 * 0.85) * 1.1
        if not (arrived and visibility['valid'] and (min(forces) > minimum_force)):
            self.view_window = []
            self.view_stability = dict(valid=False, reason='View, tracking or retained grasp not ready')
            return False
        self.view_window.append((t, p.copy(), R.copy(), np.array(visibility['projected_points_px'])))
        dwell = self.s.resolved['inspection_dwell_s']
        while len(self.view_window) > 1 and self.view_window[1][0] <= t - dwell:
            self.view_window.pop(0)
        points = np.array([x[1] for x in self.view_window])
        rotations = np.array([x[2] for x in self.view_window])
        pixels = np.array([x[3] for x in self.view_window])
        linear_span = float(np.linalg.norm(np.ptp(points, axis=0)))
        angular_span = float(np.max(Rotation.from_matrix(rotations @ rotations[0].T).magnitude()))
        pixel_span = float(np.max(np.ptp(pixels, axis=0)))
        elapsed = t - self.view_window[0][0]
        limits = self.s.resolved['inspection_stability']
        valid = elapsed >= dwell - 1e-06 and linear_span <= limits['position_span_m'] and (angular_span <= np.deg2rad(limits['orientation_from_first_deg'])) and (pixel_span <= limits['roi_span_px'])
        self.view_stability = dict(valid=bool(valid), elapsed_s=elapsed, samples=len(self.view_window), native_pose_span_m=linear_span, orientation_from_first_rad=angular_span, roi_span_px=pixel_span, minimum_grasp_force_N=minimum_force, raw_velocity_gate=False, scope='Continuous measured native-pose/image acquisition stability, not an instantaneous zero-velocity claim')
        return valid

    def step(self, o):
        if self.complete or self.failure or self.mode in ['absent', 'passive']:
            return
        s, t, m = (self.s, o['time_s'], self.m)
        if t < s.resolved['start_s']:
            return
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if self.phase == 'park':
                delta = Rotation.from_matrix(self.pick_R @ m.R.T).as_rotvec()
                actual_R = s.dual.tool_world(self.side)[1]
                if Rotation.from_matrix(m.R @ actual_R.T).magnitude() < 0.05:
                    m.R = Rotation.from_rotvec(delta * min(1, 0.3 / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12))).as_matrix() @ m.R
                if m.move(self.hold, 0.12) and np.linalg.norm(delta) < 0.005:
                    self.park_joints = s.dual.commands[self.side][m.ids].copy()
                    if self.indices:
                        self.transition('approach')
                    else:
                        self.complete = True
                s.dual.jaw_travel(self.side, 0.024)
                return
            b = s.parts[self.index]
            p, R = s.pose(b)
            grip = p + R @ np.array([0, 0, -0.002])
            if self.phase == 'approach':
                if self.index == 1 and ((s.camera_owner == 'left' and o['parts']['0']['position'][0] >= -0.2) or s.scale_owner == 'left'):
                    m.move(self.hold, 0.04)
                    self.entered = t
                    return
                if s.source_owner not in [None, self.side]:
                    self.entered = t
                    return
                if s.source_owner is None:
                    s.source_owner = self.side
                    s.events[f'{self.index}_source_claimed_s'] = t
                self.travel = 0.024
                if m.move(grip + [0, 0, 0.055], 0.18):
                    self.transition('lower')
            elif self.phase == 'lower':
                arrived = m.move(grip, 0.045)
                tool, tool_R = s.dual.tool_world(self.side)
                precise = arrived and np.linalg.norm(tool - grip) < 0.0005 and (Rotation.from_matrix(m.R @ tool_R.T).magnitude() < 0.005)
                self.since = (self.since if self.since is not None else t) if precise else None
                if self.since is not None and t - self.since > 0.15:
                    self.close_goal = grip.copy()
                    self.since = None
                    self.transition('close')
            elif self.phase == 'close':
                m.move(self.close_goal, 0.025)
                forces = o['parts'][str(self.index)]['finger_forces_by_arm'][self.side]
                good = min(forces) > 4.0
                if not good:
                    self.travel = max(0, self.travel - 0.004 / CONTROL_HZ)
                self.since = (self.since if self.since is not None else t) if good else None
                if self.since is not None and t - self.since > 0.12:
                    tool, toolR = s.dual.tool_world(self.side)
                    self.offset = toolR.T @ (p - tool)
                    self.lift = tool + [0, 0, 0.12]
                    self.transition('lift')
            elif self.phase == 'lift':
                if m.move(self.lift, 0.1):
                    if self.regrip:
                        self.regrip = False
                        self.transition('wait_camera')
                        s.dual.jaw_travel(self.side, self.travel)
                        return
                    if self.mode == 'grasp':
                        self.complete = True
                        s.source_owner = None
                        return
                    self.wait_goal = self.lift.copy()
                    s.source_owner = None
                    s.events[f'{self.index}_source_released_s'] = t
                    self.transition('wait_view')
            elif self.phase == 'wait_view':
                # Keep the non-scanning arm outside the central underside sweep.
                # Native measured clearance, not time or arm-bound QC, controls entry.
                other = 'right' if self.side == 'left' else 'left'
                clearance = s.resolved.get('camera_wait_clearance', {})
                staged = (self.side == 'left' and s.camera_owner == other
                          and o['parts']['2']['position'][0] < clearance.get('right_reverse_clear_x_m', 0.26))
                offset = clearance.get('left_tool_offset_m', [-0.12, -0.02, 0.0]) if staged else [0, 0, 0]
                self.wait_goal = self.lift + np.asarray(offset)
                m.move(self.wait_goal, clearance.get('loaded_wait_speed_m_s', 0.12))
                key = f'{self.index}_clear_wait_' + ('entered_s' if staged else 'cleared_s')
                if staged or f'{self.index}_clear_wait_entered_s' in s.events:
                    s.events.setdefault(key, t)
                self.entered = t
                if s.scale_owner is None and self.other_clear():
                    s.scale_owner = self.side
                    s.events[f'{self.index}_scale_claimed_s'] = t
                    tool = s.dual.tool_world(self.side)[0]
                    above = np.r_[weigh_scale.PAN_XY, weigh_scale.PAN_TOP_Z + 0.021 + 0.05]
                    self.scale_above = tool + (above - p)
                    self.transition('scale_move')
            elif self.phase == 'scale_move':
                if m.move(self.scale_above, 0.15):
                    self.scale_place = self.scale_above - [0, 0, 0.047]
                    self.grip_travel = self.travel
                    self.transition('scale_place')
            elif self.phase == 'scale_place':
                if m.move(self.scale_place, 0.05):
                    self.transition('scale_release')
            elif self.phase == 'scale_release':
                m.move(self.scale_place, 0.02)
                # Fully open: the gripper backs off sideways and re-grasps as from the nest.
                opened = 0.024
                self.travel = min(opened, self.travel + 0.02 / CONTROL_HZ)
                if self.travel >= opened:
                    m.command = s.dual.tool_world(self.side)[0].copy()
                    m.correction[:] = 0
                    m.orientation_correction[:] = 0
                    self.transition('scale_clear')
            elif self.phase == 'scale_clear':
                if m.move(self.scale_place + [0, 0, 0.06], 0.08):
                    self.since = None
                    self.transition('scale_read')
            elif self.phase == 'scale_read':
                # Back off up and outward so the head camera sees the readout, then return.
                sign = -1 if self.side == 'left' else 1
                back = self.scale_place + [sign * 0.13, -0.03, 0.11]
                arrived = m.move(back, 0.15)
                if arrived and f'{self.index}_weighed' in s.events:
                    self.since = self.since if self.since is not None else t
                    if t - self.since > 0.4:
                        self.since = None
                        self.transition('scale_return')
            elif self.phase == 'scale_return':
                if m.move(grip + [0, 0, 0.055], 0.15):
                    self.regrip = True
                    self.since = None
                    self.transition('lower')
            elif self.phase == 'wait_camera':
                # Hold the weighed part above the scale until the camera is free.
                m.move(self.lift, 0.04)
                self.entered = t
                if s.camera_owner is None:
                    s.camera_owner = self.side
                    s.events[f'{self.index}_camera_claimed_s'] = t
                    s.scale_owner = None
                    s.events[f'{self.index}_scale_released_s'] = t
                    self.transition('camera_setup')
            elif self.phase == 'camera_setup':
                self.face_index = 0
                self.since = None
                tool, toolR = s.dual.tool_world(self.side)
                self.offset = toolR.T @ (p - tool)
                self.grasp_rotation = toolR.T @ R
                self.view_R = toolR.copy()
                self.transition('view_move0')
            elif self.phase.startswith(('view_move', 'inspect')):
                face = s.resolved['inspection_faces'][self.face_index]
                view = s.resolved['reverse_view_by_arm'][self.side] if self.face_index == 2 else s.resolved['inspection_views'][self.face_index]
                body_target = np.asarray(view['position_m'])
                desired_R = Rotation.from_euler('xyz', view['rpy_deg'], degrees=True).as_matrix() @ self.grasp_rotation.T
                if self.phase == 'view_move2':
                    if self.reverse_path_start is None:
                        self.reverse_path_start = p.copy()
                        self.reverse_path_rotation = R.copy()
                        self.reverse_path_delta = Rotation.from_matrix(Rotation.from_euler('xyz', view['rpy_deg'], degrees=True).as_matrix() @ R.T).as_rotvec()
                        self.reverse_path_progress = 0.0
                    u = self.reverse_path_progress
                    sign = -1 if self.side == 'left' else 1
                    settings = s.resolved['reverse_path']
                    body_goal = (1 - u) * self.reverse_path_start + u * body_target + [sign * settings['outward_arc_m'] * np.sin(np.pi * u), 0, 0]
                    body_R = Rotation.from_rotvec(self.reverse_path_delta * u).as_matrix() @ self.reverse_path_rotation
                    self.view_R = body_R @ self.grasp_rotation.T
                    arrived = m.move(body_goal - self.view_R @ self.offset, 0.14, self.view_R)
                    if arrived:
                        if u >= 1:
                            self.transition('inspect2')
                        else:
                            self.reverse_path_progress = min(1.0, u + settings['waypoint_fraction'])
                    if t - self.entered > 18:
                        raise RuntimeError('Timeout coordinated reverse camera approach')
                    s.dual.jaw_travel(self.side, self.travel)
                    return
                if self.phase.startswith('view_move'):
                    if t - self.entered > 18:
                        raise RuntimeError('Timeout translating to inspection view')
                    approach = np.asarray(view.get('approach_position_m', body_target))
                    if m.move(approach - self.view_R @ self.offset, s.resolved.get('camera_approach_speed_m_s', 0.17), self.view_R):
                        self.transition('inspect' + str(self.face_index))
                    s.dual.jaw_travel(self.side, self.travel)
                    return
                rotation_ready = True if self.camera_joint_hold else self.rotate_command(desired_R)
                goal = body_target - self.view_R @ self.offset
                arrived = self.camera_joint_hold or (m.move(goal, 0.08, self.view_R) and rotation_ready)
                s.cue_state = f'FACE{self.face_index + 1}'
                s.display.set(s.cue_state)
                visibility = s.visible(self.index, face) if arrived else dict(valid=False)
                if arrived and visibility['valid']:
                    self.camera_joint_hold = True
                elif self.camera_joint_hold:
                    self.camera_joint_hold = False
                self.last_visibility = visibility
                self.last_inspection_face = face
                good = self.stable_camera_view(t, p, R, visibility, arrived, o['parts'][str(self.index)]['finger_forces_by_arm'][self.side])
                if good:
                    s.inspected.setdefault(self.index, []).append(face)
                    s.events[f'{self.index}_{face}_verified'] = dict(time_s=t, visibility=visibility, acquisition_stability=self.view_stability.copy())
                    self.face_index += 1
                    self.since = None
                    if self.face_index == 3:
                        self.last_view_goal = goal.copy()
                        s.events[f'{self.index}_qc_result_revealed'] = dict(time_s=t, result=s.resolved['process_results'][self.index], sort_result=s.resolved['sort_results'][self.index], arm=self.side, acquisitions=list(s.inspected[self.index]), scope='Independent seeded symbolic process result; not learned defect detection')
                        s.cue_state = s.resolved['process_results'][self.index]
                        s.display.set(s.cue_state)
                        self.transition('transfer')
                    else:
                        self.transition('view_move' + str(self.face_index))
            else:
                result = s.resolved['sort_results'][self.index]
                # Own-side receiver; slots along y (the 90 deg jaw opens along x).
                matching = [i for i in self.indices if s.resolved['sort_results'][i] == result]
                slot = matching.index(self.index)
                xy = s.receivers[self.side][result] + [0, s.resolved['receiver_slot_offsets_y_m'][slot]]
                target = np.r_[xy, 0.91885]
                placement_R = self.deposit_R
                goal = target - placement_R @ self.offset
                if self.phase == 'transfer':
                    # Each arm owns the receivers on its side, so no deposit wait.
                    if s.deposit_owner is None:
                        s.deposit_owner = self.side
                        s.events[f'{self.index}_deposit_claimed_s'] = t
                    if self.joint_path_kind != f'{self.index}_loaded':
                        high = np.r_[xy, s.resolved['joint_transit']['height_m']]
                        self.begin_joint_path(f'{self.index}_loaded', t, goal=high - placement_R @ self.offset, rotation=placement_R, loaded=True)
                    retained = min(o['parts'][str(self.index)]['finger_forces_by_arm'][self.side]) > 0.2 and np.linalg.norm(p - s.dual.tool_world(self.side)[0]) < 0.02
                    self.transport_lost_since = None if retained else t if self.transport_lost_since is None else self.transport_lost_since
                    if self.transport_lost_since is not None and t - self.transport_lost_since > 0.2:
                        raise RuntimeError('Native part grasp lost during shared-tray transit')
                    if self.advance_joint_path():
                        self.view_R = placement_R.copy()
                        if s.camera_owner == self.side:
                            s.camera_owner = None
                            s.events[f'{self.index}_camera_released_s'] = t
                        self.transition('place')
                elif self.phase == 'place':
                    if m.move(goal + [0, 0, 0.003], 0.065, placement_R):
                        self.release_goal = goal.copy()
                        self.transition('release')
                elif self.phase == 'release':
                    self.travel = min(0.024, self.travel + 0.02 / CONTROL_HZ)
                    if self.travel >= 0.024:
                        m.command = s.dual.tool_world(self.side)[0].copy()
                        m.correction[:] = 0
                        m.orientation_correction[:] = 0
                        self.transition('clear_release')
                elif self.phase == 'clear_release':
                    if m.move(self.release_goal + [0, 0, s.resolved['final_clearance_m']], 0.08, placement_R):
                        self.since = None
                        self.transition('settle')
                elif self.phase == 'settle':
                    m.move(self.release_goal + [0, 0, s.resolved['final_clearance_m']], 0.04, placement_R)
                    part = o['parts'][str(self.index)]
                    good = part['contained'] and part['support_N'] > 0.01 and (np.linalg.norm(b.get_linear_velocity()) < 0.012) and (np.linalg.norm(b.get_angular_velocity()) < 0.1)
                    self.since = (self.since if self.since is not None else t) if good else None
                    if self.since is not None and t - self.since > 0.3:
                        s.sorted.add(self.index)
                        self.transition('withdraw')
                elif self.phase == 'withdraw':
                    if self.ordinal + 1 == len(self.indices) and xy[1] > 0.3:
                        if self.last_empty_start is None:
                            actual_p, actual_R = s.dual.tool_world(self.side)
                            self.last_empty_start = (actual_p.copy(), actual_R.copy())
                            self.last_empty_progress = 0.0
                        start_p, start_R = self.last_empty_start
                        actual_p, actual_R = s.dual.tool_world(self.side)
                        if np.linalg.norm(actual_p - m.command) < 0.008:
                            self.last_empty_progress = min(1.0, self.last_empty_progress + 1 / 180)
                        u = self.last_empty_progress
                        end_p = np.array([-0.45 if self.side == 'left' else 0.45, 0.48, 1.12])
                        delta = Rotation.from_matrix(self.pick_R @ start_R.T).as_rotvec()
                        end_R = Rotation.from_rotvec(delta * u).as_matrix() @ start_R
                        arrived = m.move(start_p * (1 - u) + end_p * u, 0.18, end_R)
                        if u >= 1 and arrived:
                            if s.deposit_owner == self.side:
                                s.deposit_owner = None
                                s.events[f'{self.index}_deposit_released_s'] = t
                            self.complete = True
                            self.transition('complete')
                        s.dual.jaw_travel(self.side, self.travel)
                        return
                    kinds = [f'{self.index}_empty_fold', f'{self.index}_empty_swing', f'{self.index}_empty_park']
                    if self.joint_path_kind not in kinds:
                        folded = s.dual.commands[self.side][m.ids].copy()
                        folded[1:4] = [1.57, 2.4, -1.5]
                        self.begin_joint_path(kinds[0], t, joint_goal=folded, loaded=False)
                    if self.advance_joint_path():
                        if self.joint_path_kind == kinds[0]:
                            swing = s.dual.commands[self.side][m.ids].copy()
                            swing[0] = self.park_joints[0]
                            self.begin_joint_path(kinds[1], t, joint_goal=swing, loaded=False)
                        elif self.joint_path_kind == kinds[1]:
                            self.begin_joint_path(kinds[2], t, joint_goal=self.park_joints, loaded=False)
                        else:
                            if s.deposit_owner == self.side:
                                s.deposit_owner = None
                                s.events[f'{self.index}_deposit_released_s'] = t
                            self.m.R = self.pick_R.copy()
                            self.ordinal += 1
                            if self.ordinal == len(self.indices):
                                self.complete = True
                                self.transition('complete')
                            else:
                                self.index = self.indices[self.ordinal]
                                self.since = None
                                self.transition('approach')
            s.dual.jaw_travel(self.side, self.travel)
            if t - self.entered > 18:
                raise RuntimeError(f'Part{self.index}: timeout {self.phase}; visibility={getattr(self, "last_visibility", None)}')
        except RuntimeError as exc:
            self.failure = str(exc)
            print('FAILURE', self.failure, flush=True)

class InspectionARXController:
    """Two bounded native controllers with exclusive source/camera access."""

    def __init__(self, scene, mode):
        self.s, self.mode = (scene, mode)
        self.arms = [InspectionArmOperation(scene, mode, side) for side in ['left', 'right']]
        self.phase, self.failure, self.complete = ('park', None, False)
        self.last_visibility = self.view_stability = None

    def step(self, observation):
        if self.complete or self.failure or self.mode in ['absent', 'passive']:
            return
        for arm in self.arms:
            arm.step(observation)
            if arm.failure:
                self.failure = f'{arm.side}: {arm.failure}'
                return
        self.phase = ' / '.join((f'{arm.side}:{arm.phase}' for arm in self.arms))
        active = next((arm for arm in self.arms if arm.side == self.s.camera_owner), None)
        if active is not None:
            self.last_visibility = getattr(active, 'last_visibility', None)
            self.view_stability = active.view_stability
        if len(self.s.sorted) == 3:
            self.s.events.setdefault('completed_s', observation['time_s'])
            self.s.physical_completed = True
            self.s.success = self.s.final_valid(observation)
            if self.s.success:
                self.complete = True
                owner = self.s.deposit_owner
                for arm in self.arms:
                    if arm.side == owner:
                        self.s.events.setdefault(f'{arm.index}_deposit_released_s', observation['time_s'])
                self.s.deposit_owner = None
                self.s.cue_state = 'COMPLETE'
                self.s.display.set(self.s.cue_state)
                self.phase = 'complete'
                return
        if all((arm.complete for arm in self.arms)):
            self.complete = True
            self.s.events['completed_s'] = observation['time_s']
            self.s.physical_completed = len(self.s.sorted) == 3
            self.s.success = self.s.final_valid(observation)
            if self.s.success:
                self.s.cue_state = 'COMPLETE'
                self.s.display.set(self.s.cue_state)
