"""Task43: original free CAD lid, fixed locating clamp and native dispenser.

The symbolic open-time recipe is not an adhesive/curing simulation. Actual
carriage tracking completes the bead; native contact/release establishes seating.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig
from .arx_cell_task import (
    SIDES,
    ArmMotion,
    ContactBook,
    StatusDisplay,
    add_mounts,
    collide,
    jaws_open,
    remove_preview,
    set_signal,
)
from .paths import asset_path, portable_json_loads

# Only the left arm works; it keeps its recorded base and the right arm sits 0.60 m beside it
# (recorded: right [0.37, 0.08, 1.04], 0.72 m apart).
MOUNTS = rig.mounts_keeping("left", [-0.35, 0.08, 1.04])
LID = '/World/Products/WorkingCover'
HOUSING = '/World/Products/ActiveHousing'

def resolve_bonding(seed=0, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    r = dict(task_id=43, seed=seed, embodiment='dual_arx_x5', source_xy_offset_m=streams[0].uniform(-0.004, 0.004, 2).tolist() if layout else [0.0, 0.0], source_yaw_deg=float(streams[0].uniform(-5, 5)) if layout else 0.0, dispense_onset_s=float(streams[1].uniform(16, 20)) if timing else 18.0, window_relative_s=[2.0, 14.0], dispense_speed_m_s=0.04, lid_mass_scale=float(streams[2].uniform(0.9, 1.1)) if physical else 1.0, randomization=dict(layout=layout, timing=timing, physical=physical), bounds=dict(source_xy_offset_m=[-0.004, 0.004], source_yaw_deg=[-5, 5], dispense_onset_s=[16, 20], lid_mass_scale=[0.9, 1.1]), independence='Separate SeedSequence child streams for layout, process timing and physical properties', scoring='First released/contact-supported aligned seating, confirmed for0.50s, must begin2–14s after measured bead completion; earlier assembly contact invalidates timing', assumptions='Original matched CAD geometry with declared aluminium mass/friction; fixed visible housing clamp and removable lid handling stud; no chemical adhesion or product/robot attachments', controller='Privileged native-state joint-drive oracle; not a trained policy')
    r['sha256'] = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
    return r

def build_approved_dispenser(scene):
    """Reuse approved visible carriage/head on two bounded native linear axes."""
    from isaacsim.core.prims import SingleArticulation
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
    stage = scene.stage
    for name, com in [('base', [0, 0, 0]), ('x_carriage', [0.0465, 0.6, 1.35]), ('head', [0.0465, 0.4, 1.12])]:
        node = UsdGeom.Xform.Define(stage, '/World/NativeDispenser/' + name)
        initial = [0, 0, 0] if name == 'base' else [0.2, 0, 0] if name == 'x_carriage' else [0.2, 0.18, 0]
        node.AddTranslateOp().Set(Gf.Vec3d(*initial))
        p = node.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(p)
        PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0)
        m = UsdPhysics.MassAPI.Apply(p)
        m.CreateMassAttr(1 if name == 'base' else 0.15)
        m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
        m.CreateDiagonalInertiaAttr(Gf.Vec3f(0.002))
        PhysxSchema.PhysxRigidBodyAPI.Apply(p).CreateDisableGravityAttr(True)
    layer = stage.GetRootLayer()
    for name, body in [('Carriage', 'x_carriage'), ('YRail', 'x_carriage'), ('Head', 'head'), ('Nozzle', 'head')]:
        original = '/World/Dispenser/' + name
        if name == 'Carriage':
            UsdGeom.Xformable(stage.GetPrimAtPath(original)).GetOrderedXformOps()[0].Set(Gf.Vec3d(0.03, 0.6, 1.38))
        Sdf.CopySpec(layer, original, layer, '/World/NativeDispenser/' + body + '/' + name)
        stage.RemovePrim(original)
    anchor = UsdPhysics.FixedJoint.Define(stage, '/World/NativeDispenser/Anchor')
    anchor.CreateBody1Rel().SetTargets(['/World/NativeDispenser/base'])
    UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
    for axis, parent, child, initial in [('X', 'base', 'x_carriage', 0.2), ('Y', 'x_carriage', 'head', 0.18)]:
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/NativeDispenser/' + axis + 'Joint')
        joint.CreateBody0Rel().SetTargets(['/World/NativeDispenser/' + parent])
        joint.CreateBody1Rel().SetTargets(['/World/NativeDispenser/' + child])
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(-0.3)
        joint.CreateUpperLimitAttr(0.3)
        joint.CreateCollisionEnabledAttr(False)
        joint.GetPrim().AddAppliedSchema('PhysicsJointStateAPI:linear')
        joint.GetPrim().CreateAttribute('state:linear:physics:position', Sdf.ValueTypeNames.Float).Set(initial)
        joint.GetPrim().CreateAttribute('state:linear:physics:velocity', Sdf.ValueTypeNames.Float).Set(0)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        drive.CreateStiffnessAttr(6000)
        drive.CreateDampingAttr(100)
        drive.CreateMaxForceAttr(30)
        drive.CreateTargetPositionAttr(initial)
    collide(stage, ['/World/NativeDispenser/x_carriage/', '/World/NativeDispenser/head/'])
    for p in Usd.PrimRange(stage.GetPrimAtPath('/World/NativeDispenser')):
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexHull')
    return scene.world.scene.add(SingleArticulation('/World/NativeDispenser', name='bond_dispenser'))

class BondingScene:

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        from .bonding_scene import build_bonding_environment
        from .dual_arx import DualARX
        from .v2.geometry import body_envelope_points
        self.base = build_bonding_environment(output, resolution=(1280, 800))
        self.world, self.stage = (self.base.world, self.base.stage)
        self.resolved, self.events = (resolved, {})
        stage, c = (self.stage, self.base.cell)
        print('BUILD environment ready', flush=True)
        self.asset_manifest = portable_json_loads(Path(asset_path('/opt/industry/assets/v2/task43/prepared-knob40-v2/asset-manifest.json')).read_text())
        self.lid_mean = np.array(self.asset_manifest['prepared']['lid']['bounds_m']).mean(0)
        self.housing_mean = np.array(self.asset_manifest['prepared']['housing']['bounds_m']).mean(0)
        self.seat = np.array([0, 0.4, 0.824])
        source = np.array([-0.27, 0.25, 0.826 - 0.03208000183105469]) + [*resolved['source_xy_offset_m'], 0]
        yaw = np.deg2rad(resolved['source_yaw_deg'])
        R = Rotation.from_euler('z', yaw).as_matrix()
        remove_preview(c, '/World/Products/InputCover2')
        center = source + R @ self.lid_mean
        if mode == 'seat':
            R = np.eye(3)
            center = self.seat + self.lid_mean + [0, 0, 0.008]
        c.asset('Products/WorkingCover', Path(asset_path('/opt/industry/assets/v2/task43/prepared-knob40-v2/lid.usd')), center, material='steel', yaw=yaw if mode != 'seat' else 0)
        self.world.scene.remove_object('WorkingCover', registry_only=True)
        root = stage.GetPrimAtPath(LID)
        UsdPhysics.RigidBodyAPI.Apply(root)
        source_mass = self.asset_manifest['prepared']['lid']['mass_kg']
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(source_mass * resolved['lid_mass_scale'])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, np.array(self.asset_manifest['prepared']['lid']['center_of_mass_m']) - self.lid_mean)))
        dimensions = np.ptp(np.array(self.asset_manifest['prepared']['lid']['bounds_m']), axis=0)
        inertia = source_mass * resolved['lid_mass_scale'] / 12 * (np.sum(dimensions ** 2) - dimensions ** 2)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
        rb.CreateSolverPositionIterationCountAttr(64)
        rb.CreateSolverVelocityIterationCountAttr(16)
        rb.CreateSleepThresholdAttr(5e-05)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        material = PhysicsMaterial('/World/BondContactMaterial', static_friction=0.8, dynamic_friction=0.65, restitution=0)
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(True)
                if p.IsA(UsdGeom.Mesh):
                    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(p).CreateSdfResolutionAttr(1024)
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                api.CreateContactOffsetAttr(0.0003)
                api.CreateRestOffsetAttr(0)
                UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
        self.asset_manifest['runtime_collision_overrides'] = dict(lid_sdf_resolution=1024, original_prepared_resolution=512, geometry_unchanged=True, reason='Refine collision sampling across the matched CAD mating clearance')
        self.part = self.world.scene.add(SingleRigidPrim(LID, name='free_cover'))
        self.envelope = body_envelope_points(stage, root)
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS, gripper_velocity_cap=False)
        add_mounts(c, MOUNTS, base_z=0.81)
        self.static_colliders = collide(stage, ['/World/Building/Floor', '/World/Station/', '/World/Fixture/', '/World/Input/', '/World/Output/', '/World/Gantry/', '/World/ARXMounts/', HOUSING], material)
        for side in ['left', 'right']:
            for p in Usd.PrimRange(stage.GetPrimAtPath('/World/ARX/' + side)):
                if p.HasAPI(UsdPhysics.CollisionAPI) and any(('/' + n in str(p.GetPath()) for n in ['link7', 'link8'])):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
        print('BUILD products/arms ready', flush=True)
        self.dispenser = build_approved_dispenser(c)
        stage.RemovePrim('/World/Dispenser/SupplyLineC')
        self.hose = UsdGeom.BasisCurves.Define(stage, '/World/Dispenser/FlexibleSupply')
        self.hose.CreateTypeAttr('cubic')
        self.hose.CreateBasisAttr('bezier')
        self.hose.CreateWrapAttr('nonperiodic')
        self.hose.CreateCurveVertexCountsAttr([4])
        self.hose.CreateWidthsAttr([0.008])
        self.hose.SetWidthsInterpolation('constant')
        self.hose.CreatePointsAttr([[0.03, 0.75, 1.55], [0.1, 0.72, 1.59], [0.23, 0.68, 1.51], [0.23, 0.64, 1.45]])
        UsdShade.MaterialBindingAPI.Apply(self.hose.GetPrim()).Bind(c.appearance.material('black').material)
        self.contacts = ContactBook(self.world, [LID], watched_prefixes=('/World/NativeDispenser/',))
        self.glue_paths = [f'/World/Process/AdhesiveBead{i}' for i in range(4)]
        for path in self.glue_paths:
            UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
        self.cue = c.appearance.material('bond_cue', color=[0.05, 0.2, 0.8], emissive=True)
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/Control/Beacon')).Bind(self.cue.material)
        for physics_scene in stage.Traverse():
            if physics_scene.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(physics_scene).CreateEnableStabilizationAttr(True)
        self.asset_manifest['runtime_collision_overrides']['scene_stabilization'] = True
        print('BUILD resetting native physics', flush=True)
        self.world.reset()
        self.dual.initialize()
        self.dispenser.initialize()
        self.axes = [self.dispenser.dof_names.index(n) for n in ['XJoint', 'YJoint']]
        from isaacsim.core.utils.types import ArticulationAction
        self.dispenser.set_joint_positions(np.array([0.2, 0.18]), joint_indices=np.array(self.axes))
        self.dispenser.set_joint_velocities(np.zeros(2), joint_indices=np.array(self.axes))
        self.dispenser.apply_action(ArticulationAction(joint_positions=np.array([0.2, 0.18]), joint_indices=np.array(self.axes)))
        right = self.dual.commands['right']
        right[self.dual.names['right'].index('joint1')] = -1.8
        self.dual.robots['right'].set_joint_positions(right)
        self.part.initialize()
        self.part.set_world_pose(position=center, orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]) if mode != 'seat' else np.array([1.0, 0, 0, 0]))
        self.part.set_linear_velocity(np.zeros(3))
        self.part.set_angular_velocity(np.zeros(3))
        self.contacts.interval.clear()
        self.contacts.pairs.clear()
        self.contacts.minimum.clear()
        self.contacts.active.clear()
        self.contacts.unexpected.clear()
        self.dual.jaw_travel('left', 0.044)
        self.dual.robots['left'].set_joint_positions(self.dual.commands['left'])
        print('INITIAL_LID_NATIVE', self.part.get_world_pose(), flush=True)
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=43)
        self.retained_since = None
        self.success = False
        self.physical_completed = False
        opens, closes = resolved['window_relative_s']
        self.display = StatusDisplay(c, 'Recipe', {'PROCESSING': 'ADHESIVE PROCESS / WAIT', 'WINDOW': f'PLACE COVER / {opens:g}-{closes:g} s', 'EXPIRED': 'BOND WINDOW EXPIRED', 'COMPLETE': 'COVER SEATED'})
        self.display.set('PROCESSING')
        self.cue_state = None
        self.dispense_phase = 'parked'
        self.dispense_target = np.array([0.2, 0.18])
        self.segment = 0
        self.corners = np.array([[-0.093, -0.0715], [0, -0.0715], [0, 0.0715], [-0.093, 0.0715], [-0.093, -0.0715]])

    def observe(self):
        p, q = self.part.get_world_pose()
        R = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
        origin = p - R @ self.lid_mean
        relative = origin - self.seat
        points = self.envelope @ R.T + p - self.seat
        force = [self.contacts.force(LID, f'/link{i}') for i in [7, 8]]
        # Either arm may seat the cover: the jaws that must be open are those of every arm whose
        # fingers have held it, and neither arm's fingers may still touch it (the left arm's in
        # the recorded oracle, as before). `jaws` stays the left arm's: the oracle reads it.
        jaws_by_side = {side: self.dual.robots[side].get_joint_positions()[[self.dual.names[side].index(f'joint{i}') for i in [7, 8]]] for side in SIDES}
        jaws = jaws_by_side['left']
        released = bool(jaws_open(jaws_by_side, self.contacts.finger_sides(LID)) and max(force) < 0.05 and (not any((self.contacts.touching(LID, f'/World/ARX/{side}/' + self.dual.manifest['relative_body_paths'][f'link{i}']) for side in SIDES for i in [7, 8]))))
        support = self.contacts.force(LID, HOUSING)
        support_contact = any((patch['active'] and LID in patch['actors'] and any((actor.startswith(HOUSING + '/') for actor in patch['actors'])) for patch in self.contacts.active.values()))
        aligned = bool(np.linalg.norm(relative[:2]) < 0.0025 and abs(relative[2]) < 0.002 and (R[2, 2] > np.cos(np.deg2rad(2))) and (np.abs(points[:, :2]).max(0) < [0.0574, 0.0824]).all())
        stable = bool(np.linalg.norm(self.part.get_linear_velocity()) < 0.012 and np.linalg.norm(self.part.get_angular_velocity()) < 0.1)
        t = float(self.world.current_time)
        if support > 0.1 and 'first_assembly_contact_s' not in self.events:
            self.events['first_assembly_contact_s'] = t
        physical = bool((support > 0.1 or support_contact) and released and aligned and stable)
        if physical:
            self.retained_since = self.retained_since if self.retained_since is not None else t
            if t - self.retained_since >= 0.5 and (not self.physical_completed):
                self.physical_completed = True
                self.events.update(seated_release_s=self.retained_since, seating_confirmed_s=t)
                glue = self.events.get('glue_completed_s')
                first = self.events.get('first_assembly_contact_s')
                opens, closes = self.resolved['window_relative_s']
                self.success = bool(glue is not None and glue + opens <= self.retained_since <= glue + closes and (first is not None) and (first >= glue + opens) and ('bilateral_grasp_s' in self.events) and ('source_cleared_s' in self.events))
                self.events['temporal_success'] = self.success
        else:
            self.retained_since = None
        return dict(time_s=t, lid_position=p.tolist(), lid_quaternion=np.asarray(q).tolist(), lid_origin=origin.tolist(), relative_origin=relative.tolist(), lowest_point_world_z_m=float(points[:, 2].min() + self.seat[2]), source_support_force_N=self.contacts.force(LID, '/World/Input/LidPallet2'), lid_velocity=self.part.get_linear_velocity().tolist(), lid_angular_velocity=self.part.get_angular_velocity().tolist(), tool_position=self.dual.tool_world('left')[0].tolist(), finger_forces_N=force, jaws_m=jaws.tolist(), support_force_N=support, support_contact_active=support_contact, aligned=aligned, stable=stable, released=released, physical_seating=physical, dispenser_xy=self.dispenser.get_joint_positions()[self.axes].tolist(), unexpected_contacts=self.contacts.unexpected.copy())

    def process_step(self, o):
        """Adhesive dispenser and process cue on each control tick.

        Recorded inside the controller (``dispenser_step``); it is the machine's own
        schedule, independent of the robot, so the scene runs it for every policy.
        """
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import UsdGeom
        s = self
        t = o['time_s']
        actual = np.array(o['dispenser_xy'])
        x = float(0.03 + actual[0])
        s.hose.GetPointsAttr().Set([[0.03, 0.75, 1.55], [0.03 + float(actual[0]) / 2, 0.72, 1.59], [x, 0.68, 1.51], [x, 0.64, 1.45]])
        if self.dispense_phase == 'parked' and t >= s.resolved['dispense_onset_s']:
            self.dispense_phase = 'approach'
        goal = self.dispense_target.copy()
        speed = 0.1
        if self.dispense_phase == 'approach':
            goal = self.corners[0]
        elif self.dispense_phase == 'dispense':
            goal = self.corners[self.segment]
            speed = s.resolved['dispense_speed_m_s']
        elif self.dispense_phase == 'withdraw':
            goal = np.array([0.2, 0.18])
            if actual[0] > 0.14 and actual[1] > 0.1:
                s.events.setdefault('dispenser_clear_s', t)
        if np.linalg.norm(actual - self.dispense_target) < 0.003:
            delta = goal - self.dispense_target
            self.dispense_target += delta * min(1, speed / CONTROL_HZ / max(np.linalg.norm(delta), 1e-12))
        if np.linalg.norm(actual - goal) < 0.001 and np.linalg.norm(self.dispense_target - goal) < 1e-06:
            if self.dispense_phase == 'approach':
                self.dispense_phase = 'dispense'
                self.segment = 1
            elif self.dispense_phase == 'dispense':
                UsdGeom.Imageable(s.stage.GetPrimAtPath(s.glue_paths[[3, 2, 1, 0][self.segment - 1]])).MakeVisible()
                if self.segment == 4:
                    s.events['glue_completed_s'] = t
                    self.dispense_phase = 'withdraw'
                else:
                    self.segment += 1
            elif self.dispense_phase == 'withdraw':
                self.dispense_phase = 'clear'
                s.events.setdefault('dispenser_clear_s', t)
        s.dispenser.apply_action(ArticulationAction(joint_positions=self.dispense_target, joint_indices=np.array(s.axes)))
        glue = s.events.get('glue_completed_s')
        opens, closes = s.resolved['window_relative_s']
        state = 'PROCESSING' if glue is None or t < glue + opens else 'WINDOW' if t <= glue + closes else 'EXPIRED'
        if s.success:
            state = 'COMPLETE'
        if state != s.cue_state:
            color = {'PROCESSING': [0.04, 0.18, 1.0], 'WINDOW': [0.03, 0.9, 0.15], 'EXPIRED': [1.0, 0.03, 0.02], 'COMPLETE': [0.03, 0.9, 0.15]}[state]
            set_signal(s.cue, color)
            s.display.set(state)
            s.cue_state = state
        o.update(dispenser_phase=self.dispense_phase, dispenser_target_xy=self.dispense_target.tolist(), cue_state=state)

class BondingController:

    def __init__(self, scene, mode):
        self.s, self.mode = (scene, mode)
        self.motion = ArmMotion(scene.dual)
        self.phase = 'settle'
        self.entered = 0.0
        self.failure = None
        self.complete = False
        self.travel = 0.044
        self.grip_since = None
        self.missing_since = None
        self.offset = None
        self.target = None

    def transition(self, phase, target=None):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.target = None if target is None else np.array(target, float)
        self.s.events['phase_' + phase + '_s'] = self.entered
        print('PHASE', phase, self.entered, flush=True)

    def move_lid(self, o, origin, speed):
        lid_R = Rotation.from_quat(np.asarray(o['lid_quaternion'])[[1, 2, 3, 0]]).as_matrix()
        tool_p, tool_R = self.s.dual.tool_world('left')
        self.motion.orientation_feedback = True
        self.motion.R = lid_R.T @ tool_R
        error = np.asarray(origin) - np.array(o['lid_origin'])
        self.motion.move(tool_p + error, speed)
        return np.linalg.norm(error) < 0.002 and np.linalg.norm(Rotation.from_matrix(lid_R).as_rotvec()) < 0.003

    def step(self, o):
        s = self.s
        t = o['time_s']
        if self.failure or self.mode in ['passive', 'seat', 'absent']:
            return
        p = np.array(o['lid_origin'])
        grasp = p + [0, 0, 0.064]
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if s.physical_completed and (not o['physical_seating']):
                raise RuntimeError('Seated cover lost stable released support')
            if p[2] < 0.75:
                raise RuntimeError('Lid fell outside its fixture')
            if self.phase == 'settle' and t > 1.5:
                self.transition('above_source', grasp + [0, 0, 0.12])
            elif self.phase == 'above_source':
                if self.motion.move(self.target):
                    self.transition('descend', grasp)
            elif self.phase == 'descend':
                if self.motion.move(self.target, 0.025):
                    self.transition('close')
            elif self.phase == 'close':
                self.motion.move(grasp, 0.025)
                self.travel = max(0.001, self.travel - 0.01 / CONTROL_HZ)
                both = min(o['finger_forces_N']) > 8.0
                self.grip_since = (self.grip_since if self.grip_since is not None else t) if both else None
                if self.grip_since is not None and t - self.grip_since > 0.25:
                    self.offset = np.array(o['lid_position']) - np.array(o['tool_position'])
                    s.events['bilateral_grasp_s'] = t
                    self.transition('lift', np.array(o['tool_position']) + [0, 0, 0.12])
            elif self.phase == 'lift':
                if self.motion.move(self.target, 0.045):
                    if o['lowest_point_world_z_m'] < 0.86 or o['source_support_force_N'] > 0.1:
                        raise RuntimeError('Original lid has not physically cleared its source support')
                    s.events['source_cleared_s'] = t
                    if self.mode == 'grasp':
                        self.transition('hold', self.target)
                    else:
                        actual_R = Rotation.from_quat(np.asarray(o['lid_quaternion'])[[1, 2, 3, 0]]).as_matrix()
                        correction = actual_R.T
                        self.offset = correction @ (np.array(o['lid_position']) - np.array(o['tool_position']))
                        self.motion.R = correction @ self.motion.R
                        s.events['settled_grasp_alignment_R'] = correction.tolist()
                        self.transition('standby', s.seat + s.lid_mean - self.offset + [0.0, -0.3, 0.1])
            elif self.phase == 'hold':
                self.motion.move(self.target if self.target is not None else np.array(o['tool_position']))
                if t - self.entered > 1.5:
                    self.complete = True
            elif self.phase == 'standby':
                if self.move_lid(o, s.seat + [0, -0.3, 0.1], 0.07):
                    self.transition('wait_window', self.target)
            elif self.phase == 'wait_window':
                self.move_lid(o, s.seat + [0, -0.3, 0.1], 0.025)
                glue = s.events.get('glue_completed_s')
                delay = 15 if self.mode == 'late' else -1 if self.mode == 'early' else 3
                if glue is not None and 'dispenser_clear_s' in s.events and (t >= glue + delay):
                    self.motion.orientation_feedback = True
                    self.motion.position_gain = 2.0
                    self.motion.command = np.array(o['tool_position'])
                    self.motion.correction[:] = 0
                    self.transition('align', s.seat + s.lid_mean - self.offset + [0, 0, 0.1])
            elif self.phase == 'align':
                if self.move_lid(o, s.seat + [0, 0, 0.1], 0.12):
                    self.transition('lower', s.seat + s.lid_mean - self.offset + [0, 0, 0.008])
            elif self.phase == 'lower':
                if self.move_lid(o, s.seat + [0, 0, 0.015], 0.045):
                    self.motion.command = np.array(o['tool_position'])
                    self.motion.correction[:] = 0
                    self.motion.position_gain = 2.0
                    self.transition('precision_lower')
            elif self.phase == 'precision_lower':
                lid_R = Rotation.from_quat(np.asarray(o['lid_quaternion'])[[1, 2, 3, 0]]).as_matrix()
                tool_p, tool_R = s.dual.tool_world('left')
                self.motion.R = lid_R.T @ tool_R
                error = s.seat + [0, 0, 0.008] - np.array(o['lid_origin'])
                self.motion.move(tool_p + error, 0.015)
                good = np.linalg.norm(error[:2]) < 0.00025 and abs(error[2]) < 0.00025 and (np.linalg.norm(Rotation.from_matrix(lid_R).as_rotvec()) < 0.0015)
                self.seat_ready_since = getattr(self, 'seat_ready_since', None) or t if good else None
                if self.seat_ready_since is not None and t - self.seat_ready_since > 0.1:
                    self.transition('seat_contact')
            elif self.phase == 'seat_contact':
                self.move_lid(o, s.seat + [0, 0, -5e-05], 0.005)
                if o['support_force_N'] > 20:
                    raise RuntimeError('Excessive native seating contact force')
                if o['support_force_N'] > 0.2 and np.linalg.norm(np.array(o['relative_origin'])[:2]) < 0.0002 and (abs(o['relative_origin'][2]) < 0.0005):
                    self.transition('release', np.array(o['tool_position']))
                elif t - self.entered > 4:
                    raise RuntimeError('No valid supported seating contact')
            elif self.phase == 'release':
                self.motion.move(self.target, 0.025)
                self.travel = min(0.044, self.travel + 0.04 / CONTROL_HZ)
                if min(o['jaws_m']) > 0.041 and max(o['finger_forces_N']) < 0.05:
                    self.transition('withdraw', np.array(o['tool_position']) + [0, 0, 0.1])
            elif self.phase == 'withdraw':
                reached = self.motion.move(self.target, 0.045)
                if reached and s.physical_completed:
                    self.complete = True
                    self.transition('complete')
                elif t - self.entered > 8:
                    raise RuntimeError('Released original lid failed to settle in housing')
            if self.phase in ['lift', 'standby', 'wait_window', 'align', 'lower', 'precision_lower', 'seat_contact']:
                if min(o['finger_forces_N']) < 8:
                    self.travel = max(0.001, self.travel - 0.003 / CONTROL_HZ)
                if min(o['finger_forces_N']) < 0.15:
                    self.missing_since = self.missing_since if self.missing_since is not None else t
                else:
                    self.missing_since = None
                if self.missing_since is not None and t - self.missing_since > 0.1:
                    raise RuntimeError('Sustained bilateral grasp loss during transfer')
                if np.linalg.norm(np.array(o['lid_position']) - np.array(o['tool_position']) - self.offset) > 0.04:
                    raise RuntimeError('Lid separated from native hand')
            if t - self.entered > 22 and self.phase not in ['wait_window', 'complete']:
                raise RuntimeError('Phase timeout: ' + self.phase)
            s.dual.jaw_travel('left', self.travel)
        except RuntimeError as exc:
            self.failure = str(exc)
            print('FAILURE', self.failure, flush=True)
