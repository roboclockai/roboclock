"""Task41 zincate dip and rinse transfer: native contact observer and bounded oracle control.

The dry part starts on the loading pad between the baths. The arm picks it up, lowers it into
the zincate bath until it is fully immersed (the recipe clock starts there), lifts it out
inside the recipe window and leaves it released in the rinse bath.

Electrochemistry and liquid drag/buoyancy are outside this task model. Actual
mesh poses/contact events determine immersion, exit and supported release.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig
from .paths import asset_path, portable_json_loads
from .plating_scene import LOAD_PAD_CENTER, LOAD_PAD_TOP_Z
from .v2.kinematics import UrdfArm

# Only the right arm works; it keeps its recorded base and the left arm sits 0.60 m beside it
# (recorded: left [-0.43, 0.08, 1.02], right [0, 0.08, 1.02]).
MOUNTS = rig.mounts_keeping('right', [0.0, 0.08, 1.02])


def resolve_plating(seed, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    minimum = float(streams[1].uniform(18, 22)) if timing else 20.0
    duration = float(streams[1].uniform(18, 22)) if timing else 20.0
    dxy = streams[0].uniform(-0.008, 0.008, 2).tolist() if layout else [0.0, 0.0]
    yaw = float(streams[0].uniform(-8, 8)) if layout else 0.0
    mass = float(streams[2].uniform(0.135, 0.165)) if physical else 0.15
    config = dict(task_id=41, embodiment='dual_arx_x5', seed=seed, process_min_s=minimum, process_max_s=minimum + duration, recipe_provenance='Approved 20–40s demonstration recipe on October20; not expert-calibrated chemistry', layout_offset_xy_m=dxy, part_yaw_deg=yaw, part_mass_kg=mass, randomization=dict(layout=layout, timing=timing, physical=physical), bounds=dict(layout_xy_m=[-0.008, 0.008], part_yaw_deg=[-8, 8], recipe_min_s=[18, 22], window_duration_s=[18, 22], mass_kg=[0.135, 0.165]), independence='Separate SeedSequence child streams for layout, timing and physical properties', liquid_level_z_m=0.885, controller_uses_privileged_state=True)
    config['sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return config

class PlatingScene:

    def __init__(self, assets, manifest, output, resolved):
        import omni.physx
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

        from .dual_arx import DualARX
        from .plating_physics import add_mounts_and_collisions
        from .plating_scene import build_plating_environment
        self.base = build_plating_environment(assets, output, resolution=(1280, 800))
        self.world, self.stage = (self.base.world, self.base.stage)
        self.physics_hz = 1.0 / float(self.world.get_physics_dt())
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS, gripper_velocity_cap=False)
        self.collisions = add_mounts_and_collisions(self.base, self.dual.mounts)
        self.resolved = resolved
        self.output = output
        self.part_path = '/World/MetalPart'
        root = UsdGeom.Xform.Define(self.stage, self.part_path).GetPrim()
        asset = Path(asset_path('/opt/industry/assets/v2/task49/prepared-v3/port_block.usda'))
        if not asset.is_file():
            raise FileNotFoundError(asset)
        root.GetReferences().AddReference(str(asset))
        material = PhysicsMaterial('/World/ContactMaterial', static_friction=0.8, dynamic_friction=0.65, restitution=0)
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(True)
                api = PhysxSchema.PhysxCollisionAPI.Apply(p)
                api.CreateContactOffsetAttr(0.0005)
                api.CreateRestOffsetAttr(0)
                UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(self.base.appearance.material('steel').material)
        for side in self.dual.robots:
            for p in Usd.PrimRange(self.stage.GetPrimAtPath('/World/ARX/' + side)):
                if any(('/' + n + '/' in str(p.GetPath()) or str(p.GetPath()).endswith('/' + n) for n in ('link7', 'link8'))) and p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(p).Bind(material.material, materialPurpose='physics')
        # Half the part's 42 mm height above the pad, plus 2 mm to settle.
        position = np.array([LOAD_PAD_CENTER[0], LOAD_PAD_CENTER[1], LOAD_PAD_TOP_Z + 0.023])
        position[:2] += resolved['layout_offset_xy_m']
        yaw = np.deg2rad(resolved['part_yaw_deg'])
        self.part = self.world.scene.add(SingleRigidPrim(self.part_path, name='metal_part', position=position, orientation=np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])))
        mass = UsdPhysics.MassAPI.Apply(root)
        mass.CreateMassAttr(resolved['part_mass_kg'])
        inertia = np.array(mass.GetDiagonalInertiaAttr().Get()) * resolved['part_mass_kg'] / 0.15
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, inertia)))
        PhysxSchema.PhysxRigidBodyAPI.Apply(root).CreateSleepThresholdAttr(0)
        PhysxSchema.PhysxContactReportAPI.Apply(root).CreateThresholdAttr(0)
        self.signal_lights = {}
        self.signal_positions = {}
        self.signal_state = None
        self.base.appearance.rounded('/World/ProcessSignal/Housing', [0.15, 0.751, 1.102], [0.145, 0.025, 0.045], 'frame', radius=0.004)
        for index, (name, color) in enumerate([('PROCESSING', [0.03, 0.25, 1.0]), ('WINDOW', [0.03, 0.9, 0.15]), ('EXPIRED', [1.0, 0.04, 0.02])]):
            light = UsdGeom.Sphere.Define(self.stage, '/World/ProcessSignal/' + name)
            light.CreateRadiusAttr(0.01)
            op = light.AddTranslateOp()
            op.Set(Gf.Vec3d(0.112 + 0.038 * index, 0.8, 1.102))
            self.signal_positions[name] = (op, 0.112 + 0.038 * index)
            material_name = 'process_signal_' + name.lower()
            m = self.base.appearance.material(material_name, color=color, roughness=0.25, emissive=True)
            m.shaders_list[0].CreateInput('emissive_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            UsdShade.MaterialBindingAPI.Apply(light.GetPrim()).Bind(m.material)
            self.signal_lights[name] = light
        self.interval = {}
        self.unexpected = {}
        self.first_contacts = {}
        self.contact_geometry = {}

        def contact(headers, data):
            for h in headers:
                actors = sorted((str(PhysicsSchemaTools.intToSdfPath(v)) for v in (h.actor0, h.actor1)))
                part = self.part_path in actors
                arm = next((a for a in actors if a.startswith('/World/ARX/')), None)
                finger = arm is not None and arm.endswith(('/link7', '/link8'))
                for i in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    d = data[i]
                    normal = float(abs(np.dot(np.asarray(d.impulse), np.asarray(d.normal))))
                    if normal <= 1e-08:
                        continue
                    key = ' | '.join(actors)
                    separation = getattr(d, 'separation', None)
                    if separation is not None:
                        entry = self.contact_geometry.setdefault(key, dict(min_separation_m=float('inf'), samples=0))
                        entry['samples'] += 1
                        if float(separation) < entry['min_separation_m']:
                            entry.update(min_separation_m=float(separation), time_s=float(self.world.current_time), position_m=list(map(float, d.position)))
                    self.first_contacts.setdefault(key, dict(actors=actors, time_s=float(self.world.current_time)))
                    if part:
                        if finger:
                            k = 'finger7' if arm.endswith('/link7') else 'finger8'
                        elif any(('/LoadPad' in a for a in actors)):
                            k = 'load_support'
                        elif any(('/Bath0/' in a for a in actors)):
                            k = 'plating_support'
                        elif any(('/Bath1/' in a for a in actors)):
                            k = 'rinse_support'
                        else:
                            k = 'other_part_contact'
                        self.interval[k] = self.interval.get(k, 0) + normal * self.physics_hz
                    if arm and (not (part and finger)) and (normal > 1e-06):
                        entry = self.unexpected.setdefault(key, dict(actors=actors, time_s=float(self.world.current_time), impulse_Ns=normal))
                        # scoring.voiding_contacts judges the contact by how hard it pressed.
                        entry['peak_force_N'] = max(entry.get('peak_force_N', 0.0), normal * self.physics_hz)
        self.subscription = omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(contact)
        points = []
        cache = UsdGeom.XformCache()
        inv = cache.GetLocalToWorldTransform(root).GetInverse()
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                transform = cache.GetLocalToWorldTransform(p) * inv
                points.extend((tuple(transform.Transform(Gf.Vec3d(*map(float, v)))) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()))
        self.envelope = np.unique(np.array(points), axis=0)
        self.world.reset()
        self.dual.initialize()
        self.world.reset()
        self.dual.initialize()
        self.part.initialize()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=41)
        self.process_start = None
        self.first_exit = None
        self.final_exit = None
        self.retained_since = None
        self.events = {}
        self.success = False
        self.asset_record = dict(source_manifest=portable_json_loads((asset.parent / 'asset-manifest.json').read_text()), original_usd_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(), mass_kg=resolved['part_mass_kg'], collision='Existing convex-decomposition exterior-transport candidate; bore insertion is not scored', physics=dict(static_friction=0.8, dynamic_friction=0.65, restitution=0, sleep_threshold=0), limitations='No electrochemistry, buoyancy/drag or simulated change of coating thickness')

    def process_step(self, o):
        """Recipe signal lamps on each control tick (the runner called this)."""
        self.update_signal(o)

    def update_signal(self, o):
        from pxr import Gf
        elapsed = o['elapsed_s'] or 0.0
        if self.final_exit is not None:
            elapsed = self.final_exit - self.process_start
        state = 'PROCESSING' if elapsed < self.resolved['process_min_s'] else 'WINDOW' if elapsed <= self.resolved['process_max_s'] else 'EXPIRED'
        if self.final_exit is not None and elapsed < self.resolved['process_min_s']:
            state = 'EXPIRED'
        if state != self.signal_state:
            for name, (op, x) in self.signal_positions.items():
                op.Set(Gf.Vec3d(x, 0.733 if name == state else 0.8, 1.102))
            self.signal_state = state

    def observe(self):
        t = float(self.world.current_time)
        p, q = self.part.get_world_pose()
        R = Rotation.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_matrix()
        points = self.envelope @ R.T + p
        low = points.min(0)
        high = points.max(0)
        v = np.asarray(self.part.get_linear_velocity())
        w = np.asarray(self.part.get_angular_velocity())
        submerged = bool(high[2] < self.resolved['liquid_level_z_m'] - 0.001)
        plating_inside = bool(low[0] > -0.34 - 0.2 and high[0] < -0.34 + 0.2 and (low[1] > 0.43 - 0.19) and (high[1] < 0.43 + 0.19) and (low[2] > 0.65))
        if self.process_start is None and submerged and plating_inside:
            self.process_start = t
            self.events['plating_started_s'] = t
        if self.process_start is not None and self.first_exit is None and (high[2] > self.resolved['liquid_level_z_m'] + 0.001):
            self.first_exit = t
            self.events['first_surface_exit_s'] = t
        if self.process_start is not None and self.final_exit is None and (low[2] > self.resolved['liquid_level_z_m'] + 0.001):
            self.final_exit = t
            self.events['whole_part_exit_s'] = t
        inside = bool(low[0] > 0.34 - 0.2 and high[0] < 0.34 + 0.2 and (low[1] > 0.43 - 0.19) and (high[1] < 0.43 + 0.19) and (low[2] > 0.65) and submerged)
        floor = bool(self.interval.get('rinse_support', 0) > 0.4)
        released = bool(self.interval.get('finger7', 0) < 0.05 and self.interval.get('finger8', 0) < 0.05)
        stable = bool(np.linalg.norm(v) < 0.015 and np.linalg.norm(w) < 0.1 and (R[2, 2] > np.cos(np.deg2rad(5))))
        retained = inside and floor and released and stable
        self.retained_since = t if retained and self.retained_since is None else self.retained_since if retained else None
        self.success = bool(self.retained_since is not None and t - self.retained_since >= 0.5)
        if self.success:
            self.events.setdefault('supported_released_retention_s', t)
        tp, tR = self.dual.tool_world('right')
        return dict(time_s=t, elapsed_s=None if self.process_start is None else t - self.process_start, part_position=p.tolist(), part_rotation=R.tolist(), linear_velocity=v.tolist(), angular_velocity=w.tolist(), geometry_low=low.tolist(), geometry_high=high.tolist(), submerged=submerged, rinse_inside=inside, rinse_floor=floor, released=released, stable=stable, forces=self.interval.copy(), tool_position=tp.tolist(), tool_rotation=tR.tolist(), unexpected_contacts=self.unexpected.copy())

# Where the oracle dips the part: the bath centre, the part centre 2 mm above its rest height on
# the bath floor (0.7995 m), so the gripper never presses it into the floor.
DIP_XY = (-0.34, 0.43)
DIP_PART_Z = 0.8015
CARRY_PHASES = ('lift_from_load', 'above_plating', 'descend_plating', 'wait_recipe', 'lift_from_plating', 'bridge', 'above_rinse', 'descend_rinse')


class PlatingController:

    def __init__(self, scene, mode='nominal'):
        self.s = scene
        self.mode = mode
        self.phase = 'settle'
        self.entered = scene.world.current_time
        d = scene.dual
        self.kin = UrdfArm(d.manifest['derived_urdf'], 'link6', d.manifest['tool']['offset_m'], np.eye(3), [f'joint{i}' for i in range(1, 7)])
        self.ids = [d.names['right'].index(n) for n in self.kin.arm_names]
        self.R = d.rotation @ Rotation.from_euler('y', np.pi / 2).as_matrix()
        self.start = d.tool_world('right')[0]
        self.target = self.start.copy()
        self.correction = np.zeros(3)
        self.failure = None
        self.grasp_since = None
        self.travel = 0.04
        self.held_offset = None
        self.complete = False
        self.initial_posture = 'Nominal reset posture'

    def transition(self, phase, target=None):
        self.phase = phase
        self.entered = self.s.world.current_time
        self.start = self.s.dual.tool_world('right')[0]
        self.target = self.start.copy() if target is None else np.asarray(target, dtype=float)
        self.s.events['phase_' + phase + '_s'] = float(self.entered)
        print('PHASE', phase, float(self.entered), flush=True)

    def move(self, o, speed=0.045):
        d = self.s.dual
        t = o['time_s'] - self.entered
        progress = min(1.0, t / max(0.5, np.linalg.norm(self.target - self.start) / speed))
        desired = self.start + (self.target - self.start) * progress
        measured = np.asarray(o['tool_position'])
        self.correction = np.clip(self.correction + 0.8 * (desired - measured) / CONTROL_HZ, -0.015, 0.015)
        bp, bR = d.body_world('right', 'base_link')
        profile = d.manifest['profile']
        q, error = self.kin.solve(bR.T @ (desired + self.correction - bp), bR.T @ self.R, dict(zip(d.names['right'], d.commands['right'])), profile['arm_lower_rad'], profile['arm_upper_rad'])
        if error > 0.003:
            raise RuntimeError(f'Unreachable {self.phase}: {error}')
        d.commands['right'][self.ids] += np.clip(q - d.commands['right'][self.ids], -0.6 / CONTROL_HZ, 0.6 / CONTROL_HZ)
        return progress >= 1 and np.linalg.norm(measured - self.target) < 0.004

    def step(self, o):
        if self.failure:
            return
        try:
            t = o['time_s']
            age = t - self.entered
            p = np.asarray(o['part_position'])
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if p[2] < 0.63:
                raise RuntimeError('Part fell through or outside the bath support')
            if self.mode in ('passive', 'absent'):
                return
            if self.phase == 'settle':
                if age > 2 and o['forces'].get('load_support', 0) > 0.4:
                    self.transition('central_standby', [0, 0.3, 1.16])
            elif self.phase == 'central_standby':
                if self.move(o):
                    self.transition('above_load', [p[0], p[1], 1.06])
            elif self.phase == 'above_load':
                if self.move(o):
                    self.transition('descend_grasp', p + [0, 0, 0.006])
            elif self.phase == 'descend_grasp':
                if self.move(o, 0.035):
                    self.transition('close_gripper')
            elif self.phase == 'close_gripper':
                self.move(o, 0.025)
                both = min(o['forces'].get('finger7', 0), o['forces'].get('finger8', 0)) >= 2.5
                self.travel = max(0.009, self.travel - 0.01 / CONTROL_HZ)
                self.grasp_since = t if both and self.grasp_since is None else self.grasp_since if both else None
                if self.grasp_since is not None and t - self.grasp_since > 0.25:
                    self.s.events['bilateral_grasp_s'] = t
                    self.held_offset = p - np.asarray(o['tool_position'])
                    self.transition('lift_from_load', [p[0], p[1], 1.06])
            elif self.phase == 'lift_from_load':
                if self.move(o, 0.04):
                    self.transition('above_plating', [DIP_XY[0], DIP_XY[1], 1.06])
            elif self.phase == 'above_plating':
                if self.move(o):
                    # Held part centre DIP_CLEARANCE_M above its rest height on the bath floor.
                    self.transition('descend_plating', [DIP_XY[0], DIP_XY[1], DIP_PART_Z - self.held_offset[2]])
            elif self.phase == 'descend_plating':
                if self.move(o, 0.035):
                    if not o['submerged']:
                        raise RuntimeError('Reached dip depth without full immersion')
                    self.transition('wait_recipe')
            elif self.phase == 'wait_recipe':
                self.move(o, 0.025)
                departure = self.s.resolved['process_max_s'] + 1 if self.mode == 'late' else 0 if self.mode == 'early' else self.s.resolved['process_min_s'] + 1
                if o['elapsed_s'] is not None and o['elapsed_s'] >= departure:
                    self.transition('lift_from_plating', [DIP_XY[0], DIP_XY[1], 1.06])
            elif self.phase == 'lift_from_plating':
                if self.move(o, 0.04):
                    self.transition('bridge', [0, 0.43, 1.06])
            elif self.phase == 'bridge':
                if self.move(o):
                    self.transition('above_rinse', [0.34, 0.43, 1.06])
            elif self.phase == 'above_rinse':
                if self.move(o):
                    self.transition('descend_rinse', [0.34, 0.43, 0.72])
            elif self.phase == 'descend_rinse':
                reached = self.move(o, 0.035)
                if o['rinse_floor'] and o['rinse_inside']:
                    self.s.events['rinse_floor_contact_s'] = t
                    self.transition('open_gripper')
                elif reached:
                    raise RuntimeError('Reached rinse descent target without physical support')
            elif self.phase == 'open_gripper':
                self.move(o, 0.025)
                self.travel = min(0.04, self.travel + 0.02 / CONTROL_HZ)
                if self.travel >= 0.04 and age > 2:
                    self.transition('withdraw', [0.34, 0.43, 1.06])
            elif self.phase == 'withdraw':
                if self.move(o) and self.s.success:
                    self.complete = True
                    self.transition('complete')
            elif self.phase == 'complete':
                self.move(o)
            self.s.dual.jaw_travel('right', self.travel)
            if self.phase in CARRY_PHASES:
                actual = p - np.asarray(o['tool_position'])
                if np.linalg.norm(actual - self.held_offset) > 0.012:
                    raise RuntimeError('Lost physical grasp during transfer')
            if age > 22 and self.phase not in ('complete', 'wait_recipe'):
                raise RuntimeError('Controller phase timed out: ' + self.phase)
        except Exception as exc:
            self.failure = str(exc)

    def result(self):
        s = self.s
        elapsed = None if s.final_exit is None else s.final_exit - s.process_start
        temporal = elapsed is not None and s.resolved['process_min_s'] <= elapsed <= s.resolved['process_max_s']
        return dict(task_success=bool(self.complete and s.success and temporal and (not self.failure)), physical_transfer_completed=bool(self.complete and s.success), temporal_success=bool(temporal), plating_duration_s=elapsed, whole_part_exit_event='Last source-mesh point clears liquid level+1mm', phase=self.phase, failure=self.failure, events=s.events, mode=self.mode, initial_posture=self.initial_posture, recipe=[s.resolved['process_min_s'], s.resolved['process_max_s']], task_credit_eligible=self.mode == 'nominal', limitations=s.asset_record['limitations'])
