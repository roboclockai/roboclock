"""Final 45: native spring-button contact stops the motor test.

Reuses approved environment and original motor/button meshes. Fault warning
is a declared process cue; button, rotor and complete ARX arms use physics.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from roboclock.envs.rates import CONTROL_HZ

from . import rig
from .arx_cell_task import (
    ArmMotion,
    ContactBook,
    StatusDisplay,
    add_mounts,
    collide,
    remove_preview,
)
from .paths import asset_path, portable_json_loads

ROOT = Path(asset_path('/opt/industry/assets/v2/task45'))
BUTTON = '/World/MotorTest/Button'
# Only the right arm works; it keeps its recorded base and the left arm sits 0.60 m beside it
# (recorded: left [-0.96, -0.15, 0.92], 0.48 m apart).
MOUNTS = rig.mounts_keeping("right", [-0.48, -0.15, 0.92])

def resolve_motor(seed=0, *, layout=False, timing=False, physical=False):
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    d = dict(task_id=45, seed=seed, embodiment='dual_arx_x5', button_xy_offset_m=streams[0].uniform(-0.008, 0.008, 2).tolist() if layout else [0.0, 0.0], warning_onset_s=float(streams[1].uniform(14, 18)) if timing else 16.0, button_spring_N_m=float(streams[2].uniform(225, 275)) if physical else 250.0, response_deadline_s=3.0, trigger_depth_m=0.008, confirmation_dwell_s=0.2, randomization=dict(layout=layout, timing=timing, physical=physical), bounds=dict(button_xy_offset_m=[-0.008, 0.008], warning_onset_s=[14, 18], button_spring_N_m=[225, 275]), independence='Separate SeedSequence layout/event/physical streams', scoring='Native8mm button depression with real finger contact sustained0.2s within3s of visible anomaly; native motor brake and withdrawal required', assumptions='Declared10mm spring travel,5N spring cap; 15rad/s nominal drive; no calibrated failure/vibration model. Source motor/button geometry and source robot operating caps retained.', controller='Privileged-state bounded joint-drive oracle, not a trained policy')
    d['sha256'] = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
    return d

class MotorARXScene:
    title = 'MOTOR TEST / EMERGENCY STOP'
    detail = 'Press STOP within3s after warning'

    def __init__(self, manifest, output, resolved, mode):
        from isaacsim.core.prims import SingleArticulation
        from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

        from .dual_arx import DualARX
        from .remaining_environments import build_remaining_environment
        from .v2.geometry import body_envelope_points
        self.base = build_remaining_environment(45, output, resolution=(1280, 800))
        c = self.base.cell
        self.world, self.stage = (c.world, c.stage)
        stage = self.stage
        self.resolved, self.events = (resolved, {})
        self.asset_manifest = portable_json_loads((ROOT / 'asset-manifest.json').read_text())
        for p in ['/World/Rig/housing', '/World/Rig/rotor', '/World/Stop/Stop_cap', '/World/Stop/Stop_housing']:
            remove_preview(c, p)
        frame = UsdGeom.Xform.Define(stage, '/World/MotorTest/Frame').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(frame)
        UsdPhysics.MassAPI.Apply(frame).CreateMassAttr(20)
        anchor = UsdPhysics.FixedJoint.Define(stage, '/World/MotorTest/Anchor')
        anchor.CreateBody1Rel().SetTargets([frame.GetPath()])
        UsdPhysics.ArticulationRootAPI.Apply(anchor.GetPrim())
        prop = PhysxSchema.PhysxArticulationAPI.Apply(anchor.GetPrim())
        prop.CreateSolverPositionIterationCountAttr(64)
        prop.CreateSolverVelocityIterationCountAttr(16)
        prop.CreateEnabledSelfCollisionsAttr(True)
        mb = np.asarray(self.asset_manifest['prepared']['housing']['bounds_m'])
        motorp = np.array([0, 0.62, 0.94 - mb[0, 2]])
        mq = Gf.Quatf(0, Gf.Vec3f(0, 0, 1))
        for name, mass in [('housing', 5.0), ('rotor', 0.035)]:
            path = ROOT / 'prepared-v4' / (name + '.usd')
            assert hashlib.sha256(path.read_bytes()).hexdigest() == self.asset_manifest['prepared'][name]['usd_sha256']
            x = UsdGeom.Xform.Define(stage, '/World/MotorTest/' + name)
            x.GetPrim().GetReferences().AddReference(str(path))
            x.AddTranslateOp().Set(Gf.Vec3d(*motorp))
            x.AddOrientOp().Set(mq)
            UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
            UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        fix = UsdPhysics.FixedJoint.Define(stage, '/World/MotorTest/HousingMount')
        fix.CreateBody0Rel().SetTargets([frame.GetPath()])
        fix.CreateBody1Rel().SetTargets(['/World/MotorTest/housing'])
        fix.CreateLocalPos0Attr(Gf.Vec3f(*motorp))
        fix.CreateLocalRot0Attr(mq)
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/World/MotorTest/RotorJoint')
        joint.CreateBody0Rel().SetTargets(['/World/MotorTest/housing'])
        joint.CreateBody1Rel().SetTargets(['/World/MotorTest/rotor'])
        joint.CreateAxisAttr('X')
        joint.CreateCollisionEnabledAttr(False)
        self.rotor_drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
        self.rotor_drive.CreateTypeAttr('force')
        self.rotor_drive.CreateStiffnessAttr(0)
        self.rotor_drive.CreateDampingAttr(0.005)
        self.rotor_drive.CreateMaxForceAttr(0.02)
        self.rotor_drive.CreateTargetVelocityAttr(float(np.rad2deg(15)))
        self.button_position = np.array([-0.48, 0.3, 1.04])
        self.button_position[:2] += resolved['button_xy_offset_m']
        self.axis = np.array([0.0, 1.0, 0.0])
        bq = Gf.Quatf(float(np.sqrt(0.5)), Gf.Vec3f(0, 0, float(np.sqrt(0.5))))
        for name in ['cap', 'housing']:
            file = ROOT / 'button-prepared-v2' / (name + '.usd')
            assert hashlib.sha256(file.read_bytes()).hexdigest() == self.asset_manifest['button']['prepared'][name]['usd_sha256']
            x = UsdGeom.Xform.Define(stage, BUTTON if name == 'cap' else '/World/Stop/NativeHousing')
            x.GetPrim().GetReferences().AddReference(str(file))
            x.AddTranslateOp().Set(Gf.Vec3d(*self.button_position))
            x.AddOrientOp().Set(bq)
            if name == 'cap':
                UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
                UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(0.02)
                PhysxSchema.PhysxContactReportAPI.Apply(x.GetPrim()).CreateThresholdAttr(0)
        joint = UsdPhysics.PrismaticJoint.Define(stage, '/World/MotorTest/ButtonJoint')
        joint.CreateBody0Rel().SetTargets([frame.GetPath()])
        joint.CreateBody1Rel().SetTargets([BUTTON])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*self.button_position))
        joint.CreateLocalRot0Attr(bq)
        joint.CreateAxisAttr('X')
        joint.CreateLowerLimitAttr(0)
        joint.CreateUpperLimitAttr(0.01)
        spring = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
        spring.CreateTypeAttr('force')
        spring.CreateStiffnessAttr(resolved['button_spring_N_m'])
        spring.CreateDampingAttr(3)
        spring.CreateMaxForceAttr(5)
        spring.CreateTargetPositionAttr(0)
        self.machine = self.world.scene.add(SingleArticulation('/World/MotorTest', name='native_motor_test'))
        self.dual = DualARX(self.world, manifest, mounts=MOUNTS, gripper_velocity_cap=False)
        add_mounts(c, MOUNTS)
        self.static_colliders = collide(stage, ['/World/TestBench', '/World/Guard/', '/World/Rig/MotorMount', '/World/Stop/', '/World/ARXMounts/'])
        self.contacts = ContactBook(self.world, [BUTTON])
        self.display = StatusDisplay(c, 'HMI', {'NORMAL': 'MOTOR TEST / NORMAL', 'WARNING': 'FAULT WARNING / PRESS STOP <=3s', 'STOPPED': 'TEST STOPPED / SERVICE REQUIRED'})
        self.cue_state = 'NORMAL'
        self.braked = False
        self.physical_completed = False
        self.success = False
        self.trigger_since = None
        self.stop_since = None
        self.world.reset()
        self.dual.initialize()
        self.machine.initialize()
        self.button_id = self.machine.dof_names.index('ButtonJoint')
        self.rotor_id = self.machine.dof_names.index('RotorJoint')
        self.machine.set_joint_positions(np.zeros(2))
        self.machine.set_joint_velocities(np.zeros(2))
        from isaacsim.core.utils.types import ArticulationAction
        kp, kd = (np.asarray(v).reshape(-1).copy() for v in self.machine.get_articulation_controller().get_gains())
        kp[self.button_id], kd[self.button_id] = (resolved['button_spring_N_m'], 3)
        kp[self.rotor_id], kd[self.rotor_id] = (0, 0.005)
        self.machine.get_articulation_controller().set_gains(kps=kp, kds=kd)
        self.machine.apply_action(ArticulationAction(joint_positions=np.array([0.0]), joint_indices=np.array([self.button_id])))
        self.machine.apply_action(ArticulationAction(joint_velocities=np.array([15.0]), joint_indices=np.array([self.rotor_id])))
        self.dual.jaw_travel('right', 0)
        self.dual.robots['right'].set_joint_positions(self.dual.commands['right'])
        left = self.dual.commands['left']
        left[self.dual.names['left'].index('joint1')] = 1.8
        self.dual.robots['left'].set_joint_positions(left)
        self.finger_points = {name: body_envelope_points(stage, stage.GetPrimAtPath('/World/ARX/right/' + self.dual.manifest['relative_body_paths'][name])) for name in ['link7', 'link8']}
        for mapping in [self.contacts.interval, self.contacts.pairs, self.contacts.minimum, self.contacts.active, self.contacts.unexpected]:
            mapping.clear()
        # Head and wrist policy cameras and the recording overview (robodojo_rig spec).
        self.rig = rig.attach_cameras(self, MOUNTS, task=45)
        self.asset_manifest['native_scope'] = dict(button='Original cap/housing;10mm native spring travel; no pose writes after reset', rotor='Original exposed shaft/key; native limited-force velocity drive and contact-confirmed brake', static_colliders=self.static_colliders)

    def brake(self):
        if not self.braked:
            from isaacsim.core.utils.types import ArticulationAction
            kp, kd = (np.asarray(v).reshape(-1).copy() for v in self.machine.get_articulation_controller().get_gains())
            kp[self.rotor_id] = 0
            kd[self.rotor_id] = 0.003
            self.machine.get_articulation_controller().set_gains(kps=kp, kds=kd)
            self.machine.apply_action(ArticulationAction(joint_velocities=np.array([0.0]), joint_indices=np.array([self.rotor_id])))
            self.rotor_drive.GetTargetVelocityAttr().Set(0)
            self.rotor_drive.GetDampingAttr().Set(0.003)
            self.braked = True

    def observe(self):
        t = float(self.world.current_time)
        q = self.machine.get_joint_positions()
        v = self.machine.get_joint_velocities()
        depth = float(q[self.button_id])
        speed = float(v[self.rotor_id])
        forces = [self.contacts.force(BUTTON, f'/link{i}') for i in [7, 8]]
        if t >= self.resolved['warning_onset_s']:
            self.events.setdefault('warning_onset_s', t)
        if 'warning_onset_s' not in self.events and depth > 0.003:
            self.events.setdefault('premature_press_s', t)
        contact = depth >= self.resolved['trigger_depth_m'] and max(forces) > 0.1
        self.trigger_since = (self.trigger_since if self.trigger_since is not None else t) if contact else None
        if self.trigger_since is not None and t - self.trigger_since >= self.resolved['confirmation_dwell_s'] and ('button_confirmed_s' not in self.events):
            self.events['button_confirmed_s'] = t
            onset = self.events.get('warning_onset_s')
            self.events['response_s'] = t - onset if onset is not None else None
            self.success = onset is not None and t - onset <= self.resolved['response_deadline_s'] and ('premature_press_s' not in self.events)
            self.brake()
        if self.braked and abs(speed) < 0.05:
            self.stop_since = self.stop_since if self.stop_since is not None else t
            if t - self.stop_since > 0.5:
                self.events.setdefault('motor_stopped_s', t)
                self.physical_completed = True
        else:
            self.stop_since = None
        state = 'STOPPED' if self.physical_completed else 'WARNING' if 'warning_onset_s' in self.events else 'NORMAL'
        self.display.set(state)
        self.cue_state = state
        tool = self.dual.tool_world('right')[0]
        front = []
        for name, pts in self.finger_points.items():
            p, R = self.dual.body_world('right', name)
            front.extend((pts @ R.T + p - tool) @ self.axis)
        return dict(time_s=t, tool_position=tool.tolist(), button_depth_m=depth, rotor_speed_rad_s=speed, rotor_angle_rad=float(q[self.rotor_id]), finger_forces_N=forces, finger_front_m=float(max(front)), cue_state=state, stopped=self.physical_completed, unexpected_contacts=self.contacts.unexpected.copy())

    def final_valid(self, o):
        return self.physical_completed and abs(o['rotor_speed_rad_s']) < 0.05

class MotorARXController:
    # The oracle starts from the reset pose (tool pointing down) when the warning lights. The
    # recorded horizontal press needs joint4 to turn 3.0 rad from there (4.3 s at the arm's
    # joint rate); with the closed fingers pitched 60 deg below horizontal the largest joint
    # travel is 0.8 rad, and the finger tips still push the cap along its axis.
    PRESS_PITCH_DEG = 60.0
    # Where the joint-space reach ends and the straight press starts, in front of the cap.
    STANDOFF_M = 0.015

    def __init__(self, scene, mode):
        self.s, self.mode = (scene, mode)
        self.motion = ArmMotion(scene.dual, 'right')
        self.motion.orientation_feedback = True
        self.motion.position_gain = 1.2
        self.motion.R = Rotation.from_euler('x', -self.PRESS_PITCH_DEG, degrees=True).as_matrix() @ Rotation.from_euler('z', np.pi / 2).as_matrix()
        self.phase = 'settle'
        self.entered = 0.0
        self.complete = False
        self.failure = None
        self.press_goal = None
        self.reach_joints = self.reach_goal = None
        self.tip_offset = np.zeros(3)

    def transition(self, phase):
        self.phase = phase
        self.entered = float(self.s.world.current_time)
        self.s.events['phase_' + phase + '_s'] = self.entered
        print('PHASE', phase, self.entered, flush=True)

    def step(self, o):
        s = self.s
        t = o['time_s']
        if self.failure or self.mode in ['absent', 'passive']:
            return
        try:
            if o['unexpected_contacts']:
                raise RuntimeError('Unexpected native robot contact')
            if 'premature_press_s' in s.events:
                raise RuntimeError('Button pressed before warning')
            front = s.asset_manifest['button']['cap_front_offset_m']
            standby = s.button_position - self.tip_offset - s.axis * (front + o['finger_front_m'] + self.STANDOFF_M)
            if self.phase == 'settle' and t > 1:
                self.transition('standby')
            if self.phase == 'standby':
                # Hold the reset pose (no command is written) until the warning is in the
                # scene: ``warning_onset_s`` is set in the same observation that switches the
                # HMI to the red FAULT WARNING state, the first instant a camera could show it.
                # The oracle used to wait 3.5 cm in front of the button.
                onset = s.events.get('warning_onset_s')
                if onset is not None and (t - onset >= (3.5 if self.mode == 'late' else 0)):
                    self.transition('reach')
            if self.phase == 'reach':
                # Joint-space move from the reset pose to STANDOFF_M in front of the button, at
                # ArmMotion's joint command rate.
                d = s.dual
                side = 'right'
                ids = self.motion.ids
                if self.reach_joints is None:
                    bp, bR = d.body_world(side, 'base_link')
                    tool, tool_R = d.tool_world(side)
                    # The closed fingers as they will point at the button (``finger_front_m`` is
                    # measured in the current, still turned-away pose): how far their foremost
                    # points reach ahead of the tool, and where those points sit beside the
                    # button's axis, so that the tip and not the tool meets the cap's centre.
                    turn = self.motion.R @ tool_R.T
                    points = np.concatenate([
                        (s.finger_points[name] @ R.T + p - tool) @ turn.T
                        for name, (p, R) in ((name, d.body_world(side, name)) for name in ['link7', 'link8'])
                    ])
                    ahead = points @ s.axis
                    tip = points[ahead > ahead.max() - 0.002].mean(axis=0)
                    self.tip_offset = tip - s.axis * (tip @ s.axis)
                    self.reach_goal = s.button_position - self.tip_offset - s.axis * (front + ahead.max() + self.STANDOFF_M)
                    profile = d.manifest['profile']
                    self.reach_joints, error = self.motion.kin.solve(bR.T @ (self.reach_goal - bp), bR.T @ self.motion.R, dict(zip(d.names[side], d.commands[side])), profile['arm_lower_rad'], profile['arm_upper_rad'], restarts=8)
                    if error > 0.003:
                        raise RuntimeError('Standby unreachable')
                rate = self.motion.command_speed_rad_s / CONTROL_HZ
                d.commands[side][ids] += np.clip(self.reach_joints - d.commands[side][ids], -rate, rate)
                actual, R = d.tool_world(side)
                if np.linalg.norm(actual - self.reach_goal) < 0.008 and np.linalg.norm(Rotation.from_matrix(self.motion.R @ R.T).as_rotvec()) < 0.03:
                    self.motion.command = actual.copy()
                    self.motion.correction[:] = 0
                    self.press_goal = s.button_position - self.tip_offset + s.axis * (0.01 - front + 0.001 - o['finger_front_m'])
                    self.transition('press')
            elif self.phase == 'press':
                self.motion.move(self.press_goal, 0.085)
                if 'button_confirmed_s' in s.events:
                    self.transition('withdraw')
            elif self.phase == 'withdraw':
                if self.motion.move(standby, 0.075) and s.physical_completed:
                    self.complete = True
                    self.transition('complete')
            if t - self.entered > 25 and self.phase not in ['standby', 'complete']:
                raise RuntimeError('Phase timeout ' + self.phase)
            if self.phase == 'complete' and (not s.final_valid(o)):
                raise RuntimeError('Motor restarted after stop')
        except RuntimeError as exc:
            self.failure = str(exc)
            print('FAILURE', self.failure, flush=True)
