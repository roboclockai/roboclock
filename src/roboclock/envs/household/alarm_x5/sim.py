"""Alarm-clock promo scene, physics episode: free-standing alarm clock, sprung stop button, ringing cues and the oracle.

The clock is a 0.35 kg rigid body standing on its two front feet and the bottom of its case (primitive colliders
for the case, feet, bells and handle; nothing holds it to the counter). The stop button is a 5 g rigid body on a
sprung prismatic joint in the clock's handle, 6 mm of travel. The task model starts the alarm at its hidden time
and switches it off for good once the measured travel passes the catch (4 mm); from then on the catch holds the
button down (``appliance_step``: a force on the button, not a pose write).

Ringing is shown, not heard (the policy has no audio): while the alarm rings the hammer swings between the bells,
the bells shiver and the clock's visual twists and hops on the counter, every cue update (25 Hz). These move the
rendered meshes only; the colliders, and so the button the robot presses, stay with the rigid body. The second
hand ticks all the time.

Robot (privileged oracle, contact-driven, no object pose writes): the right arm waits at the ready pose with its
jaws closed until the alarm rings, moves the fingertips over the button with the tool pitched 60 deg down, presses
straight down until the measured travel reaches the stop, lifts off and goes back.
"""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.alarm_x5.scene import AXIS_Y, BUTTON_LOCAL, CAP_H, CLOCK_XY, TRAVEL
from roboclock.envs.household.x5.scene_lib import READY_Q
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
CLOCK = '/World/Objects/AlarmClock'
BUTTON = '/World/Objects/AlarmButton'
SOURCE = CLOCK+'/Source'
FILTERS = ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN', '/World/Arms/Right/link6',
           '/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Left/link6']
CLOCK_KG, BUTTON_KG = .35, .005
SPRING_N_M, LATCH_N = 300., 3.  # 1.8 N at the stop; the catch pulls harder than the spring pushes back
# Visual parts that move while the alarm rings: pivot in the clock frame and swing (deg) at the two extremes.
# The hammer head has 2 mm to the left bell and 6 mm to the right one; each bell shivers about the foot of its post.
SWING = {'Hammer': ((-.002, -.0045, .1287), (-8., 20.)), 'BellL': ((-.0355, -.006, .1187), (4., -4.)),
         'BellR': ((.0355, -.006, .1187), (4., -4.))}
RATTLE_DEG, HOP = 5., .002  # the whole clock about its vertical axis, and off the counter
DIAL = (0., -.0241, .0646)  # second hand pivot
G = {}


def press_tool(az):
    """Tool pitched 60 deg down at azimuth az (deg), jaws side by side: the closed fingertips are the lowest point."""
    e, a = np.radians(-60.), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([-np.sin(a), np.cos(a), 0.])  # jaw axis, horizontal
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def _pivot_ops(prim, pivot, axis, name):
    xf = UsdGeom.Xformable(prim)
    xf.AddTranslateOp(opSuffix=name+'Pivot').Set(Gf.Vec3d(*pivot))
    op = (xf.AddRotateYOp if axis == 'Y' else xf.AddRotateZOp)(opSuffix=name)
    op.Set(0.)
    xf.AddTranslateOp(opSuffix=name+'Unpivot').Set(Gf.Vec3d(*(-np.array(pivot)).tolist()))


def build(stage, mat):
    s = stage
    clock, button = s.GetPrimAtPath(CLOCK), s.GetPrimAtPath(BUTTON)
    P.body(clock, CLOCK_KG, com=(0., -.008, .062))
    # colliders in the clock frame; the visual meshes (which twist while it rings) carry none
    P.xform_parents(s, CLOCK+'/Collision/Case')
    case = UsdGeom.Cylinder.Define(s, CLOCK+'/Collision/Case')  # 0.6 mm clear of the counter: it stands on the three rests
    case.CreateAxisAttr('Y')
    case.CreateRadiusAttr(.064)
    case.CreateHeightAttr(.062)
    case.AddTranslateOp().Set(Gf.Vec3d(0., -.007, .0646))
    case.CreateVisibilityAttr('invisible')
    P.collision(case.GetPrim(), mat)
    for name, pos, size in [('FootL', (-.0378, -.0228, .006), (.012, .010, .012)), ('FootR', (.0378, -.0228, .006), (.012, .010, .012)),
                            ('Rest', (0., .012, .004), (.030, .016, .008)),
                            ('BellL', (-.0355, -.006, .1305), (.046, .048, .031)), ('BellR', (.0355, -.006, .1305), (.046, .048, .031)),
                            ('Handle', (0., AXIS_Y, .168), (.092, .009, .010))]:
        P.box(s, f'{CLOCK}/Collision/{name}', pos, size, mat)
    P.body(button, BUTTON_KG)
    P.collision(s.GetPrimAtPath(BUTTON+'/Cap'), mat)
    joint = UsdPhysics.PrismaticJoint.Define(s, '/World/Joints/AlarmButton')
    joint.CreateBody0Rel().SetTargets([clock.GetPath()])
    joint.CreateBody1Rel().SetTargets([button.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*BUTTON_LOCAL))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(-TRAVEL)
    joint.CreateUpperLimitAttr(0.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'linear')
    drive.CreateTargetPositionAttr(0.)
    drive.CreateStiffnessAttr(SPRING_N_M)
    drive.CreateDampingAttr(3.)
    # ringing cues: ops on the referenced visual parts
    source = UsdGeom.Xformable(s.GetPrimAtPath(SOURCE))
    source.AddTranslateOp(opSuffix='hop').Set(Gf.Vec3d(0, 0, 0))
    _pivot_ops(source.GetPrim(), (0., AXIS_Y, 0.), 'Z', 'rattle')
    for name, (pivot, _) in SWING.items():
        _pivot_ops(s.GetPrimAtPath(f'{SOURCE}/{name}'), pivot, 'Y', 'ring')
    _pivot_ops(s.GetPrimAtPath(SOURCE+'/SecondHand'), DIAL, 'Y', 'tick')
    return dict(clock=f'{CLOCK_KG} kg rigid body, free-standing on two feet and the case bottom; cylinder and box colliders',
                button=f'{BUTTON_KG*1000:g} g rigid body on a prismatic joint in the handle, {TRAVEL*1000:g} mm travel, '
                       f'spring {SPRING_N_M:g} N/m; latched down by a {LATCH_N:g} N catch force once the alarm is off',
                process='alarm state (task model); hammer, bells and the clock visual shake while it rings (appearance only)')


def _op(prim, suffix):
    return [op for op in UsdGeom.Xformable(prim).GetOrderedXformOps() if op.GetOpName().endswith(':'+suffix)][0]


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.phase = {'Left': 'done', 'Right': 'wait'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.failures, self.q_plan, self.hold = [], {}, {}
        self.travel, self.touch, self.t = 0., 0., 0.
        self.top = None  # measured centre of the button's top face (world)

    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        offset = self.task.spec.offset('clock')  # the seed's layout, before the reset: the clock slides and turns
        for path in (CLOCK, BUTTON):
            P.shift(stage.GetPrimAtPath(path), offset, CLOCK_XY)
        self.clock = world.scene.add(RigidPrim(CLOCK, name='clock', reset_xform_properties=False, track_contact_forces=True,
                                               contact_filter_prim_paths_expr=[FILTERS], max_contact_count=1024))
        self.button = world.scene.add(RigidPrim(BUTTON, name='button', reset_xform_properties=False, track_contact_forces=True,
                                                contact_filter_prim_paths_expr=[FILTERS], max_contact_count=256))
        source = stage.GetPrimAtPath(SOURCE)
        self.hop, self.rattle = _op(source, 'hop'), _op(source, 'rattle')
        self.swing = [(_op(stage.GetPrimAtPath(f'{SOURCE}/{name}'), 'ring'), angles) for name, (_, angles) in SWING.items()]
        self.tick = _op(stage.GetPrimAtPath(SOURCE+'/SecondHand'), 'tick')

    def appliance_step(self, stage):
        if self.task.cues()['silenced']:  # the catch: holds the plunger down against its spring
            self.button.apply_forces(np.array([[0., 0., -LATCH_N]]), is_global=True)

    def update_cues(self):
        beat = int(round(self.t/.04))
        side = (1 if beat % 2 else -1) if self.task.cues()['ringing'] else 0
        self.rattle.Set(float(side*RATTLE_DEG))
        self.hop.Set(Gf.Vec3d(0., 0., HOP if side > 0 else 0.))
        for op, (low, high) in self.swing:
            op.Set(float(0. if not side else high if side > 0 else low))
        self.tick.Set(float(6.*np.floor(self.t)))  # one step a second, clockwise seen from the front

    def measure(self, t, dt):
        self.t = t
        cp, cq = self.clock.get_world_poses()
        bp, _ = self.button.get_world_poses()
        cp, cq, bp = np.asarray(cp)[0], np.asarray(cq)[0], np.asarray(bp)[0]
        rc = R.from_quat([cq[1], cq[2], cq[3], cq[0]])
        if 'clock_p0' not in G:
            G.update(clock_p0=cp.copy())
        self.travel = float(max(0., BUTTON_LOCAL[2]-rc.inv().apply(bp-cp)[2])*1000)
        self.top = bp+rc.apply([0., 0., CAP_H])
        fb = np.linalg.norm(np.asarray(self.button.get_contact_force_matrix(dt=dt))[0], axis=-1)
        fc = np.linalg.norm(np.asarray(self.clock.get_contact_force_matrix(dt=dt))[0], axis=-1)
        self.touch = float(fb.max())
        tilt = float(np.degrees(np.arccos(np.clip(rc.apply([0, 0, 1])[2], -1., 1.))))
        moved = float(np.linalg.norm((cp-G['clock_p0'])[:2]))
        self.row = dict(button_travel_mm=self.travel, button_contact_N=fb.tolist(), clock_contact_N=fc.tolist(),
                        clock_tilt_deg=tilt, clock_displacement_m=moved, clock_xyz=cp.tolist())
        return dict(button_travel_mm=self.travel, hand_on_button=bool(self.touch > .2), clock_tilt_deg=tilt,
                    clock_displacement_m=moved)

    # ---------------------------------------------------------------- control
    def transition(self, side, phase, t):
        self.phase[side], self.phase_time[side] = phase, t
        self.event('phase', t, side=side, phase=phase)

    @property
    def done(self):
        return all(v == 'done' for v in self.phase.values())

    def _plan(self, servo, pos, rot, seed):
        q, pe, re = servo.kin.solve(pos, rot, seed)
        if pe > .004 or re > .05:
            raise RuntimeError(f'press pose unreachable {np.round(pos, 3)} ({pe:.4f} m, {re:.3f} rad)')
        if float(np.min(np.minimum(q-LOW, HIGH-q))) < .05:
            raise RuntimeError('press pose too close to a joint limit')
        return q

    def command(self, t):
        self.servos['Left'].move_joints(np.array(READY_Q), rate=1.)  # not needed: stays at its ready pose
        self.servos['Left'].grip(True)
        servo = self.servos['Right']
        phase, elapsed = self.phase['Right'], t-self.phase_time['Right']
        tcp, rr = servo.pose()
        servo.grip(True)  # closed jaws: the two fingertips together are the pressing tip
        if phase == 'wait':
            servo.move_joints(np.array(READY_Q), rate=1.2)
            if self.task.cues()['ringing']:
                base = servo.kin.base[:3, 3]
                top = self.top.copy()
                rot = press_tool(np.degrees(np.arctan2(top[1]-base[1], top[0]-base[0])))
                self.hold.update(top=top, rot=rot, pre=top+[0, 0, .03])
                self.q_plan['pre'] = self._plan(servo, self.hold['pre'], rot, servo.joints())
                self.transition('Right', 'approach', t)
            return
        top, rot, pre = self.hold['top'], self.hold['rot'], self.hold['pre']
        if phase == 'approach':  # joint-space move to 3 cm above the button
            servo.move_joints(self.q_plan['pre'], rate=2.)
            if np.linalg.norm(tcp-pre) < .006 and (rot*rr.inv()).magnitude() < .08:
                servo.reset_target()
                self.transition('Right', 'press', t)
            return
        if phase == 'press':  # straight down until the plunger is at its stop
            goal = np.array([top[0], top[1], top[2]-TRAVEL-.004])
            servo.move(goal, rot, speed=.06, turn_rate=.8)
            if self.travel > (TRAVEL-.0005)*1000:
                self.transition('Right', 'release', t)
        elif phase == 'release':  # let go: the catch holds the button down
            servo.move(pre, rot, speed=.12, turn_rate=.8)
            if tcp[2] > pre[2]-.008:
                self.transition('Right', 'retreat', t)
        elif phase in ('retreat', 'done'):
            q = np.array(READY_Q)
            servo.move_joints(q, rate=1.2)
            if phase == 'retreat' and np.abs(servo.joints()-q).max() < .02:
                self.transition('Right', 'done', t)
            return
        if elapsed > 10. and phase != 'done':
            self.failures.append(f'Right:phase_timeout:{phase}')

    def trace(self):
        return dict(self.row, phase=dict(self.phase), ringing=self.task.cues()['ringing'])
