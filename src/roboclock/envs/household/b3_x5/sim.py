"""B3 physics episode: a randomly timed pot fire, native burner knob, cues and oracle.

The pot body is a static triangle-mesh collider on its burner. The centre burner knob is a rigid body on a
damped revolute joint about z, from 90 deg (burner on, as placed) to its OFF stop 90 deg clockwise. The
task model integrates the flame intensity from the measured knob angle (it grows while the burner is on and
dies within a few seconds once it is off); the flame cards follow its cue.

Robot (privileged oracle, contact-driven, no object pose writes): waits at the ready pose for the flame
cue, then the right arm pinches the centre knob from above, turns it clockwise to OFF in one continuous
wrist turn, releases and retreats. The left arm stays at its ready pose.
"""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
POT = '/World/Objects/Pot/Source/Pot057_Body'
KNOB = '/World/Objects/Cooktop/Source/Stovetop012_knob_center'
KNOB_AXIS = np.array([.3075, -.1399, .8789])
JAWS = {'Left': ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN'], 'Right': ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN']}
# One wrist turn from jaw azimuth -135 deg: at -133, -127 and -120 the hand stalled 1.3-2.6 cm above the pinch, next
# to the pot's front handle (it reaches y = -.076, x .24-.36). Joint 6 stops at 2 rad, so the turn
# ends 2.5 deg short of the knob's stop, well inside OFF (10 deg).
KNOB_JAW0, KNOB_TURN = -135., 87.5
FLAME_BASE_Z, FLAME_H = .9125, .13  # flame cards: base on the oil, 13 cm tall at the initial intensity
G = {}
# Moved together by the seed's layout (the knob is part of the cooktop asset; the knob joint is anchored in the world)
COOKTOP_PARTS = ['/World/Objects/Cooktop', '/World/Objects/Pot', '/World/Objects/OilSurface',
                 '/World/Fixtures/CooktopGrates', '/World/Fixtures/CooktopFront']+[f'/World/Objects/Flame{i}' for i in range(3)]


def down(jaw_deg):
    """Tool pointing straight down with the jaw axis at azimuth jaw_deg."""
    a = np.radians(jaw_deg)
    x, y = np.array([0., 0., -1.]), np.array([np.cos(a), np.sin(a), 0.])
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def build(stage, mat):
    s = stage
    cache = UsdGeom.XformCache()
    P.collision(s.GetPrimAtPath(POT), mat, approximation='none')
    # cooktop: grates under the pots (top 0.9034) behind the knob strip, glass in front (0.8646)
    P.box(s, '/World/Fixtures/CooktopGrates', [.30, .153, .9034-.02], [.83, .446, .04], mat)  # front edge 7 cm behind the knobs
    P.box(s, '/World/Fixtures/CooktopFront', [.30, -.138, .8646-.02], [.83, .076, .04], mat)  # 4 mm under the jointed knobs
    # centre knob: rigid body on a damped revolute joint about its axis; 0 = as placed (on), -90 = OFF stop
    knob = s.GetPrimAtPath(KNOB)
    k2w = np.array(cache.GetLocalToWorldTransform(knob))
    P.collision(knob, mat)
    P.body(knob, .03)
    joint = UsdPhysics.RevoluteJoint.Define(s, '/World/Joints/CentreKnob')
    joint.CreateBody1Rel().SetTargets([knob.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, KNOB_AXIS)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float, (np.r_[KNOB_AXIS, 1.]@np.linalg.inv(k2w))[:3])))
    joint.CreateAxisAttr('Z')
    joint.CreateLowerLimitAttr(-90.)
    joint.CreateUpperLimitAttr(0.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(.002)  # valve friction (N m s / deg)
    # flame cards scale about the oil centre
    for i in range(3):
        xf = UsdGeom.Xformable(s.GetPrimAtPath(f'/World/Objects/Flame{i}'))
        base = Gf.Vec3d(.30, .10, FLAME_BASE_Z)
        xf.AddTranslateOp(opSuffix='pivot').Set(base)
        xf.AddScaleOp(opSuffix='flame').Set(Gf.Vec3f(1, 1, 1))
        xf.AddTranslateOp(opSuffix='unpivot').Set(-base)
    return dict(pot='static triangle-mesh collider on the centre burner (not displaced in this episode model)',
                knob='native centre knob, 30 g rigid body on a damped revolute joint (z); 90 deg = on as placed, OFF stop 90 deg clockwise',
                cooktop='box colliders for the grates (0.9034 m) and the glass strip in front (0.8646 m, 4 mm under the jointed knobs)',
                process='symbolic flame intensity (task model); flame cards follow the cue')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t, self.idle = [], 0., False
        self.phase = {'Left': 'done', 'Right': 'wait'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.jplan, self.jaw_travel, self.hold = {}, {}, {}
        self.knob_deg = 90.
        dx, dy, _ = task.spec.offset('cooktop')
        self.knob_axis = KNOB_AXIS+[dx, dy, 0.]  # the seed's layout

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        d = self.knob_axis-KNOB_AXIS  # the seed's layout, before the reset: the cooktop slides on the counter
        for path in COOKTOP_PARTS:
            P.shift(stage.GetPrimAtPath(path), (d[0], d[1], 0.), (0., 0.))
        UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Joints/CentreKnob')).GetLocalPos0Attr().Set(
            Gf.Vec3f(*map(float, self.knob_axis)))
        self.knob = world.scene.add(RigidPrim(KNOB, name='knob', reset_xform_properties=False, track_contact_forces=True,
                                              contact_filter_prim_paths_expr=[JAWS['Right']+JAWS['Left']], max_contact_count=256))
        self.flames = [[op for op in UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/Objects/Flame{i}')).GetOrderedXformOps()
                        if op.GetOpName().endswith('flame')][0] for i in range(3)]
        self.flame_vis = [UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/Objects/Flame{i}')) for i in range(3)]

    def update_cues(self):
        f = self.task.cues()['flame_intensity']
        h = float(np.clip(f/self.task.spec.f0, 0., 2.2))
        for op, vis in zip(self.flames, self.flame_vis):
            vis.GetVisibilityAttr().Set('inherited' if f > self.task.spec.out_threshold else 'invisible')
            op.Set(Gf.Vec3f(float(np.clip(.6+.4*h, .4, 1.3)), float(np.clip(.6+.4*h, .4, 1.3)), float(max(h, .02))))

    def measure(self, t, dt):
        self.t = t
        fk = np.linalg.norm(np.asarray(self.knob.get_contact_force_matrix(dt=dt))[0], axis=-1)  # right P, N, left P, N
        self.jaw = {'Right': fk[:2], 'Left': fk[2:]}  # the oracle's own (right) hand reads its pair
        _, kq = self.knob.get_world_poses()
        kq = np.asarray(kq)[0]
        kr = R.from_quat([kq[1], kq[2], kq[3], kq[0]])
        if 'knob_r0' not in G:
            G['knob_r0'] = kr
        rel = (kr*G['knob_r0'].inv()).as_rotvec()
        self.knob_deg = float(90.+np.degrees(rel[2]))
        self.row = dict(knob_deg=self.knob_deg, knob_contact_N=fk.tolist())
        return dict(knob_angle_deg=self.knob_deg,
                    hand_on_knob=bool(fk.max() > .3),  # judged: any jaw of either arm
                    water_contact=False, pot_displacement=0., wrong_knob_turned=False)

    # ---------------------------------------------------------------- control
    @property
    def done(self):
        return self.phase['Left'] == 'done' and self.phase['Right'] == 'done'

    def transition(self, side, phase, t):
        self.phase[side], self.phase_time[side] = phase, t
        self.event('phase', t, side=side, phase=phase)

    def plan_joints(self, servo, pos, rot, seed):
        q, pe, re = servo.kin.solve(pos, rot, seed)
        if pe > .002 or re > .02:
            raise RuntimeError(f'pose unreachable {np.round(pos, 3)} ({pe:.4f} m, {re:.3f} rad)')
        if float(np.min(np.minimum(q-LOW, HIGH-q))) < .05:
            raise RuntimeError('pose too close to a joint limit')
        return q

    def jpose(self, side, pre, rot, up=.08):
        """Joint-space approach: turn the base tucked, unfold `up` above the pre-pose, descend. True when there."""
        servo = self.servos[side]
        if side not in self.jplan:
            ready = np.array(READY_Q)
            b = servo.kin.base[:3, 3]
            ready[0] = np.arctan2(pre[1]-b[1], pre[0]-b[0])-np.pi/2
            q_hi = self.plan_joints(servo, pre+[0, 0, up], rot, ready)
            q_pre = self.plan_joints(servo, pre, rot, q_hi)
            tuck = np.array(READY_Q)
            tuck[0] = q_hi[0]
            self.jplan[side] = [tuck, q_hi, q_pre]
        plan = self.jplan[side]
        if plan:
            servo.move_joints(plan[0], rate=1.6)
            if np.max(np.abs(servo.joints()-plan[0])) < .02:
                plan.pop(0)
        if not plan:
            servo.reset_target()
            return True
        return False

    def close(self, side, rot, pos, need):
        """Each jaw closes until it touches, then both clamp. True when both pads exceed `need` N."""
        servo = self.servos[side]
        fp, fn = self.jaw[side]
        tp, tn = self.jaw_travel.get(side, [0., 0.])
        if fp > .5 and fn > .5:
            tp, tn = min(P.JAW_TRAVEL+.02, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.02, tn+servo.squeeze_step)
        else:
            tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
            tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
        self.jaw_travel[side] = [tp, tn]
        servo.grip_each(tp, tn)
        servo.move(pos, rot, speed=.01)
        return fp > need and fn > need

    def command(self, t):
        self._left(t)
        self._right(t)
        for side in ('Left', 'Right'):
            ph = self.phase[side]
            if ph not in ('done', 'wait') and t-self.phase_time[side] > 20.:
                self.failures.append(f'{side}:phase_timeout:{ph}')

    def _left(self, t):  # not needed: stays at its ready pose
        self.servos['Left'].move_joints(np.array(READY_Q), rate=1.)
        self.servos['Left'].grip(True)

    def _right(self, t):
        servo = self.servos['Right']
        ph, el = self.phase['Right'], t-self.phase_time['Right']
        tcp, rr = servo.pose()
        pinch = self.knob_axis+[0, 0, .003]
        if ph == 'wait':
            servo.move_joints(np.array(READY_Q), rate=1.)
            servo.grip(True)
            if self.task.cues()['flame_intensity'] > self.task.spec.out_threshold:
                self.transition('Right', 'approach', t)
        elif ph == 'approach':
            servo.grip(False)
            if self.jpose('Right', pinch+[0, 0, .06], down(KNOB_JAW0), up=.04):
                self.transition('Right', 'descend', t)
        elif ph == 'descend':
            servo.move(pinch, down(KNOB_JAW0), speed=.04, turn_rate=.8)
            servo.grip(False)
            if np.linalg.norm(tcp-pinch) < .002 and (down(KNOB_JAW0)*rr.inv()).magnitude() < .03:
                self.jaw_travel['Right'] = [0., 0.]
                self.transition('Right', 'close', t)
        elif ph == 'close':
            if self.close('Right', down(KNOB_JAW0), pinch, 12.) and el > .3:
                self.event('grasp', t, side='Right', jaw_N=self.jaw['Right'].tolist())
                q_turn = servo.joints()+np.radians([0, 0, 0, 0, 0, KNOB_TURN])
                q_turn[5] = min(q_turn[5], HIGH[5]-.01)  # a moved cooktop starts joint 6 up to ~3 deg further on
                self.hold['q_turn'] = q_turn
                self.transition('Right', 'turn', t)
        elif ph == 'turn':  # joint 6 alone turns the pinch clockwise about the knob axis
            # (tool straight down: joints 4 and 6 are aligned, a wrist singularity for Cartesian steps;
            # joint 6 turns about the tool axis through the tool point, +q6 = clockwise seen from above)
            servo.move_joints(self.hold['q_turn'], rate=1.)
            servo.grip_each(*self.jaw_travel['Right'])
            if np.max(np.abs(servo.joints()-self.hold['q_turn'])) < .02 and el > .3:
                self.transition('Right', 'release', t)
        elif ph == 'release':
            servo.grip(False)
            servo.move_joints(servo.joints(), rate=.1)
            if el > .5:
                self.transition('Right', 'retreat', t)
        elif ph == 'retreat':
            goal = pinch+[0, 0, .08]
            servo.move(goal, rr, speed=.10, turn_rate=.5)
            servo.grip(False)
            if np.linalg.norm(tcp-goal) < .008:
                self.transition('Right', 'home', t)
        elif ph in ('home', 'done'):
            q = np.array(READY_Q)
            servo.move_joints(q, rate=1.2)
            servo.grip(True)
            if ph == 'home' and np.max(np.abs(servo.joints()-q)) < .03:
                self.transition('Right', 'done', t)

    def trace(self):
        return dict(self.row, phase=dict(self.phase), flame=self.task.cues()['flame_intensity'], burner_z=H)
