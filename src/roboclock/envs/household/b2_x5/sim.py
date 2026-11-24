"""B2 physics episode: one of four objects is let go on a sloped board and rolls toward the counter edge.

Board: a static wedge whose slope is set per difficulty before the episode starts. Each object rests against
a kinematic peg; at the hidden release time that peg sinks into the board (scripted disturbance, the only
scripted motion). From there on the object is moved by gravity, contact and friction alone. Angular damping
stands in for rolling resistance, which PhysX does not model.

Colliders: apple = sphere. Egg = three spheres on its long axis (a smooth ovoid with its widest circle in
the middle, so it rolls straight; a real egg would curve). Bottle and cup = cylinders, the cup with the
inertia of a thin wall.

Robot (privileged oracle, left arm): both arms hold READY until the measured speed of an object shows
which one is moving; the right arm then swings aside. The left hand meets the object in joint space,
tilted so the mouth of the open gripper faces the object: the upper jaw clears the object, the lower jaw
is in its path. The object runs into the lower jaw and rides up it, between the jaws. Each jaw then
closes until it touches, the hand lifts, holds.
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.b2_x5.scene import (
    BOARD_L,
    D,
    EDGE_Y,
    HEADING,
    OBJECTS,
    TOE,
    U,
    board_points,
    object_pose,
    on_board,
    peg_pose,
    slope_rotation,
)
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW
from roboclock.envs.household.x5.task_base import gripped, place

CONTROL = dict(left='ik', right='ik')
# The left pair first: the oracle's own hand. The right pair is read for the judged signals only
# (either arm may catch: the 2026-11-10 sweep caught with the right arm and was never "held").
JAWS = ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Right/jawP', '/World/Arms/Right/jawN']
NAMES = list(OBJECTS)
ROOT = {n: '/World/Objects/'+n.title().replace('_', '') for n in NAMES}
PEG_DROP, PEG_SECONDS = .016, .08   # the peg sinks this far (m) in this time (s)
ROLLING_DAMPING = .12               # angular damping (1/s), standing in for rolling resistance
CATCH_Y = -.315                     # the hand waits on this line, 6.5 cm short of the edge along y
TOOL_AZ, TOOL_EL = HEADING-180., -55.  # the mouth faces the incoming object, tilted down
MOVE_RATE = 3.                      # rad/s, joint-space intercept
TRACK_SPEED = .6                    # m/s, sliding along the waiting line
JAW_CATCH_SPEED = .12               # m/s per jaw, the upper jaw coming down behind the object
GRIP_N = 10.                        # N per jaw before lifting (well under the 30 N crush limit)
SETTLED = .1                        # m/s: slow enough to take the object
SEAT_SPEED = .14                    # m/s
CLEAR = .006                        # upper jaw tip above the object
RIGHT_ASIDE = np.array([-1.15, *READY_Q[1:]])
MOVING = .05                        # m/s: measured speed that shows an object has been let go
G = {}


def tool(az, el, roll=0.):
    """Tool frame: x along the tool, y (jaw axis) the up direction projected off x."""
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))*R.from_euler('x', roll, degrees=True)


def lane_catch(name, offset=(0., 0., 0.)):
    """Where the object's lane crosses the line the hand waits on (board moved by the seed's layout offset)."""
    toe = place(TOE+U*OBJECTS[name]['s'], offset, TOE)
    d = R.from_euler('z', offset[2], degrees=True).apply([*D, 0.])[:2]
    return toe+d*(toe[1]-CATCH_Y)/-d[1]


def _damped(prim, mass):
    P.body(prim, mass)
    api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    api.CreateAngularDampingAttr(ROLLING_DAMPING)
    api.CreateLinearDampingAttr(0.)
    PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr(True)


def _sphere(s, path, centre, radius, mat):
    c = UsdGeom.Sphere.Define(s, path)
    c.CreateRadiusAttr(radius)
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, centre))
    c.CreatePurposeAttr('guide')
    P.collision(c.GetPrim(), mat)


def build(stage, mat):
    s = stage
    P.collision(s.GetPrimAtPath('/World/Objects/Board'), mat)
    P.box(s, '/World/Fixtures/Floor', [.3, -.3, -.05], [8., 8., .1], mat)  # a dropped object lands instead of falling forever
    for name, o in OBJECTS.items():
        root = ROOT[name]
        prim = s.GetPrimAtPath(root)
        r, ln = o['d']/2, o['l']/2
        if name == 'apple':
            _sphere(s, root+'/Hit', 0., r, mat)
        elif name == 'egg':
            for i, (z, rr) in enumerate(((0., r), (ln-r*.72, r*.72), (-ln+r*.80, r*.80))):
                _sphere(s, f'{root}/Hit{i}', z, rr, mat)
        else:
            c = UsdGeom.Cylinder.Define(s, root+'/Hit')
            c.CreateRadiusAttr(r)
            c.CreateHeightAttr(2*ln)
            c.CreateAxisAttr('Z')
            c.CreatePurposeAttr('guide')
            P.collision(c.GetPrim(), mat)
        _damped(prim, o['m'])
        if name == 'cup':  # thin wall: most of the mass sits at the full radius
            UsdPhysics.MassAPI(prim).CreateDiagonalInertiaAttr(Gf.Vec3f(float(o['m']*(r*r/2+o['l']**2/12)), float(o['m']*(r*r/2+o['l']**2/12)), float(o['m']*r*r*.9)))
        peg = s.GetPrimAtPath(root+'Peg')
        P.collision(s.GetPrimAtPath(root+'Peg/Block'), mat)
        P.body(peg, .02, kinematic=True)
    return dict(board=f'static wedge, {BOARD_L*100:.0f} cm long, slope set per difficulty',
                objects={k: f'{v["d"]*1000:.0f} mm across, {v["l"]*1000:.0f} mm long, {v["m"]*1000:.0f} g' for k, v in OBJECTS.items()},
                colliders='apple sphere; egg three spheres (rolls straight); bottle and cup cylinders',
                rolling_resistance=f'angular damping {ROLLING_DAMPING} /s',
                release='kinematic peg sinks into the board at the hidden release time (the only scripted motion)')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t, self.idle = [], 0., False
        self.phase, self.phase_time = 'wait', 0.
        self.spec = task.spec
        self.deg = self.spec.board_deg
        self.name = self.spec.target_name
        self.seen = None      # the object measured to be moving
        self.jaw_travel = [0., 0.]
        self.peak = 0.
        self.vmax = 0.
        self.done_flag = False
        self.peg_t = None
        self.offset = self.spec.offset('board')  # the seed's layout: board, objects and pegs moved as one
        self.turn = R.from_euler('z', self.offset[2], degrees=True)
        self.heading = self.turn.apply([*D, 0.])[:2]  # rolling direction

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        UsdGeom.Mesh(stage.GetPrimAtPath('/World/Objects/Board')).GetPointsAttr().Set(
            place(board_points(self.deg), self.offset, TOE).tolist())
        for name in NAMES:
            for path, (p, r) in ((ROOT[name], object_pose(name, self.deg)), (ROOT[name]+'Peg', peg_pose(name, self.deg))):
                P.pose(stage.GetPrimAtPath(path), place(p, self.offset, TOE), self.turn*r)
        self.obj = {n: world.scene.add(RigidPrim(ROOT[n], name=n, reset_xform_properties=False, track_contact_forces=True,
                                                 contact_filter_prim_paths_expr=[JAWS], max_contact_count=64)) for n in NAMES}
        self.peg = world.scene.add(RigidPrim(ROOT[self.name]+'Peg', name='peg', reset_xform_properties=False))
        self.peg0 = place(peg_pose(self.name, self.deg)[0], self.offset, TOE)
        self.peg_axis = (self.turn*slope_rotation(self.deg)).apply([0, 0, 1])

    def appliance_step(self, stage):
        if self.t < self.spec.release_time:
            return
        if self.peg_t is None:
            self.peg_t = self.t
            self.event('peg_released', self.t, object=self.name)
        u = min(1., (self.t-self.peg_t)/PEG_SECONDS)
        if u < 1. or not getattr(self, 'peg_down', False):
            self.peg.set_world_poses(positions=(self.peg0-self.peg_axis*PEG_DROP*u)[None])
            self.peg_down = u >= 1.

    def measure(self, t, dt):
        self.t = t
        rows = {}
        for n in NAMES:
            p, _ = self.obj[n].get_world_poses()
            rows[n] = dict(p=np.asarray(p)[0].copy(), v=np.asarray(self.obj[n].get_linear_velocities())[0].copy())
        if 'p0' not in G or G.get('task') is not self.task:
            G.update(task=self.task, p0={n: rows[n]['p'].copy() for n in NAMES})
        self.state = rows
        me = rows[self.name]
        f = np.linalg.norm(np.asarray(self.obj[self.name].get_contact_force_matrix(dt=dt))[0], axis=-1)  # left P, N, right P, N
        self.jaw = f[:2]  # the oracle's own (left) hand: upper jaw (P), lower jaw (N)
        self.peak = max(self.peak, float(f.max()))  # judged: any jaw of either arm
        r = OBJECTS[self.name]['d']/2
        edge = float((me['p'][1]-EDGE_Y)/-self.heading[1])
        clear = me['p'][2] > H+r+.015
        held = bool(gripped(f, 1.) and clear)  # judged: both jaws of one arm, whichever arm
        others = max(float(np.linalg.norm(rows[n]['p']-G['p0'][n])) for n in NAMES if n != self.name)
        self.row = dict(objects={n: rows[n]['p'].tolist() for n in NAMES}, target=self.name, target_pos=me['p'].tolist(),
                        target_vel=me['v'].tolist(), jaw_N=self.jaw.tolist(), right_jaw_N=f[2:].tolist(), peak_N=self.peak, peg_t=self.peg_t, edge_m=edge,
                        held=held)
        return dict(released=self.peg_t is not None, object_speed=float(np.linalg.norm(me['v'])), edge_distance=edge,
                    object_fell=bool(me['p'][2] < H-.02), object_held=held, peak_contact_n=self.peak, others_moved_m=others)

    # ---------------------------------------------------------------- control
    @property
    def done(self):
        return self.done_flag

    def transition(self, phase, t, **kw):
        self.phase, self.phase_time = phase, t
        self.event('phase', t, side='Left', phase=phase, **kw)

    def plan(self, servo, name):
        d = OBJECTS[name]['d']
        rot = tool(TOOL_AZ+self.offset[2], TOOL_EL)  # the mouth faces the (turned) rolling direction
        x, y = rot.apply([1, 0, 0]), rot.apply([0, 1, 0])
        tip = lambda sign: sign*.03015*y[2]+.002*x[2]
        z = H+d+CLEAR-tip(1)                # upper jaw tip just above the object
        xy = lane_catch(name, self.offset)
        pos = np.array([*xy, z])
        seeds = [np.array(q) for q in ([-.9, 1.6, 1.4, -1., 1., 0.], [-1.2, 1.9, 1.6, -1.2, 1.1, .3], [-.6, 1.2, .9, -.6, 1.2, .8])]
        q, pe, re = servo.kin.solve(pos, rot, np.array(READY_Q), seeds)
        if pe > .002 or re > .02:
            raise RuntimeError(f'catch pose unreachable {np.round(pos, 3)} ({pe:.4f} m, {re:.3f} rad)')
        margin = float(np.min(np.minimum(q-LOW, HIGH-q)))
        if margin < .05:
            raise RuntimeError('catch pose too close to a joint limit')
        return pos, rot, q

    def command(self, t):
        servo = self.servos['Left']
        # The right arm keeps its start pose until an object is seen moving; only then does it swing
        # aside, out of the camera's line to the board.
        self.servos['Right'].move_joints(np.array(READY_Q) if self.phase == 'wait' else RIGHT_ASIDE, rate=1.)
        self.servos['Right'].grip(False, opening=.03)
        el = t-self.phase_time
        if self.phase == 'wait':
            servo.move_joints(np.array(READY_Q), rate=1.)
            servo.grip(False)
            moving = [n for n in NAMES if np.linalg.norm(self.state[n]['v']) > MOVING]
            if moving:
                self.seen = max(moving, key=lambda n: np.linalg.norm(self.state[n]['v']))
                self.goal, self.rot, self.q = self.plan(servo, self.seen)
                self.aim = float(self.goal[0])
                self.transition('intercept', t, object=self.seen, speed=float(np.linalg.norm(self.state[self.seen]['v'])))
        elif self.phase == 'intercept':
            servo.move_joints(self.q, rate=MOVE_RATE)
            servo.grip(False)
            if np.max(np.abs(servo.joints()-self.q)) < .03:
                servo.reset_target()
                self.transition('block', t)
        elif self.phase == 'block':  # slide along the waiting line to where the measured motion crosses it
            me = self.state[self.seen]
            vy = me['v'][1]
            if vy < -.1 and me['p'][1] > self.goal[1]+.015:
                self.aim = float(me['p'][0]+me['v'][0]*(self.goal[1]-me['p'][1])/vy)
            goal = np.r_[self.aim, self.goal[1:]]
            servo.move(goal, self.rot, speed=TRACK_SPEED, turn_rate=.8)
            servo.grip(False)
            speed = float(np.linalg.norm(me['v']))
            self.vmax = max(self.vmax, speed)
            if self.jaw.max() > .3 or (self.vmax > .2 and speed < SETTLED):  # also when it stopped on the palm
                self.transition('seat', t, jaw_N=self.jaw.tolist(), speed=speed)
            elif el > 2.5:
                self.failures.append('Left:object_never_arrived')
        elif self.phase == 'seat':  # bring the pinch point onto the stopped object, so the pads close across its centre
            me = self.state[self.seen]
            tcp = servo.pose()[0]
            x, y, z = self.rot.apply(np.eye(3))
            err = me['p']-tcp
            along, across, side = float(err@x), float(err@y), float(err@z)
            # across the mouth first: the widest object leaves 4 mm per side, so going straight for the centre
            # would land the upper jaw on it. Ease off while a jaw is loaded.
            goal = tcp+y*across+z*side+(x*along if abs(across) < .003 else 0.)
            if self.jaw.max() > 6.:
                goal = tcp+y*(self.jaw[1]-self.jaw[0])*.001
            servo.move(goal, self.rot, speed=SEAT_SPEED, turn_rate=.8)
            servo.grip(False)
            if np.linalg.norm(tcp-me['p']) < .004 and np.linalg.norm(me['v']) < SETTLED:
                self.transition('close', t, offset=(tcp-me['p']).tolist())
            elif el > 2.5:
                self.failures.append('Left:not_seated')
        elif self.phase == 'close':
            servo.move(self.state[self.seen]['p'], self.rot, speed=.02, turn_rate=.5)
            fu, fl = self.jaw  # upper jaw (P), lower jaw (N)
            tu, tl = self.jaw_travel
            if fu > .5 and fl > .5:
                tu, tl = min(.05, tu+servo.squeeze_step), min(.05, tl+servo.squeeze_step)
            else:
                tu = min(.05, tu+(0. if fu > .5 else JAW_CATCH_SPEED*servo.dt))
                tl = min(.05, tl+(0. if fl > .5 else servo.close_step))
            self.jaw_travel = [tu, tl]
            servo.grip_each(tu, tl)
            if fu > GRIP_N and fl > GRIP_N:
                self.pose = (self.state[self.seen]['p'].copy(), self.rot)
                self.transition('lift', t, jaw_N=self.jaw.tolist())
            elif el > 2.:
                self.failures.append('Left:no_grasp')
        elif self.phase == 'lift':
            servo.grip_each(*self.jaw_travel)
            servo.move(self.pose[0]+[0, 0, .06], self.pose[1], speed=.10, turn_rate=.5)
            if el > 2.5:
                self.done_flag = True
                self.transition('done', t)
        elif self.phase == 'done':
            servo.grip_each(*self.jaw_travel)

    def trace(self):
        return dict(self.row, phase=self.phase)
