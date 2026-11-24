"""C3 physics episode: sort mixed spoons, forks and knives into the compartments of an open drawer.

Utensils: rigid bodies (40-45 g) with box colliders: the shared handle (16 x 8 mm section), a neck, and a
head box per kind (spoon bowl, fork tines, knife blade). All rest flat on their underside. The drawer
(floor, outer walls, three-compartment insert) is static. Before the episode starts the utensils are
moved to the seed's layout (scene.layout): which utensil lies where on the table, and the compartment
order (each compartment holds one reference utensil of its kind).

Scoring (task model, from measured state only): a loose utensil counts as sorted while it rests (its middle moved
< 1 cm over the last 0.5 s) inside the compartment of its kind, below the wall tops, touched by no jaw.

Robot (privileged oracle): each arm sorts the utensils on its side of the drawer, nearest to the drawer
first. Per utensil: joint-space move above it, top-down descent with the jaw axis across the handle,
close until both pads press, lift, joint-space move above its compartment (turned so the utensil lies
along the compartment), lower onto the stack, open, rise. Only one arm is over the drawer at a time.
"""
import numpy as np
from pxr import PhysxSchema
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.c3_x5.scene import (
    COMP_L,
    COMP_W,
    DRAWER_C,
    FLOOR,
    HANDLE,
    LOOSE_PER_KIND,
    PIECES,
    WALL_H,
    all_names,
    comp_x,
    drawer_outer,
    kind_of,
    layout,
)
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
FIXED_HORIZON = True
ARMS = ('Left', 'Right')
JAWS = ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Right/jawP', '/World/Arms/Right/jawN']
NAMES = all_names()
LOOSE = NAMES[:3*LOOSE_PER_KIND]
ROOT = {n: '/World/Objects/'+n for n in NAMES}
HEADS = {  # head collider boxes (x0, x1, half width, thickness) in the utensil frame
    'spoon': [(.031, .062, .003, .003), (.060, .120, .020, .010)],
    'fork': [(.031, .060, .004, .003), (.058, .122, .0125, .003)],
    'knife': [(.031, .152, .0075, .002)],
}
MID = {k: (HANDLE['x0']+v[-1][1])/2 for k, v in HEADS.items()}  # middle of the utensil along x
OPEN = .036              # jaw gap while approaching / releasing (m)
TCP_Z = .0035            # tool point above the counter when pinching a handle on the table (pad tips 1.9 mm up)
GRIP_N = 8.              # per pad before lifting
CARRY_Z = .13            # tool point height above the counter while carrying
DROP_CLEAR = .012        # utensil underside above the stack top when it is let go
SPEED, TURN = .16, 1.2   # Cartesian m/s, rad/s
CARRY_SPEED = .22        # m/s, utensil in hand
JOINT_RATE = 2.2         # rad/s, joint-space moves
MAX_TRIES = 2
REST_S, REST_M = .5, .01  # at rest: moved less than 1 cm over the last 0.5 s
G = {}


def down(jaw_deg):
    """Tool straight down, jaw axis at azimuth jaw_deg."""
    a = np.radians(jaw_deg)
    t, j = np.array([0., 0., -1.]), np.array([np.cos(a), np.sin(a), 0.])
    return R.from_matrix(np.column_stack((t, j, np.cross(t, j))))


def wrap(a):
    return (a+180.) % 360.-180.


def comp_of(p):
    """Compartment index (0-2) containing the xy point p, or None."""
    for i in range(3):
        if abs(p[0]-comp_x(i)) < COMP_W/2 and abs(p[1]-DRAWER_C[1]) < COMP_L/2:
            return i
    return None


def build(stage, mat):
    s = stage
    root = '/World/Objects/Drawer'
    for part in ('Floor', 'Front', 'Back', 'Wall0', 'Wall1', 'Wall2', 'Wall3', 'Felt'):
        P.collision(s.GetPrimAtPath(f'{root}/{part}'), mat)
    P.box(s, '/World/Fixtures/Floor', [.3, -.3, -.05], [8., 8., .1], mat)  # a dropped utensil lands instead of falling forever
    h = HANDLE
    for n in NAMES:
        kind = kind_of(n)
        boxes = [(h['x0'], h['x1'], h['w']/2, h['t'])]+HEADS[kind]
        for i, (x0, x1, hw, t) in enumerate(boxes):
            P.box(s, f'{ROOT[n]}/Hit{i}', [(x0+x1)/2, 0., t/2], [x1-x0, 2*hw, t], mat)
        prim = s.GetPrimAtPath(ROOT[n])
        P.body(prim, PIECES[kind]['m'], com=[PIECES[kind]['com'], 0., h['t']/2])
        PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr(True)
    ow, ol = drawer_outer()
    return dict(utensils={k: f'{v["m"]*1000:.0f} g; box colliders: handle 110 x 16 x 8 mm + head' for k, v in PIECES.items()},
                drawer=f'static, {ow*100:.1f} x {ol*100:.1f} cm, three compartments {COMP_W*100:.0f} x {COMP_L*100:.0f} cm, '
                f'walls {WALL_H*100:.1f} cm',
                layout=f'per seed: table slots of the {len(LOOSE)} loose utensils and the compartment order (scene.layout)',
                scoring='resting, released, inside the compartment of its kind, below the wall tops')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t, self.idle = [], 0., False
        self.order, self.poses = layout(task.spec.seed)
        self.comp = {kind: i for i, kind in enumerate(self.order)}
        self.phase = {a: 'next' for a in ARMS}
        self.step = {a: 0. for a in ARMS}
        self.item, self.plan, self.jaw_travel, self.hold = {}, {}, {}, {}
        self.tries = {n: 0 for n in LOOSE}
        self.skipped = set()
        self.lock = None          # arm currently over the drawer
        self.parked = {a: False for a in ARMS}
        self.row = {}
        self.hist = {n: [] for n in NAMES}

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        for n in NAMES:
            p, yaw = self.poses[n]
            P.pose(stage.GetPrimAtPath(ROOT[n]), p, R.from_euler('z', yaw, degrees=True))
        self.obj = {n: world.scene.add(RigidPrim(ROOT[n], name=n.lower(), reset_xform_properties=False, track_contact_forces=True,
                                                 contact_filter_prim_paths_expr=[JAWS], max_contact_count=64)) for n in NAMES}

    def measure(self, t, dt):
        self.t = t
        st = {}
        for n in NAMES:
            p, q = self.obj[n].get_world_poses()
            p, q = np.asarray(p)[0].copy(), np.asarray(q)[0]
            r = R.from_quat([q[1], q[2], q[3], q[0]])
            f = np.linalg.norm(np.asarray(self.obj[n].get_contact_force_matrix(dt=dt))[0], axis=-1)
            v = np.asarray(self.obj[n].get_linear_velocities())[0]
            mid = p+r.apply([MID[kind_of(n)], 0., 0.])
            st[n] = dict(p=p, r=r, f=f, v=v, mid=mid, yaw=float(np.degrees(np.arctan2(*r.apply([1., 0, 0])[1::-1]))))
        self.state = st
        # at rest: the middle moved < REST_M over the last REST_S (stacked thin bodies jitter at a few cm/s in
        # the solver while staying put, so an instantaneous speed threshold misjudges them)
        hist = self.hist
        for n in NAMES:
            hist[n].append((t, st[n]['mid'].copy()))
            while len(hist[n]) > 1 and hist[n][1][0] <= t-REST_S+1e-9:  # keep one sample at least REST_S old
                hist[n].pop(0)
        correct, wrong, placed, dropped = 0, 0, [], 0
        where = {}
        for n in LOOSE:
            d = st[n]
            c = comp_of(d['mid'])
            where[n] = c
            if d['p'][2] < H-.03:
                dropped += 1
            moved = max(float(np.linalg.norm(m-d['mid'])) for _, m in self.hist[n])
            resting = t-self.hist[n][0][0] >= REST_S-1e-6 and moved < REST_M
            if c is None or d['mid'][2] > H+WALL_H-.005 or d['f'].max() > .2 or not resting:
                continue
            if c == self.comp[kind_of(n)]:
                correct += 1
                placed.append(n)
            else:
                wrong += 1
        refs = [n for n in NAMES[3*LOOSE_PER_KIND:]]
        moved = any(comp_of(st[n]['mid']) != self.comp[kind_of(n)] or st[n]['mid'][2] > H+WALL_H for n in refs)
        self.row = dict(utensils={n: st[n]['p'].round(4).tolist() for n in NAMES}, yaw={n: round(st[n]['yaw'], 1) for n in NAMES},
                        compartment=where, order=list(self.order), correct=correct, wrong=wrong, placed=placed,
                        jaw_N={n: st[n]['f'].round(2).tolist() for n in NAMES if st[n]['f'].max() > .05}, lock=self.lock)
        return dict(correct=correct, wrong=wrong, placed=tuple(placed), dropped=dropped, references_moved=bool(moved))

    # ---------------------------------------------------------------- helpers
    @property
    def done(self):
        return all(self.parked.values())

    @property
    def finished(self):
        # Parked and the task has ended (all sorted and settled, or the stall stop) or reached its horizon: the
        # recorded episode then carries the task's termination reason.
        return self.done and (self.task.terminated or self.task.time >= self.task.spec.max_time - 1e-6)

    def transition(self, side, phase, t, **kw):
        self.phase[side], self.step[side] = phase, t
        self.event('phase', t, side=side, phase=phase, **kw)

    def solve(self, side, pos, rot, seed=None):
        """Joint solution with the largest joint-limit margin, or None."""
        servo = self.servos[side]
        b = servo.kin.base[:3, 3]
        ready = np.array(READY_Q)
        ready[0] = np.arctan2(pos[1]-b[1], pos[0]-b[0])-np.pi/2
        seeds = [ready]+[np.r_[ready[0], a, c, e, 0., j6] for a, c, e in ((2., 1.8, -1.), (1.6, 1.2, -.6), (2.4, 2.4, -1.4))
                         for j6 in (-1.2, 0., 1.2)]
        if seed is not None:
            seeds.insert(0, seed)
        best = None
        for s0 in seeds:
            q, pe, re = servo.kin.solve(pos, rot, s0)
            if pe < .002 and re < .02:
                m = float(np.min(np.minimum(q-LOW, HIGH-q)))
                if m > .05 and (best is None or m > best[1]):
                    best = (q, m)
        return None if best is None else best[0]

    def pick_list(self, side):
        """Loose utensils on this arm's side of the drawer, not yet sorted, nearest to the drawer first."""
        out = []
        for n in LOOSE:
            if n in self.skipped or n in self.item.values():
                continue
            d = self.state[n]
            if comp_of(d['mid']) is not None or d['p'][2] < H-.03:
                continue
            if (d['p'][0] < DRAWER_C[0]) == (side == 'Left'):
                out.append((abs(d['p'][0]-DRAWER_C[0])+abs(d['p'][1]-DRAWER_C[1]), n))
        return [n for _, n in sorted(out)]

    def plan_item(self, side, n):
        """Grasp pose (jaw across the handle at the centre of mass) and the turn that lays it along its compartment."""
        d = self.state[n]
        kind = kind_of(n)
        grasp = d['p']+d['r'].apply([PIECES[kind]['com'], 0., 0.])
        grasp[2] = H+TCP_Z+(d['p'][2]-H)
        best = None
        for a in (d['yaw']+90., d['yaw']-90.):
            for b in (0., 180.):  # jaw az over the drawer: the utensil then lies along y (either way round)
                turn = wrap(b-a)
                head = d['yaw']+turn  # utensil heading after the turn
                q_pick = None
                for z in (CARRY_Z, .09, .06):  # the far edge of the table is reached lower down
                    q = self.solve(side, np.r_[grasp[:2], H+z], down(a))
                    if q is not None and self.solve(side, grasp, down(a), q) is not None:
                        q_pick = (q, H+z)
                        break
                if q_pick is None:
                    continue
                cost = abs(turn)
                if best is None or cost < best[0]:
                    best = (cost, a, b, head, q_pick)
        if best is None:
            return None
        _, a, b, head, (q_pick, hi) = best
        return dict(name=n, grasp=grasp, jaw=a, drop_jaw=b, head=head, q_pick=q_pick, hi=hi, kind=kind)

    def drop_pose(self, side, plan):
        """Tool point over the compartment: the utensil middle at the compartment centre, its underside above the stack."""
        i = self.comp[plan['kind']]
        h = np.radians(plan['head'])
        ax = np.array([np.cos(h), np.sin(h), 0.])
        mid_to_grasp = PIECES[plan['kind']]['com']-MID[plan['kind']]
        xy = np.array([comp_x(i), DRAWER_C[1], 0.])+ax*mid_to_grasp
        top = H+FLOOR+.001
        for n in NAMES:  # stack top in this compartment
            d = self.state[n]
            if n != plan['name'] and comp_of(d['mid']) == i:
                top = max(top, d['p'][2]+HANDLE['t'])
        return np.array([xy[0], xy[1], top+DROP_CLEAR+TCP_Z]), down(plan['drop_jaw'])

    def plan_drop(self, side, plan):
        """Joint target above the compartment and the drop pose: the smaller turn first, then the other way round,
        carried as high as the arm reaches."""
        servo = self.servos[side]
        base = plan['drop_jaw']
        for b in sorted((base, (base+180.) % 360.), key=lambda b: abs(wrap(b-plan['jaw']))):
            p = dict(plan, drop_jaw=b, head=plan['head']+wrap(b-base))
            pos, rot = self.drop_pose(side, p)
            for z in (CARRY_Z, .10, .08):
                q = self.solve(side, np.r_[pos[:2], H+z], rot, servo.joints())
                if q is not None and self.solve(side, pos, rot, q) is not None:
                    plan.update(drop_jaw=b, head=p['head'])
                    return q, pos, rot, H+z
        return None

    def close(self, side, f, need):
        servo = self.servos[side]
        fp, fn = f
        tp, tn = self.jaw_travel.get(side, [0., 0.])
        if fp > .5 and fn > .5:
            tp, tn = min(P.JAW_TRAVEL+.01, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.01, tn+servo.squeeze_step)
        else:
            tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
            tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
        self.jaw_travel[side] = [tp, tn]
        servo.grip_each(tp, tn)
        return fp > need and fn > need

    def jaw_forces(self, side, n):
        f = self.state[n]['f']
        return f[:2] if side == 'Left' else f[2:]

    def joint_move(self, side, q, rate=JOINT_RATE, tol=.02):
        servo = self.servos[side]
        servo.move_joints(q, rate=rate)
        if np.max(np.abs(servo.joints()-q)) < tol:
            servo.reset_target()
            return True
        return False

    def cart(self, side, pos, rot, speed=SPEED, tol=.003):
        servo = self.servos[side]
        servo.move(pos, rot, speed=speed, turn_rate=TURN)
        return np.linalg.norm(servo.pose()[0]-pos) < tol and (rot*servo.pose()[1].inv()).magnitude() < .03

    def hold_jaws(self, side):
        self.servos[side].grip_each(*self.jaw_travel[side])

    def opening(self, side):
        self.servos[side].grip(False, opening=OPEN)

    # ---------------------------------------------------------------- control
    def command(self, t):
        for side in ARMS:
            self.arm(side, t)
            el = t-self.step[side]
            ph = self.phase[side]
            if ph not in ('next', 'wait', 'park', 'parked') and el > 20.:
                self.failures.append(f'{side}:phase_timeout:{ph}')

    def arm(self, side, t):
        servo = self.servos[side]
        ph, el = self.phase[side], t-self.step[side]
        plan = self.plan.get(side)
        if ph == 'next':
            self.opening(side)
            servo.move_joints(servo.target[servo.arm], rate=.1)
            todo = self.pick_list(side)
            if not todo:
                self.transition(side, 'park', t)
                return
            for n in todo:
                p = self.plan_item(side, n)
                if p is not None:
                    break
                self.skipped.add(n)
                self.event('unreachable', t, side=side, utensil=n)
            else:
                return
            self.plan[side], self.item[side] = p, n
            self.jaw_travel[side] = [0., 0.]
            self.transition(side, 'approach', t, utensil=n, kind=p['kind'], jaw=round(p['jaw'], 1), turn=round(wrap(p['drop_jaw']-p['jaw']), 1))
        elif ph == 'approach':
            self.opening(side)
            if self.joint_move(side, plan['q_pick']):
                self.transition(side, 'descend', t)
        elif ph == 'descend':
            self.opening(side)
            if self.cart(side, plan['grasp'], down(plan['jaw']), speed=.12 if servo.pose()[0][2] > H+.04 else .04, tol=.0015):
                self.jaw_travel[side] = [0., 0.]
                self.transition(side, 'close', t)
        elif ph == 'close':
            servo.move(plan['grasp'], down(plan['jaw']), speed=.01)
            f = self.jaw_forces(side, plan['name'])
            if self.close(side, f, GRIP_N) and el > .2:
                self.event('grasp', t, side=side, utensil=plan['name'], jaw_N=f.round(2).tolist())
                self.transition(side, 'lift', t)
            elif el > 3.:
                self.retry(side, t, 'no_grasp')
        elif ph == 'lift':
            self.hold_jaws(side)
            goal = np.r_[plan['grasp'][:2], plan['hi']]
            if self.cart(side, goal, down(plan['jaw']), tol=.01):
                if self.jaw_forces(side, plan['name']).min() < 1.:
                    self.retry(side, t, 'slipped')
                    return
                self.transition(side, 'wait', t)
        elif ph == 'wait':  # hold the utensil until the drawer is free
            self.hold_jaws(side)
            servo.move(*servo.pose(), speed=.01)
            if self.lock in (None, side):
                got = self.plan_drop(side, plan)
                if got is None:
                    self.failures.append(f'{side}:drop_unreachable:{plan["name"]}')
                    return
                self.lock = side
                plan['q_drop'], plan['drop'], plan['drop_rot'], plan['carry_z'] = got
                self.transition(side, 'carry', t, drop_jaw=plan['drop_jaw'], carry_z=round(got[3]-H, 3))
        elif ph == 'carry':  # straight line at carrying height (a joint-space move dipped the utensil into the walls)
            self.hold_jaws(side)
            if self.cart(side, np.r_[plan['drop'][:2], plan['carry_z']], plan['drop_rot'], speed=CARRY_SPEED, tol=.006):
                self.transition(side, 'lower', t)
        elif ph == 'lower':
            self.hold_jaws(side)
            tcp = servo.pose()[0]
            if self.cart(side, plan['drop'], plan['drop_rot'], speed=.10, tol=.002):
                self.transition(side, 'release', t)
            elif el > 1.5:  # stalled (the utensil rests on the stack, or the pose is at the edge of reach): let go here
                last = self.hold.get(side+'_z')
                if last is not None and abs(last-tcp[2]) < .0005:
                    self.transition(side, 'release', t, stalled_at=(tcp-plan['drop']).round(4).tolist())
                self.hold[side+'_z'] = tcp[2]
        elif ph == 'release':
            self.opening(side)
            servo.move(plan['drop'], plan['drop_rot'], speed=.01)
            if el > .35:
                self.transition(side, 'rise', t)
        elif ph == 'rise':
            self.opening(side)
            if self.cart(side, np.r_[plan['drop'][:2], plan['carry_z']], plan['drop_rot'], tol=.01):
                self.lock = None if self.lock == side else self.lock
                self.event('placed', t, side=side, utensil=plan['name'], compartment=self.comp[plan['kind']]+1)
                self.item.pop(side, None)
                self.transition(side, 'next', t)
        elif ph == 'back_off':  # straight up, open, before the next utensil
            self.opening(side)
            if self.cart(side, np.r_[servo.pose()[0][:2], H+CARRY_Z], servo.pose()[1], tol=.01) or el > 3.:
                self.transition(side, 'next', t)
        elif ph == 'park':
            self.opening(side)
            if self.lock == side:
                self.lock = None
            if self.joint_move(side, np.array(READY_Q), rate=1.2, tol=.03):
                self.parked[side] = True
                self.transition(side, 'parked', t)
        elif ph == 'parked':
            servo.move_joints(np.array(READY_Q), rate=1.)
            self.opening(side)
            if self.pick_list(side):  # something was knocked back onto the table: sort it too
                self.parked[side] = False
                self.transition(side, 'next', t)

    def retry(self, side, t, why):
        n = self.plan[side]['name']
        self.tries[n] += 1
        self.event('retry', t, side=side, utensil=n, why=why)
        if self.tries[n] >= MAX_TRIES:
            self.skipped.add(n)
        self.item.pop(side, None)
        self.opening(side)
        self.transition(side, 'back_off', t)

    def trace(self):
        return dict(self.row, phase=dict(self.phase))
