"""C1 physics episode (R3 serving relay at the pass): bowls, staging mat, serving board, pickups, oracle.

World (not robot control): the board holds one bowl. At each pickup time the waiter takes the bowl
resting on the board (it is moved off the counter, to a shelf out of view); the task model scores it
against the meal wanted at that pickup. A pickup that missed its meal leaves the order open: the waiter
takes that meal once it has rested on the board for task.LATE_TAKE_S (task.waiter lists every collection,
on time or late). Before the simulation starts the bowls are moved to the
seed's layout (task.layout): which meal is on the mat, which stands in which counter slot, and
small pose jitter.

Robot (privileged oracle, contact-driven, no object pose writes):
- left arm: takes the next wanted meal from its counter slot by handle B, carries it along an arc
  that stays clear of the region in front of its own base, waits beside the mat while the mat is
  busy, and sets it down on the mat;
- right arm: takes the staged meal from the mat by handle A, carries it away from the mat, holds it
  beside the board until the previous bowl has been picked up, and sets it down on the board.
The mat is shared: one arm at a time works there. Every grasp is the R2.2 close-until-contact pinch
of a 6 mm handle swept up 20 deg, with the tool pitched 20 deg down so the pads lie flat on it.
"""
import numpy as np
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.c1_x5.scene import (
    BOARD,
    BOARD_YAW,
    BOWL_SCALE,
    GRASP_R,
    HANDLE_PITCH,
    MAT_SIZE,
    STAGE,
    STAGE_YAW,
    handle_z,
)
from roboclock.envs.household.c1_x5.task import MEALS, SLOTS
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
FIXED_HORIZON = True  # running to the horizon is not an oracle failure (the task also ends early once all is served)
EL = -HANDLE_PITCH          # tool pitch for every bowl-handle grasp: pads parallel to the upswept handle
BOWL_MASS = .28*BOWL_SCALE**3  # 0.28 kg for the 12.5 cm bowl; mass follows the volume
PARK = {'Right': np.array(READY_Q), 'Left': np.array(READY_Q)+[.8, 0, 0, 0, 0, 0]}  # left parks turned outward
ACCEL, ANG_ACCEL = .25, 1.  # carry profile limits (m/s^2, rad/s^2): the pinch slips under jerks
JRATE = 2.4                # joint-space rate (rad/s) of the empty-hand moves (tuck, park, repositioning)
READY_UP = .03  # the right hand waits beside the mat this far above its pre-grasp pose (7 cm out from handle A, 2 cm up)
DESCEND = .10              # m/s: from the transit height to just above the place (then .03 to contact)
TRANSIT = H+.11            # bowl base height while carrying across the counter (clears the other bowls)
MAT_Z = H+MAT_SIZE[2]
BOARD_Z = H+BOARD['size'][2]
SHELF = (2.6, -2.5, .40)   # off-camera shelf for bowls the waiter has taken
# Left-arm carry from a counter slot (task.SLOTS index) to the mat: bowl-centre waypoints (x, y) and
# body yaw (deg) that keep the hand more than 0.3 m from the left base (no handle pose is reachable
# closer in front of it). LEFT_WAIT is where it holds the bowl while the mat is busy: more than 0.3 m
# from every point of the right arm's route.
LEFT_TO_WAIT = {0: [], 1: [], 2: [((.00, .18), 178.)]}
LEFT_WAIT = ((.14, .04), 155.)  # 0.28 m from the right arm's route off the mat; a straight carry from here to the mat stays reachable
# Right-arm route from the mat: one straight carry to beside the board, where it holds the meal until the
# pickup; it stays outside the region in front of the right base (a waypoint at (0.44, -0.10) left the grasp
# point 0.15 m from it: unreachable). The mat is released once the held bowl is MAT_CLEAR from the mat
# centre: from there the rest of the route keeps more than 0.24 m from LEFT_WAIT and the left arm's carry
# to the mat. HOVER keeps the held bowl 16 cm from the board bowl's centre.
RIGHT_HOVER = ((.48, .14), 40.)
MAT_CLEAR = .15
BOWLS = [f'/World/Objects/MealBowl{i+1}' for i in range(4)]
JAWS = ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN', '/World/Arms/Left/jawP', '/World/Arms/Left/jawN']


def tool(az, el=EL):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def build(stage, mat):
    s = stage

    def wbox(path, a, b):
        a, b = np.array(a, float), np.array(b, float)
        P.box(s, path, (a+b)/2, b-a, mat)
    bx, by = BOARD['center']
    wbox('/World/Fixtures/ServingBoard', [bx-BOARD['size'][0]/2, by-BOARD['size'][1]/2, H], [bx+BOARD['size'][0]/2, by+BOARD['size'][1]/2, BOARD_Z])
    wbox('/World/Fixtures/StagingMat', [STAGE[0]-MAT_SIZE[0]/2, STAGE[1]-MAT_SIZE[1]/2, H], [STAGE[0]+MAT_SIZE[0]/2, STAGE[1]+MAT_SIZE[1]/2, MAT_Z])
    wbox('/World/Fixtures/PickupShelf', [2.4, -2.6, .30], [3.6, -2.4, .32])  # off-camera: bowls taken by the waiter

    cache = UsdGeom.XformCache()
    for path in BOWLS:
        b = s.GetPrimAtPath(path)
        P.body(b, BOWL_MASS)
        # bowl body: a 32-sided frustum fitted to the visual mesh (the hull of the raw mesh, cut to
        # PhysX's 64-vertex limit, had a slanted base and left the bowls resting 6-8 deg tilted)
        inv = np.array(cache.GetLocalToWorldTransform(b).GetInverse())
        pts = []
        for p in Usd.PrimRange(b):
            if p.IsA(UsdGeom.Mesh) and '/Visual' in str(p.GetPath()):
                m = np.array(cache.GetLocalToWorldTransform(p))@inv
                v = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get(), float)
                pts.append(np.c_[v, np.ones(len(v))]@m)
        pts = np.concatenate(pts)[:, :3]
        r = np.hypot(pts[:, 0], pts[:, 1])
        z0, z1 = pts[:, 2].min(), pts[:, 2].max()
        r0, r1 = r[pts[:, 2] < z0+.004].max(), r[pts[:, 2] > z1-.004].max()
        ang = np.linspace(0., 2*np.pi, 32, endpoint=False)
        ring = np.c_[np.cos(ang), np.sin(ang)]
        hull = UsdGeom.Mesh.Define(s, path+'/Collider')
        hull.CreatePointsAttr([(*(r0*c), z0) for c in ring]+[(*(r1*c), z1) for c in ring])
        faces = [[k, (k+1) % 32, 32+(k+1) % 32, 32+k] for k in range(32)]+[list(range(31, -1, -1)), list(range(32, 64))]
        hull.CreateFaceVertexCountsAttr([len(f) for f in faces])
        hull.CreateFaceVertexIndicesAttr(sum(faces, []))
        hull.CreateVisibilityAttr('invisible')
        P.collision(hull.GetPrim(), mat)
        for name in ('HandleA', 'HandleB'):
            P.collision(s.GetPrimAtPath(f'{path}/{name}'), mat)
    return dict(bowls=f'four rigid bowls ({BOWL_MASS} kg, convex hull) with two box lug handles each',
                pickup='at each pickup time the bowl resting on the board is moved off the counter (the waiter takes it)',
                layout='per seed: wanted order, meal per counter slot and pose jitter (task.layout); moved at setup',
                staging_mat=list(STAGE), board=list(BOARD['center']), board_size=list(BOARD['size']), slots=[list(x) for x in SLOTS])


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        spec = task.spec
        self.order, self.slots = tuple(spec.order), tuple(spec.slots)
        self.n = len(self.order)
        # routing: the nominal counter slot nearest each waiting meal's (jittered) pose
        self.slot_index = {b: int(np.argmin([np.hypot(p[0]-x, p[1]-y) for x, y, _ in SLOTS]))
                           for b, p in enumerate(self.slots) if p is not None}
        self.queue = {'Left': [], 'Right': []}
        self.phase = {'Left': 'idle', 'Right': 'idle'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.failures, self.t = [], 0.
        self.staged = next(b for b, p in enumerate(self.slots) if p is None)  # the meal starting on the mat
        self.next_stage, self.next_serve = 1, 0  # indices into the wanted order
        self.mat_user = None
        self.taken = [False]*4
        self.handled = set()
        self.jaw_travel = {}
        self.grasp = {'Left': None, 'Right': None}
        self.offset = {}
        self.parked = {'Left': False, 'Right': False}
        self.right_ready = False  # the right hand waits beside the mat
        self.idle = False
        self.loc = ['counter']*4
        self.board_bowl = -1
        self.lost = {'Left': 0., 'Right': 0.}
        self.frozen = {}
        self.jplan = {}
        self.grasped_at = {}
        self.row = {}

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        self.stage = stage
        for b, path in enumerate(BOWLS):  # the seed's layout, before the world resets
            p = self.slots[b]
            if p is None:
                xyz, yaw = (*STAGE, MAT_Z), STAGE_YAW
            else:
                xyz, yaw = (p[0], p[1], H), p[2]
            P.pose(stage.GetPrimAtPath(path), xyz, R.from_euler('z', yaw, degrees=True))
        self.bowls = world.scene.add(RigidPrim(BOWLS, name='bowls', reset_xform_properties=False, track_contact_forces=True,
                                               contact_filter_prim_paths_expr=[JAWS]*4, max_contact_count=4096))
        self.heights = []
        for path in BOWLS:
            b = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(stage.GetPrimAtPath(path+'/Visual')).ComputeAlignedRange()
            self.heights.append(float(b.GetMax()[2]-b.GetMin()[2]))

    def appliance_step(self, stage):
        task = self.task
        for k, w in enumerate(task.waiter):  # the task's collections, scheduled pickups and overdue meals alike
            if k in self.handled:
                continue
            self.handled.add(k)
            i, b = w['pickup'], w['bowl']
            if b is not None and not self.taken[b]:
                p, q = self.bowls.get_world_poses()
                p = np.asarray(p).copy()
                p[b] = [SHELF[0]+.25*b, SHELF[1], SHELF[2]]
                self.bowls.set_world_poses(positions=p, orientations=np.asarray(q))
                self.taken[b] = True
            self.event('waiter_pickup', self.t, pickup=i+1, wanted=MEALS[self.order[i]],
                       served=None if b is None else MEALS[b], qualified=bool(task.qualified[i]), late=w['late'])

    # ---------------------------------------------------------------- signals
    def measure(self, t, dt):
        self.t = t
        p, q = self.bowls.get_world_poses()
        self.bp = np.asarray(p)
        self.br = [R.from_quat([qq[1], qq[2], qq[3], qq[0]]) for qq in np.asarray(q)]
        f = np.linalg.norm(np.asarray(self.bowls.get_contact_force_matrix(dt=dt)), axis=-1)  # (4 bowls, 4 jaws)
        self.bf = f
        bx, by = BOARD['center']
        loc, board = [], -1
        for i in range(4):
            c = self.bp[i]
            tilt = np.degrees(np.arccos(np.clip(self.br[i].apply([0, 0, 1])[2], -1, 1)))
            held = f[i, :2].min() > 1. or f[i, 2:].min() > 1.
            if self.taken[i]:
                loc.append('taken')
            elif c[2] < .84 or tilt > 40:
                loc.append('dropped')
            elif held:
                loc.append('held')
            elif abs(c[0]-bx) < BOARD['size'][0]/2 and abs(c[1]-by) < BOARD['size'][1]/2 and abs(c[2]-BOARD_Z) < .01:
                loc.append('board')
                if tilt < 10. and board < 0:
                    board = i
            elif abs(c[0]-STAGE[0]) < MAT_SIZE[0]/2 and abs(c[1]-STAGE[1]) < MAT_SIZE[1]/2 and abs(c[2]-MAT_Z) < .01:
                loc.append('mat')
            else:
                loc.append('counter')
        self.loc, self.board_bowl = loc, board
        self.row = dict(bowl_positions=self.bp.tolist(), bowl_quat_xyzw=[r.as_quat().tolist() for r in self.br], bowl_jaw_forces_N=f.tolist(),
                        bowl_location=list(loc), board_bowl=board, taken=list(self.taken), mat_user=self.mat_user, staged=self.staged)
        return dict(bowl_location=tuple(loc), board_bowl=board)

    # ------------------------------------------------------------- geometry
    def yaw(self, i):
        fwd = self.br[i].apply([1, 0, 0])
        return float(np.degrees(np.arctan2(fwd[1], fwd[0])))  # bowl body yaw (handle A at yaw-90)

    def grasp_point(self, i, side):
        """Tool point on handle A (right) or B (left) of bowl i from its measured pose (tilt included)."""
        return self.bp[i]+self.br[i].apply([0., GRASP_R*(-1 if side == 'Right' else 1), handle_z(self.heights[i])])

    def plan_joints(self, side, pos, rot, seed=None):
        """IK for a far pose, seeded with the ready pose turned toward the target (keeps the elbow-up,
        wrist-centred branch)."""
        servo = self.servos[side]
        base = servo.kin.base[:3, 3]
        ready = np.array(READY_Q)
        ready[0] = np.arctan2(pos[1]-base[1], pos[0]-base[0])-np.pi/2
        best = None
        for q0 in ((seed,) if seed is not None else (ready, servo.joints())):
            q, pe, re = servo.kin.solve(pos, rot, q0)
            margin = float(np.min(np.minimum(q-LOW, HIGH-q)))
            if pe < .002 and re < .02 and (best is None or margin > best[1]):
                best = (q, margin)
        if best is None:
            raise RuntimeError(f'{side}: pose unreachable {np.round(pos, 3)}')
        return best[0]

    def az_for(self, yaw, side):
        return yaw+90. if side == 'Right' else yaw-90.

    # --------------------------------------------------------------- planning
    def _mat_free_for_left(self):
        return self.mat_user is None and self.staged is None and 'mat' not in self.loc

    def _take_mat(self, side):
        self.mat_user = side

    def _staged(self, b):
        self.staged, self.next_stage = b, self.next_stage+1

    def _right_clear(self):
        self.staged, self.mat_user = None, None

    def _left_clear(self):
        if self.mat_user == 'Left':
            self.mat_user = None

    def _served(self, b):
        self.next_serve += 1

    def plan_right(self):
        q = self.queue['Right']
        if self.next_serve >= self.n:
            if not self.parked['Right']:
                q += [*self.home('Right'), ('call', lambda: self.parked.__setitem__('Right', True), 'right_done')]
            return
        b = self.order[self.next_serve]
        if self.staged != b or self.mat_user is not None or self.loc[b] != 'mat':
            if not self.right_ready:  # wait beside the mat, just above the pre-grasp pose (the left arm places from the far side)
                rot = tool(self.az_for(STAGE_YAW, 'Right'))
                ready = np.array(self._mat_handle_a())-rot.apply([.07, 0, 0])+[0, 0, .02+READY_UP]
                # by the park pose: from the board a straight line would cross the region in front of the right base
                q += [*self.home('Right'), ('pose', lambda p=ready: p, lambda r=rot: r, .18, .006, 'ready_Right', 'open'),
                      ('call', lambda: setattr(self, 'right_ready', True), 'at_mat')]
            return
        self.right_ready = False
        self.mat_user = 'Right'
        q += self.pick(b, 'Right')
        (hx, hy), hyaw = RIGHT_HOVER
        # one carry to beside the board; the mat is released on the way (_watch_mat)
        q += self.carry(b, 'Right', [(np.array([hx, hy, TRANSIT]), hyaw, .18, 'hover')])
        q += [('wait', lambda: self.board_bowl < 0 and 'board' not in self.loc, 'wait_board')]
        # short back-off clear of the handle, then straight up: the board lies close to the right base,
        # where a long back-off toward it flips the wrist into a joint limit
        q += self.place(b, 'Right', np.array([*BOARD['center'], BOARD_Z]), BOARD_YAW, back=(.02, .005), lift=.065, speed=.14)
        q += [('call', lambda b=b: self._served(b), 'served')]  # next: the mat (or beside it), not the park pose

    def _mat_handle_a(self):
        """Nominal handle-A tool point of a bowl set down on the mat."""
        h = self.heights[0]
        a = np.radians(STAGE_YAW-90.)
        return [STAGE[0]+GRASP_R*np.cos(a), STAGE[1]+GRASP_R*np.sin(a), MAT_Z+handle_z(h)]

    def plan_left(self):
        q = self.queue['Left']
        if self.next_stage >= self.n:
            if not self.parked['Left']:
                q += [*self.home('Left'), ('call', lambda: self.parked.__setitem__('Left', True), 'left_done')]
            return
        b = self.order[self.next_stage]
        k = self.slot_index[b]
        q += self.pick(b, 'Left')
        (wx, wy), wyaw = LEFT_WAIT
        q += self.carry(b, 'Left', [(np.array([x, y, TRANSIT]), yaw, .14, f'arc{n+1}') for n, ((x, y), yaw) in enumerate(LEFT_TO_WAIT[k])])
        if k == 0:  # the near slot has a straight, well-conditioned line to the mat: skip the waiting point when the mat is free
            q += [('skipif', self._mat_free_for_left, 1, 'direct')]
        q += self.carry(b, 'Left', [(np.array([wx, wy, TRANSIT]), wyaw, .14, 'beside_mat')])
        q += [('wait', self._mat_free_for_left, 'wait_mat'), ('call', lambda: self._take_mat('Left'), 'take_mat')]
        q += self.place(b, 'Left', np.array([*STAGE, MAT_Z]), STAGE_YAW, back=(.05, .01), speed=.14)  # one straight carry from LEFT_WAIT
        # backed off 5 cm on the far side of the bowl from handle A: the mat is free for the right arm
        q += [('call', lambda b=b: self._staged(b), 'staged'), ('call', self._left_clear, 'release_mat'), ('tuck', 'tuck_Left')]

    # step builders
    def home(self, side):
        if side == 'Right':
            return [('joint', PARK[side], 'home_'+side)]
        return [('tuck', 'tuck_'+side), ('joint', PARK[side], 'home_'+side)]  # fold first, then turn the base

    def pick(self, i, side):
        def target():
            return self.grasp_point(i, side)

        def rot():
            return tool(self.az_for(self.yaw(i), side))
        wide = 'wide' if side == 'Left' else 'open'  # on the counter: 12 mm clearance per jaw around the handle
        if side == 'Right':  # from the park pose, beside the mat or the board: an empty-hand Cartesian move to the mat
            first = ('pose', lambda: target()-rot().apply([.07, 0, 0])+[0, 0, .02], rot, .28, .006, 'pre_'+side, 'open')
        else:  # long moves from the park pose or the mat go through joint space
            first = ('jpose', lambda: target()-rot().apply([.07, 0, 0])+[0, 0, .02], rot, 'pre_'+side, wide)
        return [first,
                ('pose', lambda: target()-rot().apply([.07, 0, 0]), rot, .10, .004, 'align_'+side, wide),
                ('freeze', i, side),
                ('pose', lambda: self.frozen[side][0], lambda: self.frozen[side][1], .09, .006, 'insert_'+side, wide),
                ('close', i, side),
                ('pose', lambda: self.tool_for(side, self.grasped_at[side]+[0, 0, .012]), lambda: self.offset[side][1], .05, .004, 'lift_'+side, 'hold'),
                # rise straight to the transit height before any horizontal move (a swinging bowl scraped the counter)
                ('pose', lambda: self.tool_for(side, self._raised(side)), lambda: self.offset[side][1], .12, .006, 'raise_'+side, 'hold')]

    def _raised(self, side):
        g = self.grasped_at[side]
        return np.array([g[0], g[1], max(g[2]+.012, TRANSIT if g[2] < H+.03 else g[2]+.012)])

    def carry(self, i, side, waypoints):
        """Move the held bowl through base-centre waypoints [(base_xyz, body_yaw, speed, name)]."""
        steps = []
        for base, yaw, speed, name in waypoints:
            r = tool(self.az_for(yaw, side))
            steps.append(('pose', lambda b=base, y=yaw: self.carry_tcp(i, side, b, y), lambda r=r: r, speed, .008, name, 'hold'))
        return steps

    def place(self, i, side, base, yaw, transit=True, back=(.08, .01), lift=0., speed=.10):
        """Set bowl i down with its base centre at `base` and body yaw `yaw`, release and back off."""
        r = tool(self.az_for(yaw, side))

        def tcp_at():
            return self.carry_tcp(i, side, base, yaw)
        steps = []
        if transit:  # up to the transit height, across, then down
            steps += [('pose', lambda: self._up(side), lambda: self.servos[side].pose()[1], .10, .008, 'up_'+side, 'hold'),
                      ('pose', lambda: self.carry_tcp(i, side, np.array([base[0], base[1], TRANSIT]), yaw), lambda: r, speed, .008, 'carry_'+side, 'hold'),
                      ('pose', lambda: tcp_at()+[0, 0, .012], lambda: r, DESCEND, .004, 'over_'+side, 'hold')]
        steps += [('pose', lambda: tcp_at()+[0, 0, .001], lambda: r, .03, .003, 'lower_'+side, 'hold'),
                  ('settle', i, side),
                  ('release', i, side),
                  ('pose', lambda: tcp_at()-r.apply([back[0], 0, 0])+[0, 0, back[1]], lambda: r, .10, .008, 'back_'+side, 'open')]
        if lift:
            steps.append(('pose', lambda: tcp_at()-r.apply([back[0], 0, 0])+[0, 0, back[1]+lift], lambda: r, .10, .008, 'lift_off_'+side, 'open'))
        return steps

    def _up(self, side):
        tcp, _ = self.servos[side].pose()
        i = self.grasp[side]
        return np.array([tcp[0], tcp[1], max(tcp[2], TRANSIT+handle_z(self.heights[i]))])

    def carry_tcp(self, i, side, base, yaw):
        """TCP that puts bowl i (held with the recorded grasp offset) at base/yaw."""
        return base+R.from_euler('z', yaw, degrees=True).apply(self.offset[side][2])

    def tool_for(self, side, bowl_pos):
        return bowl_pos+self.offset[side][0]

    # ---------------------------------------------------------------- control
    def transition(self, side, phase, t):
        self.phase[side] = phase
        self.phase_time[side] = t
        self.event('phase', t, side=side, phase=phase)

    @property
    def done(self):
        return all(self.parked.values()) and not self.queue['Left'] and not self.queue['Right']

    def _watch_mat(self):
        """Release the mat once the right arm's held bowl has left it by MAT_CLEAR."""
        g = self.grasp['Right']
        if self.mat_user == 'Right' and g is not None and np.hypot(*(self.bp[g][:2]-STAGE)) > MAT_CLEAR:
            self._right_clear()
            self.event('mat_clear', self.t, side='Right')

    def command(self, t):
        self._watch_mat()
        if not self.queue['Right']:
            self.plan_right()
        if not self.queue['Left']:
            self.plan_left()
        self.idle = all(not q or q[0][0] == 'wait' for q in self.queue.values())
        for side in ('Right', 'Left'):
            self.step(side, t)

    def step(self, side, t):
        servo = self.servos[side]
        q = self.queue[side]
        tcp, rr = servo.pose()
        if not q:
            servo.move_joints(PARK[side], rate=1.)
            servo.grip(True)
            return
        s = q[0]
        kind = s[0]
        name = (s[5] if kind == 'pose' else s[3] if kind in ('jpose', 'skipif') else s[2] if kind in ('wait', 'joint', 'call')
                else s[1] if kind == 'tuck' else f'{kind}_{side}')
        if self.phase[side] != name:
            self.transition(side, name, t)
        elapsed = t-self.phase_time[side]
        advance = False
        grip = 'hold'
        if kind == 'wait':
            advance = s[1]()
            servo.move_joints(servo.target[servo.arm], rate=1.)  # hold still (no per-tick IK while waiting)
            grip = 'keep'
        elif kind == 'call':
            s[1]()
            advance = True
            grip = 'keep'
        elif kind == 'skipif':  # drop the next s[2] steps if the condition holds now
            if s[1]():
                del q[1:1+s[2]]
            advance = True
            grip = 'keep'
        elif kind == 'tuck':  # ready pose at the current base angle
            if side not in self.jplan:
                tuck = np.array(READY_Q)
                tuck[0] = servo.joints()[0]
                self.jplan[side] = [tuck]
            servo.move_joints(self.jplan[side][0], rate=JRATE)
            advance = float(np.max(np.abs(servo.joints()-self.jplan[side][0]))) < .02
            if advance:
                self.jplan.pop(side)
            grip = 'close'
        elif kind == 'joint':
            servo.move_joints(s[1], rate=JRATE)
            advance = float(np.max(np.abs(servo.joints()-s[1]))) < .03
            grip = 'close'
        elif kind == 'jpose':  # long repositioning: joint-space move to an IK solution planned once
            _, pos, rot, _, grip = s
            if side not in self.jplan:  # turn the base while tucked, unfold above the target, descend
                target = pos()
                q_above = self.plan_joints(side, target+[0, 0, .10], rot())
                q_goal = self.plan_joints(side, target, rot(), seed=q_above)
                tuck = np.array(READY_Q)
                tuck[0] = q_above[0]
                self.jplan[side] = [tuck, q_above, q_goal]
            goal = self.jplan[side][0]
            servo.move_joints(goal, rate=JRATE)
            if float(np.max(np.abs(servo.joints()-goal))) < .02:
                self.jplan[side].pop(0)
            advance = not self.jplan[side]
            if advance:
                self.jplan.pop(side)
        elif kind == 'pose':
            _, pos, rot, speed, tol, _, grip = s
            goal, r = pos(), rot()
            turn = 1.2
            if grip == 'hold':  # carrying: ramp speed and turn rate up and down (the pinch slips under jerks)
                d, ang = np.linalg.norm(goal-tcp), (r*rr.inv()).magnitude()
                speed = max(.004, min(speed, ACCEL*elapsed, np.sqrt(2*ACCEL*d)))
                turn = max(.02, min(.5, ANG_ACCEL*elapsed, np.sqrt(2*ANG_ACCEL*ang)))
            servo.move(goal, r, speed=speed, turn_rate=turn)
            advance = np.linalg.norm(tcp-goal) < tol and (r*rr.inv()).magnitude() < .06
        elif kind == 'freeze':  # handle target computed once from the bowl's current pose
            i = s[1]
            self.frozen[side] = (self.grasp_point(i, side), tool(self.az_for(self.yaw(i), side)))
            self.event('grasp_target', self.t, side=side, bowl=i+1, tcp=np.round(self.frozen[side][0], 4).tolist(),
                       bowl_pos=np.round(self.bp[i], 4).tolist(), bowl_height=round(self.heights[i], 4))
            advance, grip = True, 'open'
        elif kind == 'close':  # close-until-contact on the 6 mm handle, then clamp
            i = s[1]
            cols = [0, 1] if side == 'Right' else [2, 3]
            fp, fn = self.bf[i, cols]
            tp, tn = self.jaw_travel.get(side, [(P.JAW_OPENING-.022)/2]*2)
            if fp > .5 and fn > .5:
                tp = tn = P.JAW_TRAVEL+.025
            else:
                tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
                tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
            self.jaw_travel[side] = [tp, tn]
            servo.grip_each(tp, tn)
            servo.move(*self.frozen[side], speed=.02)
            if fp > 2. and fn > 2. and min(fp, fn) > .25*max(fp, fn):
                self.offset[side] = (tcp-self.bp[i], rr, R.from_euler('z', self.yaw(i), degrees=True).inv().apply(tcp-self.bp[i]))
                self.grasp[side] = i
                self.grasped_at[side] = self.bp[i].copy()  # lift relative to where the bowl was grasped
                advance = True
            grip = 'keep'
        elif kind == 'settle':
            advance = elapsed > .3
            servo.move(tcp, rr, speed=.02)
        elif kind == 'release':
            advance = elapsed > .4
            servo.move(tcp, rr, speed=.02)
            grip = 'open'
            self.grasp[side] = None
            self.jaw_travel.pop(side, None)
        if grip == 'open':
            servo.grip(False, opening=.022)
        elif grip == 'wide':
            servo.grip(False, opening=.030)
        elif grip == 'close':
            servo.grip(True)
        elif grip == 'hold':
            servo.grip(True, squeeze=.025)
        if advance:
            q.pop(0)
        elif elapsed > 25. and kind not in ('wait',):
            self.failures.append(f'{side}:step_timeout:{name}')
        g = self.grasp[side]
        if g is not None and kind == 'pose' and s[6] == 'hold':
            cols = [0, 1] if side == 'Right' else [2, 3]
            if self.bf[g, cols].min() < .3 and elapsed > .5:
                self.lost[side] += servo.dt
                if self.lost[side] > .3:
                    self.failures.append(f'{side}:lost_bowl{g+1}:{name}')
            else:
                self.lost[side] = 0.

    @property
    def finished(self):
        times = self.task.spec.pickup_times
        return self.done and all(f'pickup{i+1}' in self.task.events for i in range(self.n)) and self.t > times[-1]+2.

    def trace(self):
        return dict(self.row, phase=dict(self.phase), queue={k: len(v) for k, v in self.queue.items()},
                    next_stage=self.next_stage, next_serve=self.next_serve)
