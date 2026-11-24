"""A1 physics episode: colliders/rigid bodies, measured signals and the privileged oracle controller.

Sugar for a glass of tea (2026-11-19, replaces the tea infuser; contact-driven, no tool pose writes after reset).
The glass jar is a dynamic hollow rigid body (base disc, 24 wall boxes, shoulder ring, neck tube); the glass of tea
and the spoon basket are fixed fixtures. Sugar is a process model in task.py: it runs from the jar once the measured
jar tilt passes the onset angle and lands in the glass while the stream from the jar's lowest mouth point falls
inside the glass mouth (measured here, ``mouth_over_glass``); the mound, the stream and the sugar left in the jar
are drawn from the task's grams (``update_cues``).

Oracle (privileged poses, contact forces and the task's grams): the left arm pinches the jar body 3.5 cm up, lifts
it, carries it upright beside the glass and tilts it toward the glass while the mouth moves over it; the tilt rises
quickly to the first flow, is servoed to Q_TARGET g/s and tips back once the grams in the glass, in the air and the
expected tail reach the middle of the band. The jar goes back to its spot, upright, and is released. The right arm
picks the spoon from its basket meanwhile, waits above it, stirs two clockwise turns once the jar is down and
returns the spoon to the wire basket.
"""
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.a1_x5.scene import (
    CUP as CUP_XY,
    HIDDEN,
    JAR_FLOOR,
    JAR_GRIP_Z,
    JAR_MOUTH_R,
    JAR_MOUTH_Z,
    JAR_R,
    JAR_R_IN,
    JAR_SHOULDER,
    JAR_XY,
    SPOON_SHIFT,
)
from roboclock.envs.household.a1_x5.task import REPOSE_DEG, SUGAR_BULK_G_ML, Pinch, mound_radius
from roboclock.envs.household.c2_x5.sim import _annulus, _disc, _ring
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW
from roboclock.envs.household.x5.task_base import place

CUP = np.array(CUP_XY)
TEA_Z = H+.064             # liquid surface
CUP_FLOOR, CUP_RIM, CUP_R = H+.0075, H+.0814, .0385
GLASS_R_OUT = .0425
BASKET = np.array([.46, .04])+SPOON_SHIFT  # nominal basket centre; the seed's layout moves it (Oracle.basket)
BASKET_PARTS = ('/World/Objects/SpoonBasket', '/World/Fixtures/SpoonRest', '/World/Objects/Spoon')  # moved as one
OBJECTS = ['SugarJar', 'Spoon']
JAR = '/World/Objects/SugarJar'
# contact filter columns
FILTERS = ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Right/jawP', '/World/Arms/Right/jawN',
           '/World/Fixtures/Island', '/World/Fixtures/Cup/Floor', '/World/Fixtures/SpoonRest']
JAR_KG = .20               # 120 g glass jar + the sugar in it
PREGRASP = .05
CONTROL = dict(left='ik', right='ik')
PARK = np.array(READY_Q)
GRASP_TOOL = (45., -10.)   # tool azimuth, elevation at the jar pickup (reach-screened: >= 0.27 rad joint margin)
CARRY_Z = H+.105           # carried jar base clears the glass rim (8.1 cm)
OFFSET = .085              # upright jar axis this far from the glass axis (along -x) before the tilt
MOUTH_IN = .012            # mouth centre this far short of the glass axis while pouring
BLEND_DEG = 30.
MAX_TILT = 130.
Q_TARGET = 2.5             # g/s, servoed flow
TAIL_S = .30               # s of flow still to land after tipping back starts (tilt-back flow + the stream in the air)
TILT_FAST, TILT_BACK, TILT_ACC = 45., 60., 120.
ACCEL, ANG_ACCEL = .4, 1.5


def tool(az, el, roll=-90.):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))*R.from_euler('x', roll, degrees=True)


def smooth(u):
    u = min(1., max(0., u))
    return u*u*(3-2*u)


def build(stage, mat):
    """Task colliders and dynamic tools on top of the shared overlay."""
    P.vessel(stage, 'Cup', CUP, CUP_FLOOR, CUP_RIM, [.039, .039, .0395, .0395], mat)
    # Only the empty native wire basket is shown (bundled bottles hidden), with box colliders.
    src = stage.GetPrimAtPath('/World/Objects/SpoonBasket/Source')
    for prim in src.GetChildren():
        if prim.IsA(UsdGeom.Mesh) and prim.GetName() != 'Kitchen_Basket001':
            prim.SetActive(False)
    P.box(stage, '/World/Fixtures/SpoonRest/Floor', [*BASKET, H+.002], [.116, .074, .004], mat)
    for sign in [-1, 1]:
        tag = 'P' if sign > 0 else 'N'
        P.box(stage, '/World/Fixtures/SpoonRest/WallX'+tag, [BASKET[0]+sign*.061, BASKET[1], H+.023], [.003, .083, .043], mat)
        P.box(stage, '/World/Fixtures/SpoonRest/WallY'+tag, [BASKET[0], BASKET[1]+sign*.040, H+.023], [.125, .003, .043], mat)
    P.body(stage.GetPrimAtPath('/World/Fixtures/SpoonRest'), 1., kinematic=True)
    _disc(stage, JAR+'/Base', JAR_R, 0., JAR_FLOOR, mat)
    _ring(stage, JAR, JAR_R_IN, JAR_R-JAR_R_IN, 0., JAR_SHOULDER, mat)
    _annulus(stage, JAR, JAR_MOUTH_R, JAR_R, JAR_SHOULDER+.0015, .003, mat)
    _ring(stage, JAR+'/Neck', JAR_MOUTH_R, .0018, JAR_SHOULDER, JAR_MOUTH_Z, mat, n=16)
    P.body(stage.GetPrimAtPath(JAR), JAR_KG, com=(0., 0., .025))
    spoon = stage.GetPrimAtPath('/World/Objects/Spoon')
    P.body(spoon, .025)
    for p in Usd.PrimRange(spoon):
        if p.IsA(UsdGeom.Mesh):
            P.collision(p, mat)
    return dict(vessels='glass of tea and wire basket are fixed fixtures with open box-segment cavity colliders; '
                        'glass displacement is therefore 0 by construction',
                jar=f'dynamic hollow compound (base disc, 24 wall boxes, shoulder ring, neck tube), {JAR_KG*1000:.0f} g with its sugar',
                spoon='HANDAL spoon (25 g, convex hull) is dynamic',
                sugar='process model (task.py): flow from the measured jar tilt, routed by the measured stream landing '
                      'point; mound, stream and jar contents drawn from it')


def _ops(stage, path):
    ops = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()
    kinds = {op.GetOpType(): op for op in ops}
    return kinds[UsdGeom.XformOp.TypeTranslate], kinds[UsdGeom.XformOp.TypeScale]


class Arm:
    def __init__(self):
        self.queue, self.planned = [], False
        self.phase, self.phase_time = 'idle', 0.
        self.jplan, self.jaw_travel, self.grasp = [], None, None
        self.tilt_cmd = self.tilt_rate = 0.
        self.stopping = self.flowing = False


class Oracle:
    """Measures the A1 signals and runs the two-arm controller (privileged state)."""

    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.state, self.failures = {}, []
        self.left = Arm()
        self.phase = {'Left': 'idle', 'Right': 'approach'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.held, self.hold, self.unhook = {}, {}, {}
        self.right_pick = self.spoon_grip_local = self.spoon_upright = self.initial_spoon = None
        self.angle, self.clear = 0., False
        self.grasp_dwell = {'Left': 0., 'Right': 0.}
        self.grasped = {'Left': False, 'Right': False}
        self.ever = {'Left': False, 'Right': False}
        self.previous_rel = {}
        self.other = {'Left': Pinch(), 'Right': Pinch()}  # the other arm's pinch on each tool (judged signals only)
        self.turns, self.last_angle = 0., None
        self.park = {'SugarJar': 0., 'Spoon': 0.}
        self.lip = None
        self.t = 0.
        lo, hi = task.spec.band
        self.target_g = (lo+hi)/2
        off = task.spec.offset('spoon_basket')
        self.basket, self.basket_yaw = place(BASKET, off, BASKET), np.radians(off[2])

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        P.shift(stage.GetPrimAtPath(JAR), self.task.spec.offset('jar'), JAR_XY)  # the seed's layout, before the reset
        for path in BASKET_PARTS:
            P.shift(stage.GetPrimAtPath(path), self.task.spec.offset('spoon_basket'), BASKET)
        self.tools = world.scene.add(RigidPrim(['/World/Objects/'+n for n in OBJECTS], name='tools', reset_xform_properties=False,
                                               track_contact_forces=True, contact_filter_prim_paths_expr=[FILTERS, FILTERS],
                                               max_contact_count=4096))
        spoon = Usd.Stage.Open(str(P.L.REUSED/'spoon/spoon_2.usd'))
        mesh = UsdGeom.Mesh(next(p for p in spoon.Traverse() if p.IsA(UsdGeom.Mesh)))
        native = np.array([UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()).Transform(v) for v in mesh.GetPointsAttr().Get()])
        self.spoon_head = native[native[:, 0] > .030]*1.2
        a = np.linspace(0, 2*np.pi, 24, endpoint=False)
        self.rim = np.column_stack((JAR_MOUTH_R*np.cos(a), JAR_MOUTH_R*np.sin(a), np.full_like(a, JAR_MOUTH_Z)))
        self.cue_ops = {name: _ops(stage, path) for name, path in
                        (('mound', '/World/Objects/SugarMound'), ('stream', '/World/Objects/SugarStream'), ('jar', JAR+'/Sugar'))}

    def update_cues(self):
        c = self.task.cues()
        t, s = self.cue_ops['mound']
        g = c['undissolved_g']
        if g < .05:
            s.Set(Gf.Vec3f(HIDDEN))
        else:
            r = mound_radius(g)
            h = 3*g/SUGAR_BULK_G_ML*1e-6/(math.pi*r*r)  # cone of the mound's volume (steeper once it spans the bottom)
            t.Set(Gf.Vec3d(float(CUP[0]), float(CUP[1]), CUP_FLOOR+h/2))
            s.Set(Gf.Vec3f(float(r), float(r), float(h)))
        t, s = self.cue_ops['stream']
        lip = self.lip
        if c['flow_g_s'] > .05 and lip is not None:
            inside = np.linalg.norm(lip[:2]-CUP) < CUP_R
            bottom = TEA_Z if inside else H
            if lip[2] > bottom+.002:
                t.Set(Gf.Vec3d(float(lip[0]), float(lip[1]), float((lip[2]+bottom)/2)))
                s.Set(Gf.Vec3f(.0022, .0022, float(lip[2]-bottom)))
            else:
                s.Set(Gf.Vec3f(HIDDEN))
        else:
            s.Set(Gf.Vec3f(HIDDEN))
        t, s = self.cue_ops['jar']
        r = JAR_R_IN-.0007
        h = min(JAR_SHOULDER-JAR_FLOOR, c['jar_g']/SUGAR_BULK_G_ML*1e-6/(math.pi*r*r))
        if h < .0005:
            s.Set(Gf.Vec3f(HIDDEN))
        else:
            t.Set(Gf.Vec3d(0., 0., JAR_FLOOR+h/2))
            s.Set(Gf.Vec3f(float(r), float(r), float(h)))

    # ---------------------------------------------------------------- signals
    def measure(self, t, dt):
        self.t = t
        positions, quats = self.tools.get_world_poses()
        vel = self.tools.get_velocities()
        forces = np.asarray(self.tools.get_contact_force_matrix(dt=dt))
        norm = np.linalg.norm(forces, axis=-1)
        assert norm.shape == (2, len(FILTERS)), norm.shape
        st = self.state
        for i, obj in enumerate(OBJECTS):
            st[obj] = {'p': positions[i].copy(), 'r': R.from_quat(np.r_[quats[i, 1:], quats[i, 0]]), 'v': vel[i, :3].copy()}
        jp, jr = st['SugarJar']['p'], st['SugarJar']['r']
        tilt = float(np.degrees(np.arccos(np.clip(jr.apply([0, 0, 1.])[2], -1, 1))))
        rim = jr.apply(self.rim)+jp
        self.lip = lip = rim[int(np.argmin(rim[:, 2]))]
        over = bool(np.linalg.norm(lip[:2]-CUP) < CUP_R-.004 and lip[2] > TEA_Z)
        hp = st['Spoon']['r'].apply(self.spoon_head)+st['Spoon']['p']
        head = hp.mean(0)
        st.update(head_min_z=float(hp[:, 2].min()), head_max_z=float(hp[:, 2].max()), spoon_jaw_forces=norm[1, 2:4].tolist(),
                  jar_jaw_forces=norm[0, 0:2].tolist(), tilt=tilt)
        self.support = {'SugarJar': bool(norm[0, 4] > .05), 'Spoon': bool(norm[1, 6] > .04 or norm[1, 4] > .04)}
        for i, side, cols in [(0, 'Left', [0, 1]), (1, 'Right', [2, 3])]:
            tcp, rot = self.servos[side].pose()
            rel = rot.inv().apply(positions[i]-tcp)
            speed = 0. if side not in self.previous_rel else float(np.linalg.norm(rel-self.previous_rel[side])/dt)
            self.previous_rel[side] = rel
            bilateral = bool(np.all(norm[i, cols] > .04) and speed < .10)
            self.grasp_dwell[side] = self.grasp_dwell[side]+dt if bilateral else 0.
            self.grasped[side] = self.grasp_dwell[side] > .065
            if self.grasped[side] and not self.ever[side]:
                self.ever[side] = True
                self.event('bilateral_grasp', t, side=side, forces_N=norm[i, cols].tolist())
        # Judged signals must not depend on which arm holds which tool: the same pinch test (task.Pinch) with the
        # right jaws on the jar and the left jaws on the spoon. The oracle's control keeps reading self.grasped.
        other = {}
        for i, side, cols in [(0, 'Right', [2, 3]), (1, 'Left', [0, 1])]:
            tcp, rot = self.servos[side].pose()
            other[OBJECTS[i]] = self.other[side].step(norm[i, cols], rot.inv().apply(positions[i]-tcp), dt)
        jar_held = self.grasped['Left'] or other['SugarJar']
        spoon_held = self.grasped['Right'] or other['Spoon']
        radial = np.linalg.norm(hp[:, :2]-CUP, axis=1)
        in_cup = bool(radial.max() < CUP_R and hp[:, 2].max() < TEA_Z and hp[:, 2].min() > CUP_FLOOR)
        if in_cup and spoon_held:
            angle = float(np.arctan2(head[1]-CUP[1], head[0]-CUP[0]))
            if self.last_angle is not None:
                delta = float(np.arctan2(np.sin(angle-self.last_angle), np.cos(angle-self.last_angle)))
                self.turns = max(0., self.turns-delta/(2*np.pi))  # clockwise seen from above = negative angle
            self.last_angle = angle
        else:
            self.last_angle = None
        released = {o: bool(np.max(norm[i, :4]) < .03) for i, o in enumerate(OBJECTS)}
        jar_down = (self.support['SugarJar'] and released['SugarJar'] and tilt < 10. and np.linalg.norm(vel[0, :3]) < .025
                    and jp[2] < H+.005)
        c, sn = np.cos(self.basket_yaw), np.sin(self.basket_yaw)  # spoon head in the (moved, turned) basket frame
        bx = c*(hp[:, 0]-self.basket[0])+sn*(hp[:, 1]-self.basket[1])
        by = -sn*(hp[:, 0]-self.basket[0])+c*(hp[:, 1]-self.basket[1])
        spoon_park = (self.support['Spoon'] and released['Spoon'] and np.linalg.norm(vel[1, :3]) < .025
                      and bx.min() > -.061 and bx.max() < .061 and by.min() > -.040 and by.max() < .040
                      and hp[:, 2].min() > H+.002 and hp[:, 2].max() < H+.046)
        self.park = {'SugarJar': self.park['SugarJar']+dt if jar_down else 0., 'Spoon': self.park['Spoon']+dt if spoon_park else 0.}
        tipped = tilt > 60. and not jar_held and jp[2] < H+.06  # lying on the counter
        self.row = {'tool_positions': positions.tolist(), 'tool_quaternions_wxyz': quats.tolist(), 'tool_velocities': vel.tolist(),
                    'contact_forces_N': norm.tolist(), 'jar_tilt_deg': tilt, 'jar_lip': lip.tolist(), 'mouth_over_glass': over,
                    'jar_held': bool(jar_held), 'spoon_head_center': head.tolist(),
                    'head_xy_bounds': [hp[:, :2].min(0).tolist(), hp[:, :2].max(0).tolist()],
                    'head_z_interval': [float(hp[:, 2].min()), float(hp[:, 2].max())], 'head_max_cup_radius': float(radial.max()),
                    'grasped': dict(self.grasped), 'spoon_held': bool(spoon_held), 'stir_turns': self.turns,
                    'park_dwell_s': dict(self.park)}
        return dict(jar_tilt_deg=tilt, mouth_over_glass=over, jar_upright_released=bool(jar_down), spoon_in_cup=in_cup,
                    stir_turns=self.turns, spoon_in_basket_released=bool(spoon_park), glass_displacement=0.0,
                    dropped=bool(positions[:, 2].min() < .80 or tipped))

    # --------------------------------------------------------------- poses
    def tcp_for(self, o, r):
        gp, gr = self.left.grasp
        return o+r.apply(gp), r*gr

    def pour_pose(self, tilt):
        """Jar origin and rotation at `tilt` toward +x: upright OFFSET short of the glass at tilt 0, the mouth moving
        over the glass during the first BLEND_DEG, then held MOUTH_IN short of the glass axis and raised just enough
        that no part of the jar comes within 5 mm of the rim (and 2 cm of the counter)."""
        w = smooth(tilt/BLEND_DEG)
        th = np.radians(tilt)
        r = R.from_rotvec([0., th, 0.])
        axis = r.apply([0, 0, 1.])
        up_mouth = CARRY_Z+JAR_MOUTH_Z
        mz = up_mouth*(1-w)+(CUP_RIM+.035)*w
        mx = CUP[0]-OFFSET*(1-w)-MOUTH_IN*w
        for _ in range(4):
            mouth = np.array([mx, CUP[1], mz])
            need = 0.
            for sd in np.arange(0., JAR_MOUTH_Z+1e-9, .005):  # distance down the axis from the mouth
                p = mouth-axis*sd
                rad = .0168 if sd < JAR_MOUTH_Z-JAR_SHOULDER else JAR_R
                low = p[2]-rad*np.sqrt(max(0., 1-axis[2]**2))  # lowest point of that cross-section
                if np.hypot(p[0]-CUP[0], p[1]-CUP[1]) < GLASS_R_OUT+rad+.005:
                    need = max(need, CUP_RIM+.005-low)
                need = max(need, H+.02-low)
            if need <= 1e-4:
                break
            mz += need
        return np.array([mx, CUP[1], mz])-r.apply([0, 0, JAR_MOUTH_Z]), r

    def plan_joints(self, side, pos, rot, seed=None):
        servo = self.servos[side]
        seeds = [servo.joints(), PARK, np.array([-.3, 1.8, 1.2, -1., 0, 0]), np.array([.6, .6, .4, 0, 0, 0]),
                 np.array([.8, 1., .8, -.4, 0, 0])]
        best = None
        for q0 in ((seed,) if seed is not None else seeds):
            q, pe, re = servo.kin.solve(pos, rot, q0)
            m = float(np.min(np.minimum(q-LOW, HIGH-q)))
            if pe < .002 and re < .02 and (best is None or m > best[1]):
                best = (q, m)
        if best is None:
            raise RuntimeError(f'{side}: pose unreachable {np.round(pos, 3)}')
        return best[0]

    def plan_left(self):
        rot = tool(*GRASP_TOOL)
        x = rot.apply([1, 0, 0])
        jar_xy = self.state['SugarJar']['p'][:2].copy()  # measured: where the seed's layout put the jar
        grasp = np.array([*jar_xy, H+JAR_GRIP_Z])
        pre = grasp-x*.05+[0, 0, .02]
        up = R.identity()
        start = np.array([*jar_xy, CARRY_Z])
        self.left.queue = [
            dict(kind='jpose', name='reach', pos=pre, rot=rot),
            dict(kind='pose', name='insert', pos=grasp, rot=rot, speed=.03, tol=.002),
            dict(kind='close', name='close', pos=grasp, rot=rot),
            dict(kind='call', name='grasped', fn=self._record),
            dict(kind='jar', name='lift', pose=lambda: (start, up), speed=.06, tol=.006),
            dict(kind='jar', name='to_glass', pose=lambda: self.pour_pose(0.), speed=.10, tol=.006),
            dict(kind='pour', name='pour'),
            dict(kind='jar', name='upright', pose=lambda: self.pour_pose(0.), speed=.08, tol=.008, turn=1.5),
            dict(kind='jar', name='over_place', pose=lambda: (start, up), speed=.10, tol=.004),
            dict(kind='jar', name='set_down', pose=lambda: (np.array([*jar_xy, H+.002]), up), speed=.04, tol=.003),
            dict(kind='release', name='release'),
            dict(kind='call', name='clear', fn=lambda: setattr(self, 'clear', True)),
            dict(kind='pose', name='retreat', pos=None, rot=None, speed=.08, tol=.01),
            dict(kind='joint', name='home')]

    def _record(self, quiet=False):
        o, r = self.state['SugarJar']['p'], self.state['SugarJar']['r']
        tcp, rr = self.servos['Left'].pose()
        self.left.grasp = (r.inv().apply(tcp-o), r.inv()*rr)
        if not quiet:
            self.event('grasp', self.t, side='Left', jaw_N=self.state['jar_jaw_forces'])

    # ---------------------------------------------------------------- control
    def transition(self, side, phase, t):
        self.phase[side] = phase
        self.phase_time[side] = t
        self.event('phase', t, side=side, phase=phase)

    def remember(self, side, obj):
        tcp, rot = self.servos[side].pose()
        p, r = self.state[obj]['p'], self.state[obj]['r']
        self.held[side] = (r.inv().apply(tcp-p), r.inv()*rot)

    def desired_tool(self, side, p, r):
        """TCP pose that puts the held spoon at (p, r), using the measured grasp offset."""
        m = self.state['Spoon']
        tcp, rot = self.servos[side].pose()
        off = m['r'].inv().apply(tcp-m['p'])
        return p+r.apply(off), r*(m['r'].inv()*rot)

    @property
    def done(self):
        return all(v == 'done' for v in self.phase.values())

    def command(self, t):
        self._left(t)
        self._right(t)

    def _left(self, t):
        arm, servo = self.left, self.servos['Left']
        if not arm.planned:
            self.plan_left()
            arm.planned = True
        if not arm.queue:
            servo.move_joints(PARK, rate=1.)
            servo.grip(True)
            return
        s = arm.queue[0]
        kind, name = s['kind'], s['name']
        if arm.phase != name:
            arm.phase, arm.phase_time = name, t
            self.transition('Left', name, t)
        el = t-arm.phase_time
        advance, grip = False, 'hold'
        tcp, rr = servo.pose()
        if kind == 'jpose':
            if not arm.jplan:
                p, r = s['pos'], s['rot']
                for up in (.12, .09, .06):
                    try:
                        q_hi = self.plan_joints('Left', p+[0, 0, up], r)
                        break
                    except RuntimeError:
                        pass
                else:
                    q_hi = self.plan_joints('Left', p, r)
                qf = self.plan_joints('Left', p, r, seed=q_hi)
                tuck = PARK.copy()
                tuck[0] = q_hi[0]
                arm.jplan = [tuck, q_hi, qf]
            servo.move_joints(arm.jplan[0], rate=1.2)
            if np.max(np.abs(servo.joints()-arm.jplan[0])) < .02:
                arm.jplan.pop(0)
            advance, grip = not arm.jplan, 'wide'
        elif kind == 'pose':
            if s['pos'] is None:  # retreat: back off along the tool axis
                s['pos'], s['rot'] = tcp-rr.apply([.08, 0, 0])+[0, 0, .03], rr
            servo.move(s['pos'], s['rot'], speed=s['speed'], turn_rate=1.)
            ok = np.linalg.norm(tcp-s['pos']) < s['tol'] and (s['rot']*rr.inv()).magnitude() < .05
            advance, grip = ok, 'wide'
        elif kind == 'close':
            fp, fn = self.state['jar_jaw_forces']
            tp, tn = arm.jaw_travel or [(P.JAW_OPENING-.035)/2]*2
            if fp > .5 and fn > .5:
                tp, tn = min(P.JAW_TRAVEL+.025, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.025, tn+servo.squeeze_step)
            else:
                tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
                tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
            arm.jaw_travel = [tp, tn]
            servo.grip_each(tp, tn)
            servo.move(s['pos'], s['rot'], speed=.01)
            advance, grip = fp > 12. and fn > 12. and el > .3, 'keep'
        elif kind == 'call':
            s['fn']()
            advance, grip = True, 'keep' if name == 'grasped' else 'wide'
        elif kind == 'jar':
            if el == 0. and name != 'lift':  # the jar turns a little in the pinch while pouring: re-measure
                self._record(quiet=True)
            o, r = s['pose']()
            goal, rg = self.tcp_for(o, r)
            d, ang = np.linalg.norm(goal-tcp), (rg*rr.inv()).magnitude()
            speed = max(.004, min(s['speed'], ACCEL*el, np.sqrt(2*ACCEL*d)))
            turn = max(.02, min(s.get('turn', .8), ANG_ACCEL*el, np.sqrt(2*ANG_ACCEL*ang)))
            servo.move(goal, rg, speed=speed, turn_rate=turn)
            advance = d < s['tol'] and ang < .04
        elif kind == 'pour':
            advance = self._pour(t, el)
        elif kind == 'release':  # jaws fully open: the 54 mm jar needs the full opening
            servo.move(tcp, rr, speed=.01)
            advance, grip = el > .8, 'wide'
        elif kind == 'joint':
            servo.move_joints(PARK, rate=1.)
            advance, grip = np.max(np.abs(servo.joints()-PARK)) < .03, 'close'
        if grip == 'wide':
            servo.grip(False)
        elif grip == 'hold':
            servo.grip(True, squeeze=.025)
        elif grip == 'close':
            servo.grip(True)
        if advance:
            arm.queue.pop(0)
            if not arm.queue:
                self.transition('Left', 'done', t)
        elif el > (40. if kind == 'pour' else 25.):
            self.failures.append(f'Left:step_timeout:{name}')

    def _pour(self, t, el):
        """Tilt until sugar runs, servo the tilt to Q_TARGET of flow, tip back once the glass, the sugar in the air
        and the expected tail reach the middle of the band."""
        arm, servo, c = self.left, self.servos['Left'], self.task.cues()
        if el == 0.:
            arm.tilt_cmd, arm.tilt_rate, arm.stopping, arm.flowing = self.state['tilt'], 0., False, False
        q = c['flow_g_s']
        arm.flowing |= q > .3
        expected = c['sugar_g']+c['in_air_g']+q*TAIL_S
        if not arm.stopping and arm.flowing and expected >= self.target_g:
            arm.stopping = True
            self.event('stop_pour', t, sugar_g=c['sugar_g'], in_air_g=c['in_air_g'], flow_g_s=q, tilt_deg=self.state['tilt'])
        if arm.stopping:
            want = -min(TILT_BACK, np.sqrt(2*TILT_ACC*max(0., arm.tilt_cmd)))
        elif not arm.flowing:
            want = TILT_FAST
        else:
            want = float(np.clip(8.*(Q_TARGET-q), -15., 15.))
        dt = servo.dt
        arm.tilt_rate += float(np.clip(want-arm.tilt_rate, -TILT_ACC*dt, TILT_ACC*dt))
        arm.tilt_cmd = min(MAX_TILT, max(0., arm.tilt_cmd+arm.tilt_rate*dt))
        o, r = self.pour_pose(arm.tilt_cmd)
        goal_p, rg = self.tcp_for(o, r)
        servo.move(goal_p, rg, speed=.12, turn_rate=1.8)
        return arm.stopping and arm.tilt_cmd <= 0. and q < .05 and c['in_air_g'] < .05

    def _plan_spoon_grasp(self, servo, p, r):
        scored = []
        seed = servo.joints()
        stir_center = np.array([CUP[0]+.010, CUP[1], H+.17138])
        for phi, flip in [(phi, flip) for phi in [0., 90., 180., 270.] for flip in (False, True)]:
            rad = np.deg2rad(phi)
            a = np.array([0, np.cos(rad), np.sin(rad)])
            c = np.array([-1, 0, 0])
            relative = R.from_matrix(np.column_stack((a, np.cross(c, a), c)))
            if flip:  # symmetric jaws: same grasp rolled 180 deg about the approach axis (joint6 + pi)
                relative = relative*R.from_euler('x', 180, degrees=True)
            candidate = r*relative
            local = np.array([-.060, 0, -.0034])+relative.apply([.00593, 0, 0])
            grip = p+r.apply(local)
            upright = R.from_euler('z', np.pi-np.arctan2(np.cos(rad), np.sin(rad)))*R.from_euler('y', 90, degrees=True)
            q, pe, re = servo.kin.solve(grip, candidate, seed)
            # pre-grasp beyond the handle end, on the handle axis: the fingers slide onto the flat
            # handle in its own plane (a vertical descent sweeps a finger through the tilted handle)
            qh, ph, rh = servo.kin.solve(p+r.apply(local+[-PREGRASP, 0, 0]), candidate, q)
            qs, ps, rs = servo.kin.solve(stir_center+upright.apply(local), upright*relative, q)
            # keep clear of joint limits (joint6 is only +-2 rad): a wrist at its stop swings the tool
            margin = min(float(np.min(np.minimum(x-LOW, HIGH-x))) for x in (q, qh, qs))
            error = pe*10+re+ph*10+rh+ps*10+rs
            scored.append((error+max(0., .3-margin)*2, error, candidate, local, upright, qh))
        _, error, self.right_pick, self.spoon_grip_local, self.spoon_upright, self.right_pregrasp = min(scored, key=lambda item: item[0])
        if error > .15:
            raise RuntimeError('No feasible ordinary spoon-handle grasp and stir orientation')

    def _right(self, t):
        st, side, obj = self.state, 'Right', 'Spoon'
        servo = self.servos[side]
        phase = self.phase[side]
        elapsed = t-self.phase_time[side]
        p, r = st[obj]['p'], st[obj]['r']
        tcp, rr = servo.pose()
        close, speed = False, .24
        if self.right_pick is None:
            self._plan_spoon_grasp(servo, p, r)
        rot = self.right_pick
        goal = tcp.copy()
        grip = p+r.apply(self.spoon_grip_local)
        above = np.array([CUP[0]+.010, CUP[1], CUP_RIM+.030+.060])
        if phase == 'approach':  # joint-space move to the planned pre-grasp configuration
            servo.move_joints(self.right_pregrasp)
            servo.grip(False, opening=.020)
            goal = p+r.apply(self.spoon_grip_local+[-PREGRASP, 0, 0])
            if np.linalg.norm(tcp-goal) < .008 and float((rot*rr.inv()).magnitude()) < .12:
                servo.reset_target()
                self.transition(side, 'descend', t)
            return
        elif phase == 'descend':  # slide along the handle axis onto the handle
            goal, speed = grip, .04
            if np.linalg.norm(tcp-goal) < .0025 and float((rot*rr.inv()).magnitude()) < .10:
                self.transition(side, 'grasp', t)
        elif phase == 'grasp':
            goal, close = grip, True
            if self.grasped[side]:
                self.remember(side, obj)
                self.initial_spoon = (p.copy(), r)
                self.hold[side] = (tcp.copy(), rr)
                self.transition(side, 'lift', t)
        elif phase in ('lift', 'wait'):
            goal, rot = self.hold[side]
            goal, close = goal+[0, 0, .12], True
            if phase == 'lift' and np.linalg.norm(tcp-goal) < .01:
                self.transition(side, 'wait', t)
            if phase == 'wait' and self.clear:
                self.transition(side, 'above', t)
        elif phase == 'above':
            goal, rot = self.desired_tool(side, above, self.spoon_upright)
            close, speed = True, .12
            if np.linalg.norm(tcp-goal) < .006 and float((rot*rr.inv()).magnitude()) < .15 and st['head_min_z'] > CUP_RIM+.015:
                self.transition(side, 'lower', t)
        elif phase == 'lower':
            goal, rot = self.desired_tool(side, np.array([CUP[0]+.010, CUP[1], H+.035+.060]), self.spoon_upright)
            close, speed = True, .06
            if st['head_max_z'] < TEA_Z and np.linalg.norm(tcp-goal) < .006:
                self.angle = 0.
                self.transition(side, 'stir', t)
        elif phase == 'stir':
            if servo.error[0] < .004:
                self.angle -= .9*servo.dt
            target = np.array([CUP[0]+.010*np.cos(self.angle), CUP[1]+.010*np.sin(self.angle), H+.035+.060])
            goal, rot = self.desired_tool(side, target, self.spoon_upright)
            close, speed = True, .06
            if self.turns >= self.task.spec.required_turns+.05:
                self.event('measured_stir_complete', t, clockwise_turns=self.turns)
                self.transition(side, 'withdraw', t)
        elif phase == 'withdraw':
            goal, rot = self.desired_tool(side, above, self.spoon_upright)
            close, speed = True, .08
            if st['head_min_z'] > CUP_RIM+.015:
                self.transition(side, 'return', t)
        elif phase == 'return':
            target, toolrot = self.initial_spoon
            goal, rot = self.desired_tool(side, target+[0, 0, .12], toolrot)
            close, speed = True, .16
            if np.linalg.norm(tcp-goal) < .008 and float((rot*rr.inv()).magnitude()) < .15:
                self.transition(side, 'place', t)
        elif phase == 'place':
            target, toolrot = self.initial_spoon
            goal, rot = self.desired_tool(side, target+[0, 0, -.0006], toolrot)
            close, speed = True, .06
            if self.support['Spoon'] or np.linalg.norm(p-target) < .008:
                self.hold[side] = (tcp.copy(), rr)
                self.transition(side, 'release_ready', t)
        elif phase == 'release_ready':
            goal, rot = self.hold[side]
            close = True
            if elapsed > .3:
                self.transition(side, 'release', t)
        elif phase == 'release':
            goal, rot = self.hold[side]
            if elapsed > .3:
                if self.support['Spoon'] and max(st['spoon_jaw_forces']) < .03:
                    self.hold[side] = (tcp.copy(), rr)
                    self.transition(side, 'retreat', t)
                else:  # a finger hooked on the scanned handle: slide sideways off it
                    axis = rot.apply([0, 1, 0])
                    axis[2] = 0.
                    axis /= max(np.linalg.norm(axis), 1e-6)
                    direction = -axis if st['spoon_jaw_forces'][0] >= st['spoon_jaw_forces'][1] else axis
                    self.unhook[side] = (goal.copy(), rot, direction)
                    self.transition(side, 'unhook', t)
        elif phase in ('unhook', 'unhook_clear'):
            anchor, rot, direction = self.unhook[side]
            goal = anchor+direction*(.035 if phase == 'unhook' else .075)+[0, 0, -.004]
            speed = .18
            if self.support['Spoon'] and max(st['spoon_jaw_forces']) < .03:
                self.hold[side] = (tcp.copy(), rr)
                self.transition(side, 'retreat', t)
            elif phase == 'unhook' and np.linalg.norm(tcp-goal) < .005 and elapsed > .5:
                self.transition(side, 'unhook_clear', t)
        elif phase == 'retreat':  # slowly off the handle along the tool axis: one pad lies under it, and
            # the wrist is near its stops here, so at speed the clipped joints swing the hand up and fling the spoon
            goal, rot = self.hold[side]
            goal, speed = goal-rot.apply([.025, 0, 0]), .03  # 2.5 cm clears the pads; joint 5 stops at 2.9 cm
            if np.linalg.norm(tcp-goal) < .004:
                self.transition(side, 'home', t)
        elif phase in ('home', 'done'):
            servo.move_joints(np.array(READY_Q))
            servo.grip(False, opening=.030)
            if phase == 'home' and np.max(np.abs(servo.joints()-READY_Q)) < .03:
                self.transition(side, 'done', t)
            return
        if side in self.held and phase not in ('grasp', 'release', 'unhook', 'unhook_clear', 'retreat', 'done'):
            off, _ = self.held[side]
            if np.linalg.norm(p+r.apply(off)-tcp) > .035 and elapsed > .5:
                self.failures.append(f'{side}:lost_tool_retention:{phase}')
        servo.move(goal, rot, speed=speed)
        servo.grip(close, opening=.020 if phase in ('approach', 'descend') else .030)
        if elapsed > 14. and phase not in ('wait', 'stir', 'done'):
            self.failures.append(f'{side}:phase_timeout:{phase}')

    def trace(self):
        return dict(self.row, phase=dict(self.phase), tilt_cmd=self.left.tilt_cmd)
