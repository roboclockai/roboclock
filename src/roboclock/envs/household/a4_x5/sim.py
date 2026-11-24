"""A4 physics episode (R2.1): two-arm kettle pour onto a scale, measured signals, cues, oracle.

A full kettle (0.8 kg shell + up to 1.1 L water) is held by both arms, because a single X5 pinch on
its handle cannot resist the ~1.5-2 N m gravity torque: the left arm pinches the handle bar across x
with one pad through the handle opening, the right arm pinches the lid knob (tool from the front-right,
20 deg down). Both tools come from the front, so tilting the kettle about y toward the spout is mostly a
wrist roll for both arms. The arms approach one after the other: their palms would meet mid-air.
Water is a scalar model (task model): the kettle body mass follows the remaining water, the stream,
jug water level and the scale's 7-segment reading follow the task cues.

Oracle (privileged, contact-driven, no object pose writes): both arms grasp, lift the kettle, bring
the spout above the jug, tilt to ~25 deg past the pour onset, return upright early enough that the
tail flow ends near 300 g (privileged dose and remaining volume), carry back, set the kettle down
upright and release.
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.a4_x5.scene import DIGIT_X, JUG_XY, SEGMENTS, segments_on
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW
from roboclock.envs.household.x5.task_base import place

CONTROL = dict(left='ik', right='ik')
KETTLE = '/World/Objects/Kettle'
SHELL_KG = .8
# kettle frame: origin on the body axis at the base, spout toward +x (as placed)
HOLE = np.array([-.108, -.0025, .1366])   # left grasp: handle bar, one pad through the opening
KNOB = np.array([-.0025, 0., .2216])      # right grasp: lid knob centre (the pads close 1.25 cm in front of it)
SPOUT = np.array([.089, 0., .172])        # spout tip
JUG_R, JUG_RIM = .0465, H+.0225+.135
# 2026-11-04 speed-up (hard budget then 45 / 1.5 = 30 s, 36 s since 2026-11-16; the oracle set the kettle down at 38.4 s): faster pour and
# return, faster approach/insert/lift/carry, direct set-down. Old values in the comments.
POUR_EXCESS = 32.                         # deg past the onset angle while pouring (~35 ml/s; was 25, ~24 ml/s)
MAX_POUR_TILT = 88.                       # the left elbow keeps its margin up to ~85-90 deg of tilt
TILT_FAST, TILT_SLOW, TILT_BACK = 40., 18., 20.  # deg/s: to the onset, past it, back upright (was 40, 12, 12)
TILT_ACC = 40.                            # deg/s^2 (was 30)
UPRIGHT_HANDOFF = 30.                     # deg below the onset: the pour hands over to the carry (was: at 0 deg). At 15 the faster
                                          # turn during the carry slipped the grasps 4 deg and the retreat tipped the kettle (eval)
JOINT_RATE = 2.5                          # rad/s, approach moves (was 1.2)
INSERT_SPEED, LIFT_SPEED, CARRY_SPEED, SET_SPEED = .06, .10, .12, .04  # m/s (was .03, .06, .08, .04)
# spout-tip path: upright with the base 5 cm above the jug rim, then down to 3 cm above the rim while tilting;
# the pour point sits 2.5 cm short of the jug axis so the left elbow keeps ~1 rad margin at 85 deg tilt
PRE_TIP = np.array([JUG_XY[0]-.03, JUG_XY[1], JUG_RIM+.22])
POUR_TIP = np.array([JUG_XY[0]-.025, JUG_XY[1], JUG_RIM+.03])
TIP_BLEND = 40.                           # deg of tilt over which the tip moves from PRE_TIP to POUR_TIP
TOOL = {'Left': (90., 0.), 'Right': (110., -20.)}  # tool azimuth/elevation, jaw axis along x (roll -90)
ARMS = ('Left', 'Right')
JAWS = {'Left': ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN'], 'Right': ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN']}
PARK = {'Left': np.array(READY_Q), 'Right': np.array(READY_Q)}
ACCEL, ANG_ACCEL = .35, 1.2  # kettle carry ramps (was .25, 1.)
G = {}


def tool(az, el, roll=-90.):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))*R.from_euler('x', roll, degrees=True)


def smooth(u):
    u = min(1., max(0., u))
    return u*u*(3-2*u)


def _local_mesh(stage, path, world_pts, inv, mat, faces=None):
    m = UsdGeom.Mesh.Define(stage, path)
    pts = (np.c_[world_pts, np.ones(len(world_pts))]@inv)[:, :3]
    m.CreatePointsAttr(pts.tolist())
    if faces is None:
        m.CreateFaceVertexCountsAttr([3])
        m.CreateFaceVertexIndicesAttr([0, 1, 2])
    else:
        m.CreateFaceVertexCountsAttr([len(f) for f in faces])
        m.CreateFaceVertexIndicesAttr(sum(faces, []))
    m.CreateVisibilityAttr('invisible')
    P.collision(m.GetPrim(), mat)


def _box_pts(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


BOX_FACES = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]


def build(stage, mat):
    s = stage
    kettle = s.GetPrimAtPath(KETTLE)
    cache = UsdGeom.XformCache()
    k2w = np.array(cache.GetLocalToWorldTransform(kettle))
    inv = np.linalg.inv(k2w)
    mesh = [c for c in Usd.PrimRange(kettle) if c.IsA(UsdGeom.Mesh)][0]
    m2w = np.array(cache.GetLocalToWorldTransform(mesh))
    v = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), float)
    w = (np.c_[v, np.ones(len(v))]@m2w)[:, :3]
    lo = w.min(0)
    origin = np.array([lo[0]+.1189, (w[:, 1].min()+w[:, 1].max())/2, H])  # body axis on the base
    G.update(origin=origin)
    o = origin
    # body (with spout) below the lid top; lid knob box; handle bar + connectors as boxes (opening left free)
    body = w[(w[:, 0] > o[0]-.0735) & (w[:, 2] < H+.2021)]
    _local_mesh(s, KETTLE+'/BodyCollider', body[::max(1, len(body)//3000)], inv, mat)
    rel = {'KnobCollider': ((-.015, -.025, .2016), (.010, .025, .2296)),
           'HandleBar': ((-.1189, -.015, .0716), (-.0969, .015, .1866)),
           'HandleLower': ((-.1189, -.015, .0316), (-.0730, .015, .1216)),
           'HandleUpper': ((-.1150, -.015, .1516), (-.0730, .015, .1866))}
    for name, (a, b) in rel.items():
        _local_mesh(s, f'{KETTLE}/{name}', _box_pts(o+a, o+b), inv, mat, BOX_FACES)
    P.body(kettle, SHELL_KG+1.)
    jx, jy = JUG_XY
    jug = UsdGeom.Cylinder.Define(s, '/World/Fixtures/JugCollider')
    jug.CreateRadiusAttr(.05)
    jug.CreateHeightAttr(.135)
    jug.CreateAxisAttr('Z')
    jug.AddTranslateOp().Set(Gf.Vec3d(jx, jy, H+.0225+.0675))
    jug.CreateVisibilityAttr('invisible')
    P.collision(jug.GetPrim(), mat)
    P.collision(s.GetPrimAtPath('/World/Objects/KitchenScale'), mat)
    # cues: water in the jug and the pouring stream
    wm = UsdShade.Material.Define(s, '/World/Looks/PouredWater')
    sh = UsdShade.Shader.Define(s, '/World/Looks/PouredWater/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    from pxr import Sdf
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.55, .72, .85))
    sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(.55)
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.05)
    wm.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    for name in ('JugWater', 'WaterStream'):
        c = UsdGeom.Cylinder.Define(s, '/World/Objects/'+name)
        c.CreateRadiusAttr(.045 if name == 'JugWater' else .004)
        c.CreateHeightAttr(.001)
        c.CreateAxisAttr('Z')
        c.AddTranslateOp().Set(Gf.Vec3d(jx, jy, H+.0285))
        c.CreateVisibilityAttr('invisible')
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(wm)
    return dict(kettle=f'rigid kettle ({SHELL_KG} kg shell + water mass following the task model); body hull below the lid, '
                       'lid-knob box, handle bar/connector boxes leaving the handle opening free',
                jug='static fixture on the scale (cylinder collider)', water='symbolic flow model; stream, jug level and 7-segment scale reading are cues',
                grasp='left: handle bar pinched across x with one pad through the opening; right: lid knob pinched across x; '
                      'both tools point +y, so tilting toward the spout is a wrist roll')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t, self.idle = [], 0., False
        self.phase, self.phase_time = 'idle', 0.
        self.queue, self.planned = [], False
        self.jplan, self.hold_q = {}, {}
        self.jaw_travel = {}
        self.grasp = {}
        self.mass_t = -1.
        self.tilt_cmd, self.tilt_rate, self.tip_bias = 0., 0., np.zeros(3)

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        off = self.task.spec.offset('kettle')  # the seed's layout, before the reset: the kettle turns about its body axis
        P.shift(stage.GetPrimAtPath(KETTLE), off, G['origin'])
        G['origin'] = place(G['origin'], off, G['origin'])
        G['yaw0'] = R.from_euler('z', off[2], degrees=True)  # the kettle frame at rest (identity at the nominal layout)
        self.kettle = world.scene.add(RigidPrim(KETTLE, name='kettle', reset_xform_properties=False, track_contact_forces=True,
                                                contact_filter_prim_paths_expr=[JAWS['Left']+JAWS['Right']], max_contact_count=512))
        self.stage = stage
        self.jug_water = UsdGeom.Cylinder(stage.GetPrimAtPath('/World/Objects/JugWater'))
        self.stream = UsdGeom.Cylinder(stage.GetPrimAtPath('/World/Objects/WaterStream'))
        self.digits = [{seg: UsdGeom.Imageable(stage.GetPrimAtPath(f'/World/Objects/ScaleDigit{k}{seg}')) for seg in SEGMENTS}
                       for k in range(len(DIGIT_X))]
        self.shown = None

    def update_cues(self):
        c = self.task.cues()
        reading = c['scale_reading_g']
        if reading != self.shown:
            for k, on in enumerate(segments_on(reading)):
                for seg, img in self.digits[k].items():
                    img.GetVisibilityAttr().Set('inherited' if seg in on else 'invisible')
            self.shown = reading
        jx, jy = JUG_XY
        h = self.task.jug_g/1e6/(np.pi*.045**2)
        self.jug_water.GetVisibilityAttr().Set('inherited' if h > .0005 else 'invisible')
        self.jug_water.GetHeightAttr().Set(float(max(h, .001)))
        self.jug_water.GetOrderedXformOps()[0].Set(Gf.Vec3d(jx, jy, H+.0285+h/2))
        q = c['stream_ml_s']
        if q > .3 and hasattr(self, 'spout'):
            top, bottom = self.spout[2], H+.0285+h
            self.stream.GetVisibilityAttr().Set('inherited')
            self.stream.GetRadiusAttr().Set(float(.0015+.0035*np.sqrt(q/40.)))
            self.stream.GetHeightAttr().Set(float(max(.005, top-bottom)))
            self.stream.GetOrderedXformOps()[0].Set(Gf.Vec3d(float(self.spout[0]), float(self.spout[1]), float((top+bottom)/2)))
        else:
            self.stream.GetVisibilityAttr().Set('invisible')

    def measure(self, t, dt):
        self.t = t
        p, q = self.kettle.get_world_poses()
        self.kp, q = np.asarray(p)[0], np.asarray(q)[0]
        kr = R.from_quat([q[1], q[2], q[3], q[0]])
        if 'r_rest' not in G:  # kettle frame = prim rotation relative to its nominal rest pose (spout +x there)
            G['r_rest'], G['o_local'] = G.get('yaw0', R.identity()).inv()*kr, kr.inv().apply(G['origin']-self.kp)
        self.prim_r = kr
        self.kr = kr*G['r_rest'].inv()
        f = np.linalg.norm(np.asarray(self.kettle.get_contact_force_matrix(dt=dt))[0], axis=-1)
        self.jaw = {'Left': f[:2], 'Right': f[2:]}
        if t-self.mass_t > .5:  # the body mass follows the remaining water
            self.kettle.set_masses(np.array([SHELL_KG+self.task.kettle_ml/1000.]))
            self.mass_t = t
        up, spout_dir = self.kr.apply([0, 0, 1]), self.kr.apply([1, 0, 0])
        self.tilt = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))*(1. if spout_dir[2] <= 0 else -1.)
        o, _ = self.kettle_pose()
        self.spout = o+self.kr.apply(SPOUT)
        jx, jy = JUG_XY
        over = bool(np.hypot(self.spout[0]-jx, self.spout[1]-jy) < JUG_R-.008 and self.spout[2] > JUG_RIM)
        held = bool(f.max() > .5)
        base_z = o[2]
        on_counter = bool(abs(base_z-H) < .006 and abs(self.tilt) < 5.)
        self.row = dict(kettle_pos=self.kp.tolist(), kettle_quat_xyzw=self.kr.as_quat().tolist(), kettle_tilt_deg=self.tilt,
                        spout=self.spout.tolist(), jaw_contact_N={k: v.tolist() for k, v in self.jaw.items()}, kettle_base_z=float(base_z),
                        kettle_ml=self.task.kettle_ml, jug_g=self.task.jug_g, spill_g=self.task.spill_g, scale_reading_g=self.task.scale_reading())
        return dict(kettle_tilt_deg=max(0., self.tilt), spout_over_jug=over, kettle_upright_released=bool(on_counter and not held),
                    jug_on_scale=True, dropped=bool(base_z < H-.03 or abs(self.tilt) > 110))

    # -------------------------------------------------------------- frames
    def kettle_pose(self):
        """Current kettle frame: origin (body axis, base) and rotation (identity at rest)."""
        return self.kp+self.prim_r.apply(G['o_local']), self.kr

    def tcp_for(self, side, o, rot):
        gp, gr = self.grasp[side]
        return o+rot.apply(gp), rot*gr

    def pour_pose(self, tilt):
        """Kettle frame tilted `tilt` deg toward the spout, the spout tip on the PRE_TIP -> POUR_TIP path.

        Upright, the spout reaches only 1.5 cm past the body, so the base must clear the jug rim first."""
        r = R.from_euler('y', tilt, degrees=True)
        u = min(1., max(0., tilt)/TIP_BLEND)
        return PRE_TIP+(POUR_TIP+self.tip_bias-PRE_TIP)*u-r.apply(SPOUT), r

    # --------------------------------------------------------------- planning
    def plan(self):
        o0, r0 = self.kettle_pose()
        self.o0 = o0
        x0 = r0.apply([1., 0., 0.])  # the kettle's measured heading (spout +x at the nominal layout)
        yaw = np.degrees(np.arctan2(x0[1], x0[0]))
        rot = {side: tool(TOOL[side][0]+yaw, TOOL[side][1]) for side in ARMS}
        x = {side: rot[side].apply([1, 0, 0]) for side in ARMS}
        xh = x['Right']*[1, 1, 0]
        grasp_pt = {'Left': o0+r0.apply(HOLE), 'Right': o0+r0.apply(KNOB)-xh/np.linalg.norm(xh)*.0125}
        lift = o0+[0, 0, .10]
        self.queue = [
            dict(kind='jpose', name='approach_left', pos={'Left': grasp_pt['Left']-x['Left']*.06}, rot=rot),
            dict(kind='jpose', name='approach_right', pos={'Right': grasp_pt['Right']-x['Right']*.06}, rot=rot),
            dict(kind='pose', name='insert', pos={s_: grasp_pt[s_] for s_ in ARMS}, rot=rot, speed=INSERT_SPEED, tol=.002),
            dict(kind='close', name='close', pos=grasp_pt, rot=rot),
            dict(kind='call', name='grasped', fn=self._record),
            dict(kind='kettle', name='lift', pose=lambda: (lift, r0), speed=LIFT_SPEED, tol=.008),
            dict(kind='kettle', name='to_jug', pose=lambda: self.pour_pose(0.), speed=CARRY_SPEED, tol=.006),
            dict(kind='pour', name='pour'),
            # once the stream has stopped (tilt UPRIGHT_HANDOFF deg below the onset), the kettle turns the rest of the
            # way upright while it is carried back, to 6 cm above its rest spot (was: upright over the jug, then carried
            # to the 10 cm lift point)
            dict(kind='kettle', name='carry_back', pose=lambda: (o0+[0, 0, .06], R.identity()), speed=CARRY_SPEED, tol=.008, turn=.6),
            dict(kind='kettle', name='set_down', pose=lambda: (o0+[0, 0, .002], R.identity()), speed=SET_SPEED, tol=.003),
            dict(kind='release', name='release'),
            # straight back out of the handle opening first, then up: a combined back-and-up retreat lifted the left pad
            # into the handle's upper arm (1 cm clearance) and tipped the kettle over (spill)
            dict(kind='pose', name='retreat', pos=None, rot=None, speed=.08, tol=.01),
            dict(kind='pose', name='retreat_up', pos=None, rot=None, speed=.08, tol=.01, up=True),
            dict(kind='joint', name='home'),
            dict(kind='call', name='done', fn=lambda: setattr(self, 'done_flag', True))]

    def _record(self):
        o, r = self.kettle_pose()
        for side in ARMS:
            tcp, rr = self.servos[side].pose()
            self.grasp[side] = (r.inv().apply(tcp-o), r.inv()*rr)
        self.event('grasp', self.t, jaw_N={k: v.tolist() for k, v in self.jaw.items()})

    def plan_joints(self, side, pos, rot, seed=None):
        servo = self.servos[side]
        base = servo.kin.base[:3, 3]
        ready = np.array(READY_Q)
        ready[0] = np.arctan2(pos[1]-base[1], pos[0]-base[0])-np.pi/2
        best = None
        for q0 in ((seed,) if seed is not None else (ready, servo.joints())):
            q, pe, re = servo.kin.solve(pos, rot, q0)
            m = float(np.min(np.minimum(q-LOW, HIGH-q)))
            if pe < .002 and re < .02 and (best is None or m > best[1]):
                best = (q, m)
        if best is None:
            raise RuntimeError(f'{side}: pose unreachable {np.round(pos, 3)}')
        return best[0]

    # ---------------------------------------------------------------- control
    @property
    def done(self):
        return getattr(self, 'done_flag', False)

    @property
    def finished(self):
        return self.done and 'dose_final' in self.task.events

    def transition(self, phase, t):
        self.phase, self.phase_time = phase, t
        self.event('phase', t, side='both', phase=phase)

    def command(self, t):
        if not self.planned:
            self.plan()
            self.planned = True
        poses = {side: self.servos[side].pose() for side in ARMS}
        if not self.queue:
            for side in ARMS:
                self.servos[side].move_joints(PARK[side], rate=1.)
                self.servos[side].grip(True)
            return
        s = self.queue[0]
        kind, name = s['kind'], s['name']
        if self.phase != name:
            self.transition(name, t)
        el = t-self.phase_time
        advance, grip = False, 'hold'
        self.idle = False
        if kind == 'jpose':  # one arm at a time (palms would meet): turn the base tucked, unfold above the pre-grasp, descend
            done = True
            for side in ARMS:
                servo = self.servos[side]
                if side not in s['pos']:
                    servo.move_joints(self.hold_q.get(side, PARK[side]), rate=JOINT_RATE)
                    continue
                if side not in self.jplan:
                    p, r = s['pos'][side], s['rot'][side]
                    for up in (.15, .12, .09):  # highest reachable unfold point
                        try:
                            q_hi = self.plan_joints(side, p+[0, 0, up], r)
                            break
                        except RuntimeError:
                            pass
                    else:
                        q_hi = self.plan_joints(side, p+[0, 0, .05], r)
                    q_off = self.plan_joints(side, p+[0, 0, .05], r, seed=q_hi)
                    qf = self.plan_joints(side, p, r, seed=q_off)
                    tuck = np.array(READY_Q)
                    tuck[0] = q_hi[0]
                    self.jplan[side] = [tuck, q_hi, q_off, qf]
                    self.hold_q[side] = qf
                if self.jplan[side]:
                    servo.move_joints(self.jplan[side][0], rate=JOINT_RATE)
                    if np.max(np.abs(servo.joints()-self.jplan[side][0])) < .02:
                        self.jplan[side].pop(0)
                done &= not self.jplan[side]
            advance, grip = done, 'wide'
            if advance:
                self.jplan = {}
        elif kind == 'pose':
            if s['pos'] is None:  # retreat: back off along each tool axis, then up
                if s.get('up'):
                    s['pos'] = {side: poses[side][0]+[0, 0, .04] for side in ARMS}
                else:
                    s['pos'] = {side: poses[side][0]-poses[side][1].apply([.06, 0, 0]) for side in ARMS}
                s['rot'] = {side: poses[side][1] for side in ARMS}
            ok = True
            for side in ARMS:
                tcp, rr = poses[side]
                self.servos[side].move(s['pos'][side], s['rot'][side], speed=s['speed'], turn_rate=1.)
                ok &= np.linalg.norm(tcp-s['pos'][side]) < s['tol'] and (s['rot'][side]*rr.inv()).magnitude() < .05
            advance, grip = ok, 'wide' if name == 'insert' else 'open'
        elif kind == 'close':
            ok = True
            for side in ARMS:
                servo = self.servos[side]
                fp, fn = self.jaw[side]
                tp, tn = self.jaw_travel.get(side, [(P.JAW_OPENING-.035)/2]*2)
                if fp > .5 and fn > .5:
                    tp, tn = min(P.JAW_TRAVEL+.025, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.025, tn+servo.squeeze_step)
                else:
                    tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
                    tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
                self.jaw_travel[side] = [tp, tn]
                servo.grip_each(tp, tn)
                servo.move(s['pos'][side], s['rot'][side], speed=.01)
                ok &= fp > 15. and fn > 15.
            advance, grip = ok and el > .3, 'keep'
        elif kind == 'call':
            s['fn']()
            advance, grip = True, 'keep' if name == 'grasped' else 'hold'
        elif kind == 'kettle':  # both tools follow the kettle target through the recorded grasps
            o, r = s['pose']()
            ok = True
            for side in ARMS:
                tcp, rr = poses[side]
                goal, rg = self.tcp_for(side, o, r)
                d, ang = np.linalg.norm(goal-tcp), (rg*rr.inv()).magnitude()
                speed = max(.004, min(s['speed'], ACCEL*el, np.sqrt(2*ACCEL*d)))
                turn = max(.02, min(s.get('turn', .8), ANG_ACCEL*el, np.sqrt(2*ANG_ACCEL*ang)))
                self.servos[side].move(goal, rg, speed=speed, turn_rate=turn)
                ok &= d < s['tol'] and ang < .04
            advance = ok
        elif kind == 'pour':  # tilt past the onset angle; return early enough for the tail to finish near target
            spec = self.task.spec
            onset = self.task.theta0()
            target = min(onset+POUR_EXCESS, MAX_POUR_TILT)
            # fast to just below the onset, then slowly (fast wrist rolls make the two grasps fight and slip)
            # the tilt rate itself is ramped (TILT_ACC): a sudden reversal pulls the knob out of the right pinch
            if getattr(self, 'stopping', False):
                want = -min(TILT_BACK, np.sqrt(2*TILT_ACC*max(0., self.tilt_cmd)))
            else:
                want = min(TILT_FAST if self.tilt_cmd < onset-8. else TILT_SLOW, np.sqrt(2*TILT_ACC*max(0., target-self.tilt_cmd)))
            dt = self.servos['Left'].dt
            self.tilt_rate += float(np.clip(want-self.tilt_rate, -TILT_ACC*dt, TILT_ACC*dt))
            self.tilt_cmd = min(target+1., max(0., self.tilt_cmd+self.tilt_rate*dt))
            flow = self.task.flow
            # tail estimate: flow ~ excess^1.5 with the excess falling linearly while tilting back (integral 0.4 flow
            # back_s) plus the rate reversal at full flow; the exponential tail is negligible by then
            back_s = (self.tilt-onset)/TILT_BACK
            if not getattr(self, 'stopping', False) and self.task.jug_g+flow*(.4*back_s+.5*TILT_SLOW/TILT_ACC) >= spec.target_g-1.:
                self.stopping = True
                self.event('stop_pour', t, jug_g=self.task.jug_g, flow_ml_s=flow, tilt_deg=self.tilt)
            if self.tilt > TIP_BLEND:  # slow xy integrator on the measured spout: the held kettle sags ~1 cm
                err = (POUR_TIP-self.spout)*[1, 1, 0]
                self.tip_bias = np.clip(self.tip_bias+.5*err*dt, -.02, .02)
            o, r = self.pour_pose(self.tilt_cmd)
            for side in ARMS:
                goal, rg = self.tcp_for(side, o, r)
                self.servos[side].move(goal, rg, speed=.10, turn_rate=1.6)
            advance = getattr(self, 'stopping', False) and self.tilt_cmd <= max(0., onset-UPRIGHT_HANDOFF) and flow == 0.
        elif kind == 'release':
            for side in ARMS:
                self.servos[side].move(*poses[side], speed=.01)
            advance, grip = el > .6, 'open'
        elif kind == 'joint':
            ok = True
            for side in ARMS:
                self.servos[side].move_joints(PARK[side], rate=1.)
                ok &= np.max(np.abs(self.servos[side].joints()-PARK[side])) < .03
            advance, grip = ok, 'close'
        for side in ARMS:
            if grip == 'wide':
                self.servos[side].grip(False, opening=.035)
            elif grip == 'open':
                self.servos[side].grip(False, opening=.035)
            elif grip == 'hold':
                self.servos[side].grip(True, squeeze=.025)
            elif grip == 'close':
                self.servos[side].grip(True)
        if advance:
            self.queue.pop(0)
        elif el > 25.:
            self.failures.append(f'both:step_timeout:{name}')

    def trace(self):
        return dict(self.row, phase=self.phase, tilt_cmd=self.tilt_cmd,
                    jplan={k: [np.round(q, 3).tolist() for q in v] for k, v in self.jplan.items()})
