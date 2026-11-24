"""A3 physics episode (R2.1): offset turner, rigid pancake in the pan, measured signals, oracle.

The frying pan is a static triangle-mesh collider; the pancake is a rigid 45 g disc whose lower edge
is bevelled (a thin blade can wedge under it, as under a real pancake's thin edge); the turner is the
native turner mesh bent at the neck (scene.py), a 60 g rigid body with a thin bevelled blade collider.
The task model integrates the underside browning while the pancake lies on the pan; the underside
colour follows it (visible once the pancake is turned).

Robot (privileged oracle, contact-driven, no object pose writes), left arm only:
grasp the turner along its handle, hold it over the front of the pan until the underside browning
reaches the target, slide the blade under the pancake along a 50 deg approach line (blade pitched just
enough to clear the pan wall), lever the pancake about its left edge to ~85 deg (it stands on that
edge, leaning on the blade), push the blade sideways until the pancake topples past vertical (closed
loop on its measured angle), lift the blade clear and withdraw. A rigid disc cannot be tossed reliably; levering
about an edge that rests on the pan floor is deterministic.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.a3_x5.scene import COOKTOP_XY_SCALE, PAN_XY, PANCAKE_H, PANCAKE_R, TURNER_BEND, TURNER_NECK
from roboclock.envs.household.x5.scene_lib import READY_Q
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
PAN = np.array(PAN_XY)
RIM_R = .18                      # pan rim radius (m)
# pan floor/wall height above the floor vs radius (measured from the scaled pan mesh)
WALL_R = np.array([0., .10, .11, .12, .13, .14, .15, .16, .17, .19])
WALL_DZ = np.array([0., .0014, .0023, .007, .017, .026, .0345, .0394, .0411, .0411])
CAKE_MASS, TURNER_MASS = .045, .06
BLADE_T = .004                   # blade collider thickness
TARGET_DONENESS = .63            # flip when the underside reaches this browning (band .55-.75)
LEVER_DEG = 85.
APPROACH_YAW = 50.               # turner azimuth into the pan (left arm reach); the pancake is turned toward its left
LINE_OFFSET = .02                # slide line right of the pancake centre: the lever arm keeps the hand high
PALE, GOLD, DARK = np.array([.86, .70, .42]), np.array([.78, .50, .20]), np.array([.22, .11, .04])
JAWS = ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN']
TURNER = '/World/Objects/Spatula'
CAKE = '/World/Objects/Pancake'
PARK = {'Left': np.array(READY_Q)+[.8, 0, 0, 0, 0, 0], 'Right': np.array(READY_Q)}
ACCEL, ANG_ACCEL = .3, 1.5
G = {}


def tool(az, el, roll=0.):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))*R.from_euler('x', roll, degrees=True)


def smooth(u):
    u = min(1., max(0., u))
    return u*u*(3-2*u)


def build(stage, mat):
    s = stage
    floor_z = None
    for p in Usd.PrimRange(s.GetPrimAtPath('/World/Objects/FryingPan')):
        if p.IsA(UsdGeom.Mesh):
            P.collision(p, mat, approximation='none')  # static triangle mesh: the pan stays on the burner
    P.box(s, '/World/Fixtures/CooktopTop', [.30, .10, .9037-.02], [.83*COOKTOP_XY_SCALE, .55*COOKTOP_XY_SCALE, .04], mat)
    # a cooking pancake grips the pan (mu 0.8); the oiled turner blade is slicker (mu 0.4, min-combined)
    food = UsdShade.Material(s.DefinePrim('/World/Physics/FoodMat', 'Material'))
    fm = UsdPhysics.MaterialAPI.Apply(food.GetPrim())
    fm.CreateStaticFrictionAttr(.8)
    fm.CreateDynamicFrictionAttr(.75)
    fm.CreateRestitutionAttr(0.)
    blade_mat = UsdShade.Material(s.DefinePrim('/World/Physics/BladeMat', 'Material'))
    bmat = UsdPhysics.MaterialAPI.Apply(blade_mat.GetPrim())
    bmat.CreateStaticFrictionAttr(.4)
    bmat.CreateDynamicFrictionAttr(.36)
    bmat.CreateRestitutionAttr(0.)
    PhysxSchema.PhysxMaterialAPI.Apply(blade_mat.GetPrim()).CreateFrictionCombineModeAttr('min')
    # pancake: rigid disc with a bevelled lower edge (the visual body/top/underside ride on the rigid Xform)
    cake = s.GetPrimAtPath(CAKE)
    floor_z = float(UsdGeom.Xformable(cake).GetOrderedXformOps()[0].Get()[2])
    # thick backing under the flat floor: the triangle-mesh floor has no thickness and can be tunnelled
    floor = UsdGeom.Cylinder.Define(s, '/World/Fixtures/PanFloor')
    floor.CreateRadiusAttr(.105)
    floor.CreateHeightAttr(.012)
    floor.CreateAxisAttr('Z')
    floor.AddTranslateOp().Set(Gf.Vec3d(float(PAN[0]), float(PAN[1]), floor_z-.006-.0003))
    floor.CreateVisibilityAttr('invisible')
    P.collision(floor.GetPrim(), mat)
    ang = np.linspace(0, 2*np.pi, 24, endpoint=False)
    ring = np.c_[np.cos(ang), np.sin(ang)]
    # thin rounded edge: the underside rises ~10 deg over the outer 14 mm, as a real pancake thins at its rim
    rings = [(PANCAKE_R-.016, 0.), (PANCAKE_R-.002, .0025), (PANCAKE_R, .004), (PANCAKE_R-.004, PANCAKE_H)]
    pts = [(*(r*c), z) for r, z in rings for c in ring]
    faces = [list(range(23, -1, -1))]+[[24*k+j, 24*k+(j+1) % 24, 24*(k+1)+(j+1) % 24, 24*(k+1)+j] for k in range(3) for j in range(24)] + \
        [list(range(72, 96))]
    hull = UsdGeom.Mesh.Define(s, CAKE+'/Collider')
    hull.CreatePointsAttr(pts)
    hull.CreateFaceVertexCountsAttr([len(f) for f in faces])
    hull.CreateFaceVertexIndicesAttr(sum(faces, []))
    hull.CreateVisibilityAttr('invisible')
    P.collision(hull.GetPrim(), food)
    P.body(cake, CAKE_MASS)
    # turner: origin at the neck (scene), blade toward +x; thin bevelled blade plate + handle hull
    turner = s.GetPrimAtPath(TURNER)
    mesh = UsdGeom.Mesh([c for c in Usd.PrimRange(turner) if c.IsA(UsdGeom.Mesh)][0])
    v = np.asarray(mesh.GetPointsAttr().Get(), float)  # turner-local (Source has no transform)
    blade = v[v[:, 0] > .004]
    tip = float(blade[:, 0].max())
    half_w = float(np.percentile(np.abs(blade[:, 1]), 99))
    zb = float(blade[:, 2].min())
    plate = [(x, y, zb) for x in (.004, tip) for y in (-half_w, half_w)]+[(x, y, zb+BLADE_T) for x in (.004, tip-.006) for y in (-half_w, half_w)]
    bm = UsdGeom.Mesh.Define(s, TURNER+'/BladeCollider')
    bm.CreatePointsAttr(plate)
    bm.CreateFaceVertexCountsAttr([4]*6)
    bm.CreateFaceVertexIndicesAttr([0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3])
    bm.CreateVisibilityAttr('invisible')
    P.collision(bm.GetPrim(), blade_mat)
    handle = v[v[:, 0] < -.004]
    hm = UsdGeom.Mesh.Define(s, TURNER+'/HandleCollider')
    hm.CreatePointsAttr(handle[::max(1, len(handle)//2000)].tolist())
    hm.CreateFaceVertexCountsAttr([3])
    hm.CreateFaceVertexIndicesAttr([0, 1, 2])
    hm.CreateVisibilityAttr('invisible')
    P.collision(hm.GetPrim(), mat)
    P.body(turner, TURNER_MASS, com=[tip/2, 0., zb+BLADE_T/2])  # metal blade, light hollow handle
    G.update(floor_z=floor_z, tip=tip, blade_bottom=zb, half_w=half_w)
    return dict(pan='static triangle-mesh collider (pan stays on the burner); pan scaled to a 28 cm class pan',
                pancake=f'rigid {CAKE_MASS*1000:.0f} g disc, {PANCAKE_R*200:.1f} cm, thin rounded rim (underside rises ~10 deg over 14 mm); '
                        'mu 0.8 on the pan, 0.4 on the oiled blade; underside colour follows the browning',
                turner=f'native turner bent {TURNER_BEND:g} deg at the neck; {TURNER_MASS*1000:.0f} g, COM at the blade; '
                       f'{BLADE_T*1000:.0f} mm bevelled blade plate collider + handle hull',
                cooking='symbolic (task model): underside browning accrues while the pancake lies on the pan')


def wall_dz(xy):
    return float(np.interp(np.hypot(xy[0]-PAN[0], xy[1]-PAN[1]), WALL_R, WALL_DZ))


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.queue = []
        self.phase, self.phase_time = 'idle', 0.
        self.failures, self.t, self.idle = [], 0., False
        self.jplan, self.jaw_travel, self.grasp = None, None, None
        self.flip_max = 0.
        self.planned = False
        self.lever0, self.lever_pose, self.topple_dx = None, None, 0.

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        spec = self.task.spec  # the seed's layout, before the reset
        P.shift(stage.GetPrimAtPath(TURNER), spec.offset('turner'), TURNER_NECK)
        self.cake = world.scene.add(RigidPrim(CAKE, name='cake', reset_xform_properties=False, track_contact_forces=True,
                                              contact_filter_prim_paths_expr=[[TURNER]], max_contact_count=256))
        self.turner = world.scene.add(RigidPrim(TURNER, name='turner', reset_xform_properties=False, track_contact_forces=True,
                                                contact_filter_prim_paths_expr=[JAWS], max_contact_count=256))
        mat = UsdShade.Material(stage.GetPrimAtPath('/World/Looks/PancakeUnderside'))
        self.under = UsdShade.Shader(stage.GetPrimAtPath(str(mat.GetPath())+'/Surface')).GetInput('diffuseColor')

    def update_cues(self):
        d = self.task.cues()['bottom_doneness']
        c = PALE+(GOLD-PALE)*min(1., d/.65) if d < .65 else GOLD+(DARK-GOLD)*min(1., (d-.65)/.3)
        self.under.Set(Gf.Vec3f(*map(float, c)))

    def measure(self, t, dt):
        self.t = t
        p, q = self.cake.get_world_poses()
        self.cp, q = np.asarray(p)[0], np.asarray(q)[0]
        self.cr = R.from_quat([q[1], q[2], q[3], q[0]])
        up = self.cr.apply([0, 0, 1])
        self.flip = float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))
        self.flip_max = max(self.flip_max, self.flip)
        p, q = self.turner.get_world_poses()
        self.tp, q = np.asarray(p)[0], np.asarray(q)[0]
        self.tr = R.from_quat([q[1], q[2], q[3], q[0]])
        self.touch = float(np.linalg.norm(np.asarray(self.cake.get_contact_force_matrix(dt=dt))[0], axis=-1).max())
        self.jaw = np.linalg.norm(np.asarray(self.turner.get_contact_force_matrix(dt=dt))[0], axis=-1)
        v = np.asarray(self.cake.get_linear_velocities())[0]
        r = float(np.hypot(self.cp[0]-PAN[0], self.cp[1]-PAN[1]))
        fz = G['floor_z']
        on_pan = bool(abs(up[2]) > .97 and self.cp[2] < fz+wall_dz(self.cp)+.012 and self.touch < .05)
        blade_tip = self.tp+self.tr.apply([G['tip'], 0, 0])
        clear = bool(np.hypot(blade_tip[0]-PAN[0], blade_tip[1]-PAN[1]) > RIM_R+.03 or blade_tip[2] > fz+.12)
        self.row = dict(cake_pos=self.cp.tolist(), cake_quat_xyzw=self.cr.as_quat().tolist(), cake_flip_deg=self.flip,
                        cake_turner_contact_N=self.touch, turner_pos=self.tp.tolist(), turner_quat_xyzw=self.tr.as_quat().tolist(),
                        jaw_contact_N=self.jaw.tolist(), blade_tip=blade_tip.tolist())
        return dict(pancake_on_pan=on_pan,
                    pancake_rotation_deg=self.flip,
                    pancake_in_pan_released=bool(r+PANCAKE_R < RIM_R and self.touch < .05 and np.linalg.norm(v) < .02),
                    turner_clear=clear, dropped=bool(r > RIM_R+.02 or self.cp[2] < fz-.03), pan_moved=False)

    # ------------------------------------------------------------- turner frame
    def turner_target(self, n, rot):
        """TCP that puts the turner neck at n with orientation rot (recorded grasp transform)."""
        gp, gr = self.grasp
        return n+rot.apply(gp), rot*gr

    def pan_pose(self, s_tip, press=0.):
        """Turner along the approach line through the pancake centre, blade tip bottom on the floor at
        signed distance s_tip from the centre, pitched tip-down just enough to clear the pan wall."""
        fz, d = G['floor_z'], self.d
        tip_xy = self.cake0[:2]-self.left*LINE_OFFSET+d*s_tip
        tip_dz = wall_dz(tip_xy)  # the tip follows the local floor height
        beta = 0.
        for s in np.linspace(.01, G['tip'], 15):  # pitch only where the wall rises behind the tip
            dz = wall_dz(tip_xy-d*s)-tip_dz
            beta = max(beta, np.arctan2(dz+.002, s) if dz > .001 else 0.)
        beta += np.radians(3.) if press else 0.  # tip leads slightly down while sliding
        rot = R.from_euler('z', APPROACH_YAW, degrees=True)*R.from_euler('y', np.degrees(beta), degrees=True)
        n = np.array([*tip_xy, fz+tip_dz+.001-press])-rot.apply([G['tip'], 0, G['blade_bottom']])
        return n, rot

    # --------------------------------------------------------------- planning
    def plan(self):
        g, ln = np.radians(TURNER_BEND), .02
        rho = ln/g
        sg = .14  # grasp 14 cm along the handle centre line from the neck: the hand stays outside the pan rim
        grip_local = np.array([-(rho*np.sin(g)+(sg-ln)*np.cos(g)), 0., rho*(1-np.cos(g))+(sg-ln)*np.sin(g)])
        n0, r0 = self.tp.copy(), self.tr
        self.neck0 = n0  # measured rest of the turner neck (the via point is relative to it)
        grip = n0+r0.apply(grip_local)
        x0 = r0.apply([1., 0., 0.])  # measured: the turner's heading (TURNER_YAW plus the seed's turn)
        rt = tool(np.degrees(np.arctan2(x0[1], x0[0])), -TURNER_BEND)
        x = rt.apply([1, 0, 0])
        self.cake0 = self.cp.copy()
        a = np.radians(APPROACH_YAW)
        self.d, self.left = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        y0, y1 = -(PANCAKE_R+.012), PANCAKE_R+.012  # blade tip from just in front of the pancake to just past it
        self.slide_end = y1
        lift = (n0+[0, 0, .10], r0)
        hover = lambda: (self.pan_pose(y0)[0]+[0, 0, .05], self.pan_pose(y0)[1])  # noqa: E731
        park = (np.array([.16, -.20, G['floor_z']+.10]), R.from_euler('z', APPROACH_YAW, degrees=True))
        self.queue = [
            dict(kind='jpose', name='pre_grasp', pos=grip-x*.10+[0, 0, .03], rot=rt, grip='wide'),
            dict(kind='pose', name='align', pos=lambda: grip-x*.10, rot=lambda: rt, speed=.08, tol=.004, grip='wide'),
            dict(kind='pose', name='insert', pos=lambda: grip, rot=lambda: rt, speed=.03, tol=.0015, grip='wide'),
            dict(kind='close', name='close', pos=grip, rot=rt),  # a round handle squirts out of an off-centre pinch
            dict(kind='call', name='grasped', fn=self._record_grasp),
            dict(kind='turner', name='lift', pose=lambda: lift, speed=.06, tol=.006),
            dict(kind='turner', name='carry', pose=self._via, speed=.10, tol=.01),
            dict(kind='turner', name='hover', pose=hover, speed=.08, tol=.006),
            dict(kind='wait', name='wait_browning', cond=lambda: self.task.cues()['bottom_doneness'] >= TARGET_DONENESS-.02),
            dict(kind='turner', name='blade_down', pose=lambda: self.pan_pose(y0), speed=.04, tol=.004),
            dict(kind='slide', name='slide_under', y0=y0, y1=y1),
            dict(kind='lever', name='lever'),
            dict(kind='topple', name='topple'),
            dict(kind='settle', name='hold_flipped', dur=.4),
            dict(kind='withdraw', name='lift_clear'),
            dict(kind='turner', name='park_turner', pose=lambda: park, speed=.08, tol=.02, max_s=6.),  # clear of the pan; not critical
            dict(kind='call', name='done', fn=lambda: setattr(self, 'finished_flag', True))]

    def _via(self):
        n, r = self.pan_pose(-(PANCAKE_R+.012))
        return np.array([self.neck0[0]-.084, self.neck0[1]+.05, n[2]+.10]), R.from_euler('z', APPROACH_YAW/2, degrees=True)

    def _record_grasp(self):
        tcp, rr = self.servos['Left'].pose()
        self.grasp = (self.tr.inv().apply(tcp-self.tp), self.tr.inv()*rr)
        self.event('grasp', self.t, jaw_N=self.jaw.tolist())

    def plan_joints(self, pos, rot, seed=None):
        servo = self.servos['Left']
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
            raise RuntimeError(f'Left: pose unreachable {np.round(pos, 3)}')
        return best[0]

    # ---------------------------------------------------------------- control
    def transition(self, phase, t):
        self.phase, self.phase_time = phase, t
        self.event('phase', t, side='Left', phase=phase)

    @property
    def done(self):
        return getattr(self, 'finished_flag', False)

    @property
    def finished(self):
        return getattr(self, 'finished_flag', False) and 'settled_after_flip' in self.task.events

    def command(self, t):
        if not self.planned:
            self.plan()
            self.planned = True
        servo = self.servos['Left']
        self.servos['Right'].move_joints(PARK['Right'], rate=1.)
        self.servos['Right'].grip(True)
        tcp, rr = servo.pose()
        if not self.queue:
            servo.move_joints(servo.target[servo.arm], rate=1.)
            servo.grip(True, squeeze=.02)
            return
        s = self.queue[0]
        kind, name = s['kind'], s['name']
        if self.phase != name:
            self.transition(name, t)
        el = t-self.phase_time
        advance, grip = False, s.get('grip', 'hold')
        self.idle = kind == 'wait'
        if kind == 'jpose':  # long repositioning: turn the base tucked, unfold 10 cm above, descend (joint space)
            if self.jplan is None:
                q_above = self.plan_joints(s['pos']+[0, 0, .10], s['rot'])
                qf = self.plan_joints(s['pos'], s['rot'], seed=q_above)
                tuck = np.array(READY_Q)
                tuck[0] = q_above[0]
                self.jplan = [tuck, q_above, qf]
            servo.move_joints(self.jplan[0], rate=1.2)
            if np.max(np.abs(servo.joints()-self.jplan[0])) < .02:
                self.jplan.pop(0)
            if not self.jplan:
                self.jplan, advance = None, True
        elif kind in ('pose', 'turner'):
            if kind == 'pose':
                goal, r = s['pos'](), s['rot']()
            else:
                goal, r = self.turner_target(*s['pose']())
            speed, turn = s['speed'], 1.2
            if grip == 'hold':  # carrying the turner: ramp speed and turn rate
                d, ang = np.linalg.norm(goal-tcp), (r*rr.inv()).magnitude()
                speed = max(.004, min(speed, ACCEL*el, np.sqrt(2*ACCEL*d)))
                turn = max(.02, min(.8, ANG_ACCEL*el, np.sqrt(2*ANG_ACCEL*ang)))
            servo.move(goal, r, speed=speed, turn_rate=turn)
            advance = (np.linalg.norm(tcp-goal) < s['tol'] and (r*rr.inv()).magnitude() < .05) or el > s.get('max_s', 1e9)
        elif kind == 'close':  # close-until-contact on the handle, then clamp
            fp, fn = self.jaw
            tp, tn = self.jaw_travel or [(P.JAW_OPENING-.035)/2]*2
            if fp > .5 and fn > .5:  # both pads touch: ramp the squeeze (a jump flicks a round handle out)
                tp, tn = min(P.JAW_TRAVEL+.025, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.025, tn+servo.squeeze_step)
            else:
                tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
                tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
            self.jaw_travel = [tp, tn]
            servo.grip_each(tp, tn)
            servo.move(s['pos'], s['rot'], speed=.01)
            advance = fp > 15. and fn > 15. and el > .3
            grip = 'keep'
        elif kind == 'wait':
            advance = s['cond']()
            servo.move_joints(servo.target[servo.arm], rate=1.)
        elif kind == 'call':
            s['fn']()
            advance = True
        elif kind == 'slide':  # blade tip on the floor slides under the pancake; pitch follows the pan wall
            yt = s['y0']+(s['y1']-s['y0'])*smooth(el/4.5)
            n_cmd, r_cmd = self.pan_pose(yt, press=.003)
            goal, rg = self.turner_target(n_cmd, r_cmd)  # tip pressed onto the floor
            self.cmd = dict(turner_pitch_cmd=float(np.degrees(np.arcsin(-r_cmd.apply([1, 0, 0])[2]))), tcp_err_m=float(np.linalg.norm(goal-tcp)),
                            tcp_rot_err_rad=float((rg*rr.inv()).magnitude()))
            servo.move(goal, rg, speed=.04, turn_rate=.6)
            advance = el > 4.8
        elif kind == 'lever':  # rotate about the pancake's left edge line (on the floor) to LEVER_DEG: right side up
            if self.lever0 is None:
                n, r = self.pan_pose(self.slide_end)
                self.lever0 = (n, r, np.array([*(self.cp[:2]+self.left*PANCAKE_R), G['floor_z']]))
            n, r, pv = self.lever0
            Rl = R.from_rotvec(-np.r_[self.d, 0.]*np.radians(LEVER_DEG*smooth(el/3.)))
            self.lever_pose = (pv+Rl.apply(n-pv), Rl*r)
            goal, rg = self.turner_target(*self.lever_pose)
            servo.move(goal, rg, speed=.06, turn_rate=1.)
            advance = el > 3.3
        elif kind == 'topple':  # push the blade toward the left until the pancake passes vertical (closed loop)
            n, r = self.lever_pose
            self.topple_dx = min(.08, .03*el)
            goal, rg = self.turner_target(n+np.r_[self.left*self.topple_dx, .005], r)
            servo.move(goal, rg, speed=.04, turn_rate=.6)
            advance = self.flip > 120. or el > 4.
            if advance:
                self.event('toppled', t, flip_deg=self.flip, push_m=self.topple_dx)
        elif kind == 'settle':
            n, r = self.lever_pose
            goal, rg = self.turner_target(n+np.r_[self.left*self.topple_dx, .005], r)
            servo.move(goal, rg, speed=.02)
            advance = el > s['dur']
        elif kind == 'withdraw':  # lift the blade straight up, then pull it back out along the approach line
            n, r = self.lever_pose
            n = n+np.r_[self.left*self.topple_dx, .005]
            out = smooth((el-1.2)/1.5)
            goal, rg = self.turner_target(n+np.r_[-self.d*.16*out, .08*smooth(el/1.2)], r)
            servo.move(goal, rg, speed=.08, turn_rate=.8)
            advance = el > 3.
        if grip == 'wide':
            servo.grip(False, opening=.035)
        elif grip == 'hold':
            servo.grip(True, squeeze=.02)
        if advance:
            self.queue.pop(0)
        elif el > 25. and kind != 'wait':
            self.failures.append(f'Left:step_timeout:{name}')

    def trace(self):
        return dict(self.row, phase=self.phase, doneness=self.task.cues()['bottom_doneness'], **getattr(self, 'cmd', {}))
