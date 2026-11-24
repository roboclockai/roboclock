"""B1 physics episode: hinged faucet lever, measured signals, water cues and the oracle.

The native single lever (Sink054_handle, D6 flow tilt 0-15 deg about y) is a rigid body on a damped
revolute hinge; flow is proportional to its measured tilt. The right arm waits for the sudden supply surge, then reacts: it comes
in from the +x side, pushes the lever top toward the spout until the tilt reaches 0 deg, releases
and retreats. The task model integrates the water in the bowl and the spill from the measured lever
opening; the visible water surface, bubbles, stream and spill follow its cues. The oracle reads the
privileged lever angle and contact forces.
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.b1_x5.scene import BOWL_FLOOR_Z, FOAM, LEVER_OPEN_DEG, SPILLS, SPOUT_XY, STREAM_TOP
from roboclock.envs.household.x5.scene_lib import READY_Q
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
LEVER = '/World/Objects/FaucetLever'
BOWL = '/World/Objects/WaterBowl'
SPOUT = '/World/Objects/Sink/Source/Sink054_spout'  # native spout mesh: moves with the lever (the seed's layout)
# Both hands: the judged "hand on the lever" must not depend on which arm closes it (with the right arm's
# parts alone, a lever closed by the left arm never counted as closed by the robot).
FILTERS = ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN', '/World/Arms/Right/link6',
           '/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Left/link6']
G = {}
ROD_FACE_X, ROD_PUSH_Z = .020, .072  # native handle mesh: rod +x face and push height above the pivot


def tool(el, az):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))


PUSH_TOOL = tool(-45, 135)  # from the right-front, pointing toward the spout and down (reach-screened)


def build(stage, mat):
    lever = stage.GetPrimAtPath(LEVER)
    pivot = np.array(UsdGeom.Xformable(lever).GetOrderedXformOps()[0].Get())
    # The native lever is a 1 cm rod (local x .009-.019, tip at z .0916) on a round cartridge base;
    # push on the rod's +x face 2 cm below its tip, in the pivot frame.
    contact = np.array([ROD_FACE_X, 0., ROD_PUSH_Z])
    G.update(pivot=pivot, top_open=pivot+R.from_euler('y', LEVER_OPEN_DEG, degrees=True).apply(contact),
             top_closed=pivot+contact)
    P.body(lever, .08, com=[0., 0., 0.])  # balanced on the pivot: the cartridge holds any position
    P.collision(stage.GetPrimAtPath(LEVER+'/Visual'), mat)
    hinge = UsdPhysics.RevoluteJoint.Define(stage, '/World/Joints/LeverHinge')
    hinge.CreateBody1Rel().SetTargets([lever.GetPath()])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pivot)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateAxisAttr('Y')
    hinge.CreateLowerLimitAttr(0.)
    hinge.CreateUpperLimitAttr(LEVER_OPEN_DEG)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(.02)  # cartridge friction (N m s / deg)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(BOWL)):  # static triangle mesh: an arm cannot pass through it
        if prim.IsA(UsdGeom.Mesh):
            P.collision(prim, mat, approximation='none')
    for i in range(FOAM):  # bubbles float on the water: give each disc a height offset to drive
        UsdGeom.Xformable(stage.GetPrimAtPath(f'/World/Objects/Foam{i}')).AddTranslateOp(opSuffix='level').Set(Gf.Vec3d(0, 0, 0))
    return dict(lever=f'rigid native lever on a damped revolute hinge about y, 0-{LEVER_OPEN_DEG:g} deg; flow = tilt / {LEVER_OPEN_DEG:g} deg',
                water='symbolic bowl volume / spill (task model); water surface, bubbles, stream and spill driven by the cues',
                bowl='fixed in the basin, static collider; the water adds no load')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.phase = {'Left': 'done', 'Right': 'wait_surge'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.failures, self.hold, self.q_plan = [], {}, {}
        self.lever_deg, self.touch = LEVER_OPEN_DEG, 0.
        self.t, self.ran_until, self.subframes = 0., 0., 1

    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        dx, dy, _ = self.task.spec.offset('faucet')  # the seed's layout, before the reset: the faucet slides on the deck
        d = np.array([dx, dy, 0.])
        for path in (LEVER, SPOUT):
            P.shift(stage.GetPrimAtPath(path), (dx, dy, 0.), (0., 0.))
        hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath('/World/Joints/LeverHinge'))  # anchored in the world frame
        hinge.GetLocalPos0Attr().Set(Gf.Vec3f(*map(float, G['pivot']+d)))
        for key in ('pivot', 'top_open', 'top_closed'):
            G[key] = G[key]+d
        self.spout_xy = (SPOUT_XY[0]+dx, SPOUT_XY[1]+dy)  # where the stream falls; the bowl and its water stay
        self.lever = world.scene.add(RigidPrim(LEVER, name='lever', reset_xform_properties=False, track_contact_forces=True,
                                               contact_filter_prim_paths_expr=[FILTERS], max_contact_count=1024))
        path = '/World/Objects/'
        self.water_at, self.water_size = UsdGeom.Xformable(stage.GetPrimAtPath(path+'WaterSurface')).GetOrderedXformOps()
        self.water_z0 = self.water_at.Get()[2]
        self.foam = [stage.GetPrimAtPath(f'{path}Foam{i}') for i in range(FOAM)]
        self.foam_at = [UsdGeom.Xformable(p).GetOrderedXformOps()[-1] for p in self.foam]
        self.stream = UsdGeom.Cylinder(stage.GetPrimAtPath(path+'RunningWater'))
        self.stream_at = self.stream.GetOrderedXformOps()[0]
        self.spills = [stage.GetPrimAtPath(f'{path}Spill{i}') for i in range(SPILLS)]
        self.puddle = stage.GetPrimAtPath(path+'Puddle')
        self.puddle_size = UsdGeom.Xformable(self.puddle).GetOrderedXformOps()[1]

    def update_cues(self):
        c = self.task.cues()
        z = BOWL_FLOOR_Z+c['water_level_cm']/100
        self.water_at.Set(Gf.Vec3d(*SPOUT_XY, z))
        self.water_size.Set(Gf.Vec3f(c['water_radius_cm']/100, c['water_radius_cm']/100, 1))
        u = c['stream_l_s']/(self.task.spec.inflow_l_s*self.task.spec.time_scale)
        running = 'inherited' if u > .03 else 'invisible'
        if u > .03:
            self.ran_until = self.t
        # the renderer's temporal history keeps a removed stream on the transparent water for seconds: clear it
        self.subframes = 12 if u <= .03 and self.t-self.ran_until < 1. else 1
        for prim, op in zip(self.foam, self.foam_at):
            op.Set(Gf.Vec3d(0, 0, z-self.water_z0))
            prim.GetAttribute('visibility').Set(running)
        self.stream.GetVisibilityAttr().Set(running)
        self.stream.GetRadiusAttr().Set(float(.014*np.sqrt(max(u, .0))*(1.0+.08*np.sin(43*self.t) if c["surging"] else 1.0)))
        self.stream.GetHeightAttr().Set(STREAM_TOP-z)
        self.stream_at.Set(Gf.Vec3d(*self.spout_xy, (STREAM_TOP+z)/2))
        for prim in self.spills:
            prim.GetAttribute('visibility').Set('inherited' if c['overflowing'] and u > .03 else 'invisible')
        self.puddle.GetAttribute('visibility').Set('inherited' if c['spilled_l'] > 0 else 'invisible')
        radius = min(.19, .125+.13*c['spilled_l'])
        self.puddle_size.Set(Gf.Vec3f(radius, radius, 1))

    def measure(self, t, dt):
        self.t = t
        _, q = self.lever.get_world_poses()
        q = np.asarray(q)[0]
        z = R.from_quat([q[1], q[2], q[3], q[0]]).apply([0, 0, 1])
        self.lever_deg = float(np.degrees(np.arctan2(z[0], z[2])))
        f = np.linalg.norm(np.asarray(self.lever.get_contact_force_matrix(dt=dt))[0], axis=-1)
        self.touch = float(f.max())
        self.row = dict(lever_deg=self.lever_deg, lever_contact_N=f.tolist())
        return dict(lever_opening=float(np.clip(self.lever_deg/LEVER_OPEN_DEG, 0., 1.)), hand_on_lever=bool(self.touch > .3))

    def transition(self, side, phase, t):
        self.phase[side] = phase
        self.phase_time[side] = t
        self.event('phase', t, side=side, phase=phase)

    @property
    def done(self):
        return all(v == 'done' for v in self.phase.values())

    def command(self, t):
        servo = self.servos['Right']
        phase, elapsed = self.phase['Right'], t-self.phase_time['Right']
        tcp, rr = servo.pose()
        rot, speed, goal = PUSH_TOOL, .12, tcp.copy()
        if phase == 'wait_surge':
            servo.move_joints(np.array(READY_Q), rate=1.2)
            servo.grip(True)
            if self.task.cues()['surging']:
                self.transition('Right', 'approach', t)
            return
        if phase == 'approach':  # joint-space move to 4 cm on the +x side of the lever top
            pre = G['top_open']+[.04, 0, .01]
            q = self.q_plan.setdefault('pre', self._plan(servo, pre))
            servo.move_joints(q, rate=1.5)
            servo.grip(True)
            if np.linalg.norm(tcp-pre) < .006 and (PUSH_TOOL*rr.inv()).magnitude() < .08:
                servo.reset_target()
                self.transition('Right', 'push', t)
            return
        if phase == 'push':  # follow the rod face along its closing arc, leading the measured tilt
            lead = max(0., self.lever_deg-4.)
            goal = G['pivot']+R.from_euler('y', lead, degrees=True).apply([ROD_FACE_X-.003, 0., ROD_PUSH_Z])
            speed = .04
            if self.lever_deg < .3:
                self.hold['closed'] = (tcp.copy(), rr)
                self.transition('Right', 'release', t)
        elif phase == 'release':  # back off the lever before it counts as closed
            goal, rot = self.hold['closed']
            goal, speed = goal+[.05, 0, .03], .10
            if np.linalg.norm(tcp-goal) < .008:
                self.transition('Right', 'retreat', t)
        elif phase in ('retreat', 'done'):
            q = np.array(READY_Q)
            servo.move_joints(q, rate=1.2)
            servo.grip(True)
            if phase == 'retreat' and np.abs(servo.joints()-q).max() < .02:
                self.transition('Right', 'done', t)
            return
        servo.move(goal, rot, speed=speed, turn_rate=.8)
        servo.grip(True)
        if elapsed > 15. and phase not in ('done',):
            self.failures.append(f'Right:phase_timeout:{phase}')

    def _plan(self, servo, pos):
        q, pe, re = servo.kin.solve(pos, PUSH_TOOL, servo.joints())
        if pe > .004 or re > .05:
            raise RuntimeError(f'push pose unreachable {np.round(pos, 3)} ({pe:.4f} m, {re:.3f} rad)')
        if float(np.min(np.minimum(q-LOW, HIGH-q))) < .05:
            raise RuntimeError('push pose too close to a joint limit')
        return q

    def trace(self):
        return dict(self.row, phase=dict(self.phase), water_level_cm=self.task.cues()['water_level_cm'],
                    water_below_rim_cm=self.task.cues()['water_below_rim_cm'])
