"""A2 physics episode: countertop ice dispenser with a hinged paddle, a free glass, rigid ice cubes, and the oracle.

Physics:
- Dispenser: static box colliders for its body around the dispensing recess, the recess walls and the drip tray;
  the paddle is a rigid body on a revolute hinge about x at its top pivot, with a return spring (preloaded against
  its rest stop) and a 12 deg stop.
- Glass: free rigid hollow compound (base disc + 24 wall boxes), 200 g.
- Ice: POOL rigid crescent cubes (convex hull, 8.5 g, ice-on-glass friction 0.15, CCD). They wait on a hidden shelf
  in the bin. Dispenser mechanism (``appliance_step``, not robot control): each time the task model delivers a cube,
  the next waiting cube is placed in the outlet, oriented along a seeded random yaw, moving down at 0.3 m/s, and
  falls freely from there. The recess lamp is on while the paddle is held in.

Measured signals: the paddle angle (from its rigid pose), the glass pose and jaw contact, the cubes' poses (counted
inside the glass in its own frame) and the ice level (top of the highest cube in the glass over the glass's inner
height). The oracle (right arm) picks the glass up from behind, holds it under the outlet, pushes the paddle to
~8 deg with the glass rim, pulls back once the ice reaches ORACLE_LEVEL, sets the glass down where it was and parks.
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.a2_x5 import scene as A
from roboclock.envs.household.a2_x5.task import POOL
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
PADDLE = '/World/Objects/IcePaddle'
GLASS = '/World/Objects/Glass'
CUBES = [f'/World/Objects/Ice/Cube{k}' for k in range(POOL)]
# The right pair first: the oracle's own hand. The left pair is read for the judged "released" only
# (a glass held by the left gripper used to count as released on the counter).
JAWS = ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN', '/World/Arms/Left/jawP', '/World/Arms/Left/jawN']
ARM_PARTS = JAWS[:2]+['/World/Arms/Right/link6']
LAMP = '/World/Lights/DispenserLamp'
GLASS_KG, CUBE_KG, PADDLE_KG = .20, .0085, .02
PADDLE_K = .03          # N m per deg (about 2.5 N at the paddle foot at 8 deg)
PADDLE_PRELOAD = 2.     # deg against the rest stop
SPAWN = np.array([A.NOZZLE[0], A.NOZZLE[1]-.004, A.NOZZLE[2]-.012])  # over the pressed glass's axis, just inside the 70 mm outlet
SPAWN_V = .3

# oracle geometry (reach-screened on the common rig: 0.19 rad joint margin at the counter pickup, >= 0.43 elsewhere)
TOOL_AZ, TOOL_EL = 125., -20.  # wrist off the head camera's line of sight to the glass mouth
GRIP_Z = .035           # tool point above the glass base (pads across the glass; finger plates stay under the paddle)
CARRY_BASE_Z = A.PADDLE_PIVOT[2]-.212  # glass base while carried and pressed: the rim meets the paddle's foot
PRESS_DEG = 8.          # paddle angle held while dispensing (switch at 5, release at 3)
FRONT_Y = A.FRONT_Y-.055  # glass axis in front of the recess before entering
ENTER_Y = A.FRONT_Y+.03   # glass axis just short of the paddle
RELEASE_DY = .022       # pull-back once the ice is at ORACLE_LEVEL
ORACLE_LEVEL = .46      # ice level at which the oracle pulls back (band 0.40-0.60)
MOTION_WINDOW_S = .2    # cube speed = displacement over this window
CUBE_CORNERS = np.array([[sx*A.CUBE_L/2, sy*A.CUBE_R, sz*A.CUBE_R/2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
G = {}


def ice_level(glass_pos, glass_rot, cube_pos, cube_rot, mask):
    """Top of the highest cube in `mask` (its oriented bounding box, in the glass frame) over the glass's inner
    height above its floor: 0 = empty, 1 = level with the rim."""
    if not np.any(mask):
        return 0.
    tops = []
    for p, r in zip(np.asarray(cube_pos)[mask], [cube_rot[i] for i in np.flatnonzero(mask)]):
        rel = glass_rot.inv()*r
        tops.append(glass_rot.inv().apply(p-glass_pos)[2]+float(rel.apply(CUBE_CORNERS)[:, 2].max()))
    return float((max(tops)-A.GLASS_FLOOR)/(A.GLASS_RIM-A.GLASS_FLOOR))


def tool(az, el, roll=-90.):
    e, a = np.radians(el), np.radians(az)
    x = np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])
    y = np.array([0., 0., 1.])-x*x[2]
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))*R.from_euler('x', roll, degrees=True)


GRASP_ROT = tool(TOOL_AZ, TOOL_EL)


def _collider(prim, offset=.002):
    UsdPhysics.CollisionAPI.Apply(prim)
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr(offset)
    api.CreateRestOffsetAttr(0.)


def _box(stage, path, pos, size, yaw, mat):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.)
    c.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    c.AddRotateZOp().Set(float(yaw))
    c.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    c.CreateVisibilityAttr('invisible')
    _collider(c.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat, materialPurpose='physics')


def _slab(stage, path, lo, hi, mat):
    """Static axis-aligned box collider from corner lo to corner hi."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    P.box(stage, path, (lo+hi)/2, hi-lo, mat)


def build(stage, mat):
    s = stage
    # dispenser body: static colliders around the recess (the recess itself, the outlet and the bin stay open)
    bx, by, rx, f = A.BODY_X, A.BODY_Y, A.RECESS_X, A.FRONT_Y
    _slab(s, '/World/Fixtures/IceMachine/SideL', [bx[0], f, H], [rx[0], by[1], A.BODY_TOP], mat)
    _slab(s, '/World/Fixtures/IceMachine/SideR', [rx[1], f, H], [bx[1], by[1], A.BODY_TOP], mat)
    _slab(s, '/World/Fixtures/IceMachine/Front', [rx[0], f, A.CEILING_Z], [rx[1], f+.03, A.BODY_TOP], mat)
    _slab(s, '/World/Fixtures/IceMachine/Tray', [rx[0], f, H], [rx[1], by[1], A.TRAY_TOP], mat)
    _slab(s, '/World/Fixtures/IceMachine/RecessBack', [rx[0], A.RECESS_BACK_Y, A.TRAY_TOP], [rx[1], by[1], A.CEILING_Z], mat)
    px = [A.park_xy(k) for k in range(POOL)]
    lo = np.min(px, axis=0)-.03
    hi = np.max(px, axis=0)+.03
    _slab(s, '/World/Fixtures/IceMachine/IceBin', [lo[0], lo[1], A.PARK_Z-.01], [hi[0], hi[1], A.PARK_Z], mat)

    # paddle: rigid, hinged about +x at its top edge; positive angle = pushed back
    paddle = s.GetPrimAtPath(PADDLE)
    P.body(paddle, PADDLE_KG)
    PhysxSchema.PhysxRigidBodyAPI(paddle).CreateDisableGravityAttr(True)
    P.collision(s.GetPrimAtPath(PADDLE+'/Visual'), mat)
    hinge = UsdPhysics.RevoluteJoint.Define(s, '/World/Joints/PaddleHinge')
    hinge.CreateBody1Rel().SetTargets([paddle.GetPath()])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*map(float, A.PADDLE_PIVOT)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateAxisAttr('X')
    hinge.CreateLowerLimitAttr(0.)
    hinge.CreateUpperLimitAttr(A.PADDLE_MAX_DEG)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(PADDLE_K)
    drive.CreateDampingAttr(.002)
    drive.CreateTargetPositionAttr(-PADDLE_PRELOAD)

    # glass: free hollow compound
    for k in range(24):
        a = 360.*k/24
        rr = A.GLASS_R_IN+(A.GLASS_R-A.GLASS_R_IN)/2
        _box(s, f'{GLASS}/W{k}', [rr*np.cos(np.radians(a)), rr*np.sin(np.radians(a)), A.GLASS_RIM/2],
             [A.GLASS_R-A.GLASS_R_IN, 2*np.pi*rr/24*1.15, A.GLASS_RIM], a, mat)
    base = UsdGeom.Cylinder.Define(s, GLASS+'/Base')
    base.CreateRadiusAttr(A.GLASS_R)
    base.CreateHeightAttr(A.GLASS_FLOOR)
    base.CreateAxisAttr('Z')
    base.AddTranslateOp().Set(Gf.Vec3d(0, 0, A.GLASS_FLOOR/2))
    base.CreateVisibilityAttr('invisible')
    _collider(base.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(base.GetPrim()).Bind(mat, materialPurpose='physics')
    P.body(s.GetPrimAtPath(GLASS), GLASS_KG, com=(0., 0., .045))

    # ice cubes
    ice = P.contact_material(s, 'IceContact', static=.15, dynamic=.10)
    for path in CUBES:
        prim = s.GetPrimAtPath(path)
        P.body(prim, CUBE_KG)
        PhysxSchema.PhysxRigidBodyAPI(prim).CreateEnableCCDAttr(True)
        P.collision(s.GetPrimAtPath(path+'/Visual'), ice)

    lamp = UsdLux.SphereLight.Define(s, LAMP)
    lamp.CreateIntensityAttr(0.)
    lamp.CreateRadiusAttr(.01)
    lamp.CreateColorAttr(Gf.Vec3f(1., .97, .92))
    UsdGeom.Xformable(lamp).AddTranslateOp().Set(Gf.Vec3d(A.NOZZLE[0], A.NOZZLE[1]-.03, A.NOZZLE[2]-.03))
    G.update(spawn=SPAWN.tolist())
    return dict(counter_top_z=H, glass_inner_radius_m=A.GLASS_R_IN, glass_floor_m=A.GLASS_FLOOR, glass_rim_m=A.GLASS_RIM,
                island_top_xy=[[-.27, -.37], [.87, A.FRONT_Y-.01]], pool=POOL, cube_outlet_m=SPAWN.round(4).tolist(),
                bin_z=A.PARK_Z-.01, cube_half_extents_m=[A.CUBE_L/2, A.CUBE_R, A.CUBE_R/2],
                press_deg=5., release_deg=3.,
                dispenser_body='static box colliders around the dispensing recess, recess back wall and drip tray',
                paddle=f'native paddle, rigid {PADDLE_KG*1000:.0f} g on a revolute hinge about x at its top pivot, 0-{A.PADDLE_MAX_DEG:g} deg, '
                       f'return spring {PADDLE_K} N m/deg preloaded {PADDLE_PRELOAD:g} deg; dispensing switch at 5 deg (release 3 deg)',
                glass=f'free rigid hollow compound (base disc, 24 wall boxes), {GLASS_KG*1000:.0f} g',
                ice=f'{POOL} rigid crescent cubes, convex hull, {CUBE_KG*1000:.1f} g, friction 0.15/0.10, CCD',
                dispenser='appliance mechanism: on each delivery of the task model the next waiting cube is placed in the '
                          f'outlet (front half), moving down at {SPAWN_V} m/s, and falls freely; recess lamp on while pressed')


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t = [], 0.
        self.phase, self.phase_time = {'Left': 'done', 'Right': 'idle'}, {'Left': 0., 'Right': 0.}
        self.queue, self.jplan, self.jaw_travel = [], [], None
        self.emitted = 0
        self.paddle_deg, self.jaw = 0., np.zeros(2)
        self.idle, self.subframes = False, 1
        self.row, self.hist = {}, []
        self.delivered_t = np.full(POOL, np.inf)
        rng = np.random.default_rng([int(task.spec.seed), 811])
        self.yaws = rng.uniform(-60., 60., POOL)

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        P.shift(stage.GetPrimAtPath(GLASS), self.task.spec.offset('glass'), A.GLASS_XY)  # the seed's layout, before the reset
        self.paddle = world.scene.add(RigidPrim(PADDLE, name='paddle', reset_xform_properties=False))
        self.glass = world.scene.add(RigidPrim(GLASS, name='glass', reset_xform_properties=False, track_contact_forces=True,
                                               contact_filter_prim_paths_expr=[JAWS], max_contact_count=1024))
        self.cubes = world.scene.add(RigidPrim(CUBES, name='cubes', reset_xform_properties=False))
        self.lamp = UsdLux.SphereLight(stage.GetPrimAtPath(LAMP)).GetIntensityAttr()
        self.lamp_on = False

    def appliance_step(self, stage):
        """Dispenser mechanism (not robot control): deliver cubes into the outlet, lamp while pressed."""
        due = min(int(self.task.cues()['cubes_dispensed']), POOL)
        while self.emitted < due:
            k = self.emitted
            p, q = self.cubes.get_world_poses(indices=np.array([k]))
            p = np.asarray(p).copy()
            p[0] = SPAWN
            for _, old in self.hist:  # the delivery is a placement, not motion: restart its displacement history
                old[k] = SPAWN
            rot = R.from_euler('z', 90.+self.yaws[k], degrees=True)  # long axis roughly across the outlet
            x, y, z, w = rot.as_quat()
            self.cubes.set_world_poses(positions=p, orientations=np.array([[w, x, y, z]]), indices=np.array([k]))
            self.cubes.set_velocities(np.array([[0., 0., -SPAWN_V, 0., 0., 0.]]), indices=np.array([k]))
            self.delivered_t[k] = self.t
            self.emitted += 1
            self.event('cube_delivered', self.t, n=self.emitted)
        on = bool(self.task.pressed)
        if on != self.lamp_on:
            self.lamp.Set(20000. if on else 0.)
            self.lamp_on = on

    # ---------------------------------------------------------------- signals
    def measure(self, t, dt):
        self.t = t
        _, q = self.paddle.get_world_poses()
        q = np.asarray(q)[0]
        self.paddle_deg = float(np.degrees(R.from_quat([q[1], q[2], q[3], q[0]]).as_rotvec()[0]))
        gp, gq = self.glass.get_world_poses()
        self.gp = np.asarray(gp)[0]
        gq = np.asarray(gq)[0]
        self.gr = R.from_quat([gq[1], gq[2], gq[3], gq[0]])
        tilt = float(np.degrees(np.arccos(np.clip(self.gr.apply([0, 0, 1.])[2], -1, 1))))
        jaws = np.linalg.norm(np.asarray(self.glass.get_contact_force_matrix(dt=dt))[0], axis=-1)  # right P, N, left P, N
        self.jaw = jaws[:2]  # the oracle's own (right) hand
        cp, cq = self.cubes.get_world_poses()
        cp, cq = np.asarray(cp), np.asarray(cq)
        crot = [R.from_quat([q[1], q[2], q[3], q[0]]) for q in cq]
        # Cube speed from displacement over the last 0.2 s: cubes stacked in the glass report solver velocities of
        # a few cm/s while their positions change by < 0.1 mm.
        self.hist.append((t, cp.copy()))
        while len(self.hist) > 2 and self.hist[1][0] <= t-MOTION_WINDOW_S:
            self.hist.pop(0)
        t_old, p_old = self.hist[0]
        cv = np.linalg.norm(cp-p_old, axis=-1)/max(t-t_old, dt)
        rel = self.gr.inv().apply(cp-self.gp)
        inside = (np.hypot(rel[:, 0], rel[:, 1]) < A.GLASS_R_IN+.002) & (rel[:, 2] > A.GLASS_FLOOR-.006) & (rel[:, 2] < A.GLASS_RIM+.004)
        out = np.arange(POOL) < self.emitted
        in_glass = inside & out
        outside = out & ~inside & (cv < .02) & (t-self.delivered_t > .5)  # at rest outside, not just placed in the outlet
        moving = bool(np.any(out & (cv > .02)))
        on_counter = (abs(self.gp[2]-H) < .006 and tilt < 8. and jaws.max() < .3  # judged: no jaw of either arm on it
                      and -.27 < self.gp[0] < .87 and -.37 < self.gp[1] < A.FRONT_Y-.01)
        self.in_glass = int(in_glass.sum())
        self.level = ice_level(self.gp, self.gr, cp, crot, in_glass & (cv < .05))  # landed cubes, not ones falling through
        self.row = dict(paddle_deg=self.paddle_deg, glass_m=self.gp.tolist(), glass_quaternion_xyzw=self.gr.as_quat().tolist(),
                        glass_tilt_deg=tilt, jaw_N=jaws.tolist(), cube_positions=cp.round(5).tolist(), cube_speeds=cv.round(4).tolist(),
                        cube_quaternions_wxyz=cq.round(5).tolist(), ice_level=self.level, cubes_in_glass=self.in_glass, cubes_outside=int(outside.sum()), cubes_delivered=self.emitted)
        return dict(paddle_deg=max(0., self.paddle_deg), ice_level=self.level, cubes_in_glass=self.in_glass,
                    cubes_outside=int(outside.sum()),
                    cubes_moving=moving, glass_tilt_deg=tilt, glass_on_counter_released=bool(on_counter),
                    glass_dropped=bool(self.gp[2] < H-.05 or tilt > 70.))

    def update_cues(self):
        pass

    # ---------------------------------------------------------------- control
    @property
    def done(self):
        return self.phase['Right'] == 'done'

    def transition(self, phase, t):
        self.phase['Right'] = phase
        self.phase_time['Right'] = t
        self.event('phase', t, side='Right', phase=phase)

    def glass_tcp(self, axis_xy, base_z):
        return np.array([axis_xy[0], axis_xy[1], base_z+GRIP_Z])

    def plan(self):
        gx, gy = self.gp[:2]  # measured: the glass stands where the seed's layout put it
        grasp = self.glass_tcp((gx, gy), H)
        xdir = GRASP_ROT.apply([1, 0, 0])
        nx = A.NOZZLE[0]
        self.queue = [
            dict(kind='joint', name='reach', pos=grasp-xdir*.06+[0, 0, .015]),
            dict(kind='move', name='insert', pos=grasp, speed=.04, tol=.002),
            dict(kind='close', name='close', pos=grasp),
            dict(kind='move', name='lift', pos=self.glass_tcp((gx, gy), CARRY_BASE_Z), speed=.08, tol=.006, grip='hold'),
            dict(kind='move', name='to_dispenser', pos=self.glass_tcp((nx, FRONT_Y), CARRY_BASE_Z), speed=.10, tol=.005, grip='hold'),
            dict(kind='move', name='enter', pos=self.glass_tcp((nx, ENTER_Y), CARRY_BASE_Z), speed=.05, tol=.003, grip='hold'),
            dict(kind='press', name='press'),
            dict(kind='hold', name='dispensing'),
            dict(kind='move', name='release', pos=None, speed=.05, tol=.003, grip='hold'),
            dict(kind='move', name='withdraw', pos=self.glass_tcp((nx, FRONT_Y), CARRY_BASE_Z), speed=.06, tol=.005, grip='hold'),
            dict(kind='move', name='over_counter', pos=self.glass_tcp((gx, gy), CARRY_BASE_Z), speed=.08, tol=.005, grip='hold'),
            dict(kind='move', name='set_down', pos=self.glass_tcp((gx, gy), H+.002), speed=.05, tol=.003, grip='hold'),
            dict(kind='open', name='let_go'),
            dict(kind='move', name='back_off', pos=grasp-xdir*.07+[0, 0, .02], speed=.06, tol=.008, grip='wide'),
            dict(kind='move', name='rise', pos=grasp-xdir*.07+[0, 0, A.GLASS_RIM+.08], speed=.08, tol=.01, grip='wide'),  # clear the rim before the joint-space park
            dict(kind='park', name='park'),
        ]

    def plan_joints(self, servo, pos):
        best = None
        for q0 in [servo.joints(), np.array(READY_Q), np.array([.3, 1.6, 1.2, -.9, 0, 0])]:
            q, pe, re = servo.kin.solve(pos, GRASP_ROT, q0)
            m = float(np.min(np.minimum(q-LOW, HIGH-q)))
            if pe < .002 and re < .02 and (best is None or m > best[1]):
                best = (q, m)
        if best is None:
            raise RuntimeError(f'Right: pose unreachable {np.round(pos, 3)}')
        return best[0]

    def command(self, t):
        servo = self.servos['Right']
        if self.phase['Right'] == 'idle':
            self.plan()
        if not self.queue:
            if self.phase['Right'] != 'done':
                self.transition('done', t)
            servo.move_joints(np.array(READY_Q), rate=1.)
            servo.grip(True)
            self.idle = True
            return
        s = self.queue[0]
        kind, name = s['kind'], s['name']
        if self.phase['Right'] != name:
            self.transition(name, t)
        el = t-self.phase_time['Right']
        tcp, rr = servo.pose()
        advance, grip = False, s.get('grip', 'wide')
        if kind == 'joint':
            if not self.jplan:
                hi = self.plan_joints(servo, s['pos']+[0, 0, .08])
                tuck = np.array(READY_Q)
                tuck[0] = hi[0]
                self.jplan = [tuck, hi, self.plan_joints(servo, s['pos'])]
            servo.move_joints(self.jplan[0], rate=1.2)
            if np.max(np.abs(servo.joints()-self.jplan[0])) < .02:
                self.jplan.pop(0)
            advance = not self.jplan
            if advance:
                servo.reset_target()
        elif kind == 'move':
            if s['pos'] is None:  # release: straight back toward the robot
                s['pos'] = tcp-[0., RELEASE_DY, 0.]
            servo.move(s['pos'], GRASP_ROT, speed=s['speed'], turn_rate=.8)
            advance = np.linalg.norm(tcp-s['pos']) < s['tol'] and (GRASP_ROT*rr.inv()).magnitude() < .05
            if name == 'release':  # the paddle is off once it is back under 3 deg
                advance = advance or self.paddle_deg < 1.
        elif kind == 'close':
            fp, fn = self.jaw
            tp, tn = self.jaw_travel or [(P.JAW_OPENING-2*A.GLASS_R)/2-.002]*2
            if fp > .5 and fn > .5:
                tp, tn = min(P.JAW_TRAVEL+.02, tp+servo.squeeze_step), min(P.JAW_TRAVEL+.02, tn+servo.squeeze_step)
            else:
                tp = min(P.JAW_TRAVEL, tp+(0. if fp > .5 else servo.close_step))
                tn = min(P.JAW_TRAVEL, tn+(0. if fn > .5 else servo.close_step))
            self.jaw_travel = [tp, tn]
            servo.grip_each(tp, tn)
            servo.move(s['pos'], GRASP_ROT, speed=.01)
            advance, grip = fp > 10. and fn > 10. and el > .3, 'keep'
            if advance:
                self.event('grasp', t, jaw_N=self.jaw.tolist())
        elif kind == 'press':  # push the glass rim into the paddle until it is held at PRESS_DEG
            goal = s.setdefault('goal', tcp.copy())
            err = PRESS_DEG-self.paddle_deg
            goal[1] += float(np.clip(err*.0004, -.0006, .0006))  # m per control tick toward the press angle
            goal[1] = min(goal[1], self.glass_tcp((0, A.PADDLE_PIVOT[1]-A.GLASS_R-.012), 0)[1])
            servo.move(goal, GRASP_ROT, speed=.02, turn_rate=.5)
            grip = 'hold'
            advance = self.paddle_deg > PRESS_DEG-1.5 and el > .3
            if advance:
                self.queue[1]['goal'] = goal
        elif kind == 'hold':  # keep the paddle at PRESS_DEG until the requested cube has landed in the glass
            goal = s['goal']
            goal[1] += float(np.clip((PRESS_DEG-self.paddle_deg)*.0002, -.0003, .0003))
            servo.move(goal, GRASP_ROT, speed=.02, turn_rate=.5)
            grip = 'hold'
            advance = self.level >= ORACLE_LEVEL
            if advance:
                self.event('ice_at_release_level', t, ice_level=self.level, cubes_in_glass=self.in_glass, delivered=self.emitted)
        elif kind == 'open':
            servo.move(tcp, rr, speed=.01)
            advance, grip = el > .8, 'wide'
        elif kind == 'park':
            servo.move_joints(np.array(READY_Q), rate=1.)
            advance, grip = np.max(np.abs(servo.joints()-np.array(READY_Q))) < .03, 'close'
        if grip == 'wide':
            servo.grip(False)
        elif grip == 'hold':
            servo.grip(True, squeeze=.02)
        elif grip == 'close':
            servo.grip(True)
        if advance:
            self.queue.pop(0)
        elif el > (60. if kind == 'hold' else 20.):
            self.failures.append(f'Right:phase_timeout:{name}')

    def video_overlay(self, img, d, t):
        """Viewer-only ice-level panel for the demo video (never in the agent image)."""
        from PIL import ImageFont
        task = self.task
        lo, hi = task.spec.level_band
        font = ImageFont.load_default(size=20)
        W, Hh = img.size
        x0, y0 = 16, Hh-92
        d.rectangle([x0, y0, x0+470, y0+76], fill=(18, 18, 18))
        state = 'pressed' if task.pressed else ('released' if 'released' in task.events else 'not pressed')
        level = getattr(self, 'level', 0.)
        d.text((x0+12, y0+8), f'paddle: {state}   cubes in glass: {getattr(self, "in_glass", 0)}', font=font, fill=(235, 235, 235))
        colour = (120, 220, 120) if lo <= level <= hi else ((240, 200, 90) if level < lo else (240, 110, 90))
        d.text((x0+12, y0+40), f'ice level: {level*100:3.0f} % of the glass (target {lo*100:.0f}-{hi*100:.0f} %)', font=font, fill=colour)

    def trace(self):
        return dict(self.row, phase=dict(self.phase))
