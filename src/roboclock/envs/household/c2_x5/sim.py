"""C2 physics episode (R4, 2026-11-03): two arms pour cola, a PhysX particle fluid, into six glasses in a row.

The cola is a position-based-dynamics particle fluid (3 mm particles, about 27 microlitres each, water-like
material) simulated on the GPU together with the robot. It starts at rest inside two opened bottles, flows out of
a bottle's neck only when the bottle is tilted far enough, and settles in the glasses or on the counter. Every
volume the task scores is counted from particle positions: cola in each glass (inside its cylinder, below the
rim), left in each bottle, in the air, and spilled. Bottles and glasses are rigid compound colliders (a base disc,
a ring of 24 wall boxes and, for a bottle, a shoulder ring and neck tube), hollow so the fluid can sit inside;
the particles carry their own mass, so a full bottle is heavy and a filled glass is heavy.

Robot (privileged oracle, contact-driven, no object or particle writes), the same routine for both arms on their
own half of the row: pinch the bottle body 12 cm up with the tool 20 or 40 deg down, lift it 7 cm, carry it
upright to a spot 8.5 cm in front-inside of a glass, and tilt it toward the glass while the mouth moves over it.
The tilt follows the measured outflow (cola leaving the bottle): it rises quickly to the first flow, then is
servoed to a target flow, and the bottle tips back once the cola in the glass, the cola in the air and the
expected tail reach the line. Each arm pours its inner glass first and works outward.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.c2_x5.scene import (
    BODY_TOP,
    BOTTLE_R,
    BOTTLE_R_IN,
    BOTTLE_XY,
    GLASS_FLOOR,
    GLASS_R,
    GLASS_R_IN,
    GLASS_RIM,
    GLASS_XY,
    MOUTH_Z,
    NECK_R_IN,
    TARGET_ML,
)
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')
FIXED_HORIZON = True
ARMS = ('Left', 'Right')
BOTTLES = ['/World/Objects/BottleL', '/World/Objects/BottleR']
GLASSES = [f'/World/Objects/Glass{j}' for j in range(1, 7)]
COLA = '/World/Cola'
FLUID = '/World/Fluid'
ARM_GLASSES = {'Left': (2, 1, 0), 'Right': (3, 4, 5)}   # inner glass first
JAWS = {'Left': ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN'], 'Right': ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN']}
SP = .003                                   # particle spacing at rest (m)
ML = SP**3*1e6                              # ml per particle
PCO = SP/(2*.99*.6)                         # particle contact offset giving that rest spacing
BOTTLE_KG, GLASS_KG = .03, .22
GRIP_Z = .12
PHI = {'Left': 180., 'Right': 0.}           # pour direction, bottle -> glass (deg from +x): outward along the row
GRASP_TOOL = {'Left': (75., -25.), 'Right': (105., -25.)}   # tool azimuth, elevation at the pickup (bottles inside the bases)
# per glass (reach-screened on the common rig, >= 0.19 rad joint margin): tool azimuth while pouring (the bottle turns
# about its own axis while carried) and the tilt limit. Each arm's inner glass, next to its base, is reachable only
# to 100 deg; it is poured first, from the full bottle, which flows well before that.
POUR_AZ = [140., 150., 100., 80., 30., 40.]
MAX_TILT = [125., 125., 100., 100., 125., 125.]
CARRY_Z = H+.105                            # carried bottle base clears the glass rims (8.5 cm)
OFFSET = .085                               # upright bottle axis this far from the glass axis (along -PHI)
MOUTH_IN = .015                             # mouth this far short of the glass axis while pouring
BLEND_DEG = 30.
Q_TARGET = 30.                              # ml/s, servoed outflow
TAIL_S = .35                                # s of flow still to land after tipping back starts (measured)
TILT_FAST, TILT_BACK, TILT_ACC = 45., 45., 90.
ACCEL, ANG_ACCEL = .4, 1.5
PARK = np.array(READY_Q)
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


def pour_dir(side):
    a = np.radians(PHI[side])
    return np.array([np.cos(a), np.sin(a), 0.])


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


def _disc(stage, path, r, z0, z1, mat):
    c = UsdGeom.Cylinder.Define(stage, path)
    c.CreateRadiusAttr(float(r))
    c.CreateHeightAttr(float(z1-z0))
    c.CreateAxisAttr('Z')
    c.AddTranslateOp().Set(Gf.Vec3d(0, 0, (z0+z1)/2))
    c.CreateVisibilityAttr('invisible')
    _collider(c.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat, materialPurpose='physics')


def _ring(stage, root, r_in, t, z0, z1, mat, n=24):
    rr = r_in+t/2
    for k in range(n):
        a = 360.*k/n
        _box(stage, f'{root}/W{k}', [rr*np.cos(np.radians(a)), rr*np.sin(np.radians(a)), (z0+z1)/2],
             [t, 2*np.pi*rr/n*1.15, z1-z0], a, mat)


def _annulus(stage, root, r0, r1, z, t, mat, n=24):
    rr = (r0+r1)/2
    for k in range(n):
        a = 360.*k/n
        _box(stage, f'{root}/A{k}', [rr*np.cos(np.radians(a)), rr*np.sin(np.radians(a)), z], [r1-r0, 2*np.pi*r1/n*1.2, t], a, mat)


def _look(stage):
    look = UsdShade.Material.Define(stage, '/World/Looks/ColaFluid')
    sh = UsdShade.Shader.Define(stage, '/World/Looks/ColaFluid/Shader')
    sh.SetSourceAsset(Sdf.AssetPath('OmniGlass.mdl'), 'mdl')
    sh.SetSourceAssetSubIdentifier('OmniGlass', 'mdl')
    sh.CreateInput('glass_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.35, .12, .04))
    sh.CreateInput('glass_ior', Sdf.ValueTypeNames.Float).Set(1.34)
    look.CreateSurfaceOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    look.CreateVolumeOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    return look


def initial_particles():
    """Particle grid at rest spacing filling each bottle body (kept 3.5 mm off the walls)."""
    r = BOTTLE_R_IN-.0035
    layer = [(x, y) for x in np.arange(-r, r+1e-9, SP) for y in np.arange(-r, r+1e-9, SP) if x*x+y*y <= r*r]
    pts = []
    for bx, by in BOTTLE_XY:
        for z in np.arange(.0095, BODY_TOP-.012, SP):
            pts += [(bx+x, by+y, H+z) for x, y in layer]
    return np.array(pts)


def build(stage, mat):
    from omni.physx.scripts import particleUtils
    s = stage
    api = PhysxSchema.PhysxSceneAPI.Apply(s.GetPrimAtPath('/World/Physics'))
    api.CreateEnableGPUDynamicsAttr(True)       # particles run only on the GPU
    api.CreateBroadphaseTypeAttr('GPU')
    for path in BOTTLES:
        _disc(s, path+'/Base', BOTTLE_R, 0., .006, mat)
        _ring(s, path, BOTTLE_R_IN, BOTTLE_R-BOTTLE_R_IN, 0., BODY_TOP, mat)
        _annulus(s, path, NECK_R_IN, BOTTLE_R, BODY_TOP+.0015, .003, mat)
        _ring(s, path+'/Neck', NECK_R_IN, .0025, BODY_TOP, MOUTH_Z, mat, n=16)
        P.body(s.GetPrimAtPath(path), BOTTLE_KG, com=(0., 0., .13))
    for path in GLASSES:
        _disc(s, path+'/Base', GLASS_R, 0., GLASS_FLOOR, mat)
        _ring(s, path, GLASS_R_IN, GLASS_R-GLASS_R_IN, 0., GLASS_RIM, mat)
        P.body(s.GetPrimAtPath(path), GLASS_KG)
    ps = particleUtils.add_physx_particle_system(
        s, FLUID, simulation_owner=Sdf.Path('/World/Physics'), particle_contact_offset=PCO, contact_offset=PCO,
        rest_offset=.99*.6*PCO, fluid_rest_offset=.99*.6*PCO, solid_rest_offset=.99*.6*PCO, max_velocity=3.,
        solver_position_iterations=8, enable_ccd=True)
    pm = UsdShade.Material.Define(s, '/World/PhysicsMaterials/ColaPBD')
    particleUtils.add_pbd_particle_material(s, pm.GetPath(), cohesion=.01, viscosity=.0001, surface_tension=.0074,
                                            friction=.1, damping=.1)
    UsdShade.MaterialBindingAPI.Apply(ps.GetPrim()).Bind(pm, materialPurpose='physics')
    particleUtils.add_physx_particle_isosurface(s, FLUID, enabled=True)
    PhysxSchema.PhysxParticleSmoothingAPI.Apply(ps.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(ps.GetPrim()).Bind(_look(s))
    pts = initial_particles()
    particleUtils.add_physx_particleset_points(s, COLA, Vt.Vec3fArray(pts.tolist()), Vt.Vec3fArray(np.zeros_like(pts).tolist()),
                                              [SP]*len(pts), FLUID, True, True, 0, 1040*SP**3, 0.)
    G['particles'] = len(pts)
    return dict(cola=f'PhysX PBD particle fluid on the GPU: {len(pts)} particles of {SP*1000:.0f} mm ({ML*1000:.0f} ul each, '
                     f'{len(pts)*ML/2:.0f} ml per bottle), water-like material, isosurface render',
                bottles=f'rigid hollow compounds (base disc, 24 wall boxes, shoulder ring, neck tube), {BOTTLE_KG*1000:.0f} g shell; '
                        'the cola inside carries its own mass',
                glasses=f'rigid hollow compounds (base disc, 24 wall boxes), {GLASS_KG*1000:.0f} g',
                measurement='volumes counted from particle positions: in each glass (inside its cylinder, below the rim), in each '
                            'bottle, in the air and spilled (on or below the counter outside the glasses)',
                grasp='each arm pinches its bottle body 12 cm up; a pour tilts the bottle toward the glass while the mouth moves over it')


class Arm:
    def __init__(self, side):
        self.side, self.b = side, ARMS.index(side)
        self.queue, self.planned = [], False
        self.phase, self.phase_time = 'idle', 0.
        self.jplan, self.jaw_travel, self.grasp = [], None, None
        self.tilt_cmd = self.tilt_rate = 0.
        self.stopping, self.flowing = False, False
        self.finished = False


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.failures, self.t, self.idle = [], 0., False
        self.arms = {side: Arm(side) for side in ARMS}
        self.subframes = 1
        self.counts_t = -1.
        self.flow = [0., 0.]
        self.hist = [[], []]

    # ------------------------------------------------------------------ world
    def setup(self, world, stage):
        import carb
        from isaacsim.core.prims import RigidPrim
        pc = world.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type('GPU')
        carb.settings.get_settings().set('/physics/updateParticlesToUsd', True)
        spec = self.task.spec  # the seed's layout, before the reset
        cola = UsdGeom.Points(stage.GetPrimAtPath(COLA))
        pts = np.asarray(cola.GetPointsAttr().Get(), float)
        for b, (path, xy) in enumerate(zip(BOTTLES, BOTTLE_XY)):
            off = spec.offset(('bottle_left', 'bottle_right')[b])
            P.shift(stage.GetPrimAtPath(path), off, xy)
            inside = np.hypot(pts[:, 0]-xy[0], pts[:, 1]-xy[1]) < BOTTLE_R  # its cola moves with it
            pts[inside, :2] += off[:2]
        cola.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
        for j, (path, xy) in enumerate(zip(GLASSES, GLASS_XY)):
            P.shift(stage.GetPrimAtPath(path), spec.offset(f'glass{j+1}'), xy)
        jaws = JAWS['Left']+JAWS['Right']
        self.bottles = world.scene.add(RigidPrim(BOTTLES, name='bottles', reset_xform_properties=False, track_contact_forces=True,
                                                 contact_filter_prim_paths_expr=[jaws]*2, max_contact_count=1024))
        self.glasses = world.scene.add(RigidPrim(GLASSES, name='glasses', reset_xform_properties=False))
        self.points = UsdGeom.Points(stage.GetPrimAtPath(COLA))

    def _count(self, t):
        P_ = np.asarray(self.points.GetPointsAttr().Get(), float)
        glass_ml, bottle_ml = [], []
        free = np.ones(len(P_), bool)
        for j in range(6):
            rel = self.gr[j].inv().apply(P_-self.gp[j])
            inside = (rel[:, 0]**2+rel[:, 1]**2 < GLASS_R_IN**2) & (rel[:, 2] > GLASS_FLOOR-.003) & (rel[:, 2] < GLASS_RIM)
            glass_ml.append(float(inside.sum())*ML)
            free &= ~inside
        for b in range(2):
            rel = self.br[b].inv().apply(P_-self.bp[b])
            inside = (rel[:, 0]**2+rel[:, 1]**2 < (BOTTLE_R_IN+.002)**2) & (rel[:, 2] > -.002) & (rel[:, 2] < MOUTH_Z+.002)
            bottle_ml.append(float(inside.sum())*ML)
            free &= ~inside
        spilled = free & (P_[:, 2] < H+.012)
        air = free & ~spilled
        left = P_[:, 0] < .30  # each arm pours on its own half of the row
        air_side = [float((air & left).sum())*ML, float((air & ~left).sum())*ML]
        for b in range(2):
            self.hist[b].append((t, bottle_ml[b]))
            while len(self.hist[b]) > 2 and self.hist[b][0][0] < t-.2:
                self.hist[b].pop(0)
            (t0, v0), (t1, v1) = self.hist[b][0], self.hist[b][-1]
            self.flow[b] = max(0., (v0-v1)/(t1-t0)) if t1 > t0 else 0.
        self.vol = dict(glass_ml=glass_ml, bottle_ml=bottle_ml, spilled_ml=float(spilled.sum())*ML, air_ml=float(air.sum())*ML,
                        air_side_ml=air_side)

    def measure(self, t, dt):
        self.t = t
        p, q = self.bottles.get_world_poses()
        self.bp = np.asarray(p)
        self.br = [R.from_quat([x[1], x[2], x[3], x[0]]) for x in np.asarray(q)]
        f = np.linalg.norm(np.asarray(self.bottles.get_contact_force_matrix(dt=dt)), axis=-1)  # (2, [LP, LN, RP, RN])
        self.jaw = {side: f[ARMS.index(side), 2*k:2*k+2] for k, side in enumerate(ARMS)}
        gp, gq = self.glasses.get_world_poses()
        self.gp = np.asarray(gp)
        self.gr = [R.from_quat([x[1], x[2], x[3], x[0]]) for x in np.asarray(gq)]
        if 'glass0' not in G:
            G['glass0'] = self.gp.copy()
        if t-self.counts_t >= .02-1e-9 or not hasattr(self, 'vol'):  # particle counts at 50 Hz
            self._count(t)
            self.counts_t = t
        self.tilt = [float(np.degrees(np.arccos(np.clip(r.apply([0, 0, 1.])[2], -1, 1)))) for r in self.br]
        self.mouth = [self.bp[b]+self.br[b].apply([0, 0, MOUTH_Z]) for b in range(2)]
        gtilt = [float(np.degrees(np.arccos(np.clip(r.apply([0, 0, 1.])[2], -1, 1)))) for r in self.gr]
        upright = tuple(bool(gtilt[j] < 20. and np.linalg.norm(self.gp[j][:2]-G['glass0'][j][:2]) < .03 and self.gp[j][2] > H-.02)
                        for j in range(6))
        f_all = [float(f[b].max()) for b in range(2)]
        down = [bool(f_all[b] < .5 and self.tilt[b] < 5. and abs(self.bp[b][2]-H) < .006) for b in range(2)]
        dropped = [bool(self.bp[b][2] < H-.03 or (self.tilt[b] > 30. and f_all[b] < .5)) for b in range(2)]
        v = self.vol
        self.row = dict(bottle_pos=self.bp.tolist(), bottle_quat_xyzw=[r.as_quat().tolist() for r in self.br], bottle_tilt_deg=self.tilt,
                        mouth=[m.tolist() for m in self.mouth], jaw_contact_N={k: x.tolist() for k, x in self.jaw.items()},
                        bottle_jaw_N=f_all, glass_pos=self.gp.tolist(), glass_tilt_deg=gtilt, outflow_ml_s=list(self.flow), **v)
        return dict(glass_ml=tuple(v['glass_ml']), bottle_ml=tuple(v['bottle_ml']), spilled_ml=v['spilled_ml'], air_ml=v['air_ml'],
                    bottle_released_upright=tuple(down), bottle_dropped=tuple(dropped), glass_upright=upright)

    # --------------------------------------------------------------- poses
    def tcp_for(self, arm, o, r):
        gp, gr = arm.grasp
        return o+r.apply(gp), r*gr

    def pour_pose(self, arm, j, tilt):
        """Bottle origin and rotation for glass j at `tilt`: hovering OFFSET short of the glass along the pour direction
        at tilt 0, the mouth moving over the glass over the first BLEND_DEG. The bottle is then raised just enough that
        no part of it comes within 12 mm of another glass's rim (5 mm for its own glass), the mouth no lower than 3.5 cm
        above the rim; a raised mouth aims further back, toward the bottle, since the stream leaves it moving outward."""
        d = pour_dir(arm.side)
        w = smooth(tilt/BLEND_DEG)
        th = np.radians(tilt)
        glass_xy = G['glass0'][:, :2]  # measured at the start: the seed's layout
        gx, gy = glass_xy[j]
        r = R.from_rotvec(np.cross([0, 0, 1.], d)*th)*R.from_euler('z', POUR_AZ[j]-GRASP_TOOL[arm.side][0], degrees=True)
        axis = r.apply([0, 0, 1.])
        rim = H+GLASS_RIM
        mz = max(rim+.035, CARRY_Z+MOUTH_Z*np.cos(th)+BOTTLE_R*np.sin(th) if tilt < BLEND_DEG else rim+.035)
        mz = (CARRY_Z+MOUTH_Z)*(1-w)+mz*w if tilt < BLEND_DEG else mz
        for _ in range(3):  # the aim depends on the height and the clearance on the aim: two passes settle it
            back = MOUTH_IN+.25*max(0., mz-(rim+.035))
            mxy = np.array([gx, gy])-d[:2]*(OFFSET+(back-OFFSET)*w)
            mouth = np.array([*mxy, mz])
            need = 0.
            for sd in np.arange(0., MOUTH_Z+1e-9, .01):  # distance down the axis from the mouth
                p = mouth-axis*sd
                rad = .0135 if sd < MOUTH_Z-BODY_TOP-.02 else BOTTLE_R
                low = p[2]-rad*np.sqrt(max(0., 1-axis[2]**2))-rad*abs(axis[2])*(sd >= MOUTH_Z-.006)
                for k, (x, y) in enumerate(glass_xy):
                    if np.hypot(p[0]-x, p[1]-y) < GLASS_R+rad+.008:
                        need = max(need, rim+(.005 if k == j else .012)-low)
                need = max(need, H+.02-low)  # and the counter
            if need <= 1e-4:
                break
            mz += need
        else:
            back = MOUTH_IN+.25*max(0., mz-(rim+.035))
            mouth = np.array([*(np.array([gx, gy])-d[:2]*(OFFSET+(back-OFFSET)*w)), mz])
        return mouth-r.apply([0, 0, MOUTH_Z]), r

    def plan_joints(self, side, pos, rot, seed=None):
        servo = self.servos[side]
        sg = -1. if side == 'Left' else 1.
        seeds = [servo.joints(), PARK, np.array([sg*.3, 1.8, 1.2, -1., 0, 0]), np.array([.6*-sg, .6, .4, 0, 0, 0]),
                 np.array([.8*-sg, 1., .8, -.4, 0, 0])]
        best = None
        for q0 in ((seed,) if seed is not None else seeds):
            q, pe, re = servo.kin.solve(pos, rot, q0)
            m = float(np.min(np.minimum(q-LOW, HIGH-q)))
            if pe < .002 and re < .02 and (best is None or m > best[1]):
                best = (q, m)
        if best is None:
            raise RuntimeError(f'{side}: pose unreachable {np.round(pos, 3)}')
        return best[0]

    def plan(self, arm):
        side, b = arm.side, arm.b
        bx, by = self.bp[b][:2]  # measured: where the seed's layout put the bottle
        rot = tool(*GRASP_TOOL[side])
        x = rot.apply([1, 0, 0])
        grasp = np.array([bx, by, H+GRIP_Z])
        pre = grasp-x*.05+[0, 0, .02]
        up = R.identity()
        start = np.array([bx, by, CARRY_Z])
        q = [dict(kind='jpose', name='reach', pos=pre, rot=rot),
             dict(kind='pose', name='insert', pos=grasp, rot=rot, speed=.03, tol=.002),
             dict(kind='close', name='close', pos=grasp, rot=rot),
             dict(kind='call', name='grasped', fn=lambda: self._record(arm)),
             dict(kind='bottle', name='lift', pose=lambda: (start, up), speed=.06, tol=.006)]
        for j in ARM_GLASSES[side]:
            q += [dict(kind='bottle', name=f'to_glass{j+1}', pose=lambda j=j: self.pour_pose(arm, j, 0.), speed=.10, tol=.006),
                  dict(kind='pour', name=f'pour{j+1}', glass=j),
                  dict(kind='bottle', name=f'upright{j+1}', pose=lambda j=j: self.pour_pose(arm, j, 0.), speed=.08, tol=.008, turn=1.5)]
        q += [dict(kind='bottle', name='over_place', pose=lambda: (start, up), speed=.10, tol=.004),
              dict(kind='bottle', name='set_down', pose=lambda: (np.array([bx, by, H+.002]), up), speed=.04, tol=.003),
              dict(kind='release', name='release'),
              dict(kind='pose', name='retreat', pos=None, rot=None, speed=.08, tol=.01),
              dict(kind='joint', name='home'),
              dict(kind='call', name='done', fn=lambda: setattr(arm, 'finished', True))]
        arm.queue = q

    def _record(self, arm, quiet=False):
        o, r = self.bp[arm.b], self.br[arm.b]
        tcp, rr = self.servos[arm.side].pose()
        arm.grasp = (r.inv().apply(tcp-o), r.inv()*rr)
        if not quiet:
            self.event('grasp', self.t, side=arm.side, jaw_N=self.jaw[arm.side].tolist())

    # ---------------------------------------------------------------- control
    @property
    def done(self):
        return all(a.finished for a in self.arms.values())

    @property
    def finished(self):
        # The horizon (formerly the serving window's close); an early end of the task (all served and settled,
        # stall, failure) goes through the recorder's settle tail instead.
        return self.task.time >= self.task.spec.max_time

    @property
    def phase(self):
        return {side: a.phase for side, a in self.arms.items()}

    def command(self, t):
        for arm in self.arms.values():
            self._step(arm, t)

    def _step(self, arm, t):
        side, servo = arm.side, self.servos[arm.side]
        if not arm.planned:
            self.plan(arm)
            arm.planned = True
        if not arm.queue:
            servo.move_joints(PARK, rate=1.)
            servo.grip(True)
            return
        s = arm.queue[0]
        kind, name = s['kind'], s['name']
        if arm.phase != name:
            arm.phase, arm.phase_time = name, t
            self.event('phase', t, side=side, phase=name)
        el = t-arm.phase_time
        advance, grip = False, 'hold'
        tcp, rr = servo.pose()
        if kind == 'jpose':
            if not arm.jplan:
                p, r = s['pos'], s['rot']
                for up in (.12, .09, .06):
                    try:
                        q_hi = self.plan_joints(side, p+[0, 0, up], r)
                        break
                    except RuntimeError:
                        pass
                else:
                    q_hi = self.plan_joints(side, p, r)
                qf = self.plan_joints(side, p, r, seed=q_hi)
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
            fp, fn = self.jaw[side]
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
            advance, grip = True, 'keep' if name == 'grasped' else ('close' if name == 'done' else 'hold')
        elif kind == 'bottle':
            if el == 0. and name != 'lift':  # the bottle turns a little in the pinch while pouring: re-measure
                self._record(arm, quiet=True)
            o, r = s['pose']()
            goal, rg = self.tcp_for(arm, o, r)
            d, ang = np.linalg.norm(goal-tcp), (rg*rr.inv()).magnitude()
            speed = max(.004, min(s['speed'], ACCEL*el, np.sqrt(2*ACCEL*d)))
            turn = max(.02, min(s.get('turn', .8), ANG_ACCEL*el, np.sqrt(2*ANG_ACCEL*ang)))
            servo.move(goal, rg, speed=speed, turn_rate=turn)
            advance = d < s['tol'] and ang < .04
        elif kind == 'pour':
            advance = self._pour(arm, s['glass'], t, el)
        elif kind == 'release':  # jaws fully open: the 54 mm bottle needs the full opening
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
        elif el > (40. if kind == 'pour' else 25.):
            self.failures.append(f'{side}:step_timeout:{name}')

    def _pour(self, arm, j, t, el):
        """Tilt until cola flows, servo the tilt to Q_TARGET of measured outflow, tip back once the glass, the cola in
        the air and the expected tail reach the line."""
        servo, b, v = self.servos[arm.side], arm.b, self.vol
        if el == 0.:
            arm.tilt_cmd, arm.tilt_rate, arm.stopping, arm.flowing = self.tilt[b], 0., False, False
        q = self.flow[b]
        arm.flowing |= q > 3.
        expected = v['glass_ml'][j]+v['air_side_ml'][b]+q*TAIL_S
        if not arm.stopping and arm.flowing and expected >= TARGET_ML-1.:
            arm.stopping = True
            self.event('stop_pour', t, side=arm.side, glass=j+1, glass_ml=v['glass_ml'][j], air_ml=v['air_side_ml'][b], flow_ml_s=q,
                       tilt_deg=self.tilt[b])
        if arm.stopping:
            want = -min(TILT_BACK, np.sqrt(2*TILT_ACC*max(0., arm.tilt_cmd)))
        elif not arm.flowing:
            want = TILT_FAST if arm.tilt_cmd < 60. else 15.
        else:
            want = float(np.clip(.6*(Q_TARGET-q), -15., 15.))
        dt = servo.dt
        arm.tilt_rate += float(np.clip(want-arm.tilt_rate, -TILT_ACC*dt, TILT_ACC*dt))
        arm.tilt_cmd = min(MAX_TILT[j], max(0., arm.tilt_cmd+arm.tilt_rate*dt))
        o, r = self.pour_pose(arm, j, arm.tilt_cmd)
        goal_p, rg = self.tcp_for(arm, o, r)
        servo.move(goal_p, rg, speed=.12, turn_rate=1.8)
        return arm.stopping and arm.tilt_cmd <= 0. and q < .5 and v['air_side_ml'][b] < .5

    def trace(self):
        return dict(self.row, phase=self.phase, tilt_cmd={side: a.tilt_cmd for side, a in self.arms.items()})
