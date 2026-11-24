"""A2 physics episode: oven door hinge, auto-eject rack, dynamic tray/toasts, signals and oracle.

Sequence (contact-driven, no robot or object pose writes after reset):
- right arm: pinches the door handle bar and swings the drop-down door to ~45 deg, releases it (a
  damped hinge lets it settle open) and parks;
- oven mechanism: the rack is linked to the door and auto-ejects as it opens (common on countertop
  ovens), presenting the front half of the tray in front of the oven while its back half stays in
  the cavity;
- both arms: each pinches its own short rim of the tray (close-until-contact, then clamp), hold
  while the cheese finishes, lift the tray out over the open door when the browning cue reaches the
  target (the timed retrieval event), set it on the trivet, release and retreat.

The oracle reads privileged poses, contact forces and the browning cue (rendered as the cheese
colour). X5 kinematics shaped the grasps: the jaw axis is the rim normal, so the tool stays in the
y-z plane and holds one pitch from the oven to the trivet. Near the trivet, on the base line, joint 6
follows joint 1 toward its limit, which wants a steep tool; steep at the rack height, the wrist
housing (link 4) meets the oven front above the opening. A 68 deg pitch satisfies both with the pinch
at the middle of each short rim, on the tray's centre of mass: off it, the tray droops in the pads
(0.35 deg/s with the pinch 1 cm away).
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.a2_oven_x5.scene import CARRY_Y, NATIVE, OVEN_X, RACK_EJECT, TRAY_LAND_DY, TRAY_SIZE, TRIVET
from roboclock.envs.household.x5.scene_lib import READY_Q, H
from roboclock.envs.household.x5.sim import physics as P
from roboclock.envs.household.x5.sim.servo import HIGH, LOW

CONTROL = dict(left='ik', right='ik')


def _font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(f'{matplotlib.get_data_path()}/fonts/ttf/DejaVuSans.ttf', size)
    except Exception:
        return ImageFont.load_default()


B_START, B_RETRIEVE = .50, .60       # browning cue levels: start opening / lift the tray out
DOOR_RELEASE_DEG, DOOR_LIMIT_DEG = 45., 84.
RACK_TRAVEL = RACK_EJECT  # full-extension runners (the native slide travels 0.12 m at this scale)
TRAY_MASS, TOAST_MASS, DOOR_MASS, RACK_MASS = .15, .03, .8, .3
TOAST = (.0756, .0945, .020)
G = {}  # geometry measured from the composed stage in build()
FILTER_TRAY = ['/World/Arms/Left/jawP', '/World/Arms/Left/jawN', '/World/Arms/Right/jawP', '/World/Arms/Right/jawN',
               '/World/Fixtures/Trivet', '/World/Objects/OvenRack', '/World/Objects/Toast0', '/World/Objects/Toast1']
FILTER_DOOR = ['/World/Arms/Right/jawP', '/World/Arms/Right/jawN']


def _translation(stage, path):
    return np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation())


def build(stage, mat):
    root = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/Objects/CountertopOven/Source')))

    def native(p):
        return (np.r_[p, 1.]@root)[:3]
    lo, hi = native([-.311, NATIVE['y_front'], NATIVE['z_base']]), native([.311, NATIVE['y_back'], .3006])
    cav_lo = native([NATIVE['cavity_x'][0], NATIVE['cavity_y'][0], NATIVE['cavity_z'][0]])
    cav_hi = native([NATIVE['cavity_x'][1], NATIVE['cavity_y'][1], NATIVE['cavity_z'][1]])
    front = cav_lo[1]
    shell = front+.003  # shell colliders start behind the closed door's inner face (1.5 mm clearance)
    tray0 = _translation(stage, '/World/Objects/Tray')
    tray_hi = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(stage.GetPrimAtPath('/World/Objects/Tray/Visual')).ComputeAlignedRange().GetMax()
    G.update(hinge=_translation(stage, '/World/Objects/OvenDoor'), rack0=_translation(stage, '/World/Objects/OvenRack'),
             tray0=tray0, tray_h=float(tray_hi[2]-tray0[2]), floor=float(_translation(stage, '/World/Objects/Toast0')[2]-tray0[2]-.0005),
             cav_lo=cav_lo, cav_hi=cav_hi, front=front,
             trivet=np.array([*TRIVET['center'], H+TRIVET['size'][2]]))

    def wall(name, a, b):
        a, b = np.array(a), np.array(b)
        P.box(stage, '/World/Fixtures/Oven/'+name, (a+b)/2, b-a, mat)
    # Static oven shell around an open cavity (the native body mesh would fill it with a hull)
    wall('Floor', [cav_lo[0], shell, lo[2]], [cav_hi[0], hi[1], cav_lo[2]])
    wall('Top', [cav_lo[0], shell, cav_hi[2]], [cav_hi[0], hi[1], hi[2]])
    wall('Left', [lo[0], shell, lo[2]], [cav_lo[0], hi[1], hi[2]])
    wall('Right', [cav_hi[0], shell, lo[2]], [hi[0], hi[1], hi[2]])
    wall('Back', [cav_lo[0], cav_hi[1], cav_lo[2]], [cav_hi[0], hi[1], cav_hi[2]])
    P.box(stage, '/World/Fixtures/Trivet', [*TRIVET['center'], H+TRIVET['size'][2]/2], TRIVET['size'], mat)

    # Door: rigid body at the hinge, damped revolute joint to the world
    door = stage.GetPrimAtPath('/World/Objects/OvenDoor')
    s = .6
    # centre of mass 2 mm inside the hinge line: stand-in for the closing spring, the door stays shut
    P.body(door, DOOR_MASS, com=[0., .002, .14])
    # The hinge axis is ~1 mm above the counter: the panel collider starts 4.3 cm up so its outer
    # bottom edge cannot dig into the island as the door swings open (visual dip <= 5 mm).
    P.box(stage, '/World/Objects/OvenDoor/PanelCollision', [0., (-.2772+.268)*s/2+(-.2663+.268)*s/2, (.043+.279)/2],
          [.373, .0065, .279-.043], mat)
    bar_y, bar_z = (-.3047+.268)*s, (.1255-NATIVE['hinge'][2])*s
    P.box(stage, '/World/Objects/OvenDoor/BarCollision', [0., bar_y, bar_z], [.29, .0061, .0064], mat)
    for sx in (-1, 1):
        P.box(stage, f'/World/Objects/OvenDoor/Standoff{sx+1}', [sx*.15, bar_y/2-.003, bar_z], [.008, abs(bar_y)-.004, .008], mat)
    G.update(bar_local=np.array([0., bar_y, bar_z]))
    hinge = UsdPhysics.RevoluteJoint.Define(stage, '/World/Joints/DoorHinge')
    hinge.CreateBody1Rel().SetTargets([door.GetPath()])
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*map(float, G['hinge'])))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    hinge.CreateAxisAttr('X')
    hinge.CreateLowerLimitAttr(0.)
    hinge.CreateUpperLimitAttr(DOOR_LIMIT_DEG)
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
    drive.CreateStiffnessAttr(0.)
    drive.CreateDampingAttr(.08)   # hinge damper (N m s / deg): ~10-15 deg/s free fall instead of slamming

    # Rack: rigid body on a prismatic joint along -y (native 0.2 m travel x 0.6 scale)
    rack = stage.GetPrimAtPath('/World/Objects/OvenRack')
    P.body(rack, RACK_MASS)
    P.box(stage, '/World/Objects/OvenRack/Collision', [-.0032, .0007, -.0009], [.25, .182, .0064], mat)
    slide = UsdPhysics.PrismaticJoint.Define(stage, '/World/Joints/RackSlide')
    slide.CreateBody1Rel().SetTargets([rack.GetPath()])
    slide.CreateLocalPos0Attr(Gf.Vec3f(*map(float, G['rack0'])))
    slide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    slide.CreateAxisAttr('Y')
    slide.CreateLowerLimitAttr(-RACK_TRAVEL)
    slide.CreateUpperLimitAttr(0.)
    # Auto-eject rack (common on countertop ovens): a door-rack linkage, modelled as a stiff drive
    # whose target follows the measured door angle (appliance mechanism, set in appliance_step).
    rd = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), 'linear')
    rd.CreateStiffnessAttr(2000.)
    rd.CreateDampingAttr(200.)
    rd.CreateMaxForceAttr(60.)
    rd.CreateTargetPositionAttr(0.)

    # Tray (origin = bottom centre): floor + four rim walls
    tray = stage.GetPrimAtPath('/World/Objects/Tray')
    P.body(tray, TRAY_MASS)
    w, d = TRAY_SIZE[0]/2, TRAY_SIZE[1]/2
    th, fl = G['tray_h'], G['floor']
    P.box(stage, '/World/Objects/Tray/FloorCollision', [0, 0, fl/2], [2*w, 2*d, fl], mat)
    for name, pos, size in [('L', [-w+.0015, 0, th/2], [.003, 2*d, th]), ('R', [w-.0015, 0, th/2], [.003, 2*d, th]),
                            ('F', [0, -d+.0015, th/2], [2*w, .003, th]), ('B', [0, d-.0015, th/2], [2*w, .003, th])]:
        P.box(stage, f'/World/Objects/Tray/Rim{name}', pos, size, mat)
    for i in range(2):
        toast = stage.GetPrimAtPath(f'/World/Objects/Toast{i}')
        P.body(toast, TOAST_MASS)
        P.box(stage, f'/World/Objects/Toast{i}/Collision', [0, 0, TOAST[2]/2], TOAST, mat)
    return dict(oven='static box shell around an open cavity (front opening at y = %.4f m)' % front,
                door=f'rigid door ({DOOR_MASS} kg) on a damped revolute hinge, 0-{DOOR_LIMIT_DEG:.0f} deg; handle bar + standoff colliders',
                rack=f'rigid rack ({RACK_MASS} kg) on full-extension runners, {RACK_TRAVEL} m travel (native slide: '
                     '0.12 m at this scale), auto-ejected by a door linkage: drive target = travel x door angle / 80 deg',
                tray=f'dynamic tray {TRAY_SIZE[0]} x {TRAY_SIZE[1]} m, {TRAY_MASS} kg, floor + rim colliders; two toasts {TOAST_MASS} kg each',
                process='browning is symbolic (task model); cheese colour driven by the browning cue',
                front_plane_y=float(front), trivet_top_m=G['trivet'].tolist(), trivet_size_m=list(TRIVET['size']),
                tray_size_m=list(TRAY_SIZE), jaw_force_N=P.JAW_FORCE)


def rot_frame(x, y):
    x = np.asarray(x, float)/np.linalg.norm(x)
    y = np.asarray(y, float)-x*(x@np.asarray(y, float))
    y /= np.linalg.norm(y)
    return R.from_matrix(np.column_stack((x, y, np.cross(x, y))))


def tool(el, az, jaw):
    e, a = np.radians(el), np.radians(az)
    return rot_frame([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)], jaw)


DOOR_GRIP_X = OVEN_X+.10
DOOR_TOOL = rot_frame([0, .5, -.866], [0, -.866, -.5])     # 60 deg forward-down, jaws across the bar
# Short-rim pinches, steep approach from the front so the jaw axis is exactly the rim normal (a tilted
# jaw axis pushes or lifts the tray instead of clamping it). The same tool pose serves both arms, mirrored
# joint solutions; the sign picks the arm's rim.
RIM_SIDE = {'Left': -1., 'Right': 1.}
RIM_TOOL = tool(-68, 90, [-1, 0, 0])
RIM_CLEAR = .035            # joint-space approach stops this far above the pinch
PREGRASP = .06
SQUEEZE = .025  # closing target 25 mm past the stop: ~20 N clamp on the 6 mm bar
# ~15 N on the 3 mm rim, inside the 20 N jaw limit: saturated jaws have no centring force, and the tray then
# chatters along the axis through both pinches and walks the toasts across it
RIM_SQUEEZE = .017
JAW_COLS = {'Left': [0, 1], 'Right': [2, 3]}  # tray contact-matrix columns of jawP, jawN


class Oracle:
    def __init__(self, task, event):
        self.task, self.event, self.servos = task, event, None
        self.phase = {'Left': 'wait', 'Right': 'wait'}
        self.phase_time = {'Left': 0., 'Right': 0.}
        self.failures, self.hold, self.offset = [], {}, {}
        self.grasp_dwell = {'Left': 0., 'Right': 0., 'Door': 0.}
        self.grasped = {'Left': False, 'Right': False, 'Door': False}
        self.prev_rel = {}
        self.retrieved = False
        self.target = None
        self.release_dwell = 0.
        self.q_plan = {}
        self.idle = True
        self.jaw_travel = {}
        self.lost = {}

    def setup(self, world, stage):
        from isaacsim.core.prims import RigidPrim
        self.tray = world.scene.add(RigidPrim('/World/Objects/Tray', name='tray', reset_xform_properties=False, track_contact_forces=True,
                                              contact_filter_prim_paths_expr=[FILTER_TRAY], max_contact_count=4096))
        self.toasts = world.scene.add(RigidPrim(['/World/Objects/Toast0', '/World/Objects/Toast1'], name='toasts', reset_xform_properties=False))
        self.door = world.scene.add(RigidPrim('/World/Objects/OvenDoor', name='door', reset_xform_properties=False, track_contact_forces=True,
                                              contact_filter_prim_paths_expr=[FILTER_DOOR], max_contact_count=1024))
        self.rack = world.scene.add(RigidPrim('/World/Objects/OvenRack', name='rack', reset_xform_properties=False))
        # diagnostic: which scene colliders the arms touch (logged in the trace)
        self.obstacles = ['/World/Fixtures/Oven/Floor', '/World/Fixtures/Oven/Top', '/World/Fixtures/Oven/Left',
                          '/World/Fixtures/Oven/Right', '/World/Fixtures/Oven/Back', '/World/Objects/OvenDoor', '/World/Objects/OvenRack',
                          '/World/Objects/Tray', '/World/Objects/Toast0', '/World/Objects/Toast1', '/World/Fixtures/Island', '/World/Fixtures/Trivet']
        links = [f'/World/Arms/{a}/{n}' for a in ('Left', 'Right') for n in ('link3', 'link4', 'link5', 'link6', 'jawP', 'jawN')]
        self.links = links
        self.arm_view = world.scene.add(RigidPrim(links, name='arm_links', reset_xform_properties=False, track_contact_forces=True,
                                                  contact_filter_prim_paths_expr=[self.obstacles]*len(links), max_contact_count=4096))
        tex = UsdShade.Shader(stage.GetPrimAtPath('/World/Looks/CheeseToast/Texture'))
        self.cheese = tex.GetInput('scale') or tex.CreateInput('scale', Sdf.ValueTypeNames.Float4)

    def appliance_step(self, stage):
        """Oven mechanism (not robot control): the door-rack linkage ejects the rack as the door opens."""
        if not hasattr(self, 'rack_target'):
            self.rack_target = UsdPhysics.DriveAPI(stage.GetPrimAtPath('/World/Joints/RackSlide'), 'linear').GetTargetPositionAttr()
        target = -RACK_TRAVEL*float(np.clip(getattr(self, 'door_deg', 0.)/80., 0., 1.))
        if abs(target-getattr(self, '_last_rack_target', 0.)) > 5e-4:
            self.rack_target.Set(target)
            self._last_rack_target = target

    @staticmethod
    def cheese_tint(b):
        """Texture scale for browning b: pale -> golden at 0.6 -> burnt at 1.0."""
        golden, burnt = np.array([1., .80, .50]), np.array([.30, .18, .10])
        return np.ones(3)+(golden-1)*min(b, .6)/.6 if b <= .6 else golden+(burnt-golden)*min(1., (b-.6)/.4)

    def update_cues(self):
        c = self.cheese_tint(float(self.task.cues()['cheese_browning']))
        self.cheese.Set(Gf.Vec4f(*map(float, c), 1.))

    def video_overlay(self, img, d, t):
        """Viewer-only browning gauge for the demo video (the agent camera never shows it).

        The bar is painted in the rendered cheese colour; under it, the scored zones from the spec:
        too early, the success band, over-browned (partial credit) and burnt (zero)."""
        spec, b = self.task.spec, float(self.task.cues()['cheese_browning'])
        b_ret = self.task.b_at_retrieval
        lo, hi = spec.band
        zero_hi = spec.zero[1]
        font = _font(17), _font(14)
        W, H = img.size
        x0, y0, pw, ph = 16, H-128, 640, 116
        d.rectangle([x0, y0, x0+pw, y0+ph], fill=(20, 20, 20), outline=(90, 90, 90))
        d.text((x0+12, y0+8), 'Cheese browning  (task model; overlay for viewers, not in the agent camera)', fill=(225, 225, 225), font=font[1])
        bx0, bx1, by0, by1 = x0+12, x0+pw-12, y0+38, y0+58
        X = lambda v: bx0+(bx1-bx0)*min(max(v, 0.), 1.)
        base = np.array([246, 214, 110.])
        for i in range(bx1-bx0):
            col = tuple(int(v) for v in np.clip(base*self.cheese_tint(i/(bx1-bx0)), 0, 255))
            d.line([bx0+i, by0, bx0+i, by1], fill=col)
        zones = [(0., lo, (120, 120, 120), 'too early'), (lo, hi, (60, 190, 90), 'take out'),
                 (hi, zero_hi, (235, 150, 40), 'over-browned'), (zero_hi, 1., (215, 55, 45), 'burnt')]
        for a, z, col, name in zones:
            d.rectangle([X(a), by1+2, X(z), by1+7], fill=col)
            tw = d.textlength(name, font=font[1])
            if X(z)-X(a) > tw+4:
                d.text(((X(a)+X(z)-tw)/2, by1+9), name, fill=col, font=font[1])
        if b_ret is not None:  # where it left the oven
            xr = X(b_ret)
            d.line([xr, by0-4, xr, by1+4], fill=(255, 255, 255), width=2)
        xb = X(b)
        d.polygon([(xb-7, by0-10), (xb+7, by0-10), (xb, by0-1)], fill=(255, 255, 255))
        if b_ret is None:  # it keeps browning by `carryover` after leaving the oven, so the window to lift it out is shifted
            door = getattr(self, 'door_deg', 0.) > 10
            rate = spec.door_open_rate if door else 1.
            # seconds until browning reaches v at the current oven rate (the task model's heat clock)
            left = lambda v: max(0., ((spec.t_melt-spec.tau*np.log(1-v))/spec.time_scale-self.task.heat_time)/rate)
            if b < lo-spec.carryover:
                msg, col = f'{b:.2f}  not done: window opens in {left(lo-spec.carryover):.0f} s, closes in {left(hi-spec.carryover):.0f} s', (200, 200, 200)
            elif b <= hi-spec.carryover:
                msg, col = f'{b:.2f}  done: take it out within {left(hi-spec.carryover):.0f} s', (90, 220, 120)
            else:
                msg, col = f'{b:.2f}  TOO LATE: over-browned', (240, 90, 70)
            if door:
                msg += f'  (door open: x{spec.door_open_rate:g})'
        else:
            ok = lo <= b <= hi
            msg = f'taken out at {b_ret:.2f}, now {b:.2f} after carry-over: ' + ('in the window' if ok else 'outside the window')
            col = (90, 220, 120) if ok else (240, 90, 70)
        d.text((x0+12, y0+ph-26), msg, fill=col, font=font[0])

    # ---------------------------------------------------------------- signals
    def _pose(self, view, i=0):
        p, q = view.get_world_poses()
        return np.asarray(p)[i], R.from_quat(np.r_[np.asarray(q)[i, 1:], np.asarray(q)[i, 0]])

    def _pinch(self, name, side, cols, force, obj_p, dt, minimum=.3, balance=0.):
        """Opposing-pad contact (each pad >= `minimum` N, weaker pad >= `balance` x stronger) and low relative motion."""
        tcp, rot = self.servos[side].pose()
        rel = rot.inv().apply(obj_p-tcp)
        speed = 0. if name not in self.prev_rel else float(np.linalg.norm(rel-self.prev_rel[name])/dt)
        self.prev_rel[name] = rel
        f = force[cols]
        ok = bool(f.min() > minimum and f.min() >= balance*f.max() and speed < .10)
        self.grasp_dwell[name] = self.grasp_dwell[name]+dt if ok else 0.
        if self.grasp_dwell[name] > .065 and not self.grasped[name]:
            self.event('bilateral_grasp', self.t, part=name, side=side, forces_N=force[cols].tolist())
        self.grasped[name] = self.grasp_dwell[name] > .065

    def measure(self, t, dt):
        self.t = t
        tp, tr = self._pose(self.tray)
        tv = np.asarray(self.tray.get_velocities())[0, :3]
        f = np.linalg.norm(np.asarray(self.tray.get_contact_force_matrix(dt=dt))[0], axis=-1)
        dp, dr = self._pose(self.door)
        df = np.linalg.norm(np.asarray(self.door.get_contact_force_matrix(dt=dt))[0], axis=-1)
        rp, _ = self._pose(self.rack)
        z = dr.apply([0, 0, 1])
        self.door_deg = float(np.degrees(np.arctan2(-z[1], z[2])))
        self.rack_out = float(G['rack0'][1]-rp[1])
        self.tray_p, self.tray_r, self.tray_f = tp, tr, f
        self._pinch('Left', 'Left', [0, 1], f, tp, dt, minimum=2., balance=.25)
        self._pinch('Right', 'Right', [2, 3], f, tp, dt, minimum=2., balance=.25)
        self._pinch('Door', 'Right', [0, 1], df, dp, dt)
        w, d = TRAY_SIZE[0]/2, TRAY_SIZE[1]/2
        corners = tr.apply(np.array([[x, y, z] for x in (-w, w) for y in (-d, d) for z in (0, G['tray_h'])]))+tp
        in_cavity = bool(corners[:, 1].max() > G['front'])
        if not in_cavity and not self.retrieved:
            self.retrieved = True
            self.event('tray_fully_outside_cavity', t, browning=self.task.cues()['cheese_browning'], door_deg=self.door_deg)
        tilt = float(np.degrees(np.arccos(np.clip(tr.apply([0, 0, 1])[2], -1, 1))))
        food = True
        sp, sq = self.toasts.get_world_poses()
        for i in range(2):
            local = tr.inv().apply(np.asarray(sp)[i]-tp)
            food &= bool(abs(local[0]) < w and abs(local[1]) < d and -.002 < local[2] < .03)
        on_trivet = (f[4] > .3 and float(np.max(f[:4])) < .05 and np.linalg.norm(tv) < .02
                     and abs(tp[0]-G['trivet'][0]) < TRIVET['size'][0]/2 and abs(tp[1]-G['trivet'][1]) < TRIVET['size'][1]/2)
        self.release_dwell = self.release_dwell+dt if on_trivet else 0.
        dropped = bool(tp[2] < .85 or np.asarray(sp)[:, 2].min() < .85)
        cm = np.linalg.norm(np.asarray(self.arm_view.get_contact_force_matrix(dt=dt)), axis=-1)
        touches = {f'{self.links[i].split("/")[-2][0]}:{self.links[i].split("/")[-1]}>{self.obstacles[j].split("/")[-1]}': round(float(cm[i, j]), 2)
                   for i, j in zip(*np.nonzero(cm > .5))}
        self.row = dict(arm_contacts=touches, tray_position=tp.tolist(), tray_quaternion_xyzw=tr.as_quat().tolist(), tray_velocity=tv.tolist(),
                        tray_contact_forces_N=f.tolist(), toast_positions=np.asarray(sp).tolist(), door_deg=self.door_deg,
                        door_contact_forces_N=df.tolist(), rack_out_m=self.rack_out, tray_corners=corners.tolist(),
                        grasped=dict(self.grasped), tilt_deg=tilt, on_trivet_released_s=self.release_dwell)
        return dict(door_angle_deg=self.door_deg, tray_in_cavity=in_cavity, tray_on_trivet_released=bool(on_trivet),
                    food_on_tray=food, tray_tilt_deg=tilt, dropped=dropped)

    # ---------------------------------------------------------------- control
    def transition(self, side, phase, t):
        self.phase[side] = phase
        self.phase_time[side] = t
        self.event('phase', t, side=side, phase=phase)

    @property
    def done(self):
        return all(v == 'done' for v in self.phase.values())

    def _plan(self, side, pos, rot):
        servo = self.servos[side]
        q, pe, re = servo.kin.solve(pos, rot, servo.joints())
        margin = float(np.min(np.minimum(q-LOW, HIGH-q)))
        if pe > .004 or re > .05:
            raise RuntimeError(f'{side}: planned pose unreachable {np.round(pos, 3)} ({pe:.4f} m, {re:.3f} rad, margin {margin:.2f})')
        return q

    def _door_pose(self, deg):
        rx = R.from_euler('x', deg, degrees=True)
        local = G['bar_local']+[DOOR_GRIP_X-G['hinge'][0], 0, 0]
        return G['hinge']+rx.apply(local), rx*DOOR_TOOL

    def _grip_point(self, side):
        # 4 mm below the rim top: the 15 x 10 mm pads then overlap the rim and clear the tray floor
        return self.tray_p+self.tray_r.apply([RIM_SIDE[side]*(TRAY_SIZE[0]/2-.0015), 0., G['tray_h']-.004]), RIM_TOOL

    def command(self, t):
        b = self.task.cues()['cheese_browning']
        for side in ('Right', 'Left'):
            servo = self.servos[side]
            phase, elapsed = self.phase[side], t-self.phase_time[side]
            tcp, rr = servo.pose()
            close, opening, speed = False, P.JAW_OPENING, .12
            goal, rot = tcp.copy(), rr
            moved = False
            if side == 'Right':
                if phase == 'wait':
                    goal, rot = self.hold.setdefault('ready_R', (tcp.copy(), rr))
                    if b >= B_START:
                        self.idle = False
                        self.transition(side, 'pre_bar', t)
                elif phase == 'pre_bar':
                    gp, gr = self._door_pose(0.)
                    pre = gp-gr.apply([PREGRASP, 0, 0])
                    q = self.q_plan.setdefault('pre_bar', self._plan(side, pre, gr))
                    servo.move_joints(q)
                    moved = True
                    if np.linalg.norm(tcp-pre) < .006 and (gr*rr.inv()).magnitude() < .08:
                        servo.reset_target()
                        self.transition(side, 'insert_bar', t)
                elif phase == 'insert_bar':
                    goal, rot = self._door_pose(0.)
                    goal, speed = goal-rot.apply([.004, 0, 0]), .04  # bar between the distal pads
                    if np.linalg.norm(tcp-goal) < .004:
                        self.transition(side, 'close_bar', t)
                elif phase == 'close_bar':
                    goal, rot = self._door_pose(0.)
                    goal, close = goal-rot.apply([.004, 0, 0]), True
                    if self.grasped['Door'] or elapsed > .8:
                        self.transition(side, 'swing', t)
                elif phase == 'swing':  # follow the handle arc, leading the measured door angle
                    goal, rot = self._door_pose(min(DOOR_RELEASE_DEG, self.door_deg+12.))
                    close, speed = True, .10
                    if self.door_deg >= DOOR_RELEASE_DEG-3.:
                        self.hold['bar'] = (tcp.copy(), rr)
                        self.transition(side, 'release_bar', t)
                elif phase == 'release_bar':  # open and back off faster than the falling door
                    goal, rot = self.hold['bar']
                    goal, speed = goal-rot.apply([.08, 0, 0]), .20
                    if np.linalg.norm(tcp-goal) < .01:
                        self.transition(side, 'clear_bar', t)
                elif phase == 'clear_bar':
                    goal, rot = self.hold['bar']
                    goal = goal-rot.apply([.08, 0, 0])+[0, 0, .08]
                    if np.linalg.norm(tcp-goal) < .01:
                        self.transition(side, 'park', t)
                elif phase == 'park':
                    goal, rot = self.hold['ready_R']
                    if np.linalg.norm(tcp-goal) < .02 and self.phase['Left'] in ('hold',):
                        self.transition(side, 'pre_pinch', t)
            else:
                if phase == 'wait':  # until the door is open and the rack has presented the tray
                    goal, rot = self.hold.setdefault('ready_L', (tcp.copy(), rr))
                    if self.door_deg > 80. and self.rack_out > RACK_TRAVEL-.006 and self.phase['Right'] in ('park', 'pre_pinch'):
                        self.transition(side, 'pre_pinch', t)
            if phase == 'pre_pinch':
                gp, gr = self.hold.setdefault('pinch'+side, self._grip_point(side))  # frozen target
                pre = gp+[0, 0, RIM_CLEAR]
                q = self.q_plan.setdefault('pre_pinch'+side, self._plan(side, pre, gr))
                servo.move_joints(q)
                moved = True
                if np.linalg.norm(tcp-pre) < .006 and (gr*rr.inv()).magnitude() < .08:
                    servo.reset_target()
                    self.transition(side, 'insert', t)
            elif phase == 'insert':  # above the short rim ...
                goal, rot = self.hold['pinch'+side]
                goal, speed = goal+[0, 0, .02], .05
                if np.linalg.norm(tcp-goal) < .003:
                    self.transition(side, 'drop', t)
            elif phase == 'drop':  # ... then down so the open jaws straddle the rim wall
                goal, rot = self.hold['pinch'+side]
                speed = .03
                if np.linalg.norm(tcp-goal) < .003:
                    self.hold['pinch'+side] = self._grip_point(side)  # re-centre on the rim's current pose
                    self.transition(side, 'close', t)
            elif phase == 'close':
                # close-until-contact, like a parallel gripper: each pad advances until it touches the rim
                # and then holds, so the first pad does not shove the tray; clamp once both pads touch
                goal, rot = self.hold['pinch'+side]
                close = True
                f_p, f_n = self.tray_f[JAW_COLS[side]]
                tp, tn = self.jaw_travel.setdefault(side, [(P.JAW_OPENING-.012)/2]*2)
                both = f_p > .5 and f_n > .5
                if not both:
                    tp = min(P.JAW_TRAVEL, tp+(0. if f_p > .5 else servo.close_step))
                    tn = min(P.JAW_TRAVEL, tn+(0. if f_n > .5 else servo.close_step))
                else:
                    tp = tn = P.JAW_TRAVEL+RIM_SQUEEZE
                self.jaw_travel[side] = [tp, tn]
                if self.grasped[side]:
                    self.offset[side] = (self.tray_r.inv().apply(tcp-self.tray_p), self.tray_r.inv()*rr)
                    self.transition(side, 'hold', t)
            elif phase == 'hold':
                if self.target is None:
                    self.target = self.tray_p.copy()
                goal, rot = self._tool_for(side, self.target)
                close = True
                both = self.phase['Left'] == 'hold' and self.phase['Right'] == 'hold'
                if both and b >= B_RETRIEVE and side == 'Left':
                    land = G['trivet'][:2]+[0, TRAY_LAND_DY]
                    up = self.target[2]+.015  # lift, out of the cavity over the open door, sideways past the oven, down
                    self.waypoints = [self.target+[0, 0, .015], np.array([self.target[0], CARRY_Y, up]),
                                      np.array([land[0], CARRY_Y, up]), np.array([*land, up]),
                                      np.array([*land, G['trivet'][2]+.001])]
                    for s in ('Left', 'Right'):
                        self.transition(s, 'carry', t)
            elif phase == 'carry':
                close = True
                if side == 'Left':
                    self._advance_target(.05)
                goal, rot = self._tool_for(side, self.target)
                speed = .10
                if side == 'Left' and not self.waypoints and self.release_ready():
                    for s in ('Left', 'Right'):
                        self.transition(s, 'release', t)
            elif phase == 'release':
                goal, rot = self._tool_for(side, self.target)
                if elapsed > .6:
                    self.hold['release'+side] = (tcp.copy(), rr)
                    self.transition(side, 'retreat', t)
            elif phase == 'retreat':  # straight up off the rim: backing out along the tool runs joint 6 into its limit
                goal, rot = self.hold['release'+side]
                goal = goal+[0, 0, .06]
                if np.linalg.norm(tcp-goal) < .01:
                    self.transition(side, 'home', t)
            elif phase in ('home', 'done'):
                q = np.array(READY_Q)
                servo.move_joints(q, rate=1.2)
                moved = True
                if phase == 'home' and np.max(np.abs(servo.joints()-q)) < .03:
                    self.transition(side, 'done', t)
            if phase in ('pre_bar', 'insert_bar'):
                opening = .022  # the bar is only 2.2 cm in front of the door panel: keep the back finger clear
            if phase in ('pre_pinch', 'insert', 'drop'):
                opening = .012  # 3 mm rim: a narrow pre-opening limits how far the first pad can push the tray
            if phase in ('release', 'retreat'):
                opening = .024  # the pad inside the tray stays clear of the toast
            if not moved:
                servo.move(goal, rot, speed=speed, turn_rate=.8)
            if phase == 'close':
                servo.grip_each(*self.jaw_travel[side])
            else:
                servo.grip(close, opening=opening, squeeze=RIM_SQUEEZE if phase in ('hold', 'carry') else SQUEEZE)
            if elapsed > 20. and phase not in ('wait', 'hold', 'park', 'done', 'carry'):
                self.failures.append(f'{side}:phase_timeout:{phase}')
            lost = phase in ('hold', 'carry') and not self.grasped[side] and elapsed > .5
            self.lost[side] = self.lost.get(side, 0.)+servo.dt if lost else 0.
            if self.lost[side] > .3:  # debounced: a jolt from the other jaw closing is not a lost grasp
                self.failures.append(f'{side}:lost_tray_grasp:{phase}')

    def _tool_for(self, side, tray_target):
        """Gripper pose that holds the tray level and unrotated at `tray_target` (bottom centre)."""
        off, rel = self.offset[side]
        return tray_target+off, rel

    def _advance_target(self, speed):
        dt = self.servos['Left'].dt
        errs = [self.servos[s].error[0] for s in ('Left', 'Right')]
        if max(errs) > .012:   # keep both grippers in step with the shared tray target
            return
        while self.waypoints:
            d = self.waypoints[0]-self.target
            n = np.linalg.norm(d)
            if n < 1e-4:
                self.waypoints.pop(0)
                continue
            self.target = self.target+d*min(1., speed*dt/n)
            return

    def release_ready(self):
        return self.tray_f[4] > .5 and self.tray_p[2] < G['trivet'][2]+.004

    def trace(self):
        return dict(self.row, phase=dict(self.phase), browning=self.task.cues()['cheese_browning'])
