"""A2: static initial scene (appearance only; joints/rigid bodies deactivated).

R2.1 layout (physics-driven revision): the countertop oven is the native WallStackOven004 scaled
0.6 and pushed to the back edge of the island, so its drop-down door (hinged at counter height)
leaves a free strip in front of it for the trivet. The door, the lower rack, the tray and each
toast are authored as separate unscaled parts (``OvenDoor``, ``OvenRack``, ``Tray``, ``Toast0/1``)
so the simulation overlay can make them a hinged door, a sliding rack and dynamic bodies.
The upper rack is removed (it left no gripper clearance above the tray).
"""
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import (
    body_visual,
    glass_mdl,
    open_window,
    reparent,
    world,
)

K = L.KITCHEN
R = L.REUSED
OVEN = K/'WallStackOven004/WallStackOven004.usd'
SCALE = .6
BACK_Y = .375               # oven back face, 5 mm inside the island's back edge
TRAY_SIZE = (.23, .175)     # long side along x
TRAY_ZSCALE = 2.            # 2.2 cm rim (the scanned sheet's 1.1 cm rim leaves too little wall for a pinch)
# 0.60 m rig: the trivet was at (0.30, -0.285), in front of the open door and out of the head camera's view
# (y < -0.19); the open drop-down door covers the counter in front of the oven, so the trivet moved beside it.
# The oven moved 14 cm right and the trivet to its front-left (oven body x >= 0.253, trivet x <= 0.24, 2 cm
# narrower): the right arm keeps the door handle within its reach (it cannot pinch the bar left of x ~ 0.38),
# and the tray is carried out of the cavity, then sideways past the open door. The trivet sits 8 cm in front of
# the oven line: beside the oven, the right hand's palm (+-9 cm along the jaw axis) rubbed the oven's left side
# and could not rise off the released tray.
OVEN_X = .44
TRIVET = dict(center=(.12, -.08), size=(.24, .17, .024))
TRAY_LAND_DY = 0.           # the tray lands on the trivet centre
CARRY_Y = -.04              # tray centre y for the sideways carry: its back edge is clear of the oven front
RACK_EJECT = .178          # door-linked auto-eject travel of the rack (full-extension runners)
# Native WallStackOven004 geometry (metres, oven root frame), measured by ray casting the body mesh
NATIVE = dict(y_front=-.314, y_back=.2009, z_base=-.2976, hinge=(0., -.268048, -.29886),
              cavity_x=(-.2295, .2368), cavity_y=(-.2689, .1973), cavity_z=(-.2295, .1111),
              handle=(0., -.305, .1255))


def build(ctx):
    """Add the A2 task objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored, tex, wood, notes, interior, rc = (ctx.s, ctx.obj, ctx.point, ctx.authored, ctx.tex, ctx.wood,
                                                               ctx.notes, ctx.interior, ctx.rc)
    depth = (NATIVE['y_back']-NATIVE['y_front'])*SCALE
    oven = obj('CountertopOven', OVEN, [OVEN_X, BACK_Y-depth/2], H, SCALE,
               role=f'countertop oven: native WallStackOven004 uniformly scaled {SCALE}; door and lower rack re-authored as separate parts',
               native=True)
    src = str(oven.GetPath())+'/Source'
    door_m = world(s, src+'/WallStackOven004_Door001')
    pane_m = world(s, src+'/WallStackOven004_Door001_Clear')
    rack_m = world(s, src+'/WallStackOven004_Shelf002')
    root_m = world(s, src)
    hinge = door_m[3, :3]
    for part in ['WallStackOven004_Door001', 'WallStackOven004_Door001_Clear', 'WallStackOven004_Shelf001', 'WallStackOven004_Shelf002']:
        s.GetPrimAtPath(src+'/'+part).SetActive(False)
    looks = src+'/Looks'
    door = reparent(s, OVEN, '/World/Objects/OvenDoor', '/root/WallStackOven004_Door001', door_m, hinge, looks)
    pane = reparent(s, OVEN, '/World/Objects/OvenDoor/Pane', '/root/WallStackOven004_Door001_Clear', pane_m, hinge, looks)
    # The pane is a child of the door Xform: express it relative to the door origin.
    UsdGeom.Xformable(pane).GetOrderedXformOps()[0].Set(Gf.Vec3d(0., 0., 0.))
    rack = reparent(s, OVEN, '/World/Objects/OvenRack', '/root/WallStackOven004_Shelf002', rack_m, rack_m[3, :3], looks)
    removed = open_window(s, '/World/Objects/OvenDoor/Visual', '/World/Objects/OvenDoor/Pane/Visual')
    tint = glass_mdl(s, 'OvenDoorGlass', color=(.72, .70, .66))
    for m in Usd.PrimRange(pane):
        if m.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(m).Bind(tint, UsdShade.Tokens.strongerThanDescendants)
    notes.append(f'Door window opened in the static override: {removed} opaque door faces over the native clear pane removed; pane bound to tinted OmniGlass.')
    for name, prim in [('OvenDoor', door), ('OvenRack', rack)]:
        ctx.notes.append(f'{name}: native part re-authored under an unscaled Xform at {np.round(np.array(UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Get()), 4).tolist()}')

    def native_to_world(p):
        return (np.r_[p, 1.]@root_m)[:3]
    cav_lo = native_to_world([NATIVE['cavity_x'][0], NATIVE['cavity_y'][0], NATIVE['cavity_z'][0]])
    cav_hi = native_to_world([NATIVE['cavity_x'][1], NATIVE['cavity_y'][1], NATIVE['cavity_z'][1]])
    enamel = L.material(s, 'OvenEnamel', [.05, .05, .055], .35, .2)
    L.shape(s, 'OvenCavityFloor', [(cav_lo[0]+cav_hi[0])/2, (cav_lo[1]+cav_hi[1])/2+.005, cav_lo[2]+.002],
            [cav_hi[0]-cav_lo[0]-.004, cav_hi[1]-cav_lo[1]-.012, .004], enamel)
    notes.append('Authored dark enamel cavity floor added: the native oven body has no visible cavity floor when the door window is opened.')

    rack_lo, rack_hi = L.bound(rack)
    rack_c = (rack_lo+rack_hi)/2
    sheet = rc('baking_sheet', 1)
    native = L.native_size(sheet)
    tray_scale = TRAY_SIZE[0]/max(native)
    tray = body_visual(s, '/World/Objects/Tray', sheet, [rack_c[0], rack_c[1], rack_hi[2]+.0008], tray_scale, rot=(0, 0, 90),
                       zscale=TRAY_ZSCALE)
    audit_tray = L.bound(tray)
    floor_z = L.surface_z(tray, [rack_c[0], rack_c[1]], audit_tray[1][2]-.008)
    authored('Tray', tray, f'rimmed baking tray ({TRAY_SIZE[0]*100:.0f} x {TRAY_SIZE[1]*100:.1f} cm, RoboCasa baking_sheet_1 '
             f'scaled {tray_scale:.3f}, x{TRAY_ZSCALE:g} in height for a 2.2 cm rim) on the lower pull-out rack', source=sheet)
    cheese = tex('CheeseToast', 'cheese_toast_initial.png', .55)
    for i, x in enumerate([rack_c[0]-.046, rack_c[0]+.046]):
        toast = body_visual(s, f'/World/Objects/Toast{i}', R/'toast.usda', [x, rack_c[1], floor_z+.0005], 1.05)
        lo, hi = L.bound(toast)
        L.card(s, f'Toast{i}/Cheese', [0., 0., hi[2]-lo[2]+.0012], (hi-lo)[0]*.98, (hi-lo)[1]*.98, cheese)
        authored(f'Toast{i}', toast, 'bread slice proxy (authored) with cheese layer card', source=R/'toast.usda')
    L.shape(s, 'Trivet', [*TRIVET['center'], H+TRIVET['size'][2]/2], TRIVET['size'], wood)
    authored('Trivet', s.GetPrimAtPath('/World/Objects/Trivet'), 'heat-resistant landing spot for the tray, left of the oven')

    interior.append(([rack_c[0], rack_c[1]-.02, (rack_hi[2]+cav_hi[2])/2+.02], 30000., .015, (1., .86, .62)))
    interior.append(([rack_c[0], rack_c[1]+.05, (rack_hi[2]+cav_hi[2])/2+.02], 18000., .015, (1., .86, .62)))
    handle = native_to_world(NATIVE['handle'])
    tray_lo, tray_hi = audit_tray
    rim_z = tray_hi[2]-.004
    ejected = (tray_lo[1]+tray_hi[1])/2-RACK_EJECT  # middle of the short rims once the door-linked rack has ejected
    point('door_handle', [OVEN_X+.10, handle[1]-.004, handle[2]], 'Right', ['forward_down'],
          'pinch the handle bar and swing the drop-down door to ~45 deg; the damped hinge then settles it open')
    point('tray_short_rim_left', [tray_lo[0]+.0015, ejected, rim_z], 'Left', ['forward_down'],
          'left short-rim pinch after the door-linked rack ejects (jaw axis = rim normal)')
    point('tray_short_rim_right', [tray_hi[0]-.0015, ejected, rim_z], 'Right', ['forward_down'],
          'right short-rim pinch; both arms lift the tray out when the cheese is golden')
    tz = H+TRIVET['size'][2]+(tray_hi[2]-tray_lo[2])-.004
    land = TRIVET['center'][1]+TRAY_LAND_DY
    point('trivet_release_left', [TRIVET['center'][0]-TRAY_SIZE[0]/2+.0015, land, tz], 'Left', ['forward_down'],
          'two-arm release of the tray on the trivet')
    point('trivet_release_right', [TRIVET['center'][0]+TRAY_SIZE[0]/2-.0015, land, tz], 'Right', ['forward_down'],
          'two-arm release of the tray on the trivet')
    state = dict(oven='closed, broil element on (initial cue: interior light, native display)', rack_pull_out_m=0.,
                 hinge_world_m=np.round(hinge, 4).tolist(), cavity_world_m=[np.round(cav_lo, 4).tolist(), np.round(cav_hi, 4).tolist()],
                 rack_travel_m=RACK_EJECT, rack_native_travel_m=round(.2*SCALE, 4), door_open_limit_deg=84.,
                 rack='auto-ejects with the door (linkage), travel = 0.178 m x door angle / 80 deg')
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    notes.append('No toaster-oven asset exists in the reachable libraries (Lightwheel collection, RoboCasa @1b92c3d). '
                 f'The native WallStackOven004 is scaled {SCALE} as a countertop oven stand-in.')
    notes.append('R2.1: smaller oven pushed back so the open door (hinged at counter height) leaves room for the trivet; '
                 'the tray is carried with both arms, one on each short rim (one X5 gripper cannot hold a tray off-centre).')
    return state, agent
