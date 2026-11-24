"""A2: static initial scene: a countertop ice dispenser on the island, an empty glass on the counter.

Layout (rig at the island, both arms facing +y; the island at its usual place):
- Countertop ice dispenser (authored body, disclosed proxy): 36 cm wide, 33 cm deep, 52 cm tall, against the
  island's back edge with its front at y = 0.05 m. Ice bin and auger inside the upper body; an open dispensing
  recess (26 cm wide, 15.5 cm deep) over a drip tray; a nozzle housing with an octagonal ice outlet under the
  recess ceiling.
- Dispenser paddle: the native paddle of the Lightwheel Refrigerator001 door dispenser (Refrigerator001_Switch,
  "Ice | Water", with its native material), at x0.85, hinged at its top edge behind the outlet. Its hinge,
  outlet and recess keep the native dispenser's proportions at that scale. Pushing a glass rim against it
  starts the dispenser.
- Glass: lathe tumbler (56 mm across, 120 mm tall, OmniGlass) on the counter in front of the dispenser.
- Ice: POOL crescent cubes (30 x 28 x 14 mm) waiting in the bin inside the body (out of sight).
"""
import math

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdShade

from roboclock.envs.household.a2_x5.task import POOL
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import glass_mdl, lathe, reparent

FRIDGE = L.KITCHEN/'Refrigerator001/Refrigerator001.usd'  # source of the native paddle (and its material) only
S = .85                                    # paddle scale (the native dispenser at x0.85)

FRONT_Y = .05                              # dispenser front face
BODY_X = (.12, .48)
BODY_Y = (FRONT_Y, .38)                    # back flush with the island's back edge
BODY_TOP = H+.52
RECESS_X = (.17, .43)               # wide enough for the X5 palm (+-9 cm along the jaw axis) holding the glass
RECESS_BACK_Y = FRONT_Y+.155
TRAY_TOP = H+.05                           # drip grille, 0.908 m
CEILING_Z = TRAY_TOP+.30                   # recess ceiling, 1.208 m
PADDLE_PIVOT = np.array([.30, FRONT_Y+.117, CEILING_Z-.028])  # native: 11.7 cm behind the door face, 2.8 cm under the ceiling
NOZZLE = np.array([.30, FRONT_Y+.071, CEILING_Z-.030])        # outlet centre (native: 7.1 cm behind the face)
NOZZLE_R = .035
PADDLE_DOWN = np.array([0., -.258819, -.965926])              # native rest: 15 deg forward of vertical
PADDLE_FRONT = np.array([0., -.965926, .258819])*.0219*S      # pivot -> front face (native local z -2.19 cm)
PADDLE_LEN = .1457*S
PADDLE_MAX_DEG = 12.
SWITCH_NATIVE_PIVOT = np.array([-31.1049, -29.7612, 39.9167])  # cm, the Switch prim's origin in the asset

GLASS_XY = (.36, -.05)
GLASS_R, GLASS_R_IN, GLASS_FLOOR, GLASS_RIM = .028, .0255, .006, .12
GLASS_PROFILE = [(0., 0.), (GLASS_R-.002, 0.), (GLASS_R, .003), (GLASS_R, GLASS_RIM), (GLASS_R_IN, GLASS_RIM),
                 (GLASS_R_IN, GLASS_FLOOR), (0., GLASS_FLOOR)]

CUBE_L, CUBE_R = .030, .014  # crescent cube: half disc of radius CUBE_R extruded CUBE_L along x, flat side down
PARK_Z = CEILING_Z+.05       # bin shelf inside the upper body (hidden)


def park_xy(k):
    return (.215+.05*(k % 4), .225+.04*(k//4))


def crescent(stage, path, mat, segments=12):
    """Half-disc prism mesh centred on its bounding box, flat face down."""
    a = np.linspace(0, np.pi, segments+1)
    zc = -CUBE_R/2
    prof = [(CUBE_R*math.cos(t), zc+CUBE_R*math.sin(t)) for t in a]  # (y, z)
    pts, faces, counts = [], [], []
    for x in (-CUBE_L/2, CUBE_L/2):
        pts += [(x, y, z) for y, z in prof]
    n = len(prof)
    for i in range(n-1):  # curved top
        faces += [i, i+1, n+i+1, n+i]
        counts.append(4)
    faces += [n-1, 0, n, 2*n-1]  # flat bottom
    counts.append(4)
    faces += list(range(n-1, -1, -1))  # end caps
    counts.append(n)
    faces += list(range(n, 2*n))
    counts.append(n)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr('none')
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def box(s, name, x, y, z, mat):
    return L.shape(s, name, [(x[0]+x[1])/2, (y[0]+y[1])/2, (z[0]+z[1])/2], [x[1]-x[0], y[1]-y[0], z[1]-z[0]], mat)


def paddle_matrix():
    """World matrix of the native Switch mesh: native transform, scaled x S, its hinge moved to PADDLE_PIVOT."""
    native = Usd.Stage.Open(str(FRIDGE))
    m = np.array(UsdGeom.Xformable(native.GetPrimAtPath('/root/Refrigerator001_Switch')).GetLocalTransformation())
    scale = np.diag([S/100, S/100, S/100, 1.])
    move = np.eye(4)
    move[3, :3] = PADDLE_PIVOT-S/100*SWITCH_NATIVE_PIVOT  # USD row-vector convention
    return m@scale@move


def build(ctx):
    s, point, authored, interior = ctx.s, ctx.point, ctx.authored, ctx.interior
    steel = L.material(s, 'IceMachineSteel', [.62, .64, .66], .32, .85)
    front = L.material(s, 'IceMachineFront', [.05, .055, .06], .25, .3)
    recess = L.material(s, 'IceMachineRecess', [.03, .032, .035], .5)
    grille = L.material(s, 'IceMachineGrille', [.20, .21, .22], .45, .6)
    trim = L.material(s, 'IceMachineTrim', [.72, .74, .76], .22, .9)
    UsdGeom.Xform.Define(s, '/World/Objects/IceMachine')
    bx, by, rx = BODY_X, BODY_Y, RECESS_X
    f = FRONT_Y+.004  # front fascia thickness
    box(s, 'IceMachine/SideL', (bx[0], rx[0]), (f, by[1]), (H, BODY_TOP), steel)
    box(s, 'IceMachine/SideR', (rx[1], bx[1]), (f, by[1]), (H, BODY_TOP), steel)
    box(s, 'IceMachine/Upper', rx, (f, by[1]), (CEILING_Z, BODY_TOP), steel)
    box(s, 'IceMachine/Base', rx, (f, by[1]), (H, TRAY_TOP-.015), steel)
    box(s, 'IceMachine/RecessBack', rx, (RECESS_BACK_Y, by[1]), (TRAY_TOP-.015, CEILING_Z), recess)
    box(s, 'IceMachine/FasciaL', (bx[0], rx[0]), (FRONT_Y, f), (H+.01, BODY_TOP-.01), front)
    box(s, 'IceMachine/FasciaR', (rx[1], bx[1]), (FRONT_Y, f), (H+.01, BODY_TOP-.01), front)
    box(s, 'IceMachine/FasciaTop', rx, (FRONT_Y, f), (CEILING_Z, BODY_TOP-.01), front)
    box(s, 'IceMachine/FasciaBottom', rx, (FRONT_Y, f), (H+.01, TRAY_TOP-.015), front)
    box(s, 'IceMachine/Lid', (bx[0]+.004, bx[1]-.004), (by[0]+.004, by[1]-.004), (BODY_TOP, BODY_TOP+.006), trim)
    box(s, 'IceMachine/Ceiling', rx, (f, RECESS_BACK_Y), (CEILING_Z-.002, CEILING_Z), recess)
    box(s, 'IceMachine/Tray', (rx[0]+.004, rx[1]-.004), (f, RECESS_BACK_Y), (TRAY_TOP-.015, TRAY_TOP), grille)
    for i in range(14):  # grille slots
        x = rx[0]+.02+i*.0170
        box(s, f'IceMachine/Slot{i}', (x-.003, x+.003), (f+.008, RECESS_BACK_Y-.008), (TRAY_TOP-.0006, TRAY_TOP+.0004), recess)
    # nozzle housing under the ceiling, with the octagonal outlet at its bottom; the paddle hangs from its back part
    nx, ny, nz = NOZZLE
    box(s, 'IceMachine/NozzleHousing', (nx-.05, nx+.05), (f, PADDLE_PIVOT[1]+.012), (nz, CEILING_Z), trim)
    lathe(s, 'IceMachine/Outlet', [(NOZZLE_R+.004, 0.), (NOZZLE_R+.004, .006), (NOZZLE_R, .006), (NOZZLE_R, 0.)],
          (nx, ny, nz-.006), recess, segments=8)
    lathe(s, 'IceMachine/OutletMouth', [(0., 0.), (NOZZLE_R, 0.)], (nx, ny, nz-.0058), recess, segments=8)
    led = L.material(s, 'IceMachineLed', [.35, .62, .95], .3, emissive=[.20, .45, .85])
    box(s, 'IceMachine/PowerLed', (.455, .463), (FRONT_Y-.001, FRONT_Y), (BODY_TOP-.05, BODY_TOP-.042), led)
    authored('IceMachine', s.GetPrimAtPath('/World/Objects/IceMachine'),
             'countertop ice dispenser: body with ice bin, dispensing recess, drip tray, nozzle housing and outlet (disclosed proxy)')
    interior.append(([nx, FRONT_Y+.04, CEILING_Z-.04], 4000., .012, (1., .98, .95)))  # dim recess night light

    looks = '/World/Looks/FridgeNative'
    mtl = s.DefinePrim(looks)
    mtl.GetReferences().AddReference(L.rel(FRIDGE), '/root/mtl')
    paddle = reparent(s, FRIDGE, '/World/Objects/IcePaddle', '/root/Refrigerator001_Switch', paddle_matrix(), PADDLE_PIVOT, looks)
    authored('IcePaddle', paddle, 'dispenser paddle: native Refrigerator001_Switch ("Ice | Water") at x0.85 on its top hinge',
             source=FRIDGE)
    bottom = PADDLE_PIVOT+PADDLE_FRONT+PADDLE_DOWN*PADDLE_LEN
    point('paddle_bottom_front', bottom, 'Right', ['forward'], 'press with the back of the glass rim; ice falls from the outlet into the glass')
    point('ice_outlet', NOZZLE, 'Right', ['forward'], 'ice outlet: hold the glass mouth under it')

    glass_mat = glass_mdl(s, 'IceGlass')
    root = UsdGeom.Xform.Define(s, '/World/Objects/Glass')
    root.AddTranslateOp().Set(Gf.Vec3d(*GLASS_XY, H))
    lathe(s, 'Glass/Tumbler', GLASS_PROFILE, (0., 0., 0.), glass_mat)
    authored('Glass', root.GetPrim(), f'tumbler {GLASS_R*2000:.0f} mm across, {GLASS_RIM*1000:.0f} mm tall (lathe proxy, OmniGlass)')
    point('glass_grasp', [GLASS_XY[0], GLASS_XY[1], H+.035], 'Right', ['forward'], 'pinch the glass from behind, jaws across it')

    ice = L.material(s, 'Ice', [.92, .96, 1.], .45)  # frosty, opaque: partial opacity vanishes inside the glass in RTX
    UsdGeom.Xform.Define(s, '/World/Objects/Ice')
    for k in range(POOL):
        x, y = park_xy(k)
        cube = UsdGeom.Xform.Define(s, f'/World/Objects/Ice/Cube{k}')
        cube.AddTranslateOp().Set(Gf.Vec3d(x, y, PARK_Z+CUBE_R/2))
        cube.AddOrientOp().Set(Gf.Quatf(1.))
        crescent(s, f'/World/Objects/Ice/Cube{k}/Visual', ice)
    authored('Ice', s.GetPrimAtPath('/World/Objects/Ice'), f'{POOL} crescent ice cubes {CUBE_L*1000:.0f} x {CUBE_R*2000:.0f} x '
             f'{CUBE_R*1000:.0f} mm in the ice bin (authored mesh, out of sight)')
    state = dict(dispenser='idle, power on', glass='empty, upright on the counter', cubes_in_bin=POOL, paddle_deg=0.,
                 paddle_pivot_m=[round(float(v), 4) for v in PADDLE_PIVOT], outlet_m=[round(float(v), 4) for v in NOZZLE])
    return state, None
