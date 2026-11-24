"""B1: static initial scene (appearance only; joints/rigid bodies deactivated).

The faucet lever (native Sink054_handle on a D6 joint, flow tilt 0-15 deg about y) is authored as a
separate unscaled part ``FaucetLever`` rotated to the open position, so the simulation overlay can
hinge it and the robot can push it closed.

The sink is empty; a mixing bowl stands in the basin right under the spout and fills from it. The water
rises about 4 mm/s there, against a fraction of a millimetre in a full sink, so the camera sees it."""
import math

import numpy as np
from pxr import Gf, UsdGeom

from roboclock.envs.household.b1_x5.task import BOWL_LEVEL_CM, BOWL_RADIUS_CM, BOWL_VOLUME_L, B1Spec
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import (
    box_mesh,
    cut_island_top,
    island_top_material,
    reparent,
    squash,
    world,
)

K = L.KITCHEN
SINK = K/'Sink054/Sink054.usd'
LEVER_OPEN_DEG = 15.  # native D6 flow range rotY 0 (closed) .. 15 deg (fully open)
SPOUT_XY = (.4872, .13)  # where the stream falls
STREAM_TOP = H+.30
# The asset is a shallow bowl, stretched here into a 1.6 L mixing bowl: 23.5 cm across, 15 cm tall (the task's profile
# table is measured at these scales). The foot stands on the highest point of the sloped basin floor under it.
BOWL_SCALE, BOWL_Z_SCALE = 1.775, 2.727
BOWL_Z = H-.2155
BOWL_FLOOR_Z = BOWL_Z+.0767  # inner floor
BOWL_RIM_R, BOWL_RIM_Z = .1135, BOWL_Z+.15
FOAM, SPILLS = 7, 8


def build(ctx):
    """Add the B1 task objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored, island, rc = ctx.s, ctx.obj, ctx.point, ctx.authored, ctx.island, ctx.rc
    state, agent = {}, None
    stone = island_top_material(s, island)
    cut_island_top(island, opening=(-.09, -.155, .69, .315, H-.31))  # the counter strips below cover what lies outside it
    sink = obj('Sink', SINK, [.30, .08], H+.045-.29404225945, role='native island sink with pull-down faucet', native=True)
    for n in ['Shelf001', 'Shelf002']:
        pp = s.GetPrimAtPath(str(sink.GetPath())+'/Source/Sink054_'+n)
        if pp:
            pp.SetActive(False)
    for name, x0, x1, y0, y1 in [('Left', -.27481, -.09, -.38067, .38067), ('Right', .69, .87481, -.38067, .38067),
                                 ('Front', -.09, .69, -.38067, -.155), ('Back', -.09, .69, .315, .38067)]:
        box_mesh(s, 'SinkCounter'+name, [x0, y0, H-.034], [x1, y1, H], stone)
    x, y = SPOUT_XY
    bowl = obj('WaterBowl', rc('bowl', 1), [x, y], BOWL_Z, BOWL_SCALE, role='mixing bowl in the basin under the spout (fixed)')
    squash(bowl, BOWL_Z_SCALE/BOWL_SCALE, BOWL_Z)
    start_l = BOWL_VOLUME_L[-1]-B1Spec.headroom_l
    level = float(np.interp(start_l, BOWL_VOLUME_L, BOWL_LEVEL_CM))
    water_z = BOWL_FLOOR_Z+level/100
    water = L.material(s, 'BowlWater', [.60, .76, .84], .06, opacity=.72, ior=1.33)
    surface = UsdGeom.Xformable(L.disc(s, 'WaterSurface', [0, 0, 0], [1, 1], water, segments=96, lobes=(0, 0)))  # unit disc
    surface.AddTranslateOp().Set(Gf.Vec3d(x, y, water_z))
    radius = float(np.interp(level, BOWL_LEVEL_CM, BOWL_RADIUS_CM))/100
    surface.AddScaleOp().Set(Gf.Vec3f(radius, radius, 1))
    authored('WaterSurface', s.GetPrimAtPath('/World/Objects/WaterSurface'), 'surface of the water in the bowl, follows the bowl profile (disclosed proxy)')
    foam = L.material(s, 'WaterAeration', [.90, .93, .94], .5)
    rng = np.random.default_rng(7)
    for i in range(FOAM):
        ang, dist = rng.uniform(0, 6.28), (0. if i == 0 else rng.uniform(.012, .034))
        cx, cy = x+dist*math.cos(ang), y+dist*math.sin(ang)
        rad = .014 if i == 0 else rng.uniform(.004, .009)
        L.disc(s, f'Foam{i}', [cx, cy, water_z+.0012+.0002*i], [rad, rad*rng.uniform(.6, 1.)], foam, seed=i, lobes=(.06, .04), segments=48)
    authored('WaterAeration', s.GetPrimAtPath('/World/Objects/Foam0'), 'bubbles where the stream hits the water')
    stream = L.material(s, 'Stream', [.82, .9, .93], .1, opacity=.55)
    L.shape(s, 'RunningWater', [x, y, (STREAM_TOP+water_z)/2], [.0045, STREAM_TOP-water_z], stream, True)
    # hidden until the bowl overflows: thin streams over the rim, a puddle spreading on the basin floor
    for i in range(SPILLS):
        ang = 2*math.pi*i/SPILLS
        spill = L.shape(s, f'Spill{i}', [x+BOWL_RIM_R*math.cos(ang), y+BOWL_RIM_R*math.sin(ang), (BOWL_RIM_Z+BOWL_Z)/2],
                        [.0028, BOWL_RIM_Z-BOWL_Z], stream, True)
        UsdGeom.Imageable(spill).MakeInvisible()
    puddle = UsdGeom.Xformable(L.disc(s, 'Puddle', [0, 0, 0], [1, 1], water, segments=96, lobes=(0, 0)))  # unit disc
    puddle.AddTranslateOp().Set(Gf.Vec3d(x, y, BOWL_Z+.002))
    puddle.AddScaleOp().Set(Gf.Vec3f(.125, .125, 1))
    puddle.MakeInvisible()
    authored('Spill', s.GetPrimAtPath('/World/Objects/Spill0'), 'water over the rim and on the basin floor, shown only after an overflow')
    src = str(sink.GetPath())+'/Source'
    handle_m = world(s, src+'/Sink054_handle')
    pivot = handle_m[3, :3]  # native handle origin = D6 joint frame
    s.GetPrimAtPath(src+'/Sink054_handle').SetActive(False)
    lever = reparent(s, SINK, '/World/Objects/FaucetLever', '/root/Sink054_handle', handle_m, pivot, src+'/Looks')
    UsdGeom.Xformable(lever).AddRotateYOp().Set(LEVER_OPEN_DEG)
    authored('FaucetLever', lever, f'native single lever, re-authored on its pivot; tilted {LEVER_OPEN_DEG:g} deg = fully open')
    hlo, hhi = L.bound(lever)
    point('faucet_lever', [hhi[0]-.008, (hlo[1]+hhi[1])/2, hhi[2]-.012], 'Right', ['forward_down'],
          'push the lever top toward the spout until it is closed (tilt 0 deg), then release')
    state = dict(faucet='gentle stream, sudden supply surge after a delay', lever_tilt_deg=LEVER_OPEN_DEG, lever_pivot_m=[round(float(v), 4) for v in pivot],
                 bowl_capacity_l=BOWL_VOLUME_L[-1], water_in_bowl_l=round(start_l, 3))
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
