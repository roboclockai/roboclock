"""A3: static initial scene (appearance only; joints/rigid bodies deactivated)."""

import numpy as np
from pxr import Gf, Usd, UsdGeom, Vt

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import pan_on

T = L.TEXTURES
PAN_XY, PAN_LENGTH = (.30, .10), .55   # 28 cm-class pan: its flat floor leaves room to turn a 12 cm pancake over
PANCAKE_R, PANCAKE_H = .062, .008
PANCAKE_OFFSET = (.0406, -.0341)      # poured off-centre (5.3 cm against the turn direction), so the turned pancake lands as far from the wall as it started (was 4.5 cm: it landed on the wall slope at 157 deg and did not settle, eval hard 0)
TURNER_BEND = 30.                     # the native turner is flat; bent at the neck into an offset turner
# 0.60 m rig: at the 0.72 m-rig rest (neck (0.354, -0.25)) the handle grasp sat 0.23 m from the left base and
# the arm took it turned backwards (joint 1 -2.5 rad).
# Head-camera view: the turner now rests 11 cm forward (y -0.14, was -0.25 and out of view) on the counter strip
# freed by shrinking the cooktop's footprint to 64 % (front edge y -0.077; height, centre, burners and the pan
# position unchanged). Laid along +x there, its handle grasp was out of reach (a +x tool forward of the left
# base); turned 10 deg toward the left arm and 7.6 cm further right it has 0.30 rad of joint margin.
TURNER_Y = -.14
TURNER_YAW = 10.
COOKTOP_XY_SCALE = .64
TURNER_NECK = (.47, TURNER_Y)  # rest: blade flat on the counter toward +x (turned TURNER_YAW), handle rising toward -x
TURNER_GRIP = .12                     # grasp point along the handle from the neck (m)


def offset_turner(s, path, neck_xy, bend, yaw=0.):
    """Bend the native flat turner at its neck (points override) and lay it blade-flat with the handle
    rising toward -x. Returns (blade length, handle length, blade half-thickness) in metres."""
    prim = s.GetPrimAtPath(path)
    mesh = UsdGeom.Mesh([c for c in Usd.PrimRange(prim) if c.IsA(UsdGeom.Mesh)][0])
    M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()))
    v = np.asarray(mesh.GetPointsAttr().Get(), float)
    w = (np.c_[v, np.ones(len(v))]@M)[:, :3]
    m = w.mean(0)
    e1 = np.linalg.svd(w-m, full_matrices=False)[2][0]
    a = (w-m)@e1
    lo, hi = a.min(), a.max()
    width = [np.ptp((w[(a >= x) & (a < x+.01)]-m)@np.linalg.svd(w-m, full_matrices=False)[2][1]) for x in (lo, hi-.01)]
    if width[0] < width[1]:  # e1 must point from the (wide) blade toward the handle
        e1, a = -e1, -a
    lo, hi = a.min(), a.max()
    blade = a < lo+.10
    e3 = np.linalg.svd(w[blade]-w[blade].mean(0), full_matrices=False)[2][2]
    e2 = np.cross(e3, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    # neck: where the cross-section narrows from the blade to the handle
    xs = np.arange(lo, hi, .005)
    wid = np.array([np.ptp((w[(a >= x) & (a < x+.005)]-m)@e2) if ((a >= x) & (a < x+.005)).any() else 0. for x in xs])
    k = int(np.argmax(wid))
    neck = xs[k+np.argmax(wid[k:] < .5*wid.max())]  # first narrow slice past the widest one
    n = m+neck*e1
    b, c = (w-n)@e2, (w-n)@e3
    a = (w-n)@e1
    c -= np.median(c[a < -.02])  # blade mid-plane at c = 0
    # handle side: the centre line follows a 2 cm circular neck, then runs straight at the bend angle
    pm, ln = np.radians(bend), .02
    rho = ln/pm
    phi = pm*np.clip(a/ln, 0., 1.)
    run = np.maximum(a-ln, 0.)
    clx = np.where(a > 0, rho*np.sin(phi)+run*np.cos(pm), a)
    clz = np.where(a > 0, rho*(1-np.cos(phi))+run*np.sin(pm), 0.)
    ab = clx-np.where(a > 0, c*np.sin(phi), 0.)
    cb = clz+np.where(a > 0, c*np.cos(phi), c)
    half = float(np.percentile(np.abs(c[a < -.02]), 99))
    origin = np.array([neck_xy[0], neck_xy[1], H+half])
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    ex, ey = np.array([cy, sy, 0.]), np.array([-sy, cy, 0.])  # body x/y turned by yaw about z
    world = origin+np.outer(-ab, ex)+np.outer(b, -ey)+np.outer(cb, [0., 0., 1.])
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    xf.AddRotateZOp().Set(float(yaw))  # the rigid body frame carries the yaw
    inv = np.linalg.inv(np.array(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim())))
    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(((np.c_[world, np.ones(len(world))]@inv)[:, :3]).astype(np.float32)))
    mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([(world-origin).min(0), (world-origin).max(0)], np.float32)))
    return float(-a.min()), float(a.max()), half


def build(ctx):
    """Add the A3 task objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored, part_center, cooktop, flame_ring, rc = (ctx.s, ctx.obj, ctx.point, ctx.authored, ctx.part_center, ctx.cooktop, ctx.flame_ring, ctx.rc)
    state, agent = {}, None
    top = cooktop()
    ops = UsdGeom.Xformable(top).GetOrderedXformOps()  # translate, rotateXYZ, scale (scene_lib.put)
    ops[2].Set(Gf.Vec3f(COOKTOP_XY_SCALE, COOKTOP_XY_SCALE, 1.))
    lo, hi = L.bound(top)
    t = ops[0].Get()
    ops[0].Set(Gf.Vec3d(t[0]+.30-(lo[0]+hi[0])/2, t[1]+.10-(lo[1]+hi[1])/2, t[2]+H-lo[2]))
    pan_src = rc('pan', 2)
    pan, pan_d = pan_on(s, obj, 'FryingPan', pan_src, PAN_XY, PAN_LENGTH, H+.045, 'non-stick frying pan on the center burner, handle toward +x')
    bowl_c = [PAN_XY[0]+PANCAKE_OFFSET[0], PAN_XY[1]+PANCAKE_OFFSET[1]]
    fz = L.surface_z(pan, PAN_XY, H+.05)
    flame_ring('BurnerFlame', list(PAN_XY), H+.046, .27)
    batter = L.material(s, 'Batter', [.86, .70, .42], .7)
    cake = UsdGeom.Xform.Define(s, '/World/Objects/Pancake')  # one rigid body in simulation: body, top and underside
    cake.AddTranslateOp().Set(Gf.Vec3d(bowl_c[0], bowl_c[1], fz))
    L.shape(s, 'Body', [0., 0., PANCAKE_H/2], [PANCAKE_R, PANCAKE_H], batter, True, parent='/World/Objects/Pancake')
    top_m = L.material(s, 'PancakeTop', [1, 1, 1], .75, texture=T/'pancake_top_first_side.png')
    L.disc(s, 'Pancake/Top', [0., 0., PANCAKE_H+.0002], [PANCAKE_R, PANCAKE_R], top_m, lobes=(.015, .01))
    under = L.material(s, 'PancakeUnderside', [.86, .70, .42], .7)  # colour follows the browning in simulation
    L.disc(s, 'Pancake/Underside', [0., 0., -.0002], [PANCAKE_R, PANCAKE_R], under, lobes=(.015, .01), seed=1)
    authored('Pancake', cake.GetPrim(), f'pancake proxy ({PANCAKE_R*200:.1f} cm x {PANCAKE_H*1000:.0f} mm rigid disc, textured top, browned underside)')
    spat = rc('spatula', 0)
    obj('Spatula', spat, [TURNER_NECK[0]-.054, TURNER_Y-.05], H, L.scale_for(spat, .30), rot=(0, 0, 90),
        role=f'turner (native mesh bent {TURNER_BEND:g} deg at the neck into an offset turner) resting blade-flat on the counter in front of the cooktop')
    blade_len, handle_len, half = offset_turner(s, '/World/Objects/Spatula', TURNER_NECK, TURNER_BEND, TURNER_YAW)
    g = np.radians(TURNER_BEND)
    yw = np.radians(TURNER_YAW)
    point('spatula_handle', [TURNER_NECK[0]-TURNER_GRIP*np.cos(g)*np.cos(yw), TURNER_NECK[1]-TURNER_GRIP*np.cos(g)*np.sin(yw), H+half+TURNER_GRIP*np.sin(g)], 'Left', ['right_down20'],
          'grasp the turner handle along its axis, jaws across its flat faces')
    point('pancake_edge', [bowl_c[0]+.07, bowl_c[1], fz+.01], 'Right', ['left', 'down'], 'slide turner blade under pancake from the +x side')
    point('flip_above_pan', [bowl_c[0], bowl_c[1], fz+.12], 'Right', ['left', 'down'], 'lift and rotate turner (wrist roll limited to +-90 deg)')
    _c, klo, khi = part_center(str(top.GetPath())+'/Source/Stovetop012_knob_center')
    point('center_knob', [_c[0], _c[1], khi[2]+.01], 'Left', ['down'], 'burner control (optional heat adjustment)')
    state = dict(burner='center burner on, medium (gas ring visible)', pancake='first side cooking; bubbles forming', first_side_age_s='hidden; agent sees top-surface bubbles')
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
