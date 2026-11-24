"""A1: static initial scene (appearance only; joints/rigid bodies deactivated).

Sugar for a glass of tea (2026-11-19, replaces the tea-infuser version): a clear glass of tea stands where the mug
stood (same cavity, so the stir is unchanged), an open glass jar of sugar stands to its left for the left arm, the
spoon in its wire basket to its right for the right arm. The sugar that lands in the glass piles up in a white mound
on its bottom (``SugarMound``, sized by sim.py from the task's grams; empty at the start), seen through the tea from
the head camera; ``SugarStream`` is the falling stream and ``SugarJar/Sugar`` the sugar left in the jar.
"""
import json

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import glass_mdl, lathe

K = L.KITCHEN
R = L.REUSED
# Basket and spoon moved 12 cm toward the glass and 6 cm back for the 0.60 m rig (basket centre (0.34, 0.10)):
# at the old spot the spoon handle sat 0.29 m from the right base and every handle grasp put a wrist
# joint at its stop; 12 cm left alone still knocked the spoon over while sliding onto the handle.
SPOON_SHIFT = (-.12, .06)
CUP = (.2080456, .1400678)
# glass (origin at the base centre): 3 mm wall around the fixed cavity collider (inner face r 39 mm, 39.5 mm in the
# upper half), 7.5 mm base, rim 81.4 mm; tea up to 64 mm (sim.py TEA_Z)
GLASS_PROFILE = [(0., 0.), (.0420, 0.), (.0425, .0814), (.0395, .0814), (.0390, .0075), (0., .0075)]
TEA_TOP = .064
TEA_PROFILE = [(0., .0077), (.0388, .0077), (.0392, TEA_TOP), (0., TEA_TOP)]
# jar: "Glass Jar" by Extrone (CC BY 4.0), scaled to 54 mm across and 73.6 mm tall (assets/x5/objects/sugar_jar_0)
JAR = L.OBJECTS/'sugar_jar_0/sugar_jar.usda'
JAR_XY = (.06, .14)
JAR_R_IN, JAR_FLOOR, JAR_SHOULDER, JAR_MOUTH_R, JAR_MOUTH_Z, JAR_R = .0249, .003, .0598, .0150, .0736, .0270
JAR_GRIP_Z = .035
HIDDEN = 1e-4  # scale of a cue prim that is not shown (prims are switched by scale, not visibility: RTX lag)


def no_shadow(prim):
    """RTX Real-Time shadows refractive glass as if it were opaque: whatever stands inside a vessel is unlit (the
    sugar in the jar and under the tea rendered black). Glass and tea cast no shadow (as lab liquid_scene.no_shadow)."""
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdGeom.Mesh):
            UsdGeom.PrimvarsAPI(p).CreatePrimvar('doNotCastShadows', Sdf.ValueTypeNames.Bool).Set(True)


def build(ctx):
    """Add the A1 task objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored, rubber = ctx.s, ctx.obj, ctx.point, ctx.authored, ctx.rubber
    sugar = L.material(s, 'Sugar', [.93, .92, .89], .85)
    root = UsdGeom.Xform.Define(s, '/World/Objects/TeaGlass')
    root.AddTranslateOp().Set(Gf.Vec3d(CUP[0], CUP[1], H))
    lathe(s, 'TeaGlass/Glass', GLASS_PROFILE, (0., 0., 0.), glass_mdl(s, 'TeaGlassGlass'))
    lathe(s, 'TeaGlass/Tea', TEA_PROFILE, (0., 0., 0.), glass_mdl(s, 'Tea', color=(.82, .50, .20), ior=1.33))
    no_shadow(root.GetPrim())
    authored('TeaGlass', root.GetPrim(), 'glass of tea, 84 mm across, 81 mm tall, tea 56 mm deep (lathe proxy, OmniGlass; '
                                         'the tea is a tinted clear volume so the sugar on the bottom shows)')
    for name, prim in (('SugarMound', UsdGeom.Cone.Define(s, '/World/Objects/SugarMound')),
                       ('SugarStream', UsdGeom.Cylinder.Define(s, '/World/Objects/SugarStream'))):
        prim.CreateRadiusAttr(1.)
        prim.CreateHeightAttr(1.)
        prim.CreateAxisAttr('Z')
        prim.AddTranslateOp().Set(Gf.Vec3d(CUP[0], CUP[1], H+.0080))
        prim.AddScaleOp().Set(Gf.Vec3f(HIDDEN))
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(sugar)
        authored(name, prim.GetPrim(), 'sugar mound on the glass bottom (cone at the slope of settled sugar, sized by the '
                                       'grams in the glass; none at the start)' if name == 'SugarMound' else
                                       'falling sugar stream (shown while sugar runs from the jar)')
    jar = obj('SugarJar', JAR, JAR_XY, H, override=glass_mdl(s, 'JarGlass'),
              role='open glass jar of sugar, 54 x 74 mm ("Glass Jar" by Extrone, CC BY 4.0; cap left out)')
    no_shadow(jar)
    fill = UsdGeom.Cylinder.Define(s, str(jar.GetPath())+'/Sugar')
    fill.CreateRadiusAttr(1.)
    fill.CreateHeightAttr(1.)
    fill.CreateAxisAttr('Z')
    height = .048  # nominal 75 g; sim.py resizes it to the jar's actual sugar at reset
    fill.AddTranslateOp().Set(Gf.Vec3d(0., 0., JAR_FLOOR+height/2))
    fill.AddScaleOp().Set(Gf.Vec3f(JAR_R_IN-.0007, JAR_R_IN-.0007, height))
    UsdShade.MaterialBindingAPI.Apply(fill.GetPrim()).Bind(sugar)
    obj('SpoonBasket', K/'KitchenRoom_A1.usdc', [.46+SPOON_SHIFT[0], .04+SPOON_SHIFT[1]], H, .5, prim='/root/Kitchen_Basket',
        hide=['Kitchen_Basket00'+str(i) for i in [2, 3, 4, 5, 6]], override=rubber, role='utensil basket holding the spoon')
    row = json.loads((R/'a1_measured_reset_pose.json').read_text())
    sp = obj('Spoon', R/'spoon/spoon_2.usd', [.46+SPOON_SHIFT[0], .04+SPOON_SHIFT[1]], H+.004, 1.2, role='stirring spoon (HANDAL scan)')
    xf = UsdGeom.Xformable(sp)
    xf.ClearXformOpOrder()
    q = row['tool_quaternions_wxyz'][1]
    sp_pos = [v+d for v, d in zip(row['tool_positions'][1], (*SPOON_SHIFT, 0.))]
    xf.AddTranslateOp().Set(Gf.Vec3d(*sp_pos))
    xf.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    xf.AddScaleOp().Set(Gf.Vec3f(1.2))
    point('jar_grasp', [JAR_XY[0], JAR_XY[1], H+JAR_GRIP_Z], 'Left', ['forward'], 'pinch the jar body 3.5 cm up, tool 10 deg down')
    point('glass_mouth', [CUP[0], CUP[1], H+.0814], 'Left', ['pour'], 'tilt the jar mouth over the glass toward +x')
    point('spoon_handle', [.474+SPOON_SHIFT[0], .002+SPOON_SHIFT[1], H+.09], 'Right', ['down', 'forward_down'], 'grasp spoon handle in basket')
    point('stir_center', [CUP[0], CUP[1], H+.14], 'Right', ['forward_down', 'down'], 'tool point while the held spoon tip is in the tea (spoon ~0.12 m)')
    state = dict(glass='tea, no sugar', jar_sugar_g='75 nominal (hidden 60-85)', spoon='in its basket')
    return state, None  # head camera: fixed to the dual-arm rig (x5/rig.py)
