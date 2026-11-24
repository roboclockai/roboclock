"""A4: static initial scene (appearance only; joints/rigid bodies deactivated)."""

from pxr import UsdGeom

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import glass_mdl, lathe

T = L.TEXTURES
KETTLE_XY = (.285, .05)      # R2.1: placed for the two-arm hold (body axis at x .30)
JUG_XY = (.51, .05)          # jug mouth ~12 cm beyond the spout
SCALE_XY = (JUG_XY[0], JUG_XY[1]-.02)
DISPLAY = dict(center=(SCALE_XY[0], SCALE_XY[1]-.072), size=(.12, .045))
DIGIT_X = (-.010, .006, .022)    # three 7-segment digits on the LCD (card-local x)
SEGMENTS = dict(a=(0., .0105, .009, .0018), b=(.0045, .0053, .0018, .0095), c=(.0045, -.0053, .0018, .0095),
                d=(0., -.0105, .009, .0018), e=(-.0045, -.0053, .0018, .0095), f=(-.0045, .0053, .0018, .0095),
                g=(0., 0., .009, .0018))  # (x, y, width, height) in the digit cell
DIGIT_SEGS = {0: 'abcdef', 1: 'bc', 2: 'abdeg', 3: 'abcdg', 4: 'bcfg', 5: 'acdfg', 6: 'acdefg', 7: 'abc', 8: 'abcdefg', 9: 'abcdfg'}


def segments_on(value):
    """Visible segment names per digit for a right-aligned integer reading (blank leading zeros)."""
    txt = f'{int(max(0, min(999, value))):3d}'
    return [DIGIT_SEGS[int(ch)] if ch != ' ' else '' for ch in txt]


def build(ctx):
    """Add the A4 task objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored, rc = (ctx.s, ctx.obj, ctx.point, ctx.authored, ctx.rc)
    state, agent = {}, None
    ket_src = rc('kettle', 0)
    kettle = obj('Kettle', ket_src, list(KETTLE_XY), H, L.scale_for(ket_src, .23, 2), rot=(0, 0, 90),
                 role='cordless electric kettle, removed from base, filled with ~1.0 L hot water')
    scale_m = L.material(s, 'ScaleBody', [.86, .87, .86], .35)
    L.shape(s, 'KitchenScale', [*SCALE_XY, H+.011], [.19, .19, .022], scale_m)
    authored('KitchenScale', s.GetPrimAtPath('/World/Objects/KitchenScale'), 'digital kitchen scale (authored proxy, readout is the dose cue)')
    disp = L.material(s, 'ScaleLCD', [1, 1, 1], .3, texture=T/'scale_display_blank.png', emissive=(.25, .25, .25))
    L.card(s, 'ScaleDisplay', [*DISPLAY['center'], H+.0225], *DISPLAY['size'], disp)
    ink = L.material(s, 'LCDInk', [.06, .08, .06], .5)
    on = segments_on(0)
    for k, dx in enumerate(DIGIT_X):  # live 7-segment reading (simulation toggles visibility)
        for seg, (sx, sy, w, h) in SEGMENTS.items():
            prim = L.shape(s, f'ScaleDigit{k}{seg}', [DISPLAY['center'][0]+dx+sx, DISPLAY['center'][1]+sy, H+.0228], [w, h, .0004], ink)
            UsdGeom.Imageable(prim).GetVisibilityAttr().Set('inherited' if seg in on[k] else 'invisible')
    glass = glass_mdl(s, 'JugGlass')
    prof = [(0., .0), (.050, .0), (.050, .135), (.0465, .135), (.0465, .006), (0., .006)]
    jz = H+.0225
    jx, jy = JUG_XY
    jug = lathe(s, 'GlassJug', prof, [jx, jy, jz], glass)
    hm = glass
    L.shape(s, 'JugHandleTop', [jx+.065, jy, jz+.115], [.03, .012, .010], hm)
    L.shape(s, 'JugHandleBottom', [jx+.065, jy, jz+.035], [.03, .012, .010], hm)
    L.shape(s, 'JugHandleGrip', [jx+.077, jy, jz+.075], [.010, .012, .09], hm)
    grad = L.alpha_material(s, 'Graduation', T/'jug_graduations.png', emissive=False)
    L.card(s, 'JugGraduation', [jx, jy-.0505, jz+.072], .045, .12, grad, normal='front')
    authored('GlassJug', jug, 'graduated 500 ml glass jug (authored lathe proxy, OmniGlass)')
    card = L.material(s, 'RecipeCard', [1, 1, 1], .8, texture=T/'card_A4_recipe.png')
    L.card(s, 'RecipeCard', [.66, .16, H+.001], .15, .10, card)
    klo, khi = L.bound(kettle)
    point('kettle_handle', [klo[0]+.011, (klo[1]+khi[1])/2-.002, H+.1366], 'Left', ['forward'],
          'left: pinch the handle bar across x, one pad through the handle opening')
    point('kettle_lid_knob', [klo[0]+.1164, (klo[1]+khi[1])/2+.01, H+.2216], 'Right', ['forward_down'],
          'right: pinch the lid knob from the front-right (tool pitched 20 deg down)')
    point('pour_pose_spout', [jx, jy, jz+.16], 'Left', ['right', 'forward'], 'spout above jug mouth; both wrists roll to tilt the kettle')
    state = dict(kettle_water_ml=1000, jug_ml=0, scale_reading_g=0, target_g=300, tolerance_g=10)
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
