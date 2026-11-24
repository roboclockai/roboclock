"""C2: static initial scene (appearance only; joints/rigid bodies deactivated).

Layout R4 (2026-11-03, on the common rig: bases 0.60 m apart): the island is a bare counter (no cooktop, no pan).
Six tumblers stand side by side in one row, 10 cm apart, three outside each arm's base; each arm has an opened cola
bottle in front of its base (inner side). A bottle pours outward along the row: carried above the rims, it hovers beside a
glass on the inner side and tilts outward until its mouth is over that glass. The cola is a PhysX particle fluid inside the
bottle (simulation stage); the bottle is a slim 54 mm proxy so the 60.3 mm jaw opening clears it by 3 mm per side.
"""
import numpy as np
from pxr import Gf, UsdGeom

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import glass_mdl, lathe

ROW_Y = .10
GLASS_X = (-.245, -.145, -.045, .645, .745, .845)   # 10 cm apart; 1-3 left arm, 4-6 right arm (mirrored about x = 0.30)
GLASS_XY = [(x, ROW_Y) for x in GLASS_X]
# RoboDojo head camera (60 deg, 2026-11-03): at (-0.10, -0.04) / (0.70, -0.04) the 31.5 cm bottles were outside the
# head image (they rise toward the camera). Now in front of each arm, inside its base, clear of the ready-pose fingers.
BOTTLE_XY = [(.06, 0.), (.54, 0.)]
# bottle (origin at the base centre): body outer r 27 mm (wall 3 mm) to 26 cm, shoulder, neck (inner r 11 mm)
BOTTLE_R, BOTTLE_R_IN, BODY_TOP, NECK_R_IN, MOUTH_Z = .027, .024, .26, .011, .315
BOTTLE_PROFILE = [(0., 0.), (.024, 0.), (.027, .006), (.027, BODY_TOP), (.0135, BODY_TOP+.02), (.0135, MOUTH_Z),
                  (NECK_R_IN, MOUTH_Z), (NECK_R_IN, BODY_TOP+.02)]
# tumbler (origin at the base centre): outer r 29 mm, inner r 25.5 mm, 6 mm base, rim at 8.5 cm
GLASS_R, GLASS_R_IN, GLASS_FLOOR, GLASS_RIM = .029, .0255, .006, .085
GLASS_PROFILE = [(0., 0.), (GLASS_R, 0.), (GLASS_R, GLASS_RIM), (GLASS_R_IN, GLASS_RIM), (GLASS_R_IN, GLASS_FLOOR), (0., GLASS_FLOOR)]
TARGET_ML = 90.
# white line at 90 ml: the particle fluid rests 1.5 mm off the walls (its rest offset), so its free radius is 24 mm
LINE_Z = GLASS_FLOOR+TARGET_ML*1e-6/(np.pi*(GLASS_R_IN-.0015)**2)


def build(ctx):
    """Add the C2 task objects to ctx.s; return the initial process state and agent camera."""
    s, point, authored = ctx.s, ctx.point, ctx.authored
    pet = L.material(s, 'ColaBottle', [.075, .028, .014], .12)      # cola seen through brown-tinted PET
    label = L.material(s, 'ColaLabel', [.50, .05, .06], .45)       # plain label band, no text
    for k, (name, (x, y)) in enumerate(zip(('BottleL', 'BottleR'), BOTTLE_XY)):
        root = UsdGeom.Xform.Define(s, f'/World/Objects/{name}')   # one rigid body in simulation
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, H))
        lathe(s, f'{name}/Body', BOTTLE_PROFILE, (0., 0., 0.), pet)
        lathe(s, f'{name}/Label', [(BOTTLE_R+.0004, .09), (BOTTLE_R+.0004, .17)], (0., 0., 0.), label)
        side = ('left', 'right')[k]
        authored(name, root.GetPrim(), f'opened cola bottle for the {side} arm (54 x 315 mm lathe proxy, no cap; '
                                       'the cola is a particle fluid added in simulation)')
        point(f'bottle_{side}', [x, y, H+.12], ('Left', 'Right')[k], ['forward_down20'],
              f'{side} arm: pinch the bottle body 12 cm up, tool 20 deg down')
    glass = glass_mdl(s, 'TumblerGlass')
    line = L.material(s, 'FillLine', [.96, .96, .95], .35)
    for j, (x, y) in enumerate(GLASS_XY):
        n = j+1
        root = UsdGeom.Xform.Define(s, f'/World/Objects/Glass{n}')
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, H))
        lathe(s, f'Glass{n}/Tumbler', GLASS_PROFILE, (0., 0., 0.), glass)
        lathe(s, f'Glass{n}/FillLine', [(GLASS_R+.0003, LINE_Z-.0008), (GLASS_R+.0003, LINE_Z+.0008)], (0., 0., 0.), line)
        authored(f'Glass{n}', root.GetPrim(), f'tumbler {n} (about 170 ml to the rim, white line at {TARGET_ML:.0f} ml; lathe proxy, OmniGlass)')
        point(f'glass{n}_mouth', [x, y, H+GLASS_RIM], ('Left', 'Right')[j//3], ['pour'], f'pour glass {n} to the line')
    state = dict(counter='bare island counter', bottles='two opened cola bottles, one per arm', glasses='six empty tumblers in a row',
                 fill_line_ml=TARGET_ML, served=0)
    agent = ([.30, -.62, 1.48], [.30, .10, .90], 20.)
    return state, agent
