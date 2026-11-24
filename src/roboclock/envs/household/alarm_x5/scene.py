"""Alarm-clock promo scene: static initial scene (appearance only; joints/rigid bodies deactivated).

A round twin-bell wind-up alarm clock ("Alarm Clock 01", Poly Haven, CC0; assets/x5/objects/alarm_clock_0) stands on
the island counter at its native size, near the front edge left of the right arm, dial toward the front (the promo
camera looks at it from there, level with the counter) (13.2 cm wide, 17.3 cm to the top of the handle).
Its stop button is authored: a brass collar on top of the carrying handle and, as a part of its own
(``AlarmButton``), a red cap on a short stem that the simulation overlay lets slide 6 mm down into the collar.
"""
import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import glass_mdl

CLOCK_USD = L.OBJECTS/'alarm_clock_0/alarm_clock.usda'
CLOCK_XY = (.20, -.30)  # the clock's vertical axis (the middle of the handle, where the button is): 0.40 m left of the right arm
# Clock frame = the asset's: origin on the counter under the case, z up, dial toward -y.
AXIS_Y = -.006  # the handle (and the hammer) stand 6 mm in front of the origin
HANDLE_TOP = .1729
COLLAR_R, COLLAR_Z = .0065, (.171, .175)
CAP_R, CAP_H = .011, .006
STEM_R, STEM_H = .0035, .010
BUTTON_LOCAL = (0., AXIS_Y, .182)  # AlarmButton origin = centre of the cap's underside, button up
TRAVEL = .006  # down to 1 mm above the collar


def build(ctx):
    """Add the alarm-clock objects to ctx.s; return the initial process state and agent camera."""
    s, obj, point, authored = ctx.s, ctx.obj, ctx.point, ctx.authored
    clock = obj('AlarmClock', CLOCK_USD, [CLOCK_XY[0], CLOCK_XY[1]], H,
                role='twin-bell wind-up alarm clock, dial toward the robot (free-standing)')
    op = UsdGeom.Xformable(clock).GetOrderedXformOps()[0]  # put() centres the bounding box: stand the axis on CLOCK_XY
    op.Set(Gf.Vec3d(CLOCK_XY[0], CLOCK_XY[1]-AXIS_Y, H))
    glass = s.GetPrimAtPath('/World/Objects/AlarmClock/Source/Glass')
    UsdShade.MaterialBindingAPI.Apply(glass).Bind(glass_mdl(s, 'ClockGlass', thin=True))
    UsdGeom.PrimvarsAPI(glass).CreatePrimvar('doNotCastShadows', Sdf.ValueTypeNames.Bool).Set(True)
    brass = L.material(s, 'ButtonBrass', [.72, .56, .26], .35, .9)
    red = L.material(s, 'ButtonRed', [.78, .05, .04], .38)
    L.shape(s, 'ButtonCollar', [0., AXIS_Y, sum(COLLAR_Z)/2], [COLLAR_R, COLLAR_Z[1]-COLLAR_Z[0]], brass, True,
            parent='/World/Objects/AlarmClock')
    origin = np.array([CLOCK_XY[0], CLOCK_XY[1], H+BUTTON_LOCAL[2]])
    button = UsdGeom.Xform.Define(s, '/World/Objects/AlarmButton')
    button.AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    L.shape(s, 'Cap', [0., 0., CAP_H/2], [CAP_R, CAP_H], red, True, parent='/World/Objects/AlarmButton')
    L.shape(s, 'Stem', [0., 0., .001-STEM_H/2], [STEM_R, STEM_H], brass, True, parent='/World/Objects/AlarmButton')
    authored('AlarmButton', button.GetPrim(), 'stop button: red cap on a stem in a brass collar on the clock handle; '
             f'{TRAVEL*1000:g} mm travel, latches down')
    top = origin+[0., 0., CAP_H]
    point('alarm_button', top, 'Right', ['forward_down'],
          'after the alarm starts ringing: press the button straight down until it latches, then let go')
    state = dict(alarm='silent; starts ringing at a hidden time', button_travel_mm=0.,
                 button_top_m=[round(float(v), 4) for v in top], clock_axis_xy_m=list(CLOCK_XY))
    return state, None  # head camera: fixed to the dual-arm rig (x5/rig.py)
