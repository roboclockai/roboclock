"""C1: static initial scene (appearance only; joints/rigid bodies deactivated).

R3 layout (serving relay at the pass, 2026-11-06; replaces the R2.2 microwave relay):
- Each meal bowl has two lug handles on one axis: the right arm carries it by handle A, the left
  arm by handle B. The handles sweep up 20 deg and are pinched with the tool pitched 20 deg down:
  with the jaw axis near vertical the palm reaches 7.6 cm below the tool point, so a flat handle
  at rim height would put the palm on the counter. (Unchanged from R2.2.)
- An X5 holds a bowl by a handle only beyond ~0.28 m from its own base. The three waiting meals
  stand in front of the left arm (slots in task.py), out of the right arm's reach; the serving
  board stands in front of the right arm, out of the left arm's reach. Both reach the staging mat
  on the line between the bases (R2.2 position), where one handle axis suits both arms. The left
  arm never passes over the board and the right arm never over the waiting meals.
- The first meal starts on the mat. Seeded episodes move the bowls to ``task.layout(seed)`` when
  the simulation starts (sim.py); this file builds the nominal (seed 0) layout.
- Under the rendered head-camera model the mat and the bowl on it sit just below the head image:
  they are seen by the wrist cameras (x5/visibility.py WRIST_ONLY, as in R2.2).
"""
import numpy as np
from pxr import UsdGeom

from roboclock.envs.household.c1_x5.task import MEALS, layout
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H
from roboclock.envs.household.x5.scene_parts import body_visual

T = L.TEXTURES
BOWL_SCALE = .8  # 0.60 m rig: 10 cm bowls (12.5 cm before); the 6 mm lug handles keep their size for the jaws
BOWL_DIAMETER = .125*BOWL_SCALE
TAB = dict(length=.03, width=.032, thickness=.006)  # lug handle: radial length, width, thickness
HANDLE_PITCH = 20.  # handle upsweep (deg); the grasp tool pitch matches it
HANDLE_ROOT = BOWL_DIAMETER/2-.004  # handle root radius (inset into the rim)
STAGE, STAGE_YAW = (.30, -.25), 115.   # staging mat: left arm sets the bowl down (handle B), right arm takes handle A
MAT_SIZE = (.15, .15, .003)
BOARD = dict(center=(.58, .27), size=(.11, .11, .018))
BOARD_YAW = 22.5                        # served bowl body yaw (right tool azimuth 112.5 deg)
GRASP_R = HANDLE_ROOT+.0222  # grasp point (tool point) radius from the bowl axis
FOODS = [f'meal_{m}.png' for m in MEALS]


def handle_z(h):
    """Grasp-point height above the bowl base on the upswept handle's mid-plane (bowl body height h)."""
    return h-.002-TAB['thickness']/2+(GRASP_R-HANDLE_ROOT)*np.tan(np.radians(HANDLE_PITCH))


def handle_point(center, yaw, side, h):
    """Tool point on handle A (Right) or B (Left) of a bowl at `center` with body yaw `yaw` (deg)."""
    a = np.radians(yaw-90. if side == 'Right' else yaw+90.)
    return [center[0]+GRASP_R*np.cos(a), center[1]+GRASP_R*np.sin(a), center[2]+handle_z(h)]


def build(ctx):
    """Add the C1 task objects to ctx.s; return the initial process state and agent camera."""
    s, point, authored, wood, rc = ctx.s, ctx.point, ctx.authored, ctx.wood, ctx.rc
    order, slots = layout(0)
    source = rc('bowl', 5)  # the deep bowl: its handles sit high enough for the palm to clear the counter and the board
    ceramic = [L.material(s, 'CeramicSand', [.62, .58, .52], .45), L.material(s, 'CeramicSlate', [.16, .17, .18], .4)]
    h = None
    for i, food in enumerate(FOODS):
        name = f'MealBowl{i+1}'
        on_mat = slots[i] is None
        x, y, yaw = (*STAGE, STAGE_YAW) if on_mat else slots[i]
        z = H+MAT_SIZE[2] if on_mat else H
        bowl = body_visual(s, '/World/Objects/'+name, source, [x, y, z], L.scale_for(source, BOWL_DIAMETER))
        lo, hi = L.bound(bowl)
        h = hi[2]-lo[2]
        fm = L.material(s, f'Food{i}', [1, 1, 1], .55, texture=T/food)
        L.disc(s, f'{name}/Food', [0., 0., h-.012], [.045, .045], fm, seed=i, lobes=(.03, .02))
        a_up = np.radians(HANDLE_PITCH)
        n = TAB['length']+.008
        for k, sign in enumerate((-1, 1)):  # handles on the local y axis; -y is handle A (right arm side)
            L.shape(s, 'Handle'+'AB'[k], [0., sign*(HANDLE_ROOT+n/2*np.cos(a_up)), h-.002-TAB['thickness']/2+n/2*np.sin(a_up)],
                    [TAB['width'], n, TAB['thickness']], ceramic[i % 2], rot=(sign*HANDLE_PITCH, 0., 0.), parent='/World/Objects/'+name)
        UsdGeom.Xformable(bowl).AddRotateZOp().Set(float(yaw))
        authored(name, bowl, f'{MEALS[i]} in a ceramic bowl ({BOWL_DIAMETER*100:.1f} cm) with two authored lug handles', source=source)
        if on_mat:
            point(f'bowl{i+1}_mat_handle', handle_point([x, y, z], yaw, 'Right', h), 'Right', ['left_down20'],
                  f'{MEALS[i]} on the staging mat: right arm takes it by handle A')
        else:
            point(f'bowl{i+1}_handle', handle_point([x, y, z], yaw, 'Left', h), 'Left', ['forward_down20', 'forward_right_down20'],
                  f'grasp the {MEALS[i]} bowl by handle B (jaws across the handle thickness, tool pitched {HANDLE_PITCH:g} deg down)')
    L.shape(s, 'StagingMat', [*STAGE, H+MAT_SIZE[2]/2], list(MAT_SIZE), L.material(s, 'SiliconeMat', [.55, .12, .10], .7))
    authored('StagingMat', s.GetPrimAtPath('/World/Objects/StagingMat'), 'staging mat between the arms (relay point)')
    L.shape(s, 'ServingBoard', [*BOARD['center'], H+BOARD['size'][2]/2], BOARD['size'], wood)
    authored('ServingBoard', s.GetPrimAtPath('/World/Objects/ServingBoard'), 'serving board at the pass (one bowl)')
    card = L.material(s, 'OrderCard', [1, 1, 1], .8, texture=T/'card_C1_orders.png')
    L.card(s, 'OrderCard', [-.18, -.12, H+.001], .16, .10, card, yaw=0)
    point('staging_place', handle_point([*STAGE, H+MAT_SIZE[2]], STAGE_YAW, 'Left', h), 'Left', ['right_down20'],
          'left arm sets the next meal down on the staging mat (handle B)')
    point('board_place', handle_point([*BOARD['center'], H+BOARD['size'][2]], BOARD_YAW, 'Right', h), 'Right',
          ['left_forward_down20'], 'right arm sets the meal down on the serving board (handle A)')
    state = dict(meals=list(MEALS), order=[MEALS[i] for i in order], service='one bowl on the board; a waiter takes it at each pickup',
                 waiting_bowls=[dict(meal=MEALS[i], center_m=list(c[:2]), body_yaw_deg=c[2]) for i, c in enumerate(slots) if c],
                 staging_mat_m=list(STAGE), staging_yaw_deg=STAGE_YAW, board_m=list(BOARD['center']), board_yaw_deg=BOARD_YAW)
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
