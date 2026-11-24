"""B2: board on the counter with four objects held by pegs (static initial scene).

One object is let go per episode. The board is a wedge whose thin end meets the counter, so an object
gathers speed on it and keeps rolling across the counter toward the front edge, into the left hand.
"""
import math

import numpy as np
from pxr import Gf, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation as R

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H

HEADING = 240.                           # rolling direction (deg): toward the front edge and the left arm
D = np.array([math.cos(math.radians(HEADING)), math.sin(math.radians(HEADING))])
U = np.array([-D[1], D[0]])              # across the lanes
TOE = np.array([.40, -.03])              # middle of the board's thin end (+6 cm in x with the 0.60 m rig:
                                         # the cup lane's catch pose was 7 cm from the left base)
BOARD_L, BOARD_W = .27, .40
START = .225                             # objects rest this far up the board from the toe
EDGE_Y = -.38                            # front edge of the counter
OBJECTS = {  # name: lane offset, diameter, length, mass (kg)
    'cup': dict(s=-.12, d=.050, l=.086, m=.095),
    'egg': dict(s=-.04, d=.044, l=.058, m=.058),
    'pill_bottle': dict(s=.04, d=.041, l=.074, m=.060),
    'apple': dict(s=.12, d=.052, l=.052, m=.085),
}


def board_points(deg):
    """Wedge: thin end on the counter along the toe, rising uphill by deg over BOARD_L."""
    t = BOARD_L*math.tan(math.radians(deg))
    pts = []
    for along, top in ((0., 0.), (BOARD_L, t)):
        for side in (-BOARD_W/2, BOARD_W/2):
            xy = TOE-D*along+U*side
            pts += [[xy[0], xy[1], H], [xy[0], xy[1], H+top+.0005]]
    return pts


def slope_rotation(deg):
    """Local x downhill along the board, local z its normal."""
    return R.from_euler('z', HEADING, degrees=True)*R.from_euler('y', deg, degrees=True)


def on_board(s, along, height, deg):
    """World point at lane offset s, `along` uphill of the toe, `height` above the board along its normal."""
    xy = TOE+U*s-D*along
    z = H+.0005+along*math.tan(math.radians(deg))
    return np.array([*xy, z])+slope_rotation(deg).apply([0, 0, height])


def object_pose(name, deg):
    """Rest pose: on the board, against its peg. The lathe axis lies across the lane (apple: stem up)."""
    o = OBJECTS[name]
    p = on_board(o['s'], START, o['d']/2, deg)
    axis = R.from_euler('z', HEADING, degrees=True)*R.from_euler('x', 90, degrees=True)
    return p, (R.from_euler('z', HEADING, degrees=True) if name == 'apple' else axis)


def peg_pose(name, deg):
    o = OBJECTS[name]
    return on_board(o['s'], START-o['d']/2-.006, .006, deg), slope_rotation(deg)


def set_pose(prim, xyz, r):
    """Translate + orient ops (this module also loads outside Kit, without the PhysX schemas)."""
    q = r.as_quat()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))


def lathe(s, path, profile, mat, seg=48, arc=(0., 360.)):
    """Surface of revolution about local z. profile: (r, z) rows; r = 0 closes an end."""
    prof = np.asarray(profile, float)
    closed = arc[1]-arc[0] >= 360.
    n = seg if closed else seg+1
    a = np.radians(np.linspace(arc[0], arc[1], seg+1))[:n]
    pts = np.array([[r*math.cos(t), r*math.sin(t), z] for r, z in prof for t in a])
    tang = np.gradient(prof, axis=0)
    nr, nz = tang[:, 1], -tang[:, 0]
    nrm = np.array([[nr[i]*math.cos(t), nr[i]*math.sin(t), nz[i]] for i in range(len(prof)) for t in a])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    faces = []
    for i in range(len(prof)-1):
        for j in range(seg):
            k = (j+1) % n
            faces += [i*n+j, i*n+k, (i+1)*n+k, (i+1)*n+j]
    m = UsdGeom.Mesh.Define(s, path)
    m.CreatePointsAttr(pts.tolist())
    m.CreateNormalsAttr(nrm.tolist())
    m.SetNormalsInterpolation('vertex')
    m.CreateFaceVertexCountsAttr([4]*(len(faces)//4))
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateExtentAttr([pts.min(0).tolist(), pts.max(0).tolist()])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def ovoid(rmax, half, taper=0., n=33):
    """Egg profile along z: taper > 0 narrows the +z end."""
    t = np.linspace(0, math.pi, n)
    return np.c_[rmax*np.sin(t)*(1-taper*np.cos(t)), half*np.cos(t)]


def apple(rmax, n=41):
    t = np.linspace(.0, math.pi, n)
    rho = rmax*(1-.30*np.exp(-(t/.42)**2)-.22*np.exp(-((math.pi-t)/.38)**2))
    return np.c_[rho*np.sin(t), .95*rho*np.cos(t)]


def visuals(s, root, name, looks):
    o = OBJECTS[name]
    r, ln = o['d']/2, o['l']/2
    if name == 'egg':
        lathe(s, root+'/Shell', ovoid(r, ln, .10), looks['shell'])
        lathe(s, root+'/Stamp', ovoid(r+.0002, ln+.0002, .10, 33)[12:20], looks['stamp'], seg=12, arc=(-22., 22.))
    elif name == 'apple':
        lathe(s, root+'/Skin', apple(r), looks['apple'], seg=64)
        lathe(s, root+'/Blush', apple(r+.0002, 41)[8:30], looks['blush'], seg=20, arc=(-70., 70.))
        lathe(s, root+'/Stem', [[.0013, r*.62], [.0012, r*.80], [.0017, r*.96], [0., r*.97]], looks['stem'], seg=10)
    elif name == 'pill_bottle':
        body = ln*.62
        lathe(s, root+'/Body', [[0., -ln], [r*.94, -ln], [r, -ln+.002], [r, body], [r*.86, body+.002], [r*.86, body+.006]], looks['amber'], seg=64)
        lathe(s, root+'/Cap', [[r*1.0, body+.004], [r*1.0, ln-.002], [r*.94, ln], [0., ln]], looks['cap'], seg=64)
        lathe(s, root+'/Label', [[r+.0002, -ln*.72], [r+.0002, ln*.34]], looks['label'], seg=30, arc=(-115., 115.))
    else:  # open plastic tumbler
        lathe(s, root+'/Cup', [[0., -ln], [r*.90, -ln], [r, -ln+.004], [r, ln], [r*.95, ln], [r*.95, -ln+.006], [0., -ln+.006]], looks['cup'], seg=64)
        lathe(s, root+'/Band', [[r+.0002, -ln*.1], [r+.0002, ln*.4]], looks['band'], seg=24, arc=(-60., 60.))


def build(ctx):
    """Add the B2 task objects to ctx.s; return the initial process state and agent camera."""
    s, point, authored = ctx.s, ctx.point, ctx.authored
    deg = 6.
    wood = L.material(s, 'BoardWood', [.46, .29, .15], .55)
    board = UsdGeom.Mesh.Define(s, '/World/Objects/Board')
    pts = board_points(deg)
    board.CreatePointsAttr(pts)
    board.CreateFaceVertexCountsAttr([4]*6)
    board.CreateFaceVertexIndicesAttr([0, 2, 6, 4, 1, 5, 7, 3, 0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6])
    board.CreateSubdivisionSchemeAttr('none')
    board.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(board.GetPrim()).Bind(wood)
    authored('Board', board.GetPrim(), 'cutting board propped into a wedge; its thin end meets the counter')
    looks = dict(shell=L.material(s, 'EggShell', [.86, .71, .55], .5), stamp=L.material(s, 'EggStamp', [.62, .10, .12], .6),
                 apple=L.material(s, 'AppleSkin', [.62, .05, .06], .34), blush=L.material(s, 'AppleBlush', [.80, .52, .14], .36),
                 stem=L.material(s, 'AppleStem', [.20, .12, .06], .7),
                 amber=L.material(s, 'BottleAmber', [.62, .27, .05], .25), cap=L.material(s, 'BottleCap', [.90, .90, .88], .4),
                 label=L.material(s, 'BottleLabel', [.93, .93, .90], .6),
                 cup=L.material(s, 'CupBlue', [.10, .34, .62], .3), band=L.material(s, 'CupBand', [.92, .78, .22], .4))
    peg = L.material(s, 'Peg', [.13, .13, .14], .5)
    for name, o in OBJECTS.items():
        root = '/World/Objects/'+name.title().replace('_', '')
        UsdGeom.Xform.Define(s, root)
        p, r = object_pose(name, deg)
        set_pose(s.GetPrimAtPath(root), p, r)
        visuals(s, root, name, looks)
        authored(root.rsplit('/', 1)[1], s.GetPrimAtPath(root), f'{name}: {o["d"]*1000:.0f} mm across, {o["m"]*1000:.0f} g')
        pegroot = root+'Peg'
        UsdGeom.Xform.Define(s, pegroot)
        pp, pr = peg_pose(name, deg)
        set_pose(s.GetPrimAtPath(pegroot), pp, pr)
        L.shape(s, 'Block', [0, 0, 0], [.012, .030, .012], peg, parent=pegroot)
        point(name+'_lane', [*(TOE+U*o['s']), H], 'Left', ['down'], f'{name}: thin end of its lane')
    state = dict(board=f'wedge sloping {deg:.0f} deg toward the front edge', objects='cup, egg, pill bottle and apple held by pegs',
                 event='one peg retracts at a hidden time; that object rolls toward the edge')
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
