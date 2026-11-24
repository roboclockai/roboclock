"""C3: cutlery organization (static initial scene).

An open cutlery drawer (drawer box with a three-compartment insert) stands on the counter between the
two arms. Each compartment already holds one utensil of its kind (spoon, fork or knife), which shows
what goes where; the order of the compartments changes per episode. Six more utensils (two of each)
lie mixed on the table, three to the left and three to the right of the drawer (nine, three of each,
until 2026-11-09: the C-class rule makes every loose utensil a job, Q = 6).

All utensils share the same dark handle; only the head tells them apart (spoon bowl, fork tines, knife
blade). Local frame of a utensil: x along it toward the head, origin on the handle centreline at the
grasp point, z = 0 the plane it rests on.

The poses written here are the seed-0 layout; the episode moves the utensils to the seed's layout
before it starts (``layout()``).
"""
import math

import numpy as np
from pxr import Gf, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation as R

# The utensil names live in task.py, which loads without pxr.
from roboclock.envs.household.c3_x5.task import KINDS, LOOSE, LOOSE_PER_KIND, REFERENCES, name
from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H

# drawer: compartments run along y, side by side in x
DRAWER_C = np.array([.30, -.10])          # centre of the drawer box (x, y)
COMP_W, COMP_L = .080, .250               # inside of one compartment (x, y)
WALL, FLOOR, WALL_H = .008, .008, .045    # wall thickness, floor thickness, wall height above the counter
HANDLE = dict(x0=-.075, x1=.035, w=.016, t=.008)  # shared handle box (grasp point at x = 0)
PIECES = {  # per kind: mass (kg), centre of mass along x
    'spoon': dict(m=.040, com=.012),
    'fork': dict(m=.040, com=.012),
    'knife': dict(m=.045, com=.016),
}


def comp_x(i):
    """Centre x of compartment i (0 = left)."""
    return float(DRAWER_C[0]+(i-1)*(COMP_W+WALL))


def drawer_outer():
    w = 3*COMP_W+4*WALL
    return w, COMP_L+2*WALL


def all_names():
    """Loose utensils first (Spoon1-2, Fork1-2, Knife1-2), then the three references (Spoon3, Fork3, Knife3)."""
    return list(LOOSE)+list(REFERENCES)


def kind_of(n):
    return n.rstrip('0123456789').lower()


# ------------------------------------------------------------------ layouts
# Until 2026-11-09 nine slots: these three plus (-.040, -.005, 95.) and the far (.040, .205, 4.) on the left, and the
# mirror of (-.040, -.005, 95.) on the right. The six kept are the three nearest the drawer on each side, 5.5 cm apart.
_LEFT = [(.015, .005, 268.), (.070, -.010, 84.), (.125, .010, 274.)]
SLOTS = {  # table slots (x, y, yaw deg) of the six loose utensils: three left of the drawer, three right (mirrored)
    'Left': _LEFT,
    'Right': [(DRAWER_C[0]*2-x, y, (180.-yaw) % 360.) for x, y, yaw in _LEFT],
}


def slots():
    return SLOTS['Left']+SLOTS['Right']


def layout(seed):
    """Seeded layout: compartment order (kind per compartment, left to right) and a pose per loose utensil.

    Seed 0 is the nominal layout. Other seeds shuffle which utensil lies in which table slot (with
    +-0.6 cm across, +-1 cm along, +-8 deg jitter) and the compartment order."""
    order = list(KINDS)
    poses = {}
    loose = list(LOOSE)
    table = slots()
    if seed:
        rng = np.random.default_rng([int(seed), 3])
        order = [KINDS[i] for i in rng.permutation(3)]
        perm = rng.permutation(len(table))
        jit = rng.uniform(-1, 1, (len(table), 3))*[.006, .010, 8.]
        table = [tuple(np.add(table[p], jit[i])) for i, p in enumerate(perm)]
    else:
        # Spoon1, Spoon2, Fork1, Fork2, Knife1, Knife2: left (outer to inner) spoon, knife, fork; right (inner to
        # outer) spoon, fork, knife. Each arm has one of each kind.
        table = [table[i] for i in (0, 5, 4, 2, 3, 1)]
    for n, (x, y, yaw) in zip(loose, table):
        poses[n] = (np.array([x, y, H]), float(yaw))
    for i, kind in enumerate(order):
        poses[name(kind, LOOSE_PER_KIND)] = (np.array([comp_x(i), DRAWER_C[1], H+FLOOR]), 90.)
    return order, poses


def set_pose(prim, xyz, yaw):
    """Translate + orient ops (this module also loads outside Kit, without the PhysX schemas)."""
    q = R.from_euler('z', yaw, degrees=True).as_quat()
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, xyz)))
    xf.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))


# ------------------------------------------------------------------ meshes
def mesh(s, path, pts, faces, mat):
    m = UsdGeom.Mesh.Define(s, path)
    pts = np.asarray(pts, float)
    m.CreatePointsAttr(pts.tolist())
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    m.CreateExtentAttr([pts.min(0).tolist(), pts.max(0).tolist()])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m


def slab(s, path, outline, z0, z1, mat):
    """Prism over a convex-or-simple 2-D outline (counter-clockwise), from z0 to z1."""
    o = np.asarray(outline, float)
    n = len(o)
    pts = [[x, y, z0] for x, y in o]+[[x, y, z1] for x, y in o]
    faces = [list(range(n))[::-1], list(range(n, 2*n))]
    faces += [[i, (i+1) % n, n+(i+1) % n, n+i] for i in range(n)]
    return mesh(s, path, pts, faces, mat)


def rounded_rect(x0, x1, w, r, seg=4):
    """Counter-clockwise outline of an x-aligned rectangle (width w) with corners rounded by r."""
    pts = []
    for cx, cy, a0 in ((x1-r, -w/2+r, -90.), (x1-r, w/2-r, 0.), (x0+r, w/2-r, 90.), (x0+r, -w/2+r, 180.)):
        for t in np.radians(np.linspace(a0, a0+90., seg+1)):
            pts.append([cx+r*math.cos(t), cy+r*math.sin(t)])
    return pts


def bowl(s, path, cx, a, b, depth, mat, nu=24, nv=7):
    """Spoon bowl: half ellipsoid shell, rim at z = depth, bottom at z = 0, open on top."""
    pts, faces = [], []
    for i in range(nv+1):
        v = (math.pi/2)*i/nv  # 0 at the rim, pi/2 at the bottom
        for j in range(nu):
            u = 2*math.pi*j/nu
            pts.append([cx+a*math.cos(v)*math.cos(u), b*math.cos(v)*math.sin(u), depth*(1-math.sin(v))])
    for i in range(nv):
        for j in range(nu):
            k = (j+1) % nu
            faces.append([i*nu+j, i*nu+k, (i+1)*nu+k, (i+1)*nu+j])
    return mesh(s, path, pts, faces, mat)


def utensil(s, root, kind, looks):
    """Visual meshes of one utensil under root (local frame, see module doc)."""
    h = HANDLE
    slab(s, root+'/Handle', rounded_rect(h['x0'], h['x1'], h['w'], .006), 0., h['t'], looks['handle'])
    for k, x in enumerate((h['x0']+.012, h['x1']-.010)):  # two rivets
        L.shape(s, f'Rivet{k}', [x, 0, h['t']+.0002], [.0022, .0004], looks['steel'], cylinder=True, parent=root)
    if kind == 'spoon':
        slab(s, root+'/Neck', [[h['x1']-.004, -.0035], [.062, -.003], [.062, .003], [h['x1']-.004, .0035]], 0., .003, looks['steel'])
        bowl(s, root+'/Bowl', .090, .030, .020, .010, looks['steel'])
    elif kind == 'fork':
        slab(s, root+'/Neck', [[h['x1']-.004, -.0035], [.060, -.004], [.060, .004], [h['x1']-.004, .0035]], 0., .003, looks['steel'])
        slab(s, root+'/Base', [[.058, -.005], [.074, -.0125], [.080, -.0125], [.080, .0125], [.074, .0125], [.058, .005]], 0., .003, looks['steel'])
        for k in range(4):
            y = -.0105+k*.007
            slab(s, f'{root}/Tine{k}', [[.079, y-.0016], [.120, y-.0012], [.122, y], [.120, y+.0012], [.079, y+.0016]], 0., .003, looks['steel'])
    else:  # knife: straight back, curved edge to the tip
        ts = np.linspace(0, 1, 10)
        edge = [[h['x1']-.004, -.0045]]+[[.040+t*.110, -.0045-.0035*math.sin(math.pi*t)-.004*t*t] for t in ts]
        back = [[.150, -.0095], [.152, .002], [.146, .0075], [.040, .0085], [h['x1']-.004, .0045]]
        edge, back = edge[:-1], back
        slab(s, root+'/Blade', edge+back, 0., .002, looks['blade'])


def build(ctx):
    """Add the C3 task objects to ctx.s; return the initial process state and agent camera."""
    s, point, authored = ctx.s, ctx.point, ctx.authored
    looks = dict(handle=L.material(s, 'CutleryHandle', [.06, .055, .05], .45),
                 steel=L.material(s, 'CutlerySteel', [.80, .80, .82], .22, .9),
                 blade=L.material(s, 'KnifeBlade', [.84, .85, .87], .16, .9))
    wood = L.material(s, 'DrawerWood', [.70, .53, .34], .6)
    felt = L.material(s, 'DrawerFelt', [.20, .27, .24], .9)
    ow, ol = drawer_outer()
    cx, cy = DRAWER_C
    root = '/World/Objects/Drawer'
    UsdGeom.Xform.Define(s, root)
    L.shape(s, 'Floor', [cx, cy, H+FLOOR/2], [ow, ol, FLOOR], wood, parent=root)
    L.shape(s, 'Felt', [cx, cy, H+FLOOR+.0005], [ow-2*WALL, ol-2*WALL, .001], felt, parent=root)
    for side, y in (('Front', cy-ol/2+WALL/2), ('Back', cy+ol/2-WALL/2)):
        L.shape(s, side, [cx, y, H+WALL_H/2], [ow, WALL, WALL_H], wood, parent=root)
    for i in range(4):
        x = cx-ow/2+WALL/2+i*(COMP_W+WALL)
        L.shape(s, f'Wall{i}', [x, cy, H+WALL_H/2], [WALL, ol-2*WALL, WALL_H], wood, parent=root)
    L.shape(s, 'Pull', [cx, cy-ol/2-.006, H+WALL_H*.6], [.08, .006, .010], looks['steel'], parent=root)
    authored('Drawer', s.GetPrimAtPath(root), f'open cutlery drawer: three compartments {COMP_W*100:.0f} x {COMP_L*100:.0f} cm, '
             f'walls {WALL_H*100:.1f} cm high (static)')
    order, poses = layout(0)
    for n in all_names():
        kind = kind_of(n)
        r = '/World/Objects/'+n
        UsdGeom.Xform.Define(s, r)
        set_pose(s.GetPrimAtPath(r), *poses[n])
        utensil(s, r, kind, looks)
        ref = n in REFERENCES
        authored(n, s.GetPrimAtPath(r), f'{kind}' + (' (already in the drawer: marks its compartment)' if ref else ' on the table')
                 + f', {PIECES[kind]["m"]*1000:.0f} g')
        if not ref:
            p, yaw = poses[n]
            point(n.lower()+'_grasp', p+[0, 0, HANDLE['t']/2], 'Left' if p[0] < cx else 'Right', ['down'],
                  f'pinch the handle of {n} across its width (seed-0 layout)')
    for i, kind in enumerate(order):
        point(f'compartment{i+1}', [comp_x(i), cy, H+WALL_H], 'Left' if i < 1 else 'Right', ['down'],
              f'compartment {i+1} (seed 0: {kind}s)')
    state = dict(drawer='open, three compartments, one reference utensil in each', table='six utensils mixed: two spoons, '
                 'two forks, two knives', compartment_order='per seed (seed 0: spoons, forks, knives left to right)')
    agent = None  # head camera: fixed to the dual-arm rig (x5/rig.py)
    return state, agent
