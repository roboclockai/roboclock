"""Shared static-scene helpers for the household handoff package.

All asset paths written into USD layers are relative to the layer that uses
them, so the package directory can be moved or archived as a unit.
Static scenes deliberately deactivate source joints / rigid bodies; the
handoff records which native joints exist so the simulation stage can
re-enable them.
"""
import hashlib
import json
import math
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

HOUSEHOLD = Path(__file__).resolve().parents[1]
REPO = HOUSEHOLD.parents[3]
ASSETS = HOUSEHOLD/'assets/x5'
KITCHEN = ASSETS/'lightwheel_kitchen/Collected_KitchenRoom'  # not in git: scripts/household/fetch_lightwheel_kitchen.py
OBJECTS = ASSETS/'objects'
TEXTURES = ASSETS/'textures'
X5 = ASSETS/'robot'
REUSED = ASSETS/'reused'
OUTPUT = Path(os.environ.get('ROBOCLOCK_X5_INITIAL', REPO/'data/household/x5_initial'))
H = .8584  # island counter top (m)
from roboclock.envs.household.x5.rig import BASES  # noqa: E402  active base layout (rig.use(task) before building)
READY_Q = [0., .94, 1.34, -1.40, 0., 0.]  # tucked ready pose, gripper ~0.27 m above counter
SCENE_DIR = None  # set per scene so relative paths resolve from that layer


def rel(path):
    path = Path(path)
    if SCENE_DIR is None:
        return str(path)
    r = os.path.relpath(path, SCENE_DIR)
    return r if r.startswith('.') else './'+r


def repo_rel(path):
    """Repository-relative path string for metadata (falls back to the absolute path)."""
    try:  # do not follow the kitchen symlink: keep the in-repository path
        return str(Path(os.path.abspath(path)).relative_to(os.path.abspath(REPO)))
    except ValueError:
        return str(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material(stage, name, color, rough=.45, metal=0., texture=None, opacity=None, emissive=None, ior=None):
    mat = UsdShade.Material.Define(stage, '/World/Looks/'+name)
    sh = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
    if opacity is not None:
        sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).Set(opacity)
    if ior is not None:
        sh.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(ior)
    if emissive is not None:
        sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    if texture:
        uv = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/UV')
        uv.CreateIdAttr('UsdPrimvarReader_float2')
        uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Texture')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel(texture)))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
        if emissive is not None:
            sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def alpha_material(stage, name, texture, emissive=True, tint=(1, 1, 1), opacity_scale=1.):
    """Textured card with alpha cut-out (flame, smoke, display labels)."""
    mat = UsdShade.Material.Define(stage, '/World/Looks/'+name)
    sh = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    uv = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    tex = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Texture')
    tex.CreateIdAttr('UsdUVTexture')
    tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(rel(texture)))
    tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
    tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(tint[0], tint[1], tint[2], opacity_scale))
    tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
    if emissive:
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0))
        sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    else:
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    sh.CreateInput('opacity', Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(), 'a')
    sh.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(.02)
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.9)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def bound(prim):
    b = UsdGeom.BBoxCache(0, ['default', 'render']).ComputeWorldBound(prim).ComputeAlignedRange()
    return np.array(b.GetMin()), np.array(b.GetMax())


def freeze(prim):
    """Static appearance: deactivate joints, strip rigid/collision APIs in this scene's overrides."""
    for p in Usd.PrimRange(prim):
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
        for api in [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI]:
            if p.HasAPI(api):
                p.RemoveAPI(api)


def native_joints(source):
    s = Usd.Stage.Open(str(source))
    out = []
    for p in s.Traverse():
        if p.IsA(UsdPhysics.Joint):
            row = dict(joint=str(p.GetPath()), type=p.GetTypeName(),
                       body1=[str(x) for x in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()])
            for kind in (UsdPhysics.RevoluteJoint, UsdPhysics.PrismaticJoint):
                if p.IsA(kind):
                    j = kind(p)
                    row.update(axis=j.GetAxisAttr().Get(), lower=j.GetLowerLimitAttr().Get(), upper=j.GetUpperLimitAttr().Get())
            out.append(row)
    return out


def shape(stage, name, pos, size, mat, cylinder=False, rot=None, parent='/World/Objects'):
    if cylinder:
        obj = UsdGeom.Cylinder.Define(stage, f'{parent}/{name}')
        obj.CreateRadiusAttr(size[0])
        obj.CreateHeightAttr(size[1])
        obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if rot is not None:
            obj.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    else:
        obj = UsdGeom.Cube.Define(stage, f'{parent}/{name}')
        obj.CreateSizeAttr(1.)
        obj.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if rot is not None:
            obj.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
        obj.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdShade.MaterialBindingAPI.Apply(obj.GetPrim()).Bind(mat)
    return obj.GetPrim()


def disc(stage, name, center, radii, mat, z_wobble=0., segments=72, seed=0, lobes=(.08, .05)):
    """Irregular flat disc (stain, food surface, pancake top)."""
    rng = np.random.default_rng(seed)
    a = np.linspace(0, 2*np.pi, segments+1)[:-1]
    ph = rng.uniform(0, 6.28, 2)
    r = 1+lobes[0]*np.sin(a*5+ph[0])+lobes[1]*np.cos(a*9+ph[1])
    pts = np.c_[center[0]+radii[0]*r*np.cos(a), center[1]+radii[1]*r*np.sin(a), np.full(len(a), center[2])]
    pts = np.vstack([[center[0], center[1], center[2]+z_wobble], pts])
    faces = []
    for i in range(segments):
        faces += [0, 1+i, 1+(i+1) % segments]
    m = UsdGeom.Mesh.Define(stage, '/World/Objects/'+name)
    m.CreatePointsAttr(pts.tolist())
    m.CreateFaceVertexCountsAttr([3]*segments)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr('none')
    st = np.c_[(pts[:, 0]-center[0])/(2.2*radii[0])+.5, (pts[:, 1]-center[1])/(2.2*radii[1])+.5]
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(st.tolist())
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def card(stage, name, center, width, height, mat, normal='up', yaw=0.):
    """Textured rectangle; normal 'up' lies flat, 'front' faces -y (toward the robot/camera)."""
    w, h = width/2, height/2
    if normal == 'up':
        pts = [(-w, -h, 0), (w, -h, 0), (w, h, 0), (-w, h, 0)]
    else:
        pts = [(-w, 0, -h), (w, 0, -h), (w, 0, h), (-w, 0, h)]
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    pts = [(center[0]+c*x-s*y, center[1]+s*x+c*y, center[2]+z) for x, y, z in pts]
    m = UsdGeom.Mesh.Define(stage, '/World/Objects/'+name)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    m.CreateDoubleSidedAttr(True)
    m.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def put(stage, name, source, xy, z, scale=1., rot=(0, 0, 0), prim=None, hide=(), override=None):
    """Reference an asset; place its bbox center at xy and its bottom at z."""
    root = UsdGeom.Xform.Define(stage, '/World/Objects/'+name)
    child = stage.DefinePrim(str(root.GetPath())+'/Source')
    if prim:
        child.GetReferences().AddReference(rel(source), Sdf.Path(prim))
    else:
        child.GetReferences().AddReference(rel(source))
    freeze(child)
    for p in Usd.PrimRange(child):
        if p.GetName() in hide:
            p.SetActive(False)
        elif override is not None and p.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(override)
    xf = UsdGeom.Xformable(root)
    t = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    xf.AddScaleOp().Set(Gf.Vec3f(float(scale)))
    lo, hi = bound(root.GetPrim())
    t.Set(Gf.Vec3d(float(xy[0]-(lo[0]+hi[0])/2), float(xy[1]-(lo[1]+hi[1])/2), float(z-lo[2])))
    return root.GetPrim()


def offset_part(stage, path, delta):
    """Static local override of a native moving part (e.g. a toaster lever held down)."""
    p = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(p)
    ops = xf.GetOrderedXformOps()
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            op.Set(type(v)(v[0]+delta[0], v[1]+delta[1], v[2]+delta[2]))
            return
    xf.AddTranslateOp(opSuffix='handoff').Set(Gf.Vec3d(*delta))


def rotate_part(stage, path, axis, degrees, pivot_local):
    """Rotate a native part about a local pivot (static override, e.g. an oven door)."""
    p = stage.GetPrimAtPath(path)
    xf = UsdGeom.Xformable(p)
    xf.AddTranslateOp(opSuffix='pivotIn').Set(Gf.Vec3d(*pivot_local))
    rot = {'X': xf.AddRotateXOp, 'Y': xf.AddRotateYOp, 'Z': xf.AddRotateZOp}[axis](opSuffix='handoff')
    rot.Set(float(degrees))
    xf.AddTranslateOp(opSuffix='pivotOut').Set(Gf.Vec3d(*[-v for v in pivot_local]))


def surface_z(root, xy, fallback):
    """Highest mesh surface under a vertical ray at xy (world)."""
    hits = []
    cache = UsdGeom.XformCache()
    for p in Usd.PrimRange(root):
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p)
        v = np.asarray(m.GetPointsAttr().Get())
        if v is None or not len(v):
            continue
        M = np.array(cache.GetLocalToWorldTransform(p))
        w = np.c_[v, np.ones(len(v))]@M
        ind = np.array(m.GetFaceVertexIndicesAttr().Get())
        cnt = np.array(m.GetFaceVertexCountsAttr().Get())
        tris = []
        off = 0
        for n in cnt:
            for j in range(1, n-1):
                tris.append([ind[off], ind[off+j], ind[off+j+1]])
            off += n
        if not tris:
            continue
        tri = w[np.array(tris), :3]
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        v0 = b[:, :2]-a[:, :2]
        v1 = c[:, :2]-a[:, :2]
        v2 = np.array(xy)-a[:, :2]
        den = v0[:, 0]*v1[:, 1]-v1[:, 0]*v0[:, 1]
        ok = np.abs(den) > 1e-12
        den = np.where(ok, den, 1)
        u = (v2[:, 0]*v1[:, 1]-v1[:, 0]*v2[:, 1])/den
        vv = (v0[:, 0]*v2[:, 1]-v2[:, 0]*v0[:, 1])/den
        mask = ok & (u >= -1e-5) & (vv >= -1e-5) & (u+vv <= 1.00001)
        hits.extend((a[:, 2]+u*(b[:, 2]-a[:, 2])+vv*(c[:, 2]-a[:, 2]))[mask].tolist())
    return max(hits) if hits else fallback


def convert_robocasa(raw_folder, name):
    """MJCF/OBJ RoboCasa object -> visual USDC with textures copied beside it (relative)."""
    raw_folder = Path(raw_folder)
    out_dir = OBJECTS/name
    dest = out_dir/(name+'.usdc')
    if dest.exists():
        return dest
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(raw_folder/'model.xml')
    specs = {x.get('name'): x for x in tree.findall('./asset/mesh')}
    mats = {x.get('name'): x for x in tree.findall('./asset/material')}
    textures = {x.get('name'): x.get('file') for x in tree.findall('./asset/texture')}
    stage = Usd.Stage.CreateNew(str(dest))
    root = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    for i, g in enumerate(tree.findall('.//worldbody//geom')):
        if not g.get('mesh', '').endswith('_vis'):
            continue
        spec = specs[g.get('mesh')]
        vertices, uv, indices, uvi, counts = [], [], [], [], []
        for line in (raw_folder/spec.get('file')).read_text().splitlines():
            q = line.split()
            if not q:
                continue
            if q[0] == 'v':
                vertices.append([float(x) for x in q[1:4]])
            elif q[0] == 'vt':
                uv.append([float(x) for x in q[1:3]])
            elif q[0] == 'f':
                counts.append(len(q)-1)
                for item in q[1:]:
                    ids = item.split('/')
                    indices.append(int(ids[0])-1)
                    uvi.append(int(ids[1])-1 if len(ids) > 1 and ids[1] else 0)
        pts = np.array(vertices)*np.fromstring(spec.get('scale', '1 1 1'), sep=' ')
        w, x, y, z = np.fromstring(spec.get('refquat', '1 0 0 0'), sep=' ')
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        pts = (pts-np.fromstring(spec.get('refpos', '0 0 0'), sep=' '))@R
        mesh = UsdGeom.Mesh.Define(stage, f'/World/Mesh{i}')
        mesh.CreatePointsAttr(pts.tolist())
        mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateDoubleSidedAttr(True)
        if uv:
            UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying').Set(np.array(uv)[uvi].tolist())
        mat = mats.get(g.get('material'))
        tex = textures.get(mat.get('texture')) if mat is not None else None
        local_tex = None
        if tex and (raw_folder/tex).exists() and uv:
            local_tex = out_dir/Path(tex).name
            if not local_tex.exists():
                shutil.copy2(raw_folder/tex, local_tex)
        rgba = np.fromstring(mat.get('rgba', '.65 .65 .65 1'), sep=' ') if mat is not None else np.array([.65, .65, .65, 1])
        sh_mat = UsdShade.Material.Define(stage, f'/World/Looks/Surface{i}')
        sh = UsdShade.Shader.Define(stage, str(sh_mat.GetPath())+'/Surface')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in rgba[:3]]))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.45)
        if local_tex is not None:
            uvr = UsdShade.Shader.Define(stage, str(sh_mat.GetPath())+'/UV')
            uvr.CreateIdAttr('UsdPrimvarReader_float2')
            uvr.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
            t = UsdShade.Shader.Define(stage, str(sh_mat.GetPath())+'/Texture')
            t.CreateIdAttr('UsdUVTexture')
            t.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./'+local_tex.name))
            t.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
            t.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uvr.ConnectableAPI(), 'result')
            sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), 'rgb')
        sh_mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(sh_mat)
    stage.GetRootLayer().Save()
    (out_dir/'SOURCE.json').write_text(json.dumps(dict(
        source='robocasa/robocasa-assets aigen_objs.zip @1b92c3d02ca4354984fec961357db0bff7b32166',
        raw_member=str(raw_folder).split('assets/robocasa/')[-1], license='CC BY 4.0 (RoboCasa dataset card)',
        conversion='visual meshes only (model.xml *_vis geoms); collision meshes retained in raw download',
        raw_model_xml_sha256=sha(raw_folder/'model.xml')), indent=2)+'\n')
    return dest


def native_size(source):
    s = Usd.Stage.Open(str(source))
    lo, hi = bound(s.GetDefaultPrim() or s.GetPseudoRoot())
    return hi-lo


def scale_for(source, length, axis=None):
    size = native_size(source)
    return float(length/(size[axis] if axis is not None else max(size)))


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])@np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            @np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))


def _origin(e):
    m = np.eye(4)
    if e is not None:
        m[:3, :3] = _rpy(*np.fromstring(e.get('rpy', '0 0 0'), sep=' '))
        m[:3, 3] = np.fromstring(e.get('xyz', '0 0 0'), sep=' ')
    return m


def robot(stage, q=None):
    """Dual X5 visual pose from URDF FK. Visual only; articulation is rebuilt by the simulation stage."""
    q = list(READY_Q if q is None else q)
    urdf = ET.parse(X5/'arx5-sdk/models/X5.urdf')
    poses = {}
    for side, (x, y) in BASES.items():
        base = np.eye(4)
        base[:3, :3] = _rpy(0, 0, math.pi/2)
        base[:3, 3] = [x, y, H+.012]
        frames = {'base_link': base}
        for j in urdf.findall('joint'):
            T = _origin(j.find('origin'))
            if j.get('type') == 'revolute':
                k = int(j.get('name').replace('joint', ''))-1
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                v = axis*q[k]
                R = np.eye(4)
                th = np.linalg.norm(v)
                if th > 0:
                    a = v/th
                    sk = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                    R[:3, :3] = np.eye(3)+math.sin(th)*sk+(1-math.cos(th))*sk@sk
                T = T@R
            frames[j.find('child').get('link')] = frames[j.find('parent').get('link')]@T
        for link in urdf.findall('link'):
            n = link.get('name')
            vis = X5/'visual'/(n+'.usdc')
            if not vis.exists():
                continue
            p = UsdGeom.Xform.Define(stage, f'/World/Robots/{side}/{n}')
            UsdGeom.Xformable(p).AddTransformOp().Set(Gf.Matrix4d((frames[n]@_origin(link.find('visual/origin'))).T.tolist()))
            stage.DefinePrim(str(p.GetPath())+'/Visual').GetReferences().AddReference(rel(vis))
        poses[side] = dict(base_position_m=base[:3, 3].tolist(), base_yaw_deg=90., joint_positions_rad=q,
                           eef_position_m=frames['eef_link'][:3, 3].round(4).tolist(),
                           eef_approach_axis_world=frames['eef_link'][:3, 0].round(3).tolist(),
                           link6_pose_world=frames['link6'].round(5).tolist())
    xs = sorted(round(b[0], 4) for b in BASES.values())
    ys = sorted({round(b[1], 4) for b in BASES.values()})
    sep = '' if xs == [-.06, .66] else f' ({xs[1]-xs[0]:.2f} m apart)'  # legacy text unchanged
    return dict(source='real-stanford/arx5-sdk models/X5.urdf @c160db7834530a3d80fec524547c74e1d4fa8035 (MIT)',
                urdf=rel(X5/'arx5-sdk/models/X5.urdf'), mount='stationary tabletop dual mount on island counter; '
                f'bases at x={xs[0]:g} / {xs[1]:g} m{sep}, y={ys[0]:g} m, '
                'yaw +90 deg (facing +y, the back wall)',
                ready_pose_rad=q, visual_pose_only=True, poses=poses,
                tool_frame='eef_link = link6 + 0.145 m along link6 x; approach axis = eef x-axis')


def kitchen(stage, hide_island=True):
    """Lightwheel KitchenRoom_A1 backdrop (static appearance), native lights off."""
    src = KITCHEN/'KitchenRoom_A1.usdc'
    room = UsdGeom.Xform.Define(stage, '/World/Kitchen')
    room.GetPrim().GetReferences().AddReference(rel(src))
    freeze(room.GetPrim())
    hidden = []
    for path in [p.GetPath() for p in Usd.PrimRange(room.GetPrim())]:
        prim = stage.GetPrimAtPath(path)
        if not prim:
            continue
        nm = prim.GetName()
        if prim.IsA(UsdLux.BoundableLightBase) or prim.IsA(UsdLux.NonboundableLightBase):
            prim.SetActive(False)
        if prim.GetParent() == room.GetPrim() and ((hide_island and 'InsularShelf' in nm) or any(x in nm for x in ['Orange', 'Flowers', 'Kitchen_Disk002']) or nm == 'SM_P_Choppingboard_01'):
            prim.SetActive(False)
            hidden.append(nm)
    UsdGeom.Xformable(room).AddTranslateOp(opSuffix='placement').Set(Gf.Vec3d(-.20, -1.10, 0.))
    return dict(source=rel(src), translation_m=[-.20, -1.10, 0.], hidden_children=hidden,
                note='native lights disabled; joints/physics APIs removed by static overrides')


def lights(stage, interior=()):
    dome = UsdLux.DomeLight.Define(stage, '/World/Lights/Dome')
    dome.CreateIntensityAttr(850.)
    dome.CreateTextureFileAttr(Sdf.AssetPath(rel(TEXTURES/'studio_small_09_1k.hdr')))
    dome.CreateTextureFormatAttr('latlong')
    for name, pos, inten, size in [('Key', [.2, -.8, 2.3], 2000., 2.), ('Fill', [1.3, .7, 2.25], 1000., 1.4)]:
        light = UsdLux.RectLight.Define(stage, '/World/Lights/'+name)
        light.CreateIntensityAttr(inten)
        light.CreateWidthAttr(size)
        light.CreateHeightAttr(size)
        UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(*pos))
    f = UsdLux.RectLight.Define(stage, '/World/Lights/FrontBounce')
    f.CreateIntensityAttr(750.)
    f.CreateWidthAttr(2.6)
    f.CreateHeightAttr(1.2)
    xf = UsdGeom.Xformable(f)
    xf.AddTranslateOp().Set(Gf.Vec3d(.3, -1.4, 2.15))
    xf.AddRotateXOp().Set(65.)
    for i, (pos, inten, radius, color) in enumerate(interior):
        s = UsdLux.SphereLight.Define(stage, f'/World/Lights/Interior{i}')
        s.CreateIntensityAttr(inten)
        s.CreateRadiusAttr(radius)
        s.CreateColorAttr(Gf.Vec3f(*color))
        UsdGeom.Xformable(s).AddTranslateOp().Set(Gf.Vec3d(*pos))


def camera_at(stage, path, matrix, focal, h_ap, v_ap, clip):
    """Camera prim with an explicit 4x4 world pose (USD camera axes, row-major ``matrix``)."""
    cam = UsdGeom.Camera.Define(stage, path)
    cam.AddTransformOp().Set(Gf.Matrix4d(np.asarray(matrix, dtype=float).T.tolist()))
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(*clip))
    return dict(prim=path, pose_world=np.round(np.asarray(matrix, dtype=float), 6).tolist(), focal_length_mm=focal,
                horizontal_aperture_mm=h_ap, vertical_aperture_mm=v_ap, clipping_range_m=list(clip))


def camera(stage, path, eye, look, focal=34., h_ap=30., v_ap=22.5):
    cam = UsdGeom.Camera.Define(stage, path)
    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*look), Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    cam.CreateFocalLengthAttr(focal)
    cam.CreateHorizontalApertureAttr(h_ap)
    cam.CreateVerticalApertureAttr(v_ap)
    cam.CreateClippingRangeAttr(Gf.Vec2f(.02, 30.))
    return dict(prim=path, eye_m=list(eye), look_at_m=list(look), focal_length_mm=focal,
                horizontal_aperture_mm=h_ap, vertical_aperture_mm=v_ap)
