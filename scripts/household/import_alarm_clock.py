"""Import the alarm clock of the household promo scene (alarm_x5): "Alarm Clock 01" (Poly Haven, CC0), a round twin-bell wind-up clock.

    python scripts/household/import_alarm_clock.py

Writes src/roboclock/envs/household/assets/x5/objects/alarm_clock_0/: the raw glTF (1k textures), alarm_clock.usda
and SOURCE.json. The model is kept at its native size (13.2 cm wide, 6.7 cm deep, 17.3 cm tall), turned z up with
the dial facing -y and the origin on the table under the centre of the case. Its single body mesh is split by
connected component, so the parts that move while the alarm rings are prims of their own:

    Body      case, bezel, dial, back, feet, carrying handle
    BellL/R   each bell with its post and nut
    Hammer    the striker between the bells
    Glass     dial glass (its own material)
    HourHand, MinuteHand, SecondHand

The textures (diffuse, normal, AO/roughness/metal) are bound as one UsdPreviewSurface; the glass has none (the
scene binds it).
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from scipy.sparse import csgraph

BASE = 'https://dl.polyhaven.org/file/ph-assets/Models'
FILES = {
    'alarm_clock_01_1k.gltf': f'{BASE}/gltf/1k/alarm_clock_01/alarm_clock_01_1k.gltf',
    'alarm_clock_01.bin': f'{BASE}/gltf/8k/alarm_clock_01/alarm_clock_01.bin',
    'textures/alarm_clock_01_diff_1k.jpg': f'{BASE}/jpg/1k/alarm_clock_01/alarm_clock_01_diff_1k.jpg',
    'textures/alarm_clock_01_nor_gl_1k.jpg': f'{BASE}/jpg/1k/alarm_clock_01/alarm_clock_01_nor_gl_1k.jpg',
    'textures/alarm_clock_01_arm_1k.jpg': f'{BASE}/jpg/1k/alarm_clock_01/alarm_clock_01_arm_1k.jpg',
}
OUT = Path(__file__).resolve().parents[2]/'src/roboclock/envs/household/assets/x5/objects/alarm_clock_0'
Z_UP = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])  # glTF (y up, dial toward +z) -> z up, dial toward -y
ROOT = '/AlarmClock'


def part_of(lo, hi):
    """Name of the part a connected component of the body mesh belongs to, from its bounds (m, z up)."""
    x, z = (lo[0]+hi[0])/2, (lo[2]+hi[2])/2
    if z > .112 and abs(x) > .025 and hi[0]-lo[0] < .06:  # bell shells (z .114-.147), posts and nuts
        return 'BellL' if x < 0 else 'BellR'
    if lo[2] > .128 and abs(x) < .008 and -.008 < lo[1] and hi[1] < 0. and hi[0]-lo[0] < .02:  # head and stem
        return 'Hammer'
    return 'Body'


def write_mesh(stage, name, v, n, uv, faces, material):
    used = np.unique(faces)
    remap = np.full(len(v), -1)
    remap[used] = np.arange(len(used))
    v, n, uv, faces = v[used], n[used], uv[used], remap[faces]
    mesh = UsdGeom.Mesh.Define(stage, f'{ROOT}/{name}')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces.reshape(-1).tolist()))
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr([Gf.Vec3f(*map(float, v.min(0))), Gf.Vec3f(*map(float, v.max(0)))])
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray.FromNumpy(uv.astype(np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return v.min(0), v.max(0)


def pbr(stage):
    mat = UsdShade.Material.Define(stage, f'{ROOT}/Looks/Clock')
    path = str(mat.GetPath())
    sh = UsdShade.Shader.Define(stage, path+'/Surface')
    sh.CreateIdAttr('UsdPreviewSurface')
    uv = UsdShade.Shader.Define(stage, path+'/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')

    def texture(name, file, space):
        tex = UsdShade.Shader.Define(stage, f'{path}/{name}')
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./textures/'+file))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set(space)
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        return tex

    diff = texture('Diffuse', 'alarm_clock_01_diff_1k.jpg', 'sRGB')
    arm = texture('ARM', 'alarm_clock_01_arm_1k.jpg', 'raw')  # r ambient occlusion, g roughness, b metal
    nor = texture('Normal', 'alarm_clock_01_nor_gl_1k.jpg', 'raw')
    nor.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2, 2, 2, 1))
    nor.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1, -1, -1, 0))
    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(diff.ConnectableAPI(), 'rgb')
    sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).ConnectToSource(arm.ConnectableAPI(), 'g')
    sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).ConnectToSource(arm.ConnectableAPI(), 'b')
    sh.CreateInput('normal', Sdf.ValueTypeNames.Normal3f).ConnectToSource(nor.ConnectableAPI(), 'rgb')
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
    return mat


def main():
    for name, url in FILES.items():
        dest = OUT/name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'asset-fetch/1.0'})
            dest.write_bytes(urllib.request.urlopen(request, timeout=120).read())
    scene = trimesh.load(OUT/'alarm_clock_01_1k.gltf', process=False)
    usd = OUT/'alarm_clock.usda'
    if usd.exists():
        usd.unlink()
    stage = Usd.Stage.CreateNew(str(usd))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, ROOT).GetPrim())
    material = pbr(stage)
    names = {'Cylinder.030': None, 'Cylinder.030_1': 'Glass', 'Plane.012': 'HourHand', 'Plane.013': 'MinuteHand',
             'Plane.014': 'SecondHand'}
    parts = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry]
        v = trimesh.transformations.transform_points(mesh.vertices, transform)@Z_UP.T
        n = np.asarray(mesh.vertex_normals)@transform[:3, :3].T@Z_UP.T
        uv = np.asarray(mesh.visual.uv)
        faces = np.asarray(mesh.faces)
        if names[geometry] is not None:
            parts[names[geometry]] = (v, n, uv, faces)
            continue
        # body: weld the UV-seam duplicates by position, then split by connected component
        _, weld = np.unique(np.round(v, 5), axis=0, return_inverse=True)
        f = weld[faces]
        edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
        count = weld.max()+1
        _, label = csgraph.connected_components(sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                                                              shape=(count, count)), directed=False)
        face_label = label[f[:, 0]]
        owner = {}
        for c in np.unique(face_label):
            pts = v[np.unique(faces[face_label == c])]
            owner[c] = part_of(pts.min(0), pts.max(0))
        for name in ('Body', 'BellL', 'BellR', 'Hammer'):
            keep = np.array([owner[c] == name for c in face_label])
            parts[name] = (v, n, uv, faces[keep])
    size = {}
    for name, (v, n, uv, faces) in parts.items():
        uv = np.c_[uv[:, 0], uv[:, 1]]
        lo, hi = write_mesh(stage, name, v, n, uv, faces, None if name == 'Glass' else material)
        size[name] = dict(min_m=lo.round(4).tolist(), max_m=hi.round(4).tolist())
    stage.GetRootLayer().Save()
    lo = np.min([s['min_m'] for s in size.values()], axis=0)
    hi = np.max([s['max_m'] for s in size.values()], axis=0)
    (OUT/'SOURCE.json').write_text(json.dumps(dict(
        source='https://polyhaven.com/a/alarm_clock_01 ("Alarm Clock 01", modelling and texturing James Ray Cock, '
               'rigging Yann Kervran), glTF with 1k textures from dl.polyhaven.org',
        license='CC0 1.0 (Poly Haven asset license)',
        conversion='native size, turned z up with the dial toward -y and the origin on the table under the case; the '
                   'body mesh is split by connected component into Body, BellL, BellR and Hammer; glass and hands as '
                   'in the source; one UsdPreviewSurface (diffuse, normal, roughness and metal from the ARM map); '
                   'scripts/household/import_alarm_clock.py',
        size_m=(hi-lo).round(4).tolist(), bbox_min_m=lo.round(4).tolist(), bbox_max_m=hi.round(4).tolist(), parts=size,
        raw_sha256={name: hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in FILES}), indent=2)+'\n')
    print('wrote', OUT, (hi-lo).round(4))
    for name, s in size.items():
        print(' ', name, s)


if __name__ == '__main__':
    main()
