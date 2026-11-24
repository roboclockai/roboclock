"""Scene construction helpers shared by the Dual-X5 household initial scenes."""
import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

from roboclock.envs.household.x5 import scene_lib as L
from roboclock.envs.household.x5.scene_lib import H

RAW = Path(os.environ.get('ROBOCASA_RAW', '/opt/household/V1.0_X5/assets/robocasa/aigen_objs'))


def rc(cat, idx):
    return L.convert_robocasa(RAW/cat/f'{cat}_{idx}', f'{cat}_{idx}')


def squash(prim, factor, z):
    """Scale an object along z only, keeping its bottom at z (deep plate from a bowl)."""
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    sc = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale][0]
    v = sc.Get()
    sc.Set(Gf.Vec3f(v[0], v[1], v[2]*factor))
    lo, _ = L.bound(prim)
    t = [op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate][0]
    p = t.Get()
    t.Set(Gf.Vec3d(p[0], p[1], p[2]+z-lo[2]))


def glass_mdl(stage, name, color=(.96, .98, 1.), ior=1.5, thin=False):
    mat = UsdShade.Material.Define(stage, '/World/Looks/'+name)
    sh = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
    sh.SetSourceAsset(Sdf.AssetPath('OmniGlass.mdl'), 'mdl')
    sh.SetSourceAssetSubIdentifier('OmniGlass', 'mdl')
    sh.CreateInput('glass_color', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput('glass_ior', Sdf.ValueTypeNames.Float).Set(ior)
    sh.CreateInput('thin_walled', Sdf.ValueTypeNames.Bool).Set(thin)
    sh.CreateInput('frosting_roughness', Sdf.ValueTypeNames.Float).Set(0.)
    mat.CreateSurfaceOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    mat.CreateVolumeOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    mat.CreateDisplacementOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    return mat


def lathe(stage, name, profile, center, mat, segments=64):
    """Surface of revolution from (radius, z) profile, open where the profile is open."""
    prof = np.asarray(profile, float)
    a = np.linspace(0, 2*np.pi, segments, endpoint=False)
    pts = [(center[0]+r*math.cos(t), center[1]+r*math.sin(t), center[2]+z) for r, z in prof for t in a]
    faces, counts = [], []
    for i in range(len(prof)-1):
        for j in range(segments):
            k = (j+1) % segments
            faces += [i*segments+j, i*segments+k, (i+1)*segments+k, (i+1)*segments+j]
            counts.append(4)
    m = UsdGeom.Mesh.Define(stage, '/World/Objects/'+name)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def box_mesh(stage, name, lo, hi, mat, uv_tile=.6):
    """Axis-aligned box mesh with world-planar UVs (for native-material counter slabs)."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    P = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    F = [(4, 5, 6, 7), (0, 3, 2, 1), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    pts, idx, st = [], [], []
    for f in F:
        for v in f:
            p = P[v]
            idx.append(len(pts))
            pts.append(p)
            st.append((p[0]/uv_tile, p[1]/uv_tile) if f == F[0] or f == F[1] else ((p[0]+p[1])/uv_tile, p[2]/uv_tile))
    m = UsdGeom.Mesh.Define(stage, '/World/Objects/'+name)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4]*6)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr('none')
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying').Set(st)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def island_top_material(stage, island):
    """Material bound to the highest visual island mesh (the terrazzo top)."""
    best = None
    cache = UsdGeom.XformCache()
    for p in Usd.PrimRange(island):
        if not p.IsA(UsdGeom.Mesh) or 'Collisions' in str(p.GetPath()):
            continue
        mesh = UsdGeom.Mesh(p)
        pts = np.asarray(mesh.GetPointsAttr().Get())
        M = np.array(cache.GetLocalToWorldTransform(p))
        wp = (np.c_[pts, np.ones(len(pts))]@M)[:, :3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        ind = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.r_[0, np.cumsum(counts)]
        cz = np.array([wp[ind[starts[i]:starts[i+1]]][:, 2].mean() for i in range(len(counts))])
        top = int(np.argmax(cz))
        m = None
        for sub in UsdShade.MaterialBindingAPI(p).GetMaterialBindSubsets():
            if top in list(sub.GetIndicesAttr().Get() or []):
                m, _ = UsdShade.MaterialBindingAPI(sub.GetPrim()).ComputeBoundMaterial()
        if m is None:
            m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
        if m and (best is None or cz[top] > best[0]):
            best = (cz[top], m)
    if best is None:
        raise RuntimeError('No bound island top material found')
    return best[1]


def cut_island_top(island, opening=None):
    """Deactivate island faces above the cabinet so a sink can be inset (B1).

    ``opening`` = (x0, y0, x1, y1, z0) also cuts the cabinet's panels inside the basin, above z0, so the basin is open."""
    for prim in Usd.PrimRange(island):
        if not prim.IsA(UsdGeom.Mesh) or 'Collisions' in str(prim.GetPath()):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get())
        M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        wp = (np.c_[pts, np.ones(len(pts))]@M)[:, :3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        ind = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.r_[0, np.cumsum(counts)]
        faces = [wp[ind[starts[i]:starts[i+1]]] for i in range(len(counts))]
        cz = np.array([f[:, 2].mean() for f in faces])
        keep = cz < H-.04
        if opening is not None:
            x0, y0, x1, y1, z0 = opening
            keep &= [not (f[:, 2].max() > z0 and f[:, 0].max() > x0 and f[:, 0].min() < x1 and f[:, 1].max() > y0 and f[:, 1].min() < y1)
                     for f in faces]
        if keep.all():
            continue
        if not keep.any():
            prim.SetActive(False)
            continue
        kept = np.concatenate([np.arange(starts[i], starts[i+1]) for i in range(len(counts)) if keep[i]])
        mesh.GetFaceVertexCountsAttr().Set(counts[keep].tolist())
        mesh.GetFaceVertexIndicesAttr().Set(ind[kept].tolist())
        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if pv.GetInterpolation() == 'faceVarying':
                if pv.IsIndexed():
                    pv.SetIndices(np.asarray(pv.GetIndices())[kept].tolist())
                else:
                    pv.Set(np.asarray(pv.Get())[kept].tolist())
            elif pv.GetInterpolation() == 'uniform':
                v = pv.Get()
                if v is not None and len(v) == len(counts):
                    pv.Set(np.asarray(v)[keep].tolist())
        if mesh.GetNormalsInterpolation() == 'faceVarying' and mesh.GetNormalsAttr().Get() is not None:
            mesh.GetNormalsAttr().Set(np.asarray(mesh.GetNormalsAttr().Get())[kept].tolist())
        remap = np.full(len(counts), -1)
        remap[keep] = np.arange(keep.sum())
        for sub in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
            old = np.array(sub.GetIndicesAttr().Get(), dtype=int)
            new = remap[old]
            sub.GetIndicesAttr().Set(new[new >= 0].tolist())


def pan_on(s, obj, name, src, center_xy, length, z, role):
    """Place a pan with its handle toward +x and its bowl centre at center_xy."""
    p = obj(name, src, [center_xy[0], center_xy[1]], z, L.scale_for(src, length), rot=(0, 0, 90), role=role)
    lo, hi = L.bound(p)
    d = hi[1]-lo[1]
    bowl_x = lo[0]+d/2
    op = UsdGeom.Xformable(p).GetOrderedXformOps()[0]
    v = op.Get()
    op.Set(Gf.Vec3d(v[0]+center_xy[0]-bowl_x, v[1], v[2]))
    return p, d


def open_window(stage, door_path, clear_path, margin=.012):
    """Remove the door faces covering the native clear pane so the lit interior is visible."""
    clo, chi = L.bound(stage.GetPrimAtPath(clear_path))
    prim = stage.GetPrimAtPath(door_path)
    for m in Usd.PrimRange(prim):
        if not m.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(m)
        pts = np.asarray(mesh.GetPointsAttr().Get())
        M = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(m))
        wp = (np.c_[pts, np.ones(len(pts))]@M)[:, :3]
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
        ind = np.array(mesh.GetFaceVertexIndicesAttr().Get())
        starts = np.r_[0, np.cumsum(counts)]
        c = np.array([wp[ind[starts[i]:starts[i+1]]].mean(0) for i in range(len(counts))])
        inside = (c[:, 0] > clo[0]+margin) & (c[:, 0] < chi[0]-margin) & (c[:, 2] > clo[2]+margin) & (c[:, 2] < chi[2]-margin)
        keep = ~inside
        if keep.all():
            continue
        kept = np.concatenate([np.arange(starts[i], starts[i+1]) for i in range(len(counts)) if keep[i]])
        mesh.GetFaceVertexCountsAttr().Set(counts[keep].tolist())
        mesh.GetFaceVertexIndicesAttr().Set(ind[kept].tolist())
        for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
            if pv.GetInterpolation() == 'faceVarying':
                if pv.IsIndexed():
                    pv.SetIndices(np.asarray(pv.GetIndices())[kept].tolist())
                else:
                    pv.Set(np.asarray(pv.Get())[kept].tolist())
        if mesh.GetNormalsInterpolation() == 'faceVarying' and mesh.GetNormalsAttr().Get() is not None:
            mesh.GetNormalsAttr().Set(np.asarray(mesh.GetNormalsAttr().Get())[kept].tolist())
        return int(inside.sum())
    return 0


def silhouette_card(stage, name, center, direction, height, alpha, mat, cut=.08):
    """Vertical card clipped to an alpha silhouette (no rectangular card edges remain)."""
    rows, cols = alpha.shape
    pts, uv, faces = [], [], []
    c = np.asarray(center, float)
    d = np.asarray(direction, float)
    for iy in range(rows-1):
        for ix in range(cols-1):
            if alpha[iy:iy+2, ix:ix+2].mean() < cut:
                continue
            off = len(pts)
            for xx, yy in [(ix, iy), (ix+1, iy), (ix+1, iy+1), (ix, iy+1)]:
                u, v = xx/(cols-1), yy/(rows-1)
                pts.append((c+(u-.5)*d+[0, 0, v*height]).tolist())
                uv.append((u, v))
            faces += [off, off+1, off+2, off+3]
    m = UsdGeom.Mesh.Define(stage, '/World/Objects/'+name)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr([4]*(len(faces)//4))
    m.CreateFaceVertexIndicesAttr(faces)
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(m).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set(uv)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    return m.GetPrim()


def smoke_mdl(stage, name, texture):
    mat = UsdShade.Material.Define(stage, '/World/Looks/'+name)
    sh = UsdShade.Shader.Define(stage, str(mat.GetPath())+'/Shader')
    sh.SetSourceAsset(Sdf.AssetPath('OmniPBR.mdl'), 'mdl')
    sh.SetSourceAssetSubIdentifier('OmniPBR', 'mdl')
    sh.CreateInput('diffuse_color_constant', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.40, .40, .42))
    sh.CreateInput('reflection_roughness_constant', Sdf.ValueTypeNames.Float).Set(1.)
    sh.CreateInput('enable_opacity', Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput('enable_opacity_texture', Sdf.ValueTypeNames.Bool).Set(True)
    sh.CreateInput('opacity_texture', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(L.rel(texture)))
    sh.CreateInput('opacity_mode', Sdf.ValueTypeNames.Int).Set(0)
    sh.CreateInput('opacity_threshold', Sdf.ValueTypeNames.Float).Set(0.)
    sh.CreateInput('opacity_constant', Sdf.ValueTypeNames.Float).Set(.6)
    mat.CreateSurfaceOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    mat.CreateVolumeOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    mat.CreateDisplacementOutput('mdl').ConnectToSource(sh.ConnectableAPI(), 'out')
    return mat


def world(stage, path):
    return np.array(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)))


def reparent(stage, source, dest, source_prim, world_matrix, parent_translation, looks):
    """Reference one native part under a new unscaled Xform, keeping its world placement.

    `source` is the native asset file. Native bindings target /root/Looks, outside a single-prim
    reference; rebind them to the same materials under the full asset reference (`looks`)."""
    root = UsdGeom.Xform.Define(stage, dest)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*map(float, parent_translation)))
    child = stage.DefinePrim(dest+'/Visual')
    child.GetReferences().AddReference(L.rel(source), source_prim)
    L.freeze(child)
    native = Usd.Stage.Open(str(source))
    for p in Usd.PrimRange(native.GetPrimAtPath(source_prim)):
        targets = UsdShade.MaterialBindingAPI(p).GetDirectBindingRel().GetTargets()
        if targets:
            here = stage.GetPrimAtPath(str(p.GetPath()).replace(source_prim, dest+'/Visual', 1))
            mat = UsdShade.Material(stage.GetPrimAtPath(f'{looks}/{targets[0].name}'))
            UsdShade.MaterialBindingAPI.Apply(here).Bind(mat)
    local = world_matrix@np.linalg.inv(_translation(parent_translation))
    xf = UsdGeom.Xformable(child)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(Gf.Matrix4d(local.tolist()))
    return root.GetPrim()


def _translation(t):
    m = np.eye(4)
    m[3, :3] = t  # USD row-vector convention
    return m


def body_visual(stage, dest, source, origin, scale, rot=(0, 0, 0), zscale=1.):
    """Xform at `origin` (bottom centre) with a referenced visual scaled/rotated beneath it."""
    root = UsdGeom.Xform.Define(stage, dest)
    UsdGeom.Xformable(root).AddTranslateOp().Set(Gf.Vec3d(*map(float, origin)))
    child = stage.DefinePrim(dest+'/Visual')
    child.GetReferences().AddReference(L.rel(source))
    L.freeze(child)
    xf = UsdGeom.Xformable(child)
    t = xf.AddTranslateOp()
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot))
    xf.AddScaleOp().Set(Gf.Vec3f(float(scale), float(scale), float(scale*zscale)))
    lo, hi = L.bound(child)
    t.Set(Gf.Vec3d(float(origin[0]-(lo[0]+hi[0])/2), float(origin[1]-(lo[1]+hi[1])/2), float(origin[2]-lo[2])))
    return root.GetPrim()
