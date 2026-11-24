"""Render meshes with real edge radii; these meshes carry no physics APIs."""

import numpy as np


def rounded_box(size, radius=0.003, segments=6):
    """Six sampled faces projected onto an inset box plus a rounded edge radius."""
    half = np.asarray(size, dtype=float) / 2
    radius = min(radius, float(np.min(half)) * 0.8)
    core = half - radius
    points, normals, uv, indices = [], [], [], []
    for axis in range(3):
        a, b = (axis + 1) % 3, (axis + 2) % 3

        # Sampling across the bevel band rather than evenly across the large face
        # retains a flat center while resolving the tiny edge radius.
        def coords(extent):
            return np.unique(
                np.r_[
                    np.linspace(-extent, -extent + radius, segments),
                    np.linspace(extent - radius, extent, segments),
                ]
            )

        aa, bb = coords(half[a]), coords(half[b])
        for sign in (-1, 1):
            first = len(points)
            for vb in bb:
                for va in aa:
                    p = np.zeros(3)
                    p[axis], p[a], p[b] = sign * half[axis], va, vb
                    inside = np.clip(p, -core, core)
                    direction = p - inside
                    normal = direction / np.linalg.norm(direction)
                    points.append(inside + radius * normal)
                    normals.append(normal)
                    uv.append([va / size[a] + 0.5, vb / size[b] + 0.5])
            for j in range(len(bb) - 1):
                for i in range(len(aa) - 1):
                    k = first + j * len(aa) + i
                    face = [k, k + 1, k + 1 + len(aa), k + len(aa)]
                    indices.extend(face if sign > 0 else face[::-1])
    return np.asarray(points), np.asarray(normals), np.asarray(uv), np.asarray(indices)


def revolved(profile, segments=96):
    """Closed radial cross-section (radius, z), including inner glass walls."""
    points, uv, indices = [], [], []
    low, high = min(p[1] for p in profile), max(p[1] for p in profile)
    for radius, z in profile:
        for i in range(segments + 1):
            theta = 2 * np.pi * i / segments
            points.append([radius * np.cos(theta), radius * np.sin(theta), z])
            uv.append([i / segments, (z - low) / max(high - low, 1e-9)])
    for row in range(len(profile)):
        next_row = (row + 1) % len(profile)
        for i in range(segments):
            a, b = row * (segments + 1) + i, next_row * (segments + 1) + i
            indices.extend([a, a + 1, b + 1, b])
    return np.asarray(points), np.asarray(uv), np.asarray(indices)


def author_mesh(stage, path, arrays, position=(0, 0, 0), normals=None):
    from pxr import Gf, Sdf, UsdGeom

    points, uv, indices = arrays
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([4] * (len(indices) // 4))
    mesh.CreateFaceVertexIndicesAttr(indices.tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.AddTranslateOp().Set(Gf.Vec3d(*[float(x) for x in position]))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
    ).Set([Gf.Vec2f(*p) for p in uv])
    if normals is not None:
        mesh.CreateNormalsAttr([Gf.Vec3f(*p) for p in normals])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return mesh


def smooth_revolved_normals(mesh, profile, segments=96):
    """Smooth circumferential refraction while retaining cross-section corners."""
    from pxr import UsdGeom, Vt

    profile = np.asarray(profile)
    delta = np.roll(profile, -1, axis=0) - profile
    lengths = np.linalg.norm(delta, axis=1)
    radial = delta[:, 1] / np.maximum(lengths, 1e-12)
    vertical = -delta[:, 0] / np.maximum(lengths, 1e-12)
    theta = np.arange(segments + 1) * (2 * np.pi / segments)
    normals = np.zeros((len(profile), segments, 4, 3), dtype=np.float32)
    for corner, angles in enumerate([theta[:-1], theta[1:], theta[1:], theta[:-1]]):
        normals[:, :, corner, 0] = radial[:, None] * np.cos(angles)
        normals[:, :, corner, 1] = radial[:, None] * np.sin(angles)
        normals[:, :, corner, 2] = vertical[:, None]
    normals[lengths < 1e-12, :, :, 2] = 1
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.reshape(-1, 3)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
