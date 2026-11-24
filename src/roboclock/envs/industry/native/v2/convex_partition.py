"""Plane-split a source convex hull into bounded exact convex subvolumes.

Every generated point is an original hull vertex or a hull-edge intersection.
The pieces fill the original hull; no outward approximation or visual edits.
GPU cooking still requires native support/penetration validation.
"""

import numpy as np
from scipy.spatial import ConvexHull


def partition_convex(vertices, vertex_limit=32, face_limit=60):
    vertices = np.asarray(vertices, dtype=float)
    result = []

    def split(points, depth):
        hull = ConvexHull(points)
        points = points[hull.vertices]
        hull = ConvexHull(points)
        if len(points) <= vertex_limit and len(hull.simplices) <= face_limit:
            result.append((points, hull.simplices))
            return
        if depth > 32:
            raise ValueError(
                f"Source hull partition depth exceeded: {len(points)} {np.ptp(points, axis=0)}"
            )
        axis = int(np.argmax(np.ptp(points, axis=0)))
        plane = float((points[:, axis].min() + points[:, axis].max()) / 2)
        edge_faces = {}
        for i, face in enumerate(hull.simplices):
            for a, b in zip(face, np.roll(face, -1)):
                edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(i)
        edges = [
            e
            for e, faces in edge_faces.items()
            if len(faces) != 2
            or np.linalg.norm(hull.equations[faces[0]] - hull.equations[faces[1]]) > 1e-8
        ]
        cuts = []
        for a, b in edges:
            va, vb = points[a], points[b]
            if (va[axis] - plane) * (vb[axis] - plane) < 0:
                cuts.append(va + (vb - va) * (plane - va[axis]) / (vb[axis] - va[axis]))
        for sign in [-1, 1]:
            keep = points[(points[:, axis] - plane) * sign >= -1e-10]
            child = np.unique(np.vstack([keep, cuts]), axis=0)
            split(child, depth + 1)

    split(vertices, 0)
    original = ConvexHull(vertices)
    total = sum(ConvexHull(p).volume for p, _ in result)
    tolerance = max(1e-10, original.volume * 1e-8)
    assert abs(total - original.volume) < tolerance
    assert all(
        np.max(p @ original.equations[:, :3].T + original.equations[:, 3]) < 1e-8 for p, _ in result
    )
    assert all(
        any(np.linalg.norm(p - v, axis=1).min() < 1e-8 for p, _ in result)
        for v in vertices[original.vertices]
    )
    return result, dict(
        original_hull_vertices=len(original.vertices),
        pieces=len(result),
        piece_vertices=[len(p) for p, _ in result],
        piece_triangles=[len(f) for _, f in result],
        original_volume=original.volume,
        partition_volume=total,
        max_outward_error=max(
            float(np.max(p @ original.equations[:, :3].T + original.equations[:, 3]))
            for p, _ in result
        ),
        policy="Exact source convex hull split by planes; unchanged visual geometry. All source hull extrema retained, no outward padding. Native GPU cooking/support validation remains required.",
    )
