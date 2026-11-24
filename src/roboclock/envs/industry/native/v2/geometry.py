"""Collision/visual envelope points in a declared rigid-body frame.

Convex-hull vertices preserve plane-support extrema without double-aligning
USD oriented bounds. They do not change any native collision shape.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def body_envelope_points(stage, body):
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(body).GetInverse()
    bounds = UsdGeom.BBoxCache(0, ["default", "render", "proxy"])
    groups = []
    for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            if points.size == 0:
                continue
        else:
            box = bounds.ComputeUntransformedBound(prim)
            region = box.GetRange()
            lo,hi = region.GetMin(),region.GetMax()
            if region.IsEmpty():
                continue
            points = np.array([box.GetMatrix().Transform(Gf.Vec3d(x,y,z))
                               for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]])
        transform = np.asarray(cache.GetLocalToWorldTransform(prim) * inverse)
        groups.append(points @ transform[:3,:3] + transform[3,:3])
    if not groups:
        return np.empty((0,3))
    points = np.unique(np.concatenate(groups), axis=0)
    if len(points) >= 4:
        try:
            return points[ConvexHull(points).vertices]
        except QhullError:
            pass
    return points
