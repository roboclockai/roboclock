"""Pose-space skin corrective for the visual human's cylindrical power grip.

Linear-blend skinning collapses the inside of strongly flexed finger joints.
Restore the skin envelope against the measured 14 mm shaft, including triangle
interiors. This is a VISUAL corrective, not a human contact/force simulation.
It never modifies the bar, robot, bone pose, or benchmark contact predicates.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def triangle_nearest(triangles):
    """Closest point to the origin in projected XZ triangles, plus distance."""
    p = triangles
    q = np.roll(p, -1, axis=1)
    edges = q - p
    u = np.clip(-np.sum(p * edges, axis=2) / np.maximum(np.sum(edges * edges, axis=2), 1e-20), 0, 1)
    candidates = p + u[..., None] * edges
    k = np.argmin(np.sum(candidates * candidates, axis=2), axis=1)
    closest = candidates[np.arange(len(p)), k]
    cross = edges[..., 0] * (-p[..., 1]) - edges[..., 1] * (-p[..., 0])
    area = np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    inside = (np.all(cross > 1e-12, axis=1) | np.all(cross < -1e-12, axis=1)) & (area > 1e-12)
    distance = np.linalg.norm(closest, axis=1)
    distance[inside] = 0
    closest[inside] = p[inside].mean(axis=1)
    return closest, distance


class HandSurface:
    corrective_limit_m = 0.006

    def __init__(self, points, indices, weights, bind, joint_names, faces):
        self.original = np.array(points, dtype=float)
        self.all_faces = np.asarray(faces)
        hand_joint = np.array(["_Wrist" in j for j in joint_names])
        mask = (hand_joint[indices] * weights).sum(axis=1) > 0.75
        self.selected = np.flatnonzero(mask)
        self.indices, self.weights = indices[mask], weights[mask]
        self.inverse_bind = np.linalg.inv(bind)
        lookup = np.full(len(points), -1, int)
        lookup[mask] = np.arange(mask.sum())
        self.faces = lookup[faces[np.all(mask[faces], axis=1)]]
        self.points = np.c_[self.original[mask], np.ones(mask.sum())]

    def apply(self, global_bones, bar, orientation):
        matrices = (
            (global_bones @ self.inverse_bind)[self.indices] * self.weights[..., None, None]
        ).sum(axis=1)
        world = np.einsum("nij,nj->ni", matrices, self.points)[:, :3]
        rotation = Rotation.from_quat(np.roll(orientation, -1))
        relative = rotation.inv().apply(world - np.asarray(bar))
        # Conservative skin-to-shaft spacing. Only vertices/faces within the
        # actual shaft span are considered; released hands far away are untouched.
        target = 0.0155
        original = relative.copy()
        for _ in range(18):
            radial = relative[:, [0, 2]]
            norm = np.linalg.norm(radial, axis=1)
            active = (norm < target) & (np.abs(relative[:, 1]) < 0.72)
            if np.any(active):
                direction = radial[active] / np.maximum(norm[active, None], 1e-12)
                radial[active] = direction * target
                relative[:, [0, 2]] = radial
            nearest, distance = triangle_nearest(radial[self.faces])
            active = (distance < target - 0.00001) & np.all(
                np.abs(relative[self.faces, 1]) < 0.72, axis=1
            )
            if not np.any(active):
                break
            n = nearest[active]
            n /= np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-12)
            correction = np.zeros_like(radial)
            count = np.zeros(len(radial))
            for corner in range(3):
                vertex = self.faces[active, corner]
                depth = np.maximum(0.0, target - np.sum(radial[vertex] * n, axis=1))
                np.add.at(correction, vertex, n * depth[:, None])
                np.add.at(count, self.faces[active, corner], 1)
            relative[:, [0, 2]] += correction / np.maximum(count[:, None], 1)
        # Never hide a bad skeletal pose by inflating/folding the hand surface.
        # Excess required correction remains visible in metrics and fails audit.
        delta = relative - original
        requested = np.linalg.norm(delta, axis=1)
        delta *= np.minimum(1.0, self.corrective_limit_m / np.maximum(requested, 1e-12))[:, None]
        relative = original + delta
        corrected_world = rotation.apply(relative) + bar
        corrected_bind = np.linalg.solve(
            matrices[:, :3, :3], (corrected_world - matrices[:, :3, 3])[..., None]
        )[..., 0]
        points = self.original.copy()
        points[self.selected] = corrected_bind
        _, distance = triangle_nearest(relative[:, [0, 2]][self.faces])
        metrics = {
            "minimum_shaft_clearance_m": float(distance.min() - 0.014),
            "maximum_skin_corrective_m": float(np.linalg.norm(relative - original, axis=1).max()),
            "maximum_requested_corrective_m": float(requested.max()),
            "corrective_limit_m": self.corrective_limit_m,
            "checked_hand_triangles": len(self.faces),
        }
        return points, metrics

    def normals(self, points):
        """Smooth bind-space normals consistent with the corrected mesh points."""
        triangles = points[self.all_faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        normals = np.zeros_like(points)
        for corner in range(3):
            np.add.at(normals, self.all_faces[:, corner], face_normals)
        normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        return normals
