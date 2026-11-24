"""Procedural meshes for soft things that boxes cannot show: draped cloth and hoses.

Pure numpy; the arrays plug into `visual_geometry.author_mesh` (quads: points, uv, indices) and carry
per-vertex normals. Nothing here has physics.
"""

import numpy as np


def _grid_quads(rows, cols, offset=0, flip=False):
    quads = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            k = offset + j * cols + i
            quad = [k, k + 1, k + 1 + cols, k + cols]
            quads.extend(quad[::-1] if flip else quad)
    return quads


def _grid_normals(points, outward):
    """Vertex normals of a (rows, cols, 3) grid, oriented toward `outward` (a vector or None)."""
    d_row = np.gradient(points, axis=0)
    d_col = np.gradient(points, axis=1)
    normals = np.cross(d_col, d_row)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(length, 1e-9)
    if outward is not None:
        flip = np.sum(normals * np.asarray(outward), axis=-1, keepdims=True) < 0
        normals = np.where(flip, -normals, normals)
    return normals


def cloth_arrays(
    x0, x1, y0, y1, surface, drop, folds=0.008, seed=0, nx=72, ny=32, nz=10, end_drop=None
):
    """A sheet draped over a surface: a top that follows `surface(x, y)`, hanging skirts on three sides.

    surface: function (x, y arrays) -> height of the cloth top before folds.
    drop: how far the skirts hang below the top edge. folds: amplitude of the creases in meters.
    Returns (points, uv, indices, normals) with every grid merged into one mesh.
    """
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, 6)

    def crease(x, y):
        return folds * (
            np.sin(23 * x + phase[0]) * np.cos(6 * y + phase[1])
            + 0.5 * np.sin(61 * x + 9 * y + phase[2])
            + 0.35 * np.sin(11 * x - 17 * y + phase[3])
        )

    xs, ys = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
    gx, gy = np.meshgrid(xs, ys)
    top = np.stack([gx, gy, surface(gx, gy) + crease(gx, gy)], axis=-1)

    def skirt(edge_points, outward, drop=drop):
        """Hang from an edge of the top: bulge outward a little and ripple the lower hem."""
        count = len(edge_points)
        t = np.linspace(0.0, 1.0, nz)[:, None, None]
        base = np.repeat(edge_points[None], nz, axis=0)
        hem = drop * (1 + 0.12 * np.sin(np.linspace(0, 9 * np.pi, count) + phase[4]))[None, :, None]
        out = np.asarray(outward, dtype=float)
        ripple = 0.010 * np.sin(np.linspace(0, 14 * np.pi, count) + phase[5])[None, :, None]
        points = base.copy()
        points[..., 2] = base[..., 2] - t[..., 0] * hem[..., 0]
        points += out * (0.012 * np.sin(np.pi * t) + ripple * t)
        return points

    grids = [
        (top, [0, 0, 1]),
        (skirt(top[0, :], [0, -1, 0]), [0, -1, 0]),
        (skirt(top[-1, :], [0, 1, 0]), [0, 1, 0]),
        (
            skirt(top[:, 0], [-1, 0, 0], drop if end_drop is None else end_drop),
            [-1, 0, 0],
        ),  # the near end
    ]
    points, normals, uv, indices, offset = [], [], [], [], 0
    for grid, outward in grids:
        rows, cols = grid.shape[:2]
        points.append(grid.reshape(-1, 3))
        normals.append(_grid_normals(grid, outward).reshape(-1, 3))
        uv.append(np.stack([grid[..., 0] / 0.6, grid[..., 1] / 0.6], axis=-1).reshape(-1, 2))
        indices.extend(_grid_quads(rows, cols, offset))
        offset += rows * cols
    return (
        np.concatenate(points),
        np.concatenate(uv),
        np.asarray(indices),
        np.concatenate(normals),
    )


def spline(control, samples=60):
    """Smooth Catmull-Rom curve through the control points."""
    p = np.asarray(control, dtype=float)
    p = np.vstack([p[0], p, p[-1]])
    per = max(2, samples // (len(p) - 3))
    curve = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            curve.append(
                0.5
                * (
                    2 * p1
                    + (-p0 + p2) * t
                    + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                    + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
                )
            )
    curve.append(p[-2])
    return np.asarray(curve)


def tube_arrays(control, radius, sides=10, ribs=0.0):
    """A hose along a smooth path. ribs > 0 adds corrugation of that depth (breathing circuit)."""
    path = spline(control)
    tangents = np.gradient(path, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    points, normals, uv = [], [], []
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for k, (center, tangent) in enumerate(zip(path, tangents)):
        ref = up if abs(tangent @ up) < 0.95 else np.array([1.0, 0.0, 0.0])
        n = np.cross(ref, tangent)
        n /= np.linalg.norm(n)
        b = np.cross(tangent, n)
        r = radius * (1 + ribs * np.sin(along[k] / 0.006))
        for s in range(sides + 1):
            angle = 2 * np.pi * s / sides
            direction = np.cos(angle) * n + np.sin(angle) * b
            points.append(center + r * direction)
            normals.append(direction)
            uv.append([along[k] / 0.3, s / sides])
    indices = _grid_quads(len(path), sides + 1)
    return np.asarray(points), np.asarray(uv), np.asarray(indices), np.asarray(normals)


def torso_arrays(
    cx, y_head, y_waist, top, height, width_head, width_waist, n=6.0, rings=48, around=44
):
    """A torso lying on its back: wider at the shoulders, narrower at the waist, rounded at both ends.

    The top stays flat across the middle (rounding only near the sides and ends), so things placed on the chest
    rest on it. y_head/y_waist are the ends along y, `top` is the height of the flat top.
    """
    ys = np.linspace(y_head, y_waist, rings)
    center_y, half = (y_head + y_waist) / 2, abs(y_waist - y_head) / 2
    grid = np.zeros((rings, around, 3))
    centers = np.zeros((rings, 3))
    for i, y in enumerate(ys):
        s = (y - center_y) / half
        f = np.sqrt(max(1.0 - abs(s) ** 8, 1e-4))  # blunt, rounded ends
        mix = (y - y_head) / (y_waist - y_head)
        w = (width_head + (width_waist - width_head) * mix) * f
        h = height * f
        zc = top - height / 2
        centers[i] = (cx, y, zc)
        for j in range(around):
            t = 2 * np.pi * j / around
            c, sn = np.cos(t), np.sin(t)
            grid[i, j] = (
                cx + w * np.sign(c) * abs(c) ** (2 / n),
                y,
                zc + (h / 2) * np.sign(sn) * abs(sn) ** (2 / n),
            )
    normals = _grid_normals(grid, None)
    outward = grid - centers[:, None, :]
    outward[..., 1] = 0.0  # the surface normal points away from the body's axis
    flip = np.sum(normals * outward, axis=-1, keepdims=True) < 0
    normals = np.where(flip, -normals, normals)
    uv = np.stack(
        [
            np.broadcast_to(np.linspace(0, 1, around), (rings, around)),
            np.repeat(np.linspace(0, 1, rings)[:, None], around, axis=1),
        ],
        axis=-1,
    )
    quads = []
    for i in range(rings - 1):
        for j in range(around):
            a = i * around + j
            b = i * around + (j + 1) % around
            quads.extend([a, b, b + around, a + around])
    return (
        grid.reshape(-1, 3),
        uv.reshape(-1, 2),
        np.asarray(quads),
        normals.reshape(-1, 3),
    )
