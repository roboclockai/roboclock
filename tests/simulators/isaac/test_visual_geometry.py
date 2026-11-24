import numpy as np
import pytest

from roboclock.simulators.isaac.visual_geometry import rounded_box


@pytest.mark.parametrize(
    "size,radius",
    [([1.0, 1.1, 0.1], 0.006), ([0.0012, 0.0012, 0.006], 0.0003), ([0.05, 0.006, 0.022], 0.0008)],
)
def test_bevel_mesh_keeps_collision_envelope_and_outward_normals(size, radius):
    points, normals, uv, indices = rounded_box(size, radius)
    assert np.isfinite(points).all() and np.isfinite(normals).all()
    assert np.all(np.abs(points) <= np.asarray(size) / 2 + 1e-10)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)
    assert uv.min() >= 0 and uv.max() <= 1
    faces = points[indices.reshape(-1, 4)]
    geometric = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    shading = normals[indices.reshape(-1, 4)].mean(axis=1)
    assert np.all(np.sum(geometric * shading, axis=1) > 0)
