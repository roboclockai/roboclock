"""Geometry edge cases that vertex-only penetration checks miss."""

import numpy as np

from roboclock.envs.sports.bench_press.character import align
from roboclock.envs.sports.bench_press.hand_surface import triangle_nearest


def test_triangle_interior_crosses_shaft_despite_clear_vertices():
    _, distance = triangle_nearest(np.array([[[-0.03, -0.03], [0.03, -0.03], [0, 0.04]]]))
    assert distance[0] == 0


def test_triangle_edge_is_closest_surface():
    _, distance = triangle_nearest(np.array([[[0.02, -0.02], [0.02, 0.02], [0.04, 0]]]))
    assert np.isclose(distance[0], 0.02)


def test_degenerate_projection_is_not_an_enclosing_face():
    _, distance = triangle_nearest(np.array([[[0.02, 0.01], [0.02, 0.01], [0.02, 0.01]]]))
    assert np.isclose(distance[0], np.hypot(0.02, 0.01))


def test_opposite_vertical_bones_keep_proper_rotation():
    r = align([0, 0, 1], [0, 0, -1])
    assert np.allclose(r @ [0, 0, 1], [0, 0, -1])
    assert np.isclose(np.linalg.det(r), 1)


def test_bad_pose_cannot_be_hidden_by_large_skin_displacement():
    from roboclock.envs.sports.bench_press.hand_surface import HandSurface

    points = np.array([[0.002, -0.01, 0.0], [0.002, 0.01, 0.0], [0.003, 0.0, 0.001]])
    skin = HandSurface(
        points,
        np.zeros((3, 1), int),
        np.ones((3, 1)),
        np.eye(4)[None],
        ["L_Wrist"],
        np.array([[0, 1, 2]]),
    )
    corrected, metrics = skin.apply(np.eye(4)[None], np.zeros(3), [1, 0, 0, 0])
    assert np.linalg.norm(corrected - points, axis=1).max() <= skin.corrective_limit_m + 1e-12
    assert metrics["maximum_requested_corrective_m"] > skin.corrective_limit_m
    assert metrics["minimum_shaft_clearance_m"] < 0
