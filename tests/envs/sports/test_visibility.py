"""Depth diagnostics must reject arm occlusion, clipping and missing screens."""

import numpy as np

from roboclock.envs.sports.arcade.visibility import compare_depth


def test_display_depth_separates_notes_from_occluding_arm():
    depth = np.array([[1.985, 1.7, 2.0]])
    report = compare_depth(depth, np.array([[0, 0], [1, 0], [2, 0]]), np.array([2.0, 2.0, 2.0]))
    assert report["occluded"] == 1
    assert report["clear_fraction"] == 2 / 3


def test_clipped_missing_and_invalid_depth_do_not_count_as_clear():
    depth = np.array([[2.0, 2.5, np.nan, 0.0]])
    report = compare_depth(
        depth, np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]), np.full(5, 2.0)
    )
    assert report["in_frame"] == 4
    assert report["missing_surface"] == 1
    assert report["clear_fraction"] == 0.2
