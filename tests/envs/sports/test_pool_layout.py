"""Fixture resizing must preserve tool scale and collision/capture geometry."""
import numpy as np

from roboclock.envs.sports._shared.precision.pool_layout import PoolLayout
from roboclock.envs.sports._shared.precision.pool_x5 import cue_clears_jaws


def test_default_layout_preserves_existing_coordinates():
    layout = PoolLayout.from_config({})
    np.testing.assert_allclose(layout.point([.56, -.12, .774]), [.56, -.12, .774])


def test_compact_fixture_maps_throat_and_keeps_height():
    layout = PoolLayout(.6, .48, -.16)
    low, high = layout.bed_bounds
    np.testing.assert_allclose(high - low, [.564, .456])
    pocket = layout.point([1.25, .22])
    np.testing.assert_allclose(layout.throat(pocket), [high[0], pocket[1]])
    assert layout.point([.56, -.12, .774])[2] == .774
    np.testing.assert_allclose(layout.throat(layout.point([.15, -.12]))[0], low[0])


def test_capture_shrinks_around_unchanged_full_ball_envelope():
    layout = PoolLayout(.6, .48, -.16)
    # Fixture shrinks, ball radius does not; simple scaling of the original
    # center tolerance would incorrectly admit a ball through a pocket wall.
    half = layout.capture_half_extents(.023)
    np.testing.assert_allclose(half, [.0374, .029])
    assert np.all(half > 0)
    assert np.all(half < np.array([.077, .062]) * .6)


def test_cue_clearance_uses_moved_cushions_and_unscaled_shaft():
    layout = PoolLayout(.6, .48, -.16)
    tip = layout.point([.30, .05, .78])
    assert not cue_clears_jaws(tip, np.array([1., 0., 0.]), layout=layout)
    # Elevated shaft envelope clears the same translated cushion.
    tip[2] = .90
    assert cue_clears_jaws(tip, np.array([1., 0., 0.]), layout=layout)


def test_unscaled_cue_and_dock_are_inside_compact_fixture():
    layout = PoolLayout(.6, .58, -.16)
    home = layout.cue_home
    np.testing.assert_allclose(layout.point(layout.inverse_point(home)), home)
    assert cue_clears_jaws(home + [.177, 0., 0.], np.array([1., 0., 0.]),
                          travel=0., layout=layout, cue_rotation=np.eye(3))


def test_grip_saddle_can_block_a_shaft_only_clearance():
    layout = PoolLayout(.6, .58, -.16)
    # Shaft at z=.83 clears the .808-high jaw; the retained saddle extends
    # below it when rolled. The planner must check both tool components.
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    tip = np.array([.46, -.16, .83])
    assert cue_clears_jaws(tip, np.array([1., 0., 0.]), travel=0., layout=layout)
    assert not cue_clears_jaws(tip, np.array([1., 0., 0.]), travel=0.,
                               layout=layout, cue_rotation=rotation)


def test_ball_corridor_rejects_near_jaw_pot_despite_clear_centerline_to_mouth():
    layout = PoolLayout(.6, .58, -.04)
    pocket = layout.point([1.25, .22])
    assert not layout.ball_path_clear(layout.point([1.065, .10]), pocket, .023)
    assert layout.ball_path_clear(layout.point([.82, .10]), pocket, .023)


def test_training_mouth_width_changes_jaws_and_capture_together():
    original = PoolLayout(.6, .48, -.04)
    training = PoolLayout(.6, .48, -.04, .11)
    start, end = [.7076, -.0548], [.81, -.112]
    assert not original.ball_path_clear(start, end, .023)
    assert training.ball_path_clear(start, end, .023)
    assert training.end_segments[0][1] < original.end_segments[0][1]
    assert training.capture_half_extents(.023)[1] > original.capture_half_extents(.023)[1]
