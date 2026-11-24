import numpy as np
import pytest

from roboclock.envs.lab.tube_liquid import TubeLiquid
from roboclock.physics.geometry import rotation_matrix


def pose(angle):
    angle = np.deg2rad(angle)
    return np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0])


def advance(model, angle, seconds):
    for _ in range(round(seconds * 120)):
        model.step(np.array([0.4, 0, 0.9]), pose(angle), np.zeros(3), 1 / 120)


def test_stationary_tilt_retains_volume_and_world_horizontal_surface():
    model = TubeLiquid(fill_fraction=0.8)
    advance(model, 30, 0.3)
    assert model.spilled_ml == 0
    assert np.allclose(rotation_matrix(pose(30)) @ model.normal, [0, 0, 1], atol=1e-8)
    integrated = np.sum(model.heights(model.xy) - model.base) * model.area
    assert integrated == pytest.approx(model.volume, abs=1e-11)


def test_overflow_depends_on_fill_not_fixed_angle():
    low, high = TubeLiquid(0.8), TubeLiquid(0.96)
    advance(low, 40, 0.4)
    advance(high, 40, 0.4)
    assert low.spilled_ml == 0
    assert high.spilled_ml > high.spill_limit_ml
    assert high.volume + high.spilled_ml * 1e-6 == pytest.approx(high.initial_volume)
    assert high.spilled_ml == pytest.approx(
        high.puddle_ml + len(high.drops) * 0.012 + high.drop_remainder
    )


def test_rescue_stops_flow_without_restoring_lost_sample():
    model = TubeLiquid(0.968)
    advance(model, 32, 0.12)
    assert 0 < model.spilled_ml < model.spill_limit_ml
    advance(model, 0, 0.6)
    escaped = model.spilled_ml
    advance(model, 0, 0.4)
    assert model.spilled_ml == escaped
    assert model.volume < model.initial_volume
    assert model.rim_clearance > 0


def test_slosh_is_dynamic_and_damped():
    model = TubeLiquid(0.8)
    advance(model, 0, 0.1)
    advance(model, 25, 1 / 120)
    assert abs(model.normal[0]) < 0.1
    advance(model, 25, 1.0)
    assert np.allclose(rotation_matrix(pose(25)) @ model.normal, [0, 0, 1], atol=0.001)
