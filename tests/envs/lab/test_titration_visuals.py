"""A4's optical cues remain tied to volume and a legible physical meniscus."""

import numpy as np
import pytest

from roboclock.envs.lab.titration_visuals import (
    BURETTE_CAPACITY_ML,
    BURETTE_LENGTH,
    BURETTE_RADIUS,
    INITIAL_SOLUTION_ML,
    burette_profile,
    flask_profile,
    indicator_tint,
)


@pytest.mark.parametrize("delivered", [0, 16, 18, 20, 50])
def test_burette_bore_volume_and_meniscus_match_printed_scale(delivered):
    assert np.pi * BURETTE_RADIUS**2 * BURETTE_LENGTH * 1e6 == pytest.approx(50)
    profile = burette_profile(delivered)
    center = profile[-1][1]
    reading = BURETTE_CAPACITY_ML * (1 - center / BURETTE_LENGTH)
    assert reading == pytest.approx(delivered)
    assert profile[2][1] > center  # Concave wetting meniscus.


def test_flask_cross_section_contains_delivered_volume():
    for amount in [INITIAL_SOLUTION_ML, INITIAL_SOLUTION_ML + 18, INITIAL_SOLUTION_ML + 20]:
        profile = np.asarray(flask_profile(amount))
        # Integrate the radial outline, including the concave free surface.
        radii, heights = profile[:, 0], profile[:, 1]
        volume_ml = (
            np.sum(
                np.pi
                * (radii[1:] ** 2 + radii[1:] * radii[:-1] + radii[:-1] ** 2)
                / 3
                * np.diff(heights)
            )
            * 1e6
        )
        assert volume_ml == pytest.approx(amount, abs=1.5)
        assert np.all(np.isfinite(profile)) and np.max(radii) < 0.05


def test_indicator_is_clear_then_pale_pink_then_excess_pink():
    np.testing.assert_allclose(indicator_tint(16), [1, 1, 1])
    endpoint = indicator_tint(18)
    assert endpoint[0] > endpoint[2] > endpoint[1] > 0.9
    assert indicator_tint(20)[1] < endpoint[1] - 0.3
    assert np.all(indicator_tint(20) > 0)  # Absorbing transparent liquid, no solid paint.
