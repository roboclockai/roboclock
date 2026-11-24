"""Visual concentration cues must agree with fraction timing and dilution."""

import numpy as np
import pytest

from roboclock.envs.lab.chromatography_visuals import (
    BAND_SPEED,
    BED_BOTTOM,
    BED_TOP,
    CLEAR_SOLVENT,
    band_concentrations,
    packing_colors,
    solution_tint,
)
from roboclock.envs.lab.tasks.registry import make_task


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_column_edges_and_outlet_match_fraction_windows(difficulty):
    task = make_task("A3", difficulty=difficulty)
    for i, (start, end) in enumerate(task.intervals):
        # The leading/trailing edges reach the bed bottom at the scored times.
        assert band_concentrations([BED_BOTTOM], start, task.intervals)[0, i] == 0
        assert band_concentrations([BED_BOTTOM], end, task.intervals)[0, i] == 0
        assert band_concentrations([BED_BOTTOM], (start + end) / 2, task.intervals)[0, i] == 1
        center = BED_BOTTOM + (start + end) / 2 * BAND_SPEED
        assert BED_BOTTOM < center < BED_TOP
        assert band_concentrations([center], 0, task.intervals)[0, i] == pytest.approx(1)
    assert band_concentrations([BED_BOTTOM], task.intervals[-1][1] + 1, task.intervals).sum() == 0


def test_soft_band_edges_and_bounded_packing_colors():
    task = make_task("A3")
    heights = np.linspace(BED_BOTTOM, BED_TOP, 2000)
    colors = packing_colors(heights, 0, task.intervals, task.colors)
    assert np.all(np.isfinite(colors)) and np.all((colors >= 0) & (colors <= 1))
    # No abrupt flat-color plug boundary on the packed bed.
    assert np.abs(np.diff(colors, axis=0)).max() < 0.09


def test_clear_eluent_and_fraction_dilution():
    task = make_task("A3")
    np.testing.assert_allclose(solution_tint([0, 0, 0, 0], 10, task.colors), CLEAR_SOLVENT)
    strong = solution_tint([10, 0, 0, 0], 10, task.colors)
    diluted = solution_tint([10, 0, 0, 0], 20, task.colors)
    assert np.all(strong <= diluted) and np.all(diluted <= CLEAR_SOLVENT)
    assert strong[2] > strong[0]  # Blue fraction remains blue through absorption.
