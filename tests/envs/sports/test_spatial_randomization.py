"""Seeded station offsets of the sports tasks (``roboclock.envs.sports.spatial``), config only."""

import numpy as np
import pytest

from roboclock.envs.sports import spatial
from roboclock.envs.sports.tasks import make_task

TASKS = tuple(spatial.TASK_KEYS)


@pytest.mark.parametrize("task_id", TASKS)
def test_offsets_are_seeded_tier_independent_and_leave_timing_alone(task_id):
    layouts = {}
    for seed in range(4):
        tiers = [make_task(task_id, d, seed) for d in ("easy", "medium", "hard")]
        assert tiers[0].spec.layout == tiers[1].spec.layout == tiers[2].spec.layout
        assert tiers[0].spec.layout == make_task(task_id, "easy", seed).spec.layout
        assert tiers[0].native_config["spatial_randomization"] == tiers[0].spec.layout
        layouts[seed] = tiers[0].spec.layout
    # Every seed, seed 0 included, is a distinct instance.
    assert len({repr(layout) for layout in layouts.values()}) == 4


def test_offsets_reach_the_configuration_every_consumer_reads():
    from roboclock.envs.sports.arcade.tasks import MalletMoles, TaikoTask
    from roboclock.envs.sports.arena.tasks import PAD_X, pad_center

    a1 = make_task("A1", "easy", 3)
    dx, dy = a1.spec.layout["station_offset"]
    assert a1.native_config["scene"]["board_center"] == pytest.approx([0.48 + dx, dy])
    assert a1.native_config["scene"]["clock_position"][:2] == pytest.approx([0.48 + dx, -0.31 + dy])

    for task_id, cls, control, nominal in (
        ("B5", TaikoTask, "pad_0", [0.51, 0.12]),
        ("B1", MalletMoles, "mole_4", [0.62, 0.0]),
    ):
        c = make_task(task_id, "hard", 2).native_config
        dx, dy = c["v2"]["station_offset"]
        layout = cls("hard", 2, c).layout
        assert layout.controls[control].position[:2] == pytest.approx(
            [nominal[0] + dx, nominal[1] + dy]
        )
        assert layout.bodies["mallet"].position[:2] == pytest.approx([0.30 + dx, -0.30 + dy])

    w = make_task("B2", "medium", 1).native_config["workflow"]
    dx, dy = w["pad_offset"]
    assert pad_center(w, "left") == pytest.approx([PAD_X + dx, 0.23 + dy, 1.06])

    b3 = make_task("B3", "easy", 0)
    w, drawn = b3.native_config["workflow"], b3.spec.layout
    assert w["rack_x"] == pytest.approx(0.23 + drawn["rack_dx"])
    assert w["rack_height"] == pytest.approx(1.08 + drawn["rack_dz"])
    assert w["bench_x"] == pytest.approx(0.32 + drawn["bench_dx"])
    assert w["initial_bar_offset"] == [*drawn["bar_offset"], 0.0]

    offsets = make_task("C2", "easy", 0).native_config["workflow"]["cup_source_offsets"]
    for side in ("left", "right"):
        xs = [0.21 + i * 0.07 + offsets[f"{side}_{i}"][0] for i in range(3)]
        # Neighbouring cups (base radius 32.5 mm) never touch.
        assert np.min(np.diff(xs)) > 0.065


def test_tasks_without_spatial_draws_have_an_empty_layout():
    for task_id in ("A2", "B4", "C1", "C3"):
        assert make_task(task_id, "easy", 0).spec.layout == {}
