"""Seeded task instances, reset regeneration, and non-leaking public cues."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports.arena.config import config
from roboclock.envs.sports.arena.cues import shot_display
from roboclock.envs.sports.arena.tasks import Task


def test_shots_are_reproducible_with_meaningful_timing_and_pose_diversity():
    starts, positions = [], []
    for seed in range(100):
        c = config("A3", seed=seed)
        assert c == config("A3", seed=seed)
        starts.append(c["workflow"]["shot_start"])
        positions.append(c["basketball"]["pickup_xy"])
    starts, positions = np.array(starts), np.array(positions)
    assert ((2 <= starts) & (starts < 6)).all()
    assert (positions >= [0.28, -0.27]).all()
    assert (positions < [0.37, -0.17]).all()
    assert np.ptp(starts) > 3.5
    assert (np.ptp(positions, axis=0) > [0.08, 0.09]).all()
    assert len(set(starts)) == 100


@pytest.mark.parametrize("seed", [0, 1, 17, 99])
def test_difficulty_only_changes_time_budget(seed):
    easy = config("A3", "easy", seed)
    hard = config("A3", "hard", seed)
    for section in ("sim", "scene", "basketball", "randomization"):
        assert easy[section] == hard[section]
    assert easy["workflow"]["shot_start"] == hard["workflow"]["shot_start"]
    assert easy["workflow"]["shot_clock"] > hard["workflow"]["shot_clock"]


def test_public_cue_hides_future_start_then_exposes_active_window():
    c = config("A3", seed=17)["workflow"]
    start, end = c["shot_start"], c["shot_start"] + c["shot_clock"]
    assert shot_display(c, 0) == shot_display(c, start - 1e-6) == "WAIT FOR READY"
    assert shot_display(c, start).startswith("READY / SHOT CLOCK")
    assert shot_display(c, end) == "SHOT CLOCK / EXPIRED"


@pytest.mark.parametrize("seed", [0, 1, 17])
@pytest.mark.parametrize(
    "boundary,on_time", [("early", False), ("start", True), ("before_end", True), ("end", False)]
)
def test_sampled_window_controls_release_verdict(seed, boundary, on_time):
    cfg = config("A3", seed=seed)
    start = cfg["workflow"]["shot_start"]
    end = start + cfg["workflow"]["shot_clock"]
    release = {"early": start - 0.01, "start": start, "before_end": end - 0.01, "end": end}[
        boundary
    ]
    task = Task(SimpleNamespace(kind="A3", config=cfg, hoop=np.array([0.78, 0.06, 1.045])))
    obj = dict(
        position=np.array([0.5, 0.06, 1.2]),
        velocity=np.array([1.0, 0.0, 1.0]),
        forces=np.array([1.0, 1.0, 0.0]),
    )
    state = dict(
        time=release - 0.01,
        objects={"ball": obj},
        hands={"main": {"position": np.array([0.4, 0.06, 1.2])}},
    )
    task.update(state)
    obj = dict(obj, forces=np.zeros(3))
    task.update(dict(state, time=release, objects={"ball": obj}))
    assert task.release == release
    assert task.temporal == on_time
    assert ("release_outside_round" in task.failures) != on_time
