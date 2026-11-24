"""Process-window scoring for the ward A1-A2 tasks (no simulator)."""

from types import SimpleNamespace

import pytest

from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.tasks.registry import PROCESS_TASKS, make_task

DT = 1 / 120
# A1 starts its process only when the clamp is opened (its window is relative to the opening): the
# harness opens it at OPEN_AT, A2/A3 ignore the control.
OPEN_AT = 1.5


def opened(task):
    """The task as it stands once its device has been opened at OPEN_AT (A1; others unchanged)."""
    if hasattr(task, "open_time"):
        task.open_time = OPEN_AT
    return task


def probe(task_id, level="hard", seed=0):
    return opened(make_task(task_id, level, seed, load_config()))


def run(task_id, act_at=None, level="hard", seed=0, moved_at=None):
    """Advance the task with a device that completes its action at `act_at` (None: never)."""
    task = make_task(task_id, level, seed, load_config())
    task.layout = SimpleNamespace(objects={}, targets={})
    while not task.terminated:
        task.advance(DT)
        controls = {
            "device_moved": moved_at is not None and task.time >= moved_at,
            "device_opened": task.time >= OPEN_AT - DT / 2,
            "device_actuated": act_at is not None and task.time >= act_at,
        }
        task.evaluate(SimpleNamespace(controls=controls, objects={}), DT)
    return task


@pytest.mark.parametrize("task_id", sorted(PROCESS_TASKS))
def test_quantity_crosses_the_band_exactly_over_the_window(task_id):
    task = probe(task_id)
    lo, hi = task.band
    assert task.quantity(task.window_open) == pytest.approx(lo)
    assert task.quantity(task.window_close) == pytest.approx(hi)
    assert task.initial_quantity() >= 0.0


@pytest.mark.parametrize("task_id", sorted(PROCESS_TASKS))
def test_opening_is_shared_across_levels_and_width_tightens(task_id):
    config = load_config()
    for seed in range(10):
        specs = [make_task(task_id, level, seed, config).spec for level in ("easy", "medium", "hard")]
        assert len({spec.windows[0][0] for spec in specs}) == 1
        widths = [spec.windows[0][1] - spec.windows[0][0] for spec in specs]
        assert widths[0] > widths[1] > widths[2]


@pytest.mark.parametrize("task_id", sorted(PROCESS_TASKS))
def test_action_inside_the_window_succeeds_and_freezes_the_quantity(task_id):
    window = probe(task_id)
    middle = (window.window_open + window.window_close) / 2
    task = run(task_id, act_at=middle)
    assert task.temporal_success and task.physical_success and task.joint_success
    assert task.metrics()["quantity_outcome"] == "in_band"
    lo, hi = task.band
    assert lo <= task.quantity() <= hi
    assert task.time == pytest.approx(task.action_time + task.hold_after, abs=2 * DT)


@pytest.mark.parametrize("task_id", sorted(PROCESS_TASKS))
def test_early_and_late_actions_fail_on_timing_only(task_id):
    window = probe(task_id)
    early = run(task_id, act_at=window.window_open - 0.5)
    assert early.physical_success and not early.temporal_success
    assert "early_action" in early.violations and early.metrics()["quantity_outcome"] == "short"
    assert early.quantity() < early.band[0]
    late = run(task_id, act_at=window.window_close + 0.5)
    assert late.physical_success and not late.temporal_success
    assert "missed_window" in late.violations and "late_action" in late.violations
    assert late.metrics()["quantity_outcome"] == "overshoot" and late.quantity() > late.band[1]


@pytest.mark.parametrize("task_id", sorted(PROCESS_TASKS))
def test_no_action_overshoots_and_ends_at_max_time(task_id):
    task = run(task_id, act_at=None)
    assert not task.physical_success and not task.temporal_success
    assert "missed_window" in task.violations
    assert task.time == pytest.approx(task.spec.max_time, abs=2 * DT)
    assert task.metrics()["quantity_outcome"] == "overshoot"


def test_window_tasks_are_temporal_window_and_use_their_own_modules():
    names = set()
    for task_id, cls in PROCESS_TASKS.items():
        assert task_id in ("A1", "A2") and cls.category == "temporal_window"
        names.add(cls.bedside_module)
    # A3 was the syringe pump cutoff (window_syringe) until 2026-11-19; it is now the ventilation
    # rhythm task (several windows), tested in test_window_bvm.py.
    assert names == {"window_drip", "window_bags"}
