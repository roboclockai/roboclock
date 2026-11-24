from dataclasses import replace

import pytest
from helpers import snapshot

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.tasks.registry import make_task


def state(task, *, source=True, destination=False, grasp=False, released=False, bottom=None):
    s = snapshot(task)
    obj = replace(
        s.objects["sample_0"],
        grasped=grasp,
        fixture_contacts=frozenset({"destination_floor"}) if destination else frozenset(),
    )
    return replace(
        s,
        objects={"sample_0": obj},
        gripper_open=released,
        controls={
            "source_immersed": source,
            "source_fully_immersed": source,
            "destination_immersed": destination,
            **({} if bottom is None else {"sample_bottom_z": bottom}),
        },
    )


def test_film_timing_is_developer_lift_not_stop_bath_entry(task_factory):
    task = task_factory("A5")
    task.observe(state(task, grasp=True), 0.01)
    task.advance(12)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.action_time == pytest.approx(12)
    task.advance(2)
    task.observe(state(task, source=False, destination=True, grasp=True), 0.01)
    assert task.action_time == pytest.approx(12)
    assert task.temporal_success


def test_film_lifted_before_the_countdown_ends_is_early(task_factory):
    task = task_factory("A5")
    task.observe(state(task, grasp=True), 0.01)
    task.advance(11.99)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert "early_transfer" in task.violations


def test_film_lift_counts_from_the_first_centimetre_of_rise(task_factory):
    task = task_factory("A5")
    task.observe(state(task, grasp=True, bottom=0.785), 0.01)
    task.advance(11)
    # Settling on the floor and small wobble are not a lift.
    task.observe(state(task, grasp=True, bottom=0.780), 0.01)
    task.observe(state(task, grasp=True, bottom=0.789), 0.01)
    assert task.action_time is None
    task.advance(1.05)
    task.observe(state(task, grasp=True, bottom=0.795), 0.01)
    assert task.action_time == pytest.approx(12.05)
    assert task.metrics()["development_interval_s"] == pytest.approx(12.05)
    assert task.temporal_success


@pytest.mark.parametrize("difficulty,tolerance", [("easy", 4.0), ("medium", 2.0), ("hard", 1.0)])
def test_film_late_lift_window_follows_difficulty(difficulty, tolerance):
    config = load_config()
    task = make_task("A5", difficulty, 7, config)
    task.bind(make_layout(task, config))
    assert task.tolerance == tolerance
    task.observe(state(task, grasp=True), 0.01)
    task.advance(12 + tolerance + 0.01)
    task.observe(state(task, grasp=True), 0.01)
    assert task.failed and "overdevelopment" in task.violations


def test_heat_timing_is_actual_warm_bath_exit(task_factory):
    task = task_factory("A6")
    task.observe(state(task, grasp=True), 0.01)
    task.advance(30)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.action_time == pytest.approx(30)
    assert not task.physical_success
    task.advance(1)
    task.observe(state(task, source=False, destination=True, released=True), 0.25)
    assert task.joint_success


@pytest.mark.parametrize("task_id,time", [("A5", 11), ("A6", 25.9)])
def test_early_transfer_remains_invalid(task_factory, task_id, time):
    task = task_factory(task_id)
    task.observe(state(task, grasp=True), 0.01)
    task.advance(time)
    task.observe(state(task, source=False, destination=True, grasp=True), 0.01)
    task.advance(1)
    task.observe(state(task, source=False, destination=True, released=True), 0.25)
    assert task.physical_success and not task.temporal_success


@pytest.mark.parametrize("task_id,time", [("A5", 16.1), ("A6", 34.1)])
def test_missing_deadline_fails(task_factory, task_id, time):
    task = task_factory(task_id)
    task.observe(state(task, grasp=True), 0.01)
    task.advance(time)
    task.observe(state(task), 0.01)
    assert task.failed and task.terminated and not task.joint_success


def test_destination_hover_does_not_count_as_placement(task_factory):
    task = task_factory("A6")
    task.observe(state(task, grasp=True), 0.01)
    task.advance(30)
    task.observe(state(task, source=False, grasp=True), 0.01)
    s = state(task, source=False, destination=True, released=True)
    obj = replace(s.objects["sample_0"], fixture_contacts=frozenset())
    task.observe(replace(s, objects={"sample_0": obj}), 0.3)
    assert not task.physical_success


@pytest.mark.parametrize(
    "exit_time,violation",
    [(25.90, "early_transfer"), (26.10, None), (33.90, None), (34.0, None)],
)
def test_heat_shock_exit_has_the_levels_tolerance_on_both_sides(task_factory, exit_time, violation):
    # One physics tick (8 ms) was the old tolerance: no policy acting on what it sees can meet it.
    task = task_factory("A6")
    assert task.tolerance == pytest.approx(4.0)  # the factory's level is easy
    task.observe(state(task, grasp=True), 0.01)
    task.advance(exit_time)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.violations == ([violation] if violation else [])


def test_bath_instructions_state_their_windows():
    for level, tolerance in (("easy", 4.0), ("medium", 2.0), ("hard", 1.0)):
        a6 = make_task("A6", level, 0, load_config()).gen_instruction()
        assert f"between {30 - tolerance:.2f} and {30 + tolerance:.2f} seconds after immersion" in a6
        assert "immersed by 15 s" in a6
        assert "released in the ice bucket within 5 seconds of leaving" in a6
        a5 = make_task("A5", level, 0, load_config()).gen_instruction()
        assert "immersed by 15 s" in a5 and f"within {tolerance:.2f} s after the countdown ends" in a5
        assert "released within 5 s of the lift" in a5
        assert len(a5) <= 512 and len(a6) <= 512  # the observation space's limit


def test_every_bath_limit_is_at_least_one_second():
    for level in ("easy", "medium", "hard"):
        for task_id in ("A5", "A6"):
            assert make_task(task_id, level, 0, load_config()).tolerance >= 1.0


def test_film_transfer_limit_counts_from_a_late_lift(task_factory):
    # A lift near the end of the late window still has the full transfer limit.
    task = task_factory("A5")
    task.observe(state(task, grasp=True), 0.01)
    task.advance(12 + 3.9)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.temporal_success
    task.advance(4.5)
    task.observe(state(task, source=False, destination=True, grasp=True), 0.01)
    assert not task.failed
    task.observe(state(task, source=False, destination=True, released=True), 0.25)
    assert task.joint_success
    assert task.spec.max_time >= task.time + 3.0


def test_film_countdown_starts_only_after_controlled_full_immersion(task_factory):
    task = task_factory("A5")
    task.advance(5)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.source_entry_time is None and task.exit_time is None
    assert task.visual_state()["remaining"] == 12
    # A partial dip does not start the clock.
    partial = state(task, source=True, grasp=True)
    task.observe(
        replace(partial, controls={**partial.controls, "source_fully_immersed": False}), 0.01
    )
    assert task.source_entry_time is None
    task.observe(state(task, source=True, grasp=True), 0.01)
    assert task.source_entry_time == 5 and task.deadline == 17
    task.advance(10)
    assert task.visual_state()["remaining"] == 2
    task.observe(state(task, grasp=True), 0.01)
    assert task.action_time is None
    task.advance(2)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.action_time == 17 and task.temporal_success
    assert task.metrics()["development_interval_s"] == 12


def test_film_never_loaded_does_not_report_overdevelopment(task_factory):
    task = task_factory("A5")
    task.advance(16)
    task.observe(state(task, source=False), 0.01)
    assert task.failed
    assert "developer_loading_timeout" in task.physical_failures
    assert "overdevelopment" not in task.violations


def test_heat_shock_countdown_starts_at_warm_bath_immersion(task_factory):
    # The tube starts on the bench stand: the 30 s run from its immersion, not from t = 0.
    task = task_factory("A6")
    task.advance(4)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.source_entry_time is None and task.exit_time is None
    assert task.visual_state()["remaining"] == 30
    task.observe(state(task, grasp=True), 0.01)
    assert task.source_entry_time == 4 and task.deadline == 34
    task.advance(30)
    task.observe(state(task, source=False, grasp=True), 0.01)
    assert task.action_time == 34 and task.temporal_success


def test_tube_never_loaded_times_out(task_factory):
    task = task_factory("A6")
    task.advance(16)
    task.observe(state(task, source=False), 0.01)
    assert task.failed and "warm_bath_loading_timeout" in task.physical_failures
    assert "heat_exposure_exceeded" not in task.violations
