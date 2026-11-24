"""Process boundaries and scoring use measured controls, independently of rendering."""

import numpy as np
import pytest

from roboclock.envs.lab.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.mdp.state import ObjectState, PhysicalState
from roboclock.envs.lab.tasks.registry import make_task


def state(task, opening=1.0, contact=True, receiver=0):
    obj = task.layout.objects["sample_0"]
    return PhysicalState(
        {"sample_0": ObjectState(obj.position, obj.orientation)},
        np.zeros(3),
        controls={"opening": opening, "robot_contact": contact, "receiver": receiver},
    )


def bind(task_id, difficulty="easy"):
    task = make_task(task_id, difficulty=difficulty)
    task.bind(make_layout(task, load_config()))
    return task


def simulate(task, duration, opening=1.0, contact=True, receiver=0):
    for _ in range(round(duration * 120)):
        task.advance(1 / 120)
        task.observe(state(task, opening, contact, receiver), 1 / 120)
        if task.terminated:
            break


@pytest.mark.parametrize(
    "key,expected",
    [
        ("A1", "water_collection"),
        ("A2", "phase_separation"),
        ("A3", "chromatography_fractions"),
        ("A4", "titration_endpoint"),
    ],
)
def test_new_registry_and_instructions(key, expected):
    task = bind(key)
    assert task.name == expected
    assert len(task.gen_instruction()) < 512
    assert task.liquid_process


@pytest.mark.parametrize("key,volume", [("A1", 485.0), ("A2", 110.0), ("A4", 18.6)])
def test_contact_supported_closure_at_valid_amount(key, volume):
    task = bind(key)
    if key == "A1":
        task.observe(state(task), 1 / 120)
    task.volume = volume
    task.time = task.spec.valid_start + 0.1
    task.observe(state(task, opening=0.0), 1 / 120)
    simulate(task, task.completion_dwell + 0.1, opening=0.0)
    assert task.joint_success
    assert task.action_time == pytest.approx(task.spec.valid_start + 0.1)


@pytest.mark.parametrize("key", ["A1", "A2", "A4"])
def test_passive_or_pre_target_closure_is_not_success(key):
    task = bind(key)
    if key == "A1":
        task.observe(state(task, contact=False), 1 / 120)
    simulate(task, 1.0, opening=0.0, contact=False)
    assert not task.physical_success
    assert "valve_closed_without_robot_contact" in task.physical_failures
    assert "early_action" in task.violations


@pytest.mark.parametrize(
    "key,failure",
    [("A1", "overflow"), ("A2", "upper_phase_contamination"), ("A4", "excess_titrant")],
)
def test_missed_deadline_is_latched(key, failure):
    task = bind(key)
    simulate(task, task.spec.valid_end + 0.2 + getattr(task, "failure_display", 0.0))
    assert task.process_failure and task.terminated
    assert failure in task.physical_failures
    task.observe(state(task, opening=0.0), 1 / 120)
    assert not task.joint_success


def test_phase_separation_shows_upper_phase_after_contamination():
    # The failure is latched when the lower phase runs out; the episode then runs on for
    # failure_display seconds with amber draining into the beaker.
    task = bind("A2")
    simulate(task, 24.0 + 0.5)
    assert task.process_failure and not task.terminated
    assert task.failed_at == pytest.approx(24.0, abs=0.02)
    assert task.event_log[-1]["time"] == pytest.approx(24.0, abs=0.02)
    visual = task.visual_state()
    assert visual["lower_ml"] == 0
    assert visual["received_lower_ml"] == pytest.approx(120.0, abs=1e-6)
    assert visual["received_upper_ml"] == pytest.approx(2.5, abs=0.1)
    assert visual["upper_ml"] == pytest.approx(97.5, abs=0.1)
    simulate(task, 1.2)
    assert task.terminated
    assert task.physical_failures.count("upper_phase_contamination") == 1
    assert not task.joint_success


def test_reduced_valve_flow_extends_physical_deadline():
    task = bind("A1")
    simulate(task, task.spec.valid_end, opening=0.5)
    assert not task.process_failure
    assert task.volume < task.minimum_volume


def test_seed_and_geometry_do_not_change_with_timing_difficulty():
    for key in ("A1", "A2", "A3", "A4"):
        easy, hard = bind(key), bind(key, "hard")
        np.testing.assert_array_equal(
            easy.layout.objects["sample_0"].position, hard.layout.objects["sample_0"].position
        )
        if key != "A3":
            assert (
                easy.spec.valid_end - easy.spec.valid_start
                > hard.spec.valid_end - hard.spec.valid_start
            )
        else:
            assert easy.intervals[1][0] - easy.intervals[0][1] == 20.0


def test_fractions_require_separate_receivers_and_removal():
    task = bind("A3", "hard")
    for _ in range(120 * 80):
        t = task.time + 1 / 120
        receiver = next(
            (i + 1 for i, (s, e) in enumerate(task.intervals) if s - 0.5 <= t <= e + 0.05), 0
        )
        task.observe(state(task, receiver=receiver), 1 / 120)
        task.advance(1 / 120)
        if task.terminated:
            break
    assert task.joint_success
    assert len(task.correct) == 4
    assert np.count_nonzero(task.collection[1:]) == 4


def test_missing_fraction_cannot_be_erased_by_later_receivers():
    task = bind("A3", "hard")
    simulate(task, 55, receiver=0)
    assert not task.joint_success
    assert "fraction_1_collection_gap" in task.violations


def test_throttled_flow_can_close_after_nominal_clock_window():
    task = bind("A1")
    simulate(task, 2 * task.spec.valid_start + 1.0, opening=0.5)
    assert task.time > task.spec.valid_end
    task.observe(state(task, opening=0.0), 1 / 120)
    simulate(task, task.completion_dwell + 0.1, opening=0.0)
    assert task.joint_success


def test_water_flow_is_one_faucet_and_difficulty_narrows_the_volume_band():
    tasks = {level: bind("A1", level) for level in ("easy", "medium", "hard")}
    assert {task.nominal_flow for task in tasks.values()} == {10.0}
    assert [task.minimum_volume for task in tasks.values()] == [440.0, 460.0, 480.0]
    for task in tasks.values():
        assert task.spec.valid_end == pytest.approx(50.0)
        assert task.spec.valid_end - task.spec.valid_start == pytest.approx(
            (500 - task.minimum_volume) / 10.0
        )
        assert f"{task.minimum_volume:g} mL" in task.gen_instruction()


def test_water_starts_closed_without_flow_or_false_closure():
    task = bind("A1")
    initial = task.volume
    simulate(task, 3.0, opening=0.0, contact=False)
    assert task.volume == initial and task.flow == 0
    assert task.opened_at is None and task.stopped_at is None
    assert not task.terminated and not task.physical_failures
    assert not task.violations and task.action_time is None


def test_water_opening_requires_contact_and_full_cycle_is_recorded():
    task = bind("A1")
    task.observe(state(task, contact=False), 1 / 120)
    assert "valve_opened_without_robot_contact" in task.physical_failures
    task.volume = 485
    simulate(task, 2.5, opening=0.0)
    assert not task.joint_success
    assert [e["type"] for e in task.event_log if e["type"].startswith("faucet_")] == [
        "faucet_opened",
        "faucet_fully_open",
        "faucet_closed",
    ]


def test_either_arm_can_grip_the_valve():
    """The valve's contact view lists both arms' fingers; a grip by the other arm counts too."""
    from roboclock.envs.lab.liquid_scene import liquid_grips

    none = [False] * 6
    assert liquid_grips(none, 0.03, 0.03) == (False, False)
    assert liquid_grips([True, True, False, False, False, False], 0.03, 0.08) == (True, False)
    assert liquid_grips([False, False, False, True, True, False], 0.08, 0.03) == (False, True)
    # One finger, or an open gripper, is a touch and not a grip.
    assert liquid_grips([False, False, False, True, False, True], 0.08, 0.03) == (False, False)
    assert liquid_grips([False, False, False, True, True, False], 0.08, 0.07) == (False, False)
    # A single-arm rig reports three columns.
    assert liquid_grips([True, True, False], 0.03, 0.0) == (True, False)


def test_a4_cue_is_the_first_visible_pink_step():
    from roboclock.envs.lab.policies.liquid import cue_visible
    from roboclock.envs.lab.titration_visuals import floor_step

    task = bind("A4", "hard")
    task.volume = 17.6
    assert floor_step(task.volume) == 0 and not cue_visible(task)
    task.volume = 17.75
    assert floor_step(task.volume) >= 1 and cue_visible(task)
    # A level, boundary or band that approaches its mark is visible from the start.
    for task_id in ("A1", "A2", "A3"):
        assert cue_visible(bind(task_id))


def test_a4_oracle_holds_the_start_pose_until_the_flask_shows_pink():
    from types import SimpleNamespace

    from roboclock.envs.lab.policies.liquid import LiquidTrajectory
    from roboclock.envs.lab.policies.oracle import OraclePickPlace

    task = bind("A4", "hard")
    oracle = object.__new__(OraclePickPlace)
    oracle.env = SimpleNamespace(task=task)
    oracle.planner = LiquidTrajectory(0.02)
    initial = np.arange(16, dtype=float) / 10
    oracle.reset({"joint_positions": initial})
    # 16 mL is where the old oracle began to close the lever on hard; nothing shows yet.
    for volume in (0.0, 16.0, 17.6):
        task.volume = volume
        np.testing.assert_array_equal(oracle.act({"joint_positions": initial + 0.001}), initial)
    task.volume = 17.75
    assert not oracle.planner.waiting_for_cue(task)
    # Seen once, the cue stays seen.
    task.volume = 17.0
    assert not oracle.planner.waiting_for_cue(task)
