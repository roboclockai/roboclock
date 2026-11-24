"""Lab C1-C3 as batch C tasks (score_version 2, docs/c-class-timing.md)."""

import json
from dataclasses import replace

import pytest
from helpers import approach_targets, at_target, observe_for, snapshot

from roboclock.config import load_config
from roboclock.envs.lab.layout import make_layout
from roboclock.envs.lab.tasks.registry import make_task
from roboclock.evaluation import throughput
from roboclock.evaluation.schema import validate_row
from roboclock.evaluation.score import score_fields

C_TASKS = ("C1", "C2", "C3")


def seat(task, names, at, settle=0.25):
    """From time `at` on, observe `names` seated in their own targets, the rest at source."""
    task.time = at
    state = snapshot(task, **{name: at_target(task, name) for name in names})
    approach_targets(task, state)
    observe_for(task, state, settle)
    return state


def seat_in_sequence(task, finish, count=None):
    """Seat the first `count` samples one after another, the last one from `finish` on."""
    names = list(task.layout.objects)[: count or None]
    step = finish / len(names)
    state = None
    for k in range(len(names)):
        state = seat(task, names[: k + 1], step * (k + 1))
    return state


def row(task):
    return {"environment": "lab", **task.metrics()}


@pytest.mark.parametrize("task_id", C_TASKS)
def test_episode_is_the_same_at_every_difficulty(task_id):
    config = load_config()
    specs = {tier: make_task(task_id, tier, 11, config).spec for tier in throughput.TIERS}
    for tier, spec in specs.items():
        assert replace(spec, difficulty="medium") == specs["medium"]
        assert (spec.position_tolerance, spec.orientation_tolerance) == (0.018, 0.2)
        t_o = throughput.oracle_time("lab", task_id, spec.num_samples)
        assert spec.max_time == pytest.approx(throughput.horizon(t_o)) == pytest.approx(5 * t_o)
        assert spec.required_rate is None
    tasks = [make_task(task_id, tier, 11, config) for tier in ("easy", "hard")]
    layouts = [make_layout(task, config) for task in tasks]
    for name, obj in layouts[0].objects.items():
        assert (obj.position == layouts[1].objects[name].position).all()
    assert layouts[0].assignments == layouts[1].assignments


@pytest.mark.parametrize("task_id,seed,count", [("C1", 52, 29), ("C2", 50, 27), ("C3", 50, 27)])
def test_seeds_keep_their_recorded_sample_counts(task_id, seed, count):
    # data/lab/C*/demo.json and the oracle demos behind t_o used these counts.
    for tier in throughput.TIERS:
        assert make_task(task_id, tier, seed).spec.num_samples == count


@pytest.mark.parametrize("task_id", C_TASKS)
def test_instruction_states_no_seconds_and_no_rate(task_id, task_factory):
    task = task_factory(task_id)
    text = task.gen_instruction()
    assert f"Process all {task.spec.num_samples} samples as quickly as possible." in text
    assert text.count("as quickly as possible") == 1
    for word in ("second", "minute", "%", "accuracy", "within"):
        assert word not in text


@pytest.mark.parametrize("task_id", C_TASKS)
@pytest.mark.parametrize(
    "finish,tiers",
    [
        (0.5, {"easy": True, "medium": True, "hard": True}),
        (1.5, {"easy": True, "medium": True, "hard": False}),
        (3.0, {"easy": True, "medium": False, "hard": False}),
        (4.2, {"easy": False, "medium": False, "hard": False}),
    ],
)
def test_temporal_per_tier_from_t_q(task_factory, task_id, finish, tiers):
    task = task_factory(task_id)
    t_o = task.oracle_time
    seat_in_sequence(task, finish * t_o)
    result = task.metrics()
    assert result["t_q_s"] == task.completion_time == pytest.approx(finish * t_o + 0.2, abs=0.02)
    assert result["job_times_s"][-1] == pytest.approx(result["t_q_s"])
    assert result["jobs_done"] == result["jobs_required"] == task.spec.num_samples
    assert result["task_success"] and result["misplaced"] == 0
    assert result["temporal_success_by_tier"] == tiers
    assert result["joint_success_by_tier"] == tiers
    assert not task.terminated  # not yet settled for SETTLE_S, and progress kept the stall off
    for tier in throughput.TIERS:
        task.spec = replace(task.spec, difficulty=tier)
        assert task.temporal_success == tiers[tier] == task.metrics()["joint_success"]
        assert task.metrics()["temporal_violations"] == ([] if tiers[tier] else ["over_budget"])


def test_physical_needs_every_sample(task_factory):
    task = task_factory("C2")
    q = task.spec.num_samples
    seat_in_sequence(task, 0.5 * task.oracle_time, count=q - 1)
    assert task.goal_achieved is False and not task.physical_success
    result = row(task)
    assert result["jobs_done"] == q - 1 and result["t_q_s"] is None
    assert not any(result["temporal_success_by_tier"].values())
    assert result["temporal_violations"] == ["not_all_done"]
    assert score_fields("lab", result)["score"] == round(80 * (q - 1) / (q + 1), 1)


def test_physical_is_judged_at_the_end_and_independent_of_time(task_factory):
    task = task_factory("C2")
    state = seat_in_sequence(task, 0.5 * task.oracle_time)
    assert task.physical_success
    t_q = task.completion_time
    # A placed sample lifted out of its slot afterwards: physical fails, t_Q and temporal stay.
    lifted = replace(state.objects["sample_0"], grasped=True)
    task.observe(replace(state, objects={**state.objects, "sample_0": lifted}), 0.01)
    result = task.metrics()
    assert not result["task_success"] and not result["joint_success"]
    assert result["temporal_success"] and result["t_q_s"] == t_q
    # Back in its slot: physical again, scored at the frame the episode ends on.
    task.observe(state, 0.01)
    assert task.physical_success and task.completion_time == t_q


def test_misplacements_are_counted_against_the_accuracy_allowance(task_factory):
    task = task_factory("C2")
    q = task.spec.num_samples
    assert task.allowed_misplacements == throughput.allowed_misplacements(0.95, q) == 1
    wrong = at_target(task, "sample_1")  # sample_0 resting in sample_1's slot
    observe_for(task, snapshot(task, sample_0=wrong))
    assert task.misplaced() == 1
    assert task.metrics()["misplaced"] == 1


@pytest.mark.parametrize("task_id", C_TASKS)
def test_all_done_ends_after_settling(task_factory, task_id):
    task = task_factory(task_id)
    state = seat(task, list(task.layout.objects), at=1.0)
    t_q = task.completion_time
    assert t_q is not None and not task.terminated
    # One invalid frame (a sample lifted) restarts the settle clock.
    task.time += 0.5
    lifted = replace(state.objects["sample_0"], grasped=True)
    task.observe(replace(state, objects={**state.objects, "sample_0": lifted}), 0.01)
    restart = None
    while not task.terminated:
        task.time += 0.01
        task.observe(state, 0.01)
        restart = restart or task.valid_since
        assert task.time < t_q + 5.0
    assert task.time - restart == pytest.approx(throughput.SETTLE_S, abs=0.011)
    result = row(task)
    assert result["termination_reason"] == "all_done"
    assert result["t_q_s"] == t_q and result["joint_success"]
    assert {e["type"] for e in task.event_log} >= {"all_done"}


def test_stall_stop_after_t_medium(task_factory):
    task = task_factory("C3")
    t_o = task.oracle_time
    names = list(task.layout.objects)
    seat(task, names[:3], at=1.0)
    state = seat(task, names[:4], at=1.9 * t_o)
    last = task.last_job_time
    assert last == pytest.approx(1.9 * t_o + 0.2, abs=0.02)
    for now in (2.0 * t_o, last + 0.49 * t_o):
        task.time = now
        task.observe(state, 0.01)
        assert not task.terminated, now
    task.time = last + 0.5 * t_o + 0.01  # the next tick
    task.observe(state, 0.01)
    assert task.terminated and task.end_reason == "stall"
    result = row(task)
    assert result["termination_reason"] == "stall"
    assert not result["task_success"] and not result["temporal_success"]
    assert result["jobs_done"] == 4


def test_stall_is_not_armed_before_t_medium(task_factory):
    task = task_factory("C1")
    state = snapshot(task)
    task.time = throughput.MULTIPLIER["medium"] * task.oracle_time - 0.05
    task.observe(state, 0.01)
    assert not task.terminated
    task.time = throughput.MULTIPLIER["medium"] * task.oracle_time
    task.observe(state, 0.01)
    assert task.end_reason == "stall"


def test_horizon_reason_and_row_schema(task_factory):
    task = task_factory("C2")
    assert task.metrics()["termination_reason"] is None  # still running
    task.time = task.spec.max_time
    result = row(task)
    assert result["termination_reason"] == "horizon"
    assert result["horizon_s"] == pytest.approx(task.spec.max_time)
    assert validate_row(result) == []
    json.dumps(result, allow_nan=False)
    assert "required_operations_per_minute" not in result


def test_row_scores_every_tier_from_one_run(task_factory):
    task = task_factory("C3")
    seat_in_sequence(task, 1.5 * task.oracle_time)
    result = row(task)
    assert validate_row(result) == []
    fields = score_fields("lab", result)
    assert fields["score"] == 100.0  # easy
    assert fields["score_by_tier"] == {"easy": 100.0, "medium": 100.0, "hard": 80.0}


def leaning(task, name, angle, bottom_offset=(0.0, 0.0)):
    """`name` standing in its own hole, tilted by `angle` about x around its bottom face."""
    import numpy as np

    target = task.layout.targets[task.layout.assignments[name]]
    height = task.layout.dimensions[name][1]
    axis = np.array([0.0, -np.sin(angle), np.cos(angle)])
    bottom = target.position - np.array([0.0, 0.0, height / 2])
    bottom[:2] += bottom_offset
    return replace(
        at_target(task, name),
        position=bottom + axis * height / 2,
        orientation=np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]),
    )


def test_c1_tube_leaning_against_its_hole_wall_stays_valid(task_factory):
    # Oracle seed 0: all 25 tubes credited, then one leaned 0.04 rad in its hole (3.7 mm off);
    # `seated` (whole tilted envelope inside the footprint) rejected it, so the scene never
    # held valid and the episode stalled with physical false.
    from roboclock.envs.lab.mdp.geometry import in_slot, seated

    task = task_factory("C1")
    state = seat_in_sequence(task, 0.5 * task.oracle_time)
    assert task.physical_success
    name = "sample_0"
    target = task.layout.targets[task.layout.assignments[name]]
    tilted = leaning(task, name, 0.0385, bottom_offset=(0.001, 0.0))
    assert not seated(task, name, tilted, target) and in_slot(task, name, tilted, target)
    observe_for(task, replace(state, objects={**state.objects, name: tilted}), 2.5)
    result = task.metrics()
    assert result["task_success"] and result["misplaced"] == 0
    assert result["invalid_at_end"] == []
    assert task.terminated and result["termination_reason"] == "all_done"


@pytest.mark.parametrize(
    "angle,bottom_offset,lift",
    [
        (0.15, (0.0, 0.0), 0.0),  # axis crosses the rim outside the hole
        (0.0, (0.0, 0.006), 0.0),  # bottom outside the floor footprint
        (0.0, (0.0, 0.0), 0.010),  # resting on something, off the floor
    ],
)
def test_c1_tube_not_standing_in_its_hole_is_not_valid(task_factory, angle, bottom_offset, lift):
    from roboclock.envs.lab.mdp.geometry import in_slot

    task = task_factory("C1")
    state = seat_in_sequence(task, 0.5 * task.oracle_time)
    name = "sample_0"
    target = task.layout.targets[task.layout.assignments[name]]
    pose = leaning(task, name, angle, bottom_offset)
    pose = replace(pose, position=pose.position + [0.0, 0.0, lift])
    assert not in_slot(task, name, pose, target)
    task.observe(replace(state, objects={**state.objects, name: pose}), 0.01)
    assert not task.physical_success
    assert [row["name"] for row in task.metrics()["invalid_at_end"]] == [name]


def test_c1_in_slot_accepts_every_seated_pose(task_factory):
    import numpy as np

    from roboclock.envs.lab.mdp.geometry import in_slot, seated
    from roboclock.envs.lab.mdp.state import ObjectState

    task = task_factory("C1")
    name = "sample_0"
    target = task.layout.targets[task.layout.assignments[name]]
    rng = np.random.default_rng(0)
    seated_count = 0
    for _ in range(4000):
        q = np.array([1.0, *rng.normal(0, 0.03, 3)])
        pose = ObjectState(
            target.position + rng.normal(0, [0.003, 0.003, 0.002]), q / np.linalg.norm(q)
        )
        if seated(task, name, pose, target):
            seated_count += 1
            assert in_slot(task, name, pose, target)
    assert seated_count > 200
