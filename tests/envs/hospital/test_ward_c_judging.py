"""Ward C1-C4 under the C-class rule (score_version 2, docs/c-class-timing.md), no simulator."""

import re
from dataclasses import replace

import numpy as np
import pytest
from hospital_helpers import do_jobs, observe_for, snapshot, ward_solution

from roboclock.envs.hospital.config import load_config
from roboclock.envs.hospital.layout import make_layout
from roboclock.envs.hospital.tasks.base import WardThroughputTask
from roboclock.envs.hospital.tasks.registry import make_task
from roboclock.envs.lab.mdp.state import ObjectState
from roboclock.evaluation import throughput
from roboclock.evaluation.schema import validate_row
from roboclock.evaluation.score import score_fields

WARD_C = ("C1", "C2", "C3", "C4")
Q = {"C1": 8, "C2": 6, "C3": 12, "C4": 12}


def ward(task_id, difficulty="hard", seed=0):
    config = load_config()
    task = make_task(task_id, difficulty, seed, config)
    task.bind(make_layout(task, config))
    return task


def spaced(task, last):
    """Q job start times from 1 s to `last`, evenly spaced (never long enough to stall)."""
    return list(np.linspace(1.0, last, task.jobs_required))


def until_end(task, state, limit, dt=0.05):
    while not task.terminated and task.time < limit:
        task.time += dt
        task.observe(state, dt)


@pytest.mark.parametrize("task_id", WARD_C)
def test_episode_spec_is_the_same_at_every_difficulty(task_id):
    config = load_config()
    t_o = throughput.oracle_time("hospital", task_id)
    specs = [make_task(task_id, level, 4, config).spec for level in ("easy", "medium", "hard")]
    assert {replace(spec, difficulty="any") for spec in specs} == {
        replace(specs[0], difficulty="any")
    }
    spec = specs[0]
    assert spec.max_time == pytest.approx(throughput.horizon(t_o)) == pytest.approx(5 * t_o)
    assert spec.oracle_time == t_o and spec.num_items == Q[task_id]
    assert spec.cycle_time is None and spec.required_rate_per_min is None
    assert spec.orientation_tolerance == config["difficulty"]["medium"]["orientation_tolerance"]
    # The scene is seed-only.
    easy, hard = ward(task_id, "easy", 4), ward(task_id, "hard", 4)
    for name, obj in easy.layout.objects.items():
        np.testing.assert_array_equal(obj.position, hard.layout.objects[name].position)
    assert set(easy.layout.targets) == set(hard.layout.targets)


@pytest.mark.parametrize("task_id", WARD_C)
def test_instruction_asks_for_all_q_without_seconds_or_rates(task_id):
    task = ward(task_id)
    text = task.gen_instruction()
    assert isinstance(task, WardThroughputTask)
    assert f"all {Q[task_id]} " in text and "as quickly as possible" in text
    assert not re.search(r"\d\s*(s|sec|second|seconds)\b", text), text
    assert "second" not in text and "minute" not in text and "/min" not in text
    assert "%" not in text and "rate" not in text


@pytest.mark.parametrize("task_id", WARD_C)
def test_nothing_is_misplaced_at_the_start(task_id):
    for seed in range(6):
        task = ward(task_id, seed=seed)
        task.observe(snapshot(task), 0.01)
        assert task.misplaced() == 0 and not task.correct and not task.terminated


@pytest.mark.parametrize("task_id", WARD_C)
def test_all_q_done_is_physical_and_tiers_come_from_t_q(task_id):
    task = ward(task_id, "hard")
    t_o = task.oracle_time
    # The Q-th job lands between T_hard and T_medium.
    final = do_jobs(task, spaced(task, 1.5 * t_o))
    t_q = max(task.job_times.values())
    assert len(task.job_times) == Q[task_id] and 1.0 * t_o < t_q < 2.0 * t_o
    assert task.physical_success and not task.terminated  # all done, not yet settled 2 s
    observe_for(task, final, throughput.SETTLE_S + 0.05)
    assert task.terminated and task.termination_reason == "all_done"
    metrics = task.metrics()
    assert metrics["t_q_s"] == pytest.approx(t_q) and metrics["completion_time"] == metrics["t_q_s"]
    assert metrics["job_times_s"] == sorted(metrics["job_times_s"]) or task_id == "C1"
    assert len(metrics["job_times_s"]) == metrics["jobs_required"] == Q[task_id]
    assert metrics["jobs_done"] == Q[task_id] and metrics["misplaced"] == 0
    assert metrics["temporal_success_by_tier"] == {"easy": True, "medium": True, "hard": False}
    assert metrics["joint_success_by_tier"] == {"easy": True, "medium": True, "hard": False}
    assert metrics["task_success"] and not metrics["temporal_success"]  # judged at hard
    assert metrics["termination_reason"] == "all_done" and metrics["score_version"] == 2
    assert metrics["temporal_violations"] == ["late_completion"]
    assert validate_row({**metrics, "category": None}, fields={}) == []
    scored = score_fields("hospital", metrics)
    assert scored["score"] == 80.0 and scored["score_by_tier"] == {
        "easy": 100.0,
        "medium": 100.0,
        "hard": 80.0,
    }


@pytest.mark.parametrize("task_id", WARD_C)
def test_the_same_run_judged_at_another_difficulty(task_id):
    """Difficulty only picks the tier that fills temporal/joint success."""
    rows = {}
    for level in ("easy", "medium", "hard"):
        task = ward(task_id, level)
        final = do_jobs(task, spaced(task, 0.9 * task.oracle_time), dt=0.02)
        observe_for(task, final, throughput.SETTLE_S + 0.1, dt=0.05)
        rows[level] = task.metrics()
    assert {r["t_q_s"] for r in rows.values()} == {rows["easy"]["t_q_s"]}
    assert all(r["joint_success"] and r["temporal_success"] for r in rows.values())


@pytest.mark.parametrize("task_id", WARD_C)
def test_fewer_than_q_jobs_is_neither_physical_nor_temporal(task_id):
    task = ward(task_id, "easy")
    names = list(ward_solution(task))
    do_jobs(task, spaced(task, 10.0)[:-1])
    assert len(task.correct) == Q[task_id] - 1 and names[-1] not in task.correct
    metrics = task.metrics()
    assert not metrics["task_success"] and metrics["t_q_s"] is None
    assert not any(metrics["temporal_success_by_tier"].values())
    assert metrics["job_times_s"].count(None) == 1


@pytest.mark.parametrize("task_id", WARD_C)
def test_every_done_job_must_still_hold_at_the_end(task_id):
    task = ward(task_id, "easy")
    solution = ward_solution(task)
    final = do_jobs(task, spaced(task, 10.0), solution)
    assert task.physical_success
    # One item is picked back up out of its container: not valid at the end any more.
    name = next(iter(solution))
    held = ObjectState(
        solution[name].position + [0, 0, 0.08], solution[name].orientation, grasped=True
    )
    state = replace(final, objects={**final.objects, name: held})
    observe_for(task, state, 0.5)
    assert name in task.correct  # credit is never revoked: the job time stands ...
    assert not task.physical_success  # ... but physical is judged on the final state
    metrics = task.metrics()
    assert metrics["temporal_success_by_tier"]["easy"] and not metrics["task_success"]
    assert not metrics["joint_success"]
    # Put back, the episode can still end done.
    observe_for(task, final, throughput.SETTLE_S + 0.05)
    assert task.termination_reason == "all_done" and task.metrics()["joint_success"]


@pytest.mark.parametrize("task_id", WARD_C)
def test_a_drop_ends_the_episode_as_a_failure(task_id):
    task = ward(task_id, "easy")
    final = do_jobs(task, [1.0, 3.0])
    name = next(n for n in task.layout.objects if n not in task.correct)
    below = ObjectState(np.array([0.3, 0.0, 0.2]), task.layout.objects[name].orientation.copy())
    task.observe(replace(final, objects={**final.objects, name: below}), 0.01)
    assert task.terminated and task.termination_reason == "failure"
    assert not task.physical_success and not task.metrics()["task_success"]


@pytest.mark.parametrize("task_id", WARD_C)
def test_stall_stop_from_t_medium(task_id):
    t_o = throughput.oracle_time("hospital", task_id)
    idle = ward(task_id)
    until_end(idle, snapshot(idle), idle.spec.max_time)
    assert idle.termination_reason == "stall"
    t_medium = throughput.MULTIPLIER["medium"] * t_o
    assert t_medium <= idle.time < t_medium + 0.1  # nothing done: stops at T_medium
    # Jobs until past T_medium, then nothing: stops one stall window after the last one
    # (0.5 t_o, or longer for few jobs: C1 and C2 get 1.5 easy-tier job intervals).
    window = throughput.stall_window(t_o, Q[task_id])
    assert window >= 0.5 * t_o
    task = ward(task_id)
    state = do_jobs(task, [1.8 * t_o, 2.1 * t_o, 2.4 * t_o])
    last = max(task.job_times.values())
    assert last > t_medium and not task.terminated
    until_end(task, state, task.spec.max_time)
    assert task.termination_reason == "stall"
    assert last + window <= task.time < last + window + 0.1
    assert task.metrics()["termination_reason"] == "stall"


@pytest.mark.parametrize("task_id", WARD_C)
def test_horizon_reason_when_nothing_ended_it_early(task_id):
    task = ward(task_id)
    assert task.termination_reason is None and task.metrics()["termination_reason"] is None
    task.time = task.spec.max_time
    assert not task.terminated  # the environment ends it at max_time
    assert task.metrics()["termination_reason"] == "horizon"


@pytest.mark.parametrize("task_id", WARD_C)
def test_stage_score_counts_jobs_below_completion(task_id):
    task = ward(task_id, "easy")
    do_jobs(task, [1.0, 2.0, 3.0])
    metrics = task.metrics()
    scored = score_fields("hospital", metrics)
    assert scored["score_stages"][-1] == "job_3_done"
    assert scored["score"] == pytest.approx(round(80 * 5 / (Q[task_id] + 3), 1))
    every = score_fields("hospital", {**metrics, "jobs_done": Q[task_id]})
    assert every["score_stages"][-1] == f"job_{Q[task_id]}_done" and every["score"] < 80


def test_c3_vial_tipped_over_at_the_end_is_not_valid():
    task = ward("C3", "easy")
    solution = ward_solution(task)
    final = do_jobs(task, spaced(task, 10.0), solution)
    assert task.physical_success
    name = next(iter(solution))
    tilt = 0.4  # beyond the fixed 0.2 rad tolerance
    tipped = ObjectState(
        solution[name].position,
        np.array([np.cos(tilt / 2), np.sin(tilt / 2), 0.0, 0.0]),
        fixture_contacts=solution[name].fixture_contacts,
    )
    task.observe(replace(final, objects={**final.objects, name: tipped}), 0.01)
    assert not task.physical_success


def test_c4_tube_first_racked_wrong_can_still_be_moved_to_its_rack():
    task = ward("C4", "easy")
    solution = ward_solution(task)
    name = next(iter(solution))
    wrong = next(w for w, k in task.well_class.items() if k != task.tube_class[name])
    target = task.layout.targets[wrong]
    in_wrong = ObjectState(
        np.array(
            [*target.position[:2], target.support_z + 0.001 + task.layout.shapes[name].height / 2]
        ),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )
    task.time = 1.0
    observe_for(task, snapshot(task, **{name: in_wrong}), 0.3)
    assert name in task.attempted and name not in task.correct and task.misplaced() == 1
    assert task.accuracy == 0.0
    do_jobs(task, spaced(task, 10.0), solution)
    assert name in task.correct and task.physical_success
    assert task.metrics()["placements"][name]["correct"] is False  # the first well stays on record


def _bound(task_id):
    config = load_config()
    task = make_task(task_id, "easy", 0, config)
    task.bind(make_layout(task, config))
    return task


def test_item_still_held_over_a_bin_is_not_misplaced():
    """Review 2026-11-09: a spare vial lowered into a bin but still in the gripper is not resting there."""
    from hospital_helpers import in_container

    task = _bound("C1")
    solution = ward_solution(task)
    final = do_jobs(task, [1 + k for k in range(8)], solution)
    assert task.physical_success
    spare = next(n for n in task.layout.objects if n not in solution)
    held = in_container(task, spare, "bin_0", grasped=True)
    task.time = task.spec.max_time - 0.05
    task.observe(replace(final, objects={**final.objects, spare: held}), 0.01)
    assert task.misplaced() == 0 and task.physical_success


def test_no_stall_while_a_complete_scene_settles():
    """Review 2026-11-09: complete and valid just before the stall would fire ends as all_done, not stall."""
    task = _bound("C2")
    t_o = task.oracle_time
    window = throughput.stall_window(t_o, task.jobs_required)
    solution = ward_solution(task)
    final = do_jobs(task, [1, 2, 3, 4, 5, 2 * t_o - 1.0], solution)
    t_q = max(task.job_times.values())
    name = next(iter(solution))
    held = replace(solution[name], grasped=True)  # lifted back out: not valid for a while
    observe_for(
        task,
        replace(final, objects={**final.objects, name: held}),
        (t_q + window - 1.0) - task.time,
    )
    assert not task.physical_success and task.termination_reason is None
    observe_for(task, final, 3.0)  # valid again; the stall time passes during the settle
    m = task.metrics()
    assert m["termination_reason"] == "all_done" and m["task_success"] and m["joint_success"]
