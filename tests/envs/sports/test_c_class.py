"""Sports C-class episodes and rows (score_version 2, docs/c-class-timing.md), simulator-free."""

import re
from dataclasses import asdict

import pytest

from roboclock.envs.sports import c_class
from roboclock.envs.sports.stages import STAGES
from roboclock.envs.sports.tasks import make_task
from roboclock.evaluation import throughput
from roboclock.evaluation.schema import validate_row
from roboclock.evaluation.score import score_fields

LEVELS = ("easy", "medium", "hard")
SECONDS = re.compile(r"\d\s*s\b|second", re.IGNORECASE)


@pytest.mark.parametrize("task_id", ["C1", "C2"])
@pytest.mark.parametrize("seed", [0, 5])
def test_batch_episode_spec_is_identical_across_difficulties(task_id, seed):
    specs = [asdict(make_task(task_id, level, seed).spec) for level in LEVELS]
    for spec in specs:
        assert spec.pop("difficulty") in LEVELS
    assert specs[0] == specs[1] == specs[2]
    start = 2.0 if task_id == "C2" else 0.0
    t_o = throughput.oracle_time("sports", task_id)
    assert specs[0]["max_time"] == pytest.approx(start + 5 * t_o)
    timing = specs[0]["timing"]
    assert timing["time_origin"] == start and timing["oracle_time"] == t_o
    assert timing["budget_by_tier"] == throughput.budgets(t_o)


@pytest.mark.parametrize("task_id", ["C1", "C2"])
def test_batch_native_episode_is_identical_across_difficulties(task_id):
    configs = [make_task(task_id, level, 3).native_config for level in LEVELS]
    if task_id == "C1":
        assert configs[0]["rack"] == configs[1]["rack"] == configs[2]["rack"]
    else:
        cups = [
            {k: v for k, v in c["workflow"].items() if k.startswith("cups") or k == "preparation"}
            for c in configs
        ]
        assert cups[0] == cups[1] == cups[2]
    for name in ("scene", "sim", "embodiment"):
        assert configs[0][name] == configs[1][name] == configs[2][name]


@pytest.mark.parametrize("task_id", ["C1", "C2"])
@pytest.mark.parametrize("level", LEVELS)
def test_batch_instruction_asks_for_speed_without_seconds(task_id, level):
    instruction = make_task(task_id, level, 0).gen_instruction()
    assert "as quickly as possible" in instruction
    assert not SECONDS.search(instruction), instruction


def test_stream_task_keeps_its_schedule_and_horizon():
    for level in LEVELS:
        spec = make_task("C3", level, 0).spec
        timing = spec.timing
        assert timing["deadlines"][0] == timing["preparation"]
        assert spec.max_time == pytest.approx(1.25 * timing["deadlines"][-1])
        assert "served late" in spec.instruction


def batch_row(task_id, difficulty, job_times, physical, start, t_q):
    task = make_task(task_id, difficulty, 0)
    judged = c_class.judge(
        task_id,
        job_times,
        len(job_times),
        physical,
        difficulty,
        start=start,
        t_q=t_q,
        termination_reason="all_done",
    )

    class Adapter:
        ticks, time, done = 10, t_q or 1.0, True

        def result(self):
            return {
                "task": task_id,
                "physical_success": physical,
                "temporal_success": judged["temporal_success"],
                "success": judged["joint_success"],
                "failures": [],
                "completed_at": t_q,
                "throughput": judged,
                "difficulty": difficulty,
                "seed": 0,
            }

        def events(self):
            return []

        def joint_success(self):
            return judged["joint_success"]

    task.bind(Adapter())
    return task.metrics()


@pytest.mark.parametrize("level", LEVELS)
def test_metrics_row_keeps_the_judged_verdicts(level):
    t_o = throughput.oracle_time("sports", "C2")
    second = 2.0 + 1.5 * t_o  # within easy and medium, beyond hard
    row = batch_row("C2", level, [10.0, second], True, 2.0, second)
    expected = {"easy": True, "medium": True, "hard": False}
    assert row["temporal_success_by_tier"] == expected
    assert row["temporal_success"] == expected[level] == row["joint_success"]
    assert row["task_success"] and row["manipulation_success"]
    assert row["completion_time"] == second and row["termination_reason"] == "all_done"
    assert "throughput" not in row["native_result"]
    assert validate_row({**row, "environment": "sports"}) == []
    assert row["temporal_violations"] == ([] if expected[level] else ["t_q_over_budget"])
    scores = score_fields("sports", row)["score_by_tier"]
    assert scores == {tier: 100.0 if ok else 80.0 for tier, ok in expected.items()}


def test_physical_failure_row_is_never_joint():
    row = batch_row("C1", "easy", [5.0] * 15, False, 0.0, 100.0)
    assert row["temporal_success"] and not row["task_success"] and not row["joint_success"]
    assert validate_row({**row, "environment": "sports"}) == []


def test_unfinished_rack_row_has_no_completion_time():
    row = batch_row("C1", "easy", [5.0] * 15, True, 0.0, None)
    assert row["t_q_s"] is None and row["completion_time"] is None
    assert not row["temporal_success"] and row["temporal_violations"] == ["jobs_incomplete"]


def test_speed_cups_stages_count_both_cards():
    stages = STAGES["C2"]
    assert len(stages) == 12 and stages[-1].score == round(80 * 12 / 13, 1)
    first = [{"kind": "cup_placed", "card": 0}] * 6
    second = [{"kind": "cup_placed", "card": 1}] * 3
    later = [{"kind": "cup_placed", "card": 2}] * 3
    row = {"task_id": "C2", "task_success": False, "joint_success": False}
    assert score_fields("sports", {**row, "events": first})["score"] == round(80 * 6 / 13, 1)
    assert score_fields("sports", {**row, "events": first + second + later})["score"] == round(
        80 * 9 / 13, 1
    )
