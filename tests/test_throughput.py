"""C-class timing and judging (roboclock.evaluation.throughput, docs/c-class-timing.md)."""

import pytest

from roboclock.evaluation import throughput as tp
from roboclock.evaluation.schema import validate_row
from roboclock.evaluation.score import score_row


def test_multipliers_and_horizon():
    assert tp.MULTIPLIER == pytest.approx({"easy": 4.0, "medium": 5.0**0.5, "hard": 1.25})
    # hard leaves the reference oracle 20 % headroom; medium is the geometric mean
    assert tp.MULTIPLIER["hard"] * 0.8 == pytest.approx(1.0)
    assert tp.MULTIPLIER["medium"] ** 2 == pytest.approx(tp.MULTIPLIER["easy"] * tp.MULTIPLIER["hard"])
    assert tp.budgets(10.0) == pytest.approx({"easy": 40.0, "medium": 22.36068, "hard": 12.5})
    assert tp.horizon(10.0) == pytest.approx(50.0)


def test_every_batch_task_has_a_frozen_oracle_time():
    expected = (
        {("lab", t) for t in ("C1", "C2", "C3")}
        | {("hospital", t) for t in ("C1", "C2", "C3", "C4")}
        | {("industry", t) for t in ("C1", "C2", "C3", "C4")}
        | {("household", "C2"), ("household", "C3"), ("sports", "C1"), ("sports", "C2")}
    )
    assert set(tp.ORACLE_TIMES) == expected
    assert not set(tp.ORACLE_TIMES) & tp.STREAM_TASKS
    for key, entry in tp.ORACLE_TIMES.items():
        assert entry.seconds > 0 and entry.source, key


def test_per_item_oracle_time_needs_q():
    assert tp.oracle_time("lab", "C1", 25) == pytest.approx(3.977 * 25)
    with pytest.raises(ValueError):
        tp.oracle_time("lab", "C1")
    assert tp.oracle_time("industry", "C3") == 69.36


def test_stall_stop():
    t_o = 10.0
    assert not tp.stalled(22.3, None, t_o)  # not armed before T_medium = sqrt(5) t_o
    assert tp.stalled(22.4, None, t_o)  # armed, nothing done since the start
    assert not tp.stalled(22.4, 18.0, t_o)  # last job 4.4 s ago < 0.5 t_o
    assert tp.stalled(23.4, 18.0, t_o)
    assert not tp.stalled(23.4, 18.0, t_o, start=2.0)  # budget clock starts after preparation


def test_judge_tiers_from_one_run():
    t_o = 10.0
    row = tp.judge([3.0, 12.0, 25.0], 3, True, t_o, "easy")
    assert row["t_q_s"] == 25.0
    assert row["temporal_success_by_tier"] == {"easy": True, "medium": False, "hard": False}
    assert row["joint_success_by_tier"] == {"easy": True, "medium": False, "hard": False}
    assert row["temporal_success"] and row["joint_success"]
    hard = tp.judge([3.0, 12.0, 25.0], 3, True, t_o, "hard")
    assert hard["task_success"] and not hard["temporal_success"] and not hard["joint_success"]


def test_judge_physical_and_temporal_are_independent():
    # Done on time but a job fell out afterwards: temporal holds, physical does not.
    row = tp.judge([1.0, 2.0], 2, False, 10.0, "hard")
    assert row["temporal_success"] and not row["task_success"] and not row["joint_success"]
    # Not all jobs done: neither.
    row = tp.judge([1.0, None], 2, False, 10.0, "easy")
    assert row["t_q_s"] is None and not row["temporal_success"]


def test_judge_budget_counts_from_start_and_tolerance():
    row = tp.judge([14.52], 1, True, 10.0, "hard", start=2.0)
    assert row["temporal_success"]  # 14.52 - 2 = 12.52 <= T_hard + tolerance
    row = tp.judge([14.53], 1, True, 10.0, "hard", start=2.0)
    assert not row["temporal_success"]
    assert row["horizon_s"] == pytest.approx(52.0)


def test_judge_stream():
    row = tp.judge_stream([5.0, 11.0, None], [6.0, 10.0, 20.0], True)
    assert row["late_jobs"] == [1, 2] and not row["temporal_success"] and row["task_success"]
    row = tp.judge_stream([5.0, 10.01], [6.0, 10.0], True)
    assert row["joint_success"]


def test_allowed_misplacements():
    assert tp.allowed_misplacements(0.95, 25) == 1
    assert tp.allowed_misplacements(0.95, 12) == 0
    assert tp.allowed_misplacements(None, 12) == 0


def test_score_and_schema_by_tier():
    row = {
        "task_id": "C1",
        "difficulty": "easy",
        **tp.judge([3.0, 12.0, 25.0], 3, True, 10.0, "easy"),
    }
    assert validate_row(row, fields={}) == []
    assert score_row(row)["score"] == 100.0
    bad = {**row, "joint_success": False}
    assert any("joint_success_by_tier[difficulty]" in e for e in validate_row(bad, fields={}))
    forged = {**row, "joint_success_by_tier": {**row["joint_success_by_tier"], "hard": True}}
    assert any("hard" in e for e in validate_row(forged, fields={}))


def test_stall_window_follows_easy_pace_when_steps_are_few():
    t_o = 10.0
    assert tp.stall_window(t_o) == 5.0  # 0.5 t_o
    assert tp.stall_window(t_o, 30) == 5.0  # many steps: 1.5 * 40 / 30 = 2 < 5
    assert tp.stall_window(t_o, 2) == pytest.approx(30.0)  # 2 cards: 1.5 * 40 / 2
    # A policy at exactly easy-tier pace (one step every 4 t_o / steps) is never stopped.
    for steps in (2, 3, 6, 15):
        pace = tp.MULTIPLIER["easy"] * t_o / steps
        for k in range(steps):
            now = (k + 1) * pace - 1e-9
            last = k * pace if k else None
            assert not tp.stalled(now, last, t_o, steps=steps), (steps, k)


def test_judge_explicit_none_t_q_is_not_reached():
    row = tp.judge([1.0, 2.0], 2, True, 10.0, "easy", t_q=None)
    assert row["t_q_s"] is None and not row["temporal_success"]
    assert tp.judge([1.0, 2.0], 2, True, 10.0, "easy")["t_q_s"] == 2.0
