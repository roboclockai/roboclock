"""Stage scores (roboclock.evaluation.score): RoboDojo-style partial credit per episode."""

import json

import pytest

from roboclock.envs.registry import environment_names, get_environment, import_symbol
from roboclock.evaluation import score
from roboclock.evaluation.score import (
    JOINT,
    PHYSICAL,
    Stage,
    check_stages,
    given,
    happened,
    score_row,
)
from roboclock.tasks.registry import task_registry

STAGES = (
    Stage("grasped", 20, given("grasp_time")),
    Stage("lifted", 40, happened("lifted")),
    Stage("moved", 60, given("move_time")),
)


def row(**fields):
    return {"task_success": False, "joint_success": False, **fields}


def test_stages_count_in_order_and_stop_at_the_first_missing_one():
    assert score_row(row(), STAGES) == {"score": 0.0, "score_stages": []}
    first = row(grasp_time=0.0)  # a milestone at t = 0 counts
    assert score_row(first, STAGES) == {"score": 20, "score_stages": ["grasped"]}
    # "moved" without "lifted" does not count: stages are transitions, as RoboDojo's.
    skipped = row(grasp_time=1.0, move_time=3.0)
    assert score_row(skipped, STAGES)["score"] == 20
    events = row(grasp_time=1.0, events=[{"time": 2.0, "type": "lifted"}], move_time=3.0)
    assert score_row(events, STAGES) == {
        "score": 60,
        "score_stages": ["grasped", "lifted", "moved"],
    }


def test_completion_scores_80_and_joint_success_100_whatever_the_stages():
    physical = row(task_success=True)
    assert score_row(physical, STAGES) == {
        "score": PHYSICAL,
        "score_stages": ["physically_complete"],
    }
    joint = row(task_success=True, joint_success=True, grasp_time=1.0)
    result = score_row(joint, STAGES)
    assert result["score"] == JOINT
    assert result["score_stages"] == ["grasped", "physically_complete", "completed_on_time"]
    assert score_row(row(task_success=True, joint_success=True))["score"] == JOINT


@pytest.mark.parametrize(
    "stages,message",
    [
        ((Stage("a", 40, bool), Stage("b", 40, bool)), "increase"),
        ((Stage("a", 80, bool),), "increase"),
        ((Stage("a", 0, bool),), "increase"),
        ((Stage("a", 10, bool), Stage("a", 20, bool)), "repeated"),
        ((Stage("physically_complete", 10, bool),), "reserved"),
    ],
)
def test_stage_tables_are_checked(stages, message):
    with pytest.raises(ValueError, match=message):
        check_stages(stages, "x")


def test_events_are_read_from_every_domain_layout():
    sample = {
        "events": [{"type": "a"}, {"kind": "b"}, {"event": "c"}, "ignored"],
        "event_times": {"d": 1.0},
        "scene_events": {"e": 2.0},
        "measured_events": {"f": 3.0},
    }
    assert score.events(sample) == set("abcdef")
    assert score.events({"events": None, "scene_events": None}) == set()


def registered_tables():
    for name in environment_names():
        reference = get_environment(name).stages
        if reference is not None:
            yield name, import_symbol(reference)


@pytest.mark.parametrize("environment,table", list(registered_tables()))
def test_registered_stage_tables_name_real_tasks_and_are_valid(environment, table):
    tasks = set(task_registry(environment))
    assert set(table) <= tasks, f"{environment}: stages for unknown tasks {set(table) - tasks}"
    for task_id in table:
        stages = score.task_stages(environment, task_id, row())
        assert all(isinstance(stage, Stage) for stage in stages)


# Benchmark tasks with no measured physical step before completion: they score from the
# success flags alone (0, 80 or 100).
UNLISTED = {
    "lab": {"B4"},  # test-tube rescue: only the grasp precedes the lift that completes it
    "hospital": set(),
    "household": {"B1"},  # faucet: only first lever contact precedes the closed faucet
    "industry": set(),
    "sports": set(),
}


@pytest.mark.parametrize("environment", sorted(UNLISTED))
def test_every_benchmark_task_lists_stages_unless_it_has_none_to_measure(environment):
    table = import_symbol(get_environment(environment).stages)
    assert set(task_registry(environment)) - set(table) == UNLISTED[environment]


def boxing(required_per_command, hits, **fields):
    schedule = [{"hands": ["left"] * n} for n in required_per_command]
    return row(
        task_id="B2",
        native_result={"boxing_schedule": schedule, "hits": [{"hand": "left"}] * hits},
        **fields,
    )


def test_boxing_scores_each_strike_below_completion():
    assert score.score_fields("sports", boxing([2, 3, 4], 0))["score"] == 0
    assert score.score_fields("sports", boxing([2, 3, 4], 3))["score"] == 24.0  # 80 * 3 / 10
    every = score.score_fields("sports", boxing([2, 3, 4], 9))
    assert every["score"] == 72.0 and every["score_stages"][-1] == "strike_9"
    assert (
        score.score_fields("sports", boxing([2, 3, 4], 9, task_success=True))["score"] == PHYSICAL
    )
    assert score.task_stages("sports", "B2", row()) == ()  # no native result, no stages


def test_count_stages_score_k_of_n_below_completion():
    # Industry C2: three cartons; credits are monitor events, in any order.
    shelved = {"a_correct_slot_credit_s": 4.0, "b_correct_slot_credit_s": 9.0, "x_s": 1.0}
    assert score.score_fields("industry", row(task_id="C2", measured_events=shelved))["score"] == 40
    # Sports B5 (taiko): eleven notes; only stick_hit events count.
    hits = [{"kind": "stick_hit"}] * 3 + [{"kind": "control_actuated"}] * 5
    assert score.score_fields("sports", row(task_id="B5", events=hits))["score"] == 20.0
    # Sports C2: the 12 cups of the two required cards (k of 12 scores 80 k / 13); cups of
    # later cards do not add stages.
    cups = [{"kind": "cup_placed", "card": 0}] * 2 + [{"kind": "cup_placed", "card": 1}] * 6
    cups += [{"kind": "cup_placed", "card": 2}] * 3
    assert score.score_fields("sports", row(task_id="C2", events=cups))["score"] == 49.2
    # Lab C1: n samples from the episode spec.
    samples = row(task_id="C1", episode_spec={"num_samples": 3}, correct_operations=2)
    assert score.score_fields("lab", samples)["score"] == 40.0


def test_unlisted_tasks_score_from_the_success_flags():
    assert score.score_fields("lab", row(task_id="B4")) == {"score": 0.0, "score_stages": []}
    assert score.score_fields("lab", row(task_id="B4", task_success=True))["score"] == PHYSICAL
    assert score.score_fields("hospital", row(task_id="any", task_success=True))["score"] == PHYSICAL


def test_command_line_scores_episode_files(tmp_path, capsys):
    rows = [
        row(environment="household", task_id="A2", seed=0, events=[{"type": "glass_picked_up"}]),
        row(environment="household", task_id="A2", seed=1),
    ]
    (tmp_path / "run").mkdir()
    (tmp_path / "run/episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert score.main([str(tmp_path), "--episodes"]) == 0
    out = capsys.readouterr().out
    assert "household/A2 seed=0 score=20 stages=glass_picked_up" in out
    assert "household/A2" in out and "10.0" in out and "glass_picked_up 1" in out
