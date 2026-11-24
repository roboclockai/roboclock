"""Adversarial temporal and physical-predicate tests without a simulator."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports.ball_service.config import (
    HARD_INTERVAL,
    HARD_PREPARATION,
    ORACLE_ACCEPTED_S,
    TIER_FACTOR,
    config,
    tier_schedule,
)
from roboclock.envs.sports.ball_service.task import Task, cell_at
from roboclock.envs.sports.c_class import SETTLE_S
from roboclock.simulators.isaac.embodiments.config import configure


def fixture():
    c = config()
    targets = []
    b = SimpleNamespace(
        config=c,
        time=0,
        gate_drives={
            i: SimpleNamespace(GetTargetPositionAttr=lambda: SimpleNamespace(Set=targets.append))
            for i in range(4)
        },
    )
    s = dict(
        time=0,
        hands={
            side: dict(position=np.array([0.3, 0.3 if side == "left" else -0.3, 1.1]))
            for side in ("left", "right")
        },
        arm_contact_peak=0,
        objects={
            name: dict(
                position=np.array(p), velocity=np.zeros(3), forces=np.zeros(8), support_contacts={}
            )
            for name, p in c["service"]["sources"].items()
        },
    )
    b.last_state = s
    return b, Task(b), s, targets


def tick(b, t, s, time):
    b.time = s["time"] = time
    t.update(s)


def stage_acceptance(b, t, s, name="ball0"):
    c = b.config["service"]
    o = s["objects"][name]
    t.grasped[name] = "left"
    o["position"] = np.array(c["cells"][0])
    o["support_contacts"] = {"gate0": 0.56}
    t.entry.add((name, 0))


def test_seed_repeatability_and_timing_only_difficulty():
    a = config("easy", 8)
    b = config("hard", 8)
    assert a == config("easy", 8)
    assert a["service"]["sources"] == b["service"]["sources"]
    assert a["service"]["cells"] == b["service"]["cells"]
    assert a["service"]["firing_interval"] > b["service"]["firing_interval"]
    assert config("easy", 9)["service"]["sources"] != a["service"]["sources"]
    assert a["service"]["demand_count"] > a["service"]["capacity"]


def test_legacy_configuration_cannot_silently_migrate():
    c = {"workflow": {}, "keep": 1}
    with pytest.raises(ValueError, match="fixed-service"):
        configure(c, "C3")
    assert c == {"workflow": {}, "keep": 1}


def test_full_ball_envelope_required():
    c = config()["service"]
    p = np.array(c["cells"][0])
    assert cell_at(p, c) == 0
    p[0] += 0.03
    assert cell_at(p, c) is None


@pytest.mark.parametrize("defect", ["hover", "held", "side_entry", "unsupported", "ungrasped"])
def test_invalid_placement_is_not_accepted(defect):
    b, t, s, _ = fixture()
    stage_acceptance(b, t, s)
    o = s["objects"]["ball0"]
    if defect == "hover":
        o["position"][2] = 1.02
    elif defect == "held":
        o["forces"][:2] = 1
    elif defect == "side_entry":
        t.entry.clear()
    elif defect == "unsupported":
        o["support_contacts"] = {}
    else:
        t.grasped.clear()
    tick(b, t, s, 1)
    tick(b, t, s, 1.4)
    assert not t.accepted


def test_supported_released_ball_counts_once_and_dwell_resets():
    b, t, s, _ = fixture()
    stage_acceptance(b, t, s)
    tick(b, t, s, 1)
    s["objects"]["ball0"]["velocity"][0] = 0.2
    tick(b, t, s, 1.15)
    s["objects"]["ball0"]["velocity"][:] = 0
    tick(b, t, s, 1.2)
    tick(b, t, s, 1.3)
    assert not t.accepted
    tick(b, t, s, 1.45)
    tick(b, t, s, 1.8)
    assert t.accepted == {"ball0": 0}
    assert len([e for e in t.events if e["event"] == "accepted"]) == 1


def test_empty_deadline_is_late_even_if_ball_is_added_later():
    """Starvation makes that demand late; the launcher still fires the next stocked ball."""
    b, t, s, targets = fixture()
    due = b.config["service"]["preparation"]
    tick(b, t, s, due)
    assert "starvation" in t.failures and not targets
    stage_acceptance(b, t, s)
    tick(b, t, s, due + 0.1)
    tick(b, t, s, due + 0.4)
    assert "starvation" in t.failures
    assert targets == [-0.13] and t.jobs[0]["ball"] == "ball0"
    assert t.jobs[0]["job_time"] == pytest.approx(due + 0.4)  # stocked after the deadline
    s["objects"]["ball0"]["position"][2] = 0.7
    tick(b, t, s, due + 0.6)
    assert t.fired == {"ball0"}
    judged = t.result()["throughput"]
    assert judged["late_jobs"] == [0, 1, 2, 3, 4, 5] and not judged["temporal_success"]
    assert judged["job_times_s"][0] == pytest.approx(due + 0.4)
    passage = [e for e in t.events if e["event"] == "outlet_passage"]
    assert passage[0]["late"] and passage[0]["demand"] == 0


def test_gate_command_is_not_discharge_and_failed_exit_is_reported():
    b, t, s, targets = fixture()
    stage_acceptance(b, t, s)
    tick(b, t, s, 1)
    tick(b, t, s, 1.3)
    due = b.config["service"]["preparation"]
    tick(b, t, s, due)
    assert targets == [-0.13] and not t.fired
    tick(b, t, s, due + 1.5)
    assert "unverified_discharge" in t.failures


def test_actual_outlet_passage_counts_once():
    b, t, s, _ = fixture()
    stage_acceptance(b, t, s)
    tick(b, t, s, 1)
    tick(b, t, s, 1.3)
    due = b.config["service"]["preparation"]
    tick(b, t, s, due)
    s["objects"]["ball0"]["position"][2] = 0.7
    tick(b, t, s, due + 0.2)
    tick(b, t, s, due + 0.3)
    assert t.fired == {"ball0"}
    assert len([e for e in t.events if e["event"] == "outlet_passage"]) == 1


def test_physical_stack_over_capacity_is_detected_before_acceptance():
    b, t, s, _ = fixture()
    c = b.config["service"]
    for i, name in enumerate(c["sources"]):
        s["objects"][name]["position"] = np.array(c["cells"][i % 4]) + [0, 0, 0.03]
    tick(b, t, s, 1)
    assert "overflow" in t.failures and not t.accepted



@pytest.mark.parametrize("fault", ["removed", "regrasped"])
def test_supplemental_accepted_ball_no_longer_available_causes_starvation(fault):
    b, t, s, targets = fixture()
    stage_acceptance(b, t, s)
    tick(b, t, s, 1.0)
    tick(b, t, s, 1.3)
    assert t.accepted == {"ball0": 0}
    if fault == "removed":
        s["objects"]["ball0"]["position"] = np.array(t.c["sources"]["ball0"])
    else:
        s["objects"]["ball0"]["forces"][:2] = 1.0
    tick(b, t, s, t.c["preparation"])
    assert "starvation" in t.failures
    assert not targets and not t.fired


def stock(b, t, s, name, now, cell=0):
    """A grasped ball released into `cell` from above and accepted after the support dwell."""
    c = b.config["service"]
    o = s["objects"][name]
    t.grasped[name] = "left"
    o["position"] = np.array(c["cells"][cell])
    o["support_contacts"] = {f"gate{cell}": 0.56}
    t.entry.add((name, cell))
    tick(b, t, s, now)
    tick(b, t, s, now + 0.25)
    assert name in t.accepted
    return now + 0.25


def serve_all(b, t, s, late=()):
    """Stock one ball 3 s before each demand (after it for demands in `late`) and fire it."""
    c = b.config["service"]
    stocked = []
    for i, name in enumerate(c["sources"]):
        due = c["preparation"] + i * c["firing_interval"]
        if i in late:
            tick(b, t, s, due)
            stocked.append(stock(b, t, s, name, due + 0.5))
        else:
            stocked.append(stock(b, t, s, name, due - 3.0))
            tick(b, t, s, due)
        s["objects"][name]["position"][2] = 0.7  # through the gate
        tick(b, t, s, max(due, stocked[-1]) + 0.2)
        assert name in t.fired
        tick(b, t, s, max(due, stocked[-1]) + 1.5)  # gate closed again
    return stocked


def test_horizon_is_a_quarter_past_the_last_demand_and_schedule_is_per_difficulty():
    for level in ("easy", "medium", "hard"):
        c = config(level, 3)["service"]
        last = c["preparation"] + (c["demand_count"] - 1) * c["firing_interval"]
        assert c["horizon"] == pytest.approx(1.25 * last)


def deadlines(c):
    return [c["preparation"] + k * c["firing_interval"] for k in range(c["demand_count"])]


def test_every_tier_is_the_reference_schedule_stretched_by_its_factor():
    """Every demand time is m x its reference time, same demand count: hard 1.25 (20 % headroom
    over the reference oracle), easy capped at a 12 s firing interval, medium their geometric mean."""
    hard = config("hard", 3)["service"]
    assert (hard["preparation"], hard["firing_interval"]) == pytest.approx(
        (1.25 * HARD_PREPARATION, 1.25 * HARD_INTERVAL)
    )
    assert config("easy", 3)["service"]["firing_interval"] == pytest.approx(12.0)
    assert TIER_FACTOR["medium"] ** 2 == pytest.approx(TIER_FACTOR["easy"] * TIER_FACTOR["hard"])
    reference = [HARD_PREPARATION + k * HARD_INTERVAL for k in range(6)]
    for level, m in TIER_FACTOR.items():
        c = config(level, 3)["service"]
        assert c["demand_count"] == hard["demand_count"] == 6
        assert c["capacity"] == hard["capacity"] == 4
        assert deadlines(c) == pytest.approx([m * d for d in reference])
        assert c["horizon"] == pytest.approx(1.25 * m * reference[-1])


def test_schedule_does_not_depend_on_the_seed():
    """No timing jitter: a seed changes the layout only, so hard is tight on every seed."""
    for level in ("easy", "medium", "hard"):
        a, b = config(level, 0)["service"], config(level, 51)["service"]
        assert deadlines(a) == deadlines(b) and a["horizon"] == b["horizon"]
        assert a["sources"] != b["sources"]


def test_hard_schedule_is_the_tightest_the_slowest_oracle_meets():
    """Each demand is due no earlier than the slowest oracle's ball for it, with no margin."""
    due = [HARD_PREPARATION + k * HARD_INTERVAL for k in range(len(ORACLE_ACCEPTED_S))]
    assert all(d >= a for d, a in zip(due, ORACLE_ACCEPTED_S))
    # Rounded up to 0.01 s: 0.01 s earlier, the first demand or some later one beats the oracle.
    assert HARD_PREPARATION - 0.01 < ORACLE_ACCEPTED_S[0]
    assert any(
        HARD_PREPARATION + k * (HARD_INTERVAL - 0.01) < a for k, a in enumerate(ORACLE_ACCEPTED_S)
    )


def test_schedule_override_is_recorded_and_checked():
    eager = dict(preparation=0.0, firing_interval=0.01, horizon=60.0)
    c = config("hard", 0, schedule=eager)["service"]
    assert deadlines(c)[-1] == pytest.approx(0.05) and c["horizon"] == 60.0
    assert c["schedule_override"] == eager
    assert "schedule_override" not in config("hard", 0)["service"]
    stretched = config("hard", 0, schedule=dict(firing_interval=10.0))["service"]
    assert stretched["horizon"] == pytest.approx(1.25 * (1.25 * HARD_PREPARATION + 50.0))
    with pytest.raises(ValueError, match="unknown keys"):
        config("hard", 0, schedule=dict(capacity=6))


def test_eager_calibration_config_reaches_the_task():
    from roboclock.envs.sports.config import DEFAULT_CONFIG, load_config
    from roboclock.envs.sports.tasks import make_task

    task = make_task("C3", "hard", 0, load_config(DEFAULT_CONFIG.parent / "c3_eager.yaml"))
    assert task.spec.max_time == 60.0 and max(task.spec.timing["deadlines"]) < 0.1
    assert make_task("C3", "hard", 0).spec.timing["deadlines"][0] == pytest.approx(
        1.25 * HARD_PREPARATION
    )


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_instruction_states_the_tier_schedule(level):
    from roboclock.envs.sports.tasks import make_task

    preparation, interval = tier_schedule(level)
    instruction = make_task("C3", level, 0).spec.instruction
    assert f"6 timed demands (every {interval:g} s from {preparation:g} s)" in instruction
    assert "Never exceed 4 balls" in instruction


def test_on_time_service_is_judged_per_demand_from_the_stocking_time():
    b, t, s, targets = fixture()
    c = b.config["service"]
    stocked = serve_all(b, t, s)
    result = t.result()
    judged = result["throughput"]
    assert result["physical_success"] and result["temporal_success"] and result["success"]
    assert judged["job_times_s"] == pytest.approx(stocked)
    assert judged["job_deadlines_s"] == pytest.approx(
        [c["preparation"] + i * c["firing_interval"] for i in range(6)]
    )
    assert result["accepted_times"] == pytest.approx(dict(zip(c["sources"], stocked)))
    assert len(result["served_times"]) == 6 and not t.failures


def settle(b, t, s):
    tick(b, t, s, t.last_served + SETTLE_S - 0.01)
    assert not t.done
    tick(b, t, s, t.last_served + SETTLE_S + 0.001)


def test_all_served_ends_the_episode_after_settling():
    b, t, s, _ = fixture()
    serve_all(b, t, s)
    settle(b, t, s)
    assert t.done and t.termination_reason == "all_done"
    assert t.result()["completed_at"] == t.last_served
    assert t.result()["throughput"]["termination_reason"] == "all_done"


def test_late_ball_keeps_physical_success_but_fails_its_deadline():
    b, t, s, _ = fixture()
    serve_all(b, t, s, late={2})
    settle(b, t, s)
    result = t.result()
    judged = result["throughput"]
    assert "starvation" in result["failures"]
    assert result["physical_success"] and not result["temporal_success"]
    assert judged["late_jobs"] == [2] and t.termination_reason == "all_done"


def test_physical_needs_every_demand_served():
    b, t, s, _ = fixture()
    c = b.config["service"]
    stock(b, t, s, "ball0", 1.0)
    tick(b, t, s, c["preparation"])
    s["objects"]["ball0"]["position"][2] = 0.7
    tick(b, t, s, c["preparation"] + 0.2)
    tick(b, t, s, c["horizon"])
    result = t.result()
    assert t.done and t.termination_reason == "horizon"
    assert result["served"] == 1 and not result["physical_success"]
    assert result["throughput"]["job_times_s"][1:] == [None] * 5


@pytest.mark.parametrize("failure", ["dropped_ball5", "arm_collision", "occupied_cell", "overflow"])
def test_physical_failures_rule_out_physical_success(failure):
    b, t, s, _ = fixture()
    serve_all(b, t, s)
    t.failures.append(failure)
    result = t.result()
    assert result["temporal_success"] and not result["physical_success"]
    assert not result["success"]


def test_gate_retry_after_unverified_discharge_still_serves_the_demand():
    b, t, s, targets = fixture()
    c = b.config["service"]
    stock(b, t, s, "ball0", 1.0)
    due = c["preparation"]
    tick(b, t, s, due)
    tick(b, t, s, due + c["gate_timeout"] + 0.001)  # never left: gate closes, demand owed
    assert "unverified_discharge" in t.failures and targets == [-0.13, 0]
    tick(b, t, s, due + c["gate_timeout"] + 0.01)  # retried with the same ball, late
    assert targets[-1] == -0.13 and t.jobs[0]["job_time"] > due
    s["objects"]["ball0"]["position"][2] = 0.7
    tick(b, t, s, due + c["gate_timeout"] + 0.2)
    assert t.fired == {"ball0"}

