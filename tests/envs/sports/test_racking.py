from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.sports import c_class
from roboclock.envs.sports.pool_rack.config import config
from roboclock.envs.sports.pool_rack.task import Task
from roboclock.envs.sports.results import normalize
from roboclock.evaluation import throughput

T_O = throughput.oracle_time("sports", "C1")
T_MEDIUM = throughput.MULTIPLIER["medium"] * T_O
T_HARD = throughput.MULTIPLIER["hard"] * T_O


def scene(seed=0, difficulty="easy"):
    b = SimpleNamespace(config=config(difficulty, seed), time=0.0)
    t = Task(b)
    objects = {}
    for name, p in {**t.c["sources"], "cue_ball": [1.62, 0, 0.808575]}.items():
        objects[name] = dict(
            position=np.array(p),
            velocity=np.zeros(3),
            forces=np.zeros(4),
            support_contacts={"PoolBed": 1.67},
        )
    state = dict(
        time=0.0,
        objects=objects,
        arm_contact_peak=0.0,
        hands={
            s: dict(position=np.array([-1.0, 0.0, 1.0]), orientation=[1.0, 0.0, 0.0, 0.0])
            for s in ("left", "right")
        },
    )
    return b, t, state


def put_at_goals(t, s):
    for i, number in enumerate(t.c["order"]):
        name = f"ball{number:02d}"
        s["objects"][name]["position"] = np.array(t.c["slots"][i])


def tick(b, t, s, now):
    b.time = s["time"] = now
    t.update(s)


def test_normal_size_and_seeded_legal_identity_order():
    a, again, other = config(seed=4), config(seed=4), config(seed=5)
    assert a == again
    assert a["rack"]["sources"] != other["rack"]["sources"]
    assert a["rack"]["order"] != other["rack"]["order"]
    assert a["rack"]["table_length"] == 2 * a["rack"]["table_width"] == 2.032
    for c in (a, other):
        order = c["rack"]["order"]
        assert sorted(order) == list(range(1, 16))
        assert order[4] == 8 and order[10] < 8 < order[14]
        assert len(c["rack"]["sources"]) == 15


def test_difficulty_changes_nothing_in_the_rack():
    """score_version 2: one episode per seed; the difficulty only picks the judged tier."""
    easy, medium, hard = (config(level, 7)["rack"] for level in ("easy", "medium", "hard"))
    assert easy == medium == hard
    assert easy["horizon"] == pytest.approx(5 * T_O) == c_class.horizon("C1")


def test_goal_positions_without_physical_grasp_and_lift_do_not_count():
    b, t, s = scene()
    put_at_goals(t, s)
    tick(b, t, s, 1.0)
    tick(b, t, s, 4.0)
    assert not t.accepted and not t.result()["success"]


@pytest.mark.parametrize("fault", ["hover", "unsupported", "held", "moving"])
def test_placement_rejects_invalid_physical_state(fault):
    _, t, s = scene()
    name = f"ball{t.c['order'][0]:02d}"
    t.lifted.add(name)
    o = s["objects"][name]
    o["position"] = np.array(t.c["slots"][0])
    if fault == "hover":
        o["position"][2] += 0.02
    elif fault == "unsupported":
        o["support_contacts"] = {}
    elif fault == "held":
        o["forces"][:2] = 1.0
    else:
        o["velocity"][0] = 0.02
    assert t.placement(name, o) is None


def test_final_rack_rejects_wrong_eight_corner_groups_and_loose_balls():
    _, t, s = scene()
    put_at_goals(t, s)
    occupancy = {f"ball{n:02d}": i for i, n in enumerate(t.c["order"])}
    assert t.legality(s, occupancy)
    eight = "ball08"
    apex = f"ball{t.c['order'][0]:02d}"
    bad = dict(occupancy)
    bad[eight], bad[apex] = bad[apex], bad[eight]
    assert not t.legality(s, bad)
    stripe = f"ball{t.c['order'][14]:02d}"
    solid = next(n for n in occupancy if int(n[4:]) < 8 and occupancy[n] not in (10, 14))
    bad = dict(occupancy)
    bad[stripe], bad[solid] = bad[solid], bad[stripe]
    assert not t.legality(s, bad)
    s["objects"][apex]["position"][0] += 0.006
    assert not t.legality(s, occupancy)


def test_complete_batch_needs_settling_withdrawal_and_counts_once():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    tick(b, t, s, 1.0)
    tick(b, t, s, 1.6)
    assert len(t.accepted) == 15 and not t.done
    tick(b, t, s, 3.7)
    assert t.done and t.result()["success"] and t.termination_reason == "all_done"
    tick(b, t, s, 4.0)
    assert len([e for e in t.events if e["event"] == "placed"]) == 15
    result = t.result()
    assert result["completed_at"] == 3.7 and result["throughput"]["t_q_s"] == 3.7
    assert result["throughput"]["job_times_s"] == [1.6] * 15
    assert result["balls_per_budget_minute"] == pytest.approx(60 * 15 / (4 * T_O))


def test_late_placements_cannot_rescue_timeout():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    tick(b, t, s, t.c["horizon"] + 0.01)
    assert t.done and not t.accepted and not t.result()["success"]
    assert t.termination_reason == "horizon"


def test_touching_white_ball_invalidates_a_completed_rack():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    s["objects"]["cue_ball"]["forces"][0] = 1.0
    for time in (1.0, 1.6, 3.7):
        tick(b, t, s, time)
    assert "cue_ball_touched" in t.failures and not t.result()["success"]


def test_release_opens_each_asymmetric_jaw_without_pushing_ball():
    """Regression: a common absolute opening closed the wider contacting jaw."""
    from roboclock.envs.sports.pool_rack.oracle import Oracle

    b, t, state = scene()
    b.time = 0.2
    b.last_state = state
    b.command = np.zeros(4)
    b.rig = SimpleNamespace(grip_ids={"left": [0, 1], "right": [2, 3]})

    def move(side, target, rotation, *, grip, speed):
        b.command[b.rig.grip_ids[side]] = grip

    b.move = move
    oracle = Oracle(b, t)
    initial = np.array([0.040, 0.014])
    oracle.arms["left"].update(
        phase="release",
        index=0,
        offset=np.zeros(3),
        release_target=np.zeros(3),
        release_joints=initial.copy(),
    )
    oracle.arms["right"]["phase"] = "idle"
    oracle.act()
    assert np.all(b.command[:2] > initial)
    assert np.all(b.command[:2] <= 0.044)
    assert b.command[0] - initial[0] == pytest.approx(b.command[1] - initial[1])


def test_previously_credited_ball_displaced_later_blocks_completion():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    tick(b, t, s, 1.0)
    tick(b, t, s, 1.6)
    name = f"ball{t.c['order'][0]:02d}"
    s["objects"][name]["position"][0] += 0.06
    tick(b, t, s, 4.0)
    assert len(t.accepted) == 15
    assert len(t.current) == 14
    assert not t.done and not t.result()["success"]


def test_final_stability_clock_starts_after_both_hands_withdraw():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    s["hands"]["left"]["position"] = np.array(t.c["slots"][0]) + [0, 0, 0.02]
    for time in (1.0, 1.6, 4.0):
        tick(b, t, s, time)
    assert not t.done
    s["hands"]["left"]["position"] = np.array([-1.0, 0, 1])
    tick(b, t, s, 5.0)
    tick(b, t, s, 6.9)
    assert not t.done
    tick(b, t, s, 7.1)
    assert t.result()["success"]



@pytest.mark.parametrize("seed", [0, 1, 2])
def test_supplemental_arm_collision_invalidates_otherwise_legal_rack(seed):
    b, t, s = scene(seed)
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    s["arm_contact_peak"] = 50.0
    for now in (1.0, 1.6, 3.7):
        tick(b, t, s, now)
    assert "arm_collision" in t.failures and not t.result()["success"]


@pytest.mark.parametrize("peak,collision", [(3.7, False), (19.9, False), (20.5, True)])
def test_arm_collision_uses_the_c_class_threshold(peak, collision):
    # One control step decides (no dwell); grazes below c_class.ARM_COLLISION_N are harmless.
    b, t, s = scene(0)
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    s["arm_contact_peak"] = peak
    tick(b, t, s, 1.0)
    assert ("arm_collision" in t.failures) is collision


def racked(t, s, b, accept_at, hand_near=False):
    """All 15 balls lifted and settled at their goals; accepted 0.6 s after `accept_at`."""
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"])
    if hand_near:
        s["hands"]["left"]["position"] = np.array(t.c["slots"][0]) + [0, 0, 0.02]
    tick(b, t, s, accept_at)
    tick(b, t, s, accept_at + 0.6)
    assert len(t.accepted) == 15


@pytest.mark.parametrize(
    "finish, expected",
    [
        (T_HARD - 0.5, {"easy": True, "medium": True, "hard": True}),
        (T_HARD + 1.0, {"easy": True, "medium": True, "hard": False}),
        (T_MEDIUM + 1.0, {"easy": True, "medium": False, "hard": False}),
    ],
)
def test_one_run_is_judged_at_every_tier_from_the_rack_ready_time(finish, expected):
    """t_Q is the rack handed over (as the frozen t_o); each tier compares it with m * t_o."""
    rows = {}
    for level in ("easy", "medium", "hard"):
        b, t, s = scene(difficulty=level)
        # Keep the stall stop quiet: the last ball arrives within 0.5 t_o of the hand-over.
        racked(t, s, b, finish - 2.6)
        tick(b, t, s, finish)
        assert t.termination_reason == "all_done" and t.completed_at == finish
        rows[level] = t.result()
    for level, result in rows.items():
        judged = result["throughput"]
        assert judged["temporal_success_by_tier"] == expected
        assert result["physical_success"] and judged["task_success"]
        assert result["temporal_success"] == expected[level] == result["success"]
    # The difficulty changes nothing but the tier picked from the same verdicts.
    assert len({str(r["throughput"]["joint_success_by_tier"]) for r in rows.values()}) == 1


def test_physical_needs_all_fifteen_balls():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = set(t.c["sources"]) - {"ball08"}
    tick(b, t, s, 1.0)
    tick(b, t, s, 1.6)
    tick(b, t, s, t.c["horizon"])
    result = t.result()
    assert t.termination_reason in ("stall", "horizon") and len(t.accepted) == 14
    assert not result["physical_success"] and not result["temporal_success"]
    assert result["throughput"]["jobs_done"] == 14 and result["throughput"]["t_q_s"] is None
    assert "incomplete_rack_at_deadline" in result["failures"]


def test_physical_needs_the_rack_still_legal_at_the_end():
    b, t, s = scene()
    racked(t, s, b, 1.0, hand_near=True)
    name = f"ball{t.c['order'][0]:02d}"
    s["objects"][name]["position"][0] += 0.06  # knocked out after it was credited
    tick(b, t, s, T_MEDIUM + 0.6 * T_O)
    result = t.result()
    assert t.termination_reason == "stall"
    assert len(t.accepted) == 15 and not t.legal
    assert not result["physical_success"] and not result["success"]


def test_legal_rack_without_hand_over_is_physical_but_never_on_time():
    b, t, s = scene()
    racked(t, s, b, 1.0, hand_near=True)
    tick(b, t, s, T_MEDIUM - 0.01)
    assert not t.done  # the stall stop is armed only from T_medium
    tick(b, t, s, T_MEDIUM + 0.01)
    assert t.done and t.termination_reason == "stall"
    result = t.result()
    assert result["physical_success"] and not result["temporal_success"]
    assert result["throughput"]["temporal_success_by_tier"] == dict.fromkeys(
        throughput.TIERS, False
    )
    assert result["throughput"]["t_q_s"] is None and result["completed_at"] is None
    assert "rack_not_handed_over" in result["failures"]
    assert "incomplete_rack_at_deadline" not in result["failures"]


def test_stall_stop_after_no_new_ball_for_half_the_oracle_time():
    b, t, s = scene()
    put_at_goals(t, s)
    t.lifted = {f"ball{t.c['order'][0]:02d}"}
    accept = round(T_MEDIUM - 0.2 * T_O, 1)  # one ball accepted shortly before T_medium
    tick(b, t, s, accept - 0.6)
    tick(b, t, s, accept)
    tick(b, t, s, accept + 0.5 * T_O - 0.01)
    assert not t.done
    tick(b, t, s, accept + 0.5 * T_O + 0.001)
    assert t.done and t.termination_reason == "stall"
    assert t.result()["throughput"]["termination_reason"] == "stall"


def test_result_row_carries_the_judged_fields():
    b, t, s = scene(difficulty="hard")
    racked(t, s, b, 1.0)
    tick(b, t, s, 3.6)
    document = {**t.result(), "difficulty": "hard", "seed": 0}
    row = normalize(document)
    assert row["task_success"] and row["temporal_success"] and row["joint_success"]
    assert row["completion_time"] == row["t_q_s"] == 3.6
    assert row["termination_reason"] == "all_done" and row["score_version"] == 2
    assert row["budget_s_by_tier"]["hard"] == pytest.approx(T_HARD)
    assert row["temporal_violations"] == [] and row["physical_failures"] == []


def test_display_has_counts_but_no_countdown():
    b, t, s = scene()
    tick(b, t, s, 1.0)
    top, bottom = t.display()
    assert top == "00/15 BALLS" and bottom == "8-BALL / PREPARE THE BREAK"

