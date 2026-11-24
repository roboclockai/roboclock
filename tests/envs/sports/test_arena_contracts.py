"""Temporal violations and false physical positives for the workflow benchmark."""

from types import SimpleNamespace

import numpy as np
import pytest

from roboclock.envs.rates import PHYSICS_HZ
from roboclock.envs.sports import c_class
from roboclock.envs.sports.arena.config import config
from roboclock.envs.sports.arena.tasks import Task, cup_valid, retained
from roboclock.evaluation import throughput

T_O = throughput.oracle_time("sports", "C2")
WINDOW = throughput.stall_window(T_O, c_class.PROGRESS_STEPS["C2"])  # 94.2 s: 1.5 x T_easy / 2 cards


def body(p, forces=None):
    return dict(
        position=np.array(p, float),
        orientation=np.array([1, 0, 0, 0]),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        forces=np.array(forces if forces is not None else [0] * 11, float),
        support_force_z=0.3,
    )


def test_grasp_requires_both_fingers():
    assert not retained(body([0, 0, 1], [10, 0]))
    assert retained(body([0, 0, 1], [0.2, 0.3]))


def test_cup_pose_is_not_enough():
    p = np.array([0.3, 0.2, 0.786])
    o = body(p)
    assert cup_valid(o, p, 0.3, False)
    assert not cup_valid(o, p, 0, False)
    assert not cup_valid(o, p, 0.3, True)
    o["velocity"][2] = 0.5
    assert not cup_valid(o, p, 0.3, False)


def test_cup_tilt_invalidates_stack():
    p = np.array([0.3, 0.2, 0.786])
    o = body(p)
    o["orientation"] = np.array([0.7071, 0.7071, 0, 0])
    assert not cup_valid(o, p, 0.3, False)


def shot_task():
    cfg = config("A3")
    # These boundary fixtures use a known window, independent of the sampler.
    cfg["workflow"]["shot_start"] = 2.0
    b = SimpleNamespace(
        kind="A3",
        config=cfg,
        hoop=np.array([0.78, 0.06, 1.045]),
        hoop_radius=0.085,
        ball_radius=0.029,
    )
    return Task(b)


def state(time, p, forces, ee=(0.4, 0, 1.1), velocity=(0, 0, 0)):
    o = body(p, forces)
    o["velocity"] = np.array(velocity, float)
    return dict(time=time, objects={"ball": o}, hands={"main": {"position": np.array(ee)}})


def test_basket_after_buzzer_counts_if_release_on_time():
    t = shot_task()
    t.update(state(14, [0.4, 0, 1.1], [1, 1]))
    t.update(state(15, [0.6, 0.06, 1.2], [0, 0], velocity=(1, 0, 1)))
    t.update(state(16.1, [0.78, 0.06, 1.06], [0, 0], velocity=(0, 0, -1)))
    t.update(state(16.2, [0.78, 0.06, 1.04], [0, 0], velocity=(0, 0, -1)))
    t.update(state(16.3, [0.78, 0.06, 1.0], [0, 0], velocity=(0, 0, -1)))
    assert t.physical and t.temporal


def test_late_shot_separates_physics_from_timing():
    t = shot_task()
    t.update(state(16, [0.4, 0, 1.1], [1, 1]))
    t.update(state(16.2, [0.6, 0.06, 1.2], [0, 0], velocity=(1, 0, 1)))
    t.update(state(16.3, [0.78, 0.06, 1.06], [0, 0], velocity=(0, 0, -1)))
    t.update(state(16.4, [0.78, 0.06, 1.0], [0, 0], velocity=(0, 0, -1)))
    assert t.physical and not t.temporal


def test_ungasped_ball_never_scores():
    t = shot_task()
    t.update(state(3, [0.78, 0.06, 1.1], [0, 0]))
    t.update(state(4, [0.78, 0.06, 1.0], [0, 0], velocity=(0, 0, -1)))
    assert not t.physical


def test_timing_levels_do_not_change_scene_or_physics():
    a, b = config("C2", "easy"), config("C2", "hard")
    assert a["scene"] == b["scene"]
    assert a["sim"] == b["sim"]
    # score_version 2: the C2 episode is the same at every difficulty (one run, three tiers).
    cups = {k: v for k, v in a["workflow"].items() if k.startswith("cups") or k in (
        "preparation", "support_dwell")}
    assert cups == {k: b["workflow"][k] for k in cups}
    assert a["workflow"]["cups_horizon"] == pytest.approx(5 * T_O) == c_class.horizon("C2")


def boxing_task():
    return Task(SimpleNamespace(kind="B2", config=config("B2")))


def boxing_state(time, side=None, wrist_only=False):
    objects = {}
    hands = {
        s: {"position": np.array([0.35, 0.23 if s == "left" else -0.23, 1.06])}
        for s in ["left", "right"]
    }
    for s in ["left", "right"]:
        o = body([0.60 + (0.002 if side == s else 0), 0.23 if s == "left" else -0.23, 1.06], [0, 0])
        o["glove_forces"] = [0.0, 0.0]
        if side == s:
            index = 0 if s == "left" else 1
            o["forces"][index] = 3.0
            if not wrist_only:
                o["glove_forces"][index] = 3.0
            hands[s]["position"][0] = 0.52
        objects["pad_" + s] = o
    return dict(time=time, objects=objects, hands=hands)


def test_boxing_wrist_contact_does_not_score():
    t = boxing_task()
    t.update(boxing_state(3.2, "right", wrist_only=True))
    assert not t.hits


def test_boxing_rebound_is_one_strike():
    t = boxing_task()
    t.update(boxing_state(3.2, "right"))
    s = boxing_state(3.21)
    s["hands"]["right"]["position"][0] = 0.52
    t.update(s)
    t.update(boxing_state(3.22, "right"))
    assert len(t.hits) == 1
    assert "early_wrong_or_extra_strike" not in t.failures


def test_boxing_early_hit_is_a_failure():
    t = boxing_task()
    t.update(boxing_state(1, "left"))
    assert "early_wrong_or_extra_strike" in t.failures


def test_boxing_schedule_mixes_lengths_holds_and_same_hand_combinations():
    from roboclock.envs.sports.arena.boxing import schedule

    t = boxing_task()
    assert {len(c["hands"]) for c in t.commands} == {0, 1, 2, 3}
    assert any(c["hands"] == ["left", "left"] for c in t.commands)
    assert t.commands == schedule(t.c, 0)
    assert t.commands != schedule(t.c, 1)
    gaps = [b["time"] - a["end"] for a, b in zip(t.commands, t.commands[1:])]
    assert max(gaps) - min(gaps) > 0.1


def test_next_punch_waits_for_its_own_cue():
    from roboclock.envs.sports.arena.boxing import next_strike

    t = boxing_task()
    i, c = next((i, c) for i, c in enumerate(t.commands) if len(c["hands"]) == 2)
    t.hits = [dict(command=i)]
    assert next_strike(t, c["cues"][1] - 0.01) is None
    assert next_strike(t, c["cues"][1])[2] == c["hands"][1]


def test_boxing_hold_rejects_departure_from_guard():
    t = boxing_task()
    c = next(c for c in t.commands if not c["hands"])
    s = boxing_state(c["time"] + 0.1)
    s["hands"]["left"]["position"][0] += 0.06
    t.update(s)
    assert "moved_during_hold" in t.failures


def test_boxing_records_reaction_and_contact_latency_separately():
    t = boxing_task()
    s = boxing_state(3.15)
    s["hands"]["right"]["position"][0] += 0.03
    t.update(s)
    t.update(boxing_state(3.95, "right"))
    assert abs(t.reactions[0]["latency"] - 0.15) < 1e-8
    assert abs(t.hits[0]["latency"] - 0.95) < 1e-8


def test_boxing_display_uses_real_deadlines():
    from roboclock.envs.sports.arena.cues import boxing_display

    t = boxing_task()
    a = boxing_display(t, 3.2)
    assert a[1] == "GO RIGHT / 1" and a[2] == "HIT IN 1.4s"
    assert boxing_display(t, 4.7)[2] == "HIT WINDOW EXPIRED"
    t.update(boxing_state(3.95, "right"))
    assert boxing_display(t, 4.15)[1:3] == ("GUARD RIGHT", "RETURN IN 1.2s")


def test_release_confirmation_does_not_shift_deadline():
    t = shot_task()
    t.update(state(15.7, [0.4, 0, 1.1], [1, 1]))
    t.update(state(15.9, [0.41, 0, 1.1], [0, 0]))
    t.update(state(16.1, [0.50, 0, 1.1], [0, 0]))
    assert t.release == 15.9 and t.temporal


def test_release_then_regrasp_is_not_a_valid_shot():
    t = shot_task()
    t.update(state(4, [0.4, 0, 1.1], [1, 1]))
    t.update(state(5, [0.6, 0.06, 1.2], [0, 0]))
    t.update(state(6, [0.6, 0.06, 1.2], [1, 1]))
    assert "ball_recontact_after_release" in t.failures


def cups_task(level="easy"):
    from roboclock.envs.sports.arena.cup_patterns import layout

    targets, _ = layout(0)
    sources = {n: p + [0, 0.2, 0] for n, p in targets.items()}
    b = SimpleNamespace(
        kind="C2",
        config=config("C2", level),
        cup_targets=targets,
        cup_sources=sources,
        last_state={"objects": {"bell": body([0.29, -0.075, 0.775])}},
    )
    return Task(b)


def cups_state(t, time, press=False, one_support=False, hand_up=False, arm_contact=0.0):
    objects = {}
    from roboclock.envs.sports.arena.cup_patterns import layout

    _, supports = layout(t.card)
    for name, p in t.cup_targets.items():
        o = body(p)
        o["support_contacts"] = {support: 0.14 for support in supports[name]}
        if one_support and len(supports[name]) == 2:
            o["support_contacts"][supports[name][1]] = 0
        objects[name] = o
    bell = body([0.29, -0.075, 0.773 if press else 0.775])
    bell["forces"][2] = 1 if press else 0
    objects["bell"] = bell
    return dict(
        time=time,
        objects=objects,
        hands={"right": {"position": np.array([0.29, -0.075, 0.95 if hand_up else 0.79])}},
        arm_contact_peak=arm_contact,
    )


def two_cards(t, second, first=3.5, **first_card):
    """Card 0 rung at `first`, card 1 at `second` (both stable for the support dwell)."""
    t.update(cups_state(t, first - 0.5, **first_card))
    t.update(cups_state(t, first, press=True))
    t.update(cups_state(t, first + 0.1, hand_up=True))  # bell released; card 1 cups in place
    t.update(cups_state(t, second - 0.5))
    t.update(cups_state(t, second, press=True))
    assert t.card_times == [first, second]


def test_pyramid_requires_both_lower_supports():
    t = cups_task()
    t.update(cups_state(t, 3, one_support=True))
    t.update(cups_state(t, 3.5, press=True, one_support=True))
    assert t.card == 0 and "early_or_incorrect_bell" in t.failures


def test_bell_rebound_does_not_count_twice():
    t = cups_task()
    t.update(cups_state(t, 3))
    t.update(cups_state(t, 3.5, press=True))
    t.update(cups_state(t, 3.51))
    t.update(cups_state(t, 3.52, press=True))
    assert t.card == 1 and not t.failures


def test_one_card_of_two_is_neither_physical_nor_on_time():
    """Physical success needs every required card; one card is progress (stages) only."""
    t = cups_task()
    t.update(cups_state(t, 3))
    t.update(cups_state(t, 3.5, press=True))
    t.update(cups_state(t, 3.5 + WINDOW - 0.01))
    assert not t.done  # armed from T_medium, but the card was less than the window ago
    t.update(cups_state(t, 3.5 + WINDOW + 0.01))
    result = t.result()
    assert t.done and t.termination_reason == "stall"
    assert not result["physical_success"] and not result["temporal_success"]
    assert "incomplete_cards" in result["failures"]
    assert result["throughput"]["job_times_s"] == [3.5, None]
    assert result["completed_at"] is None


def test_two_cards_end_the_episode_after_settling():
    t = cups_task()
    two_cards(t, 20.0)
    t.update(cups_state(t, 20.0 + c_class.SETTLE_S - 0.01, hand_up=True))
    assert not t.done
    t.update(cups_state(t, 20.0 + c_class.SETTLE_S, hand_up=True))
    assert t.done and t.termination_reason == "all_done"
    result = t.result()
    assert result["physical_success"] and result["success"]
    assert result["completed_at"] == result["throughput"]["t_q_s"] == 20.0
    assert result["card_times"] == [3.5, 20.0]
    t.update(cups_state(t, 30.0, press=True))  # nothing counts after the end
    assert t.result()["card_times"] == [3.5, 20.0] and t.time == 22.0


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize(
    "elapsed, expected",
    [
        (1.25 * T_O - 0.5, {"easy": True, "medium": True, "hard": True}),
        (1.25 * T_O + 1.0, {"easy": True, "medium": True, "hard": False}),
        (5**0.5 * T_O + 1.0, {"easy": True, "medium": False, "hard": False}),
    ],
)
def test_tiers_count_the_second_card_from_the_end_of_preparation(level, elapsed, expected):
    t = cups_task(level)
    second = t.c["preparation"] + elapsed
    # After T_medium the stall stop needs each card within 0.5 t_o of the previous one.
    two_cards(t, second, first=max(3.5, second - 0.4 * T_O))
    t.update(cups_state(t, second + c_class.SETTLE_S, hand_up=True))
    result = t.result()
    judged = result["throughput"]
    assert judged["time_origin_s"] == t.c["preparation"] == 2.0
    assert judged["temporal_success_by_tier"] == expected
    assert result["physical_success"]
    assert result["temporal_success"] == expected[level] == result["success"]


def test_arm_collision_rules_out_physical_success():
    t = cups_task()
    two_cards(t, 20.0, arm_contact=c_class.ARM_COLLISION_N + 5.0)
    t.update(cups_state(t, 22.0, hand_up=True))
    result = t.result()
    assert t.termination_reason == "all_done" and result["temporal_success"]
    assert not result["physical_success"] and not result["success"]
    assert "arm_collision" in result["failures"]


def test_arm_graze_is_reported_but_not_a_collision():
    # A policy run grazed at 3.7 N for three control steps; collisions measured >= 96 N.
    t = cups_task()
    two_cards(t, 20.0, arm_contact=3.7)
    t.update(cups_state(t, 22.0, hand_up=True))
    result = t.result()
    assert result["physical_success"] and result["success"]
    assert result["arm_contact_peak"] == 3.7 and result["arm_collision_steps"] == 0


def test_stall_stop_without_cards():
    t = cups_task()
    start = t.c["preparation"]
    stop = start + max(2 * T_O, WINDOW)  # armed at T_medium, then the window from the start
    t.update(cups_state(t, stop - 0.01, one_support=True))
    assert not t.done
    t.update(cups_state(t, stop + 0.01, one_support=True))
    assert t.done and t.termination_reason == "stall"


def test_arm_collision_prevents_joint_success():
    t = shot_task()
    t.physical = t.temporal = True
    t.arm_collision_steps = 1
    assert not t.result()["success"]


def test_seed_changes_public_cue_schedule_reproducibly():
    assert config("B2", seed=1) == config("B2", seed=1)
    assert (
        config("B2", seed=0)["workflow"]["boxing_start"]
        != config("B2", seed=1)["workflow"]["boxing_start"]
    )




@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("edge", ["before_ready", "at_buzzer", "after_buzzer", "inside"])
def test_supplemental_shot_release_window_separate_from_basket(level, edge):
    c = config("A3", level, 2)
    t = Task(SimpleNamespace(kind="A3", config=c, hoop=np.array([0.78, 0.06, 1.045]),
                             hoop_radius=0.085, ball_radius=0.029))
    start, end = t.c["shot_start"], t.c["shot_start"] + t.c["shot_clock"]
    release = {"before_ready": start - 1 / PHYSICS_HZ, "at_buzzer": end,
               "after_buzzer": end + 1 / PHYSICS_HZ, "inside": start + 0.1}[edge]
    t.update(state(release - 0.1, [0.4, 0, 1.1], [1, 1]))
    t.update(state(release, [0.6, 0.06, 1.2], [0, 0], velocity=(1, 0, 1)))
    t.update(state(release + 0.1, [0.78, 0.06, 1.06], [0, 0], velocity=(0, 0, -1)))
    t.update(state(release + 0.2, [0.78, 0.06, 1.0], [0, 0], velocity=(0, 0, -1)))
    assert t.physical
    assert t.temporal == (edge == "inside")
    assert t.result()["success"] == (edge == "inside")


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_supplemental_boxing_wrong_hand_fails_each_schedule(level, seed):
    t = Task(SimpleNamespace(kind="B2", config=config("B2", level, seed)))
    c = next(c for c in t.commands if c["hands"])
    wrong = "left" if c["hands"][0] == "right" else "right"
    t.update(boxing_state(c["cues"][0] + 0.01, wrong))
    assert not t.hits and "early_wrong_or_extra_strike" in t.failures


@pytest.mark.parametrize("level", ["easy", "medium", "hard"])
def test_supplemental_cups_bell_before_preparation_never_counts(level):
    t = cups_task(level)
    start = t.c["preparation"]
    t.update(cups_state(t, start - 0.5))
    t.update(cups_state(t, start - 1 / PHYSICS_HZ, press=True))
    assert not t.card_times and not t.done
    assert "early_or_incorrect_bell" in t.failures
    assert not t.result()["success"] and not t.result()["physical_success"]


def test_supplemental_cups_horizon_ends_the_round():
    t = cups_task()
    end = t.c["preparation"] + t.c["cups_horizon"]
    t.card_times = [end - 10.0]  # a recent card keeps the stall stop quiet
    t.update(cups_state(t, end - 5.0, one_support=True))
    assert not t.done
    t.update(cups_state(t, end, one_support=True))
    assert t.done and t.termination_reason == "horizon"


def cups_state_two_hands(t, time, ringing=None, left_z=0.95, right_z=0.95):
    """`ringing`: the hand pressing the bell ("left" or "right"); hand heights in metres."""
    s = cups_state(t, time, press=ringing is not None)
    s["objects"]["bell"]["forces"][:4] = 0
    if ringing:
        s["objects"]["bell"]["forces"][0 if ringing == "left" else 2] = 1
    s["hands"] = {
        "left": {"position": np.array([0.29, -0.075, left_z])},
        "right": {"position": np.array([0.29, -0.075, right_z])},
    }
    return s


def test_bell_rung_by_the_left_hand_rearms_when_that_hand_lifts():
    """The re-arm read the right hand's height whichever hand rang (sweep v2 brief: judging
    must not depend on the arm): with the right hand up, a left-hand ring re-armed on the
    spring rebound and the same press counted again, as a bell on an unfinished card."""
    t = cups_task()
    t.update(cups_state_two_hands(t, 3))
    t.update(cups_state_two_hands(t, 3.5, ringing="left", left_z=0.79))
    assert t.card == 1 and t.bell_hand == "left"
    t.update(cups_state_two_hands(t, 3.51, left_z=0.79))  # rebound: at rest, hand still down
    t.update(cups_state_two_hands(t, 3.52, ringing="left", left_z=0.79))
    assert t.card == 1 and not t.failures
    # Once the left hand has lifted the bell is armed again, with the right hand low.
    t.update(cups_state_two_hands(t, 3.6, left_z=0.95, right_z=0.79))
    t.update(cups_state_two_hands(t, 5.0, ringing="left", left_z=0.79, right_z=0.79))
    assert t.card == 2 and t.card_times == [3.5, 5.0] and not t.failures


def test_bell_rung_by_the_right_hand_rearms_as_before():
    t = cups_task()
    t.update(cups_state_two_hands(t, 3))
    t.update(cups_state_two_hands(t, 3.5, ringing="right", right_z=0.79, left_z=0.79))
    assert t.card == 1 and t.bell_hand == "right"
    t.update(cups_state_two_hands(t, 3.51, right_z=0.79, left_z=0.95))  # left up: not the ringer
    t.update(cups_state_two_hands(t, 3.52, ringing="right", right_z=0.79))
    assert t.card == 1 and not t.failures
    t.update(cups_state_two_hands(t, 3.6, right_z=0.95, left_z=0.79))
    t.update(cups_state_two_hands(t, 5.0, ringing="right", right_z=0.79))
    assert t.card == 2 and not t.failures


def test_dunk_and_low_release_use_the_stated_distances():
    from roboclock.envs.sports.arena import tasks as arena

    assert (arena.NO_DUNK_RADIUS, arena.RELEASE_MIN_HEIGHT, arena.CUP_TOLERANCE) == (0.12, 0.90, 0.009)
    hoop = np.array([0.78, 0.06, 1.045])
    for offset, dunk in ((0.119, True), (0.121, False)):
        t = shot_task()
        t.update(state(4, [0.4, 0, 1.1], [1, 1]))
        p = hoop + [-offset, 0, 0.15]
        t.update(state(5, p, [0, 0], ee=p))
        t.update(state(5.1, p, [0, 0], ee=p - [0.1, 0, 0]))
        assert t.release == 5 and ("release_inside_no_dunk_region" in t.failures) is dunk
    # A ball let go below the release height is no release.
    t = shot_task()
    t.update(state(4, [0.4, 0, 0.85], [1, 1]))
    t.update(state(5, [0.4, 0, 0.85], [0, 0], ee=(0.4, 0, 0.85)))
    t.update(state(5.1, [0.4, 0, 0.85], [0, 0], ee=(0.3, 0, 0.85)))
    assert t.release is None


def test_boxing_guard_distances_are_the_stated_ones():
    from roboclock.envs.sports.arena import tasks as arena

    assert list(arena.guard("left")) == [0.35, 0.23, 1.06]
    assert list(arena.guard("right")) == [0.35, -0.23, 1.06]
    assert (arena.GUARD_RETURN, arena.GUARD_HOLD, arena.GUARD_OTHER) == (0.025, 0.035, 0.05)
