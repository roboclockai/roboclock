"""Alternating physical clock handoffs, distinct from the existing single-turn task."""

import numpy as np
import pytest

from roboclock.envs.chess.config import load_config
from roboclock.envs.sports.chess.task import DuelTask


def task(moves=None):
    c = load_config()
    c["episode"].update(history=["e2e4"], move="g8f6")
    c["duel"] = dict(moves=moves or ["g8f6", "g1f3", "b8c6", "b1c3"], max_time=140)
    t = DuelTask("easy", 0, c)
    t.piece_envelopes = {n: {"size": [0.028, 0.028, 0.045]} for n in t.initial_symbols}
    return t


def state(t, placed=True, press=False):
    objects = {
        n: dict(
            position=t.position((t.expected if placed else t.locations)[n]),
            orientation=np.array([1.0, 0, 0, 0]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            supported=True,
            robot_contact=False,
            grasped=False,
        )
        for n in t.initial_symbols
    }
    return dict(objects=objects, robot_pressed=press, opponent_robot_contact=False)


def observe(t, s, duration=0.01):
    t.observe_duel({t.actor: s}, duration)


def finish(t):
    t.grasped.update(op.piece for op in t.operations)
    for _ in range(22):
        observe(t, state(t))
    for _ in range(3):
        observe(t, state(t, press=True))


def advance(t):
    assert t.turn_complete
    t.turn_index += 1
    t.begin_turn()


def test_four_legal_moves_use_actual_previous_press():
    t = task()
    for i in range(4):
        if i:
            press = t.last_press
            t.time += 2  # Retraction consumes the next player's clock.
            advance(t)
            assert t.opponent_time == press
        finish(t)
    assert [r["actor"] for r in t.turn_results] == ["black", "white", "black", "white"]
    assert t.result()["success"]
    assert [m.uci() for m in t.board.move_stack] == ["e2e4", *t.sequence]
    assert t.turn_results[1]["latency"] > 2


def test_first_opponent_move_is_setup_but_reply_has_deadline():
    t = task()
    t.time = 31
    finish(t)
    assert t.turn_results[0]["success"]
    advance(t)
    t.time = t.last_press + 31
    finish(t)
    assert t.turn_results[1]["physical_success"]
    assert not t.turn_results[1]["temporal_success"]


@pytest.mark.parametrize("invalid", ["missing_piece", "unsupported", "held", "wrong_clock"])
def test_invalid_physical_state_cannot_pass(invalid):
    t = task()
    t.grasped.update(op.piece for op in t.operations)
    s = state(t)
    if invalid == "missing_piece":
        del s["objects"]["piece_a1"]
    elif invalid == "wrong_clock":
        s["opponent_robot_contact"] = True
    else:
        s["objects"]["piece_g8"]["supported" if invalid == "unsupported" else "robot_contact"] = (
            invalid == "held"
        )
    for _ in range(25):
        observe(t, s)
    s["robot_pressed"] = True
    for _ in range(3):
        observe(t, s)
    assert not t.turn_results[0]["success"]


def test_early_press_is_latched_even_after_valid_placement():
    t = task()
    observe(t, state(t, placed=False, press=True))
    observe(t, state(t, placed=False))
    finish(t)
    assert t.turn_results[0]["physical_success"]
    assert "clock_before_placement" in t.turn_results[0]["violations"]
    assert not t.turn_results[0]["success"]


def test_previously_moved_piece_keeps_its_physical_identity():
    t = task(["g8f6", "g1f3", "f6g8", "f3g1"])
    finish(t)
    advance(t)
    finish(t)
    advance(t)
    assert t.operations[0].piece == "piece_g8"
    assert t.operations[0].destination == "g8"


def test_illegal_reply_is_rejected():
    t = task(["g8f6", "g1g3"])
    finish(t)
    with pytest.raises(ValueError):
        advance(t)


@pytest.mark.parametrize("fault", ["displaced", "held", "unsupported"])
def test_supplemental_placement_lost_after_clock_invalidates_episode(fault):
    t = task()
    finish(t)
    s = state(t)
    piece = t.operations[0].piece
    if fault == "displaced":
        s["objects"][piece]["position"][0] += 0.03
    elif fault == "held":
        s["objects"][piece]["robot_contact"] = True
    else:
        s["objects"][piece]["supported"] = False
    observe(t, s)
    assert "placement_lost_after_clock" in t.turn_violations
    assert not t.result()["success"]


def test_black_is_not_late_for_the_time_white_takes_to_hand_over():
    """Sweep v2, seed 0: WHITE pressed in time, needed 76 s to hand the turn over, and
    BLACK's reply was booked as `2:deadline_missed`. BLACK cannot move before the handoff and
    the handoff has no time limit, so BLACK's deadline counts from its turn start."""
    t = task()
    finish(t)  # BLACK
    advance(t)
    finish(t)  # WHITE, in time
    white_press = t.last_press
    t.time += 76  # WHITE's arm takes this long to return to its waiting pose
    handoff = t.time
    advance(t)
    assert t.actor == "black" and t.opponent_time == white_press
    assert t.clock_start == handoff
    assert t.event_log[-1]["event"] == "turn_start"
    assert t.event_log[-1]["deadline"] == pytest.approx(handoff + t.spec.deadline)
    finish(t)
    black = t.turn_results[2]
    assert black["temporal_success"] and black["success"] and not black["violations"]
    assert black["latency"] > 76 > t.spec.deadline  # still the time since WHITE's press
    assert black["clock_start"] == handoff
    assert not t.violations
    advance(t)
    finish(t)
    assert t.result()["success"] and t.result()["temporal_success"]


def test_black_still_misses_its_own_deadline():
    t = task()
    finish(t)
    advance(t)
    finish(t)
    t.time += 2
    advance(t)
    t.time += t.spec.deadline + 1  # BLACK itself takes too long after getting the turn
    finish(t)
    assert not t.turn_results[2]["temporal_success"]
    assert "2:deadline_missed" in t.violations


def test_white_deadline_still_counts_from_blacks_press():
    """WHITE's instruction says "within N s of BLACK's press": BLACK's retraction is on
    WHITE's clock, as before."""
    t = task()
    finish(t)
    black_press = t.last_press
    t.time += 2  # BLACK retracts
    advance(t)
    assert t.actor == "white" and t.clock_start == black_press
    t.time = black_press + t.spec.deadline - 0.1
    finish(t)  # placed, then pressed 0.12 s after the deadline measured from BLACK's press
    assert t.turn_results[1]["latency"] > t.spec.deadline
    assert not t.turn_results[1]["temporal_success"]
    assert "1:deadline_missed" in t.violations
