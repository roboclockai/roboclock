"""Rules and temporal boundaries without a simulator; runtime tested separately."""

import chess
import numpy as np
import pytest

from roboclock.envs.chess.assets import ROOT, read_glb
from roboclock.envs.chess.config import load_config
from roboclock.envs.chess.rules import game_board, physical_plan
from roboclock.envs.chess.task import make_task


def task():
    return make_task("CH1", config=load_config())


def state(t, *, placed=False, press=False, opponent=True):
    objects = {}
    for name in t.initial_symbols:
        objects[name] = {
            "position": t.position(t.expected[name] if placed else name[6:]),
            "orientation": np.array([1.0, 0, 0, 0]),
            "velocity": np.zeros(3),
            "angular_velocity": np.zeros(3),
            "supported": True,
            "robot_contact": False,
            "grasped": False,
        }
    return {"objects": objects, "robot_pressed": press, "opponent_pressed": opponent}


def settle(t):
    t.observe(state(t), 0.01)
    t.grasped.update(op.piece for op in t.operations)
    for _ in range(21):
        t.observe(state(t, placed=True), 0.01)


def press(t):
    for _ in range(3):
        t.observe(state(t, placed=True, press=True), 0.01)


def test_legal_move_then_clock_commits_and_finishes():
    t = task()
    settle(t)
    press(t)
    assert t.joint_success and t.terminated
    assert t.board.peek().uci() == "g1f3"
    assert t.placement_time < t.press_time
    assert t.metrics()["temporal_success"]


def test_clock_before_placement_is_latched_failure():
    t = task()
    t.observe(state(t, press=True), 0.01)
    t.observe(state(t), 0.01)
    settle(t)
    press(t)
    assert t.physical_success and not t.joint_success
    assert "clock_before_placement" in t.violations


def test_wrong_square_and_unreleased_piece_never_pass():
    for kind in ("wrong_square", "held", "unsupported", "moving", "tilted", "no_grasp"):
        t = task()
        t.observe(state(t), 0.01)
        if kind != "no_grasp":
            t.grasped.update(op.piece for op in t.operations)
        s = state(t, placed=True)
        obj = s["objects"]["piece_g1"]
        if kind == "wrong_square":
            obj["position"] = t.position("h3")
        if kind == "held":
            obj["robot_contact"] = True
        if kind == "unsupported":
            obj["supported"] = False
        if kind == "moving":
            obj["velocity"][0] = 1.0
        if kind == "tilted":
            obj["orientation"] = np.array([0.707, 0.707, 0, 0])
        for _ in range(30):
            t.observe(s, 0.01)
        assert not t.placement_valid, kind


def test_placement_must_still_be_valid_at_button_press():
    t = task()
    settle(t)
    s = state(t, placed=True, press=True)
    s["objects"]["piece_g1"]["position"] = t.position("g4")
    for _ in range(3):
        t.observe(s, 0.01)
    assert not t.joint_success


def test_late_move_retains_physical_success():
    t = task()
    settle(t)
    t.time = t.opponent_time + t.spec.deadline + 0.1
    press(t)
    assert t.physical_success and not t.joint_success
    assert "deadline_missed" in t.violations


def test_exact_deadline_uses_press_onset_not_debounce_completion():
    t = task()
    settle(t)
    t.time = t.opponent_time + t.spec.deadline - 0.01
    press(t)
    assert t.joint_success


def test_missing_piece_cannot_pass():
    t = task()
    settle(t)
    s = state(t, placed=True, press=True)
    del s["objects"]["piece_a1"]
    for _ in range(3):
        t.observe(s, 0.01)
    assert not t.joint_success


def test_before_opponent_press_is_failure():
    t = task()
    s = state(t, placed=True, opponent=False)
    t.observe(s, 0.01)
    assert "move_before_opponent_press" in t.violations


def test_wrong_clock_contact_is_a_latched_violation():
    t = task()
    settle(t)
    s = state(t, placed=True)
    s["opponent_robot_contact"] = True
    t.observe(s, 0.01)
    press(t)
    assert not t.joint_success
    assert "wrong_clock_button" in t.violations


@pytest.mark.parametrize(
    "fen,uci,kinds",
    [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", ["move", "castle_rook"]),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8", ["move", "castle_rook"]),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "e5d6", ["capture", "move"]),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", ["promotion"]),
        ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", ["capture", "promotion"]),
    ],
)
def test_special_moves_require_complete_physical_change(fen, uci, kinds):
    board = chess.Board(fen)
    ops, expected = physical_plan(board, chess.Move.from_uci(uci))
    assert [op.kind for op in ops] == kinds
    if board.is_en_passant(chess.Move.from_uci(uci)):
        assert expected["piece_d5"] == "capture"
    config = load_config()
    config["episode"].update(initial_fen=fen, history=[], move=uci)
    t = make_task("CH1", config=config)
    settle(t)
    press(t)
    board.push_uci(uci)
    assert t.board.fen() == board.fen() and t.joint_success


def test_pinned_piece_illegal_move_rejected():
    board = chess.Board("4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1")
    with pytest.raises(ValueError, match="legal"):
        physical_plan(board, chess.Move.from_uci("e2f2"))


def test_terminal_positions_and_history_are_checked():
    with pytest.raises(ValueError, match="nonterminal"):
        game_board({"initial_fen": "7k/6Q1/5K2/8/8/8/8/8 b - - 0 1"})
    cfg = {"history": ["g1f3", "g8f6", "f3g1", "f6g8"] * 2}
    board = game_board(cfg)
    assert board.can_claim_threefold_repetition()
    with pytest.raises(ValueError, match="nonterminal"):
        game_board({**cfg, "claim_draw": True})


def test_seed_reproducible_and_difficulty_only_changes_deadline():
    a = make_task("CH1", "easy", 9)
    b = make_task("CH1", "hard", 9)
    assert a.spec.opponent_press_time == b.spec.opponent_press_time
    assert a.spec.fen == b.spec.fen and a.spec.move_uci == b.spec.move_uci
    assert a.spec.deadline > b.spec.deadline


def test_asset_quantization_has_correct_metric_scale():
    for filename, expected in (
        ("board.glb", [0.46, 0.46, 0.019]),
        ("clock.glb", [0.16, 0.058, 0.079]),
    ):
        vertices = np.concatenate([m[1] for m in read_glb(ROOT / "models" / filename)])
        assert np.allclose(np.ptp(vertices, axis=0), expected, atol=0.0001)
