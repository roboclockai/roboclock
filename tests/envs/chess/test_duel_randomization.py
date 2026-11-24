"""Legal, replayable opening diversity without changing temporal difficulty."""

import chess
import pytest

from roboclock.envs.sports.chess.config import config
from roboclock.envs.sports.chess.task import DuelTask


def test_openings_are_legal_replayable_and_diverse():
    scenarios = set()
    for seed in range(100):
        cfg = config(seed=seed)
        assert cfg == config(seed=seed)
        history, moves = cfg["episode"]["history"], cfg["duel"]["moves"]
        assert len(history) == 1 and len(moves) == 4
        scenarios.add(tuple(history + moves))
        board = chess.Board()
        for uci in history + moves:
            move = board.parse_uci(uci)  # Full chess legality, including king safety.
            assert not board.is_capture(move)
            assert not board.is_castling(move)
            assert not move.promotion
            board.push(move)
            assert board.is_valid() and not board.is_game_over()
        task = DuelTask("easy", seed, cfg)
        assert task.actor == "black"
        assert task.move.uci() == moves[0]
    assert len(scenarios) >= 95


@pytest.mark.parametrize("seed", [0, 1, 17, 99])
def test_difficulty_changes_deadline_not_opening(seed):
    configs = [config(level, seed) for level in ("easy", "medium", "hard")]
    for cfg in configs[1:]:
        assert cfg["episode"] == configs[0]["episode"]
        assert cfg["duel"] == configs[0]["duel"]
        assert cfg["scene"] == configs[0]["scene"]
        assert cfg["randomization"] == configs[0]["randomization"]
    deadlines = [DuelTask(cfg["difficulty"], seed, cfg).spec.deadline for cfg in configs]
    assert deadlines[0] > deadlines[1] > deadlines[2]


@pytest.mark.parametrize("seed", [-1, -42])
def test_negative_seed_rejected(seed):
    with pytest.raises(ValueError):
        config(seed=seed)
