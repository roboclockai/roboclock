"""Reproducible legal opening exchanges, independent of temporal difficulty."""

import chess
import numpy as np

from roboclock.envs.chess.config import load_config
from roboclock.envs.sports.arcade.tasks import configuration


def opening_moves(board):
    """Quiet central pawn moves and inward knight development from home squares.

    This deliberately bounded opening distribution avoids captures and special
    moves in the physical demo. Every candidate still passes the full rules engine.
    """
    candidates = []
    for move in sorted(board.legal_moves, key=lambda move: move.uci()):
        piece = board.piece_at(move.from_square)
        home_rank = 1 if board.turn == chess.WHITE else 6
        pawn = (
            piece.piece_type == chess.PAWN
            and chess.square_rank(move.from_square) == home_rank
            and chess.square_file(move.from_square) in (2, 3, 4, 5)
        )
        knight = move.uci() in {"b1c3", "g1f3", "b8c6", "g8f6"}
        if (pawn or knight) and not board.is_capture(move) and not move.promotion:
            candidates.append(move)
    return candidates


def config(difficulty="easy", seed=0):
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError("Unknown chess difficulty")
    rng = np.random.default_rng(seed)
    board = chess.Board()
    history, moves = [], []
    for ply in range(5):
        candidates = opening_moves(board)
        if not candidates:
            raise RuntimeError("Opening distribution exhausted")
        move = candidates[int(rng.integers(len(candidates)))]
        (history if ply == 0 else moves).append(move.uci())
        board.push(move)

    c = load_config()
    visual = configuration("A1", room_style="v5")
    c["sim"], c["visuals"] = visual["sim"], visual["visuals"]
    c["sim"].update(settle_steps=94, camera_resolution=[1280, 960])  # 0.375 s
    c["scene"].update(
        room_style="v5",
        piece_height=0.050,
        camera_position=[0.48, -2.20, 2.10],
        camera_look_at=[0.48, 0, 1.02],
    )
    # Move/press motion of both players' scripted controllers (2026-11-04; was
    # 0.22 m/s Cartesian, 0.16 / 0.045 m/s pawn lowering, 1.5 rad/s joints).
    c["control"].update(
        cartesian_speed_chess=0.34,
        lower_speed_chess=0.22,
        brake_height_chess=0.025,
        brake_speed_chess=0.045,
        joint_target_speed=2.4,
    )
    c["episode"].update(history=history, move=moves[0])
    c.update(task_id="A1", seed=int(seed), difficulty=difficulty)
    # Clock deadline from BLACK's press (2026-11-16 recalibration; was 30 / 18 / 10 s):
    # the reference oracle presses WHITE's clock 6.42-7.90 s after BLACK's (seeds 0-9,
    # every tier). Hard = max(t / 0.8, t + 0.3) = 9.875 -> 10 s (unchanged), medium =
    # sqrt(easy * hard) -> 17 s. The standalone chess env keeps its own 30 / 18 / 10 s.
    c["difficulties"]["medium"]["deadline"] = 17.0
    c["duel"] = dict(
        moves=moves,
        max_time=140,
        waiting_joint0={"white": 0.85, "black": -0.85},
    )
    c["randomization"] = dict(
        version="opening-v1",
        generator="numpy.PCG64",
        seed=int(seed),
        distribution="Uniform legal candidate per ply; one setup ply, four physical plies",
        scope="Quiet home-square c/d/e/f pawn advances and inward knight development",
        history=history.copy(),
        moves=moves.copy(),
        excluded="Captures, castling, en passant, promotions, arbitrary midgame positions",
    )
    return c
