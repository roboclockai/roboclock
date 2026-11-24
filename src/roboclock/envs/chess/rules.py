"""Standard chess legality and explicit physical operations for every move type."""

from dataclasses import dataclass

import chess


def game_board(episode):
    fen = episode.get("initial_fen", "startpos")
    board = chess.Board() if fen == "startpos" else chess.Board(fen)
    if not board.is_valid():
        raise ValueError("Invalid standard chess FEN")
    for uci in episode.get("history", []):
        if board.is_game_over():
            raise ValueError("History continues after a terminal position")
        board.push(board.parse_uci(uci))
    if board.is_game_over(claim_draw=episode.get("claim_draw", False)):
        raise ValueError("Episode requires a nonterminal position with a legal move")
    return board


@dataclass(frozen=True)
class Operation:
    piece: str
    destination: str
    kind: str = "move"


def piece_id(square):
    return "piece_" + chess.square_name(square)


def physical_plan(board, move):
    if move not in board.legal_moves:
        raise ValueError("Move must be legal, including king safety")
    operations = []
    expected = {piece_id(sq): chess.square_name(sq) for sq in board.piece_map()}
    if board.is_capture(move):
        captured = move.to_square
        if board.is_en_passant(move):
            captured += -8 if board.turn == chess.WHITE else 8
        name = piece_id(captured)
        expected[name] = "capture"
        operations.append(Operation(name, "capture", "capture"))
    mover = piece_id(move.from_square)
    expected[mover] = chess.square_name(move.to_square)
    operations.append(Operation(mover, expected[mover]))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
        rook = piece_id(chess.square(7 if kingside else 0, rank))
        expected[rook] = chess.square_name(chess.square(5 if kingside else 3, rank))
        operations.append(Operation(rook, expected[rook], "castle_rook"))
    if move.promotion:
        # Cylinder tokens have electronic identity labels. Promotion changes the label
        # only after physical placement; no rigid-body replacement or teleportation.
        operations[-1] = Operation(mover, expected[mover], "promotion")
    return operations, expected
