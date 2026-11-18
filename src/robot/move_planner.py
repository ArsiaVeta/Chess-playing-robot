"""Convert legal chess moves to logical arm operations."""

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import chess
from src.robot.arm import Action, MovePiece, RemovePiece, PlaceReservePiece


@dataclass(frozen=True)
class MovePlan:
    move: chess.Move
    actions: tuple[Action, ...]


def plan_move(board: chess.Board, move: chess.Move, *, capture_slots: Sequence[str] = (),
              reserve: Mapping[str, chess.Piece] | None = None) -> MovePlan:
    if move not in board.legal_moves:
        raise ValueError(f"Illegal move: {move}")
    if board.chess960:
        raise ValueError("Chess960 castling is not supported")
    actions: list[Action] = []
    slots = iter(capture_slots)

    def remove(square: chess.Square) -> None:
        slot = next(slots, None)
        if not isinstance(slot, str) or not slot or any(isinstance(a, RemovePiece) and a.slot == slot for a in actions):
            raise ValueError("A distinct empty capture slot is required for each removed piece")
        actions.append(RemovePiece(square, slot))

    if board.is_capture(move):
        square = move.to_square
        if board.is_en_passant(move):
            square += -8 if board.turn == chess.WHITE else 8
        remove(square)
    if move.promotion:
        remove(move.from_square)
        piece = chess.Piece(move.promotion, board.turn)
        used_slots = {a.slot for a in actions if isinstance(a, RemovePiece)}
        slot = next((s for s, p in (reserve or {}).items()
                     if isinstance(s, str) and s and s not in used_slots and p == piece), None)
        if slot is None:
            raise ValueError(f"No reserve piece available: {piece}")
        actions.append(PlaceReservePiece(piece, slot, move.to_square))
    else:
        actions.append(MovePiece(move.from_square, move.to_square))
        if board.is_castling(move):
            rank = chess.square_rank(move.from_square)
            kingside = board.is_kingside_castling(move)
            actions.append(MovePiece(chess.square(7 if kingside else 0, rank),
                                     chess.square(5 if kingside else 3, rank)))
    return MovePlan(move, tuple(actions))
