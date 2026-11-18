"""Conservative move inference from two board observations."""

from dataclasses import dataclass
from enum import Enum

import chess
from src.common.models import BoardObservation


class DetectionStatus(Enum):
    FOUND = "FOUND"
    NO_MOVE = "NO_MOVE"
    AMBIGUOUS = "AMBIGUOUS"
    INVALID = "INVALID"
    UNSTABLE = "UNSTABLE"


@dataclass(frozen=True)
class MoveDetection:
    status: DetectionStatus
    move: chess.Move | None = None
    candidates: tuple[chess.Move, ...] = ()
    reason: str = ""


def _occupancy(board: chess.Board) -> dict[str, bool]:
    return {chess.square_name(s): board.piece_at(s) is not None for s in chess.SQUARES}


def detect_move(board: chess.Board, before: BoardObservation, after: BoardObservation,
                min_confidence: float = 0.9) -> MoveDetection:
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")
    if not before.stable or not after.stable:
        return MoveDetection(DetectionStatus.UNSTABLE, reason="Observation is not stable")
    if any(v < min_confidence for obs in (before, after) for v in obs.confidence.values()):
        return MoveDetection(DetectionStatus.INVALID, reason="Observation confidence is too low")
    if dict(before.occupied) != _occupancy(board):
        return MoveDetection(DetectionStatus.INVALID, reason="Before observation does not match the position")

    candidates = []
    original = board.piece_map()
    for move in board.legal_moves:
        trial = board.copy(stack=False)
        trial.push(move)
        if _occupancy(trial) != dict(after.occupied):
            continue
        # An empty set means no visual change evidence is available, not proof of no changes.
        changed = {chess.square_name(s) for s in chess.SQUARES if original.get(s) != trial.piece_at(s)}
        if after.changed_squares and not changed.issubset(after.changed_squares):
            continue
        candidates.append(move)
    candidates = tuple(candidates)
    if len(candidates) == 1:
        return MoveDetection(DetectionStatus.FOUND, candidates[0], candidates, "One legal move matches")
    if candidates:
        return MoveDetection(DetectionStatus.AMBIGUOUS, candidates=candidates, reason="Multiple legal moves match")
    if dict(before.occupied) == dict(after.occupied) and not after.changed_squares:
        return MoveDetection(DetectionStatus.NO_MOVE, reason="No changes observed")
    return MoveDetection(DetectionStatus.INVALID, reason="No legal move matches the observation")
