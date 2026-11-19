"""Logical arm actions and execution result."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import chess


@dataclass(frozen=True)
class MovePiece:
    source: chess.Square
    target: chess.Square


@dataclass(frozen=True)
class RemovePiece:
    square: chess.Square
    slot: str


@dataclass(frozen=True)
class PlaceReservePiece:
    piece: chess.Piece
    slot: str
    target: chess.Square


Action = MovePiece | RemovePiece | PlaceReservePiece


class ActionStatus(Enum):
    SUCCESS = "SUCCESS"
    GRIP_FAILURE = "GRIP_FAILURE"
    TIMEOUT = "TIMEOUT"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ActionResult:
    status: ActionStatus
    message: str = ""

    @property
    def success(self) -> bool:
        return self.status is ActionStatus.SUCCESS


class Arm(ABC):
    @abstractmethod
    def execute(self, action: Action) -> ActionResult:
        """Execute one action and report its result after motion has stopped."""
