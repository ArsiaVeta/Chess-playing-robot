"""Virtual arm with reproducible failures and physical piece state."""

from collections.abc import Mapping
import chess
from src.robot.arm import Action, ActionResult, ActionStatus, Arm, MovePiece, RemovePiece, PlaceReservePiece


class MockArm(Arm):
    def __init__(self, board: chess.Board, *, reserve: Mapping[str, chess.Piece] | None = None):
        self.pieces = dict(board.piece_map())
        self.reserve = dict(reserve or {})
        self.captured: dict[str, chess.Piece] = {}
        self.log: list[tuple[Action, ActionResult]] = []
        self.next_failure: ActionStatus | None = None
        self.stopped = False

    def fail_next(self, status: ActionStatus) -> None:
        if status not in (ActionStatus.GRIP_FAILURE, ActionStatus.TIMEOUT, ActionStatus.EMERGENCY_STOP):
            raise ValueError("Unsupported simulated failure")
        self.next_failure = status

    def reset_stop(self) -> None:
        self.stopped = False

    def execute(self, action: Action) -> ActionResult:
        if self.stopped:
            result = ActionResult(ActionStatus.EMERGENCY_STOP, "Arm is stopped")
        elif self.next_failure is not None:
            status, self.next_failure = self.next_failure, None
            self.stopped = status is ActionStatus.EMERGENCY_STOP
            result = ActionResult(status, "Simulated failure; action was not applied")
        elif isinstance(action, MovePiece):
            if action.source not in self.pieces or action.target in self.pieces:
                result = ActionResult(ActionStatus.INVALID, "Source empty or target occupied")
            else:
                self.pieces[action.target] = self.pieces.pop(action.source)
                result = ActionResult(ActionStatus.SUCCESS)
        elif isinstance(action, RemovePiece):
            if action.square not in self.pieces or not action.slot or action.slot in self.captured or action.slot in self.reserve:
                result = ActionResult(ActionStatus.INVALID, "Invalid removal or occupied slot")
            else:
                self.captured[action.slot] = self.pieces.pop(action.square)
                result = ActionResult(ActionStatus.SUCCESS)
        elif isinstance(action, PlaceReservePiece):
            if self.reserve.get(action.slot) != action.piece or action.target in self.pieces:
                result = ActionResult(ActionStatus.INVALID, "Reserve piece unavailable or target occupied")
            else:
                self.pieces[action.target] = self.reserve.pop(action.slot)
                result = ActionResult(ActionStatus.SUCCESS)
        else:
            result = ActionResult(ActionStatus.INVALID, "Unknown action")
        self.log.append((action, result))
        return result
