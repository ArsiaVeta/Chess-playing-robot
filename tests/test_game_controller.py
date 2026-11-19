import chess
import pytest

from src.chess_core.game import ChessGame
from src.chess_core.move_detector import DetectionStatus
from src.common.models import BoardObservation
from src.controller.game_controller import GameController, GameState
from src.robot.arm import ActionStatus, MovePiece
from src.robot.mock_arm import MockArm


class FakeVision:
    def __init__(self, arm):
        self.arm = arm
        self.stable = True
        self.hidden_square = None
        self.confidence = 1.0
        self.reference = None
        self.confirmed = []
        self.timeouts = []
        self.report_changes = True

    def get_initial_observation(self, *, timeout):
        return self._capture(timeout, reference=None)

    def get_observation(self, *, timeout):
        assert self.reference is not None
        return self._capture(timeout, self.reference)

    def _capture(self, timeout, reference):
        self.timeouts.append(timeout)
        if not self.stable:
            raise TimeoutError("No stable frame")
        self._candidate = {
            square: piece for square, piece in self.arm.pieces.items()
            if square != self.hidden_square
        }
        occupied = {chess.square_name(s): s in self._candidate for s in chess.SQUARES}
        changed = frozenset(
            chess.square_name(s) for s in chess.SQUARES
            if reference is not None and self.report_changes
            and reference.get(s) != self._candidate.get(s)
        )
        self._observation = BoardObservation(
            occupied, dict.fromkeys(occupied, self.confidence), True, changed
        )
        return self._observation

    def confirm_observation(self, observation):
        assert observation is self._observation
        self.reference = self._candidate
        self.confirmed.append(observation)


class ScriptedEngine:
    def __init__(self, *moves):
        self.moves = iter(moves)
        self.positions = []

    def choose_move(self, board):
        self.positions.append(board.fen())
        return chess.Move.from_uci(next(self.moves))


def test_complete_fools_mate_with_vision_and_arm():
    game = ChessGame()
    arm = MockArm(game.board)
    engine = ScriptedEngine("e7e5", "d8h4")
    vision = FakeVision(arm)
    controller = GameController(game, vision, engine, arm)
    assert controller.step() is GameState.WAIT_HUMAN
    assert controller.step() is GameState.WAIT_HUMAN
    assert controller.last_detection.status is DetectionStatus.NO_MOVE

    arm.pieces[chess.F3] = arm.pieces.pop(chess.F2)
    assert controller.step() is GameState.WAIT_HUMAN
    assert arm.pieces == game.board.piece_map()
    arm.pieces[chess.G4] = arm.pieces.pop(chess.G2)
    assert controller.step() is GameState.FINISHED
    assert [move.uci() for move in game.history] == ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert game.outcome.termination is chess.Termination.CHECKMATE
    assert arm.pieces == game.board.piece_map()
    assert controller.step() is GameState.FINISHED
    assert len(arm.log) == 2
    assert [frame.changed_squares for frame in vision.confirmed] == [
        set(), {"f2", "f3"}, {"e7", "e5"}, {"g2", "g4"}, {"d8", "h4"}
    ]


def test_human_observation_timeout_leaves_reference_and_history_intact():
    game = ChessGame()
    arm = MockArm(game.board)
    vision = FakeVision(arm)
    engine = ScriptedEngine("e7e5")
    controller = GameController(game, vision, engine, arm, observation_timeout=3)
    controller.step()
    arm.pieces[chess.E4] = arm.pieces.pop(chess.E2)
    vision.stable = False
    assert controller.step() is GameState.ERROR
    assert "TimeoutError" in controller.error
    assert not game.history and not engine.positions and not arm.log
    assert len(vision.confirmed) == 1
    assert vision.timeouts == [3, 3]


def test_empty_changes_cannot_confirm_human_move():
    game = ChessGame()
    arm = MockArm(game.board)
    vision = FakeVision(arm)
    controller = GameController(game, vision, ScriptedEngine(), arm)
    controller.step()
    arm.pieces[chess.E4] = arm.pieces.pop(chess.E2)
    vision.report_changes = False
    assert controller.step() is GameState.WAIT_HUMAN
    assert controller.last_detection.status is DetectionStatus.INVALID
    assert not game.history
    assert len(vision.confirmed) == 1


def test_partial_capture_failure_keeps_logical_position_and_stops():
    class FailingArm(MockArm):
        def execute(self, action):
            if isinstance(action, MovePiece):
                self.fail_next(ActionStatus.GRIP_FAILURE)
            return super().execute(action)

    game = ChessGame("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    arm = FailingArm(game.board)
    controller = GameController(
        game, FakeVision(arm), ScriptedEngine("e4d5"), arm,
        robot_color=chess.WHITE, capture_slots=("taken",),
    )
    assert controller.step() is GameState.ERROR
    assert "GRIP_FAILURE" in controller.error
    assert game.board.piece_at(chess.D5) == chess.Piece.from_symbol("p")
    assert chess.D5 not in arm.pieces
    assert not game.history
    assert controller.step() is GameState.ERROR
    assert len(arm.log) == 2


@pytest.mark.parametrize("problem", [None, "occupancy", "changes", "timeout"])
def test_robot_move_awaits_reliable_verification_without_reexecution(problem):
    game = ChessGame()
    arm = MockArm(game.board)
    vision = FakeVision(arm)

    class UnsettledArm(MockArm):
        def execute(self, action):
            vision.confidence = 0.5
            return super().execute(action)

    arm = UnsettledArm(game.board)
    vision.arm = arm
    controller = GameController(
        game, vision, ScriptedEngine("e2e4"), arm, robot_color=chess.WHITE
    )
    assert controller.step() is GameState.VERIFY_ROBOT
    assert not game.history
    vision.confidence = 1.0
    if problem == "occupancy":
        vision.hidden_square = chess.E4
    elif problem == "changes":
        vision.report_changes = False
    elif problem == "timeout":
        vision.stable = False
    expected = GameState.ERROR if problem else GameState.WAIT_HUMAN
    assert controller.step() is expected
    assert len(game.history) == (0 if problem else 1)
    assert len(vision.confirmed) == (1 if problem else 2)
    assert len(arm.log) == 1
