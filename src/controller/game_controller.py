"""Coordinate an observed human move and a verified robot reply."""

from collections.abc import Mapping, Sequence
from enum import Enum
import math
from typing import Protocol

import chess

from src.chess_core.game import ChessGame
from src.chess_core.move_detector import DetectionStatus, MoveDetection, detect_move
from src.common.models import BoardObservation
from src.robot.arm import Arm, PlaceReservePiece, RemovePiece
from src.robot.move_planner import plan_move


class GameState(Enum):
    WAIT_HUMAN = "WAIT_HUMAN"
    VERIFY_HUMAN = "VERIFY_HUMAN"
    THINKING = "THINKING"
    EXECUTE_ROBOT = "EXECUTE_ROBOT"
    VERIFY_ROBOT = "VERIFY_ROBOT"
    ERROR = "ERROR"
    FINISHED = "FINISHED"


class Vision(Protocol):
    def get_initial_observation(self, *, timeout: float) -> BoardObservation:
        """Wait for a stable initial frame without a reference, or raise TimeoutError."""

    def get_observation(self, *, timeout: float) -> BoardObservation:
        """Wait for a fresh stable frame and compare it to the confirmed reference.

        Use frames acquired after the call starts, with no hand obstructing the
        board. Raise TimeoutError if no suitable frame arrives in time.
        """

    def confirm_observation(self, observation: BoardObservation) -> None:
        """Use the exact frame behind this observation as the new reference."""


class MoveEngine(Protocol):
    def choose_move(self, board: chess.Board) -> chess.Move: ...


class GameController:
    def __init__(
        self,
        game: ChessGame,
        vision: Vision,
        engine: MoveEngine,
        arm: Arm,
        *,
        robot_color: chess.Color = chess.BLACK,
        capture_slots: Sequence[str] = (),
        reserve: Mapping[str, chess.Piece] | None = None,
        min_confidence: float = 0.9,
        observation_timeout: float = 10.0,
    ):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not math.isfinite(observation_timeout) or observation_timeout <= 0:
            raise ValueError("observation_timeout must be positive and finite")
        self.game = game
        self.vision = vision
        self.engine = engine
        self.arm = arm
        self.robot_color = robot_color
        self.min_confidence = min_confidence
        self.observation_timeout = observation_timeout
        self.state = GameState.FINISHED if game.outcome else GameState.WAIT_HUMAN
        self.error: str | None = None
        self.last_detection: MoveDetection | None = None
        self._baseline: BoardObservation | None = None
        self._pending_move: chess.Move | None = None
        self._reserve = dict(reserve or {})
        self._capture_slots = [
            slot for slot in dict.fromkeys(capture_slots) if slot not in self._reserve
        ]

    def step(self) -> GameState:
        """Poll once, executing at most one robot move. ERROR is latched."""
        if self.state in (GameState.ERROR, GameState.FINISHED):
            return self.state
        if self.game.outcome is not None and self._pending_move is None:
            self.state = GameState.FINISHED
            return self.state
        try:
            self._step()
        except Exception as exc:
            stage = self.state
            self.error = f"{stage.value}: {type(exc).__name__}: {exc}"
            self.state = GameState.ERROR
        return self.state

    def _step(self) -> None:
        if self._pending_move is not None:
            self._verify_robot()
            return
        if self._baseline is None:
            observation = self.vision.get_initial_observation(timeout=self.observation_timeout)
            if not self._reliable(observation):
                return
            self._check_position(self.game.board, observation)
            self._confirm_observation(observation)
        elif self.game.turn != self.robot_color:
            self.state = GameState.VERIFY_HUMAN
            observation = self.vision.get_observation(timeout=self.observation_timeout)
            detection = detect_move(
                self.game.board, self._baseline, observation, self.min_confidence,
                require_changes=True,
            )
            self.last_detection = detection
            if detection.status is not DetectionStatus.FOUND or detection.move is None:
                self.state = GameState.WAIT_HUMAN
                return
            self.game.push(detection.move)
            self._confirm_observation(observation)
        if self.game.outcome is not None:
            self.state = GameState.FINISHED
        elif self.game.turn == self.robot_color:
            self._execute_robot()

    def _execute_robot(self) -> None:
        self.state = GameState.THINKING
        move = self.engine.choose_move(self.game.board.copy())
        plan = plan_move(
            self.game.board, move, capture_slots=self._capture_slots, reserve=self._reserve
        )
        self.state = GameState.EXECUTE_ROBOT
        for action in plan.actions:
            result = self.arm.execute(action)
            if not result.success:
                raise RuntimeError(f"{result.status.value}: {result.message}")
            if isinstance(action, RemovePiece):
                self._capture_slots.remove(action.slot)
            elif isinstance(action, PlaceReservePiece):
                del self._reserve[action.slot]
        self._pending_move = move
        self.state = GameState.VERIFY_ROBOT
        self._verify_robot()

    def _verify_robot(self) -> None:
        observation = self.vision.get_observation(timeout=self.observation_timeout)
        if not self._reliable(observation):
            return
        move = self._pending_move
        assert move is not None
        expected = self.game.board.copy(stack=False)
        expected.push(move)
        self._check_position(expected, observation)
        changed = {
            chess.square_name(square) for square in chess.SQUARES
            if self.game.board.piece_at(square) != expected.piece_at(square)
        }
        if not changed.issubset(observation.changed_squares):
            raise ValueError("Observed changes do not match the robot move")
        self.game.push(move)
        self._pending_move = None
        self._confirm_observation(observation)
        self.state = GameState.FINISHED if self.game.outcome else GameState.WAIT_HUMAN

    def _confirm_observation(self, observation: BoardObservation) -> None:
        self.vision.confirm_observation(observation)
        self._baseline = observation

    def _reliable(self, observation: BoardObservation) -> bool:
        return observation.stable and all(
            confidence >= self.min_confidence for confidence in observation.confidence.values()
        )

    @staticmethod
    def _check_position(board: chess.Board, observation: BoardObservation) -> None:
        occupied = {chess.square_name(square) for square in board.piece_map()}
        if occupied != observation.occupied_squares:
            raise ValueError("Observed position does not match the expected position")
