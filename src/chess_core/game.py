"""Chess session with legal move validation and explicit resignation."""

from enum import Enum

import chess


class ExternalTermination(Enum):
    RESIGNATION = "resignation"


class ChessGame:
    def __init__(self, fen: str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self._resigned: chess.Color | None = None

    def new_game(self, fen: str = chess.STARTING_FEN) -> None:
        self.board = chess.Board(fen)
        self._resigned = None

    @property
    def history(self) -> tuple[chess.Move, ...]:
        return tuple(self.board.move_stack)

    @property
    def turn(self) -> chess.Color:
        return self.board.turn

    @property
    def outcome(self) -> chess.Outcome | None:
        if self._resigned is not None:
            return chess.Outcome(ExternalTermination.RESIGNATION, not self._resigned)
        return self.board.outcome()

    def push(self, move: chess.Move | str) -> chess.Move:
        if self.outcome is not None:
            raise ValueError("The game is finished")
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        if move not in self.board.legal_moves:
            raise ValueError(f"Illegal move: {move}")
        self.board.push(move)
        return move

    def undo(self) -> chess.Move:
        if self._resigned is not None:
            raise ValueError("Start a new game after resignation")
        if not self.board.move_stack:
            raise ValueError("No moves to undo")
        return self.board.pop()

    def resign(self, color: chess.Color | None = None) -> None:
        if self.outcome is not None:
            raise ValueError("The game is finished")
        if color is not None and not isinstance(color, bool):
            raise TypeError("color must be chess.WHITE or chess.BLACK")
        self._resigned = self.turn if color is None else color
