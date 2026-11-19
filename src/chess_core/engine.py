"""Stockfish adapter using the UCI protocol."""

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import tomllib

import chess
import chess.engine


class EngineError(RuntimeError):
    """The chess engine could not produce a usable move."""


@dataclass(frozen=True)
class EngineConfig:
    path: str = "stockfish"
    time: float = 0.5
    depth: int | None = None
    timeout: float = 10.0

    def __post_init__(self):
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("Engine path must not be empty")
        for name in ("time", "timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if self.depth is not None and (isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth <= 0):
            raise ValueError("depth must be a positive integer")

    @classmethod
    def from_file(cls, path: str | Path) -> "EngineConfig":
        with Path(path).open("rb") as config_file:
            return cls(**tomllib.load(config_file)["engine"])


class StockfishEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._engine = None

    def choose_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            raise EngineError("The game is already over")
        try:
            if self._engine is None:
                if shutil.which(self.config.path) is None:
                    raise FileNotFoundError(f"Stockfish executable not found: {self.config.path}")
                self._engine = chess.engine.SimpleEngine.popen_uci(
                    self.config.path, timeout=self.config.timeout
                )
            result = self._engine.play(
                board, chess.engine.Limit(time=self.config.time, depth=self.config.depth)
            )
        except (OSError, chess.engine.EngineError, TimeoutError) as exc:
            self.close()
            raise EngineError(f"Stockfish failed: {exc}") from exc
        if result.move is None or result.move not in board.legal_moves:
            raise EngineError(f"Engine returned an illegal move: {result.move}")
        return result.move

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
            except (OSError, chess.engine.EngineError, TimeoutError):
                engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
