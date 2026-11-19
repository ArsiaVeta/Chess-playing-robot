"""Print one Stockfish move for a FEN position."""

import argparse
import sys

import chess

from src.chess_core.engine import EngineConfig, EngineError, StockfishEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask Stockfish for a legal move")
    parser.add_argument("--config", default="config/engine.example.toml")
    parser.add_argument("--fen", default=chess.STARTING_FEN)
    args = parser.parse_args()
    try:
        board = chess.Board(args.fen)
        if not board.is_valid():
            raise ValueError("Invalid chess position")
        config = EngineConfig.from_file(args.config)
        with StockfishEngine(config) as engine:
            print(engine.choose_move(board).uci())
    except (EngineError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
