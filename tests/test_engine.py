from types import SimpleNamespace
from unittest.mock import Mock

import chess
import chess.engine
import pytest

from src.chess_core.engine import EngineConfig, EngineError, StockfishEngine


def test_engine_returns_legal_move_and_closes(monkeypatch):
    monkeypatch.setattr("src.chess_core.engine.shutil.which", lambda path: path)
    fake = Mock()
    fake.play.return_value = SimpleNamespace(move=chess.Move.from_uci("e2e4"))
    start = Mock(return_value=fake)
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", start)
    board = chess.Board()
    with StockfishEngine(EngineConfig(time=0.1)) as engine:
        assert engine.choose_move(board).uci() == "e2e4"
        assert board.fen() == chess.STARTING_FEN
    fake.quit.assert_called_once()
    assert start.call_count == 1


def test_missing_engine_is_reported(monkeypatch):
    monkeypatch.setattr("src.chess_core.engine.shutil.which", lambda path: None)
    with pytest.raises(EngineError, match="executable not found"):
        StockfishEngine(EngineConfig()).choose_move(chess.Board())


def test_timeout_closes_engine(monkeypatch):
    monkeypatch.setattr("src.chess_core.engine.shutil.which", lambda path: path)
    fake = Mock()
    fake.play.side_effect = TimeoutError("timed out")
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=fake))
    with pytest.raises(EngineError, match="timed out"):
        StockfishEngine(EngineConfig()).choose_move(chess.Board())
    fake.quit.assert_called_once()


def test_illegal_engine_result_rejected(monkeypatch):
    monkeypatch.setattr("src.chess_core.engine.shutil.which", lambda path: path)
    fake = Mock()
    fake.play.return_value = SimpleNamespace(move=chess.Move.from_uci("e2e5"))
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", Mock(return_value=fake))
    with StockfishEngine(EngineConfig()) as engine:
        with pytest.raises(EngineError, match="illegal"):
            engine.choose_move(chess.Board())


def test_config_from_file():
    config = EngineConfig.from_file("config/engine.example.toml")
    assert config.time > 0


@pytest.mark.parametrize("kwargs", [{"time": 0}, {"timeout": float('nan')}, {"depth": True}, {"path": ""}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        EngineConfig(**kwargs)
