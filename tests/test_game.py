import chess
import pytest

from src.chess_core.game import ChessGame


def test_moves_and_undo():
    game = ChessGame()
    initial = game.board.fen()
    game.push("e2e4")
    assert game.turn == chess.BLACK
    assert game.history == (chess.Move.from_uci("e2e4"),)
    game.undo()
    assert game.board.fen() == initial


def test_illegal_move_does_not_change_position():
    game = ChessGame()
    initial = game.board.fen()
    with pytest.raises(ValueError):
        game.push("e2e5")
    assert game.board.fen() == initial


@pytest.mark.parametrize("fen,termination", [
    ("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", chess.Termination.CHECKMATE),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", chess.Termination.STALEMATE),
    ("7k/8/6K1/8/8/8/8/8 w - - 0 1", chess.Termination.INSUFFICIENT_MATERIAL),
])
def test_finished_positions(fen, termination):
    game = ChessGame(fen)
    assert game.outcome.termination == termination
    with pytest.raises(ValueError):
        game.push("g6g7")


def test_resignation_and_new_game():
    game = ChessGame()
    game.resign()
    assert game.outcome.winner == chess.BLACK
    with pytest.raises(ValueError):
        game.push("e2e4")
    game.new_game()
    assert game.outcome is None
