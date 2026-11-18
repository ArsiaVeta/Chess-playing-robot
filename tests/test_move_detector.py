import chess
import pytest
from src.common.models import BoardObservation
from src.chess_core.move_detector import DetectionStatus, detect_move


def observation(board, *, changed=(), stable=True, confidence=1.0):
    names = [chess.square_name(s) for s in chess.SQUARES]
    return BoardObservation({n: board.piece_at(chess.parse_square(n)) is not None for n in names},
                            {n: confidence for n in names}, stable, frozenset(changed))


def after_move(board, uci, **kwargs):
    after = board.copy()
    after.push_uci(uci)
    return observation(after, **kwargs)


def test_detect_normal_move_without_changing_board():
    board = chess.Board()
    result = detect_move(board, observation(board), after_move(board, "e2e4"))
    assert result.status is DetectionStatus.FOUND
    assert result.move.uci() == "e2e4"
    assert board.fen() == chess.STARTING_FEN


@pytest.mark.parametrize("fen,move", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
])
def test_special_and_capture_moves(fen, move):
    board = chess.Board(fen)
    assert detect_move(board, observation(board), after_move(board, move)).move.uci() == move


def test_promotion_is_ambiguous():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    result = detect_move(board, observation(board), after_move(board, "a7a8q"))
    assert result.status is DetectionStatus.AMBIGUOUS
    assert len(result.candidates) == 4


def test_rejects_inconsistent_or_unreliable_observation():
    board = chess.Board()
    other = board.copy()
    other.push_uci("e2e4")
    assert detect_move(board, observation(other), observation(other)).status is DetectionStatus.INVALID
    assert detect_move(board, observation(board), observation(board, stable=False)).status is DetectionStatus.UNSTABLE
    assert detect_move(board, observation(board), observation(board, confidence=0.2)).status is DetectionStatus.INVALID
    assert detect_move(board, observation(board), observation(board)).status is DetectionStatus.NO_MOVE
    invalid = board.copy()
    invalid.remove_piece_at(chess.E2)
    assert detect_move(board, observation(board), observation(invalid)).status is DetectionStatus.INVALID


def test_capture_ambiguity_can_be_resolved_with_changed_square():
    board = chess.Board("4k3/8/8/2p1p3/3R4/8/8/4K3 w - - 0 1")
    # A rook cannot capture diagonally; use a queen that can capture either pawn.
    board.set_piece_at(chess.D4, chess.Piece(chess.QUEEN, chess.WHITE))
    after = after_move(board, "d4c5")
    assert detect_move(board, observation(board), after).status is DetectionStatus.AMBIGUOUS
    after = after_move(board, "d4c5", changed={"d4", "c5"})
    assert detect_move(board, observation(board), after).move.uci() == "d4c5"
