import chess
import pytest
from src.robot.arm import ActionStatus, MovePiece, RemovePiece, PlaceReservePiece
from src.robot.mock_arm import MockArm
from src.robot.move_planner import plan_move


@pytest.mark.parametrize("fen,uci,slots,reserve", [
    (chess.STARTING_FEN, "e2e4", (), {}),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5", ("taken",), {}),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", ("taken",), {}),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", (), {}),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1", (), {}),
    ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n", ("pawn",), {"knight": chess.Piece.from_symbol("N")}),
    ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", ("taken", "pawn"), {"queen": chess.Piece.from_symbol("Q")}),
])
def test_plan_executes_to_expected_position(fen, uci, slots, reserve):
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    plan = plan_move(board, move, capture_slots=slots, reserve=reserve)
    arm = MockArm(board, reserve=reserve)
    for action in plan.actions:
        assert arm.execute(action).success
    board.push(move)
    assert arm.pieces == board.piece_map()
    assert len(arm.log) == len(plan.actions)


def test_capture_and_en_passant_order():
    board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    plan = plan_move(board, chess.Move.from_uci("e5d6"), capture_slots=("slot",))
    assert plan.actions == (RemovePiece(chess.D5, "slot"), MovePiece(chess.E5, chess.D6))


def test_planner_rejects_missing_resources_and_illegal_move():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    with pytest.raises(ValueError, match="reserve"):
        plan_move(board, chess.Move.from_uci("a7a8q"), capture_slots=("pawn",))
    with pytest.raises(ValueError, match="slot"):
        plan_move(board, chess.Move.from_uci("a7a8q"))
    with pytest.raises(ValueError, match="Illegal"):
        plan_move(board, chess.Move.from_uci("a7a6"))


@pytest.mark.parametrize("status", [ActionStatus.GRIP_FAILURE, ActionStatus.TIMEOUT, ActionStatus.EMERGENCY_STOP])
def test_failure_keeps_state_and_stop_is_latched(status):
    board = chess.Board()
    arm = MockArm(board)
    action = MovePiece(chess.E2, chess.E4)
    arm.fail_next(status)
    assert arm.execute(action).status == status
    assert arm.pieces == board.piece_map()
    if status is ActionStatus.EMERGENCY_STOP:
        assert arm.execute(action).status is status
        arm.reset_stop()
    assert arm.execute(action).success


def test_invalid_actions_do_not_change_state():
    board = chess.Board()
    arm = MockArm(board)
    assert arm.execute(MovePiece(chess.E2, chess.E7)).status is ActionStatus.INVALID
    assert arm.execute(PlaceReservePiece(chess.Piece.from_symbol("Q"), "missing", chess.E4)).status is ActionStatus.INVALID
    assert arm.pieces == board.piece_map()


def test_promotion_rejects_slot_collision():
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    with pytest.raises(ValueError, match="reserve"):
        plan_move(board, chess.Move.from_uci("a7a8q"), capture_slots=("same",),
                  reserve={"same": chess.Piece.from_symbol("Q")})
