import cv2
import numpy as np
import pytest

from src.common.squares import ALL_SQUARES, square_to_image_rc
from src.vision.board_transform import (
    check_corners,
    draw_grid,
    load_calibration,
    save_calibration,
    split_cells,
    warp_board,
)

FLAT = 800
CELL = FLAT // 8
FLAT_CORNERS = np.array([[0, FLAT], [FLAT, FLAT], [FLAT, 0], [0, 0]], dtype=np.float32)


def make_flat_board(marked: str) -> np.ndarray:
    board = np.zeros((FLAT, FLAT, 3), dtype=np.uint8)
    for square in ALL_SQUARES:
        row, col = square_to_image_rc(square)
        color = (90, 120, 160) if (row + col) % 2 else (200, 220, 235)
        board[row * CELL:(row + 1) * CELL, col * CELL:(col + 1) * CELL] = color
    row, col = square_to_image_rc(marked)
    center = (col * CELL + CELL // 2, row * CELL + CELL // 2)
    cv2.circle(board, center, CELL // 3, (0, 0, 255), -1)
    return board


def photograph(flat: np.ndarray, photo_corners) -> np.ndarray:
    matrix = cv2.getPerspectiveTransform(FLAT_CORNERS, np.asarray(photo_corners, dtype=np.float32))
    return cv2.warpPerspective(flat, matrix, (1000, 800))


def red_squares(warped: np.ndarray) -> set[str]:
    found = set()
    for square, cell in split_cells(warped).items():
        b, g, r = cell.reshape(-1, 3).mean(axis=0)
        if r - b > 80:
            found.add(square)
    return found


WHITE_SIDE = [(180, 700), (850, 680), (720, 150), (260, 170)]
BLACK_SIDE = [(720, 150), (260, 170), (180, 700), (850, 680)]


@pytest.mark.parametrize("marked", ["e4", "a1", "h8", "b7"])
def test_marker_lands_in_right_square_white_side(marked):
    photo = photograph(make_flat_board(marked), WHITE_SIDE)
    assert red_squares(warp_board(photo, WHITE_SIDE)) == {marked}


@pytest.mark.parametrize("marked", ["e4", "a1", "h8", "b7"])
def test_marker_lands_in_right_square_black_side(marked):
    photo = photograph(make_flat_board(marked), BLACK_SIDE)
    assert red_squares(warp_board(photo, BLACK_SIDE)) == {marked}


def test_a1_is_bottom_left_after_warp():
    photo = photograph(make_flat_board("a1"), BLACK_SIDE)
    warped = warp_board(photo, BLACK_SIDE)
    bottom_left = warped[FLAT - CELL:, :CELL].reshape(-1, 3).mean(axis=0)
    assert bottom_left[2] - bottom_left[0] > 80


def test_warp_output_size():
    photo = photograph(make_flat_board("e4"), WHITE_SIDE)
    assert warp_board(photo, WHITE_SIDE, size=400).shape == (400, 400, 3)


def test_split_cells_returns_64_equal_cells():
    cells = split_cells(np.zeros((800, 800, 3), dtype=np.uint8), margin=0.15)
    assert set(cells) == set(ALL_SQUARES)
    assert {c.shape for c in cells.values()} == {(70, 70, 3)}


def test_split_cells_without_margin_covers_cell():
    cells = split_cells(np.zeros((800, 800), dtype=np.uint8), margin=0)
    assert {c.shape for c in cells.values()} == {(100, 100)}


def test_split_cells_rejects_non_square():
    with pytest.raises(ValueError):
        split_cells(np.zeros((800, 600, 3), dtype=np.uint8))


@pytest.mark.parametrize("margin", [-0.1, 0.5, 0.7])
def test_split_cells_rejects_bad_margin(margin):
    with pytest.raises(ValueError):
        split_cells(np.zeros((800, 800, 3), dtype=np.uint8), margin=margin)


@pytest.mark.parametrize(
    "corners",
    [
        [(0, 0), (10, 0), (10, 10)],
        [(0, 0), (0, 0), (0, 0), (0, 0)],
        [(0, 0), (100, 0), (200, 0), (300, 0)],
        [(180, 700), (720, 150), (850, 680), (260, 170)],
        [(0, 0), (100, 0), (100, float("nan")), (0, 100)],
    ],
)
def test_bad_corners_rejected(corners):
    with pytest.raises(ValueError):
        check_corners(corners)


def test_draw_grid_keeps_input_and_size():
    warped = np.zeros((800, 800, 3), dtype=np.uint8)
    grid = draw_grid(warped)
    assert grid.shape == warped.shape
    assert not warped.any()
    assert grid.any()


def test_draw_grid_accepts_grayscale():
    assert draw_grid(np.zeros((400, 400), dtype=np.uint8)).shape == (400, 400, 3)


def test_calibration_round_trip(tmp_path):
    path = tmp_path / "board.json"
    save_calibration(WHITE_SIDE, (1000, 800), path)
    corners, image_size = load_calibration(path)
    assert np.allclose(corners, WHITE_SIDE)
    assert image_size == (1000, 800)


def test_missing_calibration_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibrate"):
        load_calibration(tmp_path / "nope.json")
