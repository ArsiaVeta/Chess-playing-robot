import pytest

from src.common.squares import (
    ALL_SQUARES,
    image_rc_to_square,
    is_valid_square,
    square_to_image_rc,
)


def test_all_squares_has_64_unique_names():
    assert len(ALL_SQUARES) == 64
    assert len(set(ALL_SQUARES)) == 64
    assert ALL_SQUARES[0] == "a1"
    assert ALL_SQUARES[-1] == "h8"


@pytest.mark.parametrize(
    "square, rc",
    [
        ("a1", (7, 0)),
        ("h1", (7, 7)),
        ("a8", (0, 0)),
        ("h8", (0, 7)),
        ("e4", (4, 4)),
        ("e2", (6, 4)),
        ("d5", (3, 3)),
    ],
)
def test_known_squares(square, rc):
    assert square_to_image_rc(square) == rc
    assert image_rc_to_square(*rc) == square


def test_round_trip_all_squares():
    for square in ALL_SQUARES:
        assert image_rc_to_square(*square_to_image_rc(square)) == square


def test_round_trip_all_cells():
    seen = set()
    for row in range(8):
        for col in range(8):
            square = image_rc_to_square(row, col)
            assert square_to_image_rc(square) == (row, col)
            seen.add(square)
    assert len(seen) == 64


def test_a1_and_h8_are_opposite_corners():
    assert square_to_image_rc("a1") != square_to_image_rc("h8")
    assert square_to_image_rc("a1") == (7, 0)


@pytest.mark.parametrize("bad", ["", "e", "e9", "i1", "E4", "e44", "a0", None, 5])
def test_invalid_square_names(bad):
    assert not is_valid_square(bad)
    with pytest.raises(ValueError):
        square_to_image_rc(bad)


@pytest.mark.parametrize("row, col", [(-1, 0), (8, 0), (0, 8), (0, -1), (1.0, 2), (True, 0)])
def test_invalid_image_rc(row, col):
    with pytest.raises(ValueError):
        image_rc_to_square(row, col)
