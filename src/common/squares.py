FILES = "abcdefgh"
RANKS = "12345678"

ALL_SQUARES: tuple[str, ...] = tuple(f + r for r in RANKS for f in FILES)
ALL_SQUARES_SET: frozenset[str] = frozenset(ALL_SQUARES)


def is_valid_square(name: object) -> bool:
    return isinstance(name, str) and name in ALL_SQUARES_SET


def square_to_image_rc(square: str) -> tuple[int, int]:
    if not is_valid_square(square):
        raise ValueError(f"Некорректное имя клетки: {square!r}")
    col = FILES.index(square[0])
    row = 7 - RANKS.index(square[1])
    return row, col


def image_rc_to_square(row: int, col: int) -> str:
    for value, label in ((row, "row"), (col, "col")):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 7:
            raise ValueError(f"{label} должен быть целым числом от 0 до 7, получено {value!r}")
    return FILES[col] + RANKS[7 - row]
