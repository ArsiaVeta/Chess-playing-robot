from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from src.common.squares import ALL_SQUARES_SET, is_valid_square


@dataclass(frozen=True)
class BoardObservation:
    occupied: Mapping[str, bool]
    confidence: Mapping[str, float]
    stable: bool
    changed_squares: frozenset[str] = field(default_factory=frozenset)
    frame_id: int = 0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        occupied = _check_full_board(self.occupied, "occupied")
        for square, value in occupied.items():
            if not isinstance(value, bool):
                raise TypeError(f"occupied[{square!r}] должен быть bool, получено {value!r}")

        confidence = _check_full_board(self.confidence, "confidence")
        for square, value in confidence.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"confidence[{square!r}] должен быть числом, получено {value!r}")
            if math.isnan(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"confidence[{square!r}] должен быть в [0, 1], получено {value!r}")

        if not isinstance(self.stable, bool):
            raise TypeError(f"stable должен быть bool, получено {self.stable!r}")

        if isinstance(self.changed_squares, str) or not isinstance(self.changed_squares, Iterable):
            raise TypeError("changed_squares должен быть набором имён клеток")
        changed = frozenset(self.changed_squares)
        bad = sorted(str(s) for s in changed if not is_valid_square(s))
        if bad:
            raise ValueError(f"changed_squares содержит некорректные клетки: {bad}")

        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError(f"frame_id должен быть целым >= 0, получено {self.frame_id!r}")
        if isinstance(self.timestamp, bool) or not isinstance(self.timestamp, (int, float)):
            raise TypeError(f"timestamp должен быть числом, получено {self.timestamp!r}")

        object.__setattr__(self, "occupied", MappingProxyType(dict(occupied)))
        object.__setattr__(
            self, "confidence", MappingProxyType({s: float(v) for s, v in confidence.items()})
        )
        object.__setattr__(self, "changed_squares", changed)
        object.__setattr__(self, "timestamp", float(self.timestamp))

    @property
    def occupied_squares(self) -> frozenset[str]:
        return frozenset(s for s, v in self.occupied.items() if v)


def _check_full_board(data: object, name: str) -> Mapping:
    if not isinstance(data, Mapping):
        raise TypeError(f"{name} должен быть словарём клетка -> значение")
    keys = set(data.keys())
    if keys != ALL_SQUARES_SET:
        missing = sorted(ALL_SQUARES_SET - keys)
        extra = sorted(str(k) for k in keys - ALL_SQUARES_SET)
        raise ValueError(
            f"{name} должен содержать ровно 64 клетки a1..h8; "
            f"не хватает: {missing}, лишние: {extra}"
        )
    return data
