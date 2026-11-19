import json
from pathlib import Path

import cv2
import numpy as np

from src.common.squares import ALL_SQUARES, image_rc_to_square, square_to_image_rc

CORNER_ORDER = ("a1", "h1", "h8", "a8")
DEFAULT_SIZE = 800
DEFAULT_MARGIN = 0.15
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "board.json"


def check_corners(corners) -> np.ndarray:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Нужно ровно 4 угла (x, y) в порядке {CORNER_ORDER}, получено {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("Координаты углов должны быть конечными числами")
    contour = points.reshape(-1, 1, 2)
    if cv2.contourArea(contour) < 100 or not cv2.isContourConvex(contour):
        raise ValueError(
            f"Углы не образуют выпуклый четырёхугольник: проверь порядок {CORNER_ORDER}"
        )
    return points


def warp_board(image: np.ndarray, corners, size: int = DEFAULT_SIZE) -> np.ndarray:
    src = check_corners(corners)
    dst = np.array([[0, size], [size, size], [size, 0], [0, 0]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def _cell_bounds(size: int, row: int, col: int, margin: float) -> tuple[int, int, int, int]:
    cell = size / 8
    pad = cell * margin
    y0 = round(row * cell + pad)
    y1 = round((row + 1) * cell - pad)
    x0 = round(col * cell + pad)
    x1 = round((col + 1) * cell - pad)
    return y0, y1, x0, x1


def split_cells(warped: np.ndarray, margin: float = DEFAULT_MARGIN) -> dict[str, np.ndarray]:
    height, width = warped.shape[:2]
    if height != width:
        raise ValueError(f"Выпрямленная доска должна быть квадратной, получено {width}x{height}")
    if not 0 <= margin < 0.5:
        raise ValueError(f"margin должен быть в [0, 0.5), получено {margin}")
    cells = {}
    for square in ALL_SQUARES:
        row, col = square_to_image_rc(square)
        y0, y1, x0, x1 = _cell_bounds(height, row, col, margin)
        cells[square] = warped[y0:y1, x0:x1]
    return cells


def draw_grid(warped: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR) if warped.ndim == 2 else warped.copy()
    size = canvas.shape[0]
    cell = size / 8
    for i in range(9):
        p = round(i * cell)
        cv2.line(canvas, (p, 0), (p, size), (0, 255, 0), 1)
        cv2.line(canvas, (0, p), (size, p), (0, 255, 0), 1)
    scale = cell / 150
    for row in range(8):
        for col in range(8):
            org = (round(col * cell + cell * 0.08), round(row * cell + cell * 0.3))
            label = image_rc_to_square(row, col)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_calibration(corners, image_size: tuple[int, int], path: Path = CONFIG_PATH) -> None:
    points = check_corners(corners)
    data = {
        "corner_order": list(CORNER_ORDER),
        "corners": points.tolist(),
        "image_size": [int(image_size[0]), int(image_size[1])],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: Path = CONFIG_PATH) -> tuple[np.ndarray, tuple[int, int]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла калибровки {path}. Сначала запусти: python -m src.vision.calibrate <фото>"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("corner_order") != list(CORNER_ORDER):
        raise ValueError(f"Неожиданный порядок углов в {path}: {data.get('corner_order')}")
    width, height = data["image_size"]
    return check_corners(data["corners"]), (int(width), int(height))
