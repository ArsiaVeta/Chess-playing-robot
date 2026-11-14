import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.vision.board_transform import (
    CONFIG_PATH,
    CORNER_ORDER,
    check_corners,
    draw_grid,
    save_calibration,
    warp_board,
)

PHOTO_WINDOW = "calibrate: click a1 -> h1 -> h8 -> a8"
PREVIEW_WINDOW = "warped board"
MAX_DISPLAY = 1200


def _render(image: np.ndarray, points: list[tuple[float, float]], scale: float) -> np.ndarray:
    canvas = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1 else image.copy()
    for i, (x, y) in enumerate(points):
        p = (round(x * scale), round(y * scale))
        cv2.circle(canvas, p, 6, (0, 0, 255), -1)
        cv2.putText(canvas, CORNER_ORDER[i], (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if len(points) > 1:
        poly = np.array([(round(x * scale), round(y * scale)) for x, y in points], dtype=np.int32)
        cv2.polylines(canvas, [poly], len(points) == 4, (0, 255, 0), 1)
    if len(points) < 4:
        hint = f"click {CORNER_ORDER[len(points)]} ({len(points) + 1}/4), right click = undo"
    else:
        hint = "s = save, right click = undo, Esc = quit"
    cv2.putText(canvas, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(canvas, hint, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def run(image_path: Path, output: Path) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Не удалось открыть изображение: {image_path}", file=sys.stderr)
        return 1
    height, width = image.shape[:2]
    scale = min(1.0, MAX_DISPLAY / max(width, height))
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(PHOTO_WINDOW)
    cv2.setMouseCallback(PHOTO_WINDOW, on_mouse)
    shown_points = None
    preview_open = False
    try:
        while True:
            cv2.imshow(PHOTO_WINDOW, _render(image, points, scale))
            current = tuple(points) if len(points) == 4 else None
            if current != shown_points:
                if preview_open:
                    cv2.destroyWindow(PREVIEW_WINDOW)
                    preview_open = False
                if current is not None:
                    try:
                        cv2.imshow(PREVIEW_WINDOW, draw_grid(warp_board(image, current)))
                        preview_open = True
                    except ValueError as error:
                        print(error, file=sys.stderr)
                shown_points = current
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                print("Выход без сохранения")
                return 1
            if key == ord("s") and len(points) == 4:
                try:
                    check_corners(points)
                except ValueError as error:
                    print(error, file=sys.stderr)
                    continue
                save_calibration(points, (width, height), output)
                print(f"Калибровка сохранена: {output}")
                return 0
    finally:
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Калибровка доски по 4 углам: a1 -> h1 -> h8 -> a8")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()
    return run(args.image, args.output)


if __name__ == "__main__":
    sys.exit(main())
