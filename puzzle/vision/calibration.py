"""Reference-image calibration: grid estimation, perspective warp, cell slicing.

Implements the M1 milestone: turning a photo of the puzzle box cover into a
warped, sliced grid map that later stages use as the global coordinate system.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]


def estimate_grid(piece_count: int, width_cm: float, height_cm: float) -> tuple[int, int]:
    """Estimate ``(rows, cols)`` from the piece count and finished dimensions.

    The finished puzzle keeps the print's aspect ratio, so the grid is derived
    from ``rows * cols ~= piece_count`` and ``cols / rows ~= width / height``.
    The result is an estimate only; callers may let the user correct it.
    """
    if piece_count < 1:
        raise ValueError("piece_count must be positive")
    if width_cm <= 0 or height_cm <= 0:
        raise ValueError("width_cm and height_cm must be positive")

    rows = max(2, round(math.sqrt(piece_count * height_cm / width_cm)))
    cols = max(2, round(piece_count / rows))
    return rows, cols


def _ordered_points(corners: Sequence[Point]) -> np.ndarray:
    """Validate four corners in TL, TR, BR, BL order and return them as float32."""
    if len(corners) != 4:
        raise ValueError(
            "exactly four corners are required (top-left, top-right, bottom-right, bottom-left)"
        )
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("each corner must be an (x, y) pair")
    return points


def warp_box_photo(image: np.ndarray, corners: Sequence[Point]) -> np.ndarray:
    """Warp the box-cover photo so the four marked corners form a rectangle.

    The destination size is derived from the quadrilateral's average side
    lengths, so the corrected image keeps its true aspect ratio.
    """
    points = _ordered_points(corners)
    tl, tr, br, bl = points

    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    width = max(1, round((top_w + bottom_w) / 2))
    height = max(1, round((left_h + right_h) / 2))

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, dst)
    return cv2.warpPerspective(image, matrix, (width, height))


def slice_grid(warped: np.ndarray, rows: int, cols: int) -> list[np.ndarray]:
    """Split the warped reference image into ``rows * cols`` cell patches."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    height, width = warped.shape[:2]
    cell_h = height / rows
    cell_w = width / cols
    cells: list[np.ndarray] = []
    for row in range(rows):
        y0 = round(row * cell_h)
        y1 = round((row + 1) * cell_h)
        for col in range(cols):
            x0 = round(col * cell_w)
            x1 = round((col + 1) * cell_w)
            cells.append(warped[y0:y1, x0:x1].copy())
    return cells


def draw_grid_overlay(warped: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Return a copy of the warped image with the grid drawn on top."""
    preview = warped.copy()
    height, width = preview.shape[:2]
    for row in range(1, rows):
        y = round(row * height / rows)
        cv2.line(preview, (0, y), (width - 1, y), (0, 255, 0), 2)
    for col in range(1, cols):
        x = round(col * width / cols)
        cv2.line(preview, (x, 0), (x, height - 1), (0, 255, 0), 2)
    return preview
