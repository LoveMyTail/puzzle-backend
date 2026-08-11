"""Board-state processing: segment the assembled region and locate gaps.

Implements the M2 milestone: given a photo of the partially assembled puzzle,
align it to the reference grid, decide which cells are filled, and extract
the "receiving edges" of every gap adjacent to filled cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

CELL_PX = 24
FILL_RATIO = 0.5
Point = tuple[float, float]


def segment_board(
    image: np.ndarray,
    border_margin: int = 8,
    color_distance: float = 40.0,
) -> np.ndarray:
    """Return a binary mask of puzzle pieces on a contrasting background.

    The background color is sampled as the median of the image border, and any
    pixel far enough from it is treated as part of the assembled puzzle.
    """
    if image.ndim != 3:
        raise ValueError("board photo must be a color image")
    height, width = image.shape[:2]
    margin = min(border_margin, height // 4, width // 4)
    border = np.concatenate(
        (
            image[:margin].reshape(-1, 3),
            image[height - margin :].reshape(-1, 3),
            image[:, :margin].reshape(-1, 3),
            image[:, width - margin :].reshape(-1, 3),
        )
    )
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(image.astype(np.float32) - bg.astype(np.float32), axis=2)
    return (dist > color_distance).astype(np.uint8)


def _ordered_points(corners: Sequence[Point]) -> np.ndarray:
    if len(corners) != 4:
        raise ValueError("exactly four corners are required (TL, TR, BR, BL)")
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError("each corner must be an (x, y) pair")
    return points


def align_board_to_grid(
    image: np.ndarray,
    corners: Sequence[Point],
    rows: int,
    cols: int,
    cell_px: int = CELL_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the board photo and its piece mask into grid-pixel space."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    points = _ordered_points(corners)
    grid_w = cols * cell_px
    grid_h = rows * cell_px
    dst = np.array(
        [[0, 0], [grid_w - 1, 0], [grid_w - 1, grid_h - 1], [0, grid_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(points, dst)
    mask = segment_board(image)
    warped = cv2.warpPerspective(image, matrix, (grid_w, grid_h))
    warped_mask = cv2.warpPerspective(mask, matrix, (grid_w, grid_h), flags=cv2.INTER_NEAREST)
    return warped, warped_mask


def classify_cells(
    mask: np.ndarray,
    rows: int,
    cols: int,
    cell_px: int = CELL_PX,
    fill_ratio: float = FILL_RATIO,
) -> np.ndarray:
    """Return a boolean grid marking cells whose area is mostly covered."""
    filled = np.zeros((rows, cols), dtype=bool)
    for row in range(rows):
        for col in range(cols):
            cell = mask[row * cell_px : (row + 1) * cell_px, col * cell_px : (col + 1) * cell_px]
            filled[row, col] = (cell > 0).mean() >= fill_ratio
    return filled


def find_gaps(filled: np.ndarray) -> list[dict]:
    """Return empty cells that touch at least one filled cell (1-based rows/cols)."""
    rows, cols = filled.shape
    gaps: list[dict] = []
    for row in range(rows):
        for col in range(cols):
            if filled[row, col]:
                continue
            directions = []
            if row > 0 and filled[row - 1, col]:
                directions.append("top")
            if row < rows - 1 and filled[row + 1, col]:
                directions.append("bottom")
            if col > 0 and filled[row, col - 1]:
                directions.append("left")
            if col < cols - 1 and filled[row, col + 1]:
                directions.append("right")
            if directions:
                gaps.append({"row": row + 1, "col": col + 1, "directions": directions})
    return gaps


def extract_receiving_edges(
    mask: np.ndarray,
    rows: int,
    cols: int,
    row_1based: int,
    col_1based: int,
    cell_px: int = CELL_PX,
) -> dict[str, list[tuple[int, int]]]:
    """Extract the filled neighbor contour facing a gap, per direction."""
    row = row_1based - 1
    col = col_1based - 1
    x0, y0 = col * cell_px, row * cell_px
    x1, y1 = x0 + cell_px, y0 + cell_px
    edges: dict[str, list[tuple[int, int]]] = {}
    if row > 0:
        pts = [(x, y0 - 1) for x in range(x0, x1) if mask[y0 - 1, x] > 0]
        if pts:
            edges["top"] = pts
    if row < rows - 1:
        pts = [(x, y1) for x in range(x0, x1) if mask[y1, x] > 0]
        if pts:
            edges["bottom"] = pts
    if col > 0:
        pts = [(x0 - 1, y) for y in range(y0, y1) if mask[y, x0 - 1] > 0]
        if pts:
            edges["left"] = pts
    if col < cols - 1:
        pts = [(x1, y) for y in range(y0, y1) if mask[y, x1] > 0]
        if pts:
            edges["right"] = pts
    return edges


def render_gap_preview(
    warped: np.ndarray,
    filled: np.ndarray,
    gaps: list[dict],
    rows: int,
    cols: int,
    cell_px: int = CELL_PX,
) -> np.ndarray:
    """Return a preview tinting filled cells green and gaps red."""
    overlay = warped.copy()
    for row in range(rows):
        for col in range(cols):
            if not filled[row, col]:
                continue
            x0, y0 = col * cell_px, row * cell_px
            cv2.rectangle(
                overlay,
                (x0, y0),
                (x0 + cell_px - 1, y0 + cell_px - 1),
                (60, 200, 120),
                -1,
            )
    for gap in gaps:
        x0 = (gap["col"] - 1) * cell_px
        y0 = (gap["row"] - 1) * cell_px
        cv2.rectangle(overlay, (x0, y0), (x0 + cell_px - 1, y0 + cell_px - 1), (60, 60, 230), -1)
    return cv2.addWeighted(overlay, 0.45, warped, 0.55, 0)
