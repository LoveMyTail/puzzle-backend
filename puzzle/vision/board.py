"""Board-state processing: segment the assembled region and locate gaps.

Implements the M2 milestone: given a photo of the partially assembled puzzle,
align it to the reference grid, decide which cells are filled, and extract
the "receiving edges" of every gap adjacent to filled cells.

A receiving edge is the contour of a filled neighbor that faces the gap
(including tabs/blanks protruding into it), not just the single boundary
line: the neighbor's shape is what the matcher compares against a piece side.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import cv2
import numpy as np

CELL_PX = 64
FILL_RATIO = 0.5
MIN_SOURCE_CELL_PX = 16  # default source-photo resolution floor (px per grid cell)
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


def source_cell_px(corners: Sequence[Point], rows: int, cols: int) -> float:
    """Pixels per grid cell that the marked board region provides in the photo.

    Derived from the quadrilateral's average side lengths, matching how the
    warped grid size is computed. The board photo is always warped onto the
    fixed ``CELL_PX`` grid, so the source photo must be at least as detailed or
    the receiving-edge shape signal is lost.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    tl, tr, br, bl = _ordered_points(corners)
    top_w = float(np.linalg.norm(tr - tl))
    bottom_w = float(np.linalg.norm(br - bl))
    left_h = float(np.linalg.norm(bl - tl))
    right_h = float(np.linalg.norm(br - tr))
    width = (top_w + bottom_w) / 2
    height = (left_h + right_h) / 2
    return min(width / cols, height / rows)


def validate_board_resolution(
    corners: Sequence[Point],
    rows: int,
    cols: int,
    min_cell_px: float | None = None,
) -> float:
    """Raise ``ValueError`` when the board photo is too low-res; else return px/cell.

    The threshold defaults to ``MIN_SOURCE_CELL_PX`` and can be overridden with
    the ``PUZZLE_MIN_CELL_PX`` environment variable (e.g. ``64`` for strict
    matching). Below it, the receiving-edge shape signal is too coarse for the
    matcher to separate the true gap reliably.
    """
    if min_cell_px is None:
        min_cell_px = float(os.environ.get("PUZZLE_MIN_CELL_PX", MIN_SOURCE_CELL_PX))
    px = source_cell_px(corners, rows, cols)
    if px < min_cell_px:
        raise ValueError(
            f"board photo resolution too low: about {px:.0f}px per cell, need at "
            f"least {min_cell_px:.0f}px; move the camera closer or use a "
            f"higher-resolution photo"
        )
    return px


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
    filled: np.ndarray | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Extract the filled neighbor contour facing a gap, per direction.

    For each filled neighbor, the receiving edge is the extreme contour of
    that neighbor within a window spanning the shared boundary (the neighbor
    cell plus half a cell into the gap). Sampling the extreme filled pixel
    per scan line preserves tabs/blanks protruding into the gap, which are
    the shape signal the matcher depends on.

    ``filled`` is the boolean cell grid from :func:`classify_cells`; only
    directions whose adjacent cell is filled produce an edge. When omitted it
    is derived locally from the neighbor cell's own fill ratio.
    """
    row = row_1based - 1
    col = col_1based - 1
    x0, y0 = col * cell_px, row * cell_px
    x1, y1 = x0 + cell_px, y0 + cell_px
    height, width = mask.shape[:2]
    edges: dict[str, list[tuple[int, int]]] = {}
    if row > 0 and _neighbor_filled(mask, row - 1, col, cell_px, filled):
        pts = _receiving_envelope(
            mask,
            rows=slice(max(0, y0 - cell_px), min(height, y0 + cell_px // 2)),
            cols=slice(x0, min(width, x1)),
            mode="bottom",
        )
        if pts:
            edges["top"] = pts
    if row < rows - 1 and _neighbor_filled(mask, row + 1, col, cell_px, filled):
        pts = _receiving_envelope(
            mask,
            rows=slice(max(0, y1 - cell_px // 2), min(height, y1 + cell_px)),
            cols=slice(x0, min(width, x1)),
            mode="top",
        )
        if pts:
            edges["bottom"] = pts
    if col > 0 and _neighbor_filled(mask, row, col - 1, cell_px, filled):
        pts = _receiving_envelope(
            mask,
            rows=slice(y0, min(height, y1)),
            cols=slice(max(0, x0 - cell_px), min(width, x0 + cell_px // 2)),
            mode="right",
        )
        if pts:
            edges["left"] = pts
    if col < cols - 1 and _neighbor_filled(mask, row, col + 1, cell_px, filled):
        pts = _receiving_envelope(
            mask,
            rows=slice(y0, min(height, y1)),
            cols=slice(max(0, x1 - cell_px // 2), min(width, x1 + cell_px)),
            mode="left",
        )
        if pts:
            edges["right"] = pts
    return edges


def _neighbor_filled(
    mask: np.ndarray,
    row: int,
    col: int,
    cell_px: int,
    filled: np.ndarray | None,
) -> bool:
    """Whether the neighbor cell is considered assembled."""
    if filled is not None:
        return bool(filled[row, col])
    cell = mask[
        row * cell_px : (row + 1) * cell_px,
        col * cell_px : (col + 1) * cell_px,
    ]
    return cell.size > 0 and (cell > 0).mean() > FILL_RATIO


def _receiving_envelope(
    mask: np.ndarray,
    rows: slice,
    cols: slice,
    mode: str,
) -> list[tuple[int, int]]:
    """Return the extreme filled pixels of the largest component in a window.

    ``mode`` selects the extreme to keep per scan line:
      - "bottom": deepest filled pixel per column (neighbor above a gap)
      - "top":    highest filled pixel per column (neighbor below a gap)
      - "right":  rightmost filled pixel per row (neighbor left of a gap)
      - "left":   leftmost filled pixel per row (neighbor right of a gap)

    Points are ordered along the boundary (x or y ascending), keeping the
    orientation convention of ``side_profile`` consistent with piece sides.
    """
    window = mask[rows, cols]
    if window.size == 0 or not np.any(window):
        return []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (window > 0).astype(np.uint8), connectivity=8
    )
    if num < 2:
        return []
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(labels == largest)
    if xs.size == 0:
        return []
    extremes: dict[int, int] = {}
    if mode in ("bottom", "top"):
        for x, y in zip(xs, ys):
            key = int(x)
            if key not in extremes:
                extremes[key] = int(y)
            elif mode == "bottom" and y > extremes[key]:
                extremes[key] = int(y)
            elif mode == "top" and y < extremes[key]:
                extremes[key] = int(y)
        return [
            (cols.start + x, rows.start + y) for x, y in sorted(extremes.items())
        ]
    for x, y in zip(xs, ys):
        key = int(y)
        if key not in extremes:
            extremes[key] = int(x)
        elif mode == "right" and x > extremes[key]:
            extremes[key] = int(x)
        elif mode == "left" and x < extremes[key]:
            extremes[key] = int(x)
    return [
        (cols.start + x, rows.start + y) for y, x in sorted(extremes.items())
    ]


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
