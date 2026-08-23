"""Board-state processing: segment the assembled region and locate gaps.

Implements the M2 milestone: given a photo of the partially assembled puzzle,
align it to the reference grid, decide which cells are filled, and extract
the "receiving edges" of every gap adjacent to filled cells.

A receiving edge is the contour of a filled neighbor that faces the gap
(including tabs/blanks protruding into it), not just the single boundary
line: the neighbor's shape is what the matcher compares against a piece side.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import cv2
import numpy as np

logger = logging.getLogger("puzzle.board")

CELL_PX = 64
FILL_RATIO = 0.5
MIN_SOURCE_CELL_PX = 16  # default source-photo resolution floor (px per grid cell)
Point = tuple[float, float]


def adaptive_cell_px(rows: int, cols: int) -> int:
    """Pick a grid cell size that keeps the warped board within ~1536px.

    Small puzzles (e.g. 48 pieces) get a much higher grid resolution than the
    fixed ``CELL_PX``, preserving the tab/blank detail of receiving edges;
    large puzzles fall back to ``CELL_PX`` so the warped image stays small
    enough to process in memory.
    """
    if rows < 1 or cols < 1:
        return CELL_PX
    return max(CELL_PX, min(192, 1536 // max(rows, cols)))


def segment_board(
    image: np.ndarray,
    border_margin: int = 8,
    color_distance: float = 25.0,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """Return a binary mask of puzzle pieces on a contrasting background.

    The background color is sampled as the median of the image border (or a
    caller-provided ``background`` color, e.g. sampled outside the marked
    puzzle quadrilateral), and any pixel far enough from it is treated as
    part of the assembled puzzle. The distance threshold is chosen with Otsu
    (adaptive to lighting and textured backgrounds such as tissue paper),
    floored at ``color_distance``, and the mask is cleaned with small
    morphological operations to remove background speckle.
    """
    if image.ndim != 3:
        raise ValueError("board photo must be a color image")
    if background is None:
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
    else:
        bg = np.asarray(background, dtype=np.float32).reshape(3)
    dist = np.linalg.norm(image.astype(np.float32) - bg.astype(np.float32), axis=2)
    dist_u8 = np.clip(dist, 0, 255).astype(np.uint8)
    otsu_threshold, _ = cv2.threshold(
        dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    threshold = max(int(otsu_threshold), int(color_distance))
    mask = (dist > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.medianBlur(mask, 5)
    return mask


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


def cell_region_in_photo(
    corners: Sequence[Point],
    rows: int,
    cols: int,
    cell_px: int,
    photo_shape: tuple[int, int] | None,
    row_1based: int,
    col_1based: int,
    expand_ratio: float = 0.5,
) -> dict | None:
    """Return the normalized rect of a grid cell mapped back to the photo.

    ``align_board_to_grid`` warps the uploaded board photo onto a fixed grid
    with ``cv2.getPerspectiveTransform``; mapping the target cell rectangle
    back through the inverse homography yields its pixel location in the
    original photo. The rectangle is expanded by ``expand_ratio`` of the cell
    size on each side (clamped to the board) so a crop includes neighboring
    pieces as spatial context, then normalized to the photo dimensions and
    clipped to [0, 1].

    ``photo_shape`` is the OpenCV image shape ``(height, width)`` of the
    original board photo. Returns ``None`` when the geometry cannot be
    computed (missing/invalid input or a singular homography); callers should
    degrade to the plain row/column display in that case.
    """
    if photo_shape is None or len(photo_shape) < 2:
        return None
    photo_h, photo_w = int(photo_shape[0]), int(photo_shape[1])
    if photo_h < 1 or photo_w < 1 or rows < 1 or cols < 1 or cell_px < 1:
        return None
    try:
        points = _ordered_points(corners)
        grid_w = cols * cell_px
        grid_h = rows * cell_px
        dst = np.array(
            [[0, 0], [grid_w - 1, 0], [grid_w - 1, grid_h - 1], [0, grid_h - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, dst)
        inverse = np.linalg.inv(matrix)

        row = row_1based - 1
        col = col_1based - 1
        if row < 0 or col < 0 or row >= rows or col >= cols:
            return None
        pad = cell_px * expand_ratio
        x0 = max(0.0, col * cell_px - pad)
        y0 = max(0.0, row * cell_px - pad)
        x1 = min(float(grid_w), (col + 1) * cell_px + pad)
        y1 = min(float(grid_h), (row + 1) * cell_px + pad)
        src = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src.reshape(1, -1, 2), inverse).reshape(-1, 2)
        xs = mapped[:, 0]
        ys = mapped[:, 1]
        x = float(np.clip(xs.min() / photo_w, 0.0, 1.0))
        y = float(np.clip(ys.min() / photo_h, 0.0, 1.0))
        w = float(np.clip(xs.max() / photo_w, 0.0, 1.0)) - x
        h = float(np.clip(ys.max() / photo_h, 0.0, 1.0)) - y
        if w <= 0.0 or h <= 0.0:
            return None
        return {"x": x, "y": y, "w": w, "h": h}
    except (ValueError, TypeError, cv2.error, np.linalg.LinAlgError):
        return None


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
    background: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp the board photo and its piece mask into grid-pixel space.

    ``background`` is an optional BGR color for segmentation; when omitted it
    is sampled from the four image corners, which is more robust than the full
    border when the puzzle reaches the photo edges.
    """
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
    if background is None:
        background = _corner_background(image)
    mask = segment_board(image, background=background)
    warped = cv2.warpPerspective(image, matrix, (grid_w, grid_h))
    warped_mask = cv2.warpPerspective(mask, matrix, (grid_w, grid_h), flags=cv2.INTER_NEAREST)
    logger.debug(
        "board_aligned rows=%d cols=%d grid=%dx%d mask_coverage=%.3f "
        "background=%s",
        rows,
        cols,
        grid_w,
        grid_h,
        float((warped_mask > 0).mean()),
        "provided" if background is not None else "border-median",
    )
    return warped, warped_mask


def _corner_background(image: np.ndarray) -> np.ndarray:
    """Median BGR color of the four image corner patches (table hypothesis)."""
    height, width = image.shape[:2]
    margin = max(8, min(height, width) // 40)
    patches = np.concatenate(
        (
            image[:margin, :margin].reshape(-1, 3),
            image[:margin, width - margin :].reshape(-1, 3),
            image[height - margin :, :margin].reshape(-1, 3),
            image[height - margin :, width - margin :].reshape(-1, 3),
        )
    )
    return np.median(patches, axis=0).astype(np.float32)


def exterior_background(
    image: np.ndarray, corners: Sequence[Point], strip: int = 12
) -> np.ndarray | None:
    """Median BGR sampled just outside the puzzle quadrilateral.

    Once the user (or auto-detection) marks the assembled-puzzle area, the
    table around it is the real background; sampling a thin strip just
    outside each side of the quadrilateral is far more reliable than the
    image border, which may itself contain puzzle. Returns ``None`` when the
    quadrilateral touches the photo edge (nothing to sample).
    """
    height, width = image.shape[:2]
    if height < 3 or width < 3:
        return None
    try:
        points = _ordered_points(corners)
    except ValueError:
        return None
    edges = [
        (points[0], points[1]),
        (points[1], points[2]),
        (points[2], points[3]),
        (points[3], points[0]),
    ]
    samples: list[np.ndarray] = []
    for (a, b) in edges:
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue
        nx, ny = -dy / length, dx / length  # exterior of a clockwise quad
        for t in (0.2, 0.5, 0.8):
            x = int(round(a[0] + dx * t + nx * strip))
            y = int(round(a[1] + dy * t + ny * strip))
            if 0 <= x < width and 0 <= y < height:
                samples.append(image[y, x])
    if not samples:
        return None
    return np.median(np.asarray(samples, dtype=np.float32), axis=0)


def is_full_frame_corners(
    corners: Sequence[Point], image_shape: tuple[int, int]
) -> bool:
    """Whether the caller's corners span (nearly) the whole photo.

    The app uploads full-frame corners when the user has not marked the
    puzzle quadrilateral; such corners force the whole photo onto the grid
    and make alignment fail on real photos. The backend then tries
    :func:`estimate_board_quad` instead.
    """
    height, width = image_shape[:2]
    if height < 1 or width < 1 or len(corners) != 4:
        return True
    try:
        area = float(cv2.contourArea(np.asarray(corners, dtype=np.float32)))
    except (TypeError, ValueError):
        return True
    return area >= 0.97 * height * width


def estimate_board_quad(image: np.ndarray) -> np.ndarray | None:
    """Detect the assembled-puzzle quadrilateral in a board photo.

    Assembled puzzle regions contain dense piece seams, so the detector
    thresholds local Canny edge density, keeps the largest connected region,
    and fits a minimum-area rectangle to it. Returns four corners in
    TL, TR, BR, BL order in original photo pixels, or ``None`` when no
    distinct block is found (e.g. the puzzle fills the whole frame).
    """
    if image is None or image.ndim != 3:
        return None
    height, width = image.shape[:2]
    if height < 64 or width < 64:
        return None
    small = cv2.resize(
        image,
        (min(width, 1600), min(height, 1600)),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape[:2]
    scale_x = width / sw
    scale_y = height / sh

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 150)
    density = cv2.boxFilter((edges > 0).astype(np.float32), cv2.CV_32F, (64, 64))
    region = (density > 0.05).astype(np.uint8) * 255
    region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    region = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    num, labels, stats, _ = cv2.connectedComponentsWithStats(region, connectivity=8)
    if num < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))
    if areas[best - 1] < 0.03 * sh * sw:
        return None
    component = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box[:, 0] = np.clip(box[:, 0] * scale_x, 0, width - 1)
    box[:, 1] = np.clip(box[:, 1] * scale_y, 0, height - 1)
    return _order_quad(box)


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as TL, TR, BR, BL."""
    ordered = np.asarray(points, dtype=np.float64).reshape(4, 2)
    top = ordered[np.argsort(ordered[:, 1])[:2]]
    bottom = ordered[np.argsort(ordered[:, 1])[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.asarray([tl, tr, br, bl], dtype=np.float64)


def alignment_quality(
    warped: np.ndarray,
    mask: np.ndarray,
    rows: int,
    cols: int,
    cell_px: int = CELL_PX,
) -> dict:
    """Produce an informational alignment/segmentation diagnostic.

    The edge-energy ratio and ambiguous-cell share are reported for
    debugging, but they are not a reliable gate on real photos: puzzle
    artwork is often textured (ratio stays near 1 even when aligned) and
    early-stage boards are mostly empty (nothing to align yet). ``ok`` is
    therefore only false when the photo clearly fails segmentation
    (``low_contrast``) or the evidence strongly suggests a broken input
    (``poor``); ``too_empty`` boards are treated as fine so the app does not
    nag users who simply have not assembled much yet.
    """
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    boundary_energy: list[float] = []
    interior_energy: list[float] = []
    ambiguous = 0
    total = rows * cols
    for row in range(rows):
        for col in range(cols):
            x0, y0 = col * cell_px, row * cell_px
            if col < cols - 1:
                boundary_energy.append(
                    float(magnitude[y0 : y0 + cell_px, x0 + cell_px].mean())
                )
            if row < rows - 1:
                boundary_energy.append(
                    float(magnitude[y0 + cell_px, x0 : x0 + cell_px].mean())
                )
            cell = mask[y0 : y0 + cell_px, x0 : x0 + cell_px]
            fill = float((cell > 0).mean())
            if 0.2 < fill < 0.8:
                ambiguous += 1
            interior = magnitude[
                y0 + cell_px // 4 : y0 + 3 * cell_px // 4,
                x0 + cell_px // 4 : x0 + 3 * cell_px // 4,
            ]
            interior_energy.append(float(interior.mean()))
    boundary = float(np.mean(boundary_energy)) if boundary_energy else 0.0
    interior = float(np.mean(interior_energy)) if interior_energy else 1.0
    ratio = boundary / max(interior, 1e-6)
    ambiguous_fraction = ambiguous / max(total, 1)
    coverage = float((mask > 0).mean())
    filled_cells = 0
    for row in range(rows):
        for col in range(cols):
            cell = mask[
                row * cell_px : (row + 1) * cell_px,
                col * cell_px : (col + 1) * cell_px,
            ]
            if float((cell > 0).mean()) >= 0.5:
                filled_cells += 1
    if coverage < 0.05:
        note = "low_contrast"
        ok = False
    elif filled_cells < 3:
        note = "too_empty"
        ok = True
    elif ratio < 0.75 or ambiguous_fraction > 0.6:
        note = "poor"
        ok = False
    else:
        note = "ok"
        ok = True
    return {
        "boundary_ratio": round(ratio, 2),
        "ambiguous_cells": round(ambiguous_fraction, 3),
        "mask_coverage": round(coverage, 3),
        "filled_cells": filled_cells,
        "note": note,
        "ok": ok,
    }


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
        ordered = sorted(extremes.items())
        return _smooth_envelope(ordered, cols.start, rows.start, horizontal=True)
    for x, y in zip(xs, ys):
        key = int(y)
        if key not in extremes:
            extremes[key] = int(x)
        elif mode == "right" and x > extremes[key]:
            extremes[key] = int(x)
        elif mode == "left" and x < extremes[key]:
            extremes[key] = int(x)
    ordered = sorted(extremes.items())
    return _smooth_envelope(ordered, cols.start, rows.start, horizontal=False)


def _smooth_envelope(
    ordered: list[tuple[int, int]],
    x_start: int,
    y_start: int,
    horizontal: bool,
) -> list[tuple[int, int]]:
    """Median-filter the per-scan-line extremes along the boundary.

    Single-pixel mask noise creates spikes in the receiving edge; replacing
    each extreme with the median of a 3-point window makes the envelope
    follow the real piece boundary instead of isolated mask artifacts.
    """
    if len(ordered) < 3:
        return [(x_start + x, y_start + y) for x, y in ordered]
    keys = [item[0] for item in ordered]
    values = [item[1] for item in ordered]
    smoothed: list[tuple[int, int]] = []
    for i in range(len(ordered)):
        window = values[max(0, i - 1) : i + 2]
        value = int(np.median(window))
        if horizontal:
            smoothed.append((x_start + keys[i], y_start + value))
        else:
            smoothed.append((x_start + value, y_start + keys[i]))
    return smoothed


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


_CANDIDATE_COLORS = [(60, 220, 90), (0, 165, 255), (0, 60, 255)]  # BGR: green, orange, red


def render_candidate_preview(
    warped: np.ndarray,
    candidates: list[dict],
    cell_px: int = CELL_PX,
) -> np.ndarray:
    """Return the warped board with the top candidates' gap cells marked 1..k.

    ``candidates`` must be ordered best-first (as returned by
    :func:`puzzle.solver.matcher.top_candidates`); each dict needs 1-based
    ``row``/``col``. This is the visual answer for the app: "your piece most
    likely goes in the highlighted hole".
    """
    height, width = warped.shape[:2]
    base = warped.copy()
    for rank, candidate in enumerate(candidates, start=1):
        x0 = max(0, (candidate["col"] - 1) * cell_px)
        y0 = max(0, (candidate["row"] - 1) * cell_px)
        x1 = min(width - 1, x0 + cell_px - 1)
        y1 = min(height - 1, y0 + cell_px - 1)
        color = _CANDIDATE_COLORS[(rank - 1) % len(_CANDIDATE_COLORS)]
        cv2.rectangle(base, (x0, y0), (x1, y1), color, -1)
    overlay = cv2.addWeighted(base, 0.4, warped, 0.6, 0)
    for rank, candidate in enumerate(candidates, start=1):
        x0 = max(0, (candidate["col"] - 1) * cell_px)
        y0 = max(0, (candidate["row"] - 1) * cell_px)
        x1 = min(width - 1, x0 + cell_px - 1)
        y1 = min(height - 1, y0 + cell_px - 1)
        color = _CANDIDATE_COLORS[(rank - 1) % len(_CANDIDATE_COLORS)]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 3)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        cv2.putText(
            overlay, str(rank), (cx - 10, cy + 12), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (0, 0, 0), 5, cv2.LINE_AA,
        )
        cv2.putText(
            overlay, str(rank), (cx - 10, cy + 12), cv2.FONT_HERSHEY_SIMPLEX,
            1.2, (255, 255, 255), 2, cv2.LINE_AA,
        )
    return overlay
