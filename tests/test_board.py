"""Unit tests for board-state processing (M2)."""

import math

import cv2
import numpy as np

from puzzle.solver.matcher import edge_profile
from puzzle.vision.board import (
    CELL_PX,
    align_board_to_grid,
    classify_cells,
    extract_receiving_edges,
    find_gaps,
    render_gap_preview,
    segment_board,
)


def test_segment_board_detects_pieces() -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (120, 120), (20, 120, 200), -1)
    mask = segment_board(image)
    assert mask[80, 80] > 0
    assert mask[5, 5] == 0


def test_find_gaps_around_filled_block() -> None:
    filled = np.zeros((5, 5), dtype=bool)
    filled[:2, :2] = True
    gaps = find_gaps(filled)
    positions = {(gap["row"], gap["col"]) for gap in gaps}
    assert positions == {(3, 1), (3, 2), (1, 3), (2, 3)}
    by_pos = {(gap["row"], gap["col"]): gap["directions"] for gap in gaps}
    assert "top" in by_pos[(3, 1)]
    assert "left" in by_pos[(1, 3)]


def _filled_mask(rows: int = 10, cols: int = 12) -> np.ndarray:
    mask = np.zeros((rows * CELL_PX, cols * CELL_PX), dtype=np.uint8)
    for row in range(1, 5):
        for col in range(1, 6):
            y0, x0 = row * CELL_PX, col * CELL_PX
            mask[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX] = 255
    return mask


def test_classify_cells_matches_drawn_region() -> None:
    mask = _filled_mask()
    filled = classify_cells(mask, 10, 12)
    assert filled[1:5, 1:6].all()
    assert not filled[0].any()
    assert not filled[:, 0].any()


def test_extract_receiving_edges_on_top_and_right() -> None:
    mask = np.zeros((6 * CELL_PX, 6 * CELL_PX), dtype=np.uint8)
    for row in range(1, 4):
        for col in range(1, 4):
            if row == 3 and col == 1:
                continue  # leave the gap itself empty
            mask[
                row * CELL_PX : (row + 1) * CELL_PX,
                col * CELL_PX : (col + 1) * CELL_PX,
            ] = 255
    filled = classify_cells(mask, 6, 6)
    edges = extract_receiving_edges(
        mask, 6, 6, row_1based=4, col_1based=2, filled=filled
    )
    assert "top" in edges and len(edges["top"]) == CELL_PX
    assert all(y == 3 * CELL_PX - 1 for (_, y) in edges["top"])
    assert "right" in edges and len(edges["right"]) == CELL_PX
    assert all(x == 2 * CELL_PX for (x, _) in edges["right"])
    assert "left" not in edges and "bottom" not in edges


def test_receiving_edges_preserve_tab_shape() -> None:
    """A neighbor tab protruding into a gap must survive edge extraction."""
    mask = np.zeros((5 * CELL_PX, 5 * CELL_PX), dtype=np.uint8)
    for row in range(2):
        for col in range(2):
            mask[
                row * CELL_PX : (row + 1) * CELL_PX,
                col * CELL_PX : (col + 1) * CELL_PX,
            ] = 255
    amp = CELL_PX * 0.18
    x0, y0 = CELL_PX, 2 * CELL_PX
    tab = np.array(
        [
            (x0 + CELL_PX * (i / 79), y0 - 1 + amp * math.sin(math.pi * i / 79))
            for i in range(80)
        ],
        dtype=np.float32,
    )
    cv2.fillPoly(mask, [tab.astype(np.int32)], 255)

    filled = classify_cells(mask, 5, 5)
    edges = extract_receiving_edges(
        mask, 5, 5, row_1based=3, col_1based=2, filled=filled
    )
    profile = edge_profile(np.asarray(edges["top"], dtype=np.float32))
    assert abs(profile).max() > 1e-6  # shape signal must not collapse to zero
    assert max(y for _, y in edges["top"]) >= y0 - 1 + amp * 0.9


def test_align_board_to_grid_recovers_layout() -> None:
    rows, cols = 10, 12
    grid_h, grid_w = rows * CELL_PX, cols * CELL_PX
    base = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for row in range(1, 5):
        for col in range(1, 6):
            x0, y0 = col * CELL_PX, row * CELL_PX
            cv2.rectangle(
                base,
                (x0, y0),
                (x0 + CELL_PX - 1, y0 + CELL_PX - 1),
                (30, 130, 210),
                -1,
            )
    src = np.array(
        [[0, 0], [grid_w - 1, 0], [grid_w - 1, grid_h - 1], [0, grid_h - 1]],
        dtype=np.float32,
    )
    photo_w, photo_h = 500, 420
    dst = np.array([[30, 25], [470, 35], [475, 395], [20, 385]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        base,
        matrix,
        (photo_w, photo_h),
        flags=cv2.INTER_NEAREST,
        borderValue=(255, 255, 255),
    )
    corners = [(30, 25), (470, 35), (475, 395), (20, 385)]

    warped, warped_mask = align_board_to_grid(photo, corners, rows, cols)
    assert warped.shape[:2] == (grid_h, grid_w)
    filled = classify_cells(warped_mask, rows, cols)
    assert filled[1:5, 1:6].all()
    assert not filled[0].any()
    assert find_gaps(filled)


def test_render_gap_preview_returns_image() -> None:
    warped = np.full((8 * CELL_PX, 10 * CELL_PX, 3), 255, dtype=np.uint8)
    filled = np.zeros((8, 10), dtype=bool)
    filled[0, 0] = True
    gaps = find_gaps(filled)
    assert len(gaps) == 2
    preview = render_gap_preview(warped, filled, gaps, 8, 10)
    assert preview.shape == warped.shape
