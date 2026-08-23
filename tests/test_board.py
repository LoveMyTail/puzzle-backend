"""Unit tests for board-state processing (M2)."""

import math

import cv2
import numpy as np
import pytest

from puzzle.solver.matcher import edge_profile
from puzzle.vision.board import (
    CELL_PX,
    MIN_SOURCE_CELL_PX,
    align_board_to_grid,
    alignment_quality,
    cell_region_in_photo,
    classify_cells,
    estimate_board_quad,
    extract_receiving_edges,
    find_gaps,
    is_full_frame_corners,
    render_candidate_preview,
    render_gap_preview,
    segment_board,
    source_cell_px,
    validate_board_resolution,
)


def test_segment_board_detects_pieces() -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (120, 120), (20, 120, 200), -1)
    mask = segment_board(image)
    assert mask[80, 80] > 0
    assert mask[5, 5] == 0


def test_is_full_frame_corners_detects_app_default() -> None:
    shape = (600, 900)
    assert is_full_frame_corners([[0, 0], [900, 0], [900, 600], [0, 600]], shape)
    assert not is_full_frame_corners(
        [[30, 25], [470, 35], [475, 395], [20, 385]], shape
    )
    assert is_full_frame_corners([], shape)


def test_estimate_board_quad_finds_block_on_smooth_background() -> None:
    image = np.full((600, 900, 3), 230, dtype=np.uint8)  # table
    # A block of puzzle-like tiles with visible seams.
    rows, cols = 4, 6
    cell = 60
    x0, y0 = 180, 120
    rng = np.random.default_rng(7)
    for r in range(rows):
        for c in range(cols):
            color = tuple(int(v) for v in rng.integers(40, 180, size=3))
            cv2.rectangle(
                image,
                (x0 + c * cell, y0 + r * cell),
                (x0 + (c + 1) * cell, y0 + (r + 1) * cell),
                color,
                -1,
            )
            cv2.rectangle(
                image,
                (x0 + c * cell, y0 + r * cell),
                (x0 + (c + 1) * cell, y0 + (r + 1) * cell),
                (0, 0, 0),
                2,
            )
    quad = estimate_board_quad(image)
    assert quad is not None
    # The detected quad should roughly cover the tile block.
    min_x, max_x = quad[:, 0].min(), quad[:, 0].max()
    min_y, max_y = quad[:, 1].min(), quad[:, 1].max()
    assert abs(min_x - x0) < 40
    assert abs(min_y - y0) < 40
    assert abs(max_x - (x0 + cols * cell)) < 40
    assert abs(max_y - (y0 + rows * cell)) < 40


def test_alignment_quality_penalizes_misalignment() -> None:
    size = 2 * CELL_PX
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    for r in range(2):
        for c in range(2):
            color = (60 + r * 60, 80 + c * 60, 120)
            cv2.rectangle(
                image,
                (c * CELL_PX, r * CELL_PX),
                ((c + 1) * CELL_PX, (r + 1) * CELL_PX),
                color,
                -1,
            )
            cv2.rectangle(
                image,
                (c * CELL_PX, r * CELL_PX),
                ((c + 1) * CELL_PX, (r + 1) * CELL_PX),
                (0, 0, 0),
                2,
            )
    corners = [[0, 0], [size, 0], [size, size], [0, size]]
    warped, mask = align_board_to_grid(image, corners, 2, 2)
    quality = alignment_quality(warped, mask, 2, 2)
    assert quality["ok"]
    assert quality["boundary_ratio"] > 1.25
    assert quality["note"] == "ok"
    # Shift the corners by a quarter cell: seams no longer lie on the grid.
    shifted = [
        [CELL_PX // 2, 0],
        [size + CELL_PX // 2, 0],
        [size + CELL_PX // 2, size],
        [CELL_PX // 2, size],
    ]
    warped_bad, mask_bad = align_board_to_grid(image, shifted, 2, 2)
    bad = alignment_quality(warped_bad, mask_bad, 2, 2)
    # The ratio still reports the drop even though it is informational now.
    assert bad["boundary_ratio"] < quality["boundary_ratio"]


def test_alignment_quality_flags_low_contrast_and_too_empty() -> None:
    empty_mask = np.zeros((2 * CELL_PX, 2 * CELL_PX), dtype=np.uint8)
    warped = np.full((2 * CELL_PX, 2 * CELL_PX, 3), 200, dtype=np.uint8)
    low = alignment_quality(warped, empty_mask, 2, 2)
    assert low["note"] == "low_contrast"
    assert not low["ok"]

    nearly_empty = empty_mask.copy()
    nearly_empty[:CELL_PX, :CELL_PX] = 255
    too_empty = alignment_quality(warped, nearly_empty, 2, 2)
    assert too_empty["note"] == "too_empty"
    assert too_empty["ok"]


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


def test_cell_region_in_photo_maps_grid_cell_back_to_photo() -> None:
    rows, cols = 4, 4
    cell_px = 64
    photo_shape = (512, 512)  # (height, width)
    corners = [[0, 0], [512, 0], [512, 512], [0, 512]]
    region = cell_region_in_photo(
        corners, rows, cols, cell_px, photo_shape, row_1based=2, col_1based=2
    )
    assert region is not None
    # Cell (2,2) spans [64,128) grid px; expanded by 50% -> [32,160). The
    # homography uses the same dst convention as align_board_to_grid (grid
    # width = grid_w - 1), so the mapping back is photo_x = grid_x * 512 / 255.
    assert region["x"] == pytest.approx(32 / 255)
    assert region["y"] == pytest.approx(32 / 255)
    assert region["w"] == pytest.approx(128 / 255)
    assert region["h"] == pytest.approx(128 / 255)


def test_cell_region_in_photo_clamps_at_board_border() -> None:
    rows, cols = 4, 4
    photo_shape = (512, 512)
    corners = [[0, 0], [512, 0], [512, 512], [0, 512]]
    region = cell_region_in_photo(
        corners, rows, cols, 64, photo_shape, row_1based=1, col_1based=1
    )
    assert region is not None
    assert region["x"] >= 0.0 and region["y"] >= 0.0
    assert region["x"] + region["w"] <= 1.0
    assert region["y"] + region["h"] <= 1.0


def test_cell_region_in_photo_degrades_to_none() -> None:
    rows, cols = 4, 4
    assert (
        cell_region_in_photo([], rows, cols, 64, (512, 512), 1, 1) is None
    )
    assert (
        cell_region_in_photo(
            [[0, 0], [512, 0], [512, 512], [0, 512]],
            rows,
            cols,
            64,
            None,
            1,
            1,
        )
        is None
    )
    assert (
        cell_region_in_photo(
            [[0, 0], [512, 0], [512, 512], [0, 512]],
            rows,
            cols,
            64,
            (512, 512),
            row_1based=99,
            col_1based=1,
        )
        is None
    )


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


def test_render_candidate_preview_marks_cells() -> None:
    warped = np.full((4 * CELL_PX, 5 * CELL_PX, 3), 255, dtype=np.uint8)
    candidates = [
        {"row": 2, "col": 3, "rotation": 0},
        {"row": 4, "col": 5, "rotation": 2},
    ]
    preview = render_candidate_preview(warped, candidates)
    assert preview.shape == warped.shape
    assert not np.array_equal(preview, warped)  # markers were drawn
    # The top-1 cell is tinted, not just outlined.
    x0, y0 = 2 * CELL_PX, 1 * CELL_PX
    tinted = preview[y0 : y0 + CELL_PX, x0 : x0 + CELL_PX]
    assert not np.allclose(tinted, 255)


def test_source_cell_px_measures_marked_region() -> None:
    small = [(30, 25), (470, 35), (475, 395), (20, 385)]
    big = [(30, 25), (3060, 40), (3070, 2580), (20, 2565)]
    assert source_cell_px(small, 27, 37) < MIN_SOURCE_CELL_PX
    assert source_cell_px(big, 27, 37) >= MIN_SOURCE_CELL_PX


def test_validate_board_resolution_rejects_low_res() -> None:
    small = [(30, 25), (470, 35), (475, 395), (20, 385)]
    big = [(30, 25), (3060, 40), (3070, 2580), (20, 2565)]
    with pytest.raises(ValueError):
        validate_board_resolution(small, 27, 37)
    assert validate_board_resolution(big, 27, 37) == pytest.approx(
        source_cell_px(big, 27, 37)
    )


def test_validate_board_resolution_default_threshold() -> None:
    small = [(30, 25), (470, 35), (475, 395), (20, 385)]  # ~12 px/cell: rejected
    medium = [(30, 25), (740, 35), (745, 615), (20, 605)]  # ~19 px/cell: accepted
    with pytest.raises(ValueError):
        validate_board_resolution(small, 27, 37)
    assert validate_board_resolution(medium, 27, 37) == pytest.approx(
        source_cell_px(medium, 27, 37)
    )


def test_validate_board_resolution_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    medium = [(30, 25), (740, 35), (745, 615), (20, 605)]
    monkeypatch.setenv("PUZZLE_MIN_CELL_PX", "64")
    with pytest.raises(ValueError):
        validate_board_resolution(medium, 27, 37)
