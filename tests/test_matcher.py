"""Unit tests for gap matching and Top-3 candidates (M4)."""

import math

import cv2
import numpy as np

from puzzle.solver.matcher import edge_distance, edge_profile, match_gap, top_candidates
from puzzle.vision.board import CELL_PX, classify_cells, extract_receiving_edges, find_gaps
from puzzle.vision.piece import SIDE_SAMPLES, build_signature, split_sides


def _piece_polygon(size: float, tab: float, distinct: bool) -> np.ndarray:
    per_side = 40
    if distinct:
        top_amp = -tab
        right_amp = -0.6 * tab
        bottom_amp = 1.3 * tab
        left_amp = 0.8 * tab
    else:
        top_amp = -tab
        right_amp = tab
        bottom_amp = tab
        left_amp = -tab
    points: list[tuple[float, float]] = []
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * t, top_amp * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size + right_amp * math.sin(math.pi * t), size * t))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * (1 - t), size + bottom_amp * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((left_amp * math.sin(math.pi * t), size * (1 - t)))
    return np.asarray(points, dtype=np.float32)


def _piece_image(size: int = 120, distinct: bool = False) -> np.ndarray:
    tab = size * 0.18
    pad = int(math.ceil(tab)) + 6
    image = np.full((size + 2 * pad, size + 2 * pad, 3), 255, dtype=np.uint8)
    polygon = (_piece_polygon(size, tab, distinct) + np.array([pad, pad])).astype(np.int32)
    cv2.fillPoly(image, [polygon], (30, 30, 30))
    return image


def _piece_image_with_top_blank(
    size: int = CELL_PX, tab: float = 0.18, angle: float = 0.0
) -> np.ndarray:
    """Square piece whose top side dips down (a blank), other sides distinct."""
    amp = size * tab
    pad = int(math.ceil(amp)) + 6
    points: list[tuple[float, float]] = []
    per_side = 40
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * t, amp * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size + 0.8 * amp * math.sin(math.pi * t), size * t))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * (1 - t), size + 1.2 * amp * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((0.6 * amp * math.sin(math.pi * t), size * (1 - t)))
    polygon = np.asarray(points, dtype=np.float32)
    if angle:
        center = (size / 2, size / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        polygon = cv2.transform(
            polygon.reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
    polygon = (polygon + np.array([pad, pad])).astype(np.int32)
    image = np.full((size + 2 * pad, size + 2 * pad, 3), 255, dtype=np.uint8)
    cv2.fillPoly(image, [polygon], (30, 30, 30))
    return image


def _tab_polygon(x0: float, y0: float, width: float, amp: float) -> np.ndarray:
    return np.array(
        [
            (x0 + width * (i / 79), y0 + amp * math.sin(math.pi * i / 79))
            for i in range(80)
        ],
        dtype=np.float32,
    )


def test_edge_distance_handles_mirror() -> None:
    profile = np.linspace(-0.2, 0.3, 32).astype(np.float32)
    mirrored = -profile[::-1]
    assert edge_distance(profile, mirrored) < 1e-6


def test_edge_profile_has_expected_length() -> None:
    points = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
    assert len(edge_profile(points)) == SIDE_SAMPLES


def test_match_gap_recognizes_own_edge() -> None:
    signature = build_signature(_piece_image())
    segments = split_sides(signature.contour)
    score, k = match_gap(signature, {"top": segments[0]})
    assert k == 0
    assert score < 0.05


def test_match_gap_finds_rotation_with_distinct_sides() -> None:
    signature = build_signature(_piece_image(distinct=True))
    segments = split_sides(signature.contour)
    score, k = match_gap(signature, {"top": segments[0]})
    assert k == 0 and score < 0.05
    score, k = match_gap(signature, {"top": segments[1]})
    assert k == 3 and score < 0.05
    score, k = match_gap(signature, {"bottom": segments[0]})
    assert k == 2 and score < 0.05


def test_top_candidates_orders_matching_gap_first() -> None:
    signature = build_signature(_piece_image(distinct=True))
    segments = split_sides(signature.contour)
    straight = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)]).astype(np.float32)
    gaps = [
        {"row": 1, "col": 1, "edges": {"top": segments[0], "right": segments[1]}},
        {"row": 5, "col": 5, "edges": {"top": straight, "right": straight}},
    ]
    top = top_candidates(signature, gaps, k=2)
    assert (top[0]["row"], top[0]["col"]) == (1, 1)
    assert top[0]["score"] < top[1]["score"]
    assert top[0]["confidence"] > top[1]["confidence"]
    assert abs(sum(candidate["confidence"] for candidate in top) - 1.0) < 1e-6


def test_real_chain_ranks_true_gap_first() -> None:
    """End-to-end regression: board mask -> receiving edges -> matcher.

    Previously the receiving edge collapsed to an all-zero profile, so the
    matcher could not distinguish the true gap. With tab-preserving edge
    extraction, the gap below a tabbed neighbor must rank first.
    """
    from puzzle.solver.matcher import score_gaps

    size = CELL_PX
    amp = size * 0.18
    mask = np.zeros((4 * size, 4 * size), dtype=np.uint8)
    for row in range(2):
        for col in range(2):
            mask[
                row * size : (row + 1) * size,
                col * size : (col + 1) * size,
            ] = 255
    # Tab on the bottom of cell (1,1) (0-based), protruding into gap (3,2) (1-based).
    tab = _tab_polygon(size, 2 * size - 1, size, amp)
    cv2.fillPoly(mask, [tab.astype(np.int32)], 255)

    filled = classify_cells(mask, 4, 4)
    gaps = find_gaps(filled)
    gaps_with_edges = [
        {
            **gap,
            "edges": extract_receiving_edges(
                mask, 4, 4, gap["row"], gap["col"], filled=filled
            ),
        }
        for gap in gaps
    ]
    true_gap = next(g for g in gaps_with_edges if g["row"] == 3 and g["col"] == 2)
    profile = edge_profile(np.asarray(true_gap["edges"]["top"], dtype=np.float32))
    assert abs(profile).max() > 1e-6  # receiving edge carries shape information

    signature = build_signature(_piece_image_with_top_blank())
    ranked = score_gaps(signature, gaps_with_edges)
    assert (ranked[0]["row"], ranked[0]["col"]) == (3, 2)
    assert ranked[0]["score"] < 0.05

    top = top_candidates(signature, gaps_with_edges, k=3)
    assert (top[0]["row"], top[0]["col"]) == (3, 2)


def test_real_chain_survives_arbitrary_piece_rotation() -> None:
    """End-to-end regression: a piece photographed at a non-axis-aligned angle
    must still rank the true gap first.

    The side order is anchored to the piece's own minimum-area rectangle, so
    the signature must be invariant to the piece's in-plane rotation in the
    photo; otherwise real photos (never perfectly aligned) break matching.
    """
    from puzzle.solver.matcher import score_gaps

    size = CELL_PX
    amp = size * 0.18
    mask = np.zeros((4 * size, 4 * size), dtype=np.uint8)
    for row in range(2):
        for col in range(2):
            mask[
                row * size : (row + 1) * size,
                col * size : (col + 1) * size,
            ] = 255
    tab = _tab_polygon(size, 2 * size - 1, size, amp)
    cv2.fillPoly(mask, [tab.astype(np.int32)], 255)

    filled = classify_cells(mask, 4, 4)
    gaps = find_gaps(filled)
    gaps_with_edges = [
        {
            **gap,
            "edges": extract_receiving_edges(
                mask, 4, 4, gap["row"], gap["col"], filled=filled
            ),
        }
        for gap in gaps
    ]

    signature = build_signature(_piece_image_with_top_blank(angle=37.0))

    ranked = score_gaps(signature, gaps_with_edges)
    assert (ranked[0]["row"], ranked[0]["col"]) == (3, 2)
    assert ranked[0]["score"] < 0.05
