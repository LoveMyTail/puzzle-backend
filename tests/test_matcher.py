"""Unit tests for gap matching and Top-3 candidates (M4)."""

import math

import cv2
import numpy as np

from puzzle.solver.matcher import edge_distance, edge_profile, match_gap, top_candidates
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
