"""Unit tests for single-piece segmentation and edge signatures (M3)."""

import math

import cv2
import numpy as np

from puzzle.vision.piece import (
    SIDE_SAMPLES,
    build_signature,
    match_rotation,
    resample_contour,
    segment_piece,
    side_distance,
    side_profile,
)


def _piece_polygon(size: float, tab: float) -> np.ndarray:
    per_side = 40
    points: list[tuple[float, float]] = []
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * t, -tab * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size + tab * math.sin(math.pi * t), size * t))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((size * (1 - t), size + tab * math.sin(math.pi * t)))
    for i in range(per_side):
        t = i / (per_side - 1)
        points.append((-tab * math.sin(math.pi * t), size * (1 - t)))
    return np.asarray(points, dtype=np.float32)


def _piece_image(size: int, tab_ratio: float = 0.18) -> np.ndarray:
    tab = size * tab_ratio
    pad = int(math.ceil(tab)) + 6
    image = np.full((size + 2 * pad, size + 2 * pad, 3), 255, dtype=np.uint8)
    polygon = (_piece_polygon(size, tab) + np.array([pad, pad])).astype(np.int32)
    cv2.fillPoly(image, [polygon], (30, 30, 30))
    return image


def test_segment_piece_keeps_largest_component() -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (130, 130), (20, 20, 20), -1)
    cv2.circle(image, (175, 30), 4, (20, 20, 20), -1)
    mask = segment_piece(image)
    assert mask[80, 80] > 0
    assert mask[175, 30] == 0


def test_split_sides_returns_four_resampled_sides() -> None:
    signature = build_signature(_piece_image(120))
    assert len(signature.sides) == 4
    assert all(len(side) == SIDE_SAMPLES for side in signature.sides)
    assert all(length > 0 for length in signature.side_lengths)
    assert max(abs(signature.sides[0])) > 0.02


def test_self_match_rotation_is_zero() -> None:
    signature = build_signature(_piece_image(120))
    score, k = match_rotation(signature, signature)
    assert k == 0
    assert score < 1e-4


def test_rotation_invariance() -> None:
    image = _piece_image(120)
    signature = build_signature(image)
    rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    rotated_signature = build_signature(rotated_image)
    score, k = match_rotation(signature, rotated_signature)
    # A 90-degree clockwise image rotation shifts the side order by 3.
    assert k == 3
    assert score < 0.1


def test_scale_invariance() -> None:
    big = build_signature(_piece_image(120))
    small = build_signature(_piece_image(60))
    scores = [side_distance(big.sides[i], small.sides[i]) for i in range(4)]
    assert max(scores) < 0.1


def test_side_profile_flip_negates_and_reverses() -> None:
    points = np.array([[0, 0], [2, 1], [5, 3], [8, 1], [10, 0]], dtype=np.float32)
    resampled = resample_contour(points, 32)
    forward = side_profile(resampled)
    backward = side_profile(resampled[::-1])
    assert np.allclose(backward, -forward[::-1], atol=1e-5)
