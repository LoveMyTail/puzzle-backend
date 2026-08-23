"""Unit tests for single-piece segmentation and edge signatures (M3)."""

import math

import cv2
import numpy as np
import pytest

from puzzle.vision.piece import (
    SIDE_SAMPLES,
    build_signature,
    match_rotation,
    resample_contour,
    segment_piece,
    side_distance,
    side_profile,
    validate_piece_resolution,
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


def test_segment_piece_extracts_from_textured_background() -> None:
    """A piece on a textured, non-uniform background (like tissue paper)
    must still be extracted: color-distance alone fragments into thousands of
    noise components, so GrabCut has to carry this case.
    """
    base = _piece_image(120)
    rng = np.random.default_rng(11)
    textured = rng.integers(180, 255, size=base.shape, dtype=np.uint8)
    speckle = rng.random(base.shape[:2]) < 0.05
    textured[speckle] = 120
    textured[base < 128] = base[base < 128]
    signature = build_signature(textured)
    lengths = sorted(signature.side_lengths)
    assert lengths[0] > 60
    assert lengths[-1] / lengths[0] < 3


def test_split_sides_returns_four_resampled_sides() -> None:
    signature = build_signature(_piece_image(120))
    assert len(signature.sides) == 4
    assert all(len(side) == SIDE_SAMPLES for side in signature.sides)
    assert all(length > 0 for length in signature.side_lengths)
    assert max(abs(signature.sides[0])) > 0.02


def _star_image(size: int = 160) -> np.ndarray:
    """A spiky 10-point star: high perimeter/area ratio, clearly not a piece."""
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    center = size / 2
    outer, inner = size * 0.42, size * 0.10
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = math.pi * i / 5 - math.pi / 2
        points.append(
            (center + radius * math.cos(angle), center + radius * math.sin(angle))
        )
    cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], (30, 30, 30))
    return image


def test_build_signature_rejects_non_piece_shape() -> None:
    with pytest.raises(ValueError, match="清晰的拼图块轮廓"):
        build_signature(_star_image())


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
    # The side order is anchored to the piece's own minimum-area rectangle,
    # so a pure image rotation must not shift the signature (k == 0) and the
    # profiles must stay nearly identical.
    assert k == 0
    assert score < 0.02


def test_split_sides_survives_arbitrary_image_rotation() -> None:
    image = _piece_image(120)
    signature = build_signature(image)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 37, 1.0)
    rotated_image = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR
    )
    rotated_signature = build_signature(rotated_image)
    score, k = match_rotation(signature, rotated_signature)
    assert score < 0.15
    assert all(
        0.6 * min(signature.side_lengths)
        <= length
        <= 1.6 * max(signature.side_lengths)
        for length in rotated_signature.side_lengths
    )


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


def test_validate_piece_resolution_rejects_tiny_piece() -> None:
    with pytest.raises(ValueError):
        validate_piece_resolution(build_signature(_piece_image(12)))
    validate_piece_resolution(build_signature(_piece_image(120)))  # must not raise


def test_validate_piece_resolution_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PUZZLE_MIN_SIDE_PX", "64")
    with pytest.raises(ValueError):
        validate_piece_resolution(build_signature(_piece_image(24)))
    validate_piece_resolution(build_signature(_piece_image(120)))  # 120 >= 64
