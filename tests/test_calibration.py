"""Unit tests for grid estimation and perspective calibration."""

import cv2
import numpy as np
import pytest

from puzzle.vision.calibration import (
    draw_grid_overlay,
    estimate_grid,
    slice_grid,
    warp_box_photo,
)


def test_estimate_grid_1000_pieces() -> None:
    rows, cols = estimate_grid(1000, 70, 50)
    assert (rows, cols) == (27, 37)
    assert abs(rows * cols - 1000) <= 10


def test_estimate_grid_square() -> None:
    rows, cols = estimate_grid(100, 40, 40)
    assert (rows, cols) == (10, 10)


def test_estimate_grid_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        estimate_grid(0, 70, 50)
    with pytest.raises(ValueError):
        estimate_grid(1000, 0, 50)


def _synthetic_image(width: int = 370, height: int = 270) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 2, :] = (200, 120, 60)
    image[height // 2 :, :] = (60, 120, 200)
    return image


def test_warp_box_photo_identity() -> None:
    image = _synthetic_image()
    height, width = image.shape[:2]
    corners = [(0, 0), (width, 0), (width, height), (0, height)]
    warped = warp_box_photo(image, corners)
    assert warped.shape[:2] == (height, width)
    center = (height // 2, width // 2)
    assert np.abs(warped[center].astype(int) - image[center].astype(int)).max() <= 10


def test_warp_box_photo_rejects_bad_corners() -> None:
    with pytest.raises(ValueError):
        warp_box_photo(_synthetic_image(), [(0, 0), (1, 1)])


def test_slice_grid_counts_and_shapes() -> None:
    image = _synthetic_image()
    cells = slice_grid(image, rows=27, cols=37)
    assert len(cells) == 27 * 37
    assert cells[0].shape[:2] == (10, 10)
    assert cells[-1].shape[:2] == (10, 10)


def test_draw_grid_overlay_returns_image() -> None:
    preview = draw_grid_overlay(_synthetic_image(), 27, 37)
    assert preview.shape[:2] == (270, 370)
    assert preview.dtype == np.uint8
    assert cv2.countNonZero(cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)) > 0
