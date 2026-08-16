"""Single-piece processing: segmentation and fine edge signatures.

Implements the M3 milestone: given a photo of one puzzle piece, isolate it,
split its contour into four sides, and encode each side as a dense, scale-
and rotation-invariant profile. Rotation handling is provided by cyclic
side-order shifts, ready for gap matching in the next milestone.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from puzzle.vision.board import segment_board

SIDE_SAMPLES = 64
MIN_SIDE_PX = 64  # shortest piece side in px; below this, tab/blank detail is lost


@dataclass
class PieceSignature:
    sides: list[np.ndarray]
    contour: np.ndarray
    side_lengths: list[float]


def segment_piece(
    image: np.ndarray,
    border_margin: int = 8,
    color_distance: float = 40.0,
) -> np.ndarray:
    """Return a 0/255 mask of the piece, keeping only the largest component."""
    raw = segment_board(image, border_margin, color_distance)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if num < 2:
        return raw
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    piece = np.where(labels == largest, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        filled = np.zeros_like(piece)
        cv2.drawContours(filled, contours, -1, 255, -1)
        piece = filled
    return piece


def piece_contour(mask: np.ndarray) -> np.ndarray:
    """Return the outer contour of the piece as a dense (N, 2) point array."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no piece contour found")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.float32)


def resample_contour(points: np.ndarray, samples: int) -> np.ndarray:
    """Resample a point sequence to ``samples`` evenly spaced points by arc length."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, samples, axis=0)
    lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lengths)])
    total = cum[-1]
    if total == 0:
        return np.repeat(pts[:1], samples, axis=0)
    targets = np.linspace(0.0, total, samples)
    out = np.empty((samples, 2), dtype=np.float32)
    out[:, 0] = np.interp(targets, cum, pts[:, 0])
    out[:, 1] = np.interp(targets, cum, pts[:, 1])
    return out


def split_sides(contour: np.ndarray, samples: int = SIDE_SAMPLES) -> list[np.ndarray]:
    """Split a piece contour into four resampled side point sequences."""
    if len(contour) < 8:
        raise ValueError("contour too short to split into four sides")
    rect = cv2.minAreaRect(contour.astype(np.float32))
    box = cv2.boxPoints(rect)
    indices = sorted({int(np.argmin(np.sum((contour - corner) ** 2, axis=1))) for corner in box})
    while len(indices) < 4:
        gaps = [
            ((indices[(i + 1) % len(indices)] - indices[i]) % len(contour))
            for i in range(len(indices))
        ]
        j = int(np.argmax(gaps))
        indices.append((indices[j] + gaps[j] // 2) % len(contour))
        indices.sort()

    sides: list[np.ndarray] = []
    for i in range(4):
        a = indices[i]
        b = indices[(i + 1) % 4]
        if b >= a:
            segment = contour[a : b + 1]
        else:
            segment = np.concatenate([contour[a:], contour[: b + 1]], axis=0)
        sides.append(resample_contour(segment, samples))
    return sides


def side_profile(points: np.ndarray) -> np.ndarray:
    """Encode a side as signed distances from its chord, normalized by chord length.

    The result is invariant to translation, rotation, and uniform scale. The
    sign depends on the point winding; reversing the point order negates and
    reverses the profile (useful when comparing a piece edge to a gap edge).
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=np.float32)
    chord = pts[-1] - pts[0]
    length = float(np.linalg.norm(chord))
    if length == 0:
        return np.zeros(len(pts), dtype=np.float32)
    normal = np.array([-chord[1], chord[0]], dtype=np.float32) / length
    return ((pts - pts[0]) @ normal) / length


def build_signature(
    image: np.ndarray,
    border_margin: int = 8,
    color_distance: float = 40.0,
    samples: int = SIDE_SAMPLES,
) -> PieceSignature:
    """Build a fine edge signature from a single-piece photo."""
    mask = segment_piece(image, border_margin, color_distance)
    contour = piece_contour(mask)
    segments = split_sides(contour, samples)
    return PieceSignature(
        sides=[side_profile(segment) for segment in segments],
        contour=contour,
        side_lengths=[
            float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
            for segment in segments
        ],
    )


def validate_piece_resolution(
    signature: PieceSignature, min_side_px: int = MIN_SIDE_PX
) -> None:
    """Raise ``ValueError`` when the piece is too small in the photo to match reliably.

    Below ``MIN_SIDE_PX`` pixels per side, tab/blank detail is quantized away and
    the profile stops being a faithful shape signature.
    """
    shortest = min(signature.side_lengths)
    if shortest < min_side_px:
        raise ValueError(
            f"piece is too small in the photo: shortest side is {shortest:.0f}px, "
            f"need at least {min_side_px}px; move the camera closer and retake"
        )


def rotated(signature: PieceSignature, k: int) -> PieceSignature:
    """Return a copy of the signature with sides cyclically shifted by ``k``."""
    k %= 4
    return PieceSignature(
        sides=signature.sides[k:] + signature.sides[:k],
        contour=signature.contour,
        side_lengths=signature.side_lengths[k:] + signature.side_lengths[:k],
    )


def side_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference between two side profiles."""
    n = min(len(a), len(b))
    return float(np.mean(np.abs(a[:n] - b[:n])))


def match_rotation(a: PieceSignature, b: PieceSignature) -> tuple[float, int]:
    """Return (best score, rotation k) aligning ``b`` to ``a`` over 4 rotations."""
    best_k = 0
    best_score = float("inf")
    for k in range(4):
        candidate = rotated(b, k)
        score = sum(side_distance(a.sides[i], candidate.sides[i]) for i in range(4)) / 4
        if score < best_score:
            best_score = score
            best_k = k
    return best_score, best_k
