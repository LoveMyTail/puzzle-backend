"""Single-piece processing: segmentation and fine edge signatures.

Implements the M3 milestone: given a photo of one puzzle piece, isolate it,
split its contour into four sides, and encode each side as a dense, scale-
and rotation-invariant profile. Rotation handling is provided by cyclic
side-order shifts, ready for gap matching in the next milestone.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np

from puzzle.vision.board import segment_board

logger = logging.getLogger("puzzle.piece")
_grabcut_lock = threading.Lock()

SIDE_SAMPLES = 64
MIN_SIDE_PX = 16  # default shortest-side floor (px); below this, tab/blank detail is lost


@dataclass
class PieceSignature:
    sides: list[np.ndarray]
    contour: np.ndarray
    side_lengths: list[float]


def segment_piece(
    image: np.ndarray,
    border_margin: int = 8,
    color_distance: float = 25.0,
) -> np.ndarray:
    """Return a 0/255 mask of the piece, keeping only the largest component.

    The background in real photos is rarely uniform (tables, tissue paper,
    shadows), so GrabCut is tried first with a center-foreground /
    border-background initialization. When GrabCut produces nothing useful
    it falls back to the color-distance segmentation of
    :func:`puzzle.vision.board.segment_board`.
    """
    grabcut_mask = _segment_piece_grabcut(image)
    if grabcut_mask is not None:
        refined = _refine_mask_edges(grabcut_mask, image)
        if refined is not None:
            grabcut_mask = refined
        logger.debug(
            "piece_segmentation mode=grabcut coverage=%.3f",
            float((grabcut_mask > 0).mean()),
        )
        return grabcut_mask
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
    logger.debug(
        "piece_segmentation mode=color coverage=%.3f",
        float((piece > 0).mean()),
    )
    return piece


def _segment_piece_grabcut(image: np.ndarray) -> np.ndarray | None:
    """GrabCut piece extraction with a center-foreground prior.

    The photo is downscaled to keep GrabCut fast, initialized with a certain
    background ring at the borders and probable foreground in the center
    (users photograph the piece roughly centered), then the largest
    component is scaled back to the original resolution. Returns ``None``
    when the result is degenerate (nothing, nearly everything, or a
    non-piece-like blob) so callers can fall back.
    """
    if image is None or image.ndim != 3:
        return None
    height, width = image.shape[:2]
    if height < 32 or width < 32:
        return None
    scale = min(1.0, 900.0 / max(height, width))
    small = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    sh, sw = small.shape[:2]
    init = np.zeros((sh, sw), dtype=np.uint8)
    ring = max(3, min(sh, sw) // 60)
    init[:ring, :] = cv2.GC_BGD
    init[-ring:, :] = cv2.GC_BGD
    init[:, :ring] = cv2.GC_BGD
    init[:, -ring:] = cv2.GC_BGD
    y0, y1 = sh // 6, sh - sh // 6
    x0, x1 = sw // 6, sw - sw // 6
    init[y0:y1, x0:x1] = cv2.GC_PR_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    # GrabCut's internal k-means uses OpenCV's global RNG; without a fixed
    # seed the same photo yields different masks run-to-run, which made
    # candidate rankings unstable across requests. The seed is process-level,
    # so hold a lock to keep concurrent requests from interfering with it.
    with _grabcut_lock:
        cv2.setRNGSeed(42)
        mask, _, _ = cv2.grabCut(
            small,
            init,
            None,
            background_model,
            foreground_model,
            4,
            cv2.GC_INIT_WITH_MASK,
        )
    piece = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    coverage = float((piece > 0).mean())
    if coverage < 0.005 or coverage > 0.85:
        return None
    piece = cv2.morphologyEx(piece, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(piece, connectivity=8)
    if num < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8) * 255
    if scale < 1.0:
        component = cv2.resize(
            component,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    component = cv2.morphologyEx(
        component, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    component = (cv2.GaussianBlur(component, (5, 5), 0) > 127).astype(
        np.uint8
    ) * 255
    return component


def _refine_mask_edges(mask: np.ndarray, image: np.ndarray) -> np.ndarray | None:
    """Snap the mask contour onto the photo's strongest nearby edges.

    GrabCut boundaries can include soft shadow/background gradients. This
    walks each contour point along its local normal and moves it onto the
    nearest strong Canny edge within a small window, which yields a sharper,
    more stable piece silhouette.
    """
    if image is None or image.ndim != 3:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    height, width = mask.shape[:2]
    search, step = 25, 3
    snapped = []
    for index in range(len(contour)):
        point = contour[index]
        prev = contour[(index - 1) % len(contour)]
        nxt = contour[(index + 1) % len(contour)]
        normal = np.array([-(nxt[1] - prev[1]), nxt[0] - prev[0]], dtype=np.float32)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-6:
            snapped.append(point)
            continue
        normal /= norm
        best = point
        best_distance = float("inf")
        for s in range(-search, search + 1, step):
            q = point + normal * s
            x, y = int(round(q[0])), int(round(q[1]))
            if 0 <= x < width and 0 <= y < height and edges[y, x] > 0:
                if abs(s) < best_distance:
                    best_distance = abs(s)
                    best = q
        snapped.append(best)
    refined = np.asarray(snapped, dtype=np.float32)
    if len(refined) >= 8:
        refined = cv2.GaussianBlur(
            refined.reshape(1, -1, 2), (1, 7), 1.5
        ).reshape(-1, 2)
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, [refined.astype(np.int32)], -1, 255, -1)
    refined_mask = cv2.morphologyEx(
        refined_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    # Keep the refinement only when it stays piece-like; otherwise the
    # snapped boundary (which can turn jagged on textured photos) would
    # degrade the signature.
    refined_contour = piece_contour(refined_mask)
    area = float(cv2.contourArea(refined_contour))
    perimeter = float(cv2.arcLength(refined_contour, True))
    hull = cv2.convexHull(refined_contour)
    compactness = perimeter / max(np.sqrt(area), 1e-9)
    solidity = area / max(float(cv2.contourArea(hull)), 1e-9)
    if compactness > 7.5 or solidity < 0.55:
        return None
    return refined_mask


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
    """Split a piece contour into four resampled side point sequences.

    Side junctions are located as the contour's support points in the four
    diagonal directions of the minimum-area rectangle. A tab tip protrudes
    perpendicular to one side, so it is extreme along only one rectangle
    axis; a true corner is extreme along both, which makes this selection
    robust where plain nearest-point matching would cut through a tab.
    """
    if len(contour) < 8:
        raise ValueError("contour too short to split into four sides")
    rect = cv2.minAreaRect(contour.astype(np.float32))
    (cx, cy), (w, h), angle = rect
    theta = np.deg2rad(angle)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float32,
    )
    local = (contour - np.array([cx, cy], dtype=np.float32)) @ rotation
    signs = [(1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)]
    indices: list[int] = []
    for sx, sy in signs:
        support = int(np.argmax(local[:, 0] * sx + local[:, 1] * sy))
        if support not in indices:
            indices.append(support)
    while len(indices) < 4:
        gaps = [
            ((indices[(i + 1) % len(indices)] - indices[i]) % len(contour))
            for i in range(len(indices))
        ]
        j = int(np.argmax(gaps))
        indices.append((indices[j] + gaps[j] // 2) % len(contour))
    indices = sorted(indices[:4])

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
    color_distance: float = 25.0,
    samples: int = SIDE_SAMPLES,
) -> PieceSignature:
    """Build a fine edge signature from a single-piece photo."""
    mask = segment_piece(image, border_margin, color_distance)
    contour = piece_contour(mask)
    segments = split_sides(contour, samples)
    signature = PieceSignature(
        sides=[side_profile(segment) for segment in segments],
        contour=contour,
        side_lengths=[
            float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
            for segment in segments
        ],
    )
    validate_piece_shape(signature)
    return signature


def validate_piece_shape(signature: PieceSignature) -> None:
    """Raise ``ValueError`` when the extracted blob does not look like a piece.

    A real puzzle piece is a compact, roughly square silhouette: its
    perimeter is only a few times the square root of its area, and it is
    fairly convex. Textured backgrounds (e.g. tissue paper), shadows, or
    other objects produce large, jagged, non-piece-like blobs; rejecting
    them here turns silent garbage matching into an actionable error.
    """
    contour = signature.contour
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    solidity = area / max(float(cv2.contourArea(hull)), 1e-9)
    compactness = perimeter / max(np.sqrt(area), 1e-9)
    if compactness > 7.5 or solidity < 0.55:
        raise ValueError(
            "未能提取出清晰的拼图块轮廓（背景纹理或阴影干扰）。"
            "请将碎片放在与碎片颜色反差明显的纯色背景上重新拍摄"
        )


def validate_piece_resolution(
    signature: PieceSignature, min_side_px: int | None = None
) -> None:
    """Raise ``ValueError`` when the piece is too small in the photo to match reliably.

    The threshold defaults to ``MIN_SIDE_PX`` and can be overridden with the
    ``PUZZLE_MIN_SIDE_PX`` environment variable (e.g. ``64`` for strict
    matching). Below it, tab/blank detail is quantized away and the profile
    stops being a faithful shape signature.
    """
    if min_side_px is None:
        min_side_px = int(os.environ.get("PUZZLE_MIN_SIDE_PX", MIN_SIDE_PX))
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
