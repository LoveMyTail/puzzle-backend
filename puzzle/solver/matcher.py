"""Gap matching: score a piece against board gaps and return Top-3 candidates.

Implements the M4 milestone: compare a piece's fine edge signature with the
receiving edges of every gap, search all four rotations, and rank the gaps
with normalized confidence scores.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import cv2
import numpy as np

from puzzle.vision.piece import (
    SIDE_SAMPLES,
    PieceSignature,
    resample_contour,
    side_profile,
    split_sides,
)

logger = logging.getLogger("puzzle.matcher")

DIRECTIONS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
GAP_WORST_THRESHOLD = 0.15
GAP_WORST_PENALTY = 0.5


def edge_profile(points: np.ndarray, samples: int = SIDE_SAMPLES) -> np.ndarray:
    """Encode a gap receiving edge as a chord-normalized side profile."""
    pts = np.asarray(points, dtype=np.float32)
    return side_profile(resample_contour(pts, samples))


def edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between a piece side profile and a gap edge profile.

    A piece edge and its neighbor's receiving edge are mirror images, so the
    comparison tries both orientations and keeps the closer one. Mean
    absolute difference is used; correlation-based variants were tested on
    real photos and proved less discriminative because flat edges trivially
    correlate with each other.
    """
    n = min(len(a), len(b))
    direct = float(np.mean(np.abs(a[:n] - b[:n])))
    mirrored = float(np.mean(np.abs(a[:n] + b[:n][::-1])))
    return min(direct, mirrored)


def match_gap(signature: PieceSignature, gap_edges: Mapping[str, list]) -> tuple[float, int]:
    """Return ``(best_score, rotation_k)`` aligning the piece into a gap.

    The gap score is the best single-direction match (the tightest side
    already gives strong evidence), plus a small penalty when another
    available direction matches much worse than ``GAP_WORST_THRESHOLD``.
    Averaging over directions was replaced because a noisy receiving edge
    then sinks an otherwise-correct multi-direction gap, while single-
    direction gaps win on one lucky match.
    """
    best_score = float("inf")
    best_rotation = 0
    for k in range(4):
        scores = []
        for direction, points in gap_edges.items():
            if direction not in DIRECTIONS:
                continue
            side_index = (DIRECTIONS[direction] - k) % 4
            scores.append(edge_distance(signature.sides[side_index], edge_profile(points)))
        if not scores:
            continue
        worst = max(scores)
        score = min(scores) + GAP_WORST_PENALTY * max(
            0.0, worst - GAP_WORST_THRESHOLD
        )
        if score < best_score:
            best_score = score
            best_rotation = k
    return best_score, best_rotation


def score_gaps(signature: PieceSignature, gaps_with_edges: list[dict]) -> list[dict]:
    """Score every gap and return them sorted from best to worst."""
    results = []
    for gap in gaps_with_edges:
        edges = gap.get("edges") or {}
        score, rotation = match_gap(signature, edges)
        logger.debug(
            "gap_score row=%d col=%d directions=%d score=%s rotation=%d",
            gap["row"],
            gap["col"],
            len(edges),
            f"{score:.4f}" if np.isfinite(score) else "inf",
            rotation,
        )
        results.append(
            {"row": gap["row"], "col": gap["col"], "score": score, "rotation": rotation}
        )
    results.sort(key=lambda item: item["score"])
    return results


def _softmin_confidence(scores: list[float], temperature: float = 0.1) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    exp = np.exp(-values / temperature)
    return exp / exp.sum()


def top_candidates(
    signature: PieceSignature,
    gaps_with_edges: list[dict],
    k: int = 3,
) -> list[dict]:
    """Return the top ``k`` gaps with normalized confidence scores."""
    ranked = score_gaps(signature, gaps_with_edges)[:k]
    if not ranked:
        return []
    confidences = _softmin_confidence([item["score"] for item in ranked])
    for item, confidence in zip(ranked, confidences):
        item["confidence"] = float(confidence)
    return ranked


def reweight_confidences(candidates: list[dict]) -> None:
    """Recompute softmin confidence after external re-ranking."""
    if not candidates:
        return
    temperature = float(
        os.environ.get("PUZZLE_CONFIDENCE_TEMPERATURE", "0.05")
    )
    confidences = _softmin_confidence(
        [item["score"] for item in candidates], temperature=temperature
    )
    for item, confidence in zip(candidates, confidences):
        item["confidence"] = float(confidence)


def edge_continuity(
    warped: np.ndarray,
    cell_px: int,
    piece_image: np.ndarray,
    signature: PieceSignature,
    gap_edges: Mapping[str, list],
    rotation: int,
    samples: int = SIDE_SAMPLES,
    offset_frac: float = 0.12,
) -> float:
    """Average normalized-intensity correlation along the shared edges.

    A correct placement continues the neighbor's artwork across the seam, so
    the piece's edge-adjacent interior strip and the neighbor's edge-adjacent
    interior strip should correlate after per-strip normalization (which
    absorbs lighting/white-balance differences). This provides a signal that
    pure shape matching lacks when piece shapes repeat. Returns the mean
    correlation over the gap's directions, or 0.0 when it cannot be computed.
    """
    if warped is None or warped.ndim != 3 or piece_image is None:
        return 0.0
    if piece_image.ndim != 3:
        return 0.0
    side_segments = split_sides(signature.contour, samples)
    centroid = signature.contour.mean(axis=0)
    piece_gray = cv2.cvtColor(piece_image, cv2.COLOR_BGR2GRAY)
    correlations: list[float] = []
    for direction, points in gap_edges.items():
        if direction not in DIRECTIONS:
            continue
        side_index = (DIRECTIONS[direction] - rotation) % 4
        board_strip = _board_edge_strip(
            warped, cell_px, np.asarray(points, dtype=np.float32), direction, samples
        )
        piece_strip = _piece_edge_strip(
            piece_gray, side_segments[side_index], centroid, offset_frac
        )
        if board_strip is None or piece_strip is None:
            continue
        board_norm = _normalize_profile(board_strip)
        piece_norm = _normalize_profile(piece_strip)
        if not np.any(board_norm) or not np.any(piece_norm):
            continue
        correlation = float(np.corrcoef(board_norm, piece_norm)[0, 1])
        if np.isfinite(correlation):
            correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else 0.0


def _board_edge_strip(
    warped: np.ndarray,
    cell_px: int,
    edge_points: np.ndarray,
    direction: str,
    samples: int,
) -> np.ndarray | None:
    """Sample a strip just inside the neighbor piece along a receiving edge."""
    if len(edge_points) < 8:
        return None
    resampled = resample_contour(edge_points, samples)
    if direction == "top":
        offset = np.array([[0, -1]], dtype=np.float32)
    elif direction == "bottom":
        offset = np.array([[0, 1]], dtype=np.float32)
    elif direction == "right":
        offset = np.array([[1, 0]], dtype=np.float32)
    elif direction == "left":
        offset = np.array([[-1, 0]], dtype=np.float32)
    else:
        return None
    strip = resampled + offset * (cell_px * 0.12)
    height, width = warped.shape[:2]
    xs = np.clip(strip[:, 0].astype(int), 0, width - 1)
    ys = np.clip(strip[:, 1].astype(int), 0, height - 1)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return gray[ys, xs].astype(np.float64)


def _piece_edge_strip(
    piece_gray: np.ndarray,
    side_segment: np.ndarray,
    centroid: np.ndarray,
    offset_frac: float,
) -> np.ndarray | None:
    """Sample a strip just inside the piece along one of its sides."""
    points = np.asarray(side_segment, dtype=np.float32)
    chord = points[-1] - points[0]
    length = float(np.linalg.norm(chord))
    if length < 1.0:
        return None
    normal = np.array([-chord[1], chord[0]], dtype=np.float32) / length
    midpoint = (points[0] + points[-1]) / 2
    if np.dot(normal, centroid - midpoint) < 0:
        normal = -normal
    strip = points + normal * (length * offset_frac)
    height, width = piece_gray.shape[:2]
    xs = np.clip(strip[:, 0].astype(int), 0, width - 1)
    ys = np.clip(strip[:, 1].astype(int), 0, height - 1)
    return piece_gray[ys, xs].astype(np.float64)


def _normalize_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values - values.mean()
    std = values.std()
    return values / max(std, 1e-6)
