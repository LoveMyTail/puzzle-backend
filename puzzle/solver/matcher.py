"""Gap matching: score a piece against board gaps and return Top-3 candidates.

Implements the M4 milestone: compare a piece's fine edge signature with the
receiving edges of every gap, search all four rotations, and rank the gaps
with normalized confidence scores.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np

from puzzle.vision.piece import SIDE_SAMPLES, PieceSignature, resample_contour, side_profile

logger = logging.getLogger("puzzle.matcher")

DIRECTIONS = {"top": 0, "right": 1, "bottom": 2, "left": 3}


def edge_profile(points: np.ndarray, samples: int = SIDE_SAMPLES) -> np.ndarray:
    """Encode a gap receiving edge as a chord-normalized side profile."""
    pts = np.asarray(points, dtype=np.float32)
    return side_profile(resample_contour(pts, samples))


def edge_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between a piece side profile and a gap edge profile.

    A piece edge and its neighbor's receiving edge are mirror images, so the
    comparison tries both orientations and keeps the closer one.
    """
    n = min(len(a), len(b))
    direct = float(np.mean(np.abs(a[:n] - b[:n])))
    mirrored = float(np.mean(np.abs(a[:n] + b[:n][::-1])))
    return min(direct, mirrored)


def match_gap(signature: PieceSignature, gap_edges: Mapping[str, list]) -> tuple[float, int]:
    """Return ``(best_score, rotation_k)`` aligning the piece into a gap."""
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
        score = sum(scores) / len(scores)
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
