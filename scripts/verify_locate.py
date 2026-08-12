"""M4 verification script: piece-to-gap matching and Top-3 candidates."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzle.solver import matcher
from puzzle.vision import board, piece


def parse_points(raw: str) -> list[tuple[float, float]]:
    points = []
    for token in raw.split():
        x, y = token.split(",")
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise SystemExit("expected exactly four corners: TL TR BR BL")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify piece-to-gap matching.")
    parser.add_argument("--piece", required=True, help="path to a single-piece photo")
    parser.add_argument("--board", required=True, help="path to the assembled-part photo")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--corners", required=True, help='four corners as "x,y x,y x,y x,y"')
    parser.add_argument("--out", default="/tmp/puzzle-locate", help="output directory")
    args = parser.parse_args()

    piece_image = cv2.imread(args.piece)
    board_image = cv2.imread(args.board)
    if piece_image is None or board_image is None:
        raise SystemExit("cannot read piece or board image")

    warped, warped_mask = board.align_board_to_grid(
        board_image, parse_points(args.corners), args.rows, args.cols
    )
    filled = board.classify_cells(warped_mask, args.rows, args.cols)
    gaps = board.find_gaps(filled)
    gaps_with_edges = []
    for gap in gaps:
        edges = board.extract_receiving_edges(
            warped_mask,
            args.rows,
            args.cols,
            gap["row"],
            gap["col"],
            filled=filled,
        )
        gaps_with_edges.append({**gap, "edges": edges})

    signature = piece.build_signature(piece_image)
    started = time.perf_counter()
    candidates = matcher.top_candidates(signature, gaps_with_edges, k=3)
    latency_ms = (time.perf_counter() - started) * 1000

    preview = board.render_gap_preview(warped, filled, gaps, args.rows, args.cols)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "locate_preview.jpg"), preview)

    print(f"gaps evaluated: {len(gaps_with_edges)}")
    print(f"matching latency: {latency_ms:.1f} ms")
    for candidate in candidates:
        print(
            f"  row {candidate['row']} col {candidate['col']} "
            f"score={candidate['score']:.4f} confidence={candidate['confidence']:.3f} "
            f"rotation={candidate['rotation']}"
        )
    print(f"preview: {out_dir / 'locate_preview.jpg'}")


if __name__ == "__main__":
    main()
