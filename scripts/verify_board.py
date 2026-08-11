"""M2 verification script: board alignment, gap detection, receiving edges."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzle.vision import board


def parse_points(raw: str) -> list[tuple[float, float]]:
    points = []
    for token in raw.split():
        x, y = token.split(",")
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise SystemExit("expected exactly four corners: TL TR BR BL")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify board alignment and gap extraction.")
    parser.add_argument("--image", required=True, help="path to the assembled-part photo")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--corners", required=True, help='four corners as "x,y x,y x,y x,y"')
    parser.add_argument("--out", default="/tmp/puzzle-board", help="output directory")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")

    warped, warped_mask = board.align_board_to_grid(
        image, parse_points(args.corners), args.rows, args.cols
    )
    filled = board.classify_cells(warped_mask, args.rows, args.cols)
    gaps = board.find_gaps(filled)
    preview = board.render_gap_preview(warped, filled, gaps, args.rows, args.cols)
    for gap in gaps:
        edges = board.extract_receiving_edges(
            warped_mask, args.rows, args.cols, gap["row"], gap["col"]
        )
        gap["edge_points"] = {direction: len(points) for direction, points in edges.items()}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "board_warped.jpg"), warped)
    cv2.imwrite(str(out_dir / "board_preview.jpg"), preview)

    print(f"filled cells: {int(filled.sum())} / {args.rows * args.cols}")
    print(f"gaps: {len(gaps)}")
    for gap in gaps:
        print(
            f"  row {gap['row']} col {gap['col']} neighbors={gap['directions']} "
            f"edge_points={gap['edge_points']}"
        )
    print(f"warped: {out_dir / 'board_warped.jpg'}")
    print(f"preview: {out_dir / 'board_preview.jpg'}")


if __name__ == "__main__":
    main()
