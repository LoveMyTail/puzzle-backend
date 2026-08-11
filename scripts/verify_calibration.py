"""M1 verification script: run calibration on a box-cover photo.

Example:
    python scripts/verify_calibration.py --image box.jpg --piece-count 1000 \
        --width-cm 70 --height-cm 50 \
        --points "10,20 380,25 390,290 5,280" --out /tmp/puzzle-calib
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzle.vision import calibration


def parse_points(raw: str) -> list[tuple[float, float]]:
    points = []
    for token in raw.split():
        x, y = token.split(",")
        points.append((float(x), float(y)))
    if len(points) != 4:
        raise SystemExit("expected exactly four corners: TL TR BR BL")
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify box-cover calibration and grid slicing.")
    parser.add_argument("--image", required=True, help="path to the box-cover photo")
    parser.add_argument("--piece-count", type=int, required=True)
    parser.add_argument("--width-cm", type=float, required=True)
    parser.add_argument("--height-cm", type=float, required=True)
    parser.add_argument("--points", required=True, help='four corners as "x,y x,y x,y x,y"')
    parser.add_argument("--out", default="/tmp/puzzle-calib", help="output directory")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")

    rows, cols = calibration.estimate_grid(args.piece_count, args.width_cm, args.height_cm)
    warped = calibration.warp_box_photo(image, parse_points(args.points))
    cells = calibration.slice_grid(warped, rows, cols)
    preview = calibration.draw_grid_overlay(warped, rows, cols)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    warped_path = out_dir / "box_warped.jpg"
    preview_path = out_dir / "grid_preview.jpg"
    cells_dir = out_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(preview_path), preview)
    for i, cell in enumerate(cells):
        cv2.imwrite(str(cells_dir / f"cell_{i:04d}.jpg"), cell)

    print(f"estimated grid: {rows} rows x {cols} cols")
    print(f"cells: {len(cells)}")
    print(f"warped: {warped.shape[1]}x{warped.shape[0]} -> {warped_path}")
    print(f"grid preview: {preview_path}")
    print(f"cell patches: {cells_dir}")


if __name__ == "__main__":
    main()
