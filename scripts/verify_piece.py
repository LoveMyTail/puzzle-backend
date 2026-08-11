"""M3 verification script: piece segmentation and fine edge signature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from puzzle.vision.piece import SIDE_SAMPLES, build_signature, match_rotation, split_sides


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify single-piece edge signature extraction.")
    parser.add_argument("--image", required=True, help="path to a single-piece photo")
    parser.add_argument("--out", default="/tmp/puzzle-piece", help="output directory")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"cannot read image: {args.image}")

    signature = build_signature(image)
    viz = image.copy()
    colors = [(0, 200, 255), (255, 160, 0), (120, 120, 255), (0, 180, 120)]
    for segment, color in zip(split_sides(signature.contour), colors):
        cv2.polylines(viz, [segment.astype(np.int32)], False, color, 2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "piece_sides.jpg"), viz)

    score, k = match_rotation(signature, signature)
    print(f"sides: 4 x {SIDE_SAMPLES} samples")
    print(f"side lengths (px): {[round(v, 1) for v in signature.side_lengths]}")
    print(
        f"side tab depth (max |profile|): "
        f"{[round(float(abs(s).max()), 3) for s in signature.sides]}"
    )
    print(f"self rotation match: k={k} score={score:.6f}")
    print(f"side visualization: {out_dir / 'piece_sides.jpg'}")


if __name__ == "__main__":
    main()
