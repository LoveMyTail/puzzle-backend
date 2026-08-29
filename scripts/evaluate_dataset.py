"""Batch-evaluate the locate pipeline over a dataset of project folders.

Each test case is a project directory containing:
  board_photo.jpg   - the assembled-part photo (with metadata corners)
  piece_photo.jpg   - the single-piece photo being located
  metadata.json     - rows/cols/corners (as produced by the API)

Optionally provide a labels CSV (header: ``project_id,row,col``) with the
ground-truth position for each case to get hit rates and confidence of the
true answer.

Usage:
  .venv/bin/python scripts/evaluate_dataset.py --data-dir data/projects
  .venv/bin/python scripts/evaluate_dataset.py --labels labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import main as server_main  # noqa: E402
from puzzle.vision import board, piece  # noqa: E402


def load_labels(path: str | None) -> dict[str, tuple[int, int]]:
    labels: dict[str, tuple[int, int]] = {}
    if not path:
        return labels
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[row["project_id"].strip()] = (
                int(row["row"]),
                int(row["col"]),
            )
    return labels


def _dummy_box_bytes() -> bytes:
    image = np.full((120, 90, 3), 128, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def run_test_sets(test_sets_dir: Path, labels_csv: str | None) -> list[dict]:
    """Run each case in the manifest through the real API pipeline."""
    manifest_path = test_sets_dir / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    run_dir = test_sets_dir.parent / "test_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PUZZLE_DATA_DIR"] = str(run_dir)
    client = TestClient(server_main.app)

    rows: list[dict] = []
    with open(manifest_path, encoding="utf-8") as handle:
        cases = [
            row
            for row in csv.DictReader(handle)
            if row.get("name") and not row["name"].startswith("#")
        ]
    for case in cases:
        name = case["name"]
        board_path = test_sets_dir / case["board"]
        piece_path = test_sets_dir / case["piece"]
        rows_num = int(case["rows"])
        cols_num = int(case["cols"])
        expected = (
            (int(case["expected_row"]), int(case["expected_col"]))
            if case.get("expected_row") and case.get("expected_col")
            else None
        )
        try:
            board_bytes = board_path.read_bytes()
            piece_bytes = piece_path.read_bytes()
        except OSError as exc:
            rows.append(
                {
                    "name": name,
                    "candidates": "-",
                    "truth": expected,
                    "rank": "-",
                    "confidence": "-",
                    "note": f"missing file: {exc.filename}",
                }
            )
            continue

        created = client.post(
            "/api/projects",
            files={"box_photo": ("box.jpg", _dummy_box_bytes(), "image/jpeg")},
            data={"name": name, "piece_count": "48", "width_cm": "25", "height_cm": "20"},
        )
        project_id = created.json()["project_id"]
        client.put(
            f"/api/projects/{project_id}/calibration",
            json={
                "points": [[0, 0], [90, 0], [90, 120], [0, 120]],
                "rows": rows_num,
                "cols": cols_num,
            },
        )
        corners = case.get("corners", "").strip()
        if corners:
            corners_json = corners
        else:
            board_image = cv2.imdecode(
                np.frombuffer(board_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            height, width = board_image.shape[:2]
            corners_json = json.dumps(
                [[0, 0], [width, 0], [width, height], [0, height]]
            )
        client.put(
            f"/api/projects/{project_id}/board",
            files={"board_photo": ("board.jpg", board_bytes, "image/jpeg")},
            data={
                "corners": corners_json,
                "rows": str(rows_num),
                "cols": str(cols_num),
            },
        )
        response = client.post(
            f"/api/projects/{project_id}/locate",
            files={"piece_photo": ("piece.jpg", piece_bytes, "image/jpeg")},
        )
        diagnostics = _case_diagnostics(run_dir, project_id)
        if response.status_code != 200:
            rows.append(
                {
                    "name": name,
                    "candidates": "-",
                    "truth": expected,
                    "rank": "-",
                    "confidence": "-",
                    "note": str(response.json().get("detail", response.status_code)),
                }
            )
            continue
        candidates = response.json()["candidates"]
        rank = next(
            (
                index + 1
                for index, candidate in enumerate(candidates)
                if expected
                and (candidate["row"], candidate["col"]) == expected
            ),
            None,
        )
        rows.append(
            {
                "name": name,
                "candidates": ",".join(
                    f"({c['row']},{c['col']})" for c in candidates
                ),
                "truth": expected,
                "rank": rank if rank else ("OUT" if expected else "-"),
                "confidence": (
                    f"{candidates[rank - 1]['confidence']:.2f}" if rank else "-"
                ),
                "diag": diagnostics,
                "note": "ok",
            }
        )
    return rows


def _case_diagnostics(run_dir: Path, project_id: str) -> str:
    """Compact per-case diagnostic: alignment + piece signature quality."""
    project_dir = run_dir / "projects" / project_id
    meta_path = project_dir / "metadata.json"
    if not meta_path.exists():
        return "-"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    bi = meta.get("board") or {}
    parts: list[str] = []
    if bi.get("corners"):
        photo = cv2.imread(str(project_dir / "board_photo.jpg"))
        if photo is not None:
            rows_num, cols_num = meta["rows"], meta["cols"]
            cell = bi.get("cell_px") or 192
            background = board.exterior_background(photo, bi["corners"])
            warped, mask = board.align_board_to_grid(
                photo,
                bi["corners"],
                rows_num,
                cols_num,
                cell_px=cell,
                background=background,
            )
            quality = board.alignment_quality(
                warped, mask, rows_num, cols_num, cell_px=cell
            )
            parts.append(f"align={quality['boundary_ratio']}({quality['note']})")
    piece_path = project_dir / "piece_photo.jpg"
    if piece_path.exists():
        try:
            signature = piece.build_signature(cv2.imread(str(piece_path)))
            lengths = sorted(signature.side_lengths)
            parts.append(
                f"piece_ratio={max(lengths) / max(min(lengths), 1e-6):.2f}"
            )
        except ValueError:
            parts.append("piece_rejected")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parents[1] / "data" / "projects"),
    )
    parser.add_argument("--labels", default=None, help="CSV: project_id,row,col")
    parser.add_argument(
        "--test-sets",
        default=None,
        help="evaluate a manifest-driven dataset directory (data/test_sets)",
    )
    args = parser.parse_args()

    if args.test_sets:
        rows = run_test_sets(Path(args.test_sets), args.labels)
        header = ["name", "candidates", "truth", "rank", "confidence", "diag", "note"]
        print("\t".join(header))
        for row in rows:
            print("\t".join(str(row.get(key, "-")) for key in header))
        ranked = [row for row in rows if isinstance(row["rank"], int)]
        labeled = [row for row in rows if row["truth"] != "-"]
        if labeled:
            total = len(labeled)
            for top_k in (1, 3, 6):
                hits = sum(
                    1
                    for row in labeled
                    if isinstance(row["rank"], int) and row["rank"] <= top_k
                )
                print(f"hit@{top_k}: {hits}/{total} ({100 * hits / total:.0f}%)")
        return

    labels = load_labels(args.labels)
    client = TestClient(server_main.app)
    rows: list[dict] = []

    data_dir = Path(args.data_dir)
    for project_dir in sorted(data_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        piece_path = project_dir / "piece_photo.jpg"
        meta_path = project_dir / "metadata.json"
        if not piece_path.exists() or not meta_path.exists():
            continue
        project_id = project_dir.name
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        with open(piece_path, "rb") as handle:
            response = client.post(
                f"/api/projects/{project_id}/locate",
                files={
                    "piece_photo": ("piece.jpg", handle.read(), "image/jpeg")
                },
            )
        if response.status_code != 200:
            rows.append(
                {
                    "project": project_id,
                    "grid": f"{meta.get('rows')}x{meta.get('cols')}",
                    "candidates": "-",
                    "truth": labels.get(project_id, "-"),
                    "rank": "-",
                    "confidence": "-",
                    "note": str(response.json().get("detail", response.status_code)),
                }
            )
            continue
        candidates = response.json()["candidates"]
        truth = labels.get(project_id)
        rank = next(
            (
                index + 1
                for index, candidate in enumerate(candidates)
                if truth
                and (candidate["row"], candidate["col"]) == truth
            ),
            None,
        )
        rows.append(
            {
                "project": project_id,
                "grid": f"{meta.get('rows')}x{meta.get('cols')}",
                "candidates": ",".join(
                    f"({c['row']},{c['col']})" for c in candidates
                ),
                "truth": truth if truth else "-",
                "rank": rank if rank else ("OUT" if truth else "-"),
                "confidence": (
                    f"{candidates[rank - 1]['confidence']:.2f}"
                    if rank
                    else "-"
                ),
                "note": "ok",
            }
        )

    header = ["project", "grid", "candidates", "truth", "rank", "confidence", "note"]
    print("\t".join(header))
    for row in rows:
        print("\t".join(str(row.get(key, "-")) for key in header))

    ranked = [row for row in rows if isinstance(row["rank"], int)]
    if ranked:
        total = len(ranked)
        for top_k in (1, 3, 6):
            hits = sum(1 for row in ranked if row["rank"] <= top_k)
            print(f"hit@{top_k}: {hits}/{total} ({100 * hits / total:.0f}%)")


if __name__ == "__main__":
    main()
