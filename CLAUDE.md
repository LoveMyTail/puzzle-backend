# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python backend for a **jigsaw-puzzle assistant app**. Target use case: 1000-piece puzzles with large areas of near-identical colors are hard to assemble manually. The backend:

1. Takes a photo of the puzzle box cover, calibrates a reference grid (M1).
2. Takes a photo of the partially assembled board, aligns it to that grid, and locates every gap (M2).
3. Takes a photo of a single piece, segments it, and encodes its four edges as dense shape signatures (M3).
4. Matches the piece against all board gaps, scoring all four rotations, and returns Top-3 candidates with confidence (M4).

Matching is deliberately **shape-only** — no color/texture features are used, because the target scenario is large areas of near-identical color. There is currently no piece database and no reference-image color matching; a piece is matched purely by edge geometry against the board's receiving edges.

This repo is the Python service only; the app (mobile/client) is a separate project.

## Setup

- Only interpreter available: system Python **3.9.6** (`/usr/bin/python3`, arm64). No brew/pyenv.
- Virtualenv lives at `.venv/` (git-ignored). Create it and install all deps (runtime + dev) with:

  `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`

## Commands

- Run the API: `.venv/bin/uvicorn main:app --reload` → http://127.0.0.1:8000
- Run tests: `.venv/bin/python -m pytest`
- Lint: `.venv/bin/ruff check .`
- Verify a pipeline stage against a real photo: `scripts/verify_*.py --help` (each script takes an image path, grid/corner args, and writes previews to an `--out` dir).

## Environment gotchas

- **Python 3.9 limits dependency choices.** Many current releases require ≥3.10 (e.g. fastapi ≥0.129, numpy ≥2.1). pip silently falls back to the newest version still supporting 3.9 (currently fastapi 0.128.8, numpy 2.0.2, opencv-python 5.0.0.93). When adding dependencies, expect the same fallback — or "no matching distribution" if the package dropped 3.9 entirely.
- Dev-tool versions are **pinned** in `requirements-dev.txt`. Keep them pinned: during initial setup a flaky network once made pip resolve `pytest` to 3.2.5 (a 2017 release), which broke the modern pytest config.
- `opencv-python` resolves to the 5.0.0.93 line (OpenCV 5 pre-release).

## Architecture

- `main.py` — FastAPI entry point; run via `uvicorn main:app`. Logging lives on the `puzzle` logger (level from `PUZZLE_LOG_LEVEL`, default INFO; DEBUG emits one line per evaluated gap in the matcher).
- `puzzle/vision/` — image processing (OpenCV):
  - `calibration.py` (M1) — `estimate_grid` (rows/cols from piece count + aspect ratio), `warp_box_photo` (perspective-correct the box cover from 4 marked corners), `slice_grid`, `draw_grid_overlay`.
  - `board.py` (M2) — `segment_board` (background = median of the image border, pixels beyond `color_distance=40` are puzzle), `align_board_to_grid` (warp photo + mask onto a fixed grid of `CELL_PX=64` px/cell), `classify_cells` (fill ratio ≥ `FILL_RATIO=0.5`), `find_gaps` (empty cells adjacent to filled ones), `extract_receiving_edges` (extreme filled pixel per scan line in the neighbor cell + half the gap — preserves tabs/blanks protruding into the gap), `render_gap_preview`.
  - `piece.py` (M3) — `segment_piece` (largest connected component), `piece_contour`, `split_sides` (four sides split at `minAreaRect` corners), `resample_contour` (arc-length resample to `SIDE_SAMPLES=64` points), `side_profile` (signed distance from the side's chord, normalized by chord length → invariant to translation/rotation/uniform scale), `build_signature`, `validate_piece_resolution` (rejects pieces with a side shorter than `MIN_SIDE_PX=64`).
- `puzzle/solver/matcher.py` (M4) — gap matching:
  - `edge_profile` encodes a gap receiving edge exactly like a piece side.
  - `edge_distance` returns `min(direct, mirrored)` mean-absolute-difference — a piece edge and its neighbor's receiving edge are mirror images, so both orientations are tried.
  - `match_gap` tries all 4 rotations (cyclic side shift: `(DIRECTIONS[direction] - k) % 4`) and averages over the gap's directions.
  - `score_gaps` / `top_candidates` rank all gaps and return Top-3 with softmin confidence (temperature 0.1).
- `puzzle/db/projects.py` — lightweight **file-based** storage (no database): `data/projects/{project_id}/` holds `metadata.json` plus uploaded/derived images. Data dir defaults to `<repo>/data`, override with `PUZZLE_DATA_DIR`.
- `scripts/` — `verify_calibration.py`, `verify_board.py`, `verify_piece.py`, `verify_locate.py`: manual per-stage verification against real photos.
- `tests/` — pytest suite (42 tests): `test_smoke`, `test_calibration`, `test_board`, `test_piece`, `test_matcher`, `test_api` (end-to-end via `TestClient`). Tests synthesize pieces/boards procedurally (no image fixtures).

### API

| Method & path | Preconditions | Purpose |
|---|---|---|
| `GET /health` | — | liveness |
| `POST /api/projects` | box photo, piece count, finished size cm | create project, estimate grid |
| `PUT /api/projects/{id}/calibration` | project exists | 4-corner perspective warp of box photo, slice grid |
| `PUT /api/projects/{id}/board` | calibrated | align board photo to grid, classify cells, find gaps; **422 if source photo < 64 px/cell** |
| `POST /api/projects/{id}/locate` | calibrated + board uploaded | build piece signature, score gaps, return Top-3 candidates; **422 if piece side < 64 px** |
| `GET /api/projects/{id}` | project exists | read `metadata.json` |

### Key invariants (keep in mind when changing code)

- **Scale invariance is the design contract.** Every feature is chord-normalized and arc-length resampled to a fixed 64 points, and the board is always warped onto the fixed 64 px/cell grid, so absolute photo sizes cancel out. The resolution guards (`MIN_SIDE_PX`, `MIN_SOURCE_CELL_PX`, both 64) protect the fidelity floor below which tab/blank detail is quantized away — keep them consistent with `CELL_PX` / `SIDE_SAMPLES`.
- **Receiving edges carry the neighbor's tab/blank shape**, not just the boundary line; the matcher depends on it (`test_real_chain_ranks_true_gap_first` regresses this).
- **Gap coordinates are 1-based** rows/cols in the API and in `find_gaps` output; vision code converts to 0-based internally.
- Do not switch the matcher to color/texture features without revisiting the project overview — shape-only is deliberate.
