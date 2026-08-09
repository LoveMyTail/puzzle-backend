# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Python backend for a **jigsaw-puzzle assistant app**. Target use case: 1000-piece puzzles with large areas of near-identical colors are hard to assemble manually. The backend will:

1. Take photos of individual puzzle pieces.
2. Segment each piece, identify its edges and shape features, and capture its color/texture signature.
3. Persist recognized pieces in a database.
4. Determine where each piece belongs in the overall puzzle image (position + rotation).

This repo is the Python service only; the app (mobile/client) is a separate project.

## Setup

- Only interpreter available: system Python **3.9.6** (`/usr/bin/python3`, arm64). No brew/pyenv.
- Virtualenv lives at `.venv/` (git-ignored). Create it and install all deps (runtime + dev) with:

  `python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt`

## Commands

- Run the API: `.venv/bin/uvicorn main:app --reload` → http://127.0.0.1:8000
- Run tests: `.venv/bin/python -m pytest`
- Lint: `.venv/bin/ruff check .`

## Environment gotchas

- **Python 3.9 limits dependency choices.** Many current releases require ≥3.10 (e.g. fastapi ≥0.129, numpy ≥2.1). pip silently falls back to the newest version still supporting 3.9 (currently fastapi 0.128.8, numpy 2.0.2, opencv-python 5.0.0.93). When adding dependencies, expect the same fallback — or "no matching distribution" if the package dropped 3.9 entirely.
- Dev-tool versions are **pinned** in `requirements-dev.txt`. Keep them pinned: during initial setup a flaky network once made pip resolve `pytest` to 3.2.5 (a 2017 release), which broke the modern pytest config.
- `opencv-python` resolves to the 5.0.0.93 line (OpenCV 5 pre-release).

## Architecture

- `main.py` — FastAPI entry point; the `app` object is created here and run via `uvicorn main:app`. Currently exposes only `GET /health`.
- `puzzle/` — core package:
  - `puzzle/vision/` — planned: piece segmentation, edge/shape feature extraction (OpenCV).
  - `puzzle/solver/` — planned: matching/placement, computing position + rotation per piece.
  - `puzzle/db/` — planned: persistence of piece fingerprints and board state.

  These subpackages are **empty scaffolding** (docstring-only `__init__.py`), no real logic yet — the work above is the project's roadmap.
- `tests/test_smoke.py` — smoke tests: imports the FastAPI app, checks `GET /health` via `TestClient`, and verifies `cv2` imports.
- `requirements.txt` holds runtime deps; `requirements-dev.txt` adds test/lint tools and includes `-r requirements.txt`.
