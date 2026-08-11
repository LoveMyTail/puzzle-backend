"""FastAPI entry point for the puzzle-assistant backend.

Run locally with:  uvicorn main:app --reload
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from puzzle.db import projects as project_store
from puzzle.solver import matcher
from puzzle.vision import board, calibration, piece

app = FastAPI(title="Puzzle Assistant Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/projects")
def create_project(
    box_photo: UploadFile = File(...),
    name: str = Form(""),
    piece_count: int = Form(...),
    width_cm: float = Form(...),
    height_cm: float = Form(...),
) -> dict:
    rows, cols = calibration.estimate_grid(piece_count, width_cm, height_cm)
    metadata = project_store.create_project(
        name=name,
        piece_count=piece_count,
        width_cm=width_cm,
        height_cm=height_cm,
        rows=rows,
        cols=cols,
        box_photo=box_photo.file.read(),
    )
    return {
        "project_id": metadata["id"],
        "name": metadata["name"],
        "piece_count": metadata["piece_count"],
        "width_cm": metadata["width_cm"],
        "height_cm": metadata["height_cm"],
        "estimated_rows": metadata["rows"],
        "estimated_cols": metadata["cols"],
        "calibration_status": metadata["status"],
    }


@app.put("/api/projects/{project_id}/calibration")
def calibrate_project(project_id: str, payload: dict) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")

    points = payload.get("points")
    if not isinstance(points, list) or len(points) != 4:
        raise HTTPException(status_code=422, detail="points must be four [x, y] pairs")
    try:
        corners = [(float(x), float(y)) for x, y in points]
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="points must be four [x, y] pairs") from exc

    rows = int(payload.get("rows", metadata["rows"]))
    cols = int(payload.get("cols", metadata["cols"]))
    if rows < 1 or cols < 1:
        raise HTTPException(status_code=422, detail="rows and cols must be positive")

    image = cv2.imread(str(project_store.box_photo_path(project_id)))
    if image is None:
        raise HTTPException(status_code=500, detail="box photo is missing or unreadable")

    warped = calibration.warp_box_photo(image, corners)
    cells = calibration.slice_grid(warped, rows, cols)
    directory = project_store.project_dir(project_id)
    warped_path = directory / "box_warped.jpg"
    preview_path = directory / "grid_preview.jpg"
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(preview_path), calibration.draw_grid_overlay(warped, rows, cols))

    project_store.update_project(
        project_id,
        status="calibrated",
        calibration={
            "points": [[float(x), float(y)] for x, y in corners],
            "rows": rows,
            "cols": cols,
            "cell_count": rows * cols,
            "warped_path": str(warped_path),
            "preview_path": str(preview_path),
        },
    )
    return {
        "project_id": project_id,
        "rows": rows,
        "cols": cols,
        "cell_count": len(cells),
        "warped_width": int(warped.shape[1]),
        "warped_height": int(warped.shape[0]),
    }


@app.put("/api/projects/{project_id}/board")
def update_board(
    project_id: str,
    board_photo: UploadFile = File(...),
    corners: str = Form(...),
) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    if metadata.get("status") != "calibrated":
        raise HTTPException(status_code=409, detail="project must be calibrated first")

    try:
        raw = json.loads(corners)
        corner_points = [(float(x), float(y)) for x, y in raw]
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="corners must be four [x, y] pairs") from exc
    if len(corner_points) != 4:
        raise HTTPException(status_code=422, detail="corners must be four [x, y] pairs")

    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    content = board_photo.file.read()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="unreadable board photo")

    warped, warped_mask = board.align_board_to_grid(image, corner_points, rows, cols)
    filled = board.classify_cells(warped_mask, rows, cols)
    gaps = board.find_gaps(filled)
    preview = board.render_gap_preview(warped, filled, gaps, rows, cols)

    directory = project_store.project_dir(project_id)
    photo_path = directory / "board_photo.jpg"
    warped_path = directory / "board_warped.jpg"
    preview_path = directory / "board_preview.jpg"
    mask_path = directory / "board_mask.png"
    cv2.imwrite(str(photo_path), image)
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(mask_path), warped_mask)

    project_store.update_project(
        project_id,
        board={
            "corners": [[float(x), float(y)] for x, y in corner_points],
            "filled_cells": int(filled.sum()),
            "gaps": gaps,
            "photo_path": str(photo_path),
            "warped_path": str(warped_path),
            "preview_path": str(preview_path),
            "mask_path": str(mask_path),
        },
    )
    return {
        "project_id": project_id,
        "rows": rows,
        "cols": cols,
        "filled_cells": int(filled.sum()),
        "gaps": gaps,
        "preview_path": str(preview_path),
    }


@app.post("/api/projects/{project_id}/locate")
def locate_piece(project_id: str, piece_photo: UploadFile = File(...)) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    if metadata.get("status") != "calibrated":
        raise HTTPException(status_code=409, detail="project must be calibrated first")
    board_info = metadata.get("board")
    if not board_info or not board_info.get("gaps"):
        raise HTTPException(status_code=409, detail="board state is required before locating")
    mask_path = board_info.get("mask_path")
    if mask_path is None or not Path(mask_path).exists():
        raise HTTPException(
            status_code=409,
            detail="board mask is missing; re-upload the board photo",
        )

    content = piece_photo.file.read()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="unreadable piece photo")

    signature = piece.build_signature(image)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise HTTPException(status_code=500, detail="board mask is unreadable")
    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    gaps_with_edges = []
    for gap in board_info["gaps"]:
        edges = board.extract_receiving_edges(mask, rows, cols, int(gap["row"]), int(gap["col"]))
        gaps_with_edges.append(
            {"row": int(gap["row"]), "col": int(gap["col"]), "edges": edges}
        )

    started = time.perf_counter()
    candidates = matcher.top_candidates(signature, gaps_with_edges, k=3)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    return {
        "project_id": project_id,
        "candidates": candidates,
        "latency_ms": latency_ms,
    }


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    return metadata
