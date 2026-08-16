"""FastAPI entry point for the puzzle-assistant backend.

Run locally with:  uvicorn main:app --reload
"""

import json
import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from puzzle.db import projects as project_store
from puzzle.solver import matcher
from puzzle.vision import board, calibration, piece


def _configure_logging() -> logging.Logger:
    """Configure the ``puzzle`` logger with an independent stderr handler.

    Level comes from the ``PUZZLE_LOG_LEVEL`` env var (default INFO); DEBUG
    additionally emits one line per evaluated gap in the matcher. Keeping the
    handler on the ``puzzle`` logger (with ``propagate=False``) makes the
    matching logs predictable regardless of uvicorn's own log configuration.
    """
    level = os.environ.get("PUZZLE_LOG_LEVEL", "INFO").upper()
    puzzle_logger = logging.getLogger("puzzle")
    if not puzzle_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        puzzle_logger.addHandler(handler)
    puzzle_logger.setLevel(level)
    puzzle_logger.propagate = False
    return puzzle_logger


logger = _configure_logging()

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
    try:
        board.validate_board_resolution(corner_points, rows, cols)
    except ValueError as exc:
        logger.warning(
            "board_resolution_rejected project=%s reason=%s", project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
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
            "cell_px": board.CELL_PX,
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
    content = piece_photo.file.read()
    logger.info(
        "piece_match_start project=%s photo_bytes=%d",
        project_id,
        len(content),
    )
    started_all = time.perf_counter()
    try:
        metadata = project_store.get_project(project_id)
        if metadata is None:
            raise HTTPException(status_code=404, detail="project not found")
        if metadata.get("status") != "calibrated":
            raise HTTPException(
                status_code=409, detail="project must be calibrated first"
            )
        board_info = metadata.get("board")
        if not board_info or not board_info.get("gaps"):
            raise HTTPException(
                status_code=409,
                detail="board state is required before locating",
            )
        mask_path = board_info.get("mask_path")
        if mask_path is None or not Path(mask_path).exists():
            raise HTTPException(
                status_code=409,
                detail="board mask is missing; re-upload the board photo",
            )

        image = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise HTTPException(status_code=422, detail="unreadable piece photo")

        signature = piece.build_signature(image)
        try:
            piece.validate_piece_resolution(signature)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info(
            "piece_signature project=%s contour_points=%d side_lengths=%s",
            project_id,
            len(signature.contour),
            [round(float(length), 1) for length in signature.side_lengths],
        )

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise HTTPException(status_code=500, detail="board mask is unreadable")
        rows = int(metadata["rows"])
        cols = int(metadata["cols"])
        stored_px = board_info.get("cell_px")
        if stored_px is not None:
            cell_px = int(stored_px)
        elif mask.shape[1] % cols == 0:
            cell_px = mask.shape[1] // cols
        else:
            cell_px = board.CELL_PX
        filled = board.classify_cells(mask, rows, cols, cell_px=cell_px)
        gaps_with_edges = []
        for gap in board_info["gaps"]:
            edges = board.extract_receiving_edges(
                mask,
                rows,
                cols,
                int(gap["row"]),
                int(gap["col"]),
                cell_px=cell_px,
                filled=filled,
            )
            gaps_with_edges.append(
                {"row": int(gap["row"]), "col": int(gap["col"]), "edges": edges}
            )
        logger.info(
            "piece_match_gaps project=%s gaps=%d receiving_edges=%d",
            project_id,
            len(gaps_with_edges),
            sum(len(gap["edges"]) for gap in gaps_with_edges),
        )

        started = time.perf_counter()
        candidates = matcher.top_candidates(signature, gaps_with_edges, k=3)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        total_ms = round((time.perf_counter() - started_all) * 1000, 1)
        logger.info(
            "piece_match_done project=%s candidates=%s match_latency_ms=%.1f "
            "total_ms=%.1f",
            project_id,
            [
                {
                    "row": candidate["row"],
                    "col": candidate["col"],
                    "score": round(candidate["score"], 4),
                    "rotation": candidate["rotation"],
                    "confidence": round(candidate["confidence"], 3),
                }
                for candidate in candidates
            ],
            latency_ms,
            total_ms,
        )
        preview_url = None
        warped_path = board_info.get("warped_path")
        if warped_path and Path(warped_path).exists():
            warped_image = cv2.imread(str(warped_path))
            if warped_image is not None:
                preview = board.render_candidate_preview(
                    warped_image, candidates, cell_px=cell_px
                )
                locate_preview_path = (
                    project_store.project_dir(project_id) / "locate_preview.jpg"
                )
                cv2.imwrite(str(locate_preview_path), preview)
                project_store.update_project(
                    project_id,
                    board={
                        **board_info,
                        "locate_preview_path": str(locate_preview_path),
                    },
                )
                preview_url = f"/api/projects/{project_id}/locate/preview"
        if preview_url is None:
            logger.warning(
                "locate_preview_skipped project=%s reason=warped image missing",
                project_id,
            )
        return {
            "project_id": project_id,
            "candidates": candidates,
            "latency_ms": latency_ms,
            "preview_url": preview_url,
            "board_preview_url": f"/api/projects/{project_id}/board/preview",
        }
    except HTTPException as exc:
        logger.warning(
            "piece_match_aborted project=%s reason=%s",
            project_id,
            exc.detail,
        )
        raise
    except Exception:
        logger.exception("piece_match_failed project=%s", project_id)
        raise


@app.get("/api/projects/{project_id}/board/preview")
def board_preview(project_id: str) -> FileResponse:
    """Serve the gap-tinted board overview (filled cells green, gaps red)."""
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    board_info = metadata.get("board") or {}
    path = board_info.get("preview_path")
    if path is None or not Path(path).exists():
        raise HTTPException(
            status_code=404, detail="board preview not found; upload the board photo first"
        )
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/api/projects/{project_id}/locate/preview")
def locate_preview(project_id: str) -> FileResponse:
    """Serve the last locate result: warped board with Top-1..3 gaps marked."""
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    board_info = metadata.get("board") or {}
    path = board_info.get("locate_preview_path")
    if path is None or not Path(path).exists():
        raise HTTPException(
            status_code=404, detail="locate preview not found; run locate first"
        )
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    return metadata


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
