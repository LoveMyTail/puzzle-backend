"""FastAPI entry point for the puzzle-assistant backend.

Run locally with:  uvicorn main:app --reload
"""

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from puzzle.db import projects as project_store
from puzzle.vision import calibration

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


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    return metadata
