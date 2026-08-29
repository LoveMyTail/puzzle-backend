"""FastAPI entry point for the puzzle-assistant backend.

Run locally with:  uvicorn main:app --reload
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

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


def _is_heic(content: bytes) -> bool:
    """Detect HEIC/HEIF magic bytes so the API can return an actionable error."""
    if len(content) < 12:
        return False
    brand = content[4:12]
    return brand.startswith(b"ftyphei") or brand.startswith(b"ftypmif")


def _save_operation(
    project_id: str,
    op: str,
    files: dict[str, bytes] | None = None,
    **fields: object,
) -> dict:
    """Persist one replayable operation under the project directory."""
    seq = project_store.next_operation_seq(project_id)
    saved_paths: dict[str, str] = {}
    if files:
        ops_dir = project_store.project_dir(project_id) / "ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            path = ops_dir / f"{seq:03d}_{op}_{name}"
            path.write_bytes(data)
            saved_paths[name] = str(path)
    return project_store.append_operation(
        project_id,
        {"op": op, "files": saved_paths, **fields},
        seq=seq,
    )


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
    box_photo_bytes = box_photo.file.read()
    metadata = project_store.create_project(
        name=name,
        piece_count=piece_count,
        width_cm=width_cm,
        height_cm=height_cm,
        rows=rows,
        cols=cols,
        box_photo=box_photo_bytes,
    )
    _save_operation(
        metadata["id"],
        "create",
        {"box_photo.jpg": box_photo_bytes},
        name=name,
        piece_count=piece_count,
        width_cm=width_cm,
        height_cm=height_cm,
        rows=rows,
        cols=cols,
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
    _save_operation(
        project_id,
        "calibrate",
        points=corners,
        rows=rows,
        cols=cols,
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
    rows: Optional[int] = Form(None),  # noqa: UP045 - project targets Python 3.9
    cols: Optional[int] = Form(None),  # noqa: UP045 - project targets Python 3.9
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

    if rows is not None or cols is not None:
        new_rows = rows if rows is not None else int(metadata["rows"])
        new_cols = cols if cols is not None else int(metadata["cols"])
        if new_rows < 1 or new_cols < 1:
            raise HTTPException(
                status_code=422, detail="rows and cols must be positive"
            )
        project_store.update_project(project_id, rows=new_rows, cols=new_cols)
        metadata = project_store.get_project(project_id)
    rows = int(metadata["rows"])
    cols = int(metadata["cols"])
    content = board_photo.file.read()
    if _is_heic(content):
        raise HTTPException(
            status_code=422,
            detail="HEIC/HEIF 图片不受支持，请先转换为 JPEG 或 PNG 再上传",
        )
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=422, detail="unreadable board photo")
    logger.info(
        "board_upload_start project=%s photo_bytes=%d photo_shape=%s "
        "corners=%s rows=%s cols=%s",
        project_id,
        len(content),
        list(image.shape),
        [[round(float(v), 1) for v in p] for p in corner_points],
        rows,
        cols,
    )
    corners_auto = False
    if board.is_full_frame_corners(corner_points, image.shape):
        detected = board.estimate_board_quad(image)
        if detected is not None:
            corner_points = [tuple(float(value) for value in point) for point in detected]
            corners_auto = True
            logger.info(
                "board_quad_auto_detected project=%s corners=%s",
                project_id,
                corner_points,
            )
        else:
            logger.warning(
                "board_quad_auto_detect_failed project=%s "
                "falling_back_to_full_frame",
                project_id,
            )
    background = None
    if not board.is_full_frame_corners(corner_points, image.shape):
        background = board.exterior_background(image, corner_points)
    try:
        board.validate_board_resolution(corner_points, rows, cols)
    except ValueError as exc:
        logger.warning(
            "board_resolution_rejected project=%s reason=%s", project_id, exc
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cell_px = board.adaptive_cell_px(rows, cols)
    warped, warped_mask = board.align_board_to_grid(
        image,
        corner_points,
        rows,
        cols,
        cell_px=cell_px,
        background=background,
    )
    filled = board.classify_cells(warped_mask, rows, cols, cell_px=cell_px)
    gaps = board.find_gaps(filled)
    preview = board.render_gap_preview(
        warped, filled, gaps, rows, cols, cell_px=cell_px
    )
    quality = board.alignment_quality(
        warped, warped_mask, rows, cols, cell_px=cell_px
    )
    logger.info(
        "board_uploaded project=%s corners_auto=%s mask_coverage=%.3f "
        "filled_cells=%d gaps=%d alignment=%s",
        project_id,
        corners_auto,
        float((warped_mask > 0).mean()),
        int(filled.sum()),
        len(gaps),
        quality,
    )

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
            "cell_px": cell_px,
            "photo_path": str(photo_path),
            "warped_path": str(warped_path),
            "preview_path": str(preview_path),
            "mask_path": str(mask_path),
        },
    )
    _save_operation(
        project_id,
        "board",
        {"board_photo.jpg": content},
        corners=corner_points,
        rows=rows,
        cols=cols,
        filled_cells=int(filled.sum()),
        gaps_count=len(gaps),
        alignment=quality,
        corners_auto=corners_auto,
    )
    return {
        "project_id": project_id,
        "rows": rows,
        "cols": cols,
        "filled_cells": int(filled.sum()),
        "gaps": gaps,
        "alignment": quality,
        "preview_path": str(preview_path),
    }


@app.post("/api/projects/{project_id}/locate")
def locate_piece(
    project_id: str,
    piece_photo: UploadFile = File(...),
    exclude: Optional[str] = Form(None),  # noqa: UP045 - project targets Python 3.9
) -> dict:
    content = piece_photo.file.read()
    logger.info(
        "piece_match_start project=%s photo_bytes=%d",
        project_id,
        len(content),
    )
    if _is_heic(content):
        raise HTTPException(
            status_code=422,
            detail="HEIC/HEIF 图片不受支持，请先转换为 JPEG 或 PNG 再上传",
        )
    excluded_positions: set[tuple[int, int]] = set()
    if exclude:
        try:
            excluded_positions = {
                (int(row), int(col)) for row, col in json.loads(exclude)
            }
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail="exclude must be a list of [row, col] pairs"
            ) from exc
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
        board_photo_path = board_info.get("photo_path")
        if board_photo_path is None or not Path(board_photo_path).exists():
            raise HTTPException(
                status_code=409,
                detail="board photo is missing; re-upload the board photo",
            )
        board_image = cv2.imread(str(board_photo_path))
        if board_image is None:
            raise HTTPException(status_code=500, detail="board photo is unreadable")
        corners = board_info.get("corners")
        if not corners:
            raise HTTPException(
                status_code=409,
                detail="board corners are missing; re-upload the board photo",
            )

        image = cv2.imdecode(
            np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is None:
            raise HTTPException(status_code=422, detail="unreadable piece photo")
        piece_photo_path = project_store.project_dir(project_id) / "piece_photo.jpg"
        cv2.imwrite(str(piece_photo_path), image)
        project_store.update_project(project_id, last_piece_photo=str(piece_photo_path))

        try:
            signature = piece.build_signature(image)
            piece.validate_piece_resolution(signature)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        logger.info(
            "piece_signature project=%s contour_points=%d side_lengths=%s",
            project_id,
            len(signature.contour),
            [round(float(length), 1) for length in signature.side_lengths],
        )

        rows = int(metadata["rows"])
        cols = int(metadata["cols"])
        cell_px = int(
            board_info.get("cell_px")
            or board.adaptive_cell_px(rows, cols)
        )
        background = None
        if not board.is_full_frame_corners(corners, board_image.shape):
            background = board.exterior_background(board_image, corners)
        _, mask = board.align_board_to_grid(
            board_image,
            corners,
            rows,
            cols,
            cell_px=cell_px,
            background=background,
        )
        filled = board.classify_cells(mask, rows, cols, cell_px=cell_px)
        gaps_with_edges = []
        for gap in board.find_gaps(filled):
            if (int(gap["row"]), int(gap["col"])) in excluded_positions:
                continue
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
        candidates = matcher.top_candidates(signature, gaps_with_edges, k=6)
        gaps_by_pos = {
            (gap["row"], gap["col"]): gap["edges"] for gap in gaps_with_edges
        }
        warped_for_continuity = None
        warped_path = board_info.get("warped_path")
        if warped_path and Path(warped_path).exists():
            warped_for_continuity = cv2.imread(str(warped_path))
        continuity_weight = float(
            os.environ.get("PUZZLE_CONTINUITY_WEIGHT", "0.0")
        )
        color_weight = float(os.environ.get("PUZZLE_COLOR_WEIGHT", "0.08"))
        piece_core_color = None
        if warped_for_continuity is not None and color_weight > 0:
            cx, cy = signature.contour.mean(axis=0)
            bbox = cv2.boundingRect(signature.contour.astype(np.int32))
            radius = max(8, int(0.08 * (bbox[2] + bbox[3]) / 2))
            core = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.circle(core, (int(cx), int(cy)), radius, 255, -1)
            ys, xs = np.nonzero(core)
            piece_core_color = image[ys, xs].mean(axis=0)
        for candidate in candidates:
            candidate["shape_score"] = candidate["score"]
            candidate["continuity"] = 0.0
            candidate["color_distance"] = None
            if warped_for_continuity is not None:
                candidate["continuity"] = matcher.edge_continuity(
                    warped_for_continuity,
                    cell_px,
                    image,
                    signature,
                    gaps_by_pos.get(
                        (candidate["row"], candidate["col"]), {}
                    ),
                    candidate["rotation"],
                )
                if piece_core_color is not None:
                    neighbor_distances = []
                    for direction, points in gaps_by_pos.get(
                        (candidate["row"], candidate["col"]), {}
                    ).items():
                        r, c = candidate["row"] - 1, candidate["col"] - 1
                        if direction == "top":
                            r -= 1
                        elif direction == "bottom":
                            r += 1
                        elif direction == "left":
                            c -= 1
                        else:
                            c += 1
                        if 0 <= r < rows and 0 <= c < cols:
                            cell = warped_for_continuity[
                                r * cell_px : (r + 1) * cell_px,
                                c * cell_px : (c + 1) * cell_px,
                            ]
                            neighbor_distances.append(
                                float(
                                    np.linalg.norm(
                                        piece_core_color - cell.mean(axis=(0, 1))
                                    )
                                )
                            )
                    if neighbor_distances:
                        candidate["color_distance"] = min(neighbor_distances)
            candidate["score"] = (
                candidate["shape_score"]
                - continuity_weight * candidate["continuity"]
            )
        if color_weight > 0 and any(
            c["color_distance"] is not None for c in candidates
        ):
            distances = [
                c["color_distance"]
                for c in candidates
                if c["color_distance"] is not None
            ]
            low, high = min(distances), max(distances)
            span = max(high - low, 1e-6)
            for candidate in candidates:
                if candidate["color_distance"] is None:
                    candidate["score"] += color_weight
                else:
                    normalized = (candidate["color_distance"] - low) / span
                    candidate["score"] += color_weight * normalized
        candidates.sort(key=lambda item: item["score"])
        ambiguity_band = float(
            os.environ.get("PUZZLE_AMBIGUITY_BAND", "0.02")
        )
        if (
            len(candidates) >= 3
            and candidates[2]["score"] - candidates[0]["score"]
            < ambiguity_band
        ):
            # Top scores are essentially tied: keep the wider pool so the
            # true gap is more likely to be shown for manual comparison.
            candidates = candidates[:6]
        else:
            candidates = candidates[:3]
        matcher.reweight_confidences(candidates)
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        total_ms = round((time.perf_counter() - started_all) * 1000, 1)
        rotation_band = float(
            os.environ.get("PUZZLE_ROTATION_BAND", "0.02")
        )
        for candidate in candidates:
            edges = gaps_by_pos.get(
                (candidate["row"], candidate["col"]), {}
            )
            candidate["directions"] = len(edges)
            per_rotation = []
            for k in range(4):
                scores = []
                for direction, points in edges.items():
                    side = (matcher.DIRECTIONS[direction] - k) % 4
                    scores.append(
                        matcher.edge_distance(
                            signature.sides[side],
                            matcher.edge_profile(
                                np.asarray(points, dtype=np.float32)
                            ),
                        )
                    )
                if scores:
                    worst = max(scores)
                    per_rotation.append(
                        min(scores)
                        + matcher.GAP_WORST_PENALTY
                        * max(0.0, worst - matcher.GAP_WORST_THRESHOLD)
                    )
                else:
                    per_rotation.append(float("inf"))
            order = sorted(range(4), key=lambda k: per_rotation[k])
            candidate["rotation_confident"] = (
                per_rotation[order[1]] - per_rotation[order[0]]
                >= rotation_band
            )
        corners = board_info.get("corners")
        board_image = None
        board_photo_path = board_info.get("photo_path")
        if board_photo_path:
            board_image = cv2.imread(str(board_photo_path))
        board_shape = None if board_image is None else board_image.shape
        for candidate in candidates:
            candidate["region"] = None
            if corners is not None and board_shape is not None:
                candidate["region"] = board.cell_region_in_photo(
                    corners,
                    rows,
                    cols,
                    cell_px,
                    board_shape,
                    candidate["row"],
                    candidate["col"],
                )
        if os.environ.get("PUZZLE_WRITE_DEBUG_IMAGES", "0") == "1":
            debug_dir = project_store.project_dir(project_id)
            piece_overlay_path = debug_dir / "piece_mask_overlay.jpg"
            piece_overlay = image.copy()
            cv2.drawContours(
                piece_overlay,
                [signature.contour.astype(np.int32)],
                -1,
                (0, 0, 255),
                4,
            )
            cv2.imwrite(str(piece_overlay_path), piece_overlay)
            board_edges_path = debug_dir / "board_edges_debug.jpg"
            warped_path = board_info.get("warped_path")
            board_debug = None
            if warped_path and Path(warped_path).exists():
                board_debug = cv2.imread(str(warped_path))
            if board_debug is not None:
                for gap in gaps_with_edges:
                    for points in gap["edges"].values():
                        for x, y in points:
                            cv2.circle(
                                board_debug,
                                (int(x), int(y)),
                                2,
                                (0, 255, 0),
                                -1,
                            )
                for rank, candidate in enumerate(candidates, start=1):
                    x0 = (candidate["col"] - 1) * cell_px
                    y0 = (candidate["row"] - 1) * cell_px
                    cv2.rectangle(
                        board_debug,
                        (x0, y0),
                        (x0 + cell_px - 1, y0 + cell_px - 1),
                        (0, 165, 255) if rank == 1 else (0, 60, 255),
                        3,
                    )
                    cv2.putText(
                        board_debug,
                        str(rank),
                        (x0 + cell_px // 2 - 8, y0 + cell_px // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 0),
                        5,
                        cv2.LINE_AA,
                    )
                cv2.imwrite(str(board_edges_path), board_debug)
            project_store.update_project(
                project_id,
                debug_piece_overlay=str(piece_overlay_path),
                debug_board_edges=str(board_edges_path),
            )
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
                    "directions": len(gaps_by_pos.get((candidate["row"], candidate["col"]), {})),
                    "continuity": round(candidate["continuity"], 3),
                    "color_dist": (
                        round(candidate["color_distance"], 1)
                        if candidate["color_distance"] is not None
                        else None
                    ),
                }
                for candidate in candidates
            ],
            latency_ms,
            total_ms,
        )
        if logger.isEnabledFor(logging.DEBUG):
            for candidate in candidates:
                edges = gaps_by_pos.get(
                    (candidate["row"], candidate["col"]), {}
                )
                per_direction = {}
                for direction, points in edges.items():
                    side = (matcher.DIRECTIONS[direction] - candidate["rotation"]) % 4
                    per_direction[direction] = round(
                        matcher.edge_distance(
                            signature.sides[side],
                            matcher.edge_profile(
                                np.asarray(points, dtype=np.float32)
                            ),
                        ),
                        4,
                    )
                logger.debug(
                    "candidate_detail project=%s row=%d col=%d "
                    "score=%.4f rotation=%d per_direction=%s",
                    project_id,
                    candidate["row"],
                    candidate["col"],
                    candidate["score"],
                    candidate["rotation"],
                    per_direction,
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
        _save_operation(
            project_id,
            "locate",
            {"piece_photo.jpg": content},
            exclude=sorted(list(excluded_positions)),
            latency_ms=latency_ms,
            candidates=[
                {
                    "row": candidate["row"],
                    "col": candidate["col"],
                    "score": round(float(candidate["score"]), 4),
                    "rotation": candidate["rotation"],
                    "confidence": round(float(candidate["confidence"]), 3),
                    "rotation_confident": candidate.get(
                        "rotation_confident", True
                    ),
                    "directions": candidate.get("directions", 0),
                }
                for candidate in candidates
            ],
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


@app.get("/api/projects/{project_id}/operations")
def list_operations(project_id: str) -> dict:
    """Return the replayable operation journal for a project."""
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    return {
        "project_id": project_id,
        "operations": project_store.read_operations(project_id),
    }


@app.put("/api/projects/{project_id}/placed")
def mark_placed(project_id: str, payload: dict) -> dict:
    """Record that the user placed a piece into a cell."""
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    try:
        row = int(payload["row"])
        col = int(payload["col"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="row and col are required integers"
        ) from exc
    if row < 1 or col < 1:
        raise HTTPException(
            status_code=422, detail="row and col must be positive"
        )
    record = _save_operation(project_id, "placed", row=row, col=col)
    return {
        "project_id": project_id,
        "row": row,
        "col": col,
        "seq": record["seq"],
    }


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    metadata = project_store.get_project(project_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="project not found")
    return metadata


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
