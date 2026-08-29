"""Lightweight file-based project storage for the MVP.

Each project is a directory under ``data/projects/{project_id}`` holding
``metadata.json`` plus the uploaded and derived images. No database is used
yet; the layout is documented in the tech spec.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_operation_lock = threading.Lock()


def data_dir() -> Path:
    return Path(os.environ.get("PUZZLE_DATA_DIR", DEFAULT_DATA_DIR))


def projects_dir() -> Path:
    return data_dir() / "projects"


def project_dir(project_id: str) -> Path:
    return projects_dir() / project_id


def project_meta_path(project_id: str) -> Path:
    return project_dir(project_id) / "metadata.json"


def box_photo_path(project_id: str) -> Path:
    return project_dir(project_id) / "box_photo.jpg"


def operation_log_path(project_id: str) -> Path:
    return project_dir(project_id) / "operations.jsonl"


def next_operation_seq(project_id: str) -> int:
    """Return the next per-project operation sequence number."""
    path = operation_log_path(project_id)
    if not path.exists():
        return 1
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count + 1


def append_operation(
    project_id: str, record: dict, seq: int | None = None
) -> dict:
    """Append one replayable operation record to the project journal."""
    with _operation_lock:
        if seq is None:
            seq = next_operation_seq(project_id)
        record = {
            "seq": seq,
            "ts": time.time(),
            **record,
        }
        path = operation_log_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_operations(project_id: str) -> list[dict]:
    """Read the project operation journal in chronological order."""
    path = operation_log_path(project_id)
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def new_project_id() -> str:
    return uuid.uuid4().hex[:12]


def create_project(
    *,
    name: str,
    piece_count: int,
    width_cm: float,
    height_cm: float,
    rows: int,
    cols: int,
    box_photo: bytes,
) -> dict:
    """Create a project directory, store the box photo, and persist metadata."""
    project_id = new_project_id()
    directory = project_dir(project_id)
    directory.mkdir(parents=True)
    (directory / "box_photo.jpg").write_bytes(box_photo)
    metadata = {
        "id": project_id,
        "name": name,
        "piece_count": piece_count,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "rows": rows,
        "cols": cols,
        "status": "created",
        "calibration": None,
    }
    _write_metadata(project_id, metadata)
    return metadata


def get_project(project_id: str) -> dict | None:
    path = project_meta_path(project_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_project(project_id: str, **changes: object) -> dict:
    metadata = get_project(project_id)
    if metadata is None:
        raise KeyError(f"project not found: {project_id}")
    metadata.update(changes)
    _write_metadata(project_id, metadata)
    return metadata


def _write_metadata(project_id: str, metadata: dict) -> None:
    path = project_meta_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
