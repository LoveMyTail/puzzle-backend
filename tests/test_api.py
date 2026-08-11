"""API tests for project creation and box-cover calibration."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture()
def client(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("PUZZLE_DATA_DIR", str(tmp_path / "data"))
    return TestClient(main.app)


def _box_photo_bytes() -> bytes:
    image = np.full((300, 400, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _create_project(client: TestClient) -> dict:
    resp = client.post(
        "/api/projects",
        files={"box_photo": ("box.jpg", _box_photo_bytes(), "image/jpeg")},
        data={"name": "My Puzzle", "piece_count": "1000", "width_cm": "70", "height_cm": "50"},
    )
    assert resp.status_code == 200
    return resp.json()


def test_create_project_estimates_grid(client: TestClient) -> None:
    body = _create_project(client)
    assert body["estimated_rows"] == 27
    assert body["estimated_cols"] == 37
    assert body["calibration_status"] == "created"


def test_calibrate_project(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    points = [[0, 0], [400, 0], [400, 300], [0, 300]]
    resp = client.put(f"/api/projects/{project_id}/calibration", json={"points": points})
    assert resp.status_code == 200
    body = resp.json()
    assert body["cell_count"] == 27 * 37
    assert (body["rows"], body["cols"]) == (27, 37)

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["status"] == "calibrated"
    assert detail["calibration"]["cell_count"] == 27 * 37


def test_calibrate_rejects_bad_points(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(f"/api/projects/{project_id}/calibration", json={"points": [[0, 0], [1, 1]]})
    assert resp.status_code == 422


def test_calibrate_unknown_project_returns_404(client: TestClient) -> None:
    resp = client.put(
        "/api/projects/does-not-exist/calibration",
        json={"points": [[0, 0], [1, 0], [1, 1], [0, 1]]},
    )
    assert resp.status_code == 404
