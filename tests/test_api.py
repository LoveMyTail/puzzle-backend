"""API tests for project creation and box-cover calibration."""

from pathlib import Path

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


def _board_photo_bytes() -> bytes:
    rows, cols = 10, 12
    grid_h, grid_w = rows * 24, cols * 24
    base = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for row in range(1, 5):
        for col in range(1, 6):
            x0, y0 = col * 24, row * 24
            cv2.rectangle(base, (x0, y0), (x0 + 23, y0 + 23), (30, 130, 210), -1)
    src = np.array(
        [[0, 0], [grid_w - 1, 0], [grid_w - 1, grid_h - 1], [0, grid_h - 1]],
        dtype=np.float32,
    )
    dst = np.array([[30, 25], [470, 35], [475, 395], [20, 385]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        base,
        matrix,
        (500, 420),
        flags=cv2.INTER_NEAREST,
        borderValue=(255, 255, 255),
    )
    ok, buf = cv2.imencode(".jpg", photo)
    assert ok
    return buf.tobytes()


def test_update_board_finds_gaps(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200

    corners = "[[30,25],[470,35],[475,395],[20,385]]"
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={"corners": corners},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["filled_cells"] > 0
    assert len(body["gaps"]) > 0
    assert body["gaps"][0]["directions"]

    detail = client.get(f"/api/projects/{project_id}").json()
    assert detail["board"]["filled_cells"] == body["filled_cells"]
    assert Path(detail["board"]["mask_path"]).exists()


def test_update_board_requires_calibration(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={"corners": "[[0,0],[1,0],[1,1],[0,1]]"},
    )
    assert resp.status_code == 409


def _piece_photo_bytes() -> bytes:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (160, 160), (30, 30, 30), -1)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _prepare_located_project(client: TestClient) -> str:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={"corners": "[[30,25],[470,35],[475,395],[20,385]]"},
    )
    assert resp.status_code == 200
    return project_id


def test_locate_piece_returns_candidates(client: TestClient) -> None:
    project_id = _prepare_located_project(client)
    resp = client.post(
        f"/api/projects/{project_id}/locate",
        files={"piece_photo": ("piece.jpg", _piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["candidates"]) == 3
    for candidate in body["candidates"]:
        assert {"row", "col", "score", "confidence", "rotation"} <= set(candidate)
    assert body["latency_ms"] >= 0


def test_locate_requires_board_state(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    resp = client.post(
        f"/api/projects/{project_id}/locate",
        files={"piece_photo": ("piece.jpg", _piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 409


def test_locate_unknown_project_returns_404(client: TestClient) -> None:
    resp = client.post(
        "/api/projects/does-not-exist/locate",
        files={"piece_photo": ("piece.jpg", _piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 404
