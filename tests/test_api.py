"""API tests for project creation and box-cover calibration."""

import math
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
    cell = 240  # px per cell in the rendered layout: realistic source resolution
    grid_h, grid_w = rows * cell, cols * cell
    base = np.full((grid_h, grid_w, 3), 255, dtype=np.uint8)
    for row in range(1, 5):
        for col in range(1, 6):
            x0, y0 = col * cell, row * cell
            cv2.rectangle(
                base, (x0, y0), (x0 + cell - 1, y0 + cell - 1), (30, 130, 210), -1
            )
    src = np.array(
        [[0, 0], [grid_w - 1, 0], [grid_w - 1, grid_h - 1], [0, grid_h - 1]],
        dtype=np.float32,
    )
    dst = np.array([[30, 25], [3060, 40], [3070, 2580], [20, 2565]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(
        base,
        matrix,
        (3100, 2600),
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

    corners = "[[30,25],[3060,40],[3070,2580],[20,2565]]"
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


def test_update_board_rejects_low_resolution_photo(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    # The marked region spans only ~450px -> ~12px per cell on the 27x37 grid.
    corners = "[[30,25],[470,35],[475,395],[20,385]]"
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={"corners": corners},
    )
    assert resp.status_code == 422
    assert "resolution" in resp.json()["detail"].lower()


def test_update_board_accepts_moderate_resolution_photo(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    # ~22 px per cell on the 27x37 grid: above the default 16px floor.
    corners = "[[30,25],[820,35],[825,685],[20,675]]"
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={"corners": corners},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "gaps" in body
    assert "alignment" in body
    assert "boundary_ratio" in body["alignment"]


def test_update_board_accepts_grid_override(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", _board_photo_bytes(), "image/jpeg")},
        data={
            "corners": "[[30,25],[3060,40],[3070,2580],[20,2565]]",
            "rows": "4",
            "cols": "6",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert (body["rows"], body["cols"]) == (4, 6)
    detail = client.get(f"/api/projects/{project_id}").json()
    assert (detail["rows"], detail["cols"]) == (4, 6)


def test_update_board_full_frame_corners_runs_auto_detection(
    client: TestClient,
) -> None:
    resp = client.post(
        "/api/projects",
        files={"box_photo": ("box.jpg", _box_photo_bytes(), "image/jpeg")},
        data={"name": "auto-quad", "piece_count": "24", "width_cm": "30", "height_cm": "20"},
    )
    assert resp.status_code == 200
    project_id = resp.json()["project_id"]
    resp = client.put(
        f"/api/projects/{project_id}/calibration",
        json={"points": [[0, 0], [400, 0], [400, 300], [0, 300]]},
    )
    assert resp.status_code == 200
    image = np.full((600, 900, 3), 230, dtype=np.uint8)
    rng = np.random.default_rng(3)
    for r in range(4):
        for c in range(6):
            color = tuple(int(v) for v in rng.integers(40, 180, size=3))
            cv2.rectangle(
                image,
                (180 + c * 60, 120 + r * 60),
                (240 + c * 60, 180 + r * 60),
                color,
                -1,
            )
            cv2.rectangle(
                image,
                (180 + c * 60, 120 + r * 60),
                (240 + c * 60, 180 + r * 60),
                (0, 0, 0),
                2,
            )
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    resp = client.put(
        f"/api/projects/{project_id}/board",
        files={"board_photo": ("board.jpg", buf.tobytes(), "image/jpeg")},
        data={"corners": "[[0,0],[900,0],[900,600],[0,600]]"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "alignment" in body
    # The stored board corners should reflect the detected (non-full-frame) quad.
    detail = client.get(f"/api/projects/{project_id}").json()
    stored = detail["board"]["corners"]
    assert stored != [[0.0, 0.0], [900.0, 0.0], [900.0, 600.0], [0.0, 600.0]]


def _piece_photo_bytes() -> bytes:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (40, 40), (160, 160), (30, 30, 30), -1)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _tiny_piece_photo_bytes() -> bytes:
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (26, 26), (38, 38), (30, 30, 30), -1)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return buf.tobytes()


def _star_piece_photo_bytes(size: int = 200) -> bytes:
    """A star-shaped blob: not piece-like, must be rejected with a 422."""
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    center = size / 2
    outer, inner = size * 0.42, size * 0.10
    points = []
    for i in range(10):
        radius = outer if i % 2 == 0 else inner
        angle = math.pi * i / 5 - math.pi / 2
        points.append(
            (center + radius * math.cos(angle), center + radius * math.sin(angle))
        )
    cv2.fillPoly(image, [np.asarray(points, dtype=np.int32)], (30, 30, 30))
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
        data={"corners": "[[30,25],[3060,40],[3070,2580],[20,2565]]"},
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
    assert 3 <= len(body["candidates"]) <= 6
    for candidate in body["candidates"]:
        assert {"row", "col", "score", "confidence", "rotation"} <= set(candidate)
        assert "region" in candidate
        region = candidate["region"]
        if region is not None:
            assert {"x", "y", "w", "h"} <= set(region)
            assert 0.0 <= region["x"] <= 1.0
            assert 0.0 <= region["y"] <= 1.0
            assert 0.0 < region["w"] <= 1.0
            assert 0.0 < region["h"] <= 1.0
    assert body["latency_ms"] >= 0
    # The visual answer: a board image with the candidate gaps marked, plus the
    # overall board preview URL for the app to display next to the piece photo.
    assert body["preview_url"] == f"/api/projects/{project_id}/locate/preview"
    preview = client.get(body["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/")
    board_preview = client.get(body["board_preview_url"])
    assert board_preview.status_code == 200
    assert board_preview.headers["content-type"].startswith("image/")


def test_project_operation_journal_records_full_flow(client: TestClient) -> None:
    project_id = _prepare_located_project(client)
    resp = client.post(
        f"/api/projects/{project_id}/locate",
        files={"piece_photo": ("piece.jpg", _piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    resp = client.put(
        f"/api/projects/{project_id}/placed",
        json={"row": 4, "col": 7},
    )
    assert resp.status_code == 200

    resp = client.get(f"/api/projects/{project_id}/operations")
    assert resp.status_code == 200
    operations = resp.json()["operations"]
    assert [op["op"] for op in operations] == [
        "create",
        "calibrate",
        "board",
        "locate",
        "placed",
    ]
    assert operations[0]["files"]["box_photo.jpg"].endswith(
        "box_photo.jpg"
    )
    assert operations[3]["candidates"]
    assert operations[4]["row"] == 4
    assert operations[4]["col"] == 7


def test_board_preview_missing_before_upload(client: TestClient) -> None:
    project_id = _create_project(client)["project_id"]
    resp = client.get(f"/api/projects/{project_id}/board/preview")
    assert resp.status_code == 404
    resp = client.get(f"/api/projects/{project_id}/locate/preview")
    assert resp.status_code == 404


def test_locate_piece_rejects_tiny_piece(client: TestClient) -> None:
    project_id = _prepare_located_project(client)
    resp = client.post(
        f"/api/projects/{project_id}/locate",
        files={"piece_photo": ("piece.jpg", _tiny_piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "too small" in resp.json()["detail"].lower()


def test_locate_piece_rejects_non_piece_shape(client: TestClient) -> None:
    project_id = _prepare_located_project(client)
    resp = client.post(
        f"/api/projects/{project_id}/locate",
        files={"piece_photo": ("piece.jpg", _star_piece_photo_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 422
    assert "清晰的拼图块轮廓" in resp.json()["detail"]


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
