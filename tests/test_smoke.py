"""Smoke tests: verify the skeleton imports and the health endpoint works."""

import cv2
from fastapi.testclient import TestClient

import main


def test_health_endpoint() -> None:
    client = TestClient(main.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_cv2_importable() -> None:
    assert hasattr(cv2, "__version__")
