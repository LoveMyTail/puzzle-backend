"""FastAPI entry point for the puzzle-assistant backend.

Run locally with:  uvicorn main:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Puzzle Assistant Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
