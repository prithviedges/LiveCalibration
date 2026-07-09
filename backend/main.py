"""
FastAPI entrypoint
==================
Run with:

    uvicorn backend.main:app --host 0.0.0.0 --port 8000

Then, on the tablet (same Wi-Fi network as the laptop), open:

    http://<laptop-lan-ip>:8000/

Endpoints
---------
GET  /        tablet interface (Jinja2-rendered, static after load)
GET  /health  simple liveness/connected-client check
WS   /ws      live calibration state stream (server -> tablet only)
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .calibration_state import CalibrationState
from .demo import run_demo

BASE_DIR = Path(__file__).resolve().parent.parent
DEMO_MODE = os.getenv("CALIB_DEMO_MODE", "0") != "0"

# Single shared instance. Import this from integration_example.py (or your
# own bridge script) to feed it real calibration data via `.update(...)`.
calibration_state = CalibrationState()

_demo_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _demo_task
    loop = asyncio.get_running_loop()
    calibration_state.bind_loop(loop)

    if DEMO_MODE:
        _demo_task = asyncio.create_task(run_demo(calibration_state))
        print("[calibration-dashboard] demo mode ON — set CALIB_DEMO_MODE=0 "
              "once the real engine is wired in.")
    else:
        print("[calibration-dashboard] demo mode OFF — waiting for the "
              "calibration engine to call CalibrationState.update(...).")

    yield

    if _demo_task is not None:
        _demo_task.cancel()


app = FastAPI(title="Calibration Assistant Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request, "tablet.html", {"demo_mode": DEMO_MODE}
    )


@app.get("/health")
async def health():
    return {"status": "ok", "demo_mode": DEMO_MODE, "clients": calibration_state.client_count}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    """Streams calibration state to one tablet. Multiple tablets can
    connect simultaneously — each gets its own registration."""
    await calibration_state.register(websocket)
    try:
        while True:
            # Tablets don't need to send anything; this just keeps the
            # connection open and lets us detect disconnects promptly.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await calibration_state.unregister(websocket)
