"""
CalibrationState
=================
Holds the latest snapshot published by the calibration engine and fans it
out to every connected tablet over WebSockets.

Why this shape
--------------
The calibration engine (see intrinsic_calc.py) runs its capture/processing
loop on a plain background thread — it is NOT async. FastAPI's WebSocket
connections live on the asyncio event loop. `update()` is the seam between
those two worlds:

  engine thread                         event loop
  --------------                        ----------
  state.update(...)  ──schedules──►     broadcast() sends JSON to every
  (sync, thread-safe)                    connected tablet WebSocket

`update()` can be called from ANY thread. It never blocks on network I/O
itself — it just stores the new snapshot and hands the actual send-to-all
work to the event loop via `asyncio.run_coroutine_threadsafe`.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, Set

from fastapi import WebSocket

from .instructions import Instruction


def _default_state() -> Dict[str, Any]:
    return {
        "grade": "F",
        "spatial": 0.0,
        "pose": 0.0,
        "radial": 0.0,
        "progress": 0,
        "instruction": Instruction.HOLD_POSITION.value,
        "reason": "Waiting for calibration engine…",
        "ts": time.time(),
    }


class CalibrationState:
    """Single shared instance for the whole app (see main.py)."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()
        self._state: Dict[str, Any] = _default_state()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ── wiring ──────────────────────────────────────────────────────
    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Call once at startup (see main.py's lifespan) so background
        threads know which event loop to schedule broadcasts on."""
        self._loop = loop

    # ── client lifecycle (event loop only) ─────────────────────────
    async def register(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._clients_lock:
            self._clients.add(ws)
        # Bring the newly connected tablet up to date immediately.
        await ws.send_text(json.dumps(self._state))

    async def unregister(self, ws: WebSocket) -> None:
        async with self._clients_lock:
            self._clients.discard(ws)

    @property
    def client_count(self) -> int:
        return len(self._clients)

    @property
    def state(self) -> Dict[str, Any]:
        return dict(self._state)

    # ── publishing (callable from any thread) ──────────────────────
    def update(
        self,
        *,
        grade: str,
        spatial: float,
        pose: float,
        radial: float,
        progress: float,
        instruction: str,
        reason: str = "",
    ) -> None:
        """
        Publish a new calibration snapshot. Safe to call from the
        calibration engine's own thread — this is the only method
        external code needs to call.

        Raises ValueError if `instruction` isn't one of the allowed
        Instruction enum values, so a typo in the engine fails loudly
        instead of silently confusing the field operator.
        """
        payload = {
            "grade": str(grade),
            "spatial": round(float(spatial), 4),
            "pose": round(float(pose), 4),
            "radial": round(float(radial), 4),
            "progress": max(0, min(100, round(float(progress)))),
            "instruction": Instruction(instruction).value,
            "reason": reason,
            "ts": time.time(),
        }
        self._state = payload
        self._schedule_broadcast()

    def _schedule_broadcast(self) -> None:
        if self._loop is None:
            # No event loop bound yet (e.g. called before startup) —
            # the next connecting client still gets this via register().
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(), self._loop)
        except RuntimeError:
            # Loop is shutting down; drop the update rather than crash
            # the calling (engine) thread.
            pass

    async def broadcast(self) -> None:
        """Send the current state to every connected tablet. Runs on the
        event loop — either scheduled from update() or called directly
        by async demo code."""
        payload = json.dumps(self._state)
        async with self._clients_lock:
            dead = []
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)
