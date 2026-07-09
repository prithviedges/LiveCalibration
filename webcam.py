"""
Run the dashboard against your real webcam
============================================
This starts the FastAPI dashboard (demo mode OFF) in a background thread,
then runs intrinsic_calc.py's actual capture loop in the main thread
(OpenCV's imshow/waitKey need the main thread on some platforms, notably
macOS — so the engine goes in main(), the web server goes in a thread).

No edits to intrinsic_calc.py are required. This monkey-patches
`CalibState.update` at import time so every time the engine updates its
own scores, the new snapshot is also forwarded to the dashboard via
`publish_engine_snapshot()` (see integration_example.py for the mapping
logic — weakest score picks the instruction, radial/pose/spatial priority).

Usage
-----
    pip install -r requirements.txt
    pip install opencv-python numpy          # engine dependencies
    python run_with_webcam.py --source 0     # 0 = default built-in webcam

Then:
  - Control room laptop: http://localhost:8000/
  - Field tablet (same Wi-Fi): http://<this-machine-LAN-IP>:8000/

In the OpenCV preview window: SPACE to start/stop recording checkerboard
frames, ESC to stop and run final calibration, Q to quit without
calibrating. The tablet updates live as scores change.

If you don't have the pitch-segmentation YOLO weights, pass
`--no-pitch-mask` to skip that step (it only affects which grid cells
count toward spatial coverage, not the webcam capture itself).
"""

from __future__ import annotations

import threading
import time

import uvicorn

import intrinsic_calc  # the engine script, copied alongside this file
from backend.main import app, calibration_state
from integration_example import publish_engine_snapshot

# ── monkey-patch: forward every engine score update to the dashboard ──
_original_update = intrinsic_calc.CalibState.update


def _patched_update(self, *args, **kwargs):
    _original_update(self, *args, **kwargs)
    try:
        publish_engine_snapshot(calibration_state, self.snapshot())
    except Exception as exc:  # never let a dashboard hiccup kill the capture loop
        print(f"[dashboard-bridge] failed to publish snapshot: {exc}")


intrinsic_calc.CalibState.update = _patched_update


def _run_dashboard_server() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    uvicorn.Server(config).run()


if __name__ == "__main__":
    server_thread = threading.Thread(target=_run_dashboard_server, daemon=True)
    server_thread.start()
    time.sleep(1.0)  # let the server bind before printing the URL

    print("=" * 70)
    print("Dashboard running:")
    print("  Control room (this laptop): http://localhost:8000/")
    print("  Field tablet (same Wi-Fi):  http://<this-machine-LAN-IP>:8000/")
    print("=" * 70)

    intrinsic_calc.main()  # opens the webcam preview window; blocks until ESC/Q


    # python run_with_webcam.py --source 0