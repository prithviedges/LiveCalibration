"""
Run the dashboard against your real webcam or RTSP camera stream
===============================================================
This starts the FastAPI dashboard in a background thread, then runs
intrinsic_calc.py's capture loop in the main thread.

Features:
  - Low-latency, freeze-free RTSP stream capture (TCP transport + threaded grabber).
  - Clean video session recording into the 'Videos/' directory when recording is ON.
  - Automatic persistence of the latest camera intrinsics to 'calib_params.npz'
    and timestamped calibration records to 'calib_history.json' and 'calib_history.npz'.

Usage
-----
    # Default built-in webcam:
    python webcam.py --source 0

    # Custom RTSP URL:
    python webcam.py --rtsp rtsp://admin:pass@192.168.1.100:554/stream1
    # or
    python webcam.py --source rtsp://admin:pass@192.168.1.100:554/stream1

    # Using preset RTSP sources defined in RTSP_SOURCES below:
    python webcam.py --rtsp 1
    python webcam.py --list-rtsp

    # With pitch-mask disabled (if no YOLO weights):
    python webcam.py --rtsp 1 --no-pitch-mask

Then:
  - Control room laptop: http://localhost:8000/
  - Field tablet (same Wi-Fi): http://<this-machine-LAN-IP>:8000/
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict

import uvicorn

# Low-latency TCP transport for RTSP streams across OpenCV FFmpeg
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
)

import intrinsic_calc
import intrinsic_calc_better
from backend.main import app, calibration_state
from integration_example import publish_engine_snapshot

# ── Configurable RTSP Sources / Presets ─────────────────────────────
# Add your camera links here for fast access:
RTSP_SOURCES: Dict[str, str] = {
    "1": "rtsp://192.168.1.100:554/stream1",
    "2": "rtsp://admin:admin@192.168.1.101:554/h264Preview_01_main",
    "pitch_cam": "rtsp://192.168.1.50:554/live",
    "sample": "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4",
}

# ── monkey-patch: forward every engine score update to the dashboard ──
def _patch_engine(engine_mod):
    orig = engine_mod.CalibState.update
    def _patched_update(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            publish_engine_snapshot(calibration_state, self.snapshot())
        except Exception as exc:
            print(f"[dashboard-bridge] failed to publish snapshot: {exc}")
    engine_mod.CalibState.update = _patched_update

_patch_engine(intrinsic_calc)
_patch_engine(intrinsic_calc_better)


def _run_dashboard_server() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    uvicorn.Server(config).run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Live Calibration dashboard with webcam or RTSP camera stream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video source: webcam integer (e.g. 0), RTSP URL (e.g. rtsp://...), MJPEG URL, or video path",
    )
    parser.add_argument(
        "--rtsp",
        default=None,
        help="RTSP stream URL or preset key from RTSP_SOURCES (e.g. '1', 'pitch_cam', 'sample')",
    )
    parser.add_argument(
        "--list-rtsp",
        action="store_true",
        help="List available preconfigured RTSP presets and exit",
    )
    parser.add_argument(
        "--engine",
        choices=["better", "classic"],
        default="better",
        help="Calibration engine: 'better' (intrinsic_calc_better with Stratified Assembler) or 'classic' (intrinsic_calc). Default: better",
    )
    parser.add_argument(
        "--classic",
        action="store_true",
        help="Shorthand to use classic intrinsic_calc.py",
    )
    parser.add_argument(
        "--better",
        action="store_true",
        help="Shorthand to use intrinsic_calc_better.py (default)",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    known_args, extra_args = parse_args()

    if known_args.list_rtsp:
        print("\nPreconfigured RTSP Sources:")
        print("─" * 60)
        for key, url in RTSP_SOURCES.items():
            print(f"  [{key}] -> {url}")
        print("─" * 60)
        print("Usage: python webcam.py --rtsp <key_or_url>\n")
        sys.exit(0)

    # Select engine
    engine_name = "classic" if known_args.classic else ("better" if known_args.better else known_args.engine)
    selected_engine = intrinsic_calc if engine_name == "classic" else intrinsic_calc_better

    # Determine final video source
    source_to_use = None
    if known_args.rtsp is not None:
        if known_args.rtsp in RTSP_SOURCES:
            source_to_use = RTSP_SOURCES[known_args.rtsp]
            print(f"📌 Using RTSP preset '{known_args.rtsp}': {source_to_use}")
        else:
            source_to_use = known_args.rtsp
    elif known_args.source is not None:
        source_to_use = known_args.source

    # Prepare arguments to forward to engine's main
    engine_args = list(extra_args)
    if source_to_use is not None:
        source_to_use = selected_engine.normalize_rtsp_url(str(source_to_use))
        engine_args.extend(["--source", str(source_to_use)])

    # Start FastAPI dashboard background server
    server_thread = threading.Thread(target=_run_dashboard_server, daemon=True)
    server_thread.start()
    time.sleep(1.0)  # let the server bind before printing the URL

    print("=" * 70)
    print("Dashboard running:")
    print("  Control room (this laptop): http://localhost:8000/")
    print("  Field tablet (same Wi-Fi):  http://<this-machine-LAN-IP>:8000/")
    print(f"  Calibration Engine:         {engine_name.upper()} ({selected_engine.__name__}.py)")
    if source_to_use is not None:
        print(f"  Video Source:               {source_to_use}")
    print("=" * 70)

    # Run the calibration engine with smooth capture and video recording
    selected_engine.main(engine_args)