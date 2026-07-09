# Calibration Assistant Dashboard

A real-time control-room-to-tablet dashboard for camera intrinsic
calibration. The laptop runs a FastAPI server that streams calibration
state over a WebSocket; a tablet on the same Wi-Fi network shows the
field operator one huge, color-coded instruction at a time. Everything
runs on the local network — no internet connection is required.

## Folder structure

```
calibration-dashboard/
├── backend/
│   ├── __init__.py
│   ├── main.py                 FastAPI app: GET /, GET /health, WS /ws
│   ├── calibration_state.py    CalibrationState: update() + broadcast()
│   ├── demo.py                 Demo mode (random data, cycles every 1s)
│   └── instructions.py         Instruction enum (the only allowed values)
├── templates/
│   └── tablet.html             Fullscreen tablet page
├── static/
│   ├── css/style.css           High-contrast, sunlight-readable styling
│   └── js/app.js                WebSocket client + enum→label/icon logic
├── integration_example.py      How to feed real data from intrinsic_calc.py
├── requirements.txt
└── README.md
```

## Setup

```bash
cd calibration-dashboard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run (demo mode)

Demo mode is on by default — it generates plausible random calibration
data every second so you can build/test the tablet UI without the real
OpenCV engine running.

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- **Control room laptop:** open `http://localhost:8000/`
- **Field tablet:** find the laptop's LAN IP (`ipconfig` / `ifconfig`,
  e.g. `192.168.1.23`) and open `http://192.168.1.23:8000/` in the
  tablet's browser. Add it to the home screen for a fullscreen app-like
  experience.

Both laptop and tablet must be on the **same Wi-Fi network**. No
internet access is used or required by the server, the page, or any of
its assets — fonts and icons are all inline/system, nothing is fetched
from a CDN.

## Run with the real calibration engine

1. Turn demo mode off:

   ```bash
   CALIB_DEMO_MODE=0 uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

2. Wire `intrinsic_calc.py`'s `CalibState` into the dashboard's
   `CalibrationState`. See `integration_example.py` for the full
   explanation and a ready-to-adapt `publish_engine_snapshot()` function.
   The short version — call this once per engine update, from any thread:

   ```python
   from backend.main import calibration_state
   from backend.instructions import Instruction

   calibration_state.update(
       grade="B",
       spatial=0.74,
       pose=0.61,
       radial=0.39,
       progress=81,               # 0–100
       instruction=Instruction.MOVE_TOP_LEFT.value,
       reason="Need spatial coverage",
   )
   ```

   `CalibrationState.update()` is thread-safe and non-blocking — it's
   safe to call directly from the OpenCV capture/collector thread. It
   validates `instruction` against the allowed enum and raises
   `ValueError` immediately if it doesn't recognize the value, so a typo
   in the engine fails loudly instead of confusing the field operator.

## Supported instructions

Exactly these 18 values are accepted by `CalibrationState.update()`
(see `backend/instructions.py`); the tablet UI translates each into a
readable label and pictogram automatically:

```
MOVE_LEFT, MOVE_RIGHT, MOVE_UP, MOVE_DOWN,
MOVE_TOP_LEFT, MOVE_TOP_RIGHT, MOVE_BOTTOM_LEFT, MOVE_BOTTOM_RIGHT,
MOVE_CLOSER, MOVE_FARTHER,
TILT_LEFT, TILT_RIGHT, TILT_FORWARD, TILT_BACK,
ROTATE_CLOCKWISE, ROTATE_COUNTER_CLOCKWISE,
HOLD_POSITION, CALIBRATION_COMPLETE
```

## Color logic

The tablet derives its background color client-side from the
instruction + progress (see `statusFor()` in `static/js/app.js`):

| Status  | When                                          | Color  |
|---------|------------------------------------------------|--------|
| move    | any `MOVE_*` / `TILT_*` / `ROTATE_*`, progress < 90 | Red    |
| almost  | any movement instruction, progress ≥ 90         | Yellow |
| hold    | `HOLD_POSITION`                                 | Green  |
| complete| `CALIBRATION_COMPLETE`                          | Blue   |

## Multiple tablets

`CalibrationState` keeps a set of connected WebSocket clients and
broadcasts to all of them, so any number of tablets can connect at once
and all stay in sync (e.g. one on the field operator's cart, one spare).

## Notes on the WebSocket contract

- Server → client only. The tablet doesn't need to send anything
  meaningful; the client keeps the socket open so disconnects are
  detected immediately and reconnects automatically (1.5s backoff) if
  Wi-Fi drops.
- Every message is the full current state as JSON — the client always
  renders from a complete snapshot, so a missed message never leaves the
  UI in an inconsistent state.
