"""
Integration example: wiring intrinsic_calc.py into the dashboard
==================================================================
This is the seam between the existing OpenCV calibration engine
(`CalibState` in intrinsic_calc.py) and this dashboard's `CalibrationState`
(backend/calibration_state.py). It is intentionally a standalone script
you can adapt, not something imported by the FastAPI app — the engine and
the web server are two independent processes/threads that only need to
agree on this one function call.

How to wire it in
------------------
1. Run the dashboard server in demo mode OFF:

       CALIB_DEMO_MODE=0 uvicorn backend.main:app --host 0.0.0.0 --port 8000

2. In intrinsic_calc.py's collector thread (see `collector_thread_fn`),
   right after each `state.update(...)` call on the engine's own
   `CalibState`, forward the new snapshot to the dashboard:

       from backend.main import calibration_state
       from integration_example import publish_engine_snapshot

       state.update(...)                       # existing engine call
       publish_engine_snapshot(calibration_state, state.snapshot())

   `CalibrationState.update()` is thread-safe and non-blocking, so this
   is safe to call directly from the OpenCV capture/collector thread —
   no extra queues or locks needed on your side.

3. Run both in the same Python process (simplest: import the FastAPI app
   and start uvicorn programmatically alongside the capture loop in a
   background thread), OR run them as two processes and swap this
   function for a tiny HTTP POST / local socket call into the dashboard
   process. The mapping logic below stays the same either way.

Deriving an instruction from the engine's snapshot
----------------------------------------------------
intrinsic_calc.py's `CalibState` tracks scores and a coverage grid
(`occupancy`, `active_mask`) but not a single directional instruction —
that's a UI concept the engine doesn't need to know about. This function
derives one from whichever score is weakest:

  - radial score weakest  -> alternate MOVE_CLOSER / MOVE_FARTHER
    (radial score reflects distance/scale diversity)
  - pose score weakest    -> cycle TILT_*/ROTATE_* for orientation diversity
  - spatial score weakest -> look at the occupancy grid and point toward
    the least-covered quadrant of the *active* cells
  - grade A / composite high -> CALIBRATION_COMPLETE
  - otherwise -> HOLD_POSITION while frames are being captured

Treat this as a starting point: swap in your own priority logic freely,
the only contract that matters is calling `calibration_state.update(...)`
with a valid Instruction value.
"""

from __future__ import annotations

import itertools
from typing import Any, Dict

from backend.calibration_state import CalibrationState
from backend.instructions import Instruction

# Composite weights — keep these in sync with intrinsic_calc.py's
# W_SPATIAL / W_POSE / W_DEPTH / W_RADIAL constants.
W_SPATIAL = 0.30
W_POSE = 0.30
W_DEPTH = 0.25
W_RADIAL = 0.15

_radial_cycle = itertools.cycle([Instruction.MOVE_CLOSER, Instruction.MOVE_FARTHER])
_pose_cycle = itertools.cycle(
    [
        Instruction.TILT_LEFT,
        Instruction.TILT_RIGHT,
        Instruction.TILT_FORWARD,
        Instruction.TILT_BACK,
        Instruction.ROTATE_CLOCKWISE,
        Instruction.ROTATE_COUNTER_CLOCKWISE,
    ]
)

_QUADRANT_INSTRUCTION = {
    ("top", "left"): Instruction.MOVE_TOP_LEFT,
    ("top", "right"): Instruction.MOVE_TOP_RIGHT,
    ("bottom", "left"): Instruction.MOVE_BOTTOM_LEFT,
    ("bottom", "right"): Instruction.MOVE_BOTTOM_RIGHT,
    ("top", "center"): Instruction.MOVE_UP,
    ("bottom", "center"): Instruction.MOVE_DOWN,
    ("center", "left"): Instruction.MOVE_LEFT,
    ("center", "right"): Instruction.MOVE_RIGHT,
}


def _least_covered_direction(occupancy, active_mask) -> Instruction:
    """Split the active occupancy grid into thirds and point toward the
    quadrant with the lowest coverage ratio."""
    rows, cols = occupancy.shape
    row_third = max(1, rows // 3)
    col_third = max(1, cols // 3)

    def band(size, third, index):
        if index < third:
            return "top" if size == rows else "left"
        if index >= size - third:
            return "bottom" if size == rows else "right"
        return "center"

    scores: Dict[tuple, list] = {}
    for r in range(rows):
        for c in range(cols):
            if not active_mask[r, c]:
                continue
            key = (band(rows, row_third, r), band(cols, col_third, c))
            covered, total = scores.setdefault(key, [0, 0])
            scores[key][1] += 1
            if occupancy[r, c]:
                scores[key][0] += 1

    # Prefer genuine quadrants (skip "center","center" — nothing to say there)
    scores.pop(("center", "center"), None)

    worst_key, worst_ratio = None, 2.0
    for key, (covered, total) in scores.items():
        if total == 0 or key not in _QUADRANT_INSTRUCTION:
            continue
        ratio = covered / total
        if ratio < worst_ratio:
            worst_ratio, worst_key = ratio, key

    if worst_key is None:
        return Instruction.HOLD_POSITION
    return _QUADRANT_INSTRUCTION[worst_key]


def derive_instruction(snapshot: Dict[str, Any]) -> tuple[Instruction, str]:
    spatial = snapshot["spatial"]
    pose = snapshot["pose"]
    depth = snapshot.get("depth", 0.0)
    radial = snapshot["radial"]
    composite = W_SPATIAL * spatial + W_POSE * pose + W_DEPTH * depth + W_RADIAL * radial

    if snapshot["grade"] == "A" or composite >= 0.85:
        return Instruction.CALIBRATION_COMPLETE, "All coverage targets met"

    weakest = min(
        ("depth", depth),
        ("spatial", spatial),
        ("pose", pose),
        ("radial", radial),
        key=lambda t: t[1],
    )[0]

    if weakest == "depth" and depth < 0.6:
        d_cnt = snapshot.get("depth_counts", {})
        if d_cnt.get("far", 0) < 6:
            return Instruction.MOVE_FARTHER, "Need more distance (FAR frames)"
        elif d_cnt.get("near", 0) < 6:
            return Instruction.MOVE_CLOSER, "Bring board closer (NEAR frames)"
        else:
            return Instruction.MOVE_FARTHER, "Vary distance from camera"
    if weakest == "radial" and radial < 0.5:
        return next(_radial_cycle), "Radial range too narrow"
    if weakest == "pose" and pose < 0.5:
        return next(_pose_cycle), "Need more pose diversity"
    if weakest == "spatial" and spatial < 0.85:
        instruction = _least_covered_direction(snapshot["occupancy"], snapshot["active_mask"])
        return instruction, snapshot.get("hint") or "Need spatial coverage"

    return Instruction.HOLD_POSITION, "Capturing frames — stay still"


def publish_engine_snapshot(calibration_state: CalibrationState, snapshot: Dict[str, Any]) -> None:
    """Call this right after `CalibState.update(...)` in intrinsic_calc.py's
    collector thread, passing `state.snapshot()` as `snapshot`."""
    if snapshot.get("frames", 0) == 0:
        return  # nothing recorded yet — leave the dashboard's default state

    spatial = snapshot["spatial"]
    pose = snapshot["pose"]
    depth = snapshot.get("depth", 0.0)
    radial = snapshot["radial"]
    composite = W_SPATIAL * spatial + W_POSE * pose + W_DEPTH * depth + W_RADIAL * radial
    instruction, reason = derive_instruction(snapshot)

    calibration_state.update(
        grade=snapshot["grade"],
        spatial=spatial,
        pose=pose,
        depth=depth,
        radial=radial,
        progress=composite * 100,
        instruction=instruction.value,
        reason=reason,
        occupancy=snapshot["occupancy"].astype(bool).tolist(),
        active_mask=snapshot["active_mask"].astype(bool).tolist(),
        total_active_cells=snapshot["total_active_cells"],
    )
