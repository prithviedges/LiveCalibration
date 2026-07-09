"""
Demo mode
=========
Generates a plausible, gradually-improving calibration session once per
second so the tablet UI can be built and tested without the real OpenCV
engine running. Cycles: coverage climbs -> HOLD_POSITION pauses ->
CALIBRATION_COMPLETE -> resets and runs again.

Turn this off by setting the environment variable CALIB_DEMO_MODE=0 once
the real engine is wired in via CalibrationState.update() (see
integration_example.py at the repo root).
"""

from __future__ import annotations

import asyncio
import random

from .calibration_state import CalibrationState
from .instructions import Instruction, MOVEMENT_INSTRUCTIONS

REASONS = {
    Instruction.MOVE_LEFT: "Left edge under-covered",
    Instruction.MOVE_RIGHT: "Right edge under-covered",
    Instruction.MOVE_UP: "Top rows need more frames",
    Instruction.MOVE_DOWN: "Bottom rows need more frames",
    Instruction.MOVE_TOP_LEFT: "Need spatial coverage",
    Instruction.MOVE_TOP_RIGHT: "Need spatial coverage",
    Instruction.MOVE_BOTTOM_LEFT: "Need spatial coverage",
    Instruction.MOVE_BOTTOM_RIGHT: "Need spatial coverage",
    Instruction.MOVE_CLOSER: "Radial range too narrow",
    Instruction.MOVE_FARTHER: "Radial range too narrow",
    Instruction.TILT_LEFT: "Pose diversity too low",
    Instruction.TILT_RIGHT: "Pose diversity too low",
    Instruction.TILT_FORWARD: "Pose diversity too low",
    Instruction.TILT_BACK: "Pose diversity too low",
    Instruction.ROTATE_CLOCKWISE: "Need more board rotation",
    Instruction.ROTATE_COUNTER_CLOCKWISE: "Need more board rotation",
}


def _grade_for(composite: float) -> str:
    if composite >= 0.85:
        return "A"
    if composite >= 0.70:
        return "B"
    if composite >= 0.50:
        return "C"
    if composite >= 0.30:
        return "D"
    return "F"


def _step(value: float, lo: float = -0.03, hi: float = 0.08) -> float:
    return min(1.0, max(0.0, value + random.uniform(lo, hi)))


async def run_demo(state: CalibrationState) -> None:
    """Runs forever (until the task is cancelled at shutdown)."""
    spatial = pose = radial = 0.0
    tick = 0

    while True:
        tick += 1
        spatial = _step(spatial)
        pose = _step(pose)
        radial = _step(radial, hi=0.06)

        composite = 0.40 * spatial + 0.40 * pose + 0.20 * radial
        progress = composite * 100
        grade = _grade_for(composite)

        if progress >= 97:
            instruction = Instruction.CALIBRATION_COMPLETE
            reason = "All coverage targets met"
        elif tick % 6 == 0:
            instruction = Instruction.HOLD_POSITION
            reason = "Capturing frames — stay still"
        else:
            instruction = random.choice(MOVEMENT_INSTRUCTIONS)
            reason = REASONS.get(instruction, "Improving coverage")

        state.update(
            grade=grade,
            spatial=spatial,
            pose=pose,
            radial=radial,
            progress=progress,
            instruction=instruction.value,
            reason=reason,
        )

        if instruction is Instruction.CALIBRATION_COMPLETE:
            await asyncio.sleep(5)
            spatial = pose = radial = 0.0  # loop the demo again
            tick = 0
            continue

        await asyncio.sleep(1)
