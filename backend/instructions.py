"""
Instruction enum
=================
The single source of truth for every instruction the calibration engine
is allowed to send to the tablet. Keeping this as a str-Enum means:

  - `CalibrationState.update()` can validate incoming values and fail
    fast (ValueError) if the engine ever sends something the tablet UI
    doesn't know how to render.
  - The value serializes straight to JSON (it *is* a string), so nothing
    special is needed on the wire format.
"""

from enum import Enum


class Instruction(str, Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    MOVE_UP = "MOVE_UP"
    MOVE_DOWN = "MOVE_DOWN"
    MOVE_TOP_LEFT = "MOVE_TOP_LEFT"
    MOVE_TOP_RIGHT = "MOVE_TOP_RIGHT"
    MOVE_BOTTOM_LEFT = "MOVE_BOTTOM_LEFT"
    MOVE_BOTTOM_RIGHT = "MOVE_BOTTOM_RIGHT"
    MOVE_CLOSER = "MOVE_CLOSER"
    MOVE_FARTHER = "MOVE_FARTHER"
    TILT_LEFT = "TILT_LEFT"
    TILT_RIGHT = "TILT_RIGHT"
    TILT_FORWARD = "TILT_FORWARD"
    TILT_BACK = "TILT_BACK"
    ROTATE_CLOCKWISE = "ROTATE_CLOCKWISE"
    ROTATE_COUNTER_CLOCKWISE = "ROTATE_COUNTER_CLOCKWISE"
    HOLD_POSITION = "HOLD_POSITION"
    CALIBRATION_COMPLETE = "CALIBRATION_COMPLETE"


# Instructions that represent camera *movement* (used by demo mode / bridge
# code to distinguish "keep moving" from the two terminal states).
MOVEMENT_INSTRUCTIONS = [
    i for i in Instruction
    if i not in (Instruction.HOLD_POSITION, Instruction.CALIBRATION_COMPLETE)
]

ALL_INSTRUCTIONS = [i.value for i in Instruction]
