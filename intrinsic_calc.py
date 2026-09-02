"""
Intrinsic Calibration with Live CV2 HUD  –  Parallel Edition  (v3, active-cell masking)
========================================================================================

New in this version
--------------------
  - On the first SPACE press that starts recording, we grab the current frame
    (assumed to show the empty pitch, no checkerboard on it yet), run a YOLO
    segmentation model over it, take the largest contour from the pitch mask,
    reduce it to a 4-point quadrilateral, and use that quad's (expanded)
    bounding rectangle to decide which HUD grid cells are "active".
  - Only active cells count toward spatial coverage / total_weight / hints.
    Cells outside the pitch region (e.g. crowd, sky, boundary rope) are
    permanently excluded so they can't cap your achievable composite score.

Parallelism model
-----------------
  ┌──────────────┐     frame_q    ┌─────────────────────────────┐
  │ Capture      │ ── frames ───► │ ThreadPoolExecutor (workers) │
  │ thread       │                │  · findChessboardCorners     │
  │ (main loop)  │                │  · cornerSubPix              │
  └──────────────┘                │  · solvePnP (if mtx ready)  │
                                  └──────────────┬──────────────┘
                                                 │ result_q
                                  ┌──────────────▼──────────────┐
                                  │ Collector thread             │
                                  │  · appends obj/img points    │
                                  │  · bootstraps camera matrix  │
                                  │  · updates CalibState        │
                                  └─────────────────────────────┘

Controls
--------
  SPACE  – toggle recording on/off (first press also triggers pitch segmentation)
  ESC    – stop recording, run final calibration, print intrinsics, quit
  Q      – quit immediately without calibrating
  T      – run built-in self-tests (prints to terminal, no capture needed)
"""

from __future__ import annotations
import os
import argparse
import json
import math
import queue
import random
import sys
import threading
import time
import warnings
from datetime import datetime
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple, Union
from backend.calibration_state import CalibrationState
from backend.main import calibration_state
from backend.instructions import Instruction
import cv2 as cv
import numpy as np
from integration_example import publish_engine_snapshot
from integration_example import derive_instruction

warnings.filterwarnings("ignore")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# try:
#     from sklearn.cluster import KMeans
#     from sklearn.preprocessing import StandardScaler
#     HAS_SKLEARN = True
# except ImportError:
#     HAS_SKLEARN = False

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
CHECKERBOARD_W    = 10          # inner corners, columns
CHECKERBOARD_H    = 7           # inner corners, rows
SQUARE_SIZE       = 0.092       # metres per square

GRID_COLS         = 8           # HUD coverage grid columns
GRID_ROWS         = 5           # HUD coverage grid rows
EDGE_WEIGHT       = 2.0         # extra weight for edge cells in spatial score
TARGET_STD_DEG    = 35.0        # desired rotation std-dev for pose score
N_POSE_CLUSTERS   = 6           # KMeans clusters for pose diversity
W_SPATIAL         = 0.40        # composite weight
W_POSE            = 0.40
W_RADIAL          = 0.20
GRADE_A_THRESH    = 0.85        # composite score for auto-stop

NUM_WORKERS       = 10          # parallel detection workers
MAX_QUEUE_DEPTH   = 30          # cap pending futures to bound memory
FRAME_STRIDE      = 3           # process every Nth captured frame
POSE_HISTORY_LIMIT = 300        # rolling window of stored rvecs

BOOTSTRAP_MIN_FRAMES = 25       # frames before first live calibration attempt
ROLLING_FPS_WINDOW   = 50       # frames for rolling fps estimate

# ── Pitch segmentation / active-cell config ──
_script_dir = os.path.dirname(os.path.abspath(__file__))
PITCH_SEG_MODEL_PATH = os.path.join(_script_dir, "weights", "pitch_segmentation.pt")   # path to your YOLO segmentation weights
PITCH_SEG_CONF       = 0.7             # confidence threshold for the seg model
PITCH_SEG_CLASS_ID   = 0             # None = use all predicted masks; set an int to filter to one class
PITCH_MARGIN_RATIO   = 0.15             # expand the quad's bounding box by this fraction of its w/h
PITCH_MIN_ACTIVE_CELLS = 4              # sanity floor; if segmentation gives fewer, fall back to all-active

# ─────────────────────────────────────────────
# HUD LAYOUT CONSTANTS
# ─────────────────────────────────────────────
HUD_RIGHT_W   = 240
HUD_GRID_H    = 120
HUD_TOP_H     = 44
PANEL_PAD     = 10
GRADE_BOX_SZ  = 64
BAR_H         = 14
BAR_MAX_W     = 120

FONT    = cv.FONT_HERSHEY_SIMPLEX
FONT_SM = 0.45
FONT_MD = 0.65
FONT_LG = 1.1
THICK_1 = 1
THICK_2 = 2

# Colours (BGR)
C_GREEN      = (60,  210,  80)
C_RED        = (50,   60, 210)
C_ORANGE     = (30,  150, 230)
C_BLUE       = (210, 100,  40)
C_WHITE      = (240, 240, 240)
C_PANEL      = (20,   20,  20)
C_PANEL_LT   = (40,   40,  40)
C_HINT_BG    = (28,   28,  28)
C_GRADE_A    = (50,  180,  60)
C_GRADE_B    = (170, 130,  40)
C_GRADE_C    = (40,  130, 200)
C_GRADE_D    = (40,   80, 200)
C_GRADE_F    = (50,   50, 180)
C_DROPPED    = (40,  200, 200)   # warn colour for dropped frames
C_INACTIVE   = (55,   55,  55)   # dimmed fill for cells outside the pitch region
C_INACTIVE_BORDER = (80, 80, 80)

# ─────────────────────────────────────────────
# CHECKERBOARD OBJECT POINTS
# ─────────────────────────────────────────────
# OpenCV findChessboardCorners returns corners in row-major order:
#   (col=0,row=0), (col=1,row=0), …, (col=W-1,row=0), (col=0,row=1), …
# So objp must have the same ordering: x (column index) varies fastest.
# np.mgrid[0:W, 0:H].T.reshape(-1,2) produces exactly that.
_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CHECKERBOARD_W * CHECKERBOARD_H, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_W, 0:CHECKERBOARD_H].T.reshape(-1, 2)
objp *= SQUARE_SIZE


# ─────────────────────────────────────────────
# PITCH SEGMENTATION → ACTIVE-CELL MASK
# ─────────────────────────────────────────────

_pitch_model = None   # lazily-loaded singleton

    
def get_pitch_model(model_path: str = PITCH_SEG_MODEL_PATH):
    """
    Lazily load (and cache) the YOLO segmentation model.

    Returns None if ultralytics isn't installed or the weights can't be loaded,
    in which case callers should fall back to "all cells active".
    """
    global _pitch_model
    if not HAS_YOLO:
        print("[pitch-seg] ultralytics not installed — skipping pitch segmentation, "
              "all grid cells stay active.")
        return None
    if _pitch_model is None:
        if not os.path.exists(model_path):
            print(f"[pitch-seg] model weights not found at '{model_path}' — "
                  "all grid cells stay active.")
            return None
        try:
            _pitch_model = YOLO(model_path)
        except Exception as e:
            print(f"[pitch-seg] failed to load model: {e}")
            return None
    return _pitch_model


def segment_pitch_quadrilateral(
    frame: np.ndarray,
    model=None,
    conf: float = PITCH_SEG_CONF,
    class_id: Optional[int] = PITCH_SEG_CLASS_ID,
) -> Optional[np.ndarray]:
    """
    Run pitch segmentation on a single frame and reduce the result to a
    4-point quadrilateral.

    Steps:
      1. Run the YOLO seg model, collect all predicted masks (optionally
         filtered to `class_id`).
      2. OR them together into one binary mask and take the largest contour.
      3. approxPolyDP the contour down to 4 points. If that doesn't cleanly
         yield a quad, fall back to minAreaRect's 4 corners.

    Args:
        frame: BGR image, e.g. the first frame captured after recording starts.
        model: a loaded ultralytics YOLO seg model (see get_pitch_model()).
               If None, this function loads/caches one itself.
        conf: confidence threshold passed to the model.
        class_id: if set, only masks predicted as this class are used.

    Returns:
        (4, 2) float32 array of quadrilateral corners in image coordinates,
        or None if no pitch mask could be found.
    """
    if model is None:
        model = get_pitch_model()
    if model is None:
        return None

    h, w = frame.shape[:2]
    try:
        results = model.predict(frame, conf=conf, verbose=False)
    except Exception as e:
        print(f"[pitch-seg] inference failed: {e}")
        return None

    if not results or results[0].masks is None:
        print("[pitch-seg] model returned no masks.")
        return None

    r0 = results[0]
    mask_data = r0.masks.data.cpu().numpy()          # (N, mh, mw)
    cls_data = (
        r0.boxes.cls.cpu().numpy().astype(int)
        if r0.boxes is not None and r0.boxes.cls is not None
        else np.zeros(len(mask_data), dtype=int)
    )

    combined = np.zeros((h, w), dtype=np.uint8)
    for i, m in enumerate(mask_data):
        if class_id is not None and i < len(cls_data) and cls_data[i] != class_id:
            continue
        m_resized = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
        combined |= (m_resized > 0.5).astype(np.uint8)

    if combined.sum() == 0:
        print("[pitch-seg] no mask pixels matched the requested class.")
        return None

    contours, _ = cv.findContours(
        combined * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv.contourArea)
    peri = cv.arcLength(largest, True)
    approx = cv.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        quad = approx.reshape(-1, 2).astype(np.float32)
    else:
        # Fallback: minimum-area rotated rectangle around the contour.
        rect = cv.minAreaRect(largest)
        quad = cv.boxPoints(rect).astype(np.float32)

    return quad


def compute_active_grid_from_quad(
    quad_pts: np.ndarray,
    img_w: int,
    img_h: int,
    margin_ratio: float = PITCH_MARGIN_RATIO,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """
    Turn a pitch quadrilateral into a boolean active-cell grid.

    Takes the quad's axis-aligned bounding box, expands it by `margin_ratio`
    of its own width/height on each side (so the "active rectangle" is a bit
    larger than the quad itself — you'll usually wave the checkerboard a
    little outside the exact pitch outline), clips to image bounds, then
    marks every GRID_ROWS x GRID_COLS cell that overlaps that rectangle.

    Args:
        quad_pts: (4, 2) array of quadrilateral corners.
        img_w, img_h: frame dimensions.
        margin_ratio: expansion fraction applied per side.

    Returns:
        active_mask – (GRID_ROWS, GRID_COLS) bool array.
        rect        – (x_min, y_min, x_max, y_max) the expanded rectangle used.
    """
    x_min, y_min = quad_pts.min(axis=0)
    x_max, y_max = quad_pts.max(axis=0)

    quad_w = max(x_max - x_min, 1.0)
    quad_h = max(y_max - y_min, 1.0)
    mx = quad_w * margin_ratio
    my = quad_h * margin_ratio

    x_min = max(0.0, x_min - mx)
    y_min = max(0.0, y_min - my)
    x_max = min(float(img_w), x_max + mx)
    y_max = min(float(img_h), y_max + my)

    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    active = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx1, cy1 = c * cell_w, r * cell_h
            cx2, cy2 = cx1 + cell_w, cy1 + cell_h
            # mark active if the cell rectangle overlaps the expanded quad rect at all
            if cx2 > x_min and cx1 < x_max and cy2 > y_min and cy1 < y_max:
                active[r, c] = True

    return active, (x_min, y_min, x_max, y_max)


# ─────────────────────────────────────────────
# SCORE FUNCTIONS
# ─────────────────────────────────────────────

def compute_spatial_score(
    corners_2d: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Weighted cell-occupancy score over a GRID_ROWS × GRID_COLS grid.
    Edge cells count EDGE_WEIGHT times more than interior cells.

    Args:
        corners_2d: (N, 2) float32 image coordinates.
        img_w, img_h: frame dimensions in pixels.

    Returns:
        score    – [0, 1] weighted occupancy fraction.
        occupancy – (GRID_ROWS, GRID_COLS) bool mask.
        cell_weights – (GRID_ROWS, GRID_COLS) float weight array.
    """
    cell_w = img_w / GRID_COLS
    cell_h = img_h / GRID_ROWS

    cell_weights = np.ones((GRID_ROWS, GRID_COLS), dtype=float)
    edge_rows = {0, GRID_ROWS - 1}
    edge_cols = {0, GRID_COLS - 1}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            if r in edge_rows or c in edge_cols:
                cell_weights[r, c] = EDGE_WEIGHT

    occupancy = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    for pt in corners_2d:
        col = int(min(pt[0] / cell_w, GRID_COLS - 1))
        row = int(min(pt[1] / cell_h, GRID_ROWS - 1))
        # clamp negative coordinates (shouldn't happen but be defensive)
        col = max(col, 0)
        row = max(row, 0)
        occupancy[row, col] = True

    w_total = cell_weights.sum()
    w_occ   = (occupancy.astype(float) * cell_weights).sum()
    return float(w_occ / w_total), occupancy, cell_weights


def compute_radial_score(
    corners_2d: np.ndarray,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
) -> float:
    """
    Mean squared normalised distance of corners from the principal point.
    Score → 1 when corners are near the image corners (far from centre).

    Args:
        corners_2d: (N, 2) float32 image coordinates.
        cx, cy: principal point in pixels (use img_w/2, img_h/2 before bootstrap).
        img_w, img_h: frame dimensions.

    Returns:
        score – [0, 1].
    """
    max_r = math.sqrt(
        max(cx, img_w - cx) ** 2 + max(cy, img_h - cy) ** 2
    )
    if max_r == 0.0:
        return 0.0
    dx = corners_2d[:, 0] - cx
    dy = corners_2d[:, 1] - cy
    r  = np.sqrt(dx ** 2 + dy ** 2)
    return float(((r / max_r) ** 2).mean())


def coverage_grade(score: float) -> str:
    """Map composite score to letter grade."""
    if score >= 0.88: return "A"
    if score >= 0.80: return "B"
    if score >= 0.60: return "C"
    if score >= 0.55: return "D"
    return "F"


def grade_color(grade: str) -> Tuple[int, int, int]:
    return {"A": C_GRADE_A, "B": C_GRADE_B, "C": C_GRADE_C,
            "D": C_GRADE_D, "F": C_GRADE_F}[grade]


# ─────────────────────────────────────────────
# CALIBRATION STATE  (thread-safe via lock)
# ─────────────────────────────────────────────

class CalibState:
    """All mutable calibration metrics.  All public attributes are guarded by _lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self) -> None:
        self.spatial   = 0.0
        self.pose      = 0.0
        self.radial    = 0.0
        self.composite = 0.0
        self.grade     = "F"
        self.occupancy = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        self.rot_std   = 0.0
        self.dominant  = 1.0
        self.frames    = 0
        self.hint      = "Press SPACE to start recording"
        self.fps_proc  = 0.0
        self.queue_sz  = 0
        self.dropped   = 0     # frames skipped because queue was full
        self.radial_sum = 0.0
        self.radial_count = 0
        self.cell_hits = np.zeros(
            (GRID_ROWS, GRID_COLS),
            dtype=np.uint32
        )

        self.cell_weights = np.ones(
            (GRID_ROWS, GRID_COLS),
            dtype=np.float32
        )

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                if (
                    r == 0
                    or r == GRID_ROWS - 1
                    or c == 0
                    or c == GRID_COLS - 1
                ):
                    self.cell_weights[r, c] = EDGE_WEIGHT

        # ── Active-cell mask (from pitch segmentation) ──
        # Starts all-active; get locked down once segment_pitch_quadrilateral()
        # succeeds for the first recorded frame. Only active cells count
        # toward total_weight / occupancy / hints.
        self.active_mask   = np.ones((GRID_ROWS, GRID_COLS), dtype=bool)
        self.pitch_locked  = False
        self.pitch_quad    = None    # (4,2) quad points, for HUD overlay/debug
        self.pitch_rect    = None    # (x_min, y_min, x_max, y_max) expanded rect

        self.total_weight = (self.cell_weights * self.active_mask).sum()
        # incremental pose statistics
        self.pose_n = 0
        self.pose_mean = np.zeros(3)
        self.pose_M2 = np.zeros((3,3))

    def set_active_mask(
        self,
        active_mask: np.ndarray,
        quad: Optional[np.ndarray] = None,
        rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Lock in which grid cells count toward calibration coverage.

        Cells outside `active_mask` are excluded from total_weight, from
        occupancy accounting, and from the "uncovered edge cells" hint —
        they simply don't matter for scoring or feedback anymore.
        """
        with self._lock:
            n_active = int(active_mask.sum())
            if n_active < PITCH_MIN_ACTIVE_CELLS:
                print(f"[pitch-seg] only {n_active} active cells computed — "
                      "too few to be trustworthy, keeping all cells active.")
                return
            self.active_mask  = active_mask.copy()
            self.pitch_locked = True
            self.pitch_quad   = None if quad is None else quad.copy()
            self.pitch_rect   = rect
            self.total_weight = (self.cell_weights * self.active_mask).sum()
            # Re-derive occupancy/spatial under the new mask from existing hits.
            self.occupancy = (self.cell_hits > 0) & self.active_mask
            occupied_weight = (
                self.occupancy.astype(np.float32) * self.cell_weights
            ).sum()
            self.spatial = (
                occupied_weight / self.total_weight
                if self.total_weight > 0 else 0.0
            )

    def update_spatial_incremental(self,corners: np.ndarray,img_w: int,img_h: int) -> None:
        cell_w = img_w / GRID_COLS
        cell_h = img_h / GRID_ROWS

        for pt in corners.reshape(-1, 2):

            col = int(min(pt[0] / cell_w, GRID_COLS - 1))
            row = int(min(pt[1] / cell_h, GRID_ROWS - 1))

            col = max(col, 0)
            row = max(row, 0)

            self.cell_hits[row, col] += 1

        # Only active cells can ever be "occupied" for scoring purposes.
        self.occupancy = (self.cell_hits > 0) & self.active_mask

        occupied_weight = (
            self.occupancy.astype(np.float32)
            * self.cell_weights
        ).sum()

        self.spatial = (
            occupied_weight / self.total_weight
            if self.total_weight > 0 else 0.0
        )
    def update_pose_incremental(
        self,
        rvec: np.ndarray,
    ) -> None:

        R, _ = cv.Rodrigues(
            np.asarray(rvec, dtype=np.float64).reshape(3,1)
        )

        sy = math.sqrt(
            R[0,0]**2 +
            R[1,0]**2
        )

        if sy > 1e-6:
            x = math.degrees(math.atan2(R[2,1], R[2,2]))
            y = math.degrees(math.atan2(-R[2,0], sy))
            z = math.degrees(math.atan2(R[1,0], R[0,0]))
        else:
            x = math.degrees(math.atan2(-R[1,2], R[1,1]))
            y = math.degrees(math.atan2(-R[2,0], sy))
            z = 0.0

        euler = np.array([x, y, z])

        self.pose_n += 1

        delta = euler - self.pose_mean

        self.pose_mean += (
            delta / self.pose_n
        )

        delta2 = euler - self.pose_mean

        self.pose_M2 += np.outer(
            delta,
            delta2
        )

        if self.pose_n > 1:

            cov = (
                self.pose_M2 /
                (self.pose_n - 1)
            )

            std = np.sqrt(
                np.diag(cov)
            )

            self.rot_std = float(std.max())

            self.pose = min(
                self.rot_std /
                TARGET_STD_DEG,
                1.0
            )

            self.dominant = 1.0

    def update_radial_incremental(
        self,
        corners: np.ndarray,
        cx: float,
        cy: float,
        img_w: int,
        img_h: int,
    ) -> None:
        max_r = math.sqrt(
            max(cx, img_w - cx) ** 2
            +
            max(cy, img_h - cy) ** 2
        )

        pts = corners.reshape(-1, 2)

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        r = np.sqrt(dx ** 2 + dy ** 2)

        frame_score = float(
            ((r / max_r) ** 2).mean()
        )

        self.radial = max(
        self.radial,
        frame_score
        )

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    def snapshot(self) -> Dict:
        """Return a thread-safe copy of all display fields."""
        with self._lock:
            return dict(
                spatial=self.spatial, pose=self.pose, radial=self.radial,
                composite=self.composite, grade=self.grade,
                occupancy=self.occupancy.copy(), rot_std=self.rot_std,
                dominant=self.dominant, frames=self.frames,
                hint=self.hint, fps_proc=self.fps_proc,
                queue_sz=self.queue_sz, dropped=self.dropped,
                active_mask=self.active_mask.copy(),
                pitch_locked=self.pitch_locked,
                total_active_cells=int(self.active_mask.sum()),
            )

    def update(
        self,
        imgpoints: List[np.ndarray],
        rvecs: List[np.ndarray],
        img_w: int,
        img_h: int,
        cx: float,
        cy: float,
        fps_proc: float = 0.0,
        queue_sz: int = 0,
        dropped: int = 0,
    ) -> None:
        with self._lock:
            self.frames    = len(imgpoints)
            self.fps_proc  = fps_proc
            self.queue_sz  = queue_sz
            self.dropped   = dropped
            if self.frames == 0:
                return

            
            self.composite = (W_SPATIAL * self.spatial
                              + W_POSE   * self.pose
                              + W_RADIAL * self.radial)
            self.grade     = coverage_grade(self.composite)
            self._update_hint_locked()

    def _update_hint_locked(self) -> None:
        """Must be called with _lock already held."""
        uncovered_edges = sum(
            1 for r in range(GRID_ROWS) for c in range(GRID_COLS)
            if self.active_mask[r, c]
            and not self.occupancy[r, c]
            and (r in (0, GRID_ROWS - 1) or c in (0, GRID_COLS - 1))
        )
        if self.composite >= GRADE_A_THRESH:
            self.hint = "Grade A reached!  Stopping automatically."
        elif not self.pitch_locked:
            self.hint = "Waiting on pitch segmentation — all cells active for now"
        elif self.frames < 15:
            self.hint = "Keep moving — need more frames for a solid calibration"
        elif uncovered_edges > 6:
            self.hint = (f"{uncovered_edges} active edge cells empty"
                         " — push board to pitch boundaries")
        elif self.pose < 0.50:
            self.hint = "Tilt & rotate the board more — avoid flat frontal poses"
        elif self.radial < 0.40:
            self.hint = "Push board further toward image corners for better radial coverage"
        elif self.spatial < 0.70:
            self.hint = "Cover more red cells within the active pitch region"
        else:
            self.hint = (f"Looking good — {int(self.composite * 100)}%"
                         " composite, keep covering corners")


# ─────────────────────────────────────────────
# HUD DRAWING
# ─────────────────────────────────────────────

def _filled_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    """Draw a filled rectangle, optionally blended onto the frame."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        return
    if alpha >= 1.0:
        cv.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return
    sub = img[y1:y2, x1:x2]
    if sub.size == 0:
        return
    overlay = np.full_like(sub, color, dtype=np.uint8)
    cv.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)
    img[y1:y2, x1:x2] = sub


def draw_hud(
    frame: np.ndarray,
    snap: Dict,
    recording: bool,
    found_board: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]

    # Right-side panel background
    _filled_rect(frame, w - HUD_RIGHT_W, HUD_TOP_H, w, h - HUD_GRID_H, C_PANEL, 0.82)
    # Top hint bar
    _filled_rect(frame, 0, 0, w, HUD_TOP_H, C_HINT_BG, 0.88)

    # Recording pill
    pill_color = C_GREEN if recording else (120, 120, 120)
    pill_label = "● REC" if recording else "○ IDLE"
    cv.putText(frame, pill_label, (PANEL_PAD, 28), FONT, FONT_MD, pill_color, THICK_2, cv.LINE_AA)

    # Board detected indicator
    bd_color = C_GREEN if found_board else C_RED
    cv.putText(frame, "BOARD", (100, 28), FONT, FONT_SM, bd_color, THICK_1, cv.LINE_AA)

    # Truncate hint to prevent it overflowing into the fps text
    hint = snap["hint"]
    if len(hint) > 60:
        hint = hint[:57] + "…"
    cv.putText(frame, hint, (175, 28), FONT, FONT_SM, C_WHITE, THICK_1, cv.LINE_AA)

    # Frames / fps / queue / dropped (right of hint bar)
    dropped_txt = f"  drop:{snap['dropped']}" if snap["dropped"] > 0 else ""
    fps_txt = (f"{snap['frames']} frames  "
               f"{snap['fps_proc']:.1f}fps  "
               f"Q:{snap['queue_sz']}"
               f"{dropped_txt}")
    (fw, _), _ = cv.getTextSize(fps_txt, FONT, FONT_SM, THICK_1)
    fps_col = C_DROPPED if snap["dropped"] > 0 else (160, 160, 160)
    cv.putText(frame, fps_txt,
               (w - HUD_RIGHT_W - fw - 8, 28),
               FONT, FONT_SM, fps_col, THICK_1, cv.LINE_AA)

    # ── Grade badge ──────────────────────────────────
    gx1 = w - HUD_RIGHT_W + PANEL_PAD
    gy1 = HUD_TOP_H + PANEL_PAD
    gx2 = gx1 + GRADE_BOX_SZ
    gy2 = gy1 + GRADE_BOX_SZ
    g_col = grade_color(snap["grade"])
    _filled_rect(frame, gx1, gy1, gx2, gy2, g_col, 0.9)
    (gw, gh), _ = cv.getTextSize(snap["grade"], FONT, FONT_LG, THICK_2)
    cv.putText(frame, snap["grade"],
               (gx1 + (GRADE_BOX_SZ - gw) // 2,
                gy1 + (GRADE_BOX_SZ + gh) // 2 - 4),
               FONT, FONT_LG, C_WHITE, THICK_2, cv.LINE_AA)
    comp_txt = f"{snap['composite'] * 100:.1f}%"
    (cw, _), _ = cv.getTextSize(comp_txt, FONT, FONT_SM, THICK_1)
    cv.putText(frame, comp_txt,
               (gx1 + (GRADE_BOX_SZ - cw) // 2, gy2 + 16),
               FONT, FONT_SM, g_col, THICK_1, cv.LINE_AA)

    # ── Score bars ───────────────────────────────────
    bar_x_label = gx1
    bar_x_start = gx1 + 58
    bar_x_end   = bar_x_start + BAR_MAX_W
    bar_rows = [
        ("Spatial", snap["spatial"], C_GREEN,  "×0.40"),
        ("Pose",    snap["pose"],    C_BLUE,   "×0.40"),
        ("Radial",  snap["radial"],  C_ORANGE, "×0.20"),
    ]
    by = gy2 + 38
    for label, val, col, weight in bar_rows:
        cv.putText(frame, label,  (bar_x_label,      by), FONT, FONT_SM, (180,180,180), THICK_1, cv.LINE_AA)
        cv.putText(frame, weight, (bar_x_label + 46, by), FONT, 0.35,    (100,100,100), THICK_1, cv.LINE_AA)
        cv.rectangle(frame, (bar_x_start, by - 10), (bar_x_end, by - 10 + BAR_H), C_PANEL_LT, -1)
        fill_w = int(val * BAR_MAX_W)
        if fill_w > 0:
            cv.rectangle(frame,
                         (bar_x_start, by - 10),
                         (bar_x_start + fill_w, by - 10 + BAR_H), col, -1)
        cv.putText(frame, f"{val:.2f}", (bar_x_end + 4, by),
                   FONT, FONT_SM, col, THICK_1, cv.LINE_AA)
        by += 30

    cv.putText(frame, f"rot std  {snap['rot_std']:.1f}°",
               (bar_x_label, by + 4), FONT, FONT_SM, (120,120,120), THICK_1, cv.LINE_AA)

    # ── Bottom coverage grid ─────────────────────────
    grid_y = h - HUD_GRID_H
    _filled_rect(frame, 0, grid_y, w, h, C_PANEL, 0.82)
    covered_n = int(snap["occupancy"].sum())
    total_n   = snap["total_active_cells"]
    lock_txt  = "" if snap["pitch_locked"] else "  (pitch not yet locked — all cells shown active)"
    cv.putText(frame, f"Coverage grid — {covered_n}/{total_n} active cells{lock_txt}",
               (PANEL_PAD, grid_y + 18), FONT, FONT_SM, (180,180,180), THICK_1, cv.LINE_AA)

    avail_w   = w - 2 * PANEL_PAD
    avail_h   = HUD_GRID_H - 26
    cell_w    = avail_w // GRID_COLS
    cell_h    = avail_h // GRID_ROWS
    gx_origin = PANEL_PAD
    gy_origin = grid_y + 24
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cx1 = gx_origin + c * cell_w
            cy1 = gy_origin + r * cell_h
            cx2 = cx1 + cell_w - 2
            cy2 = cy1 + cell_h - 2
            is_edge  = r in (0, GRID_ROWS - 1) or c in (0, GRID_COLS - 1)
            is_active = bool(snap["active_mask"][r, c])
            occupied = bool(snap["occupancy"][r, c])
            if not is_active:
                fill, border = C_INACTIVE, C_INACTIVE_BORDER
            elif occupied:
                fill, border = (30,120,40), C_GREEN
            elif is_edge:
                fill, border = (50,30,60), (80,50,160)
            else:
                fill, border = (50,30,30), C_RED
            _filled_rect(frame, cx1, cy1, cx2, cy2, fill)
            cv.rectangle(frame, (cx1, cy1), (cx2, cy2), border, 1)

    # ── Grade A banner ───────────────────────────────
    if snap["grade"] == "A":
        banner_h = 52
        bx1, by1 = PANEL_PAD, h // 2 - banner_h // 2
        bx2, by2 = w - PANEL_PAD, by1 + banner_h
        _filled_rect(frame, bx1, by1, bx2, by2, (20,80,20), 0.92)
        cv.rectangle(frame, (bx1,by1), (bx2,by2), C_GREEN, 2)
        msg = "Grade A reached  |  Calibration complete — stopping"
        (mw, mh), _ = cv.getTextSize(msg, FONT, FONT_MD, THICK_2)
        cv.putText(frame, msg,
                   ((w - mw) // 2, by1 + (banner_h + mh) // 2 - 2),
                   FONT, FONT_MD, C_WHITE, THICK_2, cv.LINE_AA)

    return frame


# ─────────────────────────────────────────────
# PARALLEL WORKER
# ─────────────────────────────────────────────

def detect_frame(
    item: Tuple[np.ndarray, int, int, Optional[np.ndarray], Optional[np.ndarray]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Runs in a worker thread.

    Args:
        item: (gray_frame, corners_quick, mtx_or_None, dist_or_None)

    Returns:
        (corners2, rvec) on success, or (None, None) if board not found.
        rvec is None if mtx was not yet available.
    """
    gray, corners, mtx, dist = item
    corners2 = cv.cornerSubPix(
        gray,
        corners,
        (11,11),
        (-1,-1),
        _CRITERIA
    )

    rvec = None

    if mtx is not None:
        ok, rv, _ = cv.solvePnP(
            objp,
            corners2,
            mtx,
            dist
        )

        if ok:
            rvec = rv

    return corners2, rvec


# ─────────────────────────────────────────────
# COLLECTOR THREAD
# ─────────────────────────────────────────────

def collector_thread_fn(
    result_q: queue.Queue,
    state: CalibState,
    shared: Dict,
    shared_lock: threading.Lock,
    stop_event: threading.Event,
    dropped_counter: List[int],   # [0] = mutable dropped count
) -> None:
    """
    Drains result_q, accumulates data, periodically re-scores and
    bootstraps the live camera matrix.

    shared keys
    -----------
    img_w, img_h     – frame dimensions (set by main thread before first frame)
    objpoints        – list of objp copies (one per accepted frame)
    imgpoints        – list of corner arrays (one per accepted frame)
    rvecs_live       – rolling list of rotation vectors
    mtx_live         – None until bootstrap succeeds
    dist_live        – None until bootstrap succeeds
    bootstrap_done   – bool, prevents repeated bootstrap attempts
    """
    objpoints  = shared["objpoints"]
    imgpoints  = shared["imgpoints"]
    rvecs_live = shared["rvecs_live"]

    proc_count     = 0
    t_start        = time.perf_counter()
    proc_times: deque = deque(maxlen=ROLLING_FPS_WINDOW)   # for rolling fps
    UPDATE_EVERY   = 5   # recompute scores every N accepted frames

    while not stop_event.is_set() or not result_q.empty():
        try:
            item = result_q.get(timeout=0.1)
        except queue.Empty:
            continue

        corners2, rvec = item
        if corners2 is None:
            continue

        with shared_lock:
            objpoints.append(objp.copy())
            imgpoints.append(corners2.copy())

            n_frames = len(imgpoints)

            if rvec is not None:
                rvecs_live.append(rvec)

                if len(rvecs_live) > POSE_HISTORY_LIMIT:
                    del rvecs_live[: len(rvecs_live) - POSE_HISTORY_LIMIT]

                with state._lock:
                    state.update_pose_incremental(
                        rvec
                    )

            bootstrap_done = shared["bootstrap_done"]

            img_w = shared["img_w"]
            img_h = shared["img_h"]

            mtx = shared["mtx_live"]
        cx = (
            float(mtx[0,2])
            if mtx is not None
            else img_w / 2.0
        )

        cy = (
            float(mtx[1,2])
            if mtx is not None
            else img_h / 2.0
        )
        with state._lock:

            state.update_spatial_incremental(
                corners2,
                img_w,
                img_h
            )

            state.update_radial_incremental(
                corners2,
                cx,
                cy,
                img_w,
                img_h
            )

        proc_count += 1
        now = time.perf_counter()
        proc_times.append(now)

        # ── Bootstrap camera matrix (runs once, inside collector thread) ──
        if not bootstrap_done and n_frames >= BOOTSTRAP_MIN_FRAMES:
            _bootstrap_calibration(shared, shared_lock, img_w, img_h)

        # ── Periodic score update ────────────────────────────────────────
        if proc_count % UPDATE_EVERY == 0 or proc_count <= 5:
            with shared_lock:
                mtx        = shared["mtx_live"]
                img_w_snap = shared["img_w"]
                img_h_snap = shared["img_h"]

            cx = float(mtx[0, 2]) if mtx is not None else (img_w_snap or 640) / 2.0
            cy = float(mtx[1, 2]) if mtx is not None else (img_h_snap or 480) / 2.0

            # Rolling fps: frames in the last ROLLING_FPS_WINDOW proc events
            if len(proc_times) >= 2:
                fps_proc = (len(proc_times) - 1) / (proc_times[-1] - proc_times[0] + 1e-9)
            else:
                fps_proc = 0.0

            with shared_lock:
                q_sz    = result_q.qsize()
                dropped = dropped_counter[0]

            state.update(
                imgpoints, rvecs_live,
                img_w_snap, img_h_snap, cx, cy,
                fps_proc=fps_proc,
                queue_sz=q_sz,
                dropped=dropped,
            )
            publish_engine_snapshot(calibration_state, state.snapshot())


def save_calibration_parameters(
    mtx: np.ndarray,
    dist: np.ndarray,
    save_path: str = "calib_params.npz",
    history_json_path: str = "calib_history.json",
    history_npz_path: str = "calib_history.npz",
    stage: str = "bootstrap",
    ret_rms: Optional[float] = None,
    rvecs: Optional[Any] = None,
    tvecs: Optional[Any] = None,
    num_frames: Optional[int] = None,
) -> None:
    """
    Saves the latest intrinsic matrix and distortion coefficients to calib_params.npz,
    and appends all calibrations with a timestamp to calib_history.json and calib_history.npz.
    """
    now_dt = datetime.now()
    now_iso = now_dt.isoformat()

    # 1. Update most recent parameters in calib_params.npz
    save_kwargs: Dict[str, Any] = {
        "mtx": mtx,
        "dist": dist,
        "K": mtx,
        "dist_coeffs": dist,
        "timestamp": now_iso,
        "stage": stage,
    }
    if rvecs is not None:
        save_kwargs["rvecs"] = rvecs
    if tvecs is not None:
        save_kwargs["tvecs"] = tvecs
    if ret_rms is not None:
        save_kwargs["rms"] = ret_rms
    if num_frames is not None:
        save_kwargs["num_frames"] = num_frames

    try:
        np.savez(save_path, **save_kwargs)
        print(f"💾  [calib-params] Most recent parameters stored in {save_path} (stage: {stage})")
    except Exception as e:
        print(f"❌  [calib-params] Failed to save {save_path}: {e}")

    # 2. Append to calib_history.json
    history_records = []
    if os.path.isfile(history_json_path):
        try:
            with open(history_json_path, "r", encoding="utf-8") as f:
                history_records = json.load(f)
                if not isinstance(history_records, list):
                    history_records = [history_records]
        except Exception as e:
            print(f"⚠  [calib-history] Failed reading {history_json_path}, creating fresh list: {e}")
            history_records = []

    new_record = {
        "timestamp": now_iso,
        "stage": stage,
        "num_frames": int(num_frames) if num_frames is not None else None,
        "rms_error": float(ret_rms) if ret_rms is not None else None,
        "mtx": mtx.tolist() if mtx is not None else None,
        "dist_coeffs": dist.flatten().tolist() if dist is not None else None,
    }
    history_records.append(new_record)

    try:
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(history_records, f, indent=2)
        print(f"📜  [calib-history] Appended calibration record ({stage}) to {history_json_path} (total records: {len(history_records)})")
    except Exception as e:
        print(f"❌  [calib-history] Failed writing {history_json_path}: {e}")

    # 3. Append to calib_history.npz
    try:
        timestamps_all = []
        stages_all = []
        mtx_all = []
        dist_all = []
        if os.path.isfile(history_npz_path):
            with np.load(history_npz_path, allow_pickle=True) as h_data:
                if "timestamps" in h_data:
                    timestamps_all = list(h_data["timestamps"])
                if "stages" in h_data:
                    stages_all = list(h_data["stages"])
                if "mtx_all" in h_data:
                    mtx_all = list(h_data["mtx_all"])
                if "dist_all" in h_data:
                    dist_all = list(h_data["dist_all"])

        timestamps_all.append(now_iso)
        stages_all.append(stage)
        mtx_all.append(mtx)
        dist_all.append(dist.flatten())

        np.savez(
            history_npz_path,
            timestamps=np.array(timestamps_all, dtype=object),
            stages=np.array(stages_all, dtype=object),
            mtx_all=np.array(mtx_all, dtype=np.float64),
            dist_all=np.array(dist_all, dtype=np.float64),
        )
    except Exception as e:
        print(f"⚠  [calib-history] Failed updating {history_npz_path}: {e}")


def normalize_rtsp_url(url: str) -> str:
    """Safely URL-encodes special characters in RTSP passwords (e.g. '@' in password)."""
    if not isinstance(url, str) or not (url.startswith("rtsp://") or url.startswith("rtsps://")):
        return url
    try:
        parts = url.split("://", 1)
        scheme = parts[0]
        rest = parts[1]
        if rest.count("@") > 1:
            userinfo, hostinfo = rest.rsplit("@", 1)
            if ":" in userinfo:
                user, pwd = userinfo.split(":", 1)
                import urllib.parse
                encoded_pwd = urllib.parse.quote(pwd, safe="")
                return f"{scheme}://{user}:{encoded_pwd}@{hostinfo}"
    except Exception:
        pass
    return url


class ThreadedVideoCapture:
    """
    Non-blocking, zero-latency threaded VideoCapture for RTSP streams and webcams.
    
    Prevents stream freezing by reading frames continuously in a background thread
    and always serving only the freshest frame, bypassing OpenCV's internal queue buildup.
    Robustly handles HEVC (H.265) and H.264 streams with automatic recovery from transient
    decoder exceptions (e.g. PPS/SPS packet drops).
    """
    def __init__(self, src: Union[int, str], width: Optional[int] = None, height: Optional[int] = None):
        self.src = normalize_rtsp_url(src) if isinstance(src, str) else src
        self.width = width
        self.height = height
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_event = threading.Event()
        self._last_frame_time = time.time()
        self._fps = 25.0
        
        self._is_rtsp = isinstance(self.src, str) and (self.src.startswith("rtsp://") or self.src.startswith("rtsps://"))
        if self._is_rtsp:
            # Enforce TCP transport and buffer purging without reorder_queue_size;0 which breaks HEVC/H.265
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|fflags;nobuffer|max_delay;500000"
            )
            
        self.cap = self._open_capture()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True, name="threaded-capture")
        self._thread.start()
        
        # Wait up to 5s for the first frame
        self._frame_event.wait(timeout=5.0)

    def _open_capture(self) -> cv.VideoCapture:
        if self._is_rtsp:
            cap = cv.VideoCapture(self.src, cv.CAP_FFMPEG)
            cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv.VideoCapture(self.src)
            
        if self.width:
            cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)
            
        fps = cap.get(cv.CAP_PROP_FPS)
        if fps and 0 < fps <= 120:
            self._fps = fps
        return cap

    def _capture_worker(self) -> None:
        reconnect_delay = 2.0
        consecutive_errors = 0
        while not self._stopped.is_set():
            if not self.cap.isOpened():
                if self._is_rtsp:
                    print(f"[threaded-capture] Reconnecting to RTSP stream: {self.src} ...")
                    time.sleep(reconnect_delay)
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = self._open_capture()
                    continue
                else:
                    break

            try:
                ret, frame = self.cap.read()
                consecutive_errors = 0
            except cv.error as exc:
                # Catch transient C++ decoder exceptions (e.g. HEVC PPS/SPS drops) without crashing thread
                consecutive_errors += 1
                if consecutive_errors == 30:
                    print(f"[threaded-capture] Sustained decode errors ({exc}). Waiting for clean keyframe...")
                elif consecutive_errors > 60:
                    print(f"[threaded-capture] Multiple decode errors, reconnecting stream: {self.src} ...")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    self.cap = self._open_capture()
                    consecutive_errors = 0
                time.sleep(0.005)
                continue
            except Exception:
                consecutive_errors += 1
                time.sleep(0.005)
                continue

            if ret and frame is not None:
                with self._lock:
                    self._latest_frame = frame
                    self._last_frame_time = time.time()
                self._frame_event.set()
            else:
                if self._is_rtsp:
                    if (time.time() - self._last_frame_time) > 4.0:
                        print(f"[threaded-capture] Stream stalled (timeout). Reopening: {self.src} ...")
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        time.sleep(1.0)
                        self.cap = self._open_capture()
                        self._last_frame_time = time.time()
                    else:
                        time.sleep(0.005)
                else:
                    time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._stopped.is_set():
            return False, None
            
        with self._lock:
            if self._latest_frame is not None:
                return True, self._latest_frame.copy()

        # If not yet available, wait briefly
        if self._frame_event.wait(timeout=1.0):
            with self._lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame.copy()
                    
        return False, None

    def isOpened(self) -> bool:
        return self.cap.isOpened() and not self._stopped.is_set()

    def get(self, prop_id: int) -> float:
        if prop_id == cv.CAP_PROP_FPS and self._fps > 0:
            return self._fps
        return self.cap.get(prop_id)

    def set(self, prop_id: int, value: float) -> bool:
        return self.cap.set(prop_id, value)

    def release(self) -> None:
        self._stopped.set()
        self._frame_event.set()
        try:
            self.cap.release()
        except Exception:
            pass


def _bootstrap_calibration(
    shared: Dict,
    shared_lock: threading.Lock,
    img_w: int,
    img_h: int,
) -> None:
    """
    Run an initial calibrateCamera on the first BOOTSTRAP_MIN_FRAMES frames,
    store mtx_live/dist_live, and backfill rvecs for all prior frames.

    Must only be called from the collector thread (single caller guaranteed).
    """
    with shared_lock:
        # Guard against a race where two calls slip through before the flag
        # is set (shouldn't happen in single-collector design, but be safe).
        if shared["bootstrap_done"]:
            return
        obj_snap = shared["objpoints"][:BOOTSTRAP_MIN_FRAMES]
        img_snap = shared["imgpoints"][:BOOTSTRAP_MIN_FRAMES]

    try:
        _, mtx, dist, _, _ = cv.calibrateCamera(
            obj_snap, img_snap, (img_w, img_h), None, None)
    except cv.error as e:
        print(f"[bootstrap] calibrateCamera failed: {e}")
        return

    with shared_lock:
        shared["mtx_live"]      = mtx
        shared["dist_live"]     = dist
        shared["bootstrap_done"] = True

        # Backfill rvecs for ALL frames collected so far
        # (rvecs_live is empty at this point because mtx was None during detection)
        rvecs_live = shared["rvecs_live"]
        if not rvecs_live:
            for _obj, _img in zip(shared["objpoints"], shared["imgpoints"]):
                ok, rv, _ = cv.solvePnP(_obj, _img, mtx, dist)
                if ok:
                    rvecs_live.append(rv)
            if len(rvecs_live) > POSE_HISTORY_LIMIT:
                del rvecs_live[: len(rvecs_live) - POSE_HISTORY_LIMIT]

    print(f"[bootstrap] Camera matrix estimated from {BOOTSTRAP_MIN_FRAMES} frames  "
          f"(fx={mtx[0,0]:.1f}  fy={mtx[1,1]:.1f}  "
          f"cx={mtx[0,2]:.1f}  cy={mtx[1,2]:.1f})")

    # Save bootstrap calibration parameters
    save_calibration_parameters(
        mtx=mtx,
        dist=dist,
        stage="bootstrap",
        num_frames=BOOTSTRAP_MIN_FRAMES,
    )


# ─────────────────────────────────────────────
# FINAL CALIBRATION
# ─────────────────────────────────────────────

def run_final_calibration(
    objpoints: List[np.ndarray],
    imgpoints: List[np.ndarray],
    img_shape: Tuple[int, int],   # (height, width) — same as gray.shape
    save_path: str = "calib_params.npz",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Two-pass calibration:
      1. Calibrate on up to 200 randomly sampled frames.
      2. Re-calibrate on the 50 frames with the lowest reprojection error.

    Args:
        objpoints: list of objp arrays.
        imgpoints: list of corner arrays.
        img_shape: (height, width) as returned by gray.shape.
        save_path: where to write calib_params.npz.

    Returns:
        (mtx, dist) or (None, None) if calibration failed.
    """
    MIN_FRAMES = 30
    if len(objpoints) < MIN_FRAMES:
        print(f"\n⚠  Not enough frames to calibrate (have {len(objpoints)}, need ≥{MIN_FRAMES}).")
        return None, None

    img_size = (img_shape[1], img_shape[0])   # (width, height) for OpenCV

    n_sample = min(len(objpoints), 100)
    idx      = random.sample(range(len(objpoints)), n_sample)
    print(f"\n📷  Pass 1 — calibrating on {n_sample} frames …")
    try:
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
            [objpoints[i] for i in idx],
            [imgpoints[i] for i in idx],
            img_size, None, None,
        )
    except cv.error as e:
        print(f"❌  calibrateCamera failed: {e}")
        return None, None

    # Per-frame reprojection error
    frame_errors: List[Tuple[float, np.ndarray, np.ndarray]] = []
    for obj, img in zip(objpoints, imgpoints):
        ok, rv, tv = cv.solvePnP(obj, img, mtx, dist)
        if not ok:
            continue
        proj, _ = cv.projectPoints(obj, rv, tv, mtx, dist)
        err = float(np.linalg.norm(img.reshape(-1, 2) - proj.reshape(-1, 2), axis=1).mean())
        frame_errors.append((err, obj, img))

    frame_errors.sort(key=lambda x: x[0])
    best_n   = max(80, len(frame_errors))
    best_obj = [e[1] for e in frame_errors[:best_n]]
    best_img = [e[2] for e in frame_errors[:best_n]]

    print(f"🎯  Pass 2 — re-calibrating on best {best_n} frames "
          f"(median reprojection error: {frame_errors[best_n//2][0]:.3f} px) …")
    try:
        ret_f, mtx_f, dist_f, rv_f, tv_f = cv.calibrateCamera(
            best_obj, best_img, img_size, None, None)
    except cv.error as e:
        print(f"❌  Second-pass calibrateCamera failed: {e}")
        return None, None

    # Save final calibration parameters to calib_params.npz and history files
    save_calibration_parameters(
        mtx=mtx_f,
        dist=dist_f,
        save_path=save_path,
        stage="final",
        ret_rms=ret_f,
        rvecs=rv_f,
        tvecs=tv_f,
        num_frames=best_n,
    )

    # ── Pretty terminal output ──────────────────────────────
    sep = "═" * 64
    rms_label = ("✅ excellent" if ret_f < 0.5
                 else "⚠  acceptable" if ret_f < 1.0
                 else "❌ poor — consider re-collecting data")
    print(f"\n{sep}")
    print("  🔬  FINAL INTRINSIC CALIBRATION RESULTS")
    print(sep)
    print(f"  Frames collected  : {len(objpoints)}")
    print(f"  Frames used       : {best_n}  (lowest-error subset)")
    print(f"  RMS reprojection  : {ret_f:.4f} px  ({rms_label})")
    print()
    print("  Camera Matrix (K):")
    print(f"    fx = {mtx_f[0, 0]:10.4f} px      cx = {mtx_f[0, 2]:10.4f} px")
    print(f"    fy = {mtx_f[1, 1]:10.4f} px      cy = {mtx_f[1, 2]:10.4f} px")
    print()
    print("  Distortion Coefficients  [k1, k2, p1, p2, k3]:")
    d      = dist_f.flatten()
    labels = ["k1 (radial)", "k2 (radial)", "p1 (tangential)",
              "p2 (tangential)", "k3 (radial)"]
    for lbl, val in zip(labels, d):
        bar = "▓" * max(int(abs(val) * 200), 1)
        print(f"    {lbl:<18} {val:+.6f}  {bar}")
    print()
    print("  Full matrix K:")
    for row in mtx_f:
        print(f"    {row}")
    print(f"\n  Saved → {save_path}")
    print(sep)
    return mtx_f, dist_f


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(override_args: Optional[List[str]] = None) -> None:
    save_dir = "recorded_frames"
    os.makedirs(save_dir, exist_ok=True)
    video_dir = "Videos"
    os.makedirs(video_dir, exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Intrinsic camera calibration with live HUD (parallel edition)")
    parser.add_argument("--test", action="store_true",
                        help="Run built-in self-tests and exit")
    parser.add_argument("--source", default=None,
                        help="Video source: integer index, RTSP URL, MJPEG URL, or video file path")
    parser.add_argument("--rtsp", default=None,
                        help="RTSP stream URL (e.g. rtsp://user:pass@ip:554/stream)")
    parser.add_argument("--threaded", action="store_true",
                        help="Force threaded non-blocking frame grabber (always used for RTSP)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"Worker threads for parallel detection (default {NUM_WORKERS})")
    parser.add_argument("--save", default="calib_params.npz",
                        help="Output path for calibration parameters (default: calib_params.npz)")
    parser.add_argument("--pitch-model", default=PITCH_SEG_MODEL_PATH,
                        help="Path to YOLO segmentation weights for pitch detection "
                             f"(default {PITCH_SEG_MODEL_PATH})")
    parser.add_argument("--pitch-margin", type=float, default=PITCH_MARGIN_RATIO,
                        help="Fraction to expand the pitch bounding box by on each side "
                             f"(default {PITCH_MARGIN_RATIO})")
    parser.add_argument("--no-pitch-mask", action="store_true",
                        help="Skip pitch segmentation entirely; keep all grid cells active")
    args = parser.parse_args(override_args)

    # if args.test:
    #     n_fail = run_self_tests()
    #     sys.exit(0 if n_fail == 0 else 1)

    # ── Camera / video source ────────────────────────────────────
    stream_url: any = 0
    if args.rtsp is not None:
        stream_url = args.rtsp
    elif args.source is not None:
        try:
            stream_url = int(args.source)
        except ValueError:
            stream_url = args.source
    elif HAS_REQUESTS:
        try:
            r       = requests.get("https://192.168.1.14:8443/devices",
                                   verify=False, timeout=5)
            data    = r.json()
            dev_id  = data["devices"][0]
            stream_url = f"https://192.168.1.14:8443/mjpeg?id={dev_id}"
            print(f"Connecting to device: {dev_id}")
        except Exception as e:
            print(f"Could not reach camera API ({e}).  Falling back to webcam 0.")

    is_rtsp = isinstance(stream_url, str) and (
        stream_url.startswith("rtsp://") or stream_url.startswith("rtsps://")
    )
    if is_rtsp or args.threaded:
        stream_kind = "RTSP" if is_rtsp else "Threaded"
        print(f"🌐  Opening low-latency {stream_kind} capture: {stream_url}")
        cap = ThreadedVideoCapture(stream_url, width=1920, height=1080)
    else:
        cap = cv.VideoCapture(stream_url)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

    cv.namedWindow("Intrinsic Calibration [Parallel]", cv.WINDOW_NORMAL)
    cv.resizeWindow("Intrinsic Calibration [Parallel]", 1600, 900)

    ret, frame = cap.read()
    if ret:
        print(f"Actual frame shape: {frame.shape}")
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {stream_url!r}")

    # ── Shared state ─────────────────────────────────────────────
    state       = CalibState()
    shared_lock = threading.Lock()
    shared: Dict = {
        "img_w": None, "img_h": None,
        "objpoints": [], "imgpoints": [], "rvecs_live": [],
        "mtx_live": None, "dist_live": None,
        "bootstrap_done": False,
    }
    dropped_counter = [0]   # mutable, incremented in main thread

    result_q: queue.Queue = queue.Queue()
    stop_collect = threading.Event()
    executor     = ThreadPoolExecutor(max_workers=args.workers)
    active_futures: set = set()

    # Pitch model is loaded lazily on first use, cached in get_pitch_model().
    pitch_seg_pending = not args.no_pitch_mask   # becomes False once we've tried once

    def _submit_and_forward(item):
        """Worker: detect corners and push result into result_q."""
        corners2, rvec = detect_frame(item)
        result_q.put((corners2, rvec))

    # ── Collector thread ─────────────────────────────────────────
    collector = threading.Thread(
        target=collector_thread_fn,
        args=(result_q, state, shared, shared_lock, stop_collect, dropped_counter),
        daemon=True,
        name="collector",
    )
    collector.start()

    # ── Main capture loop ────────────────────────────────────────
    recording   = False
    frame_count = 0
    last_snap   = state.snapshot()
    gray_last: Optional[np.ndarray] = None

    # Video recording state
    video_writer: Optional[cv.VideoWriter] = None
    video_path: Optional[str] = None
    video_frames_written = 0

    print("\nControls:")
    print("  SPACE – toggle recording (first press also runs pitch segmentation, saves video to Videos/)")
    print("  ESC   – stop, calibrate, quit")
    print("  Q     – quit immediately")
    print("  T     – run self-tests in terminal")
    print(f"\nParallel workers: {args.workers}  |  "
          f"Frame stride: {FRAME_STRIDE}  |  "
          f"Queue cap: {MAX_QUEUE_DEPTH}\n")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("⚠  No frame received — end of stream or camera disconnected.")
            break
        raw_frame = frame.copy()
        # Set image dimensions once
        with shared_lock:
            if shared["img_w"] is None:
                shared["img_h"], shared["img_w"] = frame.shape[:2]
            img_w = shared["img_w"]
            img_h = shared["img_h"]

        frame_count += 1

        # Write clean frame to video writer if recording is active
        if recording and video_writer is not None:
            video_writer.write(raw_frame)
            video_frames_written += 1

        # Fast board detection for HUD feedback (no subpix, every frame)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        found_quick, corners_quick = cv.findChessboardCorners(
            gray, (CHECKERBOARD_W, CHECKERBOARD_H), None)
        if found_quick:
            cv.drawChessboardCorners(
                frame, (CHECKERBOARD_W, CHECKERBOARD_H), corners_quick, found_quick)

        # Enqueue for parallel full detection (strided, bounded)
        active_futures = {f for f in active_futures if not f.done()}
        if recording and found_quick and frame_count % FRAME_STRIDE == 0:
            if len(active_futures) < MAX_QUEUE_DEPTH:
                with shared_lock:
                    mtx_snap  = shared["mtx_live"]
                    dist_snap = shared["dist_live"]
                item = (
                    gray.copy(),
                    corners_quick.copy(),
                    mtx_snap,
                    dist_snap
                )
                fut  = executor.submit(_submit_and_forward, item)
                active_futures.add(fut)
            else:
                dropped_counter[0] += 1

        # Pull latest snapshot for HUD
        last_snap = state.snapshot()

        draw_hud(frame, last_snap, recording, bool(found_quick))
        cv.imshow("Intrinsic Calibration [Parallel]", frame)
        gray_last = gray

        key = cv.waitKeyEx(1)

        if key == ord(' '):
            recording = not recording
            print(f"\n▶  Recording: {'ON' if recording else 'OFF'}")
            if recording:
                dropped_counter[0] = 0
                filename = "recorded_frames/stump_image.png"
                cv.imwrite(filename, raw_frame)

                # Initialize VideoWriter in Videos/ folder
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_path = os.path.join(video_dir, f"recording_{timestamp_str}.mp4")
                h_f, w_f = raw_frame.shape[:2]
                fps_val = cap.get(cv.CAP_PROP_FPS)
                if not fps_val or fps_val <= 0 or fps_val > 60:
                    fps_val = 25.0
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                video_writer = cv.VideoWriter(video_path, fourcc, fps_val, (w_f, h_f))
                if not video_writer.isOpened():
                    # Fallback to AVI / XVID if MP4 codec is unavailable
                    video_path = os.path.join(video_dir, f"recording_{timestamp_str}.avi")
                    fourcc = cv.VideoWriter_fourcc(*'XVID')
                    video_writer = cv.VideoWriter(video_path, fourcc, fps_val, (w_f, h_f))

                video_frames_written = 0
                print(f"🎥  [recording] Started saving video to: {video_path} ({w_f}x{h_f} @ {fps_val:.1f} fps)")

                # ── First recording start: segment the (assumed empty) pitch
                #    from the current frame and lock the active-cell mask.
                if pitch_seg_pending:
                    pitch_seg_pending = False
                    print("[pitch-seg] recording started — segmenting pitch "
                          "from current frame …")
                    quad = segment_pitch_quadrilateral(
                        frame, model=get_pitch_model(args.pitch_model))
                    if quad is not None:
                        active_mask, rect = compute_active_grid_from_quad(
                            quad, img_w, img_h, margin_ratio=args.pitch_margin)
                        state.set_active_mask(active_mask, quad=quad, rect=rect)
                        n_active = int(active_mask.sum())
                        print(f"[pitch-seg] locked {n_active}/{GRID_ROWS*GRID_COLS} "
                              f"active cells from pitch quad {quad.tolist()}")
                    else:
                        print("[pitch-seg] no pitch detected — leaving all cells active.")
            else:
                if video_writer is not None:
                    video_writer.release()
                    print(f"💾  [recording] Video saved: {video_path} ({video_frames_written} frames)")
                    video_writer = None

        elif key == 27:  # ESC
            print("\nESC — stopping and calibrating …")
            break

        elif key in (ord('q'), ord('Q')):
            if video_writer is not None:
                video_writer.release()
                print(f"💾  [recording] Video saved: {video_path} ({video_frames_written} frames)")
                video_writer = None
            stop_collect.set()
            executor.shutdown(wait=False)
            cap.release()
            cv.destroyAllWindows()
            return

        elif key in (ord('t'), ord('T')):
            print()
            # run_self_tests()
            print()

        # Auto-stop when grade A is reached
        if last_snap["grade"] == "A" and recording:
            print("\n✅  Grade A reached — stopping recording automatically.")
            recording = False
            if video_writer is not None:
                video_writer.release()
                print(f"💾  [recording] Video saved: {video_path} ({video_frames_written} frames)")
                video_writer = None
            cv.waitKey(1500)
            break

    # ── Shutdown ──────────────────────────────────────────────────
    print("\nShutting down worker pool …")
    recording = False
    if video_writer is not None:
        video_writer.release()
        print(f"💾  [recording] Video saved: {video_path} ({video_frames_written} frames)")
        video_writer = None

    executor.shutdown(wait=True)
    stop_collect.set()
    collector.join(timeout=10)
    cap.release()
    cv.destroyAllWindows()

    with shared_lock:
        obj_copy = shared["objpoints"][:]
        img_copy = shared["imgpoints"][:]

    if gray_last is None:
        print("No frames were captured.")
        return

    run_final_calibration(obj_copy, img_copy, gray_last.shape, save_path=args.save)


if __name__ == "__main__":
    main()