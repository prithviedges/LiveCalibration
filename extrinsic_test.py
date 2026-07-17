"""
Standalone extrinsic calibration test.

Loads intrinsics (mtx/K, dist/dist_coeffs) from camera_params.npz. If the
npz also contains a pitch-scene extrinsic pose (rvec/tvec, or R/t -
distinct from the per-checkerboard-frame rvecs/tvecs produced during
intrinsic calibration), that pose is used directly. Otherwise raises,
since no fresh solvePnP fallback is wired up here. Observed 2D image
points are loaded from recorded_frames/stump_image.csv (all rows, in
file order, no split). Then checks:
    1) mean/max reprojection error against the known WORLD_POINTS correspondences
    2) camera ground distance to the pitch center

Convention:
    x_cam = R @ X + t
    x_img_h = K @ x_cam  (with lens distortion applied per `dist`)
    x_img = x_img_h[:2] / x_img_h[2]

    Camera center in world coordinates: C = -R.T @ t
"""

import os
import csv
import numpy as np
import cv2


# --------------------------------------------------------------------------
# CONFIG - replace these with your real pitch correspondences
# --------------------------------------------------------------------------

CALIB_NPZ_PATH = "camera_params.npz"

RECORDED_FRAMES_DIR = "recorded_frames"
STUMP_CSV_NAME = "stump_image.csv"   # columns: image, point, x, y - all rows used, no split

USE_MANUAL_PARAMETERS = False   # True = ignore camera_params.npz completely

MANUAL_K = np.array([
    [647.0390,   0.0,    340.0715],
    [0.0,      638.5018, 234.2572],
    [0.0,        0.0,      1.0]
], dtype=np.float64)

MANUAL_DIST = np.array([
    0.013613,
    0.086619,
    0.002772,
    0.001628,
   -0.470705
], dtype=np.float64).reshape(-1, 1)

MANUAL_R = np.array([
    [ 0.9854, -0.1280, -0.1124],
    [-0.0883, -0.9481,  0.3054],
    [-0.1456, -0.2910, -0.9456]
], dtype=np.float64)

MANUAL_T = np.array([
    [0.6545],
    [-0.4335],
    [6.9230]
], dtype=np.float64)



# 3D world points (e.g. pitch markings: corners, center circle, etc.)
# and their known/measured 2D pixel correspondences.
WORLD_POINTS = np.array([
    [-1.3200,  0.0, -1.22],  # P1
        [-0.8882,  0.0, -1.22],  # P2
        # -------------------------------
        [ 0.8882,  0.0, -1.22],  # P3
        [ 1.3200,  0.0, -1.22],  # P4
        # -------------------------------
        [-1.3200,  0.0,  0.0],   # P5
        [-0.8882,  0.0,  0.0],   # P6
        # -------------------------------
        [ 0.8882,  0.0,  0.0],   # P7
        [ 1.3200,  0.0,  0.0],   # P8
        # -------------------------------
        [-1.3200,  0.0,  17.68], # P9
        [-0.8882,  0.0,  17.68], # P10
        # -------------------------------
        [ 0.8882,  0.0,  17.68],  # P11
        [ 1.3200,  0.0,  17.68],  # P12
        # -------------------------------
        [-1.3200,  0.0,  18.9],  # P13
        [-0.8882,  0.0,  18.9],  # P14
        # -------------------------------
        [ 0.8882,  0.0,  18.9],  # P15
        [ 1.3200,  0.0,  18.9],  # P16
        [-0.1143, 0.7112, -1.22],   # Q1
        [ 0.1143, 0.7112, -1.22],   # Q2
        [-0.1143, 0.7112,  18.9],   # Q3
        [ 0.1143, 0.7112,  18.9],   # Q4
], dtype=np.float64)

IMAGE_POINTS_OBSERVED = np.array([
    (326, 190),
    (367, 191),
    (410, 194),
    (453, 196),
    (318, 203),
    (364, 207),
    (410, 210),
    (458, 211),
    (253, 319),
    (328, 327),
    (408, 336),
    (491, 343),
    (227, 365),
    (312, 376),
    (406, 388),
    (506, 401),
    (374, 116),
    (388, 116),
    (317, 232),
    (350, 235),
], dtype=np.float64)

# Known pitch center in world coordinates (often (0, 0, 0))
PITCH_CENTER_WORLD = np.array([0.0, 0.0, 0.0])

# --- thresholds you supply ---
MAX_MEAN_REPROJECTION_ERROR_PX = 3.0     # acceptable average pixel error
EXPECTED_CAMERA_TO_PITCH_CENTER_M = 6.7  # ground distance, the value "you input"
DISTANCE_TOLERANCE_M = 0.5               # allowed deviation from expected

# index of the "up" axis in your world frame (0=X, 1=Y, 2=Z).
# WORLD_POINTS/PITCH_CENTER_WORLD above sit at Y=0 on the pitch plane.
UP_AXIS = 1

# Candidate key-pairs for a pitch-scene extrinsic pose, in priority order.
# The npz's own 'rvecs'/'tvecs' (plural) are per-checkerboard-frame poses
# from intrinsic calibration and are deliberately NOT in this list -
# they are not a pitch extrinsic and must never be used as one.
EXTRINSIC_KEY_CANDIDATES = [
    ("rvec", "tvec"),
    ("R", "t"),
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def load_extrinsic_from_npz(npz):
    """
    Return (rvec, tvec) for the pitch-scene extrinsic if the npz already
    contains one, else None. Accepts either a rotation vector ('rvec') or
    a rotation matrix ('R') paired with a translation ('tvec'/'t').
    """
    for r_key, t_key in EXTRINSIC_KEY_CANDIDATES:
        if r_key in npz.files and t_key in npz.files:
            r_val = npz[r_key]
            t_val = np.asarray(npz[t_key], dtype=np.float64).reshape(3, 1)
            if r_val.shape == (3, 3):
                rvec, _ = cv2.Rodrigues(r_val.astype(np.float64))
            else:
                rvec = np.asarray(r_val, dtype=np.float64).reshape(3, 1)
            print(f"Found pitch extrinsic in npz under keys "
                  f"('{r_key}', '{t_key}') - using it directly.")
            return rvec, t_val
    return None

def load_intrinsics_from_npz(npz):
    """
    Return (K, dist) from an npz, tolerant of either the checkerboard-calibration
    key names ('mtx', 'dist') or the pnp-solver output key names ('K', 'dist_coeffs').
    """
    if "mtx" in npz.files:
        K = npz["mtx"]
    elif "K" in npz.files:
        K = npz["K"]
    else:
        raise KeyError(f"No 'mtx' or 'K' key found in {CALIB_NPZ_PATH} for intrinsics.")

    if "dist" in npz.files:
        dist = npz["dist"]
    elif "dist_coeffs" in npz.files:
        dist = npz["dist_coeffs"]
    else:
        raise KeyError(f"No 'dist' or 'dist_coeffs' key found in {CALIB_NPZ_PATH} for distortion.")

    return np.asarray(K, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def load_observed_image_points(recorded_frames_dir, stump_csv_name, default_points):
    """
    Load ALL observed 2D image points from
    <recorded_frames_dir>/<stump_csv_name> (columns: image, point, x, y),
    in file order - no splitting into pitch/stump subsets. Row order must
    line up 1:1 with WORLD_POINTS above. Falls back to the hardcoded
    IMAGE_POINTS_OBSERVED default if the CSV isn't found.
    """
    csv_path = os.path.join(recorded_frames_dir, stump_csv_name)
    if os.path.isfile(csv_path):
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row["x"]), float(row["y"])))
        pts = np.array(rows, dtype=np.float64)
        print(f"Loaded {len(pts)} observed image points from {csv_path} (no split, all points used).")
        return pts
    else:
        print(f"{csv_path} not found - using hardcoded IMAGE_POINTS_OBSERVED.")
        return default_points


def camera_center(R_: np.ndarray, t_: np.ndarray) -> np.ndarray:
    """World-space camera center C = -R^T @ t."""
    return -R_.T @ t_


def ground_distance_to_point(R_, t_, point_world: np.ndarray, up_axis: int = UP_AXIS) -> float:
    """Horizontal ('ground') distance from camera center to a world point."""
    C = camera_center(R_, t_).copy()
    P = point_world.copy()
    C[up_axis] = 0.0
    P[up_axis] = 0.0
    return float(np.linalg.norm(C - P))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    # --- load intrinsics ---


    if USE_MANUAL_PARAMETERS:

        print("Using manually supplied camera parameters.")

        K = MANUAL_K.copy()
        dist = MANUAL_DIST.copy()

        R = MANUAL_R.copy()
        tvec = MANUAL_T.copy()

        rvec, _ = cv2.Rodrigues(R)

    else:

        d = np.load(CALIB_NPZ_PATH)

        K, dist = load_intrinsics_from_npz(d)

        found = load_extrinsic_from_npz(d)

        if found is None:
            raise RuntimeError(
                f"No pitch extrinsic found inside {CALIB_NPZ_PATH}.\n"
                "Either:\n"
                "1. Save R/t (or rvec/tvec) into the npz, or\n"
                "2. Set USE_MANUAL_PARAMETERS=True."
            )

        rvec, tvec = found
        R, _ = cv2.Rodrigues(rvec)

    t = tvec.ravel()

    # --- observed image points: all rows from stump_image.csv, no split ---
    image_points_observed = load_observed_image_points(
        RECORDED_FRAMES_DIR, STUMP_CSV_NAME, IMAGE_POINTS_OBSERVED
    )

    # --- reprojection error (with distortion, via cv2.projectPoints) ---
    projected, _ = cv2.projectPoints(WORLD_POINTS, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points_observed, axis=1)
    mean_error = errors.mean()
    max_error = errors.max()

    # --- ground distance to pitch center ---
    dist_to_center = ground_distance_to_point(R, t, PITCH_CENTER_WORLD.copy())

    # --- report ---
    print("K:\n", K)
    print("dist:", dist.ravel())
    print("R:\n", R)
    print("t:", t)
    print()
    print("Per-point reprojection errors (px):", np.round(errors, 3))
    print(f"Mean reprojection error: {mean_error:.3f} px")
    print(f"Max reprojection error:  {max_error:.3f} px")
    print(f"Ground distance to pitch center: {dist_to_center:.3f} m "
          f"(expected {EXPECTED_CAMERA_TO_PITCH_CENTER_M} +/- {DISTANCE_TOLERANCE_M} m)")
    print()

    if mean_error <= MAX_MEAN_REPROJECTION_ERROR_PX and EXPECTED_CAMERA_TO_PITCH_CENTER_M - dist_to_center <= DISTANCE_TOLERANCE_M:
        print(f"PASS: mean error {mean_error:.3f}px <= {MAX_MEAN_REPROJECTION_ERROR_PX}px "
              "-> extrinsic stage OK, proceed.")
    else:
        print(f"FAIL: mean error {mean_error:.3f}px > {MAX_MEAN_REPROJECTION_ERROR_PX}px "
              "-> go back to intrinsic calibration stage.")


if __name__ == "__main__":
    main()