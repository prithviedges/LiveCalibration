#imports
import os
import csv
import cv2
import numpy as np
from itertools import combinations


#---------------Helper functions-----------------#
def rotation_matrix_to_euler(R):
    """Return roll, pitch, yaw in degrees from a 3×3 rotation matrix."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll  = np.degrees(np.arctan2( R[2, 1],  R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0],  sy))
        yaw   = np.degrees(np.arctan2( R[1, 0],  R[0, 0]))
    else:
        roll  = np.degrees(np.arctan2(-R[1, 2],  R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0],  sy))
        yaw   = 0.0
    return roll, pitch, yaw


def camera_centre(R, t):
    """C = -R^T * t  (world coordinates of the optical centre)."""
    return (-R.T @ t).flatten()


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    """Mean and per-point reprojection errors (pixels)."""
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(img_pts - projected, axis=1)
    return errors, errors.mean()


def load_camera_intrinsics(calib_npz_path, default_K, default_dist_coeffs):
    """
    Load K, dist_coeffs from calib_params.npz (supports keys 'mtx'/'K', 'dist'/'dist_coeffs')
    if the file exists. Otherwise fall back to the hardcoded default K / dist_coeffs.
    """
    if os.path.isfile(calib_npz_path):
        data = np.load(calib_npz_path)
        if 'mtx' in data.files:
            K = np.array(data['mtx'], dtype=np.float64)
        elif 'K' in data.files:
            K = np.array(data['K'], dtype=np.float64)
        else:
            K = default_K

        if 'dist' in data.files:
            dist_coeffs = np.array(data['dist'], dtype=np.float64)
        elif 'dist_coeffs' in data.files:
            dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
        else:
            dist_coeffs = default_dist_coeffs
        print(f"[intrinsics] Loaded K, dist_coeffs from {calib_npz_path}")
    else:
        K = default_K
        dist_coeffs = default_dist_coeffs
        print(f"[intrinsics] {calib_npz_path} not found — using hardcoded K, dist_coeffs")
    return K, dist_coeffs


def load_keypoints_from_npz(calib_npz_path):
    """
    Try loading pitch_image and wicket_image (keypoints) from the .npz file if it exists.
    Checks for separate keys:
      - 'pitch_image' / 'pitch_points' / 'pitch_image_points' (16, 2)
      - 'wicket_image' / 'wicket_points' / 'wicket_image_points' (4, 2)
    Or a combined key:
      - 'keypoints' / 'image_points' / 'img_pts' (20, 2) -> first 16 are pitch, last 4 are wicket
    """
    if os.path.isfile(calib_npz_path):
        try:
            data = np.load(calib_npz_path)
            pitch_image = None
            wicket_image = None

            # Check separate keys
            for p_key in ["pitch_image", "pitch_points", "pitch_image_points"]:
                if p_key in data.files:
                    pitch_image = np.array(data[p_key], dtype=np.float64).reshape(-1, 2)
                    break

            for w_key in ["wicket_image", "wicket_points", "wicket_image_points"]:
                if w_key in data.files:
                    wicket_image = np.array(data[w_key], dtype=np.float64).reshape(-1, 2)
                    break

            if pitch_image is not None and wicket_image is not None:
                if pitch_image.shape[0] == 16 and wicket_image.shape[0] == 4:
                    print(f"[keypoints] Loaded pitch_image and wicket_image from {calib_npz_path}")
                    return pitch_image, wicket_image

            # Check combined keys
            for c_key in ["keypoints", "image_points", "img_pts"]:
                if c_key in data.files:
                    pts = np.array(data[c_key], dtype=np.float64).reshape(-1, 2)
                    if pts.shape[0] == 20:
                        pitch_image = pts[:16]
                        wicket_image = pts[16:20]
                        print(f"[keypoints] Loaded combined '{c_key}' from {calib_npz_path} and split into pitch/wicket")
                        return pitch_image, wicket_image
        except Exception as e:
            print(f"[keypoints] Error loading keypoints from {calib_npz_path}: {e}")
    return None, None


def load_image_points(recorded_frames_dir, stump_csv_name, default_pitch_image, default_wicket_image):
    """
    Load pitch_image / wicket_image from CSV (checks recorded_frames/, kp_output/, or direct path).
    CSV has columns: image, point, x, y.
    The first 16 rows become pitch_image, the next 4 rows become wicket_image (stump tops).
    Falls back to hardcoded defaults if the CSV isn't found.
    """
    candidate_paths = [
        os.path.join(recorded_frames_dir, stump_csv_name),
        os.path.join("kp_output", stump_csv_name),
        stump_csv_name,
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.isfile(p):
            csv_path = p
            break

    if csv_path is not None:
        rows = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((float(row["x"]), float(row["y"])))

        pitch_image = np.array(rows[:16], dtype=np.float64)
        wicket_image = np.array(rows[16:20], dtype=np.float64)
        print(f"[image points] Loaded pitch_image (16 pts) and wicket_image (4 pts) from {csv_path}")
    else:
        pitch_image = default_pitch_image
        wicket_image = default_wicket_image
        print(f"[image points] CSV not found in candidate paths -- using hardcoded pitch_image, wicket_image")
    return pitch_image, wicket_image


def save_camera_params(K, dist_coeffs, rvec, tvec, R, t, C, pitch_image, wicket_image, out_path):
    """Store intrinsics (K, dist_coeffs), extrinsics (rvec, tvec, R, t, C), and keypoints to a .npz file."""
    np.savez(
        out_path,
        K=K,
        dist_coeffs=dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        R=R,
        t=t,
        C=C,
        pitch_image=pitch_image,
        wicket_image=wicket_image,
        keypoints=np.vstack([pitch_image, wicket_image]),
    )
    print(f"\nSaved intrinsics + extrinsics + keypoints to {out_path}")


#---------------Main-----------------#
def main(
    image_path=r"recorded_frames\stump_image.png",
    calib_npz_path="calib_params.npz",
    recorded_frames_dir="recorded_frames",
    stump_csv_name="kp_output/stump_image.csv",
    extrinsics_out_path="calib_params.npz",
    show_display=True,
):
    #---------------Camera Intrinsics-----------------#
    DEFAULT_K = np.array([[2.45171310e+04, 0.00000000e+00, 1.10824991e+03],
 [0.00000000e+00, 2.42281298e+04, 2.68066558e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

    , dtype=np.float64)

    DEFAULT_dist_coeffs = np.array([[ 1.46278449e+01, -4.23045610e+03 ,-6.01742613e-02,  5.64750980e-02,
   7.91119438e+04]], dtype=np.float64)

    K, dist_coeffs = load_camera_intrinsics(calib_npz_path, DEFAULT_K, DEFAULT_dist_coeffs)

    #---------------World Coordinates-----------------#

    # --- 16 Pitch points ---
    # Bowling / popping crease corners and key pitch markings

    pitch_world = np.array([
        # Batting crease (far end)
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
        # -------------------------------
        # [ -18.3,  0.0, 176.8],   # P17
        # [ -18.3, 0.0, 189.0],    # P18
    ], dtype=np.float64)

    # --- 4 Wicket points (top of stumps) ---
    # Standard wicket: 3 stumps, 22.86 cm (9 in) apart, 71.12 cm tall
    wicket_world = np.array([
        [-0.1143, 0.7112, -1.22],   # Q1
        [ 0.1143, 0.7112, -1.22],   # Q2
        [-0.1143, 0.7112,  18.9],   # Q3
        [ 0.1143, 0.7112,  18.9],   # Q4
    ], dtype=np.float64)

    #---------------Image Coordinates-----------------#
    DEFAULT_pitch_image = np.array([
        (838, 433),   # 1
    (917, 433),   # 2
    (1263, 433),  # 3
    (1342, 433),  # 4
    (838, 456),   # 5
    (917, 453),   # 6
    (1263, 457),  # 7
    (1352, 457),  # 8
    (753, 913),   # 9
    (861, 913),   # 10
    (1348, 918),  # 11
    (1471, 918),  # 12
    (743, 967),   # 13
    (854, 961),   # 14
    (1359, 966),  # 15
    (1479, 972),  # 16

    ], dtype=np.float64)

    DEFAULT_wicket_image = np.array([
            (1071, 296),  # 17
    (1118, 296),  # 18
    (1077, 760),  # 19
    (1148, 760),  # 20
    ], dtype=np.float64)

    pitch_image, wicket_image = load_keypoints_from_npz(calib_npz_path)
    if pitch_image is None or wicket_image is None:
        pitch_image, wicket_image = load_image_points(
            recorded_frames_dir, stump_csv_name, DEFAULT_pitch_image, DEFAULT_wicket_image
        )

    #---------------Stack all points for combinations-----------------#
    all_world = np.vstack([pitch_world, wicket_world])   # (20, 3)
    all_image = np.vstack([pitch_image, wicket_image])   # (20, 2)

    labels = [f"P{i+1}"  for i in range(16)] + \
             [f"W{i+1}"  for i in range(4)]

    #---------------PnP Combinations-----------------#
    indices = list(range(len(all_world)))
    # combos  = list(combinations(indices, 6))

    pitch_indices1  = list(range(8))      # 0–7
    pitch_indices2  = list(range(8, 16))      # 8–15
    wicket_indices1 = list(range(16, 18))  # 16–17
    wicket_indices2 = list(range(18, 20))  # 18–19

    pitch_combos1  = list(combinations(pitch_indices1, 2))
    pitch_combos2  = list(combinations(pitch_indices2, 2))
    wicket_combos1 = list(combinations(wicket_indices1, 1))
    wicket_combos2 = list(combinations(wicket_indices2, 1))
    combos = [
        p_combo1 + p_combo2 + w_combo1 + w_combo2
        for p_combo1 in pitch_combos1
        for p_combo2 in pitch_combos2
        for w_combo1 in wicket_combos1
        for w_combo2 in wicket_combos2
    ]
    total   = len(combos)

    print("=" * 80)
    print(f"  PnP Solver  --  {total} combinations of 6 points from {len(all_world)} total")
    print("=" * 80)

    #--------------Calculations-----------------#
    results = []  # store (mean_err, combo_idx, rvec, tvec) for summary

    for combo_num, combo in enumerate(combos, start=1):
        obj_pts = all_world[list(combo)].reshape(-1, 1, 3)
        img_pts = all_image[list(combo)].reshape(-1, 1, 2)
        combo_labels = [labels[i] for i in combo]

        # --- Solve PnP (EPNP is robust for 6+ points) ---

        success, rvec, tvec = cv2.solvePnP(
           obj_pts,img_pts, K, dist_coeffs,
            flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            print(f"\nCombo #{combo_num:5d}  {combo_labels}  ->  FAILED")
            continue

        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist_coeffs, rvec, tvec)

        # Derived quantities
        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.flatten()
        C    = camera_centre(R, t)
        roll, pitch, yaw = rotation_matrix_to_euler(R)

        # Reprojection error on the 6 points used
        per_pt_err, mean_err = reprojection_error(
            obj_pts, img_pts.reshape(-1, 2), rvec, tvec, K, dist_coeffs
        )

        results.append((mean_err, combo_num, combo_labels, rvec, tvec, R, t, C))

        # Pretty print
        sep = "-" * 80
        print(f"\n{sep}")
        print(f"  Combo #{combo_num:5d} / {total}   Points: {combo_labels}")
        print(sep)

        print("\n  Rotation Matrix R:")
        for row in R:
            print(f"    [{row[0]:+10.6f}  {row[1]:+10.6f}  {row[2]:+10.6f}]")

        print(f"\n  Rotation Vector (Rodrigues): "
              f"[{rvec[0,0]:+.6f}, {rvec[1,0]:+.6f}, {rvec[2,0]:+.6f}]")

        print(f"  Euler Angles  ->  Roll: {roll:+8.3f} deg   "
              f"Pitch: {pitch:+8.3f} deg   Yaw: {yaw:+8.3f} deg")

        print(f"\n  Translation Vector t:  "
              f"[{t[0]:+10.6f},  {t[1]:+10.6f},  {t[2]:+10.6f}]  m")

        print(f"\n  Camera Centre C (world):  "
              f"[{C[0]:+10.4f},  {C[1]:+10.4f},  {C[2]:+10.4f}]  m")

        print(f"\n  Reprojection Errors (px):")
        for lbl, err in zip(combo_labels, per_pt_err):
            bar = "#" * min(int(err * 4), 40)
            print(f"    {lbl:>3s}  {err:6.3f} px  {bar}")
        print(f"    -- Mean: {mean_err:.4f} px")

    #--------------Best-3-----------------#
    print("\n" + "=" * 80)
    print("  SUMMARY  --  Top 3 combinations by lowest mean reprojection error")
    print("=" * 80)
    results.sort(key=lambda x: x[0])  # sort ascending by mean error
    for rank, (mean_err, combo_num, combo_labels, rvec, tvec, R, t, C) in \
            enumerate(results[:3], start=1):
        print(f"\n  Rank #{rank:2d}  Combo #{combo_num:5d}  Points: {combo_labels}")
        print(f"           Mean reprojection error : {mean_err:.4f} px")
        t_flat = tvec.flatten()
        print(f"           t = [{t_flat[0]:+.4f}, {t_flat[1]:+.4f}, {t_flat[2]:+.4f}]")
        print(f"           C = [{C[0]:+.4f}, {C[1]:+.4f}, {C[2]:+.4f}]")
        print(f"           R = [")
        print(f"             [{R[0,0]:+.4f}, {R[0,1]:+.4f}, {R[0,2]:+.4f}], ")
        print(f"             [{R[1,0]:+.4f}, {R[1,1]:+.4f}, {R[1,2]:+.4f}], ")
        print(f"             [{R[2,0]:+.4f}, {R[2,1]:+.4f}, {R[2,2]:+.4f}]")
        print(f"             ]")

    # ================= VISUALIZE BEST SOLUTION =================

    img = cv2.imread(image_path)
    #img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if img is None:
        raise FileNotFoundError(f"Could not load {image_path}")

    # Best pose
    mean_err, combo_num, combo_labels, rvec, tvec, R, t, C = results[0]

    # Refine best pose over ALL 20 points
    obj_all = all_world.reshape(-1, 1, 3)
    img_all = all_image.reshape(-1, 1, 2)
    rvec, tvec = cv2.solvePnPRefineLM(obj_all, img_all, K, dist_coeffs, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    C = camera_centre(R, t)

    print(f"\nVisualizing best solution (refined on all 20 points):")
    print(f"Combo #{combo_num}")
    print(f"Initial mean error on selected 6 points: {mean_err:.3f}px")

    # Project ALL 20 world points
    projected, _ = cv2.projectPoints(
        all_world,
        rvec,
        tvec,
        K,
        dist_coeffs
    )

    projected = projected.reshape(-1, 2)

    # Errors for all 20 correspondences
    all_errors = np.linalg.norm(
        all_image - projected,
        axis=1
    )

    print("\nErrors on ALL points:")
    for lbl, err in zip(labels, all_errors):
        print(f"{lbl:>3s}: {err:7.3f} px")

    print(f"\nMean error on ALL points = {all_errors.mean():.3f} px")

    # --------------------------------------------------
    # Draw measured points (GREEN)
    # Draw projected points (RED)
    # Draw error vectors (BLUE)
    # --------------------------------------------------

    for lbl, measured, proj, err in zip(
            labels,
            all_image,
            projected,
            all_errors):

        mx, my = np.round(measured).astype(int)
        px, py = np.round(proj).astype(int)

        # Measured point
        cv2.circle(
            img,
            (mx, my),
            10,
            (0, 255, 0),
            -1
        )

        # Projected point
        cv2.circle(
            img,
            (px, py),
            10,
            (0, 0, 255),
            -1
        )

        # Error vector
        cv2.line(
            img,
            (mx, my),
            (px, py),
            (255, 0, 0),
            2
        )

        # Label measured point
        cv2.putText(
            img,
            f"{lbl} ({err:.1f})",
            (mx + 10, my - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------
    # Save full-resolution output
    # --------------------------------------------------

    cv2.imwrite("reprojection_overlay.png", img)

    print("\nSaved: reprojection_overlay.png")

    # --------------------------------------------------
    # Display scaled image
    # --------------------------------------------------

    h, w = img.shape[:2]

    scale = min(
        1600 / w,
        900 / h
    )

    display = cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    if show_display:
        cv2.imshow("Reprojection Overlay", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for lbl, meas, proj in zip(labels, all_image, projected):
        dx = proj[0] - meas[0]
        dy = proj[1] - meas[1]

        print(lbl, f"dx={dx:+.3f}", f"dy={dy:+.3f}")

    #---------------Store intrinsics + extrinsics + keypoints-----------------#
    save_camera_params(K, dist_coeffs, rvec, tvec, R, t, C, pitch_image, wicket_image, extrinsics_out_path)

    print("\nDone.")

    return {
        "K": K,
        "dist_coeffs": dist_coeffs,
        "rvec": rvec,
        "tvec": tvec,
        "R": R,
        "t": t,
        "C": C,
        "mean_err": mean_err,
        "combo_labels": combo_labels,
        "results": results,
    }


if __name__ == "__main__":
    main()