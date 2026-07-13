
#imports
import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO

from stumps import detect_stump_keypoints


#config
CORNERS_WEIGHTS = "weights/corners.pt"      
PITCH_WEIGHTS   = "weights/pitch_segmentation.pt"         

IMAGES_DIR = "recorded_frames"               
SAVE_DIR   = "kp_output"                  

CONF_CORNERS = 0.25
CONF_PITCH   = 0.25
PITCH_STUMP_CLASS   = 1   
CORNERS_STUMP_CLASS = 2   
USE_PITCH_MODEL = False
 
os.makedirs(SAVE_DIR, exist_ok=True)

#load models
corners_model = YOLO(CORNERS_WEIGHTS)
pitch_model = YOLO(PITCH_WEIGHTS)

#helper functions
def order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)

    center = pts.mean(axis=0)

    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    pts = pts[np.argsort(angles)]

    s = pts.sum(axis=1)
    start = np.argmin(s)

    pts = np.roll(pts, -start, axis=0)

    return pts.astype(np.int32)
    #order in tl, tr, br, bl

def contour_to_quad(cnt):

    hull = cv2.convexHull(cnt)

    peri = cv2.arcLength(hull, True)

    # Search epsilon until exactly 4 vertices remain
    for eps in np.linspace(0.002, 0.08, 80):

        approx = cv2.approxPolyDP(
            hull,
            eps * peri,
            True
        )

        if len(approx) == 4:
            return order_points(approx[:, 0, :])

    # fallback: closest polygon to 4 vertices
    best = None
    best_diff = 100

    for eps in np.linspace(0.002, 0.15, 200):

        approx = cv2.approxPolyDP(
            hull,
            eps * peri,
            True
        )

        diff = abs(len(approx) - 4)

        if diff < best_diff:
            best_diff = diff
            best = approx

    if best is None:
        return None

    if len(best) > 4:

        pts = best[:, 0, :].astype(np.float32)

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)

        return order_points(box)

    return order_points(best[:, 0, :])
    #go from mask to quad (tl, tr, br, bl)

def closest_point_on_segment(p, a, b):
    """
    Closest point on segment a-b to point p.
    p, a, b: np.array([x, y], dtype=float32)
    """
    ab = b - a
    ab_len_sq = np.dot(ab, ab)

    if ab_len_sq == 0:
        return a.copy()

    t = np.dot(p - a, ab) / ab_len_sq
    t = np.clip(t, 0.0, 1.0)

    return a + t * ab


def closest_point_on_polygon_boundary(poly, p):
    """
    Closest point on the polygon boundary (edges of poly) to point p.
    poly: Nx2 array of ordered vertices
    p: (x, y) point
    """
    poly = poly.astype(np.float32)
    p = np.asarray(p, dtype=np.float32)

    best_pt = None
    best_dist = np.inf

    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        candidate = closest_point_on_segment(p, a, b)
        dist = np.linalg.norm(p - candidate)

        if dist < best_dist:
            best_dist = dist
            best_pt = candidate

    return best_pt


def clamp_quad_to_pitch(quad, pitch_quad):
    """
    For each point in quad, if it lies outside pitch_quad,
    snap it to the closest point on pitch_quad's boundary.
    quad: Nx2 array (small quad corners)
    pitch_quad: Nx2 array (ordered pitch corners)
    """
    pitch_contour = pitch_quad.reshape(-1, 1, 2).astype(np.float32)

    clamped = quad.astype(np.float32).copy()

    for i, pt in enumerate(clamped):
        inside = cv2.pointPolygonTest(
            pitch_contour,
            (float(pt[0]), float(pt[1])),
            False
        )

        if inside < 0:  # strictly outside
            clamped[i] = closest_point_on_polygon_boundary(pitch_quad, pt)

    return clamped.astype(np.int32)

#pitch quad extraction
def get_pitch_quad(pitch_result, img_shape):
    """
    Given a single ultralytics Result (from the pitch-seg model),
    return the ordered 4-point pitch quadrilateral, or None if unavailable.

    Stump detections (class == PITCH_STUMP_CLASS) are always excluded.
    When multiple pitch masks remain, the highest-confidence one is used.
    """
    if pitch_result.masks is None or len(pitch_result.masks.data) == 0:
        return None

    classes = pitch_result.boxes.cls.cpu().numpy().astype(int)
    confs = pitch_result.boxes.conf.cpu().numpy()
    masks = pitch_result.masks.data.cpu().numpy()

    keep = classes != PITCH_STUMP_CLASS
    masks = masks[keep]
    confs = confs[keep]

    if len(masks) == 0:        # <-- guard against "all detections were stumps"
        return None

    # If multiple pitch masks are returned, take the highest-confidence one
    mask = masks[int(np.argmax(confs))]

    mask = cv2.resize(
        mask,
        (img_shape[1], img_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    mask = (mask > 0.5).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    return contour_to_quad(cnt)


##corrected
# Tolerances for deciding whether points "agree" on a line
LINE_TOL_Y = 5.0   # px, for horizontal-line clustering (compares y-coords)
LINE_TOL_X = 5.0   # px, for vertical-line clustering (compares x-coords)


def classify_corners_by_centroid(quad):
    """
    Label each of a quad's 4 corners as tl/tr/bl/br using the standard robust
    sum/diff corner-ordering trick (each role comes from an independent
    argmin/argmax, so it can never collide/leave a role unassigned the way
    a quadrant-comparison approach can on skewed quads):
      tl = point with minimum (x + y)
      br = point with maximum (x + y)
      tr = point with maximum (x - y)
      bl = point with minimum (x - y)
    quad: 4x2 array
    returns: dict {'tl':pt, 'tr':pt, 'bl':pt, 'br':pt}  (pt is a length-2 np.array, float32)
    """
    pts = quad.astype(np.float32)

    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]

    labels = {
        "tl": pts[np.argmin(s)].copy(),
        "br": pts[np.argmax(s)].copy(),
        "tr": pts[np.argmax(diff)].copy(),
        "bl": pts[np.argmin(diff)].copy(),
    }

    return labels


def assign_grid_positions(quads):
    """
    Given exactly 4 quads (list of 4x2 arrays), infer which is TL/TR/BL/BR
    based on centroid position. Returns dict {'TL':quad, 'TR':quad, 'BL':quad, 'BR':quad}
    or None if len(quads) != 4.
    """
    if len(quads) != 4:
        return None

    centroids = [q.astype(np.float32).mean(axis=0) for q in quads]

    idx_by_y = sorted(range(4), key=lambda i: centroids[i][1])
    top_idx, bottom_idx = idx_by_y[:2], idx_by_y[2:]

    top_sorted = sorted(top_idx, key=lambda i: centroids[i][0])
    bottom_sorted = sorted(bottom_idx, key=lambda i: centroids[i][0])

    tl_i, tr_i = top_sorted[0], top_sorted[1]
    bl_i, br_i = bottom_sorted[0], bottom_sorted[1]

    return {"TL": quads[tl_i], "TR": quads[tr_i], "BL": quads[bl_i], "BR": quads[br_i]}


# Each group: (axis, [(quad_name, corner_key), ...four references...])
# axis 'h' -> cluster/fit on y (line is roughly horizontal)
# axis 'v' -> cluster/fit on x (line is roughly vertical, may be slanted)
LINE_GROUPS = [
    ("h", [("TL", "tl"), ("TL", "tr"), ("TR", "tl"), ("TR", "tr")]),  # top row, top edge
    ("h", [("TL", "bl"), ("TL", "br"), ("TR", "bl"), ("TR", "br")]),  # top row, bottom edge
    ("h", [("BL", "tl"), ("BL", "tr"), ("BR", "tl"), ("BR", "tr")]),  # bottom row, top edge
    ("h", [("BL", "bl"), ("BL", "br"), ("BR", "bl"), ("BR", "br")]),  # bottom row, bottom edge
    ("v", [("TL", "tl"), ("TL", "bl"), ("BL", "tl"), ("BL", "bl")]),  # left col, left edge
    ("v", [("TL", "tr"), ("TL", "br"), ("BL", "tr"), ("BL", "br")]),  # left col, right edge
    ("v", [("TR", "tl"), ("TR", "bl"), ("BR", "tl"), ("BR", "bl")]),  # right col, left edge
    ("v", [("TR", "tr"), ("TR", "br"), ("BR", "tr"), ("BR", "br")]),  # right col, right edge
]


def find_majority_cluster(values, tol):
    """
    values: list/array of 4 scalars.
    Returns (inlier_indices, outlier_indices) for the largest tight cluster
    (points within tol of each other), or (None, None) if no cluster of size >= 3.
    """
    n = len(values)
    best_cluster = []

    for i in range(n):
        cluster = [j for j in range(n) if abs(values[j] - values[i]) <= tol]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) >= 3:
        inliers = best_cluster
        outliers = [j for j in range(n) if j not in inliers]
        return inliers, outliers

    return None, None


def correct_line_consensus(quads_labeled):
    """
    quads_labeled: dict {'TL':{'tl':pt,'tr':pt,'bl':pt,'br':pt}, 'TR':{...}, 'BL':{...}, 'BR':{...}}
    Mutates points in place (snaps outliers onto the consensus line). Returns count corrected.
    """
    n_corrected = 0

    for axis, refs in LINE_GROUPS:
        pts = [quads_labeled[qname][ckey] for qname, ckey in refs]

        if axis == "h":
            values = [p[1] for p in pts]  # y-coords
            tol = LINE_TOL_Y
        else:
            values = [p[0] for p in pts]  # x-coords
            tol = LINE_TOL_X

        inliers, outliers = find_majority_cluster(values, tol)

        if inliers is None:
            continue  # no majority agreement, leave this group untouched

        if axis == "h":
            # Horizontal line: y = const (median of inlier y-values)
            line_y = float(np.median([pts[i][1] for i in inliers]))

            for o in outliers:
                pts[o][1] = line_y
                n_corrected += 1

        else:
            # Vertical (possibly slanted) line: fit x = m*y + b from inliers
            ys = np.array([pts[i][1] for i in inliers], dtype=np.float64)
            xs = np.array([pts[i][0] for i in inliers], dtype=np.float64)

            if len(set(ys.tolist())) < 2:
                # Degenerate (all inlier y-values identical) — fall back to median x
                fit_x = float(np.median(xs))
                for o in outliers:
                    pts[o][0] = fit_x
                    n_corrected += 1
            else:
                m, b = np.polyfit(ys, xs, 1)

                for o in outliers:
                    pts[o][0] = float(m * pts[o][1] + b)
                    n_corrected += 1

    return n_corrected

def main():
    CORNER_DRAW_ORDER = ["tl", "tr", "br", "bl"]
    GRID_ORDER = ["TL", "TR", "BR", "BL"]
    DEDUPE_DIST_THRESH = 40.0  # px — center-distance duplicate test
    DEDUPE_IOU_THRESH = 0.3    # overlap fraction — catches duplicates whose centers
                                # differ by more than DEDUPE_DIST_THRESH but whose
                                # boxes still substantially overlap (e.g. the model
                                # firing twice on the same physical corner box)

    # Maps (grid_name, corner_key) -> the fixed 1-16 keypoint index the user
    # wants used everywhere instead of the "TL_tl" style name.
    CORNER_INDEX = {
        ("TL", "tl"): 1,  ("TL", "tr"): 2,  ("TL", "br"): 6,  ("TL", "bl"): 5,
        ("TR", "tl"): 3,  ("TR", "tr"): 4,  ("TR", "br"): 8,  ("TR", "bl"): 7,
        ("BR", "tl"): 11, ("BR", "tr"): 12, ("BR", "br"): 16, ("BR", "bl"): 15,
        ("BL", "tl"): 9,  ("BL", "tr"): 10, ("BL", "br"): 14, ("BL", "bl"): 13,
    }

    # Per-image CSV header: 16 rows per image (in that image's own CSV), one row
    # per keypoint index 1..16, with separate x and y columns.
    KP_HEADER = ["image", "point", "x", "y"]

    image_files = [
        f for f in sorted(os.listdir(IMAGES_DIR))
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"Found {len(image_files)} images in {IMAGES_DIR}")

    corners_results = corners_model.predict(
        source=IMAGES_DIR,
        conf=CONF_CORNERS,
        stream=True,
        verbose=False
    )

    # Only run the pitch model — and only use it to clamp corner-quad points — when
    # USE_PITCH_MODEL is True. When False, it's skipped entirely and
    # correct_line_consensus() (majority agreement on horizontal + slanted lines)
    # is the sole correction mechanism.
    pitch_by_name = {}
    if USE_PITCH_MODEL:
        pitch_results_list = list(pitch_model.predict(
            source=IMAGES_DIR,
            conf=CONF_PITCH,
            stream=True,
            verbose=False
        ))
        pitch_by_name = {
            os.path.basename(r.path): r for r in pitch_results_list
        }


    def quad_center(q):
        return q.mean(axis=0)


    def quad_area(q):
        # simple polygon area (shoelace) for a 4-point quad
        x = q[:, 0]
        y = q[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


    def quad_iou(q1, q2):
        """
        Intersection-over-union of two convex quads. Returns 0.0 if they don't
        overlap or the convex intersection routine fails (e.g. degenerate quad).
        """
        try:
            inter_area, _ = cv2.intersectConvexConvex(
                q1.astype(np.float32), q2.astype(np.float32)
            )
        except cv2.error:
            return 0.0

        if inter_area <= 0:
            return 0.0

        union = quad_area(q1) + quad_area(q2) - inter_area
        if union <= 0:
            return 0.0

        return float(inter_area / union)


    def dedupe_quads(quads, classes, confs, dist_thresh=40.0, iou_thresh=0.3):
        """
        Remove near-duplicate quads. Two quads are considered duplicates if
        EITHER their centers are within dist_thresh px of each other OR their
        polygon IoU exceeds iou_thresh (catches same-box double detections whose
        centers happen to sit further apart than dist_thresh). When two quads
        are duplicates, keep the one with the higher detection confidence; if
        confidences are unavailable/equal, fall back to keeping the larger-area
        quad (spurious detections tend to be small slivers near a real corner).
        """
        removed = set()

        for i in range(len(quads)):
            if i in removed:
                continue
            for j in range(i + 1, len(quads)):
                if j in removed:
                    continue

                d = np.linalg.norm(quad_center(quads[i]) - quad_center(quads[j]))
                iou = quad_iou(quads[i], quads[j])

                if d < dist_thresh or iou > iou_thresh:
                    conf_i = confs[i] if confs is not None else None
                    conf_j = confs[j] if confs is not None else None

                    if conf_i is not None and conf_j is not None and conf_i != conf_j:
                        drop = j if conf_i > conf_j else i
                    else:
                        drop = j if quad_area(quads[i]) >= quad_area(quads[j]) else i

                    removed.add(drop)
                    if drop == i:
                        break  # i got removed, stop comparing it further

        kept_idx = [i for i in range(len(quads)) if i not in removed]
        dedup_quads = [quads[i] for i in kept_idx]
        dedup_classes = [classes[i] for i in kept_idx]
        dedup_confs = [confs[i] for i in kept_idx] if confs is not None else None
        return dedup_quads, dedup_classes, dedup_confs


    for result in corners_results:

        img = result.orig_img.copy()
        fname = os.path.basename(result.path)

        # --- per-image 16-kp CSV setup: one row per point (1..16), x/y columns ---
        csv_name = os.path.splitext(fname)[0] + ".csv"
        csv_path = os.path.join(SAVE_DIR, csv_name)
        kp_rows = {
            idx: {"image": fname, "point": idx, "x": "", "y": ""}
            for idx in range(1, 21)
        }

        # --- get matching pitch result & pitch quad for this frame (flag-gated) ---
        pitch_quad = None
        if USE_PITCH_MODEL:
            pitch_result = pitch_by_name.get(fname)
            if pitch_result is not None:
                pitch_quad = get_pitch_quad(pitch_result, img.shape)

        quads = []
        classes = []
        confs = []

        if result.masks is not None:

            masks = result.masks.data.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            conf_scores = result.boxes.conf.cpu().numpy()

            for mask, cls, conf in zip(masks, cls_ids, conf_scores):
                if cls == CORNERS_STUMP_CLASS:
                    continue
                mask = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                mask = (mask > 0.5).astype(np.uint8) * 255

                contours, _ = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if len(contours) == 0:
                    continue

                cnt = max(contours, key=cv2.contourArea)

                quad = contour_to_quad(cnt)

                if quad is None:
                    continue

                # --- clamp small-quad corners to the pitch boundary (only if enabled) ---
                if USE_PITCH_MODEL and pitch_quad is not None:
                    quad = clamp_quad_to_pitch(quad, pitch_quad)

                quads.append(quad.astype(np.float32))
                classes.append(cls)
                confs.append(float(conf))

        n_raw = len(quads)

        # --- remove near-duplicate/spurious quads (highest confidence wins) before the grid check ---
        quads, classes, confs = dedupe_quads(
            quads, classes, confs,
            dist_thresh=DEDUPE_DIST_THRESH, iou_thresh=DEDUPE_IOU_THRESH
        )

        n_deduped = len(quads)
        trimmed_extra = False

        # --- if a spurious extra detection survives dedup, keep only the 4
        # highest-confidence quads rather than giving up on the whole frame ---
        if len(quads) > 4:
            order = sorted(range(len(quads)), key=lambda i: confs[i], reverse=True)[:4]
            quads = [quads[i] for i in order]
            classes = [classes[i] for i in order]
            confs = [confs[i] for i in order]
            trimmed_extra = True

        # --- cross-quad line-consensus correction (only when exactly 4 quads found) ---
        # This runs regardless of USE_PITCH_MODEL: it's the required fallback when the
        # pitch model isn't used, and a further refinement pass on top of pitch-clamped
        # points when it is.
        grid = assign_grid_positions(quads) if len(quads) == 4 else None

        if grid is not None:
            quads_labeled = {
                name: classify_corners_by_centroid(q) for name, q in grid.items()
            }

            correct_line_consensus(quads_labeled)

            # class id per grid position, matched back to original detection order
            # via centroid identity
            grid_names = list(grid.keys())
            grid_cls = {}
            for name in grid_names:
                gq = grid[name]
                match_idx = min(
                    range(len(quads)),
                    key=lambda i: np.linalg.norm(quads[i].mean(axis=0) - gq.mean(axis=0))
                )
                grid_cls[name] = classes[match_idx]

            for name, corners in quads_labeled.items():
                cls = grid_cls[name]
                if cls == CORNERS_STUMP_CLASS:
                    continue
                ordered_pts = [corners[k] for k in CORNER_DRAW_ORDER]

                for ckey, p in zip(CORNER_DRAW_ORDER, ordered_pts):
                    p_int = tuple(map(int, p))
                    cv2.circle(img, p_int, 6, (0, 0, 255), -1)

                    kp_idx = CORNER_INDEX[(name, ckey)]
                    kp_rows[kp_idx]["x"] = p_int[0]
                    kp_rows[kp_idx]["y"] = p_int[1]

                    cv2.putText(
                        img,
                        str(kp_idx),
                        (p_int[0] + 8, p_int[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 0, 0),
                        1
                    )

                label_pt = ordered_pts[0]
                cv2.putText(
                    img,
                    f"{name}:{cls}",
                    (int(label_pt[0]), int(label_pt[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            status = f"OK ({n_raw} raw -> {n_deduped} deduped"
            status += " -> trimmed to top-4" if trimmed_extra else ""
            status += ")"
            print(f"[{fname}] {status} — 16-kp row written")

        else:
            # Fewer than 4 distinct quads survived dedup — can't reliably infer
            # TL/TR/BR/BL grid positions, so draw as-is with raw per-quad/per-corner
            # indices, and this image's 16-kp CSV row is left blank.
            for i, (quad, cls) in enumerate(zip(quads, classes)):
                for j, p in enumerate(quad):
                    p_int = tuple(map(int, p))
                    cv2.circle(img, p_int, 6, (0, 0, 255), -1)

                    cv2.putText(
                        img,
                        str(j),
                        (p_int[0] + 8, p_int[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2
                    )

                cv2.putText(
                    img,
                    str(cls),
                    (int(quad[0][0]), int(quad[0][1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            print(
                f"[{fname}] SKIPPED — only {len(quads)} distinct quad(s) after dedup "
                f"({n_raw} raw detections); need exactly 4. CSV row left blank."
            )

        # Optional: also draw the pitch quad itself for sanity-checking
        if pitch_quad is not None:
            cv2.polylines(img, [pitch_quad.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

        keypoints = detect_stump_keypoints(img, pitch_model)
        if keypoints is not None:
        # Returned as:
        # [far_tl, far_tr, near_tr, near_tl]

            for idx, pt in enumerate(keypoints, start=17):
                kp_rows[idx]["x"] = int(pt[0])
                kp_rows[idx]["y"] = int(pt[1])

                cv2.circle(img, tuple(map(int, pt)), 6, (255, 255, 0), -1)

                cv2.putText(
                    img,
                    str(idx),
                    (int(pt[0]) + 8, int(pt[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
        out_path = os.path.join(SAVE_DIR, fname)
        cv2.imwrite(out_path, img)

        # --- write this image's own 16-kp CSV (16 rows: point, x, y) ---
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=KP_HEADER)
            writer.writeheader()
            for idx in range(1, 21):
                writer.writerow(kp_rows[idx])

    print("Done. Images + per-image 16-kp CSVs saved to", SAVE_DIR)


if __name__ == "__main__":
    main()