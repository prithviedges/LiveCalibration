import cv2
import numpy as np

def detect_stump_keypoints(image, model, stump_class=1):
    """
    Returns four points in the order:
        [far_tl, far_tr, near_tr, near_tl]

    Returns None if two stump instances are not found.
    """

    results = model(image, verbose=False)[0]

    if results.masks is None:
        return None

    masks = results.masks.data.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    stumps = []

    for mask, cls in zip(masks, classes):

        if cls != stump_class:
            continue

        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)

        # Force rectangle
        rect = cv2.minAreaRect(cnt)
        quad = cv2.boxPoints(rect).astype(np.float32)

        # Order corners
        quad = quad[np.argsort(quad[:, 1])]

        top = quad[:2]
        bottom = quad[2:]

        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        tl, tr = top
        bl, br = bottom

        centroid = quad.mean(axis=0)

        stumps.append({
            "centroid": centroid,
            "tl": tuple(np.round(tl).astype(int)),
            "tr": tuple(np.round(tr).astype(int))
        })

    if len(stumps) != 2:
        return None

    # Smaller y => farther stump
    stumps.sort(key=lambda s: s["centroid"][1])

    far = stumps[0]
    near = stumps[1]

    return [
        far["tl"],
        far["tr"],
        near["tl"],
        near["tr"],
    ]