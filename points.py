import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================

image_path = r"C:\Users\prith\Project\pitch_correction\frames\frame1.jpg"


# ============================================================
# GLOBALS
# ============================================================

clicked_points = []

zoom_scale = 1.0
min_zoom = 0.1
max_zoom = 20.0
zoom_step = 1.1

pan_x = 0
pan_y = 0

dragging = False
last_mouse_pos = None

mouse_x = 0
mouse_y = 0

# ============================================================
# LOAD IMAGE
# ============================================================

original_img = cv2.imread(image_path)
#original_img = cv2.rotate(original_img, cv2.ROTATE_90_COUNTERCLOCKWISE)

if original_img is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

h, w = original_img.shape[:2]

window_name = "Image Annotator"
magnifier_name = "Magnifier"

# ============================================================
# COORDINATE CONVERSIONS
# ============================================================

def image_to_screen(ix, iy):
    sx = int(ix * zoom_scale + pan_x)
    sy = int(iy * zoom_scale + pan_y)
    return sx, sy


def screen_to_image(sx, sy):
    ix = int(round((sx - pan_x) / zoom_scale))
    iy = int(round((sy - pan_y) / zoom_scale))
    return ix, iy


# ============================================================
# REDRAW MAIN CANVAS
# ============================================================

def redraw():

    interpolation = (
        cv2.INTER_NEAREST
        if zoom_scale > 3
        else cv2.INTER_LINEAR
    )

    canvas = cv2.resize(
        original_img,
        None,
        fx=zoom_scale,
        fy=zoom_scale,
        interpolation=interpolation
    )

    # ---------------------------
    # Draw clicked points
    # ---------------------------

    for idx, (x, y) in enumerate(clicked_points, start=1):

        sx, sy = image_to_screen(x, y)

        draw_x = sx - pan_x
        draw_y = sy - pan_y

        cv2.circle(
            canvas,
            (draw_x, draw_y),
            5,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            canvas,
            str(idx),
            (draw_x + 10, draw_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # ---------------------------
    # Crosshair
    # ---------------------------

    cv2.line(
        canvas,
        (mouse_x - 20, mouse_y),
        (mouse_x + 20, mouse_y),
        (0, 255, 255),
        1
    )

    cv2.line(
        canvas,
        (mouse_x, mouse_y - 20),
        (mouse_x, mouse_y + 20),
        (0, 255, 255),
        1
    )

    # ---------------------------
    # Cursor pixel coordinates
    # ---------------------------

    ix, iy = screen_to_image(mouse_x, mouse_y)

    cv2.putText(
        canvas,
        f"Pixel: ({ix}, {iy})",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"Zoom: {zoom_scale:.2f}x",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        f"Points: {len(clicked_points)}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return canvas


# ============================================================
# MAGNIFIER
# ============================================================

def draw_magnifier():

    ix, iy = screen_to_image(mouse_x, mouse_y)

    patch_radius = 25

    x1 = max(0, ix - patch_radius)
    y1 = max(0, iy - patch_radius)

    x2 = min(w, ix + patch_radius)
    y2 = min(h, iy + patch_radius)

    patch = original_img[y1:y2, x1:x2]

    if patch.size == 0:
        return

    magnified = cv2.resize(
        patch,
        None,
        fx=12,
        fy=12,
        interpolation=cv2.INTER_NEAREST
    )

    mh, mw = magnified.shape[:2]

    # Center crosshair

    cv2.line(
        magnified,
        (mw // 2, 0),
        (mw // 2, mh),
        (0, 255, 0),
        1
    )

    cv2.line(
        magnified,
        (0, mh // 2),
        (mw, mh // 2),
        (0, 255, 0),
        1
    )

    # Grid

    step = 12

    # Center crosshair

    cx = mw // 2
    cy = mh // 2

    cv2.line(
        magnified,
        (cx, 0),
        (cx, mh),
        (0, 0, 255),
        2
    )

    cv2.line(
        magnified,
        (0, cy),
        (mw, cy),
        (0, 0, 255),
        2
    )

    # Center box

    cv2.rectangle(
        magnified,
        (cx - 6, cy - 6),
        (cx + 6, cy + 6),
        (0, 255, 0),
        1
    )

    cv2.putText(
        magnified,
        f"({ix}, {iy})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow(magnifier_name, magnified)


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):

    global clicked_points
    global zoom_scale
    global pan_x, pan_y
    global dragging
    global last_mouse_pos
    global mouse_x, mouse_y

    mouse_x = x
    mouse_y = y

    # --------------------------------
    # LEFT CLICK
    # --------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        ix, iy = screen_to_image(x, y)

        if 0 <= ix < w and 0 <= iy < h:

            clicked_points.append((ix, iy))

            print(
                f"Added {len(clicked_points)} "
                f"-> Pixel: ({ix}, {iy})"
            )

    # --------------------------------
    # RIGHT CLICK
    # --------------------------------

    elif event == cv2.EVENT_RBUTTONDOWN:

        if clicked_points:
            removed = clicked_points.pop()
            print(f"Removed {removed}")

    # --------------------------------
    # PAN
    # --------------------------------

    elif event == cv2.EVENT_MBUTTONDOWN:

        dragging = True
        last_mouse_pos = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:

        dx = x - last_mouse_pos[0]
        dy = y - last_mouse_pos[1]

        pan_x += dx
        pan_y += dy

        last_mouse_pos = (x, y)

    elif event == cv2.EVENT_MBUTTONUP:

        dragging = False

    # --------------------------------
    # ZOOM
    # --------------------------------

    elif event == cv2.EVENT_MOUSEWHEEL:

        old_zoom = zoom_scale

        ix, iy = screen_to_image(x, y)

        if flags > 0:
            zoom_scale *= zoom_step
        else:
            zoom_scale /= zoom_step

        zoom_scale = max(
            min_zoom,
            min(max_zoom, zoom_scale)
        )

        pan_x = int(x - ix * zoom_scale)
        pan_y = int(y - iy * zoom_scale)


# ============================================================
# WINDOWS
# ============================================================

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(magnifier_name, cv2.WINDOW_NORMAL)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)

# ============================================================
# HELP
# ============================================================

print("\nControls")
print("------------------------------------------------")
print("Left Click        -> Add point")
print("Right Click       -> Remove last point")
print("Middle Drag       -> Pan")
print("Mouse Wheel       -> Zoom")
print("W A S D           -> Move last point")
print("Backspace         -> Undo")
print("S                 -> Save")
print("Q                 -> Quit")
print("------------------------------------------------")

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    canvas = redraw()

    cv2.imshow(window_name, canvas)

    draw_magnifier()

    key = cv2.waitKey(20) & 0xFF

    # --------------------------------
    # NUDGE LAST POINT
    # --------------------------------

    if clicked_points:

        x, y = clicked_points[-1]

        if key == ord('w'):
            clicked_points[-1] = (x, y - 1)

        elif key == ord('s'):
            clicked_points[-1] = (x, y + 1)

        elif key == ord('a'):
            clicked_points[-1] = (x - 1, y)

        elif key == ord('d'):
            clicked_points[-1] = (x + 1, y)

    # --------------------------------
    # SAVE
    # --------------------------------

    if key == ord('p'):

        with open("clicked_points.txt", "w") as f:

            for idx, (x, y) in enumerate(
                clicked_points,
                start=1
            ):
                f.write(
                    f"{idx}: ({x}, {y})\n"
                )

        print("Saved to clicked_points.txt")

    # --------------------------------
    # UNDO
    # --------------------------------

    elif key == 8:

        if clicked_points:
            removed = clicked_points.pop()
            print(f"Undo -> {removed}")

    # --------------------------------
    # QUIT
    # --------------------------------

    elif key == ord('q'):
        break

cv2.destroyAllWindows()

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\nFinal Ordered Pixel Locations:")
print("[")

for pt in clicked_points:
    print(f"    {pt},")

print("]")