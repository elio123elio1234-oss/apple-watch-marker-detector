#!/usr/bin/env python3
"""
Marker Generator v4 — SplitScreen + QuadDot
=============================================

v4 DEFAULT: "split" mode — fills the ENTIRE watch screen with color.
    Left half = CYAN, Right half = GREEN.

WHY: At steep angles (V5/V6 ECG position, ≥60° tilt), individual dots
become 2-3 pixels each and are impossible to detect reliably. A full-
screen color rectangle gives 100-500× more pixel area to work with.

Modes:
    split    (default)  Full-screen: left=cyan, right=green
    quaddot  (legacy)   4 colored dots on black OLED background (v3)

Usage:
    python generate_marker.py --watch-model 45mm
    python generate_marker.py --watch-model 45mm --brightness 50
    python generate_marker.py --mode quaddot --watch-model 45mm
"""

import cv2
import numpy as np
import argparse
import os

WATCH_SIZES = {
    "38mm": {"px": (272, 340), "mm": (26.5, 33.3)},
    "40mm": {"px": (324, 394), "mm": (30.7, 37.3)},
    "41mm": {"px": (352, 430), "mm": (33.5, 41.0)},
    "42mm": {"px": (312, 390), "mm": (28.7, 36.1)},
    "44mm": {"px": (368, 448), "mm": (34.5, 42.0)},
    "45mm": {"px": (396, 484), "mm": (37.6, 46.0)},
    "49mm": {"px": (410, 502), "mm": (39.0, 48.0)},
}


# ================================================================== #
#                   SplitScreen Marker (v4 default)                   #
# ================================================================== #

def generate_split_marker(
    size=400,
    watch_model=None,
    output_path="marker.png",
    brightness=60,
    show_preview=True,
):
    """
    Generate a SplitScreen marker: left half cyan, right half green.

    Fills the entire screen with color, maximizing the detectable area
    at any viewing angle. The cyan/green boundary gives orientation.
    """
    if watch_model and watch_model in WATCH_SIZES:
        w, h = WATCH_SIZES[watch_model]["px"]
        physical = WATCH_SIZES[watch_model]["mm"]
    else:
        w = h = size
        physical = None

    img = np.zeros((h, w, 3), dtype=np.uint8)
    scale = brightness / 100.0
    val = int(255 * scale)

    # BGR format: (B, G, R)
    cyan_bgr = (val, val, 0)    # cyan  = high B + high G
    green_bgr = (0, val, 0)     # green = high G only

    mid = w // 2
    img[:, :mid] = cyan_bgr
    img[:, mid:] = green_bgr

    cv2.imwrite(output_path, img)

    print("=" * 60)
    print("  SplitScreen Marker v4 Generated!")
    print("=" * 60)
    print(f"  Output:       {os.path.abspath(output_path)}")
    print(f"  Image size:   {w} x {h} px")
    print(f"  Brightness:   {brightness}%")
    print(f"  Left half:    CYAN  BGR={cyan_bgr}")
    print(f"  Right half:   GREEN BGR={green_bgr}")

    if physical:
        pw, ph = physical
        print(f"\n  Watch model:  {watch_model}")
        print(f"  Screen size:  {pw:.1f} x {ph:.1f} mm")
        print(f"\n  Detector command:")
        print(f"    python detect_marker.py --watch-model {watch_model}")
    else:
        print(f"\n  Measure screen width & height in mm, then:")
        print(f"    python detect_marker.py --screen-width W --screen-height H")

    print(f"\n  Watch model screen sizes (mm):")
    for model, specs in WATCH_SIZES.items():
        pw, ph = specs["mm"]
        print(f"    {model:>5s}: {pw:.1f} x {ph:.1f}")
    print("=" * 60)

    if show_preview:
        preview = cv2.resize(img, (400, int(400 * h / w)),
                             interpolation=cv2.INTER_AREA)
        ph, pw = preview.shape[:2]
        cv2.putText(preview, "CYAN", (pw // 4 - 30, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(preview, "GREEN", (3 * pw // 4 - 40, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow("SplitScreen v4 (press any key)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ================================================================== #
#                   QuadDot Marker (legacy v3)                        #
# ================================================================== #

BASE_COLORS = {
    "green":   (0, 255, 0),
    "cyan":    (255, 255, 0),
    "magenta": (255, 0, 255),
    "yellow":  (0, 255, 255),
}


def generate_quaddot_marker(
    size=400,
    watch_model=None,
    output_path="marker.png",
    large_ratio=1.4,
    margin_pct=0.22,
    color_name="green",
    brightness=60,
    show_preview=True,
):
    """Legacy v3: Generate 4-dot QuadDot marker."""
    base_color = BASE_COLORS.get(color_name, BASE_COLORS["green"])
    scale = brightness / 100.0
    circle_color = tuple(int(c * scale) for c in base_color)

    if watch_model and watch_model in WATCH_SIZES:
        w, h = WATCH_SIZES[watch_model]["px"]
    else:
        w = h = size

    img = np.zeros((h, w, 3), dtype=np.uint8)
    base = min(w, h)
    margin = int(base * margin_pct)
    cx, cy = w // 2, h // 2
    half_side = (base - 2 * margin) // 2

    small_r = int(half_side * 0.32)
    large_r = int(small_r * large_ratio)

    min_edge = max(large_r + 5, small_r + 5)
    if cx - half_side < min_edge:
        half_side = cx - min_edge
    if cy - half_side < min_edge:
        half_side = cy - min_edge

    centers = {
        "TL": (cx - half_side, cy - half_side),
        "TR": (cx + half_side, cy - half_side),
        "BR": (cx + half_side, cy + half_side),
        "BL": (cx - half_side, cy + half_side),
    }
    radii = {"TL": large_r, "TR": small_r, "BR": small_r, "BL": small_r}

    for name in ["TL", "TR", "BR", "BL"]:
        cv2.circle(img, centers[name], radii[name], circle_color, -1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print("=" * 60)
    print("  QuadDot Marker (legacy v3) Generated!")
    print("=" * 60)
    print(f"  Output: {os.path.abspath(output_path)}")
    print(f"  Color:  {color_name.upper()} at {brightness}% brightness")
    print("=" * 60)

    if show_preview:
        preview = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow("QuadDot Marker (press any key)", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


# ================================================================== #
#                   HTML Display                                      #
# ================================================================== #

def generate_html_display(output_path="marker.html", marker_image="marker.png"):
    """Generate a full-screen HTML page for displaying the marker."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Marker Display</title>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            overflow: hidden;
        }}
        img {{
            max-width: 100vw;
            max-height: 100vh;
            object-fit: contain;
        }}
    </style>
</head>
<body>
    <img src="{marker_image}" alt="Marker">
    <script>
        if (navigator.wakeLock) {{ navigator.wakeLock.request('screen').catch(e => {{}}); }}
        document.body.addEventListener('click', () => {{
            document.documentElement.requestFullscreen().catch(e => {{}});
        }});
    </script>
</body>
</html>"""
    with open(output_path, "w") as f:
        f.write(html)
    print(f"  HTML: {os.path.abspath(output_path)}")


# ================================================================== #
#                   CLI                                                #
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate marker v4")
    parser.add_argument("--mode", type=str, default="split",
                        choices=["split", "quaddot"],
                        help="split=full-screen (default), quaddot=4 dots (legacy)")
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--watch-model", type=str, default=None,
                        choices=list(WATCH_SIZES.keys()))
    parser.add_argument("--brightness", type=int, default=60,
                        help="Brightness 0-100 (default: 60)")
    parser.add_argument("--color", type=str, default="green",
                        choices=list(BASE_COLORS.keys()),
                        help="Dot color (quaddot mode only)")
    parser.add_argument("--output", type=str, default="marker.png")
    parser.add_argument("--html", action="store_true")
    args = parser.parse_args()

    if args.mode == "split":
        generate_split_marker(
            size=args.size, watch_model=args.watch_model,
            output_path=args.output, brightness=args.brightness,
        )
    else:
        generate_quaddot_marker(
            size=args.size, watch_model=args.watch_model,
            output_path=args.output, color_name=args.color,
            brightness=args.brightness,
        )

    if args.html:
        generate_html_display(marker_image=args.output)
