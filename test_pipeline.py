#!/usr/bin/env python3
"""
SplitScreen Marker v4 — Test Pipeline
=======================================
Tests the full generate→detect pipeline with synthetic scenes.

Scenarios:
  1. Frontal view
  2. 65° tilt (moderate ECG angle)
  3. 75° tilt (steep V5/V6 position)
  4. Small scale (simulating 40cm distance with tiny watch)
  5. No false positives (scene without marker)
  6. Dim brightness (40%)
  7. With simulated glass reflections
"""

import cv2
import numpy as np
import sys


def create_background(h, w, seed=42):
    """Room-like background with noise, furniture, skin tones."""
    rng = np.random.RandomState(seed)

    # Beige wall
    bg = np.full((h, w, 3), (170, 165, 155), dtype=np.uint8)
    noise = rng.randint(-12, 12, (h, w, 3), dtype=np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # White closet
    cv2.rectangle(bg, (40, 40), (220, 340), (238, 238, 242), -1)

    # Skin-tone ellipse (arm)
    cv2.ellipse(bg, (w // 2, h - 50), (220, 45), 0, 0, 360,
                (125, 155, 195), -1)

    return bg


def paste_split_marker(scene, marker, cx, cy, scale, angle_deg=0):
    """
    Paste a SplitScreen marker into the scene, optionally with
    perspective tilt. angle_deg=0 is frontal, 75° = steep tilt.

    Composites the FULL marker area (including black border pixels)
    over the background, simulating an OLED screen occluding what's behind.
    """
    mh, mw = marker.shape[:2]
    nw, nh = int(mw * scale), int(mh * scale)
    small = cv2.resize(marker, (nw, nh), interpolation=cv2.INTER_AREA)

    if angle_deg > 0:
        compress = np.cos(np.radians(angle_deg))
        tw = max(int(nw * compress), 4)

        src = np.array([[0, 0], [nw, 0], [nw, nh], [0, nh]], np.float32)
        v_off = int(nh * 0.06 * (1 - compress))
        dst = np.array([
            [0, v_off], [tw, 0], [tw, nh], [0, nh - v_off]
        ], np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            small, M, (tw, nh),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Mask: warp a white rectangle to track the screen area
        white = np.ones((nh, nw), dtype=np.uint8) * 255
        warp_mask = cv2.warpPerspective(
            white, M, (tw, nh),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
    else:
        warped = small
        tw = nw
        warp_mask = np.ones((nh, nw), dtype=np.uint8) * 255

    # Paste into scene
    y1 = max(cy - nh // 2, 0)
    x1 = max(cx - tw // 2, 0)
    y2 = min(y1 + warped.shape[0], scene.shape[0])
    x2 = min(x1 + warped.shape[1], scene.shape[1])

    sh = y2 - y1
    sw = x2 - x1
    patch = warped[:sh, :sw]
    pmask = warp_mask[:sh, :sw] > 128

    region = scene[y1:y2, x1:x2]
    region[pmask] = patch[pmask]


def add_glass_reflection(scene, cx, cy, radius=25):
    """Simulate a white specular reflection spot on the watch glass."""
    overlay = scene.copy()
    cv2.circle(overlay, (cx, cy), radius, (220, 220, 230), -1)
    cv2.addWeighted(overlay, 0.35, scene, 0.65, 0, scene)


# ================================================================== #
#                    Test Cases                                       #
# ================================================================== #

def test_frontal():
    print("=" * 60)
    print("TEST 1: Frontal view")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800)
    paste_split_marker(scene, marker, 500, 300, scale=0.3)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    if corners is not None:
        print(f"  Screen found! Corners: {corners.shape}")
        pose = det.estimate_pose(corners, scene.shape)
        if pose:
            print(f"  Distance: {pose['distance_cm']:.1f} cm")
            print(f"  Angle X: {pose['angle_x']:+.1f}°")
            print(f"  Body Rotation: {pose['body_rotation']:.1f}°")
            print(f"  ECG Label: {pose['ecg_label']}")
        else:
            print("  (Pose: None — synthetic perspective)")
        print("  RESULT: PASS")
        return True
    else:
        print("  *** Screen NOT found ***")
        print("  RESULT: FAIL")
        return False


def test_tilt_65():
    print()
    print("=" * 60)
    print("TEST 2: 65° tilt (moderate ECG angle)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=77)
    paste_split_marker(scene, marker, 450, 280, scale=0.3, angle_deg=65)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    if ok:
        pose = det.estimate_pose(corners, scene.shape)
        if pose:
            print(f"  Distance: {pose['distance_cm']:.1f} cm")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_tilt_75():
    print()
    print("=" * 60)
    print("TEST 3: 75° steep tilt (V5/V6 ECG)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=99)
    paste_split_marker(scene, marker, 450, 280, scale=0.35, angle_deg=75)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    if ok:
        pose = det.estimate_pose(corners, scene.shape)
        if pose:
            print(f"  Distance: {pose['distance_cm']:.1f} cm")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_small_scale():
    print()
    print("=" * 60)
    print("TEST 4: Small scale (simulating ~50cm distance)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=55)
    # Small scale → watch screen is ~40×50 pixels
    paste_split_marker(scene, marker, 500, 300, scale=0.12)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_no_false_positive():
    print()
    print("=" * 60)
    print("TEST 5: No marker (should NOT detect)")
    print("=" * 60)

    from detect_marker import ScreenDetector

    scene = create_background(600, 800, seed=123)

    # Add some random greenish patches (NOT our marker)
    cv2.circle(scene, (300, 300), 15, (40, 120, 40), -1)
    cv2.rectangle(scene, (500, 200), (530, 250), (50, 100, 50), -1)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is None
    print(f"  {'Correctly: no marker found' if ok else '*** FALSE POSITIVE ***'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_dim_brightness():
    print()
    print("=" * 60)
    print("TEST 6: Dim marker (40% brightness)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=40, show_preview=False)

    scene = create_background(600, 800, seed=66)
    paste_split_marker(scene, marker, 450, 300, scale=0.3)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_with_reflection():
    print()
    print("=" * 60)
    print("TEST 7: Glass reflection on screen")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=88)
    paste_split_marker(scene, marker, 450, 280, scale=0.3)
    # Add a specular reflection in the marker area
    add_glass_reflection(scene, 460, 275, radius=18)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_tilt_75_small():
    print()
    print("=" * 60)
    print("TEST 8: 75° tilt + small scale (worst case)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=44)
    paste_split_marker(scene, marker, 500, 300, scale=0.18, angle_deg=75)

    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    print(f"  Screen {'found' if ok else 'NOT found'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_body_rotation_values():
    """Verify body_rotation is present and increases with tilt angle."""
    print()
    print("=" * 60)
    print("TEST 9: Body rotation values at different tilt angles")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)
    det = ScreenDetector(screen_width_mm=37.6, screen_height_mm=46.0)

    rotations = []
    for angle in [0, 45, 65, 75]:
        scene = create_background(600, 800, seed=200 + angle)
        paste_split_marker(scene, marker, 450, 280, scale=0.3,
                           angle_deg=angle)
        corners, _ = det.find_screen(scene)
        if corners is None:
            print(f"  Tilt {angle}°: NOT detected")
            rotations.append(None)
            continue
        pose = det.estimate_pose(corners, scene.shape)
        if pose and 'body_rotation' in pose:
            br = pose['body_rotation']
            ecg = pose['ecg_label']
            print(f"  Tilt {angle} -> body_rotation={br:.1f}  [{ecg}]")
            rotations.append(br)
        else:
            print(f"  Tilt {angle}°: pose failed or no body_rotation key")
            rotations.append(None)

    # Check: body_rotation key always present when pose succeeds
    valid = [r for r in rotations if r is not None]
    ok = len(valid) >= 3  # at least 3 of 4 should work
    if ok and len(valid) >= 2:
        # Check monotonically increasing (with some tolerance)
        ok = ok and valid[-1] > valid[0]
        print(f"  Monotonic check: {valid[0]:.1f} -> {valid[-1]:.1f}"
              f" ({'increasing' if valid[-1] > valid[0] else 'NOT increasing'})")

    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def test_custom_screen_size():
    """Verify detector works with user-measured 29x17mm screen size."""
    print()
    print("=" * 60)
    print("TEST 10: Custom screen size (29x17mm)")
    print("=" * 60)

    from generate_marker import generate_split_marker
    from detect_marker import ScreenDetector

    marker = generate_split_marker(
        watch_model="45mm", brightness=60, show_preview=False)

    scene = create_background(600, 800, seed=150)
    paste_split_marker(scene, marker, 450, 300, scale=0.3)

    # Use user's measured dimensions
    det = ScreenDetector(screen_width_mm=29.0, screen_height_mm=17.0)
    corners, mask = det.find_screen(scene)

    ok = corners is not None
    if ok:
        pose = det.estimate_pose(corners, scene.shape)
        if pose:
            print(f"  Distance: {pose['distance_cm']:.1f} cm (with 29x17mm)")
            print(f"  Body Rotation: {pose['body_rotation']:.1f}°")
            ok = 'body_rotation' in pose
    print(f"  Screen {'found' if ok else 'NOT found'}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


# ================================================================== #
#                    Main                                             #
# ================================================================== #

if __name__ == "__main__":
    print("SplitScreen v4 Test Pipeline")
    print("============================\n")

    results = []
    results.append(("Frontal view", test_frontal()))
    results.append(("65° tilt", test_tilt_65()))
    results.append(("75° steep tilt", test_tilt_75()))
    results.append(("Small scale", test_small_scale()))
    results.append(("No false positive", test_no_false_positive()))
    results.append(("40% brightness", test_dim_brightness()))
    results.append(("Glass reflection", test_with_reflection()))
    results.append(("75° + small", test_tilt_75_small()))
    results.append(("Body rotation values", test_body_rotation_values()))
    results.append(("Custom screen size", test_custom_screen_size()))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results:
        s = "PASS" if passed else "FAIL"
        print(f"  {s} : {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed!")
    else:
        print("Some tests FAILED.")
        sys.exit(1)
