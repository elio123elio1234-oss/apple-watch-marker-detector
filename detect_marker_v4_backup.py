#!/usr/bin/env python3
"""
Marker Detector v4 — Screen Rectangle Detection
=================================================

v4 STRATEGY: detect the watch SCREEN as a colored rectangle, not
individual dots. The SplitScreen marker (left=cyan, right=green)
fills the entire watch display, giving 100-500× more pixel area
than 4 small dots.

WHY: At V5/V6 ECG position (wrist tilted 60-80° against the chest),
a 45mm Apple Watch screen projects to roughly 20×80 pixels at 40cm.
Four individual dots within that area would be 3-5 pixels each —
impossible to detect reliably. But the FULL screen rectangle (even
compressed to 20px wide) is easy to find as a colored blob.

Detection pipeline:
  1. HSV filter (wide range covers green + cyan + angle-shifted)
  2. Find largest rectangular green/cyan blob (screen)
  3. Verify: two-color split (cyan + green) = our marker, not random
  4. minAreaRect → 4 corners
  5. Order corners using cyan/green centroids for orientation
  6. solvePnP → full 6DOF pose

Usage:
    python detect_marker.py --watch-model 45mm
    python detect_marker.py --watch-model 45mm --debug
    python detect_marker.py --screen-width 37.6 --screen-height 46.0
    python detect_marker.py --screen-size 29x17        # 2.9cm x 1.7cm measured
"""

import cv2
import numpy as np
import argparse
import time

# Watch screen physical dimensions (width, height in mm)
SCREEN_SIZES = {
    "38mm": (26.5, 33.3),
    "40mm": (30.7, 37.3),
    "41mm": (33.5, 41.0),
    "42mm": (28.7, 36.1),
    "44mm": (34.5, 42.0),
    "45mm": (37.6, 46.0),
    "49mm": (39.0, 48.0),
}


class ScreenDetector:
    """
    Detects a SplitScreen marker (left=cyan, right=green) as a
    colored rectangle and estimates 6DOF pose.
    """

    # Wide HSV range: catches green + cyan + color-shifted variants.
    # Sat ≥ 40 allows partially-reflected areas (glass reflections
    # add white light, reducing saturation).
    WIDE_HSV = (np.array([25, 40, 25]), np.array([115, 255, 255]))

    # Narrower ranges for separating cyan from green (orientation).
    # Higher saturation threshold for reliable color identification.
    CYAN_HSV = (np.array([78, 70, 25]), np.array([115, 255, 255]))
    GREEN_HSV = (np.array([25, 70, 25]), np.array([76, 255, 255]))

    def __init__(self, screen_width_mm=37.6, screen_height_mm=46.0):
        self.screen_w = screen_width_mm
        self.screen_h = screen_height_mm

        W, H = screen_width_mm, screen_height_mm
        self.model_points = np.array([
            [0, 0, 0],      # TL (cyan side, top)
            [W, 0, 0],      # TR (green side, top)
            [W, H, 0],      # BR (green side, bottom)
            [0, H, 0],      # BL (cyan side, bottom)
        ], dtype=np.float64)

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        self._prev_tvec = None
        self._prev_rvec = None
        self._smooth_factor = 0.55

        # Corner consistency: remember previous corners to prevent flicker
        self._prev_corners = None
        self._corner_max_jump = 80  # px — reject if corners jump more

        self._last_bbox = None
        self._frames_lost = 0
        self._max_lost_frames = 25

    def _get_camera_matrix(self, shape):
        if self.camera_matrix is None:
            h, w = shape[:2]
            f = w * 0.85
            self.camera_matrix = np.array([
                [f, 0, w / 2.0],
                [0, f, h / 2.0],
                [0, 0, 1],
            ], dtype=np.float64)
        return self.camera_matrix

    # ================================================================ #
    #                   Screen Mask                                     #
    # ================================================================ #

    def _create_screen_mask(self, frame):
        """Create binary mask covering all green+cyan pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.WIDE_HSV[0], self.WIDE_HSV[1])

        # Light morphology: close small gaps, remove noise
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

        return mask, hsv

    # ================================================================ #
    #                   Screen Detection                                #
    # ================================================================ #

    def find_screen(self, frame):
        """
        Find the watch screen as a colored rectangle.

        Returns (ordered_corners, mask) or (None, mask).
        ordered_corners is a (4, 2) float64 array: [TL, TR, BR, BL].
        """
        mask, hsv = self._create_screen_mask(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        max_frame_area = frame.shape[0] * frame.shape[1] * 0.6
        best = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25 or area > max_frame_area:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 1 or rh < 1:
                continue

            rect_area = rw * rh
            rectangularity = area / rect_area
            if rectangularity < 0.45:
                continue

            # Reject extremely thin slivers (likely edge artifacts)
            aspect = min(rw, rh) / max(rw, rh)
            if aspect < 0.03:
                continue

            # Verify two-color split inside the blob
            box = cv2.boxPoints(rect).astype(np.int32)
            poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(poly_mask, box, 255)

            cyan_m = cv2.inRange(hsv, self.CYAN_HSV[0], self.CYAN_HSV[1])
            green_m = cv2.inRange(hsv, self.GREEN_HSV[0], self.GREEN_HSV[1])

            cyan_in = np.sum((cyan_m & poly_mask) > 0)
            green_in = np.sum((green_m & poly_mask) > 0)

            has_split = (cyan_in > area * 0.12 and green_in > area * 0.12)

            # Light dark-surround check
            has_dark = self._check_surround(gray, poly_mask)

            # Must pass at least one: two-color split OR dark surround.
            # A random green object on a bright wall has neither.
            if not has_split and not has_dark:
                continue

            # Score: prefer large rectangular blobs with the two-color split
            score = area * rectangularity
            if has_split:
                score *= 3.0
            if has_dark:
                score *= 1.5

            if score > best_score:
                best_score = score
                best = (cnt, rect, has_split)

        if best is None:
            return None, mask

        cnt, rect, has_split = best
        box = cv2.boxPoints(rect)

        ordered = self._order_corners(box, frame, hsv, has_split)
        return ordered, mask

    def _check_surround(self, gray, screen_mask):
        """Check that the screen is brighter than its immediate surround."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(screen_mask, kernel, iterations=1)
        ring = dilated & ~screen_mask

        screen_px = gray[screen_mask > 0]
        ring_px = gray[ring > 0]

        if len(ring_px) < 5 or len(screen_px) < 5:
            return True

        return np.median(ring_px) < np.median(screen_px) * 0.75

    # ================================================================ #
    #                   Corner Ordering                                 #
    # ================================================================ #

    def _order_corners(self, box, frame, hsv, has_split):
        """
        Order minAreaRect corners as [TL, TR, BR, BL].

        Primary: use cyan/green centroids (reliable at moderate angles).
        Fallback: geometric ordering (longest side = vertical, top-left first).
        Always validates convexity to prevent crossed (bowtie) corners.
        """
        if has_split:
            ordered = self._order_by_color_split(box, frame, hsv)
            if ordered is not None:
                ordered = self._fix_crossed_corners(ordered)
                return ordered

        ordered = self._order_geometric(box)
        ordered = self._fix_crossed_corners(ordered)
        return ordered

    @staticmethod
    def _is_convex(pts):
        """
        Check if 4 ordered points form a convex (non-crossed) polygon.
        All cross products of consecutive edge pairs must have the same sign.
        """
        n = len(pts)
        sign = None
        for i in range(n):
            o = pts[i]
            a = pts[(i + 1) % n] - o
            b = pts[(i + 2) % n] - pts[(i + 1) % n]
            cross = a[0] * b[1] - a[1] * b[0]
            if abs(cross) < 1e-6:
                continue
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    @staticmethod
    def _fix_crossed_corners(corners):
        """
        If 4 corners form a crossed (bowtie) quadrilateral, uncross
        them by swapping the problematic pair.

        A bowtie happens when two adjacent corners are swapped.
        Try swapping middle pairs until the polygon is convex.
        """
        if ScreenDetector._is_convex(corners):
            return corners

        # Three possible pairings of 4 points into a quadrilateral:
        # Original: 0-1-2-3  (current, crossed)
        # Swap 1,3: 0-3-2-1
        # Swap 1,2: 0-2-1-3
        swaps = [
            [0, 2, 1, 3],  # swap indices 1 and 2
            [0, 3, 2, 1],  # swap indices 1 and 3
            [0, 1, 3, 2],  # swap indices 2 and 3
        ]
        for idx in swaps:
            candidate = corners[idx]
            if ScreenDetector._is_convex(candidate):
                return candidate

        # If nothing works, return original (shouldn't happen)
        return corners

    def _order_by_color_split(self, box, frame, hsv):
        """Order corners using cyan (left) and green (right) centroids."""
        poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, box.astype(np.int32), 255)

        cyan_m = cv2.inRange(hsv, self.CYAN_HSV[0], self.CYAN_HSV[1]) & poly_mask
        green_m = cv2.inRange(hsv, self.GREEN_HSV[0], self.GREEN_HSV[1]) & poly_mask

        cy_coords = np.where(cyan_m > 0)
        gr_coords = np.where(green_m > 0)

        if len(cy_coords[0]) < 5 or len(gr_coords[0]) < 5:
            return None

        # Centroids (x, y)
        cyan_c = np.array([np.mean(cy_coords[1]), np.mean(cy_coords[0])])
        green_c = np.array([np.mean(gr_coords[1]), np.mean(gr_coords[0])])

        # Split corners: closer to cyan = left side, closer to green = right
        left, right = [], []
        for pt in box:
            if np.linalg.norm(pt - cyan_c) <= np.linalg.norm(pt - green_c):
                left.append(pt)
            else:
                right.append(pt)

        if len(left) != 2 or len(right) != 2:
            return None

        # Within each pair, sort by y (top first)
        left.sort(key=lambda p: p[1])
        right.sort(key=lambda p: p[1])

        ordered = np.array([
            left[0],    # TL (cyan, top)
            right[0],   # TR (green, top)
            right[1],   # BR (green, bottom)
            left[1],    # BL (cyan, bottom)
        ], dtype=np.float64)

        # Corner consistency: if previous corners exist, check that this
        # ordering is consistent (no sudden jumps). If a corner swapped,
        # re-order using nearest-neighbor matching to previous frame.
        if self._prev_corners is not None:
            ordered = self._enforce_corner_consistency(ordered)

        self._prev_corners = ordered.copy()
        return ordered

    def _validate_corner_winding(self, corners):
        """
        Ensure corners wind clockwise: TL→TR→BR→BL.
        If counter-clockwise, reverse to fix.
        """
        # Cross product of (TR-TL) x (BL-TL)
        v1 = corners[1] - corners[0]  # TL→TR
        v2 = corners[3] - corners[0]  # TL→BL
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if cross < 0:
            # Counter-clockwise → reverse to TL, BL, BR, TR → remap
            corners = corners[[0, 3, 2, 1]]
        return corners

    def _enforce_corner_consistency(self, corners):
        """
        Prevent corner ordering flicker by matching to previous frame.
        If the new ordering is very different, re-assign corners using
        nearest-neighbor to the previous known positions.
        """
        prev = self._prev_corners
        # Check how far each corner moved
        dists = np.linalg.norm(corners - prev, axis=1)
        max_jump = np.max(dists)

        if max_jump < self._corner_max_jump:
            # Ordering is consistent, no fix needed
            return corners

        # Corners may have swapped — re-assign by closest match
        used = [False] * 4
        reordered = np.zeros_like(corners)
        for i in range(4):
            best_j = -1
            best_d = 1e9
            for j in range(4):
                if used[j]:
                    continue
                d = np.linalg.norm(corners[j] - prev[i])
                if d < best_d:
                    best_d = d
                    best_j = j
            reordered[i] = corners[best_j]
            used[best_j] = True

        return reordered

    def _order_geometric(self, box):
        """Fallback: order by y (top pair) then x (left first)."""
        sorted_y = box[np.argsort(box[:, 1])]
        top = sorted_y[:2]
        bottom = sorted_y[2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]

        return np.array([
            top[0],       # TL
            top[1],       # TR
            bottom[1],    # BR
            bottom[0],    # BL
        ], dtype=np.float64)

    # ================================================================ #
    #                   Pose Estimation                                 #
    # ================================================================ #

    def estimate_pose(self, corners, frame_shape):
        """Estimate 6DOF pose from 4 ordered screen corners."""
        cam = self._get_camera_matrix(frame_shape)

        try:
            ok, rvec, tvec = cv2.solvePnP(
                self.model_points, corners, cam, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
        except cv2.error:
            ok = False

        if not ok:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    self.model_points, corners, cam, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            except cv2.error:
                ok = False

        if not ok:
            return None

        # Reprojection error check: reject truly insane PnP solutions.
        # A crossed-corner bowtie produces errors >> screen diagonal.
        # Normal uncalibrated errors are high but proportional to screen size.
        reproj, _ = cv2.projectPoints(
            self.model_points, rvec, tvec, cam, self.dist_coeffs
        )
        reproj = reproj.reshape(-1, 2)
        reproj_err = np.mean(np.linalg.norm(reproj - corners, axis=1))
        screen_diag = np.linalg.norm(corners[0] - corners[2])
        if screen_diag > 1 and reproj_err > screen_diag * 3.0:
            # Extremely bad — likely crossed corners or flipped solution
            return None

        dist = np.linalg.norm(tvec)
        if dist < 5 or dist > 5000:
            return None

        # Temporal smoothing with outlier rejection
        if self._prev_tvec is not None:
            prev_dist = np.linalg.norm(self._prev_tvec)
            jump_ratio = abs(dist - prev_dist) / max(prev_dist, 1)

            if jump_ratio > 0.6:
                # Large jump — likely a bad frame. Use heavy dampening
                a = 0.85
            else:
                a = self._smooth_factor

            tvec = a * self._prev_tvec + (1 - a) * tvec

            # Proper rotation smoothing via SLERP-like interpolation:
            # Convert both to rotation matrices, blend, re-extract rvec
            R_prev, _ = cv2.Rodrigues(self._prev_rvec)
            R_new, _ = cv2.Rodrigues(rvec)
            R_blend = R_prev * a + R_new * (1 - a)
            # Re-orthogonalize via SVD
            U, _, Vt = np.linalg.svd(R_blend)
            R_ortho = U @ Vt
            # Ensure proper rotation (det = +1)
            if np.linalg.det(R_ortho) < 0:
                R_ortho = -R_ortho
            rvec, _ = cv2.Rodrigues(R_ortho)

        self._prev_tvec = tvec.copy()
        self._prev_rvec = rvec.copy()

        distance_mm = np.linalg.norm(tvec)
        angle_x = np.degrees(np.arctan2(tvec[0][0], tvec[2][0]))
        angle_y = np.degrees(np.arctan2(tvec[1][0], tvec[2][0]))
        rmat, _ = cv2.Rodrigues(rvec)

        # Body rotation: angle between screen normal and camera Z-axis.
        # The screen normal in camera coordinates is the 3rd column of R.
        # When the screen faces the camera: normal ≈ [0,0,1] → angle = 0°.
        # When tilted to V6 position: normal rotates → angle ≈ 70-80°.
        screen_normal = rmat[:, 2]  # 3rd column = screen Z in camera frame
        cos_angle = abs(screen_normal[2])  # dot product with [0,0,1]
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        body_rotation = np.degrees(np.arccos(cos_angle))

        # ECG position estimate
        if body_rotation < 15:
            ecg_label = "V1-V2 (frontal)"
        elif body_rotation < 35:
            ecg_label = "V3 (mid)"
        elif body_rotation < 55:
            ecg_label = "V4 (mid-lateral)"
        elif body_rotation < 70:
            ecg_label = "V5 (lateral)"
        else:
            ecg_label = "V6 (far lateral)"

        return {
            'rvec': rvec, 'tvec': tvec,
            'distance_mm': distance_mm,
            'distance_cm': distance_mm / 10.0,
            'angle_x': angle_x,
            'angle_y': angle_y,
            'body_rotation': body_rotation,
            'ecg_label': ecg_label,
            'rotation_matrix': rmat,
            'image_points': corners,
        }

    def estimate_distance_simple(self, rect, frame_shape):
        """
        Simple distance estimate when solvePnP fails.

        Uses: distance = focal * real_size / apparent_size
        The longer dimension of the rect ≈ screen height.
        """
        rw, rh = rect[1]
        apparent = max(rw, rh)
        if apparent < 2:
            return None

        cam = self._get_camera_matrix(frame_shape)
        focal = cam[0][0]
        real = self.screen_h  # physical height in mm

        distance_mm = focal * real / apparent
        return distance_mm / 10.0  # cm

    # ================================================================ #
    #                   Visualization                                   #
    # ================================================================ #

    def draw_results(self, frame, corners, pose):
        """Draw detected screen outline and pose info."""
        pts = corners.astype(int)
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 0), (255, 255, 0)]

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)

        for i, (label, color) in enumerate(zip(labels, colors)):
            pt = tuple(pts[i])
            cv2.circle(frame, pt, 5, color, -1)
            cv2.putText(frame, label, (pt[0] + 8, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if pose:
            self._draw_axes(frame, pose)
            self._draw_info_panel(frame, pose)

    def _draw_axes(self, frame, pose):
        cam = self._get_camera_matrix(frame.shape)
        al = self.screen_w * 0.4
        pts3d = np.array([
            [0, 0, 0], [al, 0, 0], [0, al, 0], [0, 0, -al]
        ], dtype=np.float64)
        img_pts, _ = cv2.projectPoints(
            pts3d, pose['rvec'], pose['tvec'], cam, self.dist_coeffs
        )
        o = tuple(img_pts[0].ravel().astype(int))
        x = tuple(img_pts[1].ravel().astype(int))
        y = tuple(img_pts[2].ravel().astype(int))
        z = tuple(img_pts[3].ravel().astype(int))
        cv2.line(frame, o, x, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(frame, o, y, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(frame, o, z, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def _draw_info_panel(self, frame, pose):
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (380, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Body rotation (big, prominent)
        br = pose['body_rotation']
        ecg = pose['ecg_label']
        # Color: green < 30°, yellow 30-60°, orange 60-80°, red > 80°
        if br < 30:
            rc = (0, 255, 0)
        elif br < 60:
            rc = (0, 220, 255)
        elif br < 80:
            rc = (0, 140, 255)
        else:
            rc = (0, 80, 255)
        cv2.putText(frame, f"Body Rotation: {br:.0f} deg",
                    (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, rc, 2)
        cv2.putText(frame, f"~ {ecg}",
                    (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1)

        # Distance
        d = pose['distance_cm']
        dc = (0, 255, 0) if d < 100 else (0, 200, 255)
        cv2.putText(frame, f"Distance: {d:.1f} cm",
                    (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc, 2)

        # Position angles
        cv2.putText(frame, f"Pos X: {pose['angle_x']:+.1f}  Y: {pose['angle_y']:+.1f}",
                    (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 220), 1)

        # Screen info
        cv2.putText(frame,
                    f"Screen: {self.screen_w:.1f}x{self.screen_h:.1f}mm",
                    (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

        # Draw rotation arc indicator
        self._draw_rotation_gauge(frame, br)

    def _draw_rotation_gauge(self, frame, body_rotation):
        """Draw a visual arc gauge showing body rotation 0-90°."""
        cx, cy = 340, 105
        radius = 55

        # Background arc (0-90°)
        cv2.ellipse(frame, (cx, cy), (radius, radius), 180, 0, 90,
                    (60, 60, 60), 2, cv2.LINE_AA)

        # Filled arc (current rotation)
        sweep = min(body_rotation, 90)
        if sweep > 0:
            if body_rotation < 30:
                ac = (0, 200, 0)
            elif body_rotation < 60:
                ac = (0, 180, 220)
            else:
                ac = (0, 120, 255)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 180, 0,
                        int(sweep), ac, 4, cv2.LINE_AA)

        # Needle
        angle_rad = np.radians(180 + sweep)
        nx = int(cx + radius * np.cos(angle_rad))
        ny = int(cy + radius * np.sin(angle_rad))
        cv2.line(frame, (cx, cy), (nx, ny), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 3, (255, 255, 255), -1)

        # Labels
        cv2.putText(frame, "0", (cx - radius - 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(frame, "90", (cx - 5, cy - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def draw_debug(self, frame, mask):
        """Show the HSV mask overlay in corner."""
        h, w = frame.shape[:2]
        ms = cv2.resize(mask, (w // 4, h // 4))
        mb = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)
        mb[:, :, 1] = ms  # green-tint
        frame[h - mb.shape[0]:h, w - mb.shape[1]:w] = mb
        cv2.putText(frame, "HSV Mask",
                    (w - mb.shape[1] + 5, h - mb.shape[0] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ================================================================ #
    #                   ROI Tracking                                    #
    # ================================================================ #

    def _compute_roi(self, frame_shape):
        if self._last_bbox is None or self._frames_lost > self._max_lost_frames:
            return None
        h, w = frame_shape[:2]
        bx, by, bw, bh = self._last_bbox
        expand = 1.2
        ex, ey = int(bw * expand), int(bh * expand)
        return (max(bx - ex, 0), max(by - ey, 0),
                min(bx + bw + ex, w), min(by + bh + ey, h))

    def _update_tracking(self, corners):
        if corners is not None:
            xs = corners[:, 0]
            ys = corners[:, 1]
            pad = 10
            self._last_bbox = (
                int(np.min(xs)) - pad, int(np.min(ys)) - pad,
                int(np.ptp(xs)) + 2 * pad, int(np.ptp(ys)) + 2 * pad,
            )
            self._frames_lost = 0
        else:
            self._frames_lost += 1

    # ================================================================ #
    #                   Full Pipeline                                   #
    # ================================================================ #

    def process_frame(self, frame, debug=False):
        """
        Run full detection + pose estimation on one frame.

        Returns (annotated_frame, pose_dict, mask).
        """
        # ROI search
        roi = self._compute_roi(frame.shape)
        sf = frame
        ox, oy = 0, 0

        if roi is not None:
            x1, y1, x2, y2 = roi
            sf = frame[y1:y2, x1:x2]
            ox, oy = x1, y1
            if debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)

        corners, mask = self.find_screen(sf)

        # Offset if we searched in ROI
        if corners is not None and (ox or oy):
            corners = corners + np.array([ox, oy], dtype=np.float64)

        # Fallback: search full frame if ROI failed
        if corners is None and roi is not None:
            corners, mask = self.find_screen(frame)

        pose = None
        if corners is not None:
            pose = self.estimate_pose(corners, frame.shape)
            if pose is not None:
                self.draw_results(frame, corners, pose)
            else:
                # Draw outline even without pose
                pts = corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1)%4]),
                             (0, 200, 255), 2, cv2.LINE_AA)
            self._update_tracking(corners)
        else:
            self._update_tracking(None)
            # Only reset smoothing after several lost frames (not on first miss)
            if self._frames_lost > 5:
                self._prev_tvec = None
                self._prev_rvec = None
                self._prev_corners = None
            cv2.putText(frame, "Searching for screen marker...",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
            cv2.putText(frame, f"Screen: {self.screen_w:.0f}x{self.screen_h:.0f}mm",
                        (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if debug:
            self.draw_debug(frame, mask)

        return frame, pose, mask


# ==================================================================== #
#                           CLI Main                                    #
# ==================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="SplitScreen Marker Detector v4"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--watch-model", type=str, default=None,
                        choices=list(SCREEN_SIZES.keys()),
                        help="Auto-set screen dimensions from watch model")
    parser.add_argument("--screen-width", type=float, default=None,
                        help="Physical screen width in mm (e.g. 29 for 2.9cm)")
    parser.add_argument("--screen-height", type=float, default=None,
                        help="Physical screen height in mm (e.g. 17 for 1.7cm)")
    parser.add_argument("--screen-size", type=str, default=None,
                        help="Screen WxH in mm, e.g. '29x17' (2.9cm x 1.7cm)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--smooth", type=float, default=0.4)
    args = parser.parse_args()

    # Determine screen dimensions
    sw, sh = 37.6, 46.0  # default: 45mm Apple Watch
    if args.watch_model:
        sw, sh = SCREEN_SIZES[args.watch_model]
    if args.screen_size:
        parts = args.screen_size.lower().split('x')
        sw, sh = float(parts[0]), float(parts[1])
    if args.screen_width:
        sw = args.screen_width
    if args.screen_height:
        sh = args.screen_height

    detector = ScreenDetector(screen_width_mm=sw, screen_height_mm=sh)
    detector._smooth_factor = args.smooth

    model_name = args.watch_model or "custom"
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera: {aw}x{ah} | Screen: {sw:.1f}x{sh:.1f}mm ({model_name})")
    print("Keys: q=quit  d=debug  s=screenshot  +/-=smoothing")

    show_debug = args.debug
    fps_timer = time.time()
    fc = 0
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, pose, _ = detector.process_frame(frame, show_debug)

        fc += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = fc / elapsed
            fc = 0
            fps_timer = time.time()

        cv2.putText(annotated, f"FPS: {fps:.0f}",
                    (annotated.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)
        if show_debug:
            cv2.putText(annotated, "DEBUG",
                        (annotated.shape[1] - 80, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        cv2.imshow("SplitScreen Detector v4", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('d'):
            show_debug = not show_debug
        elif key == ord('s'):
            fn = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fn, annotated)
            print(f"Saved: {fn}")
        elif key in (ord('+'), ord('=')):
            detector._smooth_factor = min(detector._smooth_factor + 0.05, 0.9)
            print(f"Smooth: {detector._smooth_factor:.2f}")
        elif key == ord('-'):
            detector._smooth_factor = max(detector._smooth_factor - 0.05, 0.0)
            print(f"Smooth: {detector._smooth_factor:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
