#!/usr/bin/env python3
"""
Marker Detector v5 — Clinical-Grade Screen Tracking
=====================================================

CLINICAL-GRADE rewrite for ECG electrode positioning.
Tracks a SplitScreen marker (left=cyan, right=green) on an Apple Watch
and computes body rotation angle for V1-V6 lead placement guidance.

Major improvements over v4:
  - One Euro Filter (Casiez et al. 2012) for adaptive noise reduction:
    rock-solid when stationary, zero-lag when moving.
  - Dual IPPE solution disambiguation using temporal pose continuity
    (eliminates the random angle jumps between the two PnP solutions).
  - solvePnP with useExtrinsicGuess for frame-to-frame consistency.
  - Tracking state machine: SEARCH -> LOCKED -> COASTING, like a
    fighter-jet lock-on. Persistent display during brief dropouts.
  - Velocity-aware ROI prediction for continuous tracking.
  - Adaptive detection thresholds (strict in search, relaxed when locked).
  - Confidence metric showing tracking quality.

Usage:
    python detect_marker.py --watch-model 45mm
    python detect_marker.py --screen-size 29x17
    python detect_marker.py --watch-model 45mm --debug
"""

import cv2
import numpy as np
import argparse
import time
from collections import deque

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


# ================================================================== #
#  One Euro Filter — adaptive low-pass for clinical-grade tracking   #
# ================================================================== #

class OneEuroFilter:
    """
    One Euro Filter (Casiez, Roussel, Vogel — CHI 2012).

    Adaptive low-pass filter that automatically balances jitter
    reduction (when still) vs. responsiveness (when moving).

    Parameters:
        min_cutoff : Minimum cutoff frequency (Hz). Lower = stronger
                     smoothing at rest. 0.3-1.0 for medical tracking.
        beta       : Speed coefficient. Higher = less lag during fast
                     motion. 0.001-0.01 for smooth tracking.
        d_cutoff   : Cutoff for the derivative (speed) estimation.
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _smoothing_factor(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()

        if self._x_prev is None:
            self._x_prev = float(x)
            self._dx_prev = 0.0
            self._t_prev = t
            return float(x)

        dt = t - self._t_prev
        if dt <= 1e-9:
            dt = 1.0 / 30.0  # assume 30 fps

        # Estimate speed (filtered derivative)
        dx = (float(x) - self._x_prev) / dt
        a_d = self._smoothing_factor(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adaptive cutoff: higher speed -> higher cutoff -> less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Apply low-pass
        a = self._smoothing_factor(cutoff, dt)
        x_hat = a * float(x) + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None


class OneEuroFilterVec:
    """One Euro Filter for N-dimensional vectors (e.g., 3D position)."""

    def __init__(self, dim, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(dim)
        ]

    def __call__(self, x, t=None):
        flat = x.flatten()
        out = np.array([f(v, t) for f, v in zip(self._filters, flat)])
        return out.reshape(x.shape)

    def reset(self):
        for f in self._filters:
            f.reset()


# ================================================================== #
#                   ScreenDetector  (clinical-grade)                  #
# ================================================================== #

class ScreenDetector:
    """
    Clinical-grade SplitScreen marker detector with 6DOF tracking.

    Tracking state machine:
        SEARCH   -> full-frame scan, strict thresholds
        LOCKED   -> ROI tracking, relaxed thresholds, One Euro filtering
        COASTING -> detection lost <N frames, show last pose with warning

    The detector resolves the IPPE planar-pose ambiguity by choosing
    the solution closest to the previous frame's rotation, eliminating
    the random angle jumps that plague single-frame PnP.
    """

    MODE_SEARCH = 0
    MODE_LOCKED = 1
    MODE_COASTING = 2

    # --- HSV colour ranges ---
    WIDE_HSV = (np.array([25, 40, 30]), np.array([115, 255, 255]))
    # Relaxed: wider for brightness robustness when locked
    WIDE_HSV_RELAXED = (np.array([18, 20, 12]), np.array([120, 255, 255]))
    CYAN_HSV = (np.array([78, 50, 20]), np.array([115, 255, 255]))
    GREEN_HSV = (np.array([25, 50, 20]), np.array([76, 255, 255]))

    def __init__(self, screen_width_mm=37.6, screen_height_mm=46.0):
        self.screen_w = screen_width_mm
        self.screen_h = screen_height_mm

        # Expected aspect ratio of screen (for scoring)
        self._expected_aspect = (min(screen_width_mm, screen_height_mm)
                                 / max(screen_width_mm, screen_height_mm))

        W, H = screen_width_mm, screen_height_mm
        self.model_points = np.array([
            [0, 0, 0],      # TL (cyan side, top)
            [W, 0, 0],      # TR (green side, top)
            [W, H, 0],      # BR (green side, bottom)
            [0, H, 0],      # BL (cyan side, bottom)
        ], dtype=np.float64)

        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # ---- Tracking state ----
        self._mode = self.MODE_SEARCH
        self._frames_locked = 0
        self._frames_lost = 0
        self._max_coast_frames = 12
        self._max_search_frames = 30

        # ---- Pose history (raw, before filtering) ----
        self._prev_rvec = None
        self._prev_tvec = None
        self._last_pose = None
        self._prev_corners = None
        self._corner_max_jump = 80

        # ---- Size gating (prevents lock-on to wrong objects) ----
        self._prev_area = None        # last known marker area in pixels
        self._area_ratio_max = 2.5    # reject if area changes >2.5x

        # ---- One Euro Filters ----
        self._filt_body_rot = OneEuroFilter(
            min_cutoff=0.4, beta=0.004, d_cutoff=1.0)
        self._filt_tvec = OneEuroFilterVec(
            3, min_cutoff=1.2, beta=0.008, d_cutoff=1.0)
        self._filt_dist = OneEuroFilter(
            min_cutoff=1.0, beta=0.006, d_cutoff=1.0)
        self._filt_ax = OneEuroFilter(
            min_cutoff=1.5, beta=0.01, d_cutoff=1.0)
        self._filt_ay = OneEuroFilter(
            min_cutoff=1.5, beta=0.01, d_cutoff=1.0)

        # ---- ROI tracking with velocity ----
        self._last_bbox = None
        self._bbox_velocity = np.array([0.0, 0.0])
        self._prev_center = None

        # ---- Confidence ----
        self._det_history = deque(maxlen=30)
        self._confidence = 0.0

        # ---- Running median pre-filter for body rotation ----
        self._rot_buffer = deque(maxlen=5)

        # ---- Corner smoothing (stops bounding box dancing) ----
        self._filt_corners = OneEuroFilterVec(
            8, min_cutoff=2.5, beta=0.015, d_cutoff=1.0)

        # ---- Lock confirmation (prevents false lock-on) ----
        self._confirm_count = 0
        self._confirm_required = 3

        # CLI compat
        self._smooth_factor = 0.4

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

    def _create_screen_mask(self, frame, relaxed=False):
        """Create binary mask covering all green+cyan pixels."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo, hi = (self.WIDE_HSV_RELAXED if relaxed else self.WIDE_HSV)
        mask = cv2.inRange(hsv, lo, hi)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=1)

        return mask, hsv

    # ================================================================ #
    #                   Screen Detection                                #
    # ================================================================ #

    def find_screen(self, frame, relaxed=False):
        """
        Find the watch screen as a coloured rectangle.

        Returns (ordered_corners, mask) or (None, mask).
        ordered_corners is a (4, 2) float64 array: [TL, TR, BR, BL].
        """
        mask, hsv = self._create_screen_mask(frame, relaxed=relaxed)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pre-compute colour masks ONCE (not per-contour — huge speedup)
        cyan_full = cv2.inRange(hsv, self.CYAN_HSV[0], self.CYAN_HSV[1])
        green_full = cv2.inRange(hsv, self.GREEN_HSV[0], self.GREEN_HSV[1])

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Speed: only examine largest candidates
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        max_frame_area = frame.shape[0] * frame.shape[1] * 0.6
        min_area = 12 if relaxed else 25
        min_rect = 0.30 if relaxed else 0.45
        min_aspect = 0.015 if relaxed else 0.03

        best = None
        best_score = -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_frame_area:
                continue

            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 1 or rh < 1:
                continue

            rect_area = rw * rh
            rectangularity = area / rect_area
            if rectangularity < min_rect:
                continue

            aspect = min(rw, rh) / max(rw, rh)
            if aspect < min_aspect:
                continue

            # Verify two-colour split (bounding-rect ROI — fast)
            box = cv2.boxPoints(rect).astype(np.int32)
            bx, by, bw_r, bh_r = cv2.boundingRect(box)
            bx = max(bx, 0); by = max(by, 0)
            bx2 = min(bx + bw_r, frame.shape[1])
            by2 = min(by + bh_r, frame.shape[0])
            if bx2 - bx < 2 or by2 - by < 2:
                continue
            small_mask = np.zeros((by2 - by, bx2 - bx), dtype=np.uint8)
            cv2.fillConvexPoly(small_mask, box - [bx, by], 255)

            cyan_in = cv2.countNonZero(
                cv2.bitwise_and(cyan_full[by:by2, bx:bx2], small_mask))
            green_in = cv2.countNonZero(
                cv2.bitwise_and(green_full[by:by2, bx:bx2], small_mask))

            split_thr = 0.06 if relaxed else 0.12
            has_split = (cyan_in > area * split_thr
                         and green_in > area * split_thr)

            # At extreme angles one colour dominates, but we STILL
            # require at least a TRACE of the minority colour.
            # This prevents locking onto random green/cyan objects.
            has_weak_split = False
            if relaxed and not has_split:
                minor = min(cyan_in, green_in)
                major = max(cyan_in, green_in)
                # Major colour must be strong, minor must exist at all
                has_weak_split = (major > area * 0.15
                                  and minor > max(area * 0.02, 3))

            has_dark = self._check_surround(gray, box)

            # ALWAYS require some colour evidence — dark surround alone
            # is not enough (too many false positives from dark objects)
            if not has_split and not has_weak_split:
                continue

            # Size gating: if we have a previous size, reject candidates
            # that are wildly different (prevents jumping to bg objects)
            if self._prev_area is not None and self._mode != self.MODE_SEARCH:
                size_ratio = area / max(self._prev_area, 1)
                if (size_ratio > self._area_ratio_max
                        or size_ratio < 1.0 / self._area_ratio_max):
                    continue

            # ===== MULTI-FACTOR SCORING =====
            # The old `area * rectangularity` let big background blobs
            # dominate. The new score uses what makes our marker UNIQUE:
            # a bright rectangle with the right aspect ratio and a
            # spatially-separated left-right cyan/green split.

            # 1) Aspect ratio closeness to expected watch screen shape
            aspect_err = abs(aspect - self._expected_aspect)
            aspect_err_rel = aspect_err / max(self._expected_aspect, 0.01)
            aspect_score = max(0.0, 1.0 - aspect_err_rel * 2.0)

            # 2) Colour split balance (ideal = 50/50 cyan:green)
            total_c = cyan_in + green_in
            balance_score = 0.0
            if total_c > 5:
                balance = min(cyan_in, green_in) / total_c
                balance_score = min(balance * 2.0, 1.0)

            # 3) SPATIAL split: cyan and green on OPPOSITE sides.
            #    This is THE key differentiator vs random reflections
            #    where colours are randomly mixed.
            spatial_score = 0.0
            cy_roi = cv2.bitwise_and(
                cyan_full[by:by2, bx:bx2], small_mask)
            gr_roi = cv2.bitwise_and(
                green_full[by:by2, bx:bx2], small_mask)
            cy_xs = np.where(cy_roi > 0)[1]
            gr_xs = np.where(gr_roi > 0)[1]
            if len(cy_xs) >= 3 and len(gr_xs) >= 3:
                sep = abs(float(np.mean(cy_xs)) - float(np.mean(gr_xs)))
                spatial_score = min(sep / max(bw_r * 0.25, 1), 1.0)

            # 4) Brightness: OLED is self-emitting (bright), not a
            #    dim reflection from ambient light.
            v_roi = hsv[by:by2, bx:bx2, 2]
            v_in = v_roi[small_mask > 0]
            mean_v = float(np.mean(v_in)) if len(v_in) > 0 else 0
            bright_score = min(mean_v / 120.0, 1.0)

            # 5) Colour coverage: what fraction of the blob is our
            #    colour? Real marker is nearly 100% coloured.
            coverage = total_c / max(area, 1)
            coverage_score = min(coverage / 0.7, 1.0)

            # Combine (multiplicative — each factor gates the rest)
            score = ((0.3 + rectangularity)
                     * (0.1 + aspect_score * 3.0)      # aspect   huge
                     * (0.1 + spatial_score * 4.0)      # spatial  huge
                     * (0.3 + balance_score * 2.0)      # balance  large
                     * (0.5 + bright_score)              # bright   moderate
                     * (0.5 + coverage_score)            # coverage moderate
                     * (1.0 + np.log1p(area) * 0.08))   # area     mild

            if has_split:
                score *= 2.0
            if has_dark:
                score *= 1.3
            if has_weak_split and not has_split:
                score *= 0.7  # weak split is suspicious

            if score > best_score:
                best_score = score
                best = (cnt, rect, has_split)

        if best is None:
            return None, mask

        cnt, rect, has_split = best
        box = cv2.boxPoints(rect)

        ordered = self._order_corners(box, frame, hsv, has_split)
        return ordered, mask

    def _check_surround(self, gray, box_pts):
        """Screen should be brighter than its immediate surround."""
        bx, by, bw, bh = cv2.boundingRect(box_pts)
        pad = 12
        y1, x1 = max(by - pad, 0), max(bx - pad, 0)
        y2, x2 = (min(by + bh + pad, gray.shape[0]),
                   min(bx + bw + pad, gray.shape[1]))
        roi = gray[y1:y2, x1:x2]
        if roi.size < 25:
            return True
        sm = np.zeros(roi.shape, dtype=np.uint8)
        cv2.fillConvexPoly(sm, box_pts - [x1, y1], 255)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        dilated = cv2.dilate(sm, kernel, iterations=1)
        ring = dilated & ~sm
        screen_px = roi[sm > 0]
        ring_px = roi[ring > 0]
        if len(ring_px) < 5 or len(screen_px) < 5:
            return True
        return float(np.median(ring_px)) < float(np.median(screen_px)) * 0.75

    # ================================================================ #
    #                   Corner Ordering                                 #
    # ================================================================ #

    def _order_corners(self, box, frame, hsv, has_split):
        """
        Order minAreaRect corners as [TL, TR, BR, BL].
        Primary: cyan/green colour centroids.  Fallback: geometric.
        Always validates convexity.
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
        n = len(pts)
        sign = None
        for i in range(n):
            a = pts[(i + 1) % n] - pts[i]
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
        if ScreenDetector._is_convex(corners):
            return corners
        for idx in ([0, 2, 1, 3], [0, 3, 2, 1], [0, 1, 3, 2]):
            candidate = corners[idx]
            if ScreenDetector._is_convex(candidate):
                return candidate
        return corners

    def _order_by_color_split(self, box, frame, hsv):
        poly_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(poly_mask, box.astype(np.int32), 255)

        cyan_m = (cv2.inRange(hsv, self.CYAN_HSV[0], self.CYAN_HSV[1])
                  & poly_mask)
        green_m = (cv2.inRange(hsv, self.GREEN_HSV[0], self.GREEN_HSV[1])
                   & poly_mask)

        cy_coords = np.where(cyan_m > 0)
        gr_coords = np.where(green_m > 0)

        if len(cy_coords[0]) < 5 or len(gr_coords[0]) < 5:
            return None

        cyan_c = np.array([np.mean(cy_coords[1]), np.mean(cy_coords[0])])
        green_c = np.array([np.mean(gr_coords[1]), np.mean(gr_coords[0])])

        left, right = [], []
        for pt in box:
            if np.linalg.norm(pt - cyan_c) <= np.linalg.norm(pt - green_c):
                left.append(pt)
            else:
                right.append(pt)

        if len(left) != 2 or len(right) != 2:
            return None

        left.sort(key=lambda p: p[1])
        right.sort(key=lambda p: p[1])

        ordered = np.array([
            left[0], right[0], right[1], left[1]
        ], dtype=np.float64)

        if self._prev_corners is not None:
            ordered = self._enforce_corner_consistency(ordered)

        self._prev_corners = ordered.copy()
        return ordered

    def _enforce_corner_consistency(self, corners):
        prev = self._prev_corners
        dists = np.linalg.norm(corners - prev, axis=1)
        if np.max(dists) < self._corner_max_jump:
            return corners

        used = [False] * 4
        reordered = np.zeros_like(corners)
        for i in range(4):
            best_j, best_d = -1, 1e9
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
        sorted_y = box[np.argsort(box[:, 1])]
        top = sorted_y[:2]
        bottom = sorted_y[2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        return np.array([
            top[0], top[1], bottom[1], bottom[0]
        ], dtype=np.float64)

    # ================================================================ #
    #           Pose Estimation  (clinical-grade)                       #
    # ================================================================ #

    def estimate_pose(self, corners, frame_shape):
        """
        Clinical-grade 6DOF pose estimation with IPPE disambiguation.

        Strategy:
          1. solvePnPGeneric (IPPE_SQUARE) -> get BOTH planar solutions.
          2. If previous pose exists, also try ITERATIVE with
             useExtrinsicGuess (pre-conditioned on last frame).
          3. Pick the solution closest to the previous rotation
             (eliminates the IPPE random-flip problem).
          4. Feed through One Euro Filters for adaptive smoothing.
          5. Compute body rotation from the selected rotation matrix.
        """
        cam = self._get_camera_matrix(frame_shape)
        t_now = time.time()

        # ------ Collect candidate solutions ------
        candidates = []

        # When LOCKED with previous pose, skip IPPE (expensive) and
        # use ITERATIVE with extrinsic guess directly (much faster).
        skip_ippe = (self._prev_rvec is not None
                     and self._mode == self.MODE_LOCKED)

        # Method 1: IPPE_SQUARE -> two solutions (skip when locked)
        if not skip_ippe:
            try:
                n_sol, rvecs, tvecs, reproj_errs = cv2.solvePnPGeneric(
                    self.model_points, corners, cam, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                for i in range(n_sol):
                    candidates.append(
                        (rvecs[i].copy(), tvecs[i].copy(),
                         float(reproj_errs[i][0])))
            except cv2.error:
                pass

        # Method 2: ITERATIVE with previous pose as seed
        if self._prev_rvec is not None:
            try:
                ok, rv_it, tv_it = cv2.solvePnP(
                    self.model_points, corners, cam, self.dist_coeffs,
                    rvec=self._prev_rvec.copy(),
                    tvec=self._prev_tvec.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    rp, _ = cv2.projectPoints(
                        self.model_points, rv_it, tv_it, cam,
                        self.dist_coeffs)
                    err = float(np.mean(np.linalg.norm(
                        rp.reshape(-1, 2) - corners, axis=1)))
                    candidates.append((rv_it, tv_it, err))
            except cv2.error:
                pass

        # Method 3: plain ITERATIVE (no seed) as fallback
        if not candidates:
            try:
                ok, rv, tv = cv2.solvePnP(
                    self.model_points, corners, cam, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    rp, _ = cv2.projectPoints(
                        self.model_points, rv, tv, cam, self.dist_coeffs)
                    err = float(np.mean(np.linalg.norm(
                        rp.reshape(-1, 2) - corners, axis=1)))
                    candidates.append((rv, tv, err))
            except cv2.error:
                pass

        if not candidates:
            return None

        # ------ Filter bad solutions ------
        screen_diag = np.linalg.norm(corners[0] - corners[2])
        valid = []
        for rv, tv, err in candidates:
            dist = np.linalg.norm(tv)
            if dist < 5 or dist > 5000:
                continue
            if screen_diag > 1 and err > screen_diag * 2.5:
                continue

            # PHYSICS CONSTRAINT: the screen must face the camera.
            # The screen normal in camera coords = R[:,2].
            # For a visible screen, the normal's Z component (towards
            # camera) should be positive (screen faces us).
            # In some setups (synthetic tests, flipped coords), all
            # solutions can be negative — we handle this by checking
            # that tz and normal_z have CONSISTENT signs.
            R_check, _ = cv2.Rodrigues(rv)
            screen_normal_z = R_check[2, 2]
            tz_val = tv[2][0]

            # Both should have the same sign (consistent geometry).
            # If tz > 0 (in front) then normal_z should be > 0 (facing us).
            # If both negative (synthetic), that's also internally consistent.
            # Reject only if they DISAGREE (one positive, one negative).
            signs_consistent = (screen_normal_z * tz_val > 0
                                or abs(screen_normal_z) < 0.1)
            if not signs_consistent:
                continue

            valid.append((rv, tv, err))

        if not valid:
            return None

        # ------ Select best using temporal continuity ------
        if self._prev_rvec is not None and len(valid) > 1:
            # Among candidates, prefer the one with physics-correct
            # normal AND closest to previous frame
            best = min(valid,
                       key=lambda c: self._pose_distance(c[0], c[1]))
        else:
            # First frame: pick the solution where screen faces camera
            # most directly (highest |normal_z|)
            def _initial_score(c):
                R_c, _ = cv2.Rodrigues(c[0])
                facing = abs(R_c[2, 2])  # higher = more frontal
                return -facing + c[2] * 0.01  # prefer facing + low error
            best = min(valid, key=_initial_score)

        rvec_raw, tvec_raw, reproj_err = best

        # Store raw pose for next frame's disambiguation
        self._prev_rvec = rvec_raw.copy()
        self._prev_tvec = tvec_raw.copy()

        # ------ One Euro Filtering ------
        dist_raw = float(np.linalg.norm(tvec_raw))
        distance_mm = self._filt_dist(dist_raw, t_now)

        ax_raw = float(np.degrees(np.arctan2(
            tvec_raw[0][0], tvec_raw[2][0])))
        ay_raw = float(np.degrees(np.arctan2(
            tvec_raw[1][0], tvec_raw[2][0])))
        angle_x = self._filt_ax(ax_raw, t_now)
        angle_y = self._filt_ay(ay_raw, t_now)

        # Rotation matrix from the raw (selected) rvec.
        # We filter the output body_rotation scalar, not the rvec,
        # because filtering Rodrigues vectors is numerically unstable
        # near gimbal-lock singularities.
        rmat, _ = cv2.Rodrigues(rvec_raw)

        # ------ Body Rotation ------
        # Screen normal in camera frame = R's 3rd column.
        # dot(normal, camera_Z=[0,0,1]) = rmat[2,2].
        # Use abs() because in some configurations (synthetic test,
        # flipped coords) the normal consistently points backward.
        # What matters is the ANGLE between normal and view axis.
        cos_angle = float(np.clip(abs(rmat[2, 2]), 0.0, 1.0))
        body_rot_raw = float(np.degrees(np.arccos(cos_angle)))

        # Running median pre-filter: reject outlier spikes BEFORE
        # One Euro Filter. A single bad frame can't fool the median.
        self._rot_buffer.append(body_rot_raw)
        if len(self._rot_buffer) >= 3:
            body_rot_median = float(np.median(list(self._rot_buffer)))
        else:
            body_rot_median = body_rot_raw

        body_rotation = self._filt_body_rot(body_rot_median, t_now)
        body_rotation = float(np.clip(body_rotation, 0.0, 90.0))

        # Update size gating reference
        self._prev_area = float(cv2.contourArea(
            corners.astype(np.float32)))

        # ECG position mapping
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
            'rvec': rvec_raw,
            'tvec': tvec_raw,
            'distance_mm': distance_mm,
            'distance_cm': distance_mm / 10.0,
            'angle_x': float(angle_x),
            'angle_y': float(angle_y),
            'body_rotation': body_rotation,
            'body_rotation_raw': float(body_rot_raw),
            'ecg_label': ecg_label,
            'rotation_matrix': rmat,
            'image_points': corners,
            'reproj_error': reproj_err,
            'confidence': self._confidence,
        }

    def _pose_distance(self, rvec, tvec):
        """
        Weighted distance between a candidate pose and the previous.
        Rotation distance (geodesic on SO(3)) weighted heavily because
        IPPE ambiguity flips are primarily rotational.
        """
        if self._prev_rvec is None:
            return 0.0

        # Rotation distance: angle of R_prev^T @ R_new
        R_prev, _ = cv2.Rodrigues(self._prev_rvec)
        R_new, _ = cv2.Rodrigues(rvec)
        R_diff = R_prev.T @ R_new
        cos_a = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        rot_deg = float(np.degrees(np.arccos(cos_a)))

        # Translation distance (normalised by distance)
        prev_d = max(float(np.linalg.norm(self._prev_tvec)), 1.0)
        t_dist = float(np.linalg.norm(tvec - self._prev_tvec)) / prev_d

        # Weight rotation much higher (IPPE flips are rotational)
        return rot_deg * 3.0 + t_dist * 50.0

    # ================================================================ #
    #                   Visualization                                   #
    # ================================================================ #

    def draw_results(self, frame, corners, pose, coasting=False):
        """Draw detected screen outline and pose info on frame."""
        pts = corners.astype(int)
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(255, 255, 0), (0, 255, 0), (0, 255, 0), (255, 255, 0)]

        line_color = (0, 140, 255) if coasting else (0, 200, 255)
        thickness = 1 if coasting else 2

        for i in range(4):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % 4])
            cv2.line(frame, p1, p2, line_color, thickness, cv2.LINE_AA)

        if not coasting:
            for i, (label, color) in enumerate(zip(labels, colors)):
                pt = tuple(pts[i])
                cv2.circle(frame, pt, 5, color, -1)
                cv2.putText(frame, label, (pt[0] + 8, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        if pose:
            if not coasting:
                self._draw_axes(frame, pose)
            self._draw_info_panel(frame, pose, coasting)

    def _draw_axes(self, frame, pose):
        cam = self._get_camera_matrix(frame.shape)
        al = self.screen_w * 0.4
        pts3d = np.array([
            [0, 0, 0], [al, 0, 0], [0, al, 0], [0, 0, -al]
        ], dtype=np.float64)
        try:
            img_pts, _ = cv2.projectPoints(
                pts3d, pose['rvec'], pose['tvec'], cam, self.dist_coeffs)
        except cv2.error:
            return
        o = tuple(img_pts[0].ravel().astype(int))
        x = tuple(img_pts[1].ravel().astype(int))
        y = tuple(img_pts[2].ravel().astype(int))
        z = tuple(img_pts[3].ravel().astype(int))
        cv2.line(frame, o, x, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.line(frame, o, y, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.line(frame, o, z, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, "X", x, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Y", y, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.putText(frame, "Z", z, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)

    def _draw_info_panel(self, frame, pose, coasting=False):
        overlay = frame.copy()
        ph = 230
        cv2.rectangle(overlay, (5, 5), (390, ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y0 = 32

        # Lock status indicator
        if coasting:
            status_text = "COASTING"
            sc = (0, 180, 255)
        elif self._mode == self.MODE_LOCKED:
            status_text = "LOCKED"
            sc = (0, 255, 0)
        else:
            status_text = "ACQUIRING"
            sc = (0, 255, 255)

        cv2.putText(frame, status_text, (295, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, sc, 2)

        # Confidence bar
        conf = pose.get('confidence', self._confidence)
        bar_w = int(60 * min(conf, 1.0))
        cv2.rectangle(frame, (295, y0 + 8), (355, y0 + 16),
                      (50, 50, 50), -1)
        if bar_w > 0:
            if conf > 0.7:
                bc = (0, 255, 0)
            elif conf > 0.4:
                bc = (0, 220, 255)
            else:
                bc = (0, 80, 255)
            cv2.rectangle(frame, (295, y0 + 8), (295 + bar_w, y0 + 16),
                          bc, -1)

        # Body rotation (big, prominent)
        br = pose['body_rotation']
        ecg = pose['ecg_label']
        if br < 30:
            rc = (0, 255, 0)
        elif br < 60:
            rc = (0, 220, 255)
        elif br < 80:
            rc = (0, 140, 255)
        else:
            rc = (0, 80, 255)
        cv2.putText(frame, f"Body Rotation: {br:.0f} deg",
                    (15, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.85, rc, 2)
        cv2.putText(frame, f"~ {ecg}",
                    (15, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rc, 1)

        # Distance
        d = pose['distance_cm']
        dc = (0, 255, 0) if d < 100 else (0, 200, 255)
        cv2.putText(frame, f"Distance: {d:.1f} cm",
                    (15, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dc, 2)

        # Position angles
        cv2.putText(frame,
                    f"Pos X: {pose['angle_x']:+.1f}  "
                    f"Y: {pose['angle_y']:+.1f}",
                    (15, y0 + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 220), 1)

        # Diagnostics line
        re = pose.get('reproj_error', 0)
        raw = pose.get('body_rotation_raw', br)
        cv2.putText(frame,
                    f"Reproj: {re:.1f}px  Raw: {raw:.0f}  "
                    f"Conf: {conf:.0%}",
                    (15, y0 + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (120, 120, 120), 1)

        # Screen info
        cv2.putText(frame,
                    f"Screen: {self.screen_w:.1f}x{self.screen_h:.1f}mm",
                    (15, y0 + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (100, 100, 100), 1)

        # Rotation arc gauge
        self._draw_rotation_gauge(frame, br)

    def _draw_rotation_gauge(self, frame, body_rotation):
        cx, cy = 345, 135
        radius = 55

        cv2.ellipse(frame, (cx, cy), (radius, radius), 180, 0, 90,
                    (60, 60, 60), 2, cv2.LINE_AA)

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

        angle_rad = np.radians(180 + sweep)
        nx = int(cx + radius * np.cos(angle_rad))
        ny = int(cy + radius * np.sin(angle_rad))
        cv2.line(frame, (cx, cy), (nx, ny), (255, 255, 255), 2,
                 cv2.LINE_AA)
        cv2.circle(frame, (nx, ny), 3, (255, 255, 255), -1)

        cv2.putText(frame, "0", (cx - radius - 12, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(frame, "90", (cx - 5, cy - radius - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    def draw_debug(self, frame, mask):
        h, w = frame.shape[:2]
        ms = cv2.resize(mask, (w // 4, h // 4))
        mb = cv2.cvtColor(ms, cv2.COLOR_GRAY2BGR)
        mb[:, :, 1] = ms
        frame[h - mb.shape[0]:h, w - mb.shape[1]:w] = mb
        cv2.putText(frame, "HSV Mask",
                    (w - mb.shape[1] + 5, h - mb.shape[0] + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ================================================================ #
    #               ROI Tracking with Velocity Prediction               #
    # ================================================================ #

    def _compute_roi(self, frame_shape):
        """Compute search ROI with velocity prediction."""
        if self._last_bbox is None:
            return None
        if self._frames_lost > self._max_search_frames:
            return None

        h, w = frame_shape[:2]
        bx, by, bw, bh = self._last_bbox

        # Predict position using velocity
        predict = min(self._frames_lost, 5)
        bx = int(bx + self._bbox_velocity[0] * predict)
        by = int(by + self._bbox_velocity[1] * predict)

        # Larger ROI when coasting
        expand = 1.2 + 0.15 * min(self._frames_lost, 10)
        ex = int(bw * expand)
        ey = int(bh * expand)

        return (max(bx - ex, 0), max(by - ey, 0),
                min(bx + bw + ex, w), min(by + bh + ey, h))

    def _update_tracking(self, corners):
        """Update ROI, velocity, and confidence."""
        if corners is not None:
            xs, ys = corners[:, 0], corners[:, 1]
            pad = 10
            cx, cy = float(np.mean(xs)), float(np.mean(ys))

            if self._prev_center is not None:
                vx = cx - self._prev_center[0]
                vy = cy - self._prev_center[1]
                self._bbox_velocity = (
                    0.7 * self._bbox_velocity
                    + 0.3 * np.array([vx, vy]))
            self._prev_center = np.array([cx, cy])

            self._last_bbox = (
                int(np.min(xs)) - pad, int(np.min(ys)) - pad,
                int(np.ptp(xs)) + 2 * pad, int(np.ptp(ys)) + 2 * pad,
            )
            self._frames_lost = 0
            self._frames_locked += 1
            self._det_history.append(1)
        else:
            self._frames_lost += 1
            self._frames_locked = 0
            self._det_history.append(0)

        if len(self._det_history) > 0:
            self._confidence = (
                sum(self._det_history) / len(self._det_history))
        else:
            self._confidence = 0.0

    # ================================================================ #
    #           Full Pipeline with State Machine                        #
    # ================================================================ #

    def process_frame(self, frame, debug=False):
        """
        Full detection + pose estimation with tracking state machine.

        States:
            SEARCH   -> scanning full frame, strict thresholds
            LOCKED   -> ROI + relaxed thresholds + filtering
            COASTING -> lost briefly, show last pose with warning

        Returns (annotated_frame, pose_dict, mask).
        """
        # ---- Detection strategy by mode ----
        corners = None
        mask = None

        if self._mode == self.MODE_SEARCH:
            # Downscale full-frame search for speed
            sf = frame
            scale = 1.0
            h_f, w_f = frame.shape[:2]
            if w_f > 640:
                scale = 640.0 / w_f
                sf = cv2.resize(sf, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            corners, mask = self.find_screen(sf, relaxed=False)
            if corners is None:
                corners, mask = self.find_screen(sf, relaxed=True)
            if corners is not None and scale < 1.0:
                corners = corners / scale
        else:
            # LOCKED / COASTING — velocity-predicted ROI
            roi = self._compute_roi(frame.shape)
            sf = frame
            ox, oy = 0, 0
            if roi is not None:
                x1, y1, x2, y2 = roi
                if x2 > x1 + 10 and y2 > y1 + 10:
                    sf = frame[y1:y2, x1:x2]
                    ox, oy = x1, y1
                    if debug:
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (255, 255, 0), 1)
            corners, mask = self.find_screen(sf, relaxed=True)
            if corners is not None and (ox or oy):
                corners = corners + np.array([ox, oy], dtype=np.float64)
            # Fallback: full frame if ROI failed
            if corners is None and roi is not None:
                corners, mask = self.find_screen(frame, relaxed=True)

        pose = None
        coasting = False
        t_now = time.time()

        if corners is not None:
            pose = self.estimate_pose(corners, frame.shape)
            if pose is not None:
                self._last_pose = pose.copy()
                self._last_pose['image_points'] = corners.copy()

                # Lock confirmation: require consecutive detections
                if self._mode == self.MODE_SEARCH:
                    self._confirm_count += 1
                    if self._confirm_count >= self._confirm_required:
                        self._mode = self.MODE_LOCKED
                else:
                    self._mode = self.MODE_LOCKED

                # Smooth corners for stable bounding box display
                if self._mode == self.MODE_LOCKED:
                    sm_c = self._filt_corners(
                        corners.flatten(), t_now).reshape(4, 2)
                else:
                    sm_c = corners
                self.draw_results(frame, sm_c, pose)
            else:
                pts = corners.astype(int)
                for i in range(4):
                    cv2.line(frame, tuple(pts[i]),
                             tuple(pts[(i + 1) % 4]),
                             (0, 200, 255), 2, cv2.LINE_AA)
            self._update_tracking(corners)

        else:
            self._confirm_count = 0
            self._update_tracking(None)

            if (self._frames_lost <= self._max_coast_frames
                    and self._last_pose is not None):
                # COASTING — show last known pose
                self._mode = self.MODE_COASTING
                coasting = True
                pose = self._last_pose
                lc = pose.get('image_points')
                if lc is not None:
                    self.draw_results(frame, lc, pose, coasting=True)
                remaining = self._max_coast_frames - self._frames_lost
                cv2.putText(
                    frame,
                    f"COASTING ({remaining})",
                    (15, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)
            else:
                # Fully lost
                self._mode = self.MODE_SEARCH
                if self._frames_lost > 12:
                    self._prev_rvec = None
                    self._prev_tvec = None
                    self._prev_corners = None
                    self._last_pose = None
                    self._reset_filters()

                cv2.putText(
                    frame, "Searching for screen marker...",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 200), 2)
                cv2.putText(
                    frame,
                    f"Screen: {self.screen_w:.0f}x"
                    f"{self.screen_h:.0f}mm",
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (150, 150, 150), 1)

        if debug:
            self.draw_debug(frame, mask)
            mode_names = {0: "SEARCH", 1: "LOCKED", 2: "COASTING"}
            cv2.putText(frame,
                        f"Mode: {mode_names.get(self._mode, '?')}",
                        (frame.shape[1] - 180, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

        return frame, pose, mask

    def _reset_filters(self):
        """Reset all One Euro Filters (after fully losing track)."""
        self._filt_body_rot.reset()
        self._filt_tvec.reset()
        self._filt_dist.reset()
        self._filt_ax.reset()
        self._filt_ay.reset()
        self._rot_buffer.clear()
        self._prev_area = None
        self._filt_corners.reset()
        self._confirm_count = 0

    # ================================================================ #
    #                   Utility                                         #
    # ================================================================ #

    def estimate_distance_simple(self, rect, frame_shape):
        rw, rh = rect[1]
        apparent = max(rw, rh)
        if apparent < 2:
            return None
        cam = self._get_camera_matrix(frame_shape)
        focal = cam[0][0]
        distance_mm = focal * self.screen_h / apparent
        return distance_mm / 10.0


# ==================================================================== #
#                           CLI Main                                    #
# ==================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="SplitScreen Marker Detector v5 (Clinical-Grade)"
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--watch-model", type=str, default=None,
                        choices=list(SCREEN_SIZES.keys()),
                        help="Auto-set screen dimensions from watch model")
    parser.add_argument("--screen-width", type=float, default=None,
                        help="Physical screen width in mm")
    parser.add_argument("--screen-height", type=float, default=None,
                        help="Physical screen height in mm")
    parser.add_argument("--screen-size", type=str, default=None,
                        help="Screen WxH in mm, e.g. '29x17'")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--smooth", type=float, default=0.4,
                        help="Smoothing (0.1=very smooth, 1.0=raw). "
                             "Controls One Euro Filter min_cutoff.")
    args = parser.parse_args()

    sw, sh = 37.6, 46.0
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

    # Map --smooth to One Euro Filter strength
    mc = max(0.1, args.smooth * 2.0)
    detector._filt_body_rot = OneEuroFilter(
        min_cutoff=mc * 0.4, beta=0.004, d_cutoff=1.0)
    detector._filt_tvec = OneEuroFilterVec(
        3, min_cutoff=mc * 1.2, beta=0.008, d_cutoff=1.0)
    detector._filt_dist = OneEuroFilter(
        min_cutoff=mc, beta=0.006, d_cutoff=1.0)
    detector._filt_ax = OneEuroFilter(
        min_cutoff=mc * 1.5, beta=0.01, d_cutoff=1.0)
    detector._filt_ay = OneEuroFilter(
        min_cutoff=mc * 1.5, beta=0.01, d_cutoff=1.0)

    model_name = args.watch_model or "custom"
    print("SplitScreen Detector v5 (Clinical-Grade)")
    print("=========================================")
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
    print(f"Smoothing: {args.smooth:.2f} (One Euro min_cutoff={mc:.2f})")
    print("Keys: q=quit  d=debug  s=screenshot  +/-=smoothing")
    print()

    show_debug = args.debug
    fps_timer = time.time()
    fc = 0
    fps = 0.0
    mc_current = mc

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

        cv2.imshow("SplitScreen Detector v5", annotated)

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
            mc_current = min(mc_current + 0.1, 3.0)
            detector._filt_body_rot.min_cutoff = mc_current * 0.4
            print(f"Smoothing cutoff: {mc_current:.2f} (less smooth)")
        elif key == ord('-'):
            mc_current = max(mc_current - 0.1, 0.1)
            detector._filt_body_rot.min_cutoff = mc_current * 0.4
            print(f"Smoothing cutoff: {mc_current:.2f} (more smooth)")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
