"""Small, stateful capture gate for a live ChArUco calibration preview.

This gate selects steady, sharp, different board views. Its conservative quality
thresholds are heuristics, not a guarantee of a good lens calibration: the final
solver error and corrected-image review still matter. Sharpness is measured on
the board ROI so an OBS canvas's black sidebars do not bias the result.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


class CalibrationCaptureGate:
    def __init__(self, total=10, *, automatic=True, start_delay=5.0, hold_seconds=1.0,
                 cooldown_seconds=2.0, board_columns=6, board_rows=4,
                 min_corners=10, min_sharpness=35.0, stable_motion=0.025,
                 max_similar_tilt=3, min_motion_pixels=6.0, max_motion_pixels=8.0):
        if total < 1 or board_columns < 2 or board_rows < 2 or min_corners < 4 or max_similar_tilt < 1:
            raise ValueError("Capture and board counts must be positive.")
        if any(not math.isfinite(value) or value < 0 for value in
               (start_delay, hold_seconds, cooldown_seconds, min_sharpness, stable_motion,
                min_motion_pixels, max_motion_pixels)):
            raise ValueError("Capture thresholds must be finite and non-negative.")
        if max_motion_pixels < min_motion_pixels:
            raise ValueError("Maximum motion tolerance must be at least the minimum.")
        self.total = int(total)
        self.automatic = bool(automatic)
        self.start_delay = float(start_delay)
        self.hold_seconds = float(hold_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.board_columns, self.board_rows = int(board_columns), int(board_rows)
        self.min_corners = int(min_corners)
        self.min_sharpness = float(min_sharpness)
        self.stable_motion = float(stable_motion)
        self.min_motion_pixels, self.max_motion_pixels = float(min_motion_pixels), float(max_motion_pixels)
        self.max_similar_tilt = int(max_similar_tilt)
        self._started = None
        self._last_time = None
        self._stable_since = None
        self._anchor = None
        self._last_capture = None
        self._poses = []
        self._hold_resets = 0
        self._last_hold_reset_reason = None
        self._hold_reset = False
        self._diagnostics = {}

    @property
    def count(self):
        return len(self._poses)

    def _reset_hold(self, reason=None):
        if reason is not None and self._anchor is not None:
            self._hold_resets += 1
            self._last_hold_reset_reason = reason
            self._hold_reset = True
        self._anchor = None
        self._stable_since = None

    def _status(self, now, message, reason, *, capture=False, ready=False, sharpness=None):
        elapsed = max(0.0, now - self._started)
        stable = 0.0 if self._stable_since is None else max(0.0, now - self._stable_since)
        return {"capture": capture, "message": message, "reason": reason,
                "count": self.count, "accepted": self.count, "total": self.total,
                "progress": min(1.0, self.count / self.total),
                "complete": self.count >= self.total, "ready": ready,
                "stableProgress": min(1.0, stable / self.hold_seconds) if self.hold_seconds else 1.0,
                "startProgress": min(1.0, elapsed / self.start_delay) if self.start_delay else 1.0,
                "sharpness": sharpness if sharpness is not None else self._diagnostics.get("sharpness"),
                "holdResets": self._hold_resets, "holdReset": self._hold_reset,
                "lastHoldResetReason": self._last_hold_reset_reason, **self._diagnostics}

    def _view(self, gray, corners, ids, image_size):
        self._diagnostics = {"detectedCorners": 0, "validCorners": 0, "poseCorners": 0,
                             "poseCornerIds": [], "grossCornerOutliers": 0,
                             "sharedCorners": 0, "motionPixels": None, "motionRatio": None,
                             "motionLimitPixels": None, "motionBasis": None}
        width, height = image_size
        if (not isinstance(gray, np.ndarray) or gray.ndim != 2
                or gray.shape != (height, width) or width <= 0 or height <= 0):
            return None, "Camera frame dimensions changed; keep the feed settings fixed.", "frame_size"
        if corners is None or ids is None:
            return None, "Show the ChArUco board to the camera.", "no_board"
        try:
            points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
            self._diagnostics["detectedCorners"] = len(points)
            raw_ids = np.asarray(ids, dtype=np.float64).reshape(-1)
            if (not np.isfinite(raw_ids).all() or np.any(raw_ids < 0)
                    or np.any(raw_ids >= self.board_columns * self.board_rows)):
                return None, "Wait for a clear board detection.", "corners"
            indices = raw_ids.astype(np.int64)
        except (TypeError, ValueError, OverflowError):
            return None, "Wait for a clear board detection.", "corners"
        if (len(points) != len(indices) or len(points) < self.min_corners
                or not np.isfinite(points).all()
                or not np.equal(raw_ids, indices).all() or len(set(indices)) != len(indices)
                or np.any(indices < 0) or np.any(indices >= self.board_columns * self.board_rows)):
            return None, "Show more of the board; at least ten clear inner corners are needed.", "corners"
        if (np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width)
                or np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height)):
            return None, "Keep the board inside the camera view.", "coverage"
        order = np.argsort(indices)
        points, indices = points[order], indices[order]
        minimum, maximum = points.min(axis=0), points.max(axis=0)
        extent = maximum - minimum
        area = abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))
        if (extent[0] < 48 or extent[1] < 48 or area < 2000
                or area < extent[0] * extent[1] * .3):
            return None, "Move closer or show a broader, less edge-on view of the board.", "coverage"
        self._diagnostics["validCorners"] = len(points)

        board = np.column_stack((indices % self.board_columns, indices // self.board_columns)).astype(np.float64)
        # An isolated bad corner must not distort the inferred whole-board pose.
        # This robust fit is ONLY a pose/motion estimate. Lens distortion itself
        # is not homographic: its large edge residuals are useful measurements
        # for the solver and must not be removed from a calibration capture.
        homography, _mask = cv2.findHomography(board, points, method=cv2.LMEDS,
                                              maxIters=256, confidence=.99)
        if homography is None or not np.isfinite(homography).all():
            return None, "Show a broader view of the board.", "coverage"
        projected = cv2.perspectiveTransform(board.reshape(-1, 1, 2), homography).reshape(-1, 2)
        residuals = np.linalg.norm(projected - points, axis=1)
        median = float(np.median(residuals))
        deviation = float(np.median(np.abs(residuals - median)))
        cutoff = max(3.0, .02 * math.sqrt(area), median + 4.5 * max(.5, deviation))
        inliers = np.isfinite(residuals) & (residuals <= cutoff)
        self._diagnostics.update(poseCorners=int(np.count_nonzero(inliers)),
                                 poseCornerIds=indices[inliers].tolist())
        if not np.all(inliers) and np.count_nonzero(inliers) >= 4:
            refined, _mask = cv2.findHomography(board[inliers], points[inliers], method=0)
            if refined is not None and np.isfinite(refined).all():
                homography = refined
        gross = []
        for index, residual in enumerate(residuals):
            if residual <= max(10.0, .08 * math.sqrt(area)):
                continue
            # A smooth curved row/edge can have a large H residual. Reject a
            # frame only for a large isolated discontinuity, not a curved edge.
            distance = np.sum(np.abs(board - board[index]), axis=1)
            nearby = residuals[(distance > 0) & (distance <= 1)]
            if len(nearby) >= 2 and np.max(nearby) < max(5.0, residual * .25):
                gross.append(int(indices[index]))
        self._diagnostics["grossCornerOutliers"] = len(gross)

        # Work only on the detected board; padding includes marker edges without
        # including large letterbox areas. Downsampling bounds per-frame cost.
        margin = max(2, int(min(extent) * .04))
        x0, y0 = np.maximum(0, np.floor(minimum).astype(int) - margin)
        x1, y1 = np.minimum((width, height), np.ceil(maximum).astype(int) + margin + 1)
        roi = gray[y0:y1, x0:x1]
        scale = min(1.0, 480.0 / max(roi.shape))
        if scale < 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(roi.astype(np.float32), cv2.CV_32F).var())
        self._diagnostics["sharpness"] = sharpness
        if not math.isfinite(sharpness) or sharpness < self.min_sharpness:
            return None, "The board is blurry. Hold still and let the camera focus.", "blur"

        anchors = np.array([[[0, 0], [self.board_columns - 1, 0],
                             [self.board_columns - 1, self.board_rows - 1], [0, self.board_rows - 1]]], np.float64)
        quad = cv2.perspectiveTransform(anchors, homography)[0]
        if not np.isfinite(quad).all():
            return None, "Wait for a clear board detection.", "corners"
        top, right, bottom, left = (float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4))
        if min(top, right, bottom, left) <= 1:
            return None, "Show a broader view of the board.", "coverage"
        board_scale = math.sqrt(max(1.0, abs(cv2.contourArea(quad.astype(np.float32)))))
        pose = {"center": quad.mean(axis=0), "scale": board_scale,
                "angle": math.atan2(*(quad[1] - quad[0])[::-1]),
                "tilt": np.array([math.log(top / bottom), math.log(left / right),
                    math.log(((top + bottom) / (left + right)) /
                             ((self.board_columns - 1) / (self.board_rows - 1)))])}
        return {"points": points, "ids": tuple(indices.tolist()), "pose": pose, "quad": quad,
                "sharpness": sharpness, "scale": board_scale, "grossOutliers": gross}, None, None

    def _motion(self, view):
        anchor = self._anchor
        common = sorted(set(anchor["ids"]) & set(view["ids"]))
        grid = np.array([(index % self.board_columns, index // self.board_columns) for index in common], np.float32)
        spread = len(common) >= 6 and abs(cv2.contourArea(cv2.convexHull(grid))) >= 1
        if spread:
            previous = dict(zip(anchor["ids"], anchor["points"]))
            current = dict(zip(view["ids"], view["points"]))
            distances = np.linalg.norm(np.array([current[index] - previous[index] for index in common]), axis=1)
            movement = float(np.quantile(distances, .85))
            basis = "shared-corners"
        else:
            # Partial occlusion can leave disjoint/collinear subsets. A robust
            # board homography supplies the same canonical anchors in both views.
            movement = float(np.quantile(np.linalg.norm(view["quad"] - anchor["quad"], axis=1), .85))
            basis = "board-pose"
        limit = max(self.min_motion_pixels, min(self.max_motion_pixels,
                                               self.stable_motion * anchor["scale"]))
        self._diagnostics.update(sharedCorners=len(common), motionPixels=movement,
                                 motionRatio=movement / max(1.0, anchor["scale"]),
                                 motionLimitPixels=limit, motionBasis=basis)
        return movement > limit

    def _novel(self, view):
        pose = view["pose"]
        similar_tilt = 0
        for previous in self._poses:
            prior = previous["pose"]
            scale = (pose["scale"] + prior["scale"]) / 2
            common = sorted(set(previous["ids"]) & set(view["ids"]))
            grid = np.array([(index % self.board_columns, index // self.board_columns) for index in common], np.float32)
            if len(common) >= 6 and abs(cv2.contourArea(cv2.convexHull(grid))) >= 1:
                before = dict(zip(previous["ids"], previous["points"]))
                current = dict(zip(view["ids"], view["points"]))
                movement = float(np.quantile(np.linalg.norm(
                    np.array([current[index] - before[index] for index in common]), axis=1), .85))
                limit = max(self.min_motion_pixels, min(self.max_motion_pixels, self.stable_motion * scale))
                if movement <= limit:
                    return False, "Move or tilt the camera to a different view of the board.", "duplicate"
            shift = float(np.linalg.norm(pose["center"] - prior["center"])) / scale
            size_change = abs(math.log(pose["scale"] / prior["scale"]))
            angle = abs(math.atan2(math.sin(pose["angle"] - prior["angle"]), math.cos(pose["angle"] - prior["angle"])))
            tilt = float(np.max(np.abs(pose["tilt"] - prior["tilt"])))
            if shift < .16 and size_change < .14 and angle < math.radians(8) and tilt < .07:
                return False, "Move or tilt the camera to a different view of the board.", "duplicate"
            if tilt < .07:
                similar_tilt += 1
        if similar_tilt >= self.max_similar_tilt:
            return False, "Tilt the camera to a different angle; sideways movement alone is not enough.", "angle"
        return True, None, None

    def observe(self, now, frame_gray, charuco_corners, charuco_ids, image_size, *, manual=False):
        now = float(now)
        if not math.isfinite(now) or self._last_time is not None and now < self._last_time:
            raise ValueError("Capture time must be finite and monotonic.")
        self._last_time = now
        self._hold_reset = False
        if self._started is None:
            self._started = now
        if self.count >= self.total:
            return self._status(now, "All calibration views captured.", "complete")
        view, message, reason = self._view(frame_gray, charuco_corners, charuco_ids, image_size)
        if view is None:
            self._reset_hold(reason)
            return self._status(now, message, reason)
        sharpness = view["sharpness"]
        novel, message, reason = self._novel(view)
        if not novel:
            self._reset_hold(reason)
            return self._status(now, message, reason, sharpness=sharpness)

        if self._anchor is None:
            self._anchor, self._stable_since = view, now
        elif self._motion(view):
            self._reset_hold("motion")
            self._anchor, self._stable_since = view, now
        if view["grossOutliers"]:
            return self._status(now, "Wait for a clearer board detection; an isolated corner is misplaced.",
                                "corner_outlier", sharpness=sharpness)
        if manual:
            return self._accept(now, view)
        if not self.automatic:
            return self._status(now, "Press C to capture this view.", "manual", ready=True, sharpness=sharpness)
        if now - self._started < self.start_delay:
            seconds = max(1, math.ceil(self.start_delay - (now - self._started)))
            return self._status(now, f"Starting in {seconds}… Show the board and hold still.", "starting", sharpness=sharpness)
        if self._last_capture is not None and now - self._last_capture < self.cooldown_seconds:
            return self._status(now, "Move to the next view, then hold still.", "cooldown", sharpness=sharpness)
        if now - self._stable_since < self.hold_seconds:
            return self._status(now, "Hold this view still…", "holding", ready=True, sharpness=sharpness)
        return self._accept(now, view)

    def _accept(self, now, view):
        self._poses.append(view)
        self._last_capture = now
        self._reset_hold()
        message = ("All calibration views captured." if self.count >= self.total else
                   f"Captured {self.count}/{self.total}. Move or tilt the camera, then hold still.")
        return self._status(now, message, "captured", capture=True, sharpness=view["sharpness"])
