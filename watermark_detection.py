"""Find the known Camera for OBS Studio caption without modifying camera pixels.

The small glyph templates contain the three-line caption only. Their slight
shape differences accommodate the current phone's lens-corrected feed. This
detector does not recognize arbitrary text or reconstruct anything it covers.
"""

from dataclasses import dataclass
import hashlib
import math

import cv2
import numpy as np

from app_paths import resource_path


TEMPLATE_PATHS = (
    "assets/watermarks/obs-caption-center.png",
    "assets/watermarks/obs-caption-left.png",
)


@dataclass(frozen=True)
class WatermarkObservation:
    mask: np.ndarray
    exclusion_mask: np.ndarray
    bbox: tuple[int, int, int, int] | None
    confidence: float
    provenance: dict | None = None

    @property
    def detected(self) -> bool:
        return self.bbox is not None


def _white_feature(frame):
    # Both brightness and low chroma matter: red bases and skin are not text.
    bright = cv2.inRange(frame, (160, 160, 160), (255, 255, 255))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pale = cv2.inRange(hsv, (0, 0, 0), (180, 100, 255))
    return cv2.bitwise_and(bright, pale)


class ObsWatermarkDetector:
    """Locate one translating caption in a post-undistortion BGR camera frame.

    ``mask`` marks glyphs plus a small positional margin for candidate rejection.
    ``exclusion_mask`` adds a wider margin for blurred background comparisons.
    Neither mask should be painted into the input image or used as image data.
    """

    def __init__(self, *, threshold=0.58, search_width=480):
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Watermark threshold must be between zero and one")
        if not isinstance(search_width, int) or search_width < 160:
            raise ValueError("Watermark search width must be at least 160 pixels")
        self.threshold = float(threshold)
        self.search_width = search_width
        self.templates = []
        for name in TEMPLATE_PATHS:
            template = cv2.imread(resource_path(name), cv2.IMREAD_GRAYSCALE)
            if template is None or template.ndim != 2 or cv2.countNonZero(template) < 100:
                raise ValueError(f"Missing or invalid watermark template: {name}")
            self.templates.append(np.where(template > 0, 255, 0).astype(np.uint8))
        self._scaled_templates = {}

    def _variants(self, frame_height, image_scale, scales):
        key = (frame_height, image_scale, tuple(scales))
        cached = self._scaled_templates.get(key)
        if cached is None:
            cached = []
            for template in self.templates:
                for scale in scales:
                    factor = frame_height / 1080.0 * scale
                    full = cv2.resize(template, None, fx=factor, fy=factor,
                                      interpolation=cv2.INTER_NEAREST)
                    small = cv2.resize(full, None, fx=image_scale, fy=image_scale,
                                       interpolation=cv2.INTER_AREA)
                    cached.append((full, small))
            self._scaled_templates[key] = cached
        return cached

    @staticmethod
    def _best_match(feature, variants):
        best = (0.0, None, None)
        for full, small in variants:
            if small.shape[0] > feature.shape[0] or small.shape[1] > feature.shape[1]:
                continue
            scores = cv2.matchTemplate(feature, small, cv2.TM_CCOEFF_NORMED)
            _minimum, score, _min_location, location = cv2.minMaxLoc(scores)
            if math.isfinite(score) and score > best[0]:
                best = (float(score), location, full)
        return best

    def detect(self, frame_bgr) -> WatermarkObservation:
        if (not isinstance(frame_bgr, np.ndarray) or frame_bgr.dtype != np.uint8
                or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3
                or min(frame_bgr.shape[:2]) < 32):
            raise ValueError("Watermark detection requires a uint8 BGR camera frame")
        height, width = frame_bgr.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        feature = _white_feature(frame_bgr)
        scale = min(1.0, self.search_width / float(width))
        small = cv2.resize(feature, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        best = self._best_match(small, self._variants(height, scale, (1.0,)))
        # Typical frames use the exact size. Only harder matches pay for a
        # small scale bank, accounting for position-dependent lens correction.
        if best[0] < .72:
            alternative = self._best_match(small, self._variants(height, scale, (.96, 1.04)))
            if alternative[0] > best[0]:
                best = alternative
        score, location, template = best
        if location is None or score < self.threshold:
            return WatermarkObservation(mask, mask.copy(), None, score)

        # Refine quarter-size localization to full pixels before making masks.
        x, y = (int(round(value / scale)) for value in location)
        th, tw = template.shape
        margin = max(5, int(math.ceil(2 / scale)))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + tw + margin), min(height, y + th + margin)
        roi = feature[y0:y1, x0:x1]
        if roi.shape[0] < th or roi.shape[1] < tw:
            return WatermarkObservation(mask, mask.copy(), None, score)
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        _minimum, refined_score, _min_location, refined = cv2.minMaxLoc(result)
        if not math.isfinite(refined_score) or refined_score < self.threshold:
            return WatermarkObservation(mask, mask.copy(), None, max(0.0, float(refined_score)))
        x, y = x0 + refined[0], y0 + refined[1]
        # A single word (or two matching lines) is insufficient. Requiring the
        # complete layout avoids masking ordinary CAMERA/STUDIO labels.
        patch = feature[y:y + th, x:x + tw]
        boundaries = (0, int(round(th / 3)), int(round(th * 125 / 180)), th)
        for a, b in zip(boundaries, boundaries[1:]):
            line_score = float(cv2.matchTemplate(
                patch[a:b], template[a:b], cv2.TM_CCOEFF_NORMED)[0, 0])
            if not math.isfinite(line_score) or line_score < .45:
                return WatermarkObservation(mask, mask.copy(), None, score)
        radius = max(2, int(round(5 * height / 1080.0)))
        comparison_radius = max(radius + 1, int(round(11 * height / 1080.0)))
        # Dilate only the caption patch, avoiding two full-HD morphology passes.
        glyph_mask = self._placed_mask(template, x, y, radius, (height, width))
        comparison_mask = self._placed_mask(template, x, y, comparison_radius, (height, width))
        return WatermarkObservation(glyph_mask, comparison_mask, (x, y, tw, th),
                                    min(score, float(refined_score)))

    @staticmethod
    def _placed_mask(template, x, y, radius, shape):
        padded = cv2.copyMakeBorder(template, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=0)
        patch = cv2.dilate(padded, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        result = np.zeros(shape, np.uint8)
        left, top = x - radius, y - radius
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(shape[1], left + patch.shape[1]), min(shape[0], top + patch.shape[0])
        result[y0:y1, x0:x1] = patch[y0 - top:y1 - top, x0 - left:x1 - left]
        return result


class BoardCaptionDetector:
    """Register the known caption after camera-to-board rectification.

    Rectification can rotate, stretch and clip camera-overlay lettering. A
    camera-space translation matcher cannot describe that image. Here SIFT
    correspondences propose a homography of the SAME two glyph templates.
    Independent visible-image correlation must then corroborate all three
    caption lines. No changed-pixel regions, object positions, identities or
    previous detections are inputs. Absence never reuses an old caption mask.

    Only the geometrically verified glyphs are excluded, with a small margin;
    the input pixels are untouched. This does not recover objects behind text.
    """

    METHOD = "known-caption-homography-v1"
    _LINES = ((0, 60), (60, 125), (125, 180))

    def __init__(self, *, search_width=640):
        if not isinstance(search_width, int) or search_width < 160:
            raise ValueError("Caption search width must be at least 160 pixels")
        self.search_width = search_width
        self._sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=.02,
                                     edgeThreshold=15)
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.templates = ObsWatermarkDetector().templates
        self._references = []
        for template in self.templates:
            points, descriptors = self._sift.detectAndCompute(template, None)
            self._references.append((template, points, descriptors,
                                      hashlib.sha256(template.tobytes()).hexdigest()))

    def _absent(self, shape):
        mask = np.zeros(shape, np.uint8)
        return WatermarkObservation(mask, mask.copy(), None, 0.,
                                    {"method": self.METHOD, "detected": False,
                                     "search_width": self.search_width,
                                     "opencv_version": cv2.__version__})

    def detect(self, board_bgr) -> WatermarkObservation:
        if (not isinstance(board_bgr, np.ndarray) or board_bgr.dtype != np.uint8
                or board_bgr.ndim != 3 or board_bgr.shape[2] != 3
                or min(board_bgr.shape[:2]) < 32):
            raise ValueError("Caption detection requires a uint8 BGR board image")
        height, width = board_bgr.shape[:2]
        # White glyph contrast survives fading better than a fixed brightness
        # threshold. The minimum channel also reduces saturated colored edges.
        feature = np.min(board_bgr, axis=2)
        scale = min(1., self.search_width / width)
        small = cv2.resize(feature, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        points, descriptors = self._sift.detectAndCompute(small, None)
        if descriptors is None or len(points) < 12:
            return self._absent((height, width))
        best = None
        for template, reference_points, reference_descriptors, fingerprint in self._references:
            matches = self._matcher.knnMatch(reference_descriptors, descriptors, k=2)
            good = [pair[0] for pair in matches
                    if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
            # Stable ordering and a local, nonparallel USAC seed avoid depending
            # on OpenCV's process-global RNG or another detector's random calls.
            good.sort(key=lambda item: (item.queryIdx, item.trainIdx, item.distance))
            if len(good) < 12:
                continue
            source = np.float32([reference_points[item.queryIdx].pt for item in good])
            target = np.float32([points[item.trainIdx].pt for item in good])
            params = cv2.UsacParams()
            params.randomGeneratorState = 0
            params.isParallel = False
            params.threshold = 2.
            params.confidence = .999
            params.maxIterations = 3000
            transform, inliers = cv2.findHomography(source, target, params)
            if transform is None or inliers is None:
                continue
            keep = inliers.ravel() > 0
            if np.count_nonzero(keep) < 12 or np.mean(keep) < .40:
                continue
            supported = source[keep]
            if (np.ptp(supported[:, 0]) < .30 * template.shape[1]
                    or np.ptp(supported[:, 1]) < .50 * template.shape[0]):
                continue
            transform = np.diag((1 / scale, 1 / scale, 1.)) @ transform
            result = self._verify(feature, template, transform)
            if result is None:
                continue
            score, visible, line_scores = result
            # The full image verification, not feature-match count alone,
            # selects between the two known lens-corrected glyph templates.
            rank = (score, int(np.count_nonzero(keep)), fingerprint)
            if best is None or rank > best[0]:
                best = (rank, template, transform, {"method": self.METHOD,
                    "detected": True, "template_sha256": fingerprint,
                    "search_width": self.search_width, "opencv_version": cv2.__version__,
                    "matches": len(good), "inliers": int(np.count_nonzero(keep)),
                    "visible_fraction": visible, "line_correlations": line_scores,
                    "homography": transform.tolist(), "ransac_seed": 0})
        if best is None:
            return self._absent((height, width))
        rank, template, transform, provenance = best
        # Dilate glyphs in their canonical space, so the margins follow the
        # same local scale/perspective as the lettering rather than a big box.
        masks = []
        for radius in (2, 4):
            glyph = cv2.dilate(template, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
            masks.append(cv2.warpPerspective(glyph, transform, (width, height),
                                             flags=cv2.INTER_NEAREST))
        nonzero = cv2.findNonZero(masks[0])
        if nonzero is None:
            return self._absent((height, width))
        return WatermarkObservation(masks[0], masks[1], cv2.boundingRect(nonzero),
                                    float(rank[0]), provenance)

    def _verify(self, feature, template, transform):
        """Reject spurious feature homographies against every visible line."""
        height, width = feature.shape
        th, tw = template.shape
        corners = np.float32(((0, 0), (tw, 0), (tw, th), (0, th)))
        homogeneous = np.column_stack((corners, np.ones(4))) @ transform.T
        if not np.isfinite(homogeneous).all() or np.any(homogeneous[:, 2] <= .20):
            return None
        projected = (homogeneous[:, :2] / homogeneous[:, 2:]).astype(np.float32)
        contour = projected.reshape(-1, 1, 2)
        area = abs(cv2.contourArea(contour))
        edges = np.linalg.norm(np.roll(projected, -1, axis=0) - projected, axis=1)
        if (not cv2.isContourConvex(contour) or area < .005 * width * height
                or area > .65 * width * height or edges.min() < 30
                or edges.max() / edges.min() > 5):
            return None
        try:
            inverse = np.linalg.inv(transform)
        except np.linalg.LinAlgError:
            return None
        patch = cv2.warpPerspective(feature, inverse, (tw, th))
        visible = cv2.warpPerspective(np.full(feature.shape, 255, np.uint8),
                                      inverse, (tw, th)) > 250
        if visible.mean() < .50:
            return None
        scores = []
        for start, end in self._LINES:
            valid = visible[start:end]
            # A cropped template is supported only when a substantial part of
            # EACH of its three lines remains in the actual image. Two words
            # cannot authorize an invented third line elsewhere on the board.
            if valid.mean() < .35:
                return None
            observed = patch[start:end][valid].astype(np.float64)
            expected = template[start:end][valid].astype(np.float64)
            if observed.std() < 4 or expected.std() < 4:
                return None
            observed -= observed.mean()
            expected -= expected.mean()
            score = float(observed @ expected / (np.linalg.norm(observed) * np.linalg.norm(expected)))
            if not math.isfinite(score) or score < .35:
                return None
            scores.append(score)
        score = float(np.mean(scores))
        if score < .45:
            return None
        return score, float(visible.mean()), scores
