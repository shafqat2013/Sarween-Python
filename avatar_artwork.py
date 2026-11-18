"""Conservative, bounded matching of exposed Foundry artwork candidates.

This is an additional negative observation, not a foreground segmenter. The
caller decides when competing detections warrant a check and which catalog
belongs to the camera geometry. A match never supplies a miniature identity.
Whole-object pattern agreement is required: partly covered art remains unknown.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


MAX_TEMPLATES = 32
MAX_IMAGE_SIDE = 96
MAX_COMPARISONS = 96
SCALES = (.9, 1., 1.1)


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value))


def _features(bgr, mode):
    floating = bgr.astype(np.float32)
    if mode == "chroma":
        blue, green, red = cv2.split(floating)
        return np.dstack((red - green, blue - green))
    return cv2.cvtColor(floating, cv2.COLOR_BGR2GRAY)[..., None]


def _energy(pixels):
    centered = pixels - pixels.mean(axis=(0, 1), keepdims=True)
    return float(np.mean(centered * centered))


def _complex_pattern(image):
    """Refuse plain disks, annuli and other nearly radial color patterns.

    Alpha can carry the detail of a monochrome icon. Its cutouts must therefore
    participate in this check, rather than treating every solid ink color as a
    low-information template. A photograph can carry internal color detail.
    """
    height, width = image.shape[:2]
    yy, xx = np.mgrid[:height, :width]
    bins = np.floor(np.hypot((xx - (width - 1) / 2) / max(width, 1),
                            (yy - (height - 1) / 2) / max(height, 1)) * 32).astype(int)
    flat_bins = bins.ravel()
    counts = np.bincount(flat_bins).clip(1)
    residual = 0.
    for channel in range(image.shape[2]):
        values = image[..., channel]
        profile = np.bincount(flat_bins, weights=values.ravel()) / counts
        residual += float(np.mean((values - profile[bins]) ** 2))
    total = _energy(image) * image.shape[2]
    return total >= 36. and residual / max(total, 1e-6) >= .16


@dataclass(frozen=True)
class _Template:
    template_id: str
    pixels: np.ndarray
    mode: str
    full_width: int
    full_height: int
    left: int
    top: int
    offset: int
    foreground: np.ndarray
    mean_ink_chroma: np.ndarray
    blur_sigma: float


class AvatarArtworkMatcher:
    """Cache small pattern features and reject only fully explained candidates.

    ``set_catalog`` accepts decoded uint8 BGRA thumbnails. ``board_width`` and
    ``board_height`` are the rendered tile dimensions mapped to rectified camera
    pixels, including transparent padding. No dimensions are inferred from the
    candidate itself. Missing or invalid catalog data clears rejection authority.
    """
    def __init__(self):
        self._signature = None
        self._templates = ()
        self.scene_id = None
        self.catalog_id = None
        self.last_catalog_status = {"ready": False, "reason": "no_catalog"}

    def clear(self, reason="no_catalog"):
        self._signature = None
        self._templates = ()
        self.scene_id = self.catalog_id = None
        self.last_catalog_status = {"ready": False, "reason": reason}

    def set_catalog(self, catalog):
        if not isinstance(catalog, dict):
            self.clear()
            return False
        scene_id, catalog_id, revision = (catalog.get(key) for key in ("sceneId", "catalogId", "revision"))
        fingerprint, content_sha = catalog.get("fingerprint"), catalog.get("contentSha256")
        items = catalog.get("templates")
        if (not all(isinstance(value, str) and 0 < len(value) <= 256 for value in (scene_id, catalog_id))
                or not isinstance(revision, int) or isinstance(revision, bool) or revision < 0
                or any(value is not None and (not isinstance(value, str) or not 0 < len(value) <= 256)
                       for value in (fingerprint, content_sha))
                or not isinstance(items, list) or len(items) > MAX_TEMPLATES):
            self.clear("invalid_catalog")
            return False
        geometry = tuple((item.get("templateId") if isinstance(item.get("templateId"), str) else None,
                          item.get("board_width") if _number(item.get("board_width")) else None,
                          item.get("board_height") if _number(item.get("board_height")) else None)
                         for item in items if isinstance(item, dict))
        signature = scene_id, catalog_id, revision, fingerprint, content_sha, geometry
        images_valid = all(isinstance(item, dict)
            and isinstance(item.get("image_bgra"), np.ndarray)
            and item["image_bgra"].dtype == np.uint8 and item["image_bgra"].ndim == 3
            and item["image_bgra"].shape[2] == 4
            and 4 <= min(item["image_bgra"].shape[:2]) <= max(item["image_bgra"].shape[:2]) <= MAX_IMAGE_SIDE
            for item in items)
        if signature == self._signature and images_valid:
            return bool(self._templates)
        compiled, skipped, seen = [], [], set()
        for item in items:
            if not isinstance(item, dict):
                skipped.append("invalid_template")
                continue
            template_id, bgra = item.get("templateId"), item.get("image_bgra")
            width, height = item.get("board_width"), item.get("board_height")
            if (not isinstance(template_id, str) or not template_id or template_id in seen
                    or not isinstance(bgra, np.ndarray) or bgra.dtype != np.uint8
                    or bgra.ndim != 3 or bgra.shape[2] != 4
                    or min(bgra.shape[:2]) < 4 or max(bgra.shape[:2]) > MAX_IMAGE_SIDE
                    or not all(_number(value) and 8 <= value <= MAX_IMAGE_SIDE for value in (width, height))):
                skipped.append("invalid_template")
                continue
            seen.add(template_id)
            variants = self._compile(template_id, bgra.copy(), width, height)
            compiled.extend(variants)
            if not variants:
                skipped.append("uninformative_template")
        self._signature = signature
        self._templates = tuple(compiled)
        self.scene_id, self.catalog_id = scene_id, catalog_id
        self.last_catalog_status = {"ready": bool(compiled), "reason": "ready" if compiled else "no_informative_templates",
                                    "template_count": len(seen), "variant_count": len(compiled), "skipped": skipped}
        return bool(compiled)

    @staticmethod
    def _compile(template_id, bgra, width, height):
        output = []
        for scale in SCALES:
            w, h = round(width * scale), round(height * scale)
            if min(w, h) < 8 or max(w, h) > MAX_IMAGE_SIDE:
                continue
            resized = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA if w < bgra.shape[1] else cv2.INTER_LINEAR)
            alpha = resized[..., 3].astype(np.float32) / 255.
            foreground = alpha >= .8
            if np.count_nonzero(foreground) < 24:
                continue
            # Transparent holes are part of the artwork's spatial signature.
            # Trim distant padding, but retain a small surrounding context.
            yy, xx = np.where(alpha > .05)
            if not len(xx):
                continue
            left, top = max(0, int(xx.min()) - 1), max(0, int(yy.min()) - 1)
            right, bottom = min(w, int(xx.max()) + 2), min(h, int(yy.max()) + 2)
            bgr = resized[..., :3].astype(np.float32) * alpha[..., None]
            chroma = _features(bgr, "chroma")[top:bottom, left:right]
            mode = "chroma" if _energy(chroma) >= 36. else "gray"
            full_pixels = _features(bgr, mode)
            unblurred = full_pixels[top:bottom, left:right]
            if min(unblurred.shape[:2]) < 6 or not _complex_pattern(unblurred):
                continue
            if mode == "gray" and _energy(unblurred) < 100.:
                continue
            foreground = foreground[top:bottom, left:right].copy()
            foreground.flags.writeable = False
            ink_chroma = _features(resized[..., :3], "chroma")[top:bottom, left:right][foreground].mean(axis=0)
            # Small fixed point-spread alternatives model an optically sampled
            # screen rather than assuming the camera sees a perfectly crisp
            # renderer texture. They do not change the visible-pattern gate.
            for sigma in dict.fromkeys((0., min(.6, .04 * min(w, h)), min(1.2, .06 * min(w, h)))):
                pixels = (cv2.GaussianBlur(full_pixels, (0, 0), sigma, borderType=cv2.BORDER_CONSTANT)
                          if sigma else full_pixels)[top:bottom, left:right]
                if pixels.ndim == 2:
                    pixels = pixels[..., None]
                if min(pixels.shape[:2]) < 6 or not _complex_pattern(pixels):
                    continue
                # Neutral art must retain substantial internal spatial detail.
                if mode == "gray" and _energy(pixels) < 100.:
                    continue
                pixels = np.ascontiguousarray(pixels, dtype=np.float32)
                pixels.flags.writeable = False
                output.append(_Template(template_id, pixels, mode, w, h, left, top,
                                        max(2, min(6, math.ceil(.1 * min(w, h)))), foreground, ink_chroma, sigma))
        return output

    @staticmethod
    def _color_supported(template, bgr):
        ink = _features(bgr, "chroma")[template.foreground]
        if not len(ink):
            return False
        observed = ink.mean(axis=0)
        expected = template.mean_ink_chroma
        reference_strength = float(np.linalg.norm(expected))
        observed_strength = float(np.linalg.norm(observed))
        if template.mode == "gray":
            return float(np.sqrt(np.mean(ink * ink))) <= 12.
        if reference_strength < 12.:
            return True
        return (observed_strength >= max(6., .05 * reference_strength)
                and float(expected @ observed) / (reference_strength * observed_strength) >= .94)

    @staticmethod
    def _full_visible(template, patch):
        """A high average match must also explain each part of the object."""
        centered_t = template - template.mean(axis=(0, 1), keepdims=True)
        centered_p = patch - patch.mean(axis=(0, 1), keepdims=True)
        denominator = float(np.sum(centered_t * centered_t))
        if denominator <= 1e-6:
            return False, 0.
        gain = float(np.sum(centered_t * centered_p) / denominator)
        if not .1 <= gain <= 3.:
            return False, 0.
        error = centered_p - gain * centered_t
        scale = max(6., float(np.sqrt(np.mean(centered_p * centered_p))))
        per_pixel = np.sqrt(np.mean(error * error, axis=2))
        coverage = float(np.mean(per_pixel <= .6 * scale))
        if coverage < .95:
            return False, coverage
        for rows in np.array_split(np.arange(template.shape[0]), 3):
            for columns in np.array_split(np.arange(template.shape[1]), 3):
                tile = error[np.ix_(rows, columns)]
                if float(np.sqrt(np.mean(tile * tile))) > .4 * scale:
                    return False, coverage
        return True, coverage

    def filter_candidates(self, board, candidates, excluded_mask=None):
        if (not isinstance(board, np.ndarray) or board.dtype != np.uint8
                or board.ndim != 3 or board.shape[2] != 3):
            raise ValueError("Artwork matching needs a uint8 BGR board image")
        if excluded_mask is not None and (not isinstance(excluded_mask, np.ndarray)
                or excluded_mask.shape != board.shape[:2] or excluded_mask.ndim != 2):
            raise ValueError("Artwork matching exclusion mask must match the board")
        items = list(candidates)
        diagnostics = {**self.last_catalog_status, "policy": "whole-artwork-pattern-v1", "catalogId": self.catalog_id,
                       "evaluated_candidates": 0, "comparisons": 0, "rejected": [], "observations": [], "budget_exhausted": False}
        if not self._templates:
            return items, diagnostics
        kept = []
        height, width = board.shape[:2]
        for index, candidate in enumerate(items):
            values = [candidate.get(key) for key in ("cx", "cy", "radius")] if isinstance(candidate, dict) else []
            if len(values) != 3 or not all(_number(value) for value in values) or values[2] <= 0:
                kept.append(candidate)
                continue
            x, y, radius = values
            best_score, match = -1., None
            diagnostics["evaluated_candidates"] += 1
            for template in self._templates:
                # A visible avatar inside a larger physical ring does not
                # explain that ring's boundary and cannot veto its presence.
                diameter = 2 * radius
                if not .65 * max(template.full_width, template.full_height) <= diameter <= 1.25 * max(template.full_width, template.full_height):
                    continue
                if diagnostics["comparisons"] >= MAX_COMPARISONS:
                    diagnostics["budget_exhausted"] = True
                    break
                th, tw = template.pixels.shape[:2]
                left = math.floor(x - template.full_width / 2 + template.left) - template.offset
                top = math.floor(y - template.full_height / 2 + template.top) - template.offset
                right, bottom = left + tw + 2 * template.offset + 1, top + th + 2 * template.offset + 1
                if left < 0 or top < 0 or right > width or bottom > height:
                    continue
                roi = board[top:bottom, left:right]
                observed = _features(roi, template.mode)
                scores = cv2.matchTemplate(observed, template.pixels, cv2.TM_CCOEFF_NORMED)
                diagnostics["comparisons"] += 1
                _minimum, score, _minimum_location, location = cv2.minMaxLoc(scores)
                if not math.isfinite(score):
                    continue
                best_score = max(best_score, score)
                if score < (.96 if template.mode == "gray" else .94):
                    continue
                px, py = location
                if excluded_mask is not None and np.any(excluded_mask[top + py:top + py + th, left + px:left + px + tw]):
                    continue
                patch = observed[py:py + th, px:px + tw]
                if not self._color_supported(template, roi[py:py + th, px:px + tw]):
                    continue
                visible, coverage = self._full_visible(template.pixels, patch)
                if not visible:
                    continue
                match = {"candidate": index, "templateId": template.template_id, "score": float(score),
                         "pattern_coverage": coverage, "reason": "strong_exposed_artwork_match",
                         "blur_sigma": template.blur_sigma,
                         "match_bounds": [left + px, top + py, tw, th]}
                break
            if match is None:
                kept.append(candidate)
                diagnostics["observations"].append({"candidate": index, "reason": "unknown",
                                                      "best_score": max(0., float(best_score))})
            else:
                diagnostics["rejected"].append(match)
        return kept, diagnostics
