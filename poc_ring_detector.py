"""Shared, experimental ring/base observations for the offline single-mover POC.

The development setup has warm-lit figures on a predominantly blue/green
screen. Two broad chroma cues propose foreground components, without learning
individual identity profiles. Inscribed-circle peaks nominate base regions;
robust fits to component boundaries refine their centers while ignoring the
projecting miniature bodies and reflective head stickers.

This restricted detector does not establish physical-vs-display segmentation
on arbitrary maps. Warm printed circles can also be candidates. The caller
must associate observations and abstain on missing or ambiguous evidence.
No recording names, starting cells, identities, or expected endpoints are used.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


class RingDetector:
    """Find shared base candidates in a rectified uint8 BGR board image.

    ``uncertainty_px`` is an engineering allowance, not a statistically
    calibrated interval. ``last_obstruction_mask`` marks oversized foreground
    components; it is not a semantic hand classifier and may include artwork.
    """

    def __init__(self, grid_px: float):
        self.grid_px = float(grid_px)
        if not math.isfinite(self.grid_px) or self.grid_px < 6:
            raise ValueError("grid_px must be finite and at least six pixels")
        self.last_obstruction_mask = None
        self.last_foreground_mask = None
        self.last_diagnostics = {}

    def detect(self, image, excluded_mask=None) -> list[dict]:
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3):
            raise ValueError("Ring detection requires a uint8 BGR image")
        height, width = image.shape[:2]
        if min(height, width) < 8:
            raise ValueError("Image is too small for ring detection")
        if excluded_mask is not None and (
                not isinstance(excluded_mask, np.ndarray)
                or excluded_mask.shape != (height, width)
                or excluded_mask.ndim != 2):
            raise ValueError("excluded_mask must match the image dimensions")

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        a, b = lab[:, :, 1], lab[:, :, 2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        # Both families are applied to every object. They never select identity.
        cues = (
            ("lab", ((a > 128) & (b > 125)) | (a > 140)),
            ("hsv", ((hue < 40) | (hue > 135)) & (saturation > 35) & (value > 25)),
        )
        unavailable = excluded_mask > 0 if excluded_mask is not None else None
        obstruction = np.zeros((height, width), np.uint8)
        union = np.zeros((height, width), np.uint8)
        candidates = []
        rejected = {"size": 0, "edge": 0, "brightness": 0, "excluded": 0, "geometry": 0}
        kernel = np.ones((3, 3), np.uint8)
        grid = self.grid_px
        for cue_name, cue in cues:
            mask = np.where(cue, 255, 0).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            if unavailable is not None:
                mask[unavailable] = 0
            union |= mask
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for label in range(1, count):
                x, y, w, h, area = (int(v) for v in stats[label])
                local = labels[y:y+h, x:x+w] == label
                if area > 7 * grid * grid or max(w, h) > 4 * grid:
                    if area > .5 * grid * grid:
                        obstruction[y:y+h, x:x+w][local] = 255
                    rejected["size"] += 1
                    continue
                if (area < .5 * grid * grid or min(w, h) < .66 * grid
                        or max(w, h) / min(w, h) > 2.4):
                    rejected["size"] += 1
                    continue
                if x <= 0 or y <= 0 or x+w >= width or y+h >= height:
                    rejected["edge"] += 1
                    continue
                # Specular TV highlights can leave a warm rim around a white
                # hole in the chroma mask. Reject the whole bright neighborhood
                # before fitting its rim as an apparent base.
                if np.mean(np.min(image[y:y+h, x:x+w], axis=2) > 235) > .15:
                    rejected["brightness"] += 1
                    continue
                component = local.astype(np.uint8)
                distance = cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
                peak_kernel = max(3, int(round(.25 * grid)) | 1)
                maxima = cv2.dilate(distance, np.ones((peak_kernel, peak_kernel), np.uint8))
                peak_y, peak_x = np.where((distance >= .33 * grid) & (distance >= maxima - 1e-5))
                order = sorted(zip(peak_x, peak_y), key=lambda p: (-distance[p[1], p[0]], p[1], p[0]))
                if not order:
                    continue
                contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contour = max(contours, key=len)[:, 0, :].astype(float) + (x, y)
                fits = self._circle_proposals(contour)
                accepted_local = []
                for px, py in order:
                    peak = (x+px, y+py)
                    if any(math.hypot(peak[0]-old[0], peak[1]-old[1]) < .8*grid for old in accepted_local):
                        continue
                    fit = self._fit_near_peak(fits, peak)
                    if fit is None:
                        rejected["geometry"] += 1
                        continue
                    cx, cy, radius, confidence, coverage = fit
                    if any(math.hypot(cx-old[0], cy-old[1]) < 1.8*max(radius, old[2])
                           for old in accepted_local):
                        continue
                    yy, xx = np.mgrid[max(0, int(cy-radius)):min(height, int(cy+radius+1)),
                                       max(0, int(cx-radius)):min(width, int(cx+radius+1))]
                    disk = (yy-cy)**2 + (xx-cx)**2 <= radius*radius
                    if unavailable is not None and np.mean(unavailable[yy, xx][disk]) > .03:
                        rejected["excluded"] += 1
                        continue
                    pixels = image[yy, xx][disk]
                    clipped = float(np.mean(np.min(pixels, axis=1) > 235)) if len(pixels) else 1.
                    if clipped > .08:
                        rejected["brightness"] += 1
                        continue
                    uncertainty = max(2.5, grid * (.25 if coverage < .65 else .12 if coverage < .8 else .09))
                    candidates.append({"cx": cx, "cy": cy, "radius": radius,
                                       "confidence": confidence, "uncertainty_px": uncertainty,
                                       "support_coverage": coverage, "partial": coverage < .7,
                                       "component_area": area, "clipped_fraction": clipped,
                                       "source": cue_name})
                    accepted_local.append((cx, cy, radius))

        output = []
        # High angular support wins when the two cues describe the same base.
        for candidate in sorted(candidates, key=lambda v: (-v["confidence"], -v["radius"], v["cy"], v["cx"])):
            if any(math.hypot(candidate["cx"]-old["cx"], candidate["cy"]-old["cy"])
                   < 1.8*max(candidate["radius"], old["radius"]) for old in output):
                continue
            output.append(candidate)
        output.sort(key=lambda v: (v["cy"], v["cx"]))
        self.last_obstruction_mask = obstruction
        self.last_foreground_mask = union
        self.last_diagnostics = {"candidate_count": len(output), "rejected": rejected,
                                 "obstruction_fraction": float(np.mean(obstruction > 0))}
        return output

    def _circle_proposals(self, xy):
        # The boundary is on a pixel lattice. Every second point retains useful
        # angular support while reducing the number of pairwise trial distances.
        xy = xy[::2]
        if len(xy) < 20:
            return None
        triples = xy[np.random.default_rng(0).integers(0, len(xy), (256, 3))]
        a = triples[:, 0]
        b, c = triples[:, 1]-a, triples[:, 2]-a
        determinant = 2*(b[:, 0]*c[:, 1]-b[:, 1]*c[:, 0])
        usable = np.abs(determinant) > 1e-5
        a, b, c, determinant = a[usable], b[usable], c[usable], determinant[usable]
        bb, cc = np.sum(b*b, axis=1), np.sum(c*c, axis=1)
        offset = np.column_stack(((bb*c[:, 1]-cc*b[:, 1])/determinant,
                                  (b[:, 0]*cc-c[:, 0]*bb)/determinant))
        centers = a+offset
        radii = np.linalg.norm(offset, axis=1)
        usable = (radii >= .43*self.grid_px) & (radii <= self.grid_px)
        centers, radii = centers[usable], radii[usable]
        if not len(centers):
            return None
        vectors = xy[:, None, :]-centers[None, :, :]
        distance = np.linalg.norm(vectors, axis=2)
        inliers = np.abs(distance-radii[None, :]) <= 1.5
        bins = (np.floor((np.arctan2(vectors[:, :, 1], vectors[:, :, 0])+np.pi)
                         *36/(2*np.pi)).astype(int) % 36)
        ids = bins + np.arange(len(centers))[None, :]*36
        histogram = np.bincount(ids[inliers], minlength=len(centers)*36).reshape(-1, 36)
        coverage = np.sum(histogram > 0, axis=1)/36
        density = np.minimum(1., 2*np.sum(inliers, axis=0)/(2*np.pi*radii))
        score = .7*coverage+.3*density
        return centers, radii, score, coverage

    def _fit_near_peak(self, fits, peak):
        if fits is None:
            return None
        centers, radii, scores, coverage = fits
        usable = np.linalg.norm(centers-np.asarray(peak), axis=1) < radii*.8
        valid_scores = np.where(usable, scores, -1)
        best = int(np.argmax(valid_scores))
        if valid_scores[best] < .52 or coverage[best] < .5:
            return None
        cx, cy = centers[best]
        return float(cx), float(cy), float(radii[best]), float(scores[best]), float(coverage[best])
