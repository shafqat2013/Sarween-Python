"""Experimental global ring-boundary observations, without identity colors.

Unlike the earlier solid-component detector, proposals come from circular image
edges. Verification looks for an outer boundary and a roughly concentric inner
boundary, so a pale annulus need not contain a filled warm disk. Initialization
may pool ONE radius and appearance model from the starting image; labels and
individual color profiles are neither retained nor used.

Circular map artwork can still be a candidate. These measurements alone do not
establish that an object is physical or that changing fog is understood.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import cv2
import numpy as np


class AnnularRingDetector:
    def __init__(self, grid_px):
        self.grid_px = float(grid_px)
        if not math.isfinite(self.grid_px) or self.grid_px < 6:
            raise ValueError("Grid spacing must be finite and at least six pixels")
        self.expected_radius = .75 * self.grid_px
        self._appearance = None
        self._appearance_observed = None
        self.last_foreground_mask = None
        self.last_obstruction_mask = None
        self.last_diagnostics = {}
        self._angles = np.arange(72, dtype=np.float32) * (2 * np.pi / 72)
        self._cos, self._sin = np.cos(self._angles), np.sin(self._angles)

    @staticmethod
    def _inputs(image, excluded_mask):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8):
            raise ValueError("Annular detection needs a uint8 BGR image")
        if excluded_mask is None:
            mask = np.zeros(image.shape[:2], np.uint8)
        elif not isinstance(excluded_mask, np.ndarray) or excluded_mask.shape != image.shape[:2]:
            raise ValueError("Caption mask dimensions must match the board")
        else:
            mask = np.where(excluded_mask > 0, 255, 0).astype(np.uint8)
        return mask

    def _proposals(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        proposals = []
        for channel, name, high, votes in ((gray, "gray", 90, 25),
                                          (lab[:, :, 1], "lab-a", 50, 15),
                                          (lab[:, :, 2], "lab-b", 50, 15)):
            circles = cv2.HoughCircles(cv2.GaussianBlur(channel, (3, 3), 0),
                cv2.HOUGH_GRADIENT, 1.25, max(8, .75 * self.grid_px),
                param1=high, param2=votes,
                minRadius=max(3, round(.55 * self.grid_px)),
                maxRadius=max(4, math.ceil(.96 * self.grid_px)))
            if circles is not None:
                proposals.extend((float(x), float(y), float(radius), name)
                                 for x, y, radius in circles[0])
        return lab, proposals

    @staticmethod
    def _bins(lab):
        quantized = lab.astype(np.int32) // 16
        return quantized[..., 0] * 256 + quantized[..., 1] * 16 + quantized[..., 2]

    def initialize(self, board, positions, excluded_mask=None):
        mask = self._inputs(board, excluded_mask)
        positions = list(positions.values()) if isinstance(positions, Mapping) else list(positions)
        parsed = []
        for point in positions:
            x, y = map(float, point)
            if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < board.shape[1] and 0 <= y < board.shape[0]):
                raise ValueError("Starting centers must be inside the board")
            parsed.append((x, y))
        if not parsed:
            raise ValueError("Supply at least one starting base center")
        lab, proposals = self._proposals(board)
        radii = []
        for x, y in parsed:
            nearby = []
            for cx, cy, r, _source in proposals:
                if (math.hypot(cx - x, cy - y) > .35 * self.grid_px
                        or not .58 * self.grid_px <= r <= .96 * self.grid_px):
                    continue
                excluded = self._samples(mask, cx, cy, np.array((.85 * r, r, 1.1 * r)),
                                          interpolation=cv2.INTER_NEAREST)
                if np.mean(excluded > 0) <= .05:
                    nearby.append(r)
            if nearby:
                radii.append(max(nearby))
        if radii:
            self.expected_radius = float(np.median(radii))
        positive, negative = np.zeros(4096), np.zeros(4096)
        height, width = board.shape[:2]
        for x, y in parsed:
            radius = self.expected_radius
            x0, x1 = max(0, int(x - 1.7 * radius)), min(width, int(x + 1.7 * radius + 1))
            y0, y1 = max(0, int(y - 1.7 * radius)), min(height, int(y + 1.7 * radius + 1))
            yy, xx = np.mgrid[y0:y1, x0:x1]
            distance = np.hypot(xx - x, yy - y)
            valid = mask[y0:y1, x0:x1] == 0
            bins = self._bins(lab[y0:y1, x0:x1])
            ring = (distance >= .65 * radius) & (distance <= 1.03 * radius) & valid
            outside = (distance >= 1.25 * radius) & (distance <= 1.65 * radius) & valid
            positive += np.bincount(bins[ring], minlength=4096)
            negative += np.bincount(bins[outside], minlength=4096)
        # A single pooled likelihood table corroborates geometry. It does not
        # store which start supplied a color, or attach an identity to colors.
        self._appearance_observed = positive > 0
        positive = (positive + .5) / (positive.sum() + 2048)
        negative = (negative + .5) / (negative.sum() + 2048)
        self._appearance = np.log(positive) - np.log(negative)

    def _samples(self, pixels, x, y, radii, *, interpolation=cv2.INTER_LINEAR):
        radii = np.asarray(radii, np.float32)
        mx = (x + self._cos[:, None] * radii[None, :]).astype(np.float32)
        my = (y + self._sin[:, None] * radii[None, :]).astype(np.float32)
        return cv2.remap(pixels, mx, my, interpolation, borderMode=cv2.BORDER_CONSTANT)

    def detect(self, image, excluded_mask=None):
        mask = self._inputs(image, excluded_mask)
        lab, proposals = self._proposals(image)
        # Verify each neighborhood once even when two channels proposed it.
        unique = []
        for proposal in proposals:
            if not any(math.hypot(proposal[0] - old[0], proposal[1] - old[1]) < .35 * self.grid_px
                       for old in unique):
                unique.append(proposal)
        verified = [item for proposal in unique
                    if (item := self._verify_edges(lab, mask, proposal)) is not None]
        output = []
        for item in sorted(verified, key=lambda v: (-v["confidence"], v["cy"], v["cx"])):
            if any(math.hypot(item["cx"] - old["cx"], item["cy"] - old["cy"]) < .5 * self.grid_px
                   for old in output):
                continue
            output.append(item)
        output.sort(key=lambda v: (v["cy"], v["cx"]))
        self.last_foreground_mask = (np.where(self._appearance[self._bins(lab)] > 0, 255, 0).astype(np.uint8)
                                     if self._appearance is not None else np.zeros(mask.shape, np.uint8))
        self.last_obstruction_mask = np.zeros(mask.shape, np.uint8)
        self.last_diagnostics = {"proposal_count": len(proposals), "verified_count": len(verified),
                                 "candidate_count": len(output), "expected_radius": self.expected_radius,
                                 "pooled_appearance": self._appearance is not None}
        return output

    def _verify_edges(self, lab, mask, proposal):
        """Fit observed circular perimeter samples, without a filled-disk seed."""
        px, py, _proposal_radius, source = proposal
        height, width = mask.shape
        radius = self.expected_radius
        if min(px, py, width - px, height - py) <= radius * 1.2 + 2:
            return None
        radii = np.arange(.60 * radius, 1.28 * radius + .5, .5, dtype=np.float32)
        before = self._samples(lab, px, py, radii - 1.25).astype(np.float32)
        after = self._samples(lab, px, py, radii + 1.25).astype(np.float32)
        energy = np.linalg.norm(after - before, axis=2)
        invalid = self._samples(mask, px, py, radii,
                                interpolation=cv2.INTER_NEAREST) > 0
        local = energy >= cv2.dilate(energy, np.ones((1, 5), np.uint8)) - 1e-5
        local &= (energy >= 8) & ~invalid
        points = []
        for angle in range(len(self._angles)):
            peaks = np.where(local[angle])[0]
            selected = []
            for index in sorted(peaks, key=lambda i: -energy[angle, i]):
                if any(abs(radii[index] - radii[other]) < 3 for other in selected):
                    continue
                selected.append(index)
                points.append((px + radii[index] * self._cos[angle],
                               py + radii[index] * self._sin[angle]))
                if len(selected) == 3:
                    break
        if len(points) < 30:
            return None
        xy = np.asarray(points, np.float64)
        triples = xy[np.random.default_rng(0).integers(0, len(xy), (512, 3))]
        a, b, c = triples[:, 0], triples[:, 1] - triples[:, 0], triples[:, 2] - triples[:, 0]
        determinant = 2 * (b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0])
        valid = np.abs(determinant) > 1e-6
        a, b, c, determinant = a[valid], b[valid], c[valid], determinant[valid]
        bb, cc = np.sum(b * b, axis=1), np.sum(c * c, axis=1)
        offset = np.column_stack(((bb * c[:, 1] - cc * b[:, 1]) / determinant,
                                  (b[:, 0] * cc - c[:, 0] * bb) / determinant))
        centers = a + offset
        rs = np.linalg.norm(offset, axis=1)
        valid = ((rs >= .83 * radius) & (rs <= 1.17 * radius)
                 & (np.linalg.norm(centers - (px, py), axis=1) <= .32 * self.grid_px))
        centers, rs = centers[valid], rs[valid]
        if not len(rs):
            return None
        vectors = xy[:, None, :] - centers[None, :, :]
        distances = np.linalg.norm(vectors, axis=2)
        inliers = np.abs(distances - rs[None, :]) <= 1.8
        bins = np.floor((np.arctan2(vectors[:, :, 1], vectors[:, :, 0]) + np.pi)
                        * 36 / (2 * np.pi)).astype(int) % 36
        histogram = np.bincount((bins + np.arange(len(rs))[None, :] * 36)[inliers],
                                minlength=len(rs) * 36).reshape(-1, 36)
        coverage = (histogram > 0).mean(axis=1)
        support = np.minimum(inliers.sum(axis=0) / 72, 1)
        scores = (.8 * coverage + .2 * support - .4 * np.abs(rs / radius - 1)
                  - .15 * np.linalg.norm(centers - (px, py), axis=1) / self.grid_px)
        chosen_models = []
        best_result = None
        for index in np.argsort(-scores):
            if coverage[index] < .70 or scores[index] < .72:
                continue
            center, fitted_radius = centers[index], rs[index]
            if any(np.linalg.norm(center - old_center) < 2 and abs(fitted_radius - old_radius) < 1
                   for old_center, old_radius in chosen_models):
                continue
            chosen_models.append((center, fitted_radius))
            result = self._finish_model(lab, mask, xy, center, fitted_radius,
                                        float(scores[index]), float(coverage[index]), source, (px, py))
            if result is not None and (best_result is None or result["confidence"] > best_result["confidence"]):
                best_result = result
            if len(chosen_models) >= 20:
                break
        return best_result

    def _finish_model(self, lab, mask, xy, center, fitted_radius, geometric_score, coverage, source, proposal):
        radius = self.expected_radius
        px, py = proposal
        for _iteration in range(2):
            good = np.abs(np.linalg.norm(xy - center, axis=1) - fitted_radius) <= 2.0
            chosen = xy[good] - center
            if len(chosen) < 25:
                return None
            solution = np.linalg.lstsq(np.column_stack((2 * chosen, np.ones(len(chosen)))),
                                       np.sum(chosen * chosen, axis=1), rcond=None)[0]
            correction = solution[:2]
            squared = solution[2] + correction @ correction
            if squared <= 0:
                return None
            center = center + correction
            fitted_radius = math.sqrt(squared)
        x, y = center
        if not .83 * radius <= fitted_radius <= 1.17 * radius:
            return None
        outer_radii = fitted_radius + np.array((-1.25, 0., 1.25), np.float32)
        outer_before = self._samples(lab, x, y, outer_radii - 1.25).astype(np.float32)
        outer_after = self._samples(lab, x, y, outer_radii + 1.25).astype(np.float32)
        outer_vectors = outer_after - outer_before
        outer_energy = np.linalg.norm(outer_vectors, axis=2)
        strongest_outer = outer_vectors[np.arange(len(self._angles)), outer_energy.argmax(axis=1)]
        eo = outer_energy.max(axis=1)
        inner_radii = np.linspace(.62 * fitted_radius, .87 * fitted_radius, 8)
        inner_before = self._samples(lab, x, y, inner_radii - 1.25).astype(np.float32)
        inner_after = self._samples(lab, x, y, inner_radii + 1.25).astype(np.float32)
        inner_vectors = inner_after - inner_before
        inner_energy = np.linalg.norm(inner_vectors, axis=2)
        ei = inner_energy.max(axis=1)
        opposite = np.sum(inner_vectors * strongest_outer[:, None, :], axis=2)
        opposite /= inner_energy * eo[:, None] + 1e-6
        opposed = np.any((opposite < -.35) & (inner_energy >= 10), axis=1)
        excluded = self._samples(mask, x, y, np.linspace(.55 * fitted_radius, 1.1 * fitted_radius, 12),
                                 interpolation=cv2.INTER_NEAREST)
        valid = ~np.any(excluded > 0, axis=1)
        if valid.mean() < .85:
            return None
        outer_support = float(np.mean(eo[valid] >= 13))
        inner_support = float(np.mean(ei[valid] >= 10))
        paired = float(np.mean((eo[valid] >= 13) & (ei[valid] >= 10)))
        if outer_support < .72 or inner_support < .45 or paired < .4:
            return None
        appearance = None
        if self._appearance is not None:
            pixels = self._samples(lab, x, y, np.linspace(.78 * fitted_radius, .96 * fitted_radius, 5))
            appearance = float(np.mean(self._appearance[self._bins(pixels)] > 0))
            if appearance < .30:
                return None
        strong_valid = valid & (eo >= 13)
        unit_vectors = strongest_outer / (eo[:, None] + 1e-6)
        coherence = float(np.linalg.norm(np.mean(unit_vectors[strong_valid], axis=0)))
        opposition_support = float(np.mean(opposed[strong_valid]))
        if opposition_support < .45:
            return None
        # Ring-to-background transitions can oppose each other on a changing
        # light/dark background. A strong majority of the measured annular band
        # may instead corroborate the ONE pooled appearance model. Require
        # observed ring colors with at least 2:1 ring/background likelihood:
        # smoothing priors for unseen colors are not physical ring evidence.
        strong_appearance = None
        if self._appearance is not None:
            bins = self._bins(pixels)
            strong_appearance = float(np.mean(self._appearance_observed[bins]
                                              & (self._appearance[bins] >= math.log(2))))
        if coherence < .60 and (strong_appearance is None or strong_appearance < .75):
            return None
        score = .72 * geometric_score + .28 * coherence
        uncertainty = max(6., float(np.linalg.norm(center - (px, py))) + 1.,
                          self.grid_px * (.12 if coverage >= .85 else .22))
        return {"cx": float(x), "cy": float(y), "radius": float(fitted_radius),
                "confidence": score, "uncertainty_px": float(uncertainty),
                "support_coverage": float(coverage), "inner_support": inner_support,
                "paired_support": paired, "valid_fraction": float(valid.mean()),
                "outer_coherence": coherence, "opposition_support": opposition_support,
                "strong_appearance_support": strong_appearance,
                "appearance_support": appearance, "partial": bool(coverage < .8),
                "source": "annular-" + source}
