"""Complementary proposals with one trained physical-ring acceptance rule.

Pale hollow rings need annular edge proposals; a base whose inner ring is hidden
by the miniature may still have a useful partial color boundary. This adapter
preserves both measurement families, selects an existing supported measurement
within each compatible circle cluster, and reports which family supplied it.
The chosen center is retained, but its allowance encloses every accepted
measurement's center and existing allowance in that cluster. This engineering
envelope accounts for disagreement; it is not a calibrated confidence interval.

The two proposal families do not independently establish presence. After
initialization every proposal must pass the same measured diameter, visible
ring-band geometry and starting-ring chroma checks. Compatible starting labels
are evidence constraints, not identities inferred from color. Screen artwork
which genuinely matches every observed feature remains ambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

import cv2
import numpy as np

from annular_ring_detector import AnnularRingDetector
from poc_ring_detector import RingDetector


class HybridRingDetector:
    def __init__(self, grid_px):
        self.grid_px = float(grid_px)
        self.warm = RingDetector(self.grid_px)
        self.annular = AnnularRingDetector(self.grid_px)
        self.last_foreground_mask = None
        self.last_obstruction_mask = None
        self.last_diagnostics = {}
        self._ring_models = {}
        self._verification_initialized = False
        self._named_profiles = False
        self.missing_profile_ids = []
        self.profiles_ready = False

    def initialize(self, board, positions, excluded_mask=None):
        positions = dict(positions) if hasattr(positions, "items") else list(positions)
        self.annular.initialize(board, positions, excluded_mask=excluded_mask)
        self._named_profiles = hasattr(positions, "items")
        starts = dict(positions) if self._named_profiles else dict(enumerate(positions))
        lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB).astype(np.float32)
        mask = self.annular._inputs(board, excluded_mask)
        # Training must see each source's measured boundary. Loose arbitration
        # can otherwise discard a valid outer ring in favor of a precise inner
        # body fit before any profile exists to verify the alternatives.
        proposals = [*self.warm.detect(board, excluded_mask),
                     *self.annular.detect(board, excluded_mask)]
        previous_models = self._ring_models
        self._ring_models = {key: model for key, model in previous_models.items() if key in starts}
        expected = self.annular.expected_radius
        for key, (x, y) in starts.items():
            if key in self._ring_models:
                continue
            nearby = [candidate for candidate in proposals
                      if math.hypot(candidate["cx"] - x, candidate["cy"] - y) <= .4 * self.grid_px
                      and .82 * expected <= candidate["radius"] <= 1.18 * expected]
            for candidate in sorted(nearby, key=self._quality):
                # Hough's pooled radius is only a proposal hint. Prefer the
                # measured outer diameter when its actual band verifies; a
                # biased partial-color fit can instead verify the pooled size.
                for radius in dict.fromkeys((candidate["radius"], expected)):
                    features = self._ring_features(lab, mask, {**candidate, "radius": radius})
                    if features is None or not self._ring_geometry(features):
                        continue
                    # Median chroma ignores exposure changes and projecting
                    # bodies. Most of the visible annular band must agree.
                    chroma = features["band"][features["valid"], 1:]
                    center = np.median(chroma, axis=0)
                    distances = np.linalg.norm(chroma - center, axis=1)
                    tolerance = float(np.clip(np.percentile(distances, 75) + 8., 14., 24.))
                    if np.mean(distances <= tolerance) < .65:
                        continue
                    self._ring_models[key] = {"radius": float(radius),
                                              "chroma": center, "tolerance": tolerance}
                    break
                if key in self._ring_models:
                    break
        self.missing_profile_ids = [key for key in starts if key not in self._ring_models]
        self.profiles_ready = not self.missing_profile_ids
        self._verification_initialized = True

    def _ring_features(self, lab, mask, candidate):
        """Sample a small polar neighborhood, independent of proposal source."""
        x, y, radius = (float(candidate[key]) for key in ("cx", "cy", "radius"))
        height, width = mask.shape
        if radius <= 0 or min(x, y, width - x, height - y) < 1.28 * radius:
            return None
        sample = lambda values: self.annular._samples(lab, x, y, np.asarray(values) * radius)
        band = sample((.82, .90, .98)).mean(axis=1)
        inside = sample((.68, .72, .76)).mean(axis=1)
        outside = sample((1.10, 1.18, 1.26)).mean(axis=1)
        excluded = self.annular._samples(mask, x, y, np.linspace(.55, 1.26, 12) * radius,
                                           interpolation=cv2.INTER_NEAREST)
        valid = ~np.any(excluded > 0, axis=1)
        if np.mean(valid) < .75:
            return None
        outer, inner = outside - band, band - inside
        outer_energy = np.linalg.norm(outer, axis=1)
        inner_energy = np.linalg.norm(inner, axis=1)
        paired = (outer_energy >= 10) & (inner_energy >= 10)
        opposition = np.sum(outer * inner, axis=1) / (outer_energy * inner_energy + 1e-6)
        return {"band": band, "valid": valid,
                "outer_support": float(np.mean(outer_energy[valid] >= 10)),
                "inner_support": float(np.mean(inner_energy[valid] >= 10)),
                "paired_support": float(np.mean(paired[valid])),
                "opposed_support": float(np.mean((paired & (opposition < -.25))[valid]))}

    @staticmethod
    def _ring_geometry(features):
        # Partial occlusion is allowed, but the visible evidence still needs a
        # band with an outside and inside boundary. A filled colored disk is
        # not enough, even if its outer contour contains a circular arc.
        return (features["outer_support"] >= .65
                and features["inner_support"] >= .45
                and features["paired_support"] >= .40
                and features["opposed_support"] >= .35)

    @staticmethod
    def _chroma_support(chroma, model):
        """Preserve color direction while allowing exposure-related saturation."""
        reference = model["chroma"] - 128.
        strength = float(reference @ reference)
        if strength < 144.:
            # Neutral rings have no stable hue direction to normalize.
            return float(np.mean(np.linalg.norm(chroma - model["chroma"], axis=1) <= model["tolerance"]))
        observed = chroma - 128.
        scale = observed @ reference / strength
        distance = np.linalg.norm(observed - scale[:, None] * reference, axis=1)
        return float(np.mean((scale >= .5) & (scale <= 1.75) & (distance <= model["tolerance"])))

    def _verify_ring(self, lab, mask, candidate):
        radius = float(candidate["radius"])
        possible = {key: model for key, model in self._ring_models.items()
                    if .82 * model["radius"] <= radius <= 1.18 * model["radius"]}
        if not possible:
            return None, "diameter_mismatch"
        samples = {}
        support = {}
        best = None
        available = geometry = False
        for key, model in possible.items():
            measured_radius = model["radius"]
            if measured_radius not in samples:
                samples[measured_radius] = self._ring_features(lab, mask, {**candidate, "radius": measured_radius})
            features = samples[measured_radius]
            if features is None:
                continue
            available = True
            if not self._ring_geometry(features):
                continue
            geometry = True
            chroma = features["band"][features["valid"], 1:]
            fraction = self._chroma_support(chroma, model)
            if fraction < .65:
                continue
            support[key] = fraction
            quality = (fraction, features["opposed_support"], features["paired_support"])
            if best is None or quality > best[0]:
                best = (quality, measured_radius, features)
        compatible = list(support)
        if not compatible:
            return None, ("ring_color_mismatch" if geometry else
                          "ring_band_geometry" if available else "ring_band_unavailable")
        _quality, measured_radius, features = best
        item = deepcopy(candidate)
        item["proposal_radius"] = item["radius"]
        item["radius"] = measured_radius
        item["ring_verified"] = True
        if self._named_profiles:
            item["compatible_minis"] = compatible
        item["ring_verification"] = {"policy": "measured-band-v1",
            **{key: value for key, value in features.items() if key not in {"band", "valid"}},
            "appearance_support": max(support.values()),
            "compatible_profile_count": len(compatible)}
        return item, None

    @staticmethod
    def _quality(item):
        # Positional uncertainty is primary. Boundary support resolves equal
        # uncertainty, rather than comparing confidence scores across families
        # as if they were calibrated probabilities.
        return (float(item.get("uncertainty_px", float("inf"))),
                -float(item.get("support_coverage", 0)),
                -float(item.get("confidence", 0)))

    def _compatible(self, left, right):
        return (math.hypot(left["cx"] - right["cx"], left["cy"] - right["cy"]) <= .4 * self.grid_px
                and abs(left["radius"] - right["radius"]) <= .5 * self.grid_px)

    @staticmethod
    def _uncertainty_envelope(chosen, cluster):
        # The representative is an observed center, not an average. Retaining
        # its optimistic allowance would discard accepted alternative centers.
        # A circle of radius distance + allowance about this center encloses
        # each alternative's existing allowance without claiming independence
        # or statistical calibration of the two measurement families.
        offsets = [math.hypot(item["cx"] - chosen["cx"], item["cy"] - chosen["cy"])
                   for _family, item in cluster]
        envelope = max(offset + float(item["uncertainty_px"])
                       for offset, (_family, item) in zip(offsets, cluster))
        return float(envelope), float(max(offsets))

    def _propose(self, board, excluded_mask=None, verify=None):
        warm = self.warm.detect(board, excluded_mask)
        annular = self.annular.detect(board, excluded_mask)
        measurements = [("color-boundary", item) for item in warm]
        measurements.extend(("annular-boundary", item) for item in annular)
        if verify is not None:
            # A permissive inner-body fit must not win arbitration and discard
            # a correctly measured outer ring. Only independently verified
            # physical measurements can affect the representative or allowance.
            measurements = [(family, checked) for family, item in measurements
                            if (checked := verify(item)) is not None]
        measurements.sort(key=lambda pair: (*self._quality(pair[1]), pair[1]["cy"], pair[1]["cx"], pair[0]))
        clusters = []
        for family, item in measurements:
            compatible = [(math.hypot(item["cx"] - cluster[0][1]["cx"],
                                     item["cy"] - cluster[0][1]["cy"]), cluster)
                          for cluster in clusters if self._compatible(item, cluster[0][1])]
            if compatible:
                # Compare with the selected representative, never transitively
                # merge two objects through a lower-quality bridge candidate.
                min(compatible, key=lambda pair: pair[0])[1].append((family, item))
            else:
                clusters.append([(family, item)])
        output = []
        for cluster in clusters:
            family, chosen = cluster[0]
            candidate = deepcopy(chosen)
            candidate["chosen_measurement_source"] = family
            candidate["measurement_sources"] = sorted({source for source, _item in cluster})
            candidate["source_measurements"] = [
                {"family": source,
                 **{key: deepcopy(item.get(key)) for key in (
                     "cx", "cy", "radius", "uncertainty_px", "support_coverage", "confidence", "source",
                     "compatible_minis", "ring_verification") if key in item}}
                for source, item in cluster]
            if self._named_profiles and verify is not None:
                candidate["compatible_minis"] = list(dict.fromkeys(
                    mini for _family, item in cluster for mini in item.get("compatible_minis", [])))
            envelope, disagreement = self._uncertainty_envelope(chosen, cluster)
            candidate["selected_uncertainty_px"] = float(chosen["uncertainty_px"])
            candidate["measurement_center_disagreement_px"] = disagreement
            candidate["uncertainty_px"] = envelope
            candidate["uncertainty_basis"] = "accepted-measurement-envelope"
            output.append(candidate)
        output.sort(key=lambda item: (item["cy"], item["cx"]))
        return output

    def detect(self, board, excluded_mask=None):
        rejected = []
        if self._verification_initialized:
            lab = cv2.cvtColor(board, cv2.COLOR_BGR2LAB).astype(np.float32)
            mask = self.annular._inputs(board, excluded_mask)
            def verify(candidate):
                verified, reason = self._verify_ring(lab, mask, candidate)
                if verified is None:
                    rejected.append({"cx": candidate["cx"], "cy": candidate["cy"],
                                     "radius": candidate["radius"], "reason": reason})
                return verified
            output = self._propose(board, excluded_mask, verify)
        else:
            # Untrained observations are proposals only. Production presence
            # must never silently fall back to the old permissive detector.
            self._propose(board, excluded_mask)
            output = []
        raw_count = (self.warm.last_diagnostics.get("candidate_count", 0)
                     + self.annular.last_diagnostics.get("candidate_count", 0))
        shape = board.shape[:2]
        zero = np.zeros(shape, np.uint8)
        self.last_foreground_mask = cv2.bitwise_or(
            self.warm.last_foreground_mask if self.warm.last_foreground_mask is not None else zero,
            self.annular.last_foreground_mask if self.annular.last_foreground_mask is not None else zero)
        self.last_obstruction_mask = (self.warm.last_obstruction_mask
                                      if self.warm.last_obstruction_mask is not None else zero)
        self.last_diagnostics = {"candidate_count": len(output), "proposal_count": raw_count,
                                 "full_board_search": True,
                                 "color_candidates": self.warm.last_diagnostics.get("candidate_count", 0),
                                 "annular_candidates": self.annular.last_diagnostics.get("candidate_count", 0),
                                 "ring_verification": self._verification_initialized,
                                 "trained_profile_count": len(self._ring_models),
                                 "missing_profile_ids": list(self.missing_profile_ids),
                                 "ring_rejections": rejected,
                                 "expanded_uncertainty_candidates": sum(
                                     item["uncertainty_px"] > item["selected_uncertainty_px"] + 1e-9
                                     for item in output),
                                 "color_diagnostics": deepcopy(self.warm.last_diagnostics),
                                 "annular_diagnostics": deepcopy(self.annular.last_diagnostics)}
        return output

    def detect_tracked(self, board, positions, excluded_mask=None, *, full_search=False):
        """Check known neighborhoods, widening immediately if presence is unclear.

        This changes only proposal search area. Every local measurement passes
        the same verifier before arbitration. Callers must periodically request
        a full search to inspect unexplained objects elsewhere on the board,
        and must not learn a whole-board reference from local-only evidence.
        """
        mask = self.annular._inputs(board, excluded_mask)
        if not isinstance(positions, Mapping) or not positions:
            raise ValueError("Tracked searches need named base centers")
        height, width = board.shape[:2]
        for point in positions.values():
            if (not isinstance(point, (tuple, list, np.ndarray)) or len(point) != 2
                    or not all(math.isfinite(float(value)) for value in point)
                    or not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise ValueError("Tracked centers must be inside the board")
        if (full_search or not self.profiles_ready or not self._named_profiles
                or set(positions) - self._ring_models.keys()):
            return self.detect(board, mask)

        # Leave room for the full accepted anchor displacement and the outer
        # ring-band samples. Cropping never supplies missing edge evidence.
        anchor_gate = .45 * self.grid_px
        radius = max(model["radius"] for model in self._ring_models.values())
        padding = math.ceil(1.3 * max(radius, self.annular.expected_radius) + anchor_gate + 4)
        regions = [(max(0, math.floor(x) - padding), max(0, math.floor(y) - padding),
                    min(width, math.ceil(x) + padding + 1), min(height, math.ceil(y) + padding + 1))
                   for x, y in positions.values()]
        searched_pixels = sum((right - left) * (bottom - top) for left, top, right, bottom in regions)
        if searched_pixels >= width * height or any(
                min(right - left, bottom - top) < 8 for left, top, right, bottom in regions):
            return self.detect(board, mask)

        foreground = np.zeros((height, width), np.uint8)
        obstruction = np.zeros_like(foreground)
        measurements, rejected = [], []
        color_count = annular_count = checked_pixels = 0
        for (mini, (anchor_x, anchor_y)), (left, top, right, bottom) in zip(positions.items(), regions):
            crop, crop_mask = board[top:bottom, left:right], mask[top:bottom, left:right]
            checked_pixels += crop.shape[0] * crop.shape[1]
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

            def verify(candidate):
                verified, reason = self._verify_ring(lab, crop_mask, candidate)
                if verified is None:
                    rejected.append({"cx": candidate["cx"] + left, "cy": candidate["cy"] + top,
                                     "radius": candidate["radius"], "reason": reason})
                return verified

            region_candidates = self._propose(crop, crop_mask, verify)
            own_matches = [item for item in region_candidates if mini in item["compatible_minis"]
                           and math.hypot(item["cx"] + left - anchor_x,
                                          item["cy"] + top - anchor_y) <= anchor_gate]
            if len(own_matches) != 1:
                # A departure is a reason to search, never to wait for a later
                # frame or spend work checking the remaining stationary bases.
                output = self.detect(board, mask)
                self.last_diagnostics["local_fallback_reason"] = "anchor_missing_or_ambiguous"
                self.last_diagnostics["local_searched_pixels"] = checked_pixels
                return output
            for candidate in region_candidates:
                candidate["cx"] += left
                candidate["cy"] += top
                for source in candidate["source_measurements"]:
                    source["cx"] += left
                    source["cy"] += top
                measurements.append(candidate)
            color_count += self.warm.last_diagnostics.get("candidate_count", 0)
            annular_count += self.annular.last_diagnostics.get("candidate_count", 0)
            for source in (self.warm, self.annular):
                if source.last_foreground_mask is not None:
                    foreground[top:bottom, left:right] |= source.last_foreground_mask
                if source.last_obstruction_mask is not None:
                    obstruction[top:bottom, left:right] |= source.last_obstruction_mask

        # Overlapping search windows may observe the same physical ring twice.
        # Keep one real measurement with an allowance enclosing both estimates.
        output = []
        for candidate in sorted(measurements, key=self._quality):
            duplicate = next((item for item in output if math.hypot(
                item["cx"] - candidate["cx"], item["cy"] - candidate["cy"]) <= .18 * self.grid_px), None)
            if duplicate is None:
                output.append(candidate)
            else:
                distance = math.hypot(duplicate["cx"] - candidate["cx"], duplicate["cy"] - candidate["cy"])
                duplicate["uncertainty_px"] = max(duplicate["uncertainty_px"], distance + candidate["uncertainty_px"])
                duplicate["compatible_minis"] = list(dict.fromkeys(
                    [*duplicate["compatible_minis"], *candidate["compatible_minis"]]))

        counts = {mini: 0 for mini in positions}
        unambiguous = True
        for candidate in output:
            matches = [mini for mini, (x, y) in positions.items()
                       if mini in candidate["compatible_minis"]
                       and math.hypot(candidate["cx"] - x, candidate["cy"] - y) <= anchor_gate]
            if len(matches) != 1:
                unambiguous = False
            else:
                counts[matches[0]] += 1
        if not unambiguous or any(count != 1 for count in counts.values()):
            output = self.detect(board, mask)
            self.last_diagnostics["local_fallback_reason"] = "anchor_missing_or_ambiguous"
            self.last_diagnostics["local_searched_pixels"] = searched_pixels
            return output

        output.sort(key=lambda item: (item["cy"], item["cx"]))
        self.last_foreground_mask, self.last_obstruction_mask = foreground, obstruction
        self.last_diagnostics = {"candidate_count": len(output), "proposal_count": color_count + annular_count,
            "color_candidates": color_count, "annular_candidates": annular_count,
            "full_board_search": False, "search_regions": len(regions), "searched_pixels": searched_pixels,
            "ring_verification": True, "trained_profile_count": len(self._ring_models),
            "missing_profile_ids": list(self.missing_profile_ids), "ring_rejections": rejected,
            "expanded_uncertainty_candidates": sum(
                item["uncertainty_px"] > item["selected_uncertainty_px"] + 1e-9 for item in output)}
        return output
