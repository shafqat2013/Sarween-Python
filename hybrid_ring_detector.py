"""Complementary global color-boundary and annular measurements.

Pale hollow rings need annular edge proposals; a base whose inner ring is hidden
by the miniature may still have a useful partial color boundary. This adapter
preserves both measurement families, selects an existing supported measurement
within each compatible circle cluster, and reports which family supplied it.
It neither averages centers nor reduces an existing uncertainty allowance.

No identities, expected destinations or recording-specific coordinates are
used. Digital circular artwork can still be observed; temporal association and
the experimental background assumptions remain separate concerns.
"""

from __future__ import annotations

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

    def initialize(self, board, positions, excluded_mask=None):
        self.annular.initialize(board, positions, excluded_mask=excluded_mask)

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

    def detect(self, board, excluded_mask=None):
        warm = self.warm.detect(board, excluded_mask)
        annular = self.annular.detect(board, excluded_mask)
        measurements = [("color-boundary", item) for item in warm]
        measurements.extend(("annular-boundary", item) for item in annular)
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
                     "cx", "cy", "radius", "uncertainty_px", "support_coverage", "confidence", "source")}}
                for source, item in cluster]
            output.append(candidate)
        output.sort(key=lambda item: (item["cy"], item["cx"]))
        shape = board.shape[:2]
        zero = np.zeros(shape, np.uint8)
        self.last_foreground_mask = cv2.bitwise_or(
            self.warm.last_foreground_mask if self.warm.last_foreground_mask is not None else zero,
            self.annular.last_foreground_mask if self.annular.last_foreground_mask is not None else zero)
        self.last_obstruction_mask = (self.warm.last_obstruction_mask
                                      if self.warm.last_obstruction_mask is not None else zero)
        self.last_diagnostics = {"candidate_count": len(output), "color_candidates": len(warm),
                                 "annular_candidates": len(annular),
                                 "color_diagnostics": deepcopy(self.warm.last_diagnostics),
                                 "annular_diagnostics": deepcopy(self.annular.last_diagnostics)}
        return output
