"""Complementarity and measurement arbitration without identity-specific cues."""

import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector
from poc_ring_detector import RingDetector
from observer_engine import BoardGeometry


def image():
    board = np.full((240, 400, 3), (150, 90, 40), np.uint8)
    # An inner-boundary-occluded warm base: the legacy partial perimeter is
    # useful, while two concentric annular boundaries are absent.
    cv2.circle(board, (100, 120), 20, (35, 75, 180), -1)
    cv2.rectangle(board, (76, 96), (93, 144), (95, 95, 95), -1)
    # A pale hollow ring with no filled warm-component proposal.
    cv2.circle(board, (280, 120), 21, (245, 245, 245), -1)
    cv2.circle(board, (280, 120), 16, (150, 90, 40), -1)
    cv2.circle(board, (280, 120), 10, (95, 95, 95), -1)
    return board


def measurement(x, *, uncertainty=6, coverage=.8, radius=20, y=120):
    return {"cx": float(x), "cy": float(y), "radius": float(radius),
            "uncertainty_px": float(uncertainty), "support_coverage": float(coverage),
            "confidence": .8, "source": "fixture", "partial": coverage < .7}


class HybridDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def test_pale_and_partial_occluded_bases_are_complementary(self):
        board = image()
        detector = HybridRingDetector(28)
        detector.initialize(board, [(100, 120), (280, 120)])
        candidates = detector.detect(board)
        self.assertEqual(len(candidates), 2, candidates)
        left, right = candidates
        self.assertLess(np.hypot(left["cx"] - 100, left["cy"] - 120), 3)
        self.assertEqual(left["chosen_measurement_source"], "color-boundary")
        self.assertLess(left["support_coverage"], .75)
        original = min(RingDetector(28).detect(board),
                       key=lambda item: np.hypot(item["cx"] - 100, item["cy"] - 120))
        self.assertEqual(left["uncertainty_px"], original["uncertainty_px"])
        self.assertLess(np.hypot(right["cx"] - 280, right["cy"] - 120), 3)
        self.assertEqual(right["chosen_measurement_source"], "annular-boundary")
        json.dumps(candidates, allow_nan=False)

    def fake(self, warm, annular):
        detector = HybridRingDetector(28)
        detector.warm.detect = Mock(return_value=warm)
        detector.annular.detect = Mock(return_value=annular)
        return detector

    def test_selected_center_is_retained_but_allowance_encloses_accepted_alternatives(self):
        color = measurement(100, uncertainty=3.5, coverage=.78)
        annular = measurement(106, uncertainty=7, coverage=.95)
        detector = self.fake([color], [annular])
        candidates = detector.detect(image())
        self.assertEqual(len(candidates), 1)
        chosen = candidates[0]
        self.assertEqual((chosen["cx"], chosen["uncertainty_px"]), (100, 13))
        self.assertEqual(chosen["selected_uncertainty_px"], 3.5)
        self.assertEqual(chosen["measurement_center_disagreement_px"], 6)
        self.assertEqual(chosen["uncertainty_basis"], "accepted-measurement-envelope")
        self.assertEqual(chosen["measurement_sources"], ["annular-boundary", "color-boundary"])
        self.assertEqual(len(chosen["source_measurements"]), 2)
        self.assertEqual([item["uncertainty_px"] for item in chosen["source_measurements"]], [3.5, 7])
        self.assertNotIn("measurement_sources", color)

    def test_partial_color_measurement_is_not_discarded_when_annulus_is_absent(self):
        color = measurement(100, uncertainty=7, coverage=.61)
        detector = self.fake([color], [])
        chosen = detector.detect(image())[0]
        self.assertEqual(chosen["cx"], 100)
        self.assertEqual(chosen["uncertainty_px"], 7)
        self.assertEqual(chosen["measurement_sources"], ["color-boundary"])

    def test_equal_uncertainty_prefers_stronger_boundary_support(self):
        detector = self.fake([measurement(100, coverage=.6)], [measurement(104, coverage=.9)])
        chosen = detector.detect(image())[0]
        self.assertEqual(chosen["cx"], 104)
        self.assertEqual(chosen["chosen_measurement_source"], "annular-boundary")

    def test_adjacent_objects_are_not_transitively_merged_by_a_bridge(self):
        detector = self.fake([measurement(100, uncertainty=3), measurement(120, uncertainty=3)],
                             [measurement(110, uncertainty=7)])
        candidates = detector.detect(image())
        self.assertEqual(len(candidates), 2)
        self.assertEqual(sorted(c["cx"] for c in candidates), [100, 120])
        self.assertEqual([item["uncertainty_px"] for item in candidates], [17, 3])

    def test_accepted_family_disagreement_near_grid_border_requires_abstention(self):
        geometry = BoardGeometry({"registration": {"left": 0, "top": 0, "right": 400, "bottom": 240},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
            "gridSize": 28}, (400, 240))
        color = measurement(77, uncertainty=3.36, y=126)
        annular = measurement(84.6, uncertainty=6, y=126)
        self.assertEqual(geometry.describe((77, 126), color["uncertainty_px"])["cell"], "C5")
        chosen = self.fake([color], [annular]).detect(image())[0]
        self.assertEqual((chosen["cx"], chosen["cy"]), (77, 126))
        self.assertAlmostEqual(chosen["uncertainty_px"], 13.6)
        destination = geometry.describe((chosen["cx"], chosen["cy"]), chosen["uncertainty_px"])
        self.assertIsNone(destination["cell"])
        self.assertEqual(destination["cells"], ["C5", "D5"])

    def test_incompatible_family_outlier_is_a_separate_hypothesis_not_an_average(self):
        for outlier in (measurement(130, uncertainty=6), measurement(104, uncertainty=6, radius=40)):
            with self.subTest(outlier=outlier):
                chosen = self.fake([measurement(100, uncertainty=3)], [outlier]).detect(image())
                self.assertEqual(len(chosen), 2)
                self.assertEqual(chosen[0]["cx"], 100)
                self.assertEqual(chosen[0]["uncertainty_px"], 3)
                self.assertEqual(chosen[0]["measurement_sources"], ["color-boundary"])
                self.assertEqual(chosen[1]["cx"], outlier["cx"])
                self.assertEqual(chosen[1]["uncertainty_px"], 6)
                self.assertEqual(chosen[1]["measurement_sources"], ["annular-boundary"])

    def test_shared_initialization_and_caption_masks_are_forwarded(self):
        detector = HybridRingDetector(28)
        detector.annular.initialize = Mock()
        detector.warm.detect = Mock(return_value=[])
        detector.annular.detect = Mock(return_value=[])
        board = image()
        mask = np.zeros(board.shape[:2], np.uint8)
        points = {"arbitrary-label": (100, 120)}
        detector.initialize(board, points, mask)
        detector.annular.initialize.assert_called_once_with(board, points, excluded_mask=mask)
        self.assertEqual(detector.detect(board, mask), [])
        detector.warm.detect.assert_called_once_with(board, mask)
        detector.annular.detect.assert_called_once_with(board, mask)


if __name__ == "__main__":
    unittest.main()
