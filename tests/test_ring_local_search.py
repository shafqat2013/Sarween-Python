"""Local anchor searches retain the verifier and fall back in the same frame."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector
from tests.test_physical_ring_verification import canvas, ring


POSITIONS = {"left": (100, 130), "right": (320, 130)}


class RingLocalSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def make(self):
        initial = canvas()
        for point in POSITIONS.values():
            ring(initial, point)
        detector = HybridRingDetector(28)
        detector.initialize(initial, POSITIONS)
        self.assertTrue(detector.profiles_ready)
        return detector, initial

    def test_stationary_anchors_use_crops_and_preserve_board_coordinates_and_masks(self):
        detector, initial = self.make()
        original = deepcopy(detector._ring_models)
        detector.detect = Mock(wraps=detector.detect)
        candidates = detector.detect_tracked(initial, POSITIONS)
        detector.detect.assert_not_called()
        self.assertEqual(len(candidates), 2)
        self.assertFalse(detector.last_diagnostics["full_board_search"])
        self.assertLess(detector.last_diagnostics["searched_pixels"], initial.shape[0] * initial.shape[1])
        self.assertEqual(detector.last_foreground_mask.shape, initial.shape[:2])
        self.assertEqual(detector.last_obstruction_mask.shape, initial.shape[:2])
        for point in POSITIONS.values():
            candidate = min(candidates, key=lambda item: np.hypot(item["cx"] - point[0], item["cy"] - point[1]))
            self.assertLess(np.hypot(candidate["cx"] - point[0], candidate["cy"] - point[1]), 3)
            self.assertTrue(candidate["ring_verified"])
            for measurement in candidate["source_measurements"]:
                self.assertLess(np.hypot(measurement["cx"] - point[0], measurement["cy"] - point[1]), 4)
        for key, model in original.items():
            self.assertEqual(detector._ring_models[key]["radius"], model["radius"])
            np.testing.assert_array_equal(detector._ring_models[key]["chroma"], model["chroma"])
        json.dumps(candidates, allow_nan=False)
        json.dumps(detector.last_diagnostics, allow_nan=False)

    def test_departure_broadens_search_immediately_and_returns_real_destination(self):
        detector, initial = self.make()
        moved = canvas()
        ring(moved, (200, 130))
        ring(moved, POSITIONS["right"])
        detector.detect = Mock(wraps=detector.detect)
        candidates = detector.detect_tracked(moved, POSITIONS)
        detector.detect.assert_called_once()
        self.assertTrue(detector.last_diagnostics["full_board_search"])
        self.assertEqual(detector.last_diagnostics["local_fallback_reason"], "anchor_missing_or_ambiguous")
        self.assertLess(detector.last_diagnostics["local_searched_pixels"], initial.shape[0] * initial.shape[1] / 2)
        self.assertTrue(any(np.hypot(item["cx"] - 200, item["cy"] - 130) < 3 for item in candidates))

    def test_global_audit_finds_objects_outside_anchor_neighborhoods(self):
        detector, initial = self.make()
        ring(initial, (210, 130))
        candidates = detector.detect_tracked(initial, POSITIONS, full_search=True)
        self.assertTrue(detector.last_diagnostics["full_board_search"])
        self.assertEqual(len(candidates), 3)

    def test_masked_origin_and_wrong_color_avatar_cannot_use_local_presence_shortcut(self):
        for mode in ("mask", "avatar"):
            with self.subTest(mode=mode):
                detector, image = self.make()
                mask = np.zeros(image.shape[:2], np.uint8)
                if mode == "mask":
                    mask[95:165, 65:135] = 255
                else:
                    image = canvas()
                    ring(image, POSITIONS["right"])
                    cv2.circle(image, POSITIONS["left"], 14, (180, 30, 180), -1)
                candidates = detector.detect_tracked(image, POSITIONS, mask)
                self.assertTrue(detector.last_diagnostics["full_board_search"])
                self.assertFalse(any(np.hypot(item["cx"] - 100, item["cy"] - 130) < 10 for item in candidates))

    def test_nearby_duplicate_anchors_cannot_count_one_circle_twice(self):
        detector, image = self.make()
        candidates = detector.detect_tracked(image, {"left": (100, 130), "right": (102, 130)})
        self.assertTrue(detector.last_diagnostics["full_board_search"])
        self.assertEqual(len(candidates), 2)

    def test_untrained_or_partial_profiles_never_take_the_local_fast_path(self):
        image = canvas()
        detector = HybridRingDetector(28)
        self.assertEqual(detector.detect_tracked(image, POSITIONS), [])
        self.assertTrue(detector.last_diagnostics["full_board_search"])

    def test_search_crops_can_clip_at_board_edges_without_losing_coordinates(self):
        positions = {"left": (35, 130), "right": (430, 130)}
        image = canvas()
        for point in positions.values():
            ring(image, point)
        detector = HybridRingDetector(28)
        detector.initialize(image, positions)
        self.assertTrue(detector.profiles_ready)
        candidates = detector.detect_tracked(image, positions)
        self.assertEqual(len(candidates), 2)
        self.assertFalse(detector.last_diagnostics["full_board_search"])
        for point in positions.values():
            item = min(candidates, key=lambda row: np.hypot(row["cx"] - point[0], row["cy"] - point[1]))
            self.assertLess(np.hypot(item["cx"] - point[0], item["cy"] - point[1]), 3)
        self.assertEqual(detector.last_foreground_mask.shape, image.shape[:2])


if __name__ == "__main__":
    unittest.main()
