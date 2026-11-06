"""Independent regression: a map circle must not hijack a colored base."""

import math
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import v3_tracking as tracking
import tests.test_presence_tracking as presence_tests


class BaseGeometryDistractorTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(0)

    @staticmethod
    def scene(rotation, *, circle_offset=35, circle_radius=20):
        image = np.full((220, 220, 3), 25, np.uint8)
        # This pale circle is map artwork behind an elongated same-color mini
        # body. Its center lies inside the color blob's bounding rectangle.
        cv2.circle(image, (90 + circle_offset, 110), circle_radius, (230, 230, 230), 2)
        colored = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(colored, (90, 110), 14, 255, -1)
        cv2.line(colored, (90, 110), (136, 110), 255, 8)
        image[colored > 0] = (0, 0, 255)
        true_center = np.array((90.0, 110.0))
        for _ in range(rotation):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            true_center = np.array((219 - true_center[1], true_center[0]))
        return image, true_center

    def check_scene(self, *, circle_offset, circle_radius):
        fixture = presence_tests.PresenceTrackingTest()
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                image, true_center = self.scene(rotation, circle_offset=circle_offset,
                                                circle_radius=circle_radius)
                original_pixels = image.copy()
                bundle = fixture.bundle(image)
                bundle.marker_mode = "viewport"
                arguments = (bundle, {"red": fixture.red_profile()}, 28)
                previous = {"red": fixture.settled_state()}
                with patch.object(tracking, "estimate_base_circle", return_value=None):
                    baseline, _ = tracking.detect_minis(*arguments, prev_state=previous)
                result, _ = tracking.detect_minis(*arguments, prev_state=previous)
                self.assertIsNotNone(baseline["red"])
                self.assertIsNotNone(result["red"])
                self.assertGreater(baseline["red"].score, .7)
                previous_error = math.dist((baseline["red"].cx, baseline["red"].cy), true_center)
                new_error = math.dist((result["red"].cx, result["red"].cy), true_center)
                self.assertLessEqual(new_error, previous_error + 1,
                    "Unrelated grayscale circle pulled a strong color detection away from its base")
                np.testing.assert_array_equal(image, original_pixels)

    def test_unrelated_circle_inside_body_bbox_cannot_worsen_base_center(self):
        self.check_scene(circle_offset=35, circle_radius=20)

    def test_large_map_circle_containing_majority_color_still_cannot_hijack_base(self):
        # This false circle contains about 71% of the colored body's pixels.
        # A majority-inside-circle check alone cannot establish base identity.
        self.check_scene(circle_offset=25, circle_radius=26)


if __name__ == "__main__":
    unittest.main()
