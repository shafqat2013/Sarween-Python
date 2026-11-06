"""Conservative base geometry using generated images, without GUI or camera."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from base_geometry import estimate_base_circle
import tests.test_presence_tracking as presence_tests
import v3_tracking as tracking


class BaseGeometryTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(0)

    def estimate(self, image, *args, **kwargs):
        if "color_mask" not in kwargs and isinstance(image, np.ndarray) and image.ndim == 3:
            kwargs["color_mask"] = np.where((image[:, :, 2] > 100) & (image[:, :, 1] < 100), 255, 0).astype(np.uint8)
        return estimate_base_circle(image, *args, **kwargs)

    def test_same_color_body_appendages_in_all_four_directions(self):
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            with self.subTest(direction=direction):
                image = np.full((220, 220, 3), 25, np.uint8)
                mask = np.zeros((220, 220), np.uint8)
                cv2.circle(mask, (110, 110), 20, 255, -1)
                end = (110 + 43 * direction[0], 110 + 43 * direction[1])
                cv2.line(mask, (110, 110), end, 255, 10)
                image[mask > 0] = (0, 0, 230)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = contours[0]
                center, _ = cv2.minEnclosingCircle(contour)
                fit = self.estimate(image, center, cv2.boundingRect(contour), 28)
                self.assertIsNotNone(fit)
                self.assertAlmostEqual(fit.cx, 110, delta=2)
                self.assertAlmostEqual(fit.cy, 110, delta=2)

    def test_partial_round_ring_and_insufficient_arc(self):
        image = np.full((220, 220, 3), 25, np.uint8)
        cv2.ellipse(image, (110, 110), (20, 20), 0, 30, 330, (0, 0, 230), 6)
        fit = self.estimate(image, (110, 110), (86, 86, 49, 49), 28)
        self.assertIsNotNone(fit)
        self.assertAlmostEqual(fit.cx, 110, delta=3)
        self.assertAlmostEqual(fit.cy, 110, delta=3)
        sparse = np.full_like(image, 25)
        cv2.ellipse(sparse, (110, 110), (20, 20), 0, 0, 80, (0, 0, 230), 3)
        self.assertIsNone(self.estimate(sparse, (110, 110), (86, 86, 49, 49), 28))

    def test_competing_centers_abstain(self):
        image = np.full((220, 220, 3), 25, np.uint8)
        for center in ((102, 110), (118, 110)):
            cv2.circle(image, center, 20, (180, 180, 180), 2)
        self.assertIsNone(self.estimate(image, (110, 110), (80, 80, 60, 60), 28,
            color_mask=np.where(image[:, :, 0] > 100, 255, 0).astype(np.uint8)))

    def test_nearby_background_circle_and_grid_cannot_replace_small_color_object(self):
        image = np.full((220, 220, 3), 25, np.uint8)
        for pos in range(0, 220, 28):
            cv2.line(image, (pos, 0), (pos, 219), (90, 90, 90), 1)
            cv2.line(image, (0, pos), (219, pos), (90, 90, 90), 1)
        cv2.circle(image, (137, 110), 20, (230, 230, 230), 2)
        cv2.circle(image, (110, 110), 5, (0, 0, 230), -1)
        self.assertIsNone(self.estimate(image, (110, 110), (105, 105, 11, 11), 28))

    def test_caption_edges_are_unavailable_and_source_pixels_are_unchanged(self):
        image = np.full((220, 220, 3), 25, np.uint8)
        cv2.circle(image, (110, 110), 20, (0, 0, 230), -1)
        before = image.copy()
        fit = self.estimate(image, (110, 110), (90, 90, 41, 41), 28)
        self.assertIsNotNone(fit)
        self.assertIsNone(self.estimate(image, (110, 110), (90, 90, 41, 41), 28,
                                              excluded_mask=np.full((220, 220), 255, np.uint8)))
        np.testing.assert_array_equal(image, before)

    def test_non_finite_or_invalid_inputs_do_not_create_geometry(self):
        image = np.zeros((120, 120, 3), np.uint8)
        for center, bbox, grid in (((float("nan"), 60), (40, 40, 40, 40), 28),
                                   ((60, 60), (40, 40, -1, 40), 28),
                                   ((60, 60), (40, 40, 40, 40), float("inf"))):
            self.assertIsNone(self.estimate(image, center, bbox, grid))
        self.assertIsNone(self.estimate(image[:, :, 0], (60, 60), (40, 40, 40, 40), 28))

    def test_geometry_never_bypasses_weak_color_rejection(self):
        fixture = presence_tests.PresenceTrackingTest()
        profile = fixture.red_profile()
        profile["presence_min_score"] = 1.0
        image = fixture.frame((72, 94, 13))
        with patch.object(tracking, "estimate_base_circle") as fit:
            detections, _state = tracking.detect_minis(fixture.bundle(image), {"red": profile},
                fixture.GRID_PX, prev_state={"red": fixture.settled_state()})
        self.assertIsNone(detections["red"])
        fit.assert_not_called()

    def test_color_outside_search_crop_still_counts_against_circle_ownership(self):
        image = np.full((320, 360, 3), 25, np.uint8)
        cv2.circle(image, (240, 160), 20, (0, 0, 230), -1)
        cv2.line(image, (10, 160), (240, 160), (0, 0, 230), 12)
        # The visible circle is a minority of the complete accepted component.
        # Its distant same-color extension cannot vanish from the denominator.
        self.assertIsNone(self.estimate(image, (240, 160), (4, 140, 257, 41), 28))


if __name__ == "__main__":
    unittest.main()
