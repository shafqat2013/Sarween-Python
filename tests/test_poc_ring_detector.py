"""Image-level checks for the offline POC's shared, restricted base detector."""
import unittest

import cv2
import numpy as np

from poc_ring_detector import RingDetector


BACKGROUND = (115, 95, 45)
WARM_FIGURE = (35, 65, 105)


def board(width=340, height=240):
    return np.full((height, width, 3), BACKGROUND, np.uint8)


def figure(image, center, *, direction=(-1, 0), radius=20, ring=(20, 40, 210)):
    x, y = center
    cv2.circle(image, center, radius, ring, -1)
    cv2.circle(image, center, radius-3, WARM_FIGURE, -1)
    end = (x+direction[0]*39, y+direction[1]*39)
    cv2.line(image, center, end, WARM_FIGURE, 9)
    cv2.circle(image, end, 3, (245, 245, 245), -1)


class SharedRingDetectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def test_six_identical_figures_are_located_without_identity_profiles(self):
        image = board()
        centers = [(65, 60), (165, 60), (270, 60), (65, 170), (165, 170), (270, 170)]
        for index, center in enumerate(centers):
            figure(image, center, direction=(-1, 0) if index % 2 else (1, 0))
        observations = RingDetector(28).detect(image)
        self.assertEqual(len(observations), 6)
        for x, y in centers:
            self.assertLess(min(np.hypot(row["cx"]-x, row["cy"]-y) for row in observations), 3)
        self.assertTrue(all("identity" not in row for row in observations))

    def test_head_sticker_and_body_direction_do_not_define_base_center(self):
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            with self.subTest(direction=direction):
                image = board(220, 220)
                figure(image, (110, 110), direction=direction)
                observations = RingDetector(28).detect(image)
                self.assertEqual(len(observations), 1)
                self.assertLess(np.hypot(observations[0]["cx"]-110,
                                         observations[0]["cy"]-110), 3)

    def test_disconnected_head_glint_is_not_an_additional_base(self):
        image = board(220, 220)
        figure(image, (110, 110))
        cv2.circle(image, (50, 160), 4, (255, 255, 255), -1)
        observations = RingDetector(28).detect(image)
        self.assertEqual(len(observations), 1)

    def test_two_bases_in_one_foreground_component_remain_distinct(self):
        image = board(220, 220)
        for center in ((85, 110), (125, 110)):
            cv2.circle(image, center, 18, WARM_FIGURE, -1)
        cv2.line(image, (103, 110), (107, 110), WARM_FIGURE, 3)
        observations = RingDetector(28).detect(image)
        self.assertEqual(len(observations), 2)
        for x in (85, 125):
            self.assertLess(min(np.hypot(row["cx"]-x, row["cy"]-110) for row in observations), 3)

    def test_excluded_pixels_cannot_be_used_as_base_evidence(self):
        image = board(220, 220)
        figure(image, (110, 110))
        before = image.copy()
        detector = RingDetector(28)
        self.assertEqual(detector.detect(image, np.full(image.shape[:2], 255, np.uint8)), [])
        np.testing.assert_array_equal(image, before)

    def test_specular_white_disk_with_warm_rim_is_rejected(self):
        image = board(220, 220)
        cv2.circle(image, (110, 110), 24, (90, 145, 210), -1)
        cv2.circle(image, (110, 110), 20, (255, 255, 255), -1)
        self.assertEqual(RingDetector(28).detect(image), [])

    def test_oversized_foreground_is_obstruction_evidence_not_a_base(self):
        image = board(220, 220)
        cv2.rectangle(image, (30, 35), (190, 180), WARM_FIGURE, -1)
        detector = RingDetector(28)
        self.assertEqual(detector.detect(image), [])
        self.assertGreater(np.count_nonzero(detector.last_obstruction_mask), 10000)

    def test_empty_cool_board_has_no_candidates(self):
        self.assertEqual(RingDetector(28).detect(board()), [])

    def test_each_call_depends_only_on_current_image(self):
        image = board(220, 220)
        figure(image, (110, 110))
        detector = RingDetector(28)
        first = detector.detect(image)
        detector.detect(board(220, 220))
        self.assertEqual(detector.detect(image), first)

    def test_invalid_geometry_or_image_is_rejected(self):
        for grid in (float("nan"), float("inf"), 0, 5):
            with self.assertRaises(ValueError):
                RingDetector(grid)
        detector = RingDetector(28)
        for image in (None, np.zeros((200, 200), np.uint8), np.zeros((200, 200, 3), np.float32)):
            with self.assertRaises(ValueError):
                detector.detect(image)
        with self.assertRaises(ValueError):
            detector.detect(board(), np.zeros((3, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
