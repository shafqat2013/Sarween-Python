"""Image-level contracts for annular boundaries, not colored identity profiles."""

import json
import unittest

import cv2
import numpy as np

from annular_ring_detector import AnnularRingDetector


POINTS = [(65, 65), (195, 65), (325, 65), (65, 235), (195, 235), (325, 235)]


def board(points=POINTS, *, radius=30, inner=21, appendages=False, sticker=False):
    image = np.full((300, 400, 3), (150, 90, 40), np.uint8)
    for x, y in points:
        cv2.circle(image, (x, y), radius, (245, 245, 245), -1)
        cv2.circle(image, (x, y), inner, (150, 90, 40), -1)
        cv2.circle(image, (x, y), 15, (95, 95, 95), -1)
        if appendages:
            cv2.line(image, (x, y), (x - 35, y - 4), (95, 95, 95), 7)
        if sticker:
            cv2.circle(image, (x - 14, y - 12), 4, (255, 255, 255), -1)
    return image


class AnnularDetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def detect(self, image, *, initial=None, positions=POINTS, mask=None):
        detector = AnnularRingDetector(40)
        detector.initialize(image if initial is None else initial, positions, mask)
        return detector, detector.detect(image, mask)

    def assert_centers(self, candidates, expected, error=3):
        self.assertEqual(len(candidates), len(expected), candidates)
        for point in expected:
            self.assertLess(min(np.hypot(c["cx"] - point[0], c["cy"] - point[1])
                                for c in candidates), error)

    def test_six_identical_pale_hollow_rings_need_no_filled_color_component(self):
        detector, candidates = self.detect(board())
        self.assert_centers(candidates, POINTS, error=1)
        self.assertTrue(detector.last_diagnostics["pooled_appearance"])
        self.assertTrue(all(c["inner_support"] > .8 for c in candidates))
        self.assertTrue(all(c["radius"] > 28 for c in candidates))

    def test_thinner_annulus_retains_outer_centers(self):
        _detector, candidates = self.detect(board(radius=25, inner=21))
        self.assert_centers(candidates, POINTS, error=2)
        self.assertTrue(all(23 <= c["radius"] <= 27 for c in candidates))

    def test_head_stickers_and_body_appendages_do_not_become_base_centers(self):
        _detector, candidates = self.detect(board(appendages=True, sticker=True))
        self.assert_centers(candidates, POINTS, error=4)

    def test_proposals_cover_new_locations_without_destination_initialization(self):
        initial = board()
        moved = [*POINTS[:-1], (325, 150)]
        _detector, candidates = self.detect(board(moved), initial=initial)
        self.assert_centers(candidates, moved, error=2)
        self.assertGreater(min(np.hypot(c["cx"] - POINTS[-1][0], c["cy"] - POINTS[-1][1])
                               for c in candidates), 40)

    def test_caption_mask_makes_covered_ring_unavailable_without_repainting(self):
        image = board()
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, POINTS[0], 38, 255, -1)
        before = image.copy()
        _detector, candidates = self.detect(image, mask=mask)
        self.assert_centers(candidates, POINTS[1:], error=2)
        np.testing.assert_array_equal(image, before)

    def test_masked_caption_circles_cannot_train_the_shared_radius(self):
        image = board()
        mask = np.zeros(image.shape[:2], np.uint8)
        for point in POINTS[:3]:
            cv2.circle(image, point, 37, (245, 245, 245), -1)
            cv2.circle(mask, point, 45, 255, -1)
        detector, candidates = self.detect(image, mask=mask)
        self.assertLess(abs(detector.expected_radius - 30), 2)
        self.assert_centers(candidates, POINTS[3:], error=2)

    def test_filled_bright_disks_are_not_enough_without_an_inner_ring_boundary(self):
        initial = board()
        image = np.full_like(initial, (150, 90, 40))
        for point in POINTS:
            cv2.circle(image, point, 30, (245, 245, 245), -1)
        _detector, candidates = self.detect(image, initial=initial)
        self.assertEqual(candidates, [])

    def test_linear_grid_and_small_text_are_not_base_annuli(self):
        image = np.full((300, 400, 3), (150, 90, 40), np.uint8)
        for x in range(0, 400, 40):
            cv2.line(image, (x, 0), (x, 299), (200, 200, 200), 1)
        for y in range(0, 300, 40):
            cv2.line(image, (0, y), (399, y), (200, 200, 200), 1)
        for x, y in POINTS:
            cv2.putText(image, "AB12", (x - 12, y), cv2.FONT_HERSHEY_SIMPLEX, .25,
                        (245, 245, 245), 1, cv2.LINE_AA)
        _detector, candidates = self.detect(image, initial=board())
        self.assertEqual(candidates, [])

    def test_conflicting_texture_edges_do_not_replace_a_coherent_ring(self):
        image = np.full((240, 400, 3), 128, np.uint8)
        cv2.circle(image, (100, 120), 30, (245, 245, 245), -1)
        # These fragments have complete inner/outer circular edge coverage,
        # but alternating bright/dark transitions do not describe one uniform
        # ring boundary. Geometry alone previously accepted this false object.
        for quadrant in range(4):
            shade = 224 if quadrant % 2 else 32
            cv2.ellipse(image, (280, 120), (30, 30), 0,
                        quadrant * 90, (quadrant + 1) * 90,
                        (shade, shade, shade), -1)
        for point in ((100, 120), (280, 120)):
            cv2.circle(image, point, 21, (128, 128, 128), -1)
            cv2.circle(image, point, 15, (95, 95, 95), -1)
        # No trained appearance table: the image-level boundary evidence must
        # distinguish the valid ring from conflicting texture fragments.
        candidates = AnnularRingDetector(40).detect(image)
        self.assert_centers(candidates, [(100, 120)], error=2)

    def test_pooled_ring_evidence_handles_opposing_background_transitions(self):
        initial = np.full((240, 400, 3), 80, np.uint8)
        cv2.circle(initial, (100, 120), 30, (180, 180, 180), -1)
        cv2.circle(initial, (100, 120), 21, (80, 80, 80), -1)
        cv2.circle(initial, (100, 120), 15, (95, 95, 95), -1)
        image = np.full_like(initial, 80)
        image[:, 280:] = 245
        cv2.circle(image, (280, 120), 30, (180, 180, 180), -1)
        cv2.ellipse(image, (280, 120), (21, 21), 0, 90, 270, (80, 80, 80), -1)
        cv2.ellipse(image, (280, 120), (21, 21), 0, -90, 90, (245, 245, 245), -1)
        cv2.circle(image, (280, 120), 15, (95, 95, 95), -1)
        # The new position is never supplied to initialization. Half of the
        # ring is brighter than its background, half darker; color-gradient
        # coherence alone cannot corroborate this valid uniform annulus.
        self.assertEqual(AnnularRingDetector(40).detect(image), [])
        detector = AnnularRingDetector(40)
        detector.initialize(initial, [(100, 120)])
        candidates = detector.detect(image)
        self.assert_centers(candidates, [(280, 120)], error=2)
        self.assertLess(candidates[0]["outer_coherence"], .60)
        self.assertGreaterEqual(candidates[0]["strong_appearance_support"], .75)

    def test_pooled_colors_do_not_rescue_a_filled_textured_disk(self):
        initial = np.full((240, 400, 3), 80, np.uint8)
        cv2.circle(initial, (100, 120), 30, (180, 180, 180), -1)
        cv2.circle(initial, (100, 120), 21, (80, 80, 80), -1)
        image = np.full_like(initial, 80)
        image[:, 280:] = 245
        cv2.circle(image, (280, 120), 30, (180, 180, 180), -1)
        # Inner texture is not a concentric annular boundary. The fitted outer
        # band matches trained ring colors, so appearance must not bypass the
        # existing inner/outer geometry and opposite-boundary checks.
        for y in range(112, 129, 4):
            cv2.line(image, (273, y), (287, y), (80, 80, 80), 1)
        detector = AnnularRingDetector(40)
        detector.initialize(initial, [(100, 120)])
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        samples = detector._samples(lab, 280, 120, np.linspace(.78 * 30, .96 * 30, 5))
        bins = detector._bins(samples)
        support = np.mean(detector._appearance_observed[bins]
                          & (detector._appearance[bins] >= np.log(2)))
        self.assertGreater(support, .95)
        self.assertEqual(detector.detect(image), [])

    def test_unseen_texture_colors_are_not_pooled_ring_evidence(self):
        initial = board(points=[(100, 120)])
        detector = AnnularRingDetector(40)
        detector.initialize(initial, [(100, 120)])
        image = np.full((240, 400, 3), 128, np.uint8)
        for quadrant in range(4):
            shade = 224 if quadrant % 2 else 32
            cv2.ellipse(image, (280, 120), (30, 30), 0,
                        quadrant * 90, (quadrant + 1) * 90,
                        (shade, shade, shade), -1)
        cv2.circle(image, (280, 120), 21, (128, 128, 128), -1)
        cv2.circle(image, (280, 120), 15, (95, 95, 95), -1)
        self.assertEqual(detector.detect(image), [])

    def test_one_pooled_model_is_independent_of_identity_names_and_input_order(self):
        image = board()
        first = AnnularRingDetector(40)
        second = AnnularRingDetector(40)
        first.initialize(image, {f"first-{i}": p for i, p in enumerate(POINTS)})
        second.initialize(image, {f"other-{i}": p for i, p in enumerate(reversed(POINTS))})
        self.assertEqual(first.expected_radius, second.expected_radius)
        np.testing.assert_array_equal(first._appearance, second._appearance)
        np.testing.assert_array_equal(first._appearance_observed, second._appearance_observed)
        self.assertEqual(first.detect(image), second.detect(image))

    def test_measurement_records_are_json_native_and_deterministic(self):
        detector, first = self.detect(board())
        second = detector.detect(board())
        self.assertEqual(json.dumps(first, allow_nan=False), json.dumps(second, allow_nan=False))
        self.assertTrue(all(c["uncertainty_px"] >= 4 for c in first))

    def test_invalid_images_masks_and_starting_positions_are_rejected(self):
        for grid in (0, 5, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                AnnularRingDetector(grid)
        detector = AnnularRingDetector(40)
        for image in (np.zeros((5, 5, 3), np.uint8), np.zeros((300, 400), np.uint8),
                      np.zeros((300, 400, 4), np.uint8)):
            with self.assertRaises(ValueError):
                detector.detect(image)
        with self.assertRaises(ValueError):
            detector.detect(board(), np.zeros((12, 12), np.uint8))
        for positions in ([], [(float("nan"), 4)], [(-1, 30)], [(400, 20)]):
            with self.assertRaises(ValueError):
                detector.initialize(board(), positions)


if __name__ == "__main__":
    unittest.main()
