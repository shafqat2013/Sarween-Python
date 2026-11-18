"""All proposal sources must pass the same physical-ring evidence checks."""
import json
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector


BACKGROUND = (150, 90, 40)
YELLOW = (15, 180, 240)
PURPLE = (180, 30, 180)


def canvas():
    return np.full((260, 460, 3), BACKGROUND, np.uint8)


def ring(image, point, *, color=YELLOW, radius=21, inner=16):
    cv2.circle(image, point, radius, color, -1)
    cv2.circle(image, point, inner, BACKGROUND, -1)
    cv2.circle(image, point, 10, (90, 90, 90), -1)


class PhysicalRingVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def detector(self):
        initial = canvas()
        ring(initial, (100, 130))
        detector = HybridRingDetector(28)
        detector.initialize(initial, {"same-mini": (100, 130)})
        self.assertTrue(detector.profiles_ready)
        return detector

    def test_wrong_color_or_smaller_avatar_does_not_keep_old_anchor_occupied(self):
        for radius in (12, 21):
            with self.subTest(radius=radius):
                detector = self.detector()
                moved = canvas()
                # Screen avatar at the old position, physical ring elsewhere.
                cv2.circle(moved, (100, 130), radius, PURPLE, -1)
                cv2.line(moved, (94, 130), (106, 130), (40, 40, 40), 3)
                ring(moved, (320, 130))
                candidates = detector.detect(moved)
                self.assertEqual(len(candidates), 1, candidates)
                self.assertLess(np.hypot(candidates[0]["cx"] - 320, candidates[0]["cy"] - 130), 3)
                self.assertEqual(candidates[0]["compatible_minis"], ["same-mini"])
                self.assertTrue(candidates[0]["ring_verified"])
                json.dumps(candidates, allow_nan=False)

    def test_filled_disk_and_wrong_diameter_ring_fail_even_when_color_matches(self):
        detector = self.detector()
        image = canvas()
        cv2.circle(image, (100, 130), 21, YELLOW, -1)
        cv2.circle(image, (100, 130), 10, (90, 90, 90), -1)
        ring(image, (320, 130), radius=13, inner=10)
        self.assertEqual(detector.detect(image), [])

    def test_ring_of_wrong_color_cannot_borrow_an_unrelated_starting_profile(self):
        detector = self.detector()
        image = canvas()
        ring(image, (320, 130), color=PURPLE)
        self.assertEqual(detector.detect(image), [])
        self.assertTrue(any(item["reason"] == "ring_color_mismatch"
                            for item in detector.last_diagnostics["ring_rejections"]))

    def test_appearance_can_support_multiple_identical_rings_without_assigning_ids(self):
        initial = canvas()
        ring(initial, (100, 130))
        ring(initial, (320, 130))
        detector = HybridRingDetector(28)
        detector.initialize(initial, {"left": (100, 130), "right": (320, 130)})
        candidates = detector.detect(initial)
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(set(item["compatible_minis"]) == {"left", "right"} for item in candidates))

    def test_moderate_exposure_change_preserves_ring_chroma_evidence(self):
        detector = self.detector()
        moved = canvas()
        ring(moved, (320, 130))
        moved = np.clip(moved.astype(float) * .75, 0, 255).astype(np.uint8)
        candidates = detector.detect(moved)
        self.assertEqual(len(candidates), 1, candidates)
        self.assertEqual(candidates[0]["compatible_minis"], ["same-mini"])

    def test_color_proposal_does_not_bypass_the_common_verifier(self):
        detector = self.detector()
        image = canvas()
        ring(image, (320, 130))
        proposal = {"cx": 320., "cy": 130., "radius": 21., "confidence": .9,
                    "uncertainty_px": 3., "support_coverage": .8, "source": "color-only"}
        detector.warm.detect = Mock(return_value=[proposal])
        detector.annular.detect = Mock(return_value=[])
        candidates = detector.detect(image)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["measurement_sources"], ["color-boundary"])
        self.assertTrue(candidates[0]["ring_verified"])
        # Identical proposal geometry over a filled blob must fail.
        cv2.circle(image, (320, 130), 21, YELLOW, -1)
        self.assertEqual(detector.detect(image), [])

    def test_invalid_inner_fit_cannot_discard_a_verified_outer_ring_measurement(self):
        detector = self.detector()
        image = canvas()
        ring(image, (320, 130))
        candidate = {"cx": 320., "cy": 130., "confidence": .9,
                     "support_coverage": .9, "source": "fixture"}
        # Old arbitration selected the more precise but physically wrong
        # inner fit, then threw away the entire cluster on verification.
        detector.warm.detect = Mock(return_value=[{**candidate, "radius": 15., "uncertainty_px": 2.5}])
        detector.annular.detect = Mock(return_value=[{**candidate, "radius": 21., "uncertainty_px": 6.}])
        measured = detector.detect(image)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["measurement_sources"], ["annular-boundary"])
        self.assertEqual(measured[0]["compatible_minis"], ["same-mini"])
        self.assertEqual(measured[0]["uncertainty_px"], 6.)

    def test_training_cannot_discard_a_valid_outer_ring_for_an_inner_fit(self):
        initial = canvas()
        ring(initial, (100, 130))
        detector = HybridRingDetector(28)
        candidate = {"cx": 100., "cy": 130., "confidence": .9,
                     "support_coverage": .9, "source": "fixture"}
        detector.warm.detect = Mock(return_value=[{**candidate, "radius": 15., "uncertainty_px": 2.5}])
        detector.annular.detect = Mock(return_value=[{**candidate, "radius": 21., "uncertainty_px": 6.}])
        detector.annular.initialize = Mock()
        detector.annular.expected_radius = 21.875
        detector.initialize(initial, {"ring": (100, 130)})
        self.assertTrue(detector.profiles_ready)
        self.assertEqual(detector.missing_profile_ids, [])
        self.assertEqual(detector._ring_models["ring"]["radius"], 21.)

    def test_bad_starting_evidence_stays_unready_and_retry_only_learns_missing_profile(self):
        detector = self.detector()
        model = detector._ring_models["same-mini"]
        image = canvas()
        ring(image, (100, 130))
        detector.initialize(image, {"same-mini": (100, 130), "new-mini": (320, 130)})
        self.assertFalse(detector.profiles_ready)
        self.assertEqual(detector.missing_profile_ids, ["new-mini"])
        self.assertIs(detector._ring_models["same-mini"], model)
        ring(image, (320, 130))
        detector.initialize(image, {"same-mini": (100, 130), "new-mini": (320, 130)})
        self.assertTrue(detector.profiles_ready)
        self.assertIs(detector._ring_models["same-mini"], model)

    def test_masked_start_cannot_train_a_profile_or_emit_unverified_presence(self):
        detector = HybridRingDetector(28)
        image = canvas()
        ring(image, (100, 130))
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, (100, 130), 30, 255, -1)
        detector.initialize(image, {"masked": (100, 130)}, mask)
        self.assertFalse(detector.profiles_ready)
        self.assertEqual(detector.detect(image, mask), [])

    def test_uninitialized_detector_never_promotes_loose_proposals_to_presence(self):
        image = canvas()
        ring(image, (100, 130))
        detector = HybridRingDetector(28)
        self.assertEqual(detector.detect(image), [])
        self.assertFalse(detector.profiles_ready)

    def test_measured_outer_band_is_not_replaced_by_a_biased_hough_radius_hint(self):
        image = canvas()
        cv2.circle(image, (200, 130), 30, YELLOW, -1)
        cv2.circle(image, (200, 130), 24, (160, 160, 160), -1)
        detector = HybridRingDetector(40)
        initialize = detector.annular.initialize
        def biased_hint(*args, **kwargs):
            initialize(*args, **kwargs)
            detector.annular.expected_radius = 26.
        detector.annular.initialize = biased_hint
        detector.initialize(image, {"ring": (200, 130)})
        self.assertTrue(detector.profiles_ready)
        self.assertGreater(detector._ring_models["ring"]["radius"], 28.)
        measured = detector.detect(image)
        self.assertEqual(len(measured), 1)
        self.assertEqual(measured[0]["compatible_minis"], ["ring"])


if __name__ == "__main__":
    unittest.main()
