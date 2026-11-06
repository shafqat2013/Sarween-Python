"""A separately sampled base appearance supplies geometry, never identity alone."""

import copy
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

import v3_tracking as tracking
import tests.test_presence_tracking as presence_tests


class BaseAppearanceTests(unittest.TestCase):
    GRID = 40.0
    BASE_BGR = (55, 65, 88)
    IDENTITY_BGR = (0, 0, 255)

    def setUp(self):
        cv2.setNumThreads(0)
        self.fixture = presence_tests.PresenceTrackingTest()

    @staticmethod
    def lab(color):
        value = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2Lab)[0, 0]
        return [float(value[0]) * 100 / 255, float(value[1]) - 128,
                float(value[2]) - 128]

    def profile(self):
        return {**self.fixture.red_profile(),
                "base_appearance": {"lab": self.lab(self.BASE_BGR)}}

    def scene(self, rotation=0, *, base=True, identity=True):
        image = np.full((220, 220, 3), 235, np.uint8)
        center = (92.0, 110.0)
        if base:
            cv2.circle(image, (92, 110), 14, self.BASE_BGR, -1)
        if identity:
            # The brightly colored projected body is visibly offset from the
            # desaturated circular support. Its center must not become the base.
            cv2.ellipse(image, (118, 110), (12, 8), 0, 0, 360,
                        self.IDENTITY_BGR, -1)
        for _ in range(rotation):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            center = (219 - center[1], center[0])
        return image, center

    def detect(self, image, profile=None, *, watermark=None, board_ratio=0.0,
               largest_area=0.0, previous=None):
        bundle = self.fixture.bundle(image, watermark_mask=watermark,
                                     board_ratio=board_ratio,
                                     largest_area=largest_area)
        bundle.marker_mode = "viewport"
        return tracking.detect_minis(bundle,
            {"red": self.profile() if profile is None else profile}, self.GRID,
            {"red": self.fixture.settled_state() if previous is None else previous})

    def test_body_projection_in_four_directions_keeps_the_physical_base_center(self):
        for rotation in range(4):
            with self.subTest(rotation=rotation):
                image, center = self.scene(rotation)
                original = image.copy()
                profile = self.profile()
                profile["base_appearance"]["diameter_squares"] = .7
                original_profile = copy.deepcopy(profile)
                with patch.object(tracking, "estimate_base_circle",
                                  side_effect=AssertionError("Configured base must supply its own center")):
                    detected, state = self.detect(image, profile)
                self.assertIsNotNone(detected["red"])
                self.assertLess(math.dist((detected["red"].cx, detected["red"].cy), center), 2)
                self.assertEqual(state["red"]["pending_hits"], 1)
                np.testing.assert_array_equal(image, original)
                self.assertEqual(profile, original_profile)

    def test_neither_cue_can_identify_a_mini_by_itself(self):
        for base, identity in ((True, False), (False, True), (False, False)):
            with self.subTest(base=base, identity=identity):
                image, _ = self.scene(base=base, identity=identity)
                detected, state = self.detect(image)
                self.assertIsNone(detected["red"])
                self.assertEqual(state["red"]["pending_hits"], 0)

    def test_distant_identity_does_not_validate_unrelated_base_artwork(self):
        image, _ = self.scene(identity=False)
        cv2.circle(image, (178, 110), 9, self.IDENTITY_BGR, -1)
        detected, _ = self.detect(image)
        self.assertIsNone(detected["red"])

    def test_identity_does_not_bypass_base_size_and_shape_safeguards(self):
        for shape in ("tiny", "oversized", "thin"):
            with self.subTest(shape=shape):
                image = np.full((220, 220, 3), 235, np.uint8)
                if shape == "thin":
                    cv2.rectangle(image, (45, 107), (145, 113), self.BASE_BGR, -1)
                else:
                    cv2.circle(image, (92, 110), 6 if shape == "tiny" else 50,
                               self.BASE_BGR, -1)
                cv2.circle(image, (118, 100), 7, self.IDENTITY_BGR, -1)
                detected, _ = self.detect(image)
                self.assertIsNone(detected["red"])

    def test_appearance_only_cue_preserves_existing_profile_area_limits(self):
        image, _ = self.scene()
        for setting, value in (("presence_max_area_frac", .30),
                               ("presence_min_area_frac", .50)):
            with self.subTest(setting=setting):
                profile = {**self.profile(), setting: value}
                detected, _ = self.detect(image, profile)
                self.assertIsNone(detected["red"],
                    "A base appearance sample must not override the existing physical size limits")

    def test_known_base_diameter_rejects_oversized_same_appearance_body(self):
        image = np.full((220, 220, 3), 235, np.uint8)
        # This plausible upper-body color component passes the usual area and
        # aspect checks, but cannot be the learned 28-pixel physical base.
        cv2.ellipse(image, (92, 110), (30, 15), 0, 0, 360, self.BASE_BGR, -1)
        cv2.circle(image, (128, 110), 6, self.IDENTITY_BGR, -1)
        legacy, _ = self.detect(image)
        self.assertIsNotNone(legacy["red"])
        profile = self.profile()
        profile["base_appearance"]["diameter_squares"] = .7
        detected, state = self.detect(image, profile)
        self.assertIsNone(detected["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)

    def test_known_base_diameter_allows_visible_partial_components(self):
        for shape in ("small_circle", "short_axis"):
            with self.subTest(shape=shape):
                image = np.full((220, 220, 3), 235, np.uint8)
                if shape == "small_circle":
                    cv2.circle(image, (92, 110), 12, self.BASE_BGR, -1)
                else:
                    # Occlusion can narrow the visible region independently
                    # of the physical base's measured full diameter.
                    cv2.ellipse(image, (92, 110), (19, 10), 0, 0, 360,
                                self.BASE_BGR, -1)
                cv2.circle(image, (128, 110), 6, self.IDENTITY_BGR, -1)
                legacy, _ = self.detect(image)
                self.assertIsNotNone(legacy["red"])
                profile = self.profile()
                profile["base_appearance"]["diameter_squares"] = 1.0
                detected, state = self.detect(image, profile)
                self.assertIsNotNone(detected["red"])
                self.assertTrue(math.isfinite(detected["red"].cx))
                self.assertTrue(math.isfinite(detected["red"].cy))
                self.assertEqual(state["red"]["pending_hits"], 1)

    def test_malformed_base_diameter_fails_closed(self):
        image, _ = self.scene()
        for diameter in (float("nan"), float("inf"), True, False, 0, -1, None):
            with self.subTest(diameter=diameter):
                profile = self.profile()
                profile["base_appearance"]["diameter_squares"] = diameter
                detected, state = self.detect(image, profile)
                self.assertIsNone(detected["red"])
                self.assertEqual(state["red"]["pending_hits"], 0)

    def test_one_identity_between_two_strong_bases_is_ambiguous(self):
        image = np.full((220, 220, 3), 235, np.uint8)
        cv2.circle(image, (92, 110), 13, self.BASE_BGR, -1)
        cv2.circle(image, (128, 110), 13, self.BASE_BGR, -1)
        cv2.circle(image, (110, 88), 6, self.IDENTITY_BGR, -1)
        for last_xy in (None, (92.0, 110.0), (128.0, 110.0)):
            with self.subTest(last_xy=last_xy):
                detected, state = self.detect(image, previous=self.fixture.settled_state(
                    last_xy=last_xy, pending_xy=(92, 110), pending_hits=2))
                self.assertIsNone(detected["red"], "A remembered point cannot resolve current identity ambiguity")
                self.assertEqual(state["red"]["pending_hits"], 0)

    def test_malformed_base_cue_fails_closed_instead_of_using_body_center(self):
        image, _ = self.scene()
        invalid = [None, True, [], "red", {}, {"lab": [20, 5]},
                   {"lab": [float("nan"), 5, 5]},
                   {"lab": [20, 5, float("inf")]},
                   {"lab": self.lab(self.BASE_BGR), "presence_lab_dist": -1},
                   {"lab": self.lab(self.BASE_BGR), "presence_lab_dist": float("nan")}]
        for cue in invalid:
            with self.subTest(cue=cue):
                profile = {**self.fixture.red_profile(), "base_appearance": cue}
                detected, _ = self.detect(image, profile)
                self.assertIsNone(detected["red"])

    def test_hand_and_unsettled_board_gates_still_suppress_configured_base(self):
        image, _ = self.scene()
        for kwargs in ({"board_ratio": .04, "largest_area": 8000},
                       {"board_ratio": .025, "previous": {"change_hist": [.020, .010, .017]}}):
            with self.subTest(kwargs=kwargs):
                detected, state = self.detect(image, **kwargs)
                self.assertIsNone(detected["red"])
                self.assertEqual(state["red"]["pending_hits"], 0)

    def test_watermark_over_base_abstains_and_clear_frame_resumes(self):
        image, center = self.scene()
        original = image.copy()
        watermark = np.zeros(image.shape[:2], np.uint8)
        watermark[105:115, 85:99] = 255
        detected, state = self.detect(image, watermark=watermark)
        self.assertIsNone(detected["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)
        watermark[:] = 0
        detected, _ = self.detect(image, watermark=watermark)
        self.assertIsNotNone(detected["red"])
        self.assertLess(math.dist((detected["red"].cx, detected["red"].cy), center), 2)
        np.testing.assert_array_equal(image, original)

    def test_weak_candidate_cannot_bypass_existing_score_guard(self):
        image, _ = self.scene()
        original_finder = tracking._find_presence_candidates

        def weak_candidates(*args, **kwargs):
            return [replace(candidate, score=.18)
                    for candidate in original_finder(*args, **kwargs)]

        with patch.object(tracking, "_find_presence_candidates", side_effect=weak_candidates):
            detected, state = self.detect(image)
        self.assertIsNone(detected["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)

    def test_weak_near_red_fragment_cannot_establish_ownership_of_a_strong_base(self):
        image, _ = self.scene(identity=False)
        # This is inside the red Lab tolerance but close to its boundary. Its
        # small round footprint scores about .22, below the existing .35 guard.
        cv2.circle(image, (118, 110), 5, (40, 80, 255), -1)
        detected, state = self.detect(image)
        self.assertIsNone(detected["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)

    def test_lab_curve_base_profile_and_processing_scale_keep_full_resolution_center(self):
        image = np.full((1000, 1000, 3), 235, np.uint8)
        cv2.circle(image, (600, 500), 28, self.BASE_BGR, -1)
        cv2.ellipse(image, (652, 500), (24, 16), 0, 0, 360, self.IDENTITY_BGR, -1)
        profile = {**self.fixture.red_profile(), "base_appearance": {
            "lab_curve": [self.lab(self.BASE_BGR)], "presence_lab_dist": 18}}
        bundle = self.fixture.bundle(image)
        bundle.marker_mode = "viewport"
        detected, _ = tracking.detect_minis(bundle, {"red": profile}, 80,
            {"red": self.fixture.settled_state()})
        self.assertIsNotNone(detected["red"])
        self.assertAlmostEqual(detected["red"].cx, 600, delta=2)
        self.assertAlmostEqual(detected["red"].cy, 500, delta=2)

    def test_profile_without_base_cue_retains_legacy_detection_and_refinement(self):
        image = np.full((220, 220, 3), 235, np.uint8)
        cv2.circle(image, (80, 100), 14, self.IDENTITY_BGR, -1)
        with patch.object(tracking, "estimate_base_circle", return_value=None) as fitter:
            detected, _ = self.detect(image, self.fixture.red_profile())
        self.assertIsNotNone(detected["red"])
        self.assertAlmostEqual(detected["red"].cx, 80, delta=1)
        self.assertAlmostEqual(detected["red"].cy, 100, delta=1)
        self.assertEqual(fitter.call_count, 1)


if __name__ == "__main__":
    unittest.main()
