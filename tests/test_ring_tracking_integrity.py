"""Image-level avatar rejection and conservative background adaptation."""

from copy import deepcopy
import unittest

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector
from observer_engine import BoardGeometry, CONTINUOUS_DISPLAY_POLICY, ObserverEngine
from tests.test_observer_expected_recovery import POSITIONS, VIEW


def pixels(positions, *, avatar=None):
    image = np.full((360, 400, 3), (150, 90, 40), np.uint8)
    if avatar == "identical-ring":
        paint_ring(image, POSITIONS["mini1"])
    elif avatar == "purple-disk":
        cv2.circle(image, POSITIONS["mini1"], 22, (170, 35, 170), -1)
    elif avatar == "warm-disk":
        cv2.circle(image, POSITIONS["mini1"], 30, (35, 75, 180), -1)
    for point in positions.values():
        paint_ring(image, point)
    return image


def paint_ring(image, point):
    point = tuple(map(round, point))
    cv2.circle(image, point, 30, (35, 75, 180), -1)
    cv2.circle(image, point, 24, (160, 160, 160), -1)
    cv2.line(image, point, (point[0] - 43, point[1]), (160, 160, 160), 8)


class RingTrackingIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def engine(self):
        return ObserverEngine(BoardGeometry(VIEW, (400, 360)), POSITIONS,
            detector_factory=HybridRingDetector, display_transition_policy=CONTINUOUS_DISPLAY_POLICY)

    def acquire(self, engine):
        for now in (0, .125, .25):
            row = engine.process(pixels(POSITIONS), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"], row["state"])
        return row

    def test_exposed_wrong_color_or_filled_avatar_cannot_keep_a_departed_mini_present(self):
        for avatar in ("purple-disk", "warm-disk"):
            with self.subTest(avatar=avatar):
                engine = self.engine()
                self.acquire(engine)
                moved = deepcopy(POSITIONS)
                moved["mini1"] = (80, 175)
                for now in (.375, .5, .625, .75, .875):
                    row = engine.process(pixels(moved, avatar=avatar), None, now, scene_visual_revision=0)
                self.assertEqual([item["mini"] for item in engine.events], ["mini1"], row["state"])
                self.assertLess(np.linalg.norm(np.array(engine.events[0]["to_xy"]) - moved["mini1"]), 6)
                self.assertTrue(all(item.get("ring_verified") for item in row["candidates"]))
                self.assertTrue(all(item.get("compatible_minis") for item in row["candidates"]))

    def test_lookalike_stays_uncertain_and_does_not_erase_the_real_new_position(self):
        engine = self.engine()
        initial = self.acquire(engine)
        reference_updates = initial["reference_updates"]
        moved = deepcopy(POSITIONS)
        moved["mini1"] = (80, 175)
        # A deliberately indistinguishable drawn ring cannot be certified as
        # physical by this single-camera appearance check. Preserve ambiguity.
        # Start after the periodic full-board audit is due. Local anchor checks
        # never learn unseen areas into the reference between those audits.
        for now in (1.375, 1.5, 1.625, 1.75):
            row = engine.process(pixels(moved, avatar="identical-ring"), None, now, scene_visual_revision=0)
            self.assertEqual(row["events"], [])
            self.assertEqual(row["state"]["phase"], "uncertain")
            self.assertEqual(row["reference_updates"], reference_updates)
        # When the screen lookalike clears, the physical arrival is still new
        # evidence, rather than an object silently absorbed into the map.
        for now in (1.875, 2., 2.125, 2.25):
            row = engine.process(pixels(moved), None, now, scene_visual_revision=0)
        self.assertEqual([item["mini"] for item in engine.events], ["mini1"], row["state"])
        self.assertLess(np.linalg.norm(np.array(engine.events[0]["to_xy"]) - moved["mini1"]), 6)

    def test_profile_mismatch_cannot_acquire_a_mini_from_its_old_screen_location(self):
        engine = self.engine()
        self.acquire(engine)
        # Acquisition after a reset must train from visible rings; it must not
        # fall back to accepting this filled on-screen disk as a clicked mini.
        missing = deepcopy(POSITIONS)
        del missing["mini1"]
        engine.reset(POSITIONS)
        for now in (0, .125, .25):
            row = engine.process(pixels(missing, avatar="purple-disk"), None, now, scene_visual_revision=0)
        self.assertFalse(row["acquired"])
        self.assertIn("mini1", row["acquisition_missing"])
        self.assertIn("mini1", row["acquisition"]["ring_profiles_missing"])
        self.assertEqual(engine.events, [])

    def test_local_checks_do_not_reseed_areas_outside_their_search(self):
        engine = self.engine()
        initial = self.acquire(engine)
        for now in (.375, .5, .625):
            row = engine.process(pixels(POSITIONS), None, now, scene_visual_revision=0)
            self.assertFalse(row["detector_diagnostics"]["full_board_search"])
            self.assertEqual(row["reference_updates"], initial["reference_updates"])
            self.assertEqual(row["events"], [])
        row = engine.process(pixels(POSITIONS), None, 1.5, scene_visual_revision=0)
        self.assertTrue(row["detector_diagnostics"]["full_board_search"])
        self.assertGreater(row["reference_updates"], initial["reference_updates"])
        self.assertEqual(row["events"], [])

    def test_new_ring_during_acquisition_is_not_silently_seeded_as_background(self):
        engine = self.engine()
        engine.process(pixels(POSITIONS), None, 0, scene_visual_revision=0)
        extra = pixels(POSITIONS)
        paint_ring(extra, (200, 175))
        for now in (.125, .25, .375):
            row = engine.process(extra, None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"])
        self.assertEqual(row["state"]["phase"], "uncertain")
        self.assertEqual(row["state"]["reason"], "extra_candidates_with_full_layout")
        self.assertEqual(row["reference_updates"], 1)
        self.assertTrue(row["state"]["unassigned"])
        self.assertEqual(engine.events, [])


if __name__ == "__main__":
    unittest.main()
