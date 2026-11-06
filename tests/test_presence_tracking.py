import unittest
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import v3_tracking as tracking


class PresenceTrackingTest(unittest.TestCase):
    GRID_PX = 40.0

    @staticmethod
    def red_profile():
        pixel = np.uint8([[[0, 0, 255]]])
        lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2Lab)[0, 0]
        return {
            "lab": [
                float(lab[0]) * 100.0 / 255.0,
                float(lab[1]) - 128.0,
                float(lab[2]) - 128.0,
            ],
            "presence_lab_dist": 20.0,
        }

    @staticmethod
    def bundle(image, *, board_ratio=0.0, largest_area=0.0, watermark_mask=None):
        height, width = image.shape[:2]
        return SimpleNamespace(
            warp_bgr=image,
            mask_warp=None,
            raw_motion_ratio=board_ratio,
            largest_motion_area=largest_area,
            warp_w=width,
            warp_h=height,
            watermark_mask_warp=watermark_mask,
        )

    @staticmethod
    def frame(*circles):
        image = np.full((180, 180, 3), 255, dtype=np.uint8)
        for x, y, radius in circles:
            cv2.circle(image, (x, y), radius, (0, 0, 255), thickness=-1)
        return image

    @staticmethod
    def settled_state(**values):
        state = {"change_hist": [0.0] * (tracking.PRESENCE_SETTLE_FRAMES - 1)}
        state.update(values)
        return state

    def test_stationary_ring_is_detected_without_motion(self):
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((72, 94, 13))),
            {"red": self.red_profile()},
            self.GRID_PX,
            prev_state={"red": self.settled_state()},
        )

        detection = detections["red"]
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.cx, 72.0, delta=1.0)
        self.assertAlmostEqual(detection.cy, 94.0, delta=1.0)
        self.assertEqual(state["red"]["pending_hits"], 1)

    def test_small_same_color_screen_token_is_rejected(self):
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((72, 94, 6))),
            {"red": self.red_profile()},
            self.GRID_PX,
            prev_state={"red": self.settled_state()},
        )

        self.assertIsNone(detections["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)

    def test_hand_obstruction_suppresses_ring_detection(self):
        detections, state = tracking.detect_minis(
            self.bundle(
                self.frame((72, 94, 13)),
                board_ratio=0.04,
                largest_area=8000.0,
            ),
            {"red": self.red_profile()},
            self.GRID_PX,
            prev_state={"red": self.settled_state(last_xy=(72.0, 94.0))},
        )

        self.assertIsNone(detections["red"])
        self.assertEqual(state["red"]["last_xy"], (72.0, 94.0))

    def test_detection_does_not_move_confirmed_anchor(self):
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((130, 130, 13))),
            {"red": self.red_profile()},
            self.GRID_PX,
            prev_state={"red": self.settled_state(last_xy=(35.0, 35.0))},
        )

        self.assertIsNotNone(detections["red"])
        self.assertEqual(state["red"]["last_xy"], (35.0, 35.0))

    def test_pending_location_wins_over_competing_color_blob(self):
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((48, 52, 13), (132, 126, 15))),
            {"red": self.red_profile()},
            self.GRID_PX,
            prev_state={
                "red": self.settled_state(
                    last_xy=None,
                    pending_xy=(48.0, 52.0),
                    pending_hits=2,
                )
            },
        )

        detection = detections["red"]
        self.assertIsNotNone(detection)
        self.assertAlmostEqual(detection.cx, 48.0, delta=1.0)
        self.assertAlmostEqual(detection.cy, 52.0, delta=1.0)
        self.assertEqual(state["red"]["pending_hits"], 3)

    def test_changing_board_must_settle_before_detection(self):
        profile = {"red": self.red_profile()}
        state = {
            "red": {
                "change_hist": [0.020, 0.018, 0.017],
                "pending_xy": (72.0, 94.0),
                "pending_hits": 3,
            }
        }
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((72, 94, 13)), board_ratio=0.019),
            profile,
            self.GRID_PX,
            prev_state=state,
        )
        self.assertIsNone(detections["red"])
        self.assertEqual(state["red"]["pending_hits"], 0)

        for _ in range(tracking.PRESENCE_SETTLE_FRAMES):
            detections, state = tracking.detect_minis(
                self.bundle(self.frame((72, 94, 13)), board_ratio=0.002),
                profile,
                self.GRID_PX,
                prev_state=state,
            )
        self.assertIsNotNone(detections["red"])

    def test_caption_over_ring_abstains_then_resumes_without_changing_pixels(self):
        image = self.frame((72, 94, 13))
        original = image.copy()
        caption = np.zeros(image.shape[:2], np.uint8)
        caption[90:98, 78:84] = 255
        state = {"red": self.settled_state(last_xy=(72.0, 94.0))}
        detections, state = tracking.detect_minis(
            self.bundle(image, watermark_mask=caption), {"red": self.red_profile()},
            self.GRID_PX, state)
        self.assertIsNone(detections["red"])
        self.assertEqual(state["red"]["last_xy"], (72.0, 94.0))
        self.assertEqual(state["red"]["pending_hits"], 0)
        np.testing.assert_array_equal(image, original)
        detections, _ = tracking.detect_minis(
            self.bundle(image, watermark_mask=np.zeros_like(caption)),
            {"red": self.red_profile()}, self.GRID_PX, state)
        self.assertIsNotNone(detections["red"])

    def test_visible_ring_can_move_outside_caption_covering_old_anchor(self):
        image = self.frame((130, 130, 13))
        caption = np.zeros(image.shape[:2], np.uint8)
        caption[30:60, 30:60] = 255
        detections, _ = tracking.detect_minis(
            self.bundle(image, watermark_mask=caption), {"red": self.red_profile()},
            self.GRID_PX, {"red": self.settled_state(last_xy=(45, 45))})
        self.assertIsNotNone(detections["red"])
        self.assertAlmostEqual(detections["red"].cx, 130, delta=1)

    def test_clear_crescent_stays_accepted_but_caption_over_its_missing_side_blocks(self):
        image = self.frame((80, 80, 22))
        cv2.circle(image, (90, 80), 19, (255, 255, 255), -1)
        frame = self.bundle(image)
        frame.marker_mode = "viewport"
        state = {"red": self.settled_state()}
        detections, _ = tracking.detect_minis(frame, {"red": self.red_profile()}, self.GRID_PX, state)
        self.assertIsNotNone(detections["red"])
        caption = np.zeros(image.shape[:2], np.uint8)
        caption[76:84, 94:99] = 255
        frame.watermark_mask_warp = caption
        detections, _ = tracking.detect_minis(frame, {"red": self.red_profile()}, self.GRID_PX, state)
        self.assertIsNone(detections["red"])

    def test_caption_mask_uses_full_warp_coordinates_after_processing_scale(self):
        image = np.full((1000, 1000, 3), 255, np.uint8)
        cv2.circle(image, (600, 500), 26, (0, 0, 255), -1)
        caption = np.zeros(image.shape[:2], np.uint8)
        caption[495:505, 595:605] = 255
        detections, _ = tracking.detect_minis(
            self.bundle(image, watermark_mask=caption), {"red": self.red_profile()}, 80,
            {"red": self.settled_state()})
        self.assertIsNone(detections["red"])
        caption[:] = 0
        caption[245:255, 295:305] = 255
        detections, _ = tracking.detect_minis(
            self.bundle(image, watermark_mask=caption), {"red": self.red_profile()}, 80,
            {"red": self.settled_state()})
        self.assertIsNotNone(detections["red"])
        self.assertAlmostEqual(detections["red"].cx, 600, delta=2)

    def test_weak_fragment_is_not_selected_even_when_it_is_the_only_candidate(self):
        candidate = tracking.ComboDetection("red", 72, 94, 19.2, 198, .18, .156,
                                           (60, 85, 24, 18), (45, 28, 36))
        for override in (None, float("nan"), True, -1, 2):
            profile = self.red_profile()
            if override is not None:
                profile["presence_min_score"] = override
            with self.subTest(override=override), patch.object(
                    tracking, "_find_presence_candidates", return_value=[candidate]):
                detections, _ = tracking.detect_minis(self.bundle(self.frame()), {"red": profile},
                    self.GRID_PX, {"red": self.settled_state()})
                self.assertIsNone(detections["red"])

    def test_weak_near_color_fragment_is_rejected_but_clear_round_ring_is_kept(self):
        fragment = np.full((180, 180, 3), 255, np.uint8)
        color = (40, 80, 255)  # Within the profile's Lab tolerance, near its edge.
        cv2.fillPoly(fragment, [np.array([[65, 70], [84, 70], [84, 77], [78, 77],
                                         [78, 92], [70, 92], [70, 79], [65, 79]])], color)
        profile = self.red_profile()
        state = {"red": self.settled_state()}
        unguarded, _ = tracking.detect_minis(self.bundle(fragment),
            {"red": {**profile, "presence_min_score": 0}}, 28, state)
        self.assertIsNotNone(unguarded["red"])
        self.assertLess(unguarded["red"].score, tracking.PRESENCE_MIN_SCORE)
        guarded, _ = tracking.detect_minis(self.bundle(fragment), {"red": profile}, 28, state)
        self.assertIsNone(guarded["red"])
        ring = np.full_like(fragment, 255)
        cv2.circle(ring, (80, 80), 13, color, -1)
        detections, _ = tracking.detect_minis(self.bundle(ring), {"red": profile}, 28, state)
        self.assertIsNotNone(detections["red"])

    def test_engine_holds_last_cell_during_caption_and_moves_after_it_clears(self):
        from tracking_engine import TrackingContext, TrackingEngine
        frame = self.bundle(self.frame((50, 50, 13)))
        frame.locked, frame.grid_w, frame.grid_h = True, 4, 4
        engine = TrackingEngine(tracking.detect_minis, consensus_n=4, consensus_k=3)
        now = 0.0

        def step():
            nonlocal now
            now += .1
            return engine.step(frame, {"red": self.red_profile()},
                TrackingContext(now, ("scene",), 1, self.GRID_PX,
                                lambda x, y: (int(x // self.GRID_PX), int(y // self.GRID_PX))))

        initial_moves = [move for _ in range(8) for move in step().moves]
        self.assertEqual([move.raw_to for move in initial_moves], ["r1c1"])
        frame.warp_bgr = self.frame((130, 50, 13))
        frame.watermark_mask_warp = np.zeros(frame.warp_bgr.shape[:2], np.uint8)
        frame.watermark_mask_warp[40:60, 120:140] = 255
        self.assertEqual([move for _ in range(8) for move in step().moves], [])
        self.assertEqual(engine.last_emitted["red"], "r1c1")
        frame.watermark_mask_warp[:] = 0
        resumed = [move for _ in range(8) for move in step().moves]
        self.assertEqual([move.raw_to for move in resumed], ["r1c3"])

    def test_caption_does_not_disable_existing_global_hand_obstruction_safeguards(self):
        image = self.frame((72, 94, 13))
        caption = np.zeros(image.shape[:2], np.uint8)
        caption[140:160, 140:160] = 255
        detections, _ = tracking.detect_minis(
            self.bundle(image, board_ratio=.04, largest_area=8000, watermark_mask=caption),
            {"red": self.red_profile()}, self.GRID_PX,
            {"red": self.settled_state(last_xy=(72, 94))})
        self.assertIsNone(detections["red"])

    def test_invalid_caption_coordinates_fail_closed(self):
        detections, state = tracking.detect_minis(
            self.bundle(self.frame((72, 94, 13)), watermark_mask=np.zeros((90, 90), np.uint8)),
            {"red": self.red_profile()}, self.GRID_PX,
            {"red": self.settled_state(last_xy=(72, 94))})
        self.assertIsNone(detections["red"])
        self.assertEqual(state["red"]["last_xy"], (72, 94))

    def test_watermark_status_only_changes_the_camera_preview(self):
        image = self.frame()
        original = image.copy()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[100:105, 80:85] = 255
        bundle = SimpleNamespace(cam_bgr=image, locked=True, watermark_mask_cam=mask)
        preview = tracking._camera_preview(bundle)
        np.testing.assert_array_equal(image, original)
        self.assertFalse(np.shares_memory(image, preview))
        self.assertFalse(np.array_equal(preview, original))
        mask[:] = 0
        np.testing.assert_array_equal(tracking._camera_preview(bundle), original)

    def check_candidates(self, candidates, order):
        def find(_lab, _index, name, *args, **kwargs):
            x, y, score = candidates[name]
            return [tracking.ComboDetection(name, x, y, 1, 500, .9, score,
                                            (0, 0, 25, 25), (50, 30, 30))]
        profiles = {name: {} for name in order}
        state = {name: self.settled_state(last_xy=(10, 10)) for name in profiles}
        with patch.object(tracking, "_find_presence_candidates", find):
            return tracking.detect_minis(self.bundle(self.frame()), profiles, self.GRID_PX, state)

    def test_rejected_candidate_cannot_eliminate_third_mini(self):
        candidates = {"A": (90, 90, .8), "B": (106, 90, .9), "C": (74, 90, .7)}
        for order in permutations(candidates):
            with self.subTest(order=order):
                detections, state = self.check_candidates(candidates, order)
                self.assertEqual({name for name, det in detections.items() if det}, {"B", "C"})
                self.assertEqual(state["A"]["pending_hits"], 0)
                self.assertEqual(state["A"]["last_xy"], (10, 10))

    def test_five_mini_survivors_are_order_independent_including_ties(self):
        candidates = {"A": (40, 40, .8), "B": (40, 40, .8), "C": (110, 40, .7),
                      "D": (40, 110, .7), "E": (110, 110, .7)}
        for order in permutations(candidates):
            detections, _ = self.check_candidates(candidates, order)
            self.assertEqual({name for name, det in detections.items() if det}, {"A", "C", "D", "E"})


if __name__ == "__main__":
    unittest.main()
