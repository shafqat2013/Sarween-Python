import unittest
from types import SimpleNamespace

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
    def bundle(image, *, board_ratio=0.0, largest_area=0.0):
        height, width = image.shape[:2]
        return SimpleNamespace(
            warp_bgr=image,
            mask_warp=None,
            raw_motion_ratio=board_ratio,
            largest_motion_area=largest_area,
            warp_w=width,
            warp_h=height,
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


if __name__ == "__main__":
    unittest.main()
