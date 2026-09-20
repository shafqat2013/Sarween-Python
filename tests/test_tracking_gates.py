import unittest
from types import SimpleNamespace

import v3_tracking as tracking


class TrackingGateTest(unittest.TestCase):
    @staticmethod
    def bundle(board_ratio, largest_area):
        return SimpleNamespace(
            raw_motion_ratio=board_ratio,
            largest_motion_area=largest_area,
            warp_w=1280,
            warp_h=720,
        )

    def test_arm_sweep_is_rejected(self):
        self.assertTrue(tracking._has_large_obstruction(self.bundle(0.03452, 27051)))

    def test_scattered_foundry_change_does_not_block_minis(self):
        self.assertFalse(tracking._has_large_obstruction(self.bundle(0.027, 8116)))

    def test_fragmented_hand_motion_is_still_an_obstruction(self):
        self.assertTrue(tracking._has_large_obstruction(self.bundle(0.04, 8000)))

    def test_gray_sample_is_rejected_for_blue_ring(self):
        reason = tracking._calibration_rejection_reason("blue", (71.4, 0.0, -5.0))
        self.assertIn("nearly gray", reason)

    def test_blue_sample_is_accepted_for_blue_ring(self):
        self.assertIsNone(
            tracking._calibration_rejection_reason("blue", (40.0, 18.0, -35.0))
        )

    def test_gray_sample_is_allowed_for_white_ring(self):
        self.assertIsNone(
            tracking._calibration_rejection_reason("white", (80.0, 0.0, -2.0))
        )


if __name__ == "__main__":
    unittest.main()
