"""Steady, sharp and diverse calibration capture decisions without hardware."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from calibration_capture import CalibrationCaptureGate, CalibrationViewArchive


def checker(width=640, height=480):
    return (((np.arange(height)[:, None] // 8 + np.arange(width)[None, :] // 8) % 2) * 255).astype(np.uint8)


def board(*, offset=(120, 100), scale=60, angle=0, perspective=(0, 0), subset=None):
    ids = np.arange(24, dtype=np.int32) if subset is None else np.asarray(subset, np.int32)
    radians = np.deg2rad(angle)
    cosine, sine = np.cos(radians), np.sin(radians)
    matrix = np.array([[scale * cosine, -scale * sine, offset[0]],
                       [scale * sine, scale * cosine, offset[1]],
                       [perspective[0], perspective[1], 1]], np.float64)
    grid = np.column_stack((ids % 6, ids // 6)).astype(np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(grid, matrix), ids.reshape(-1, 1)


class CalibrationCaptureTests(unittest.TestCase):
    def observe(self, gate, now, *, gray=None, view=None, manual=False):
        gray = checker() if gray is None else gray
        corners, ids = board() if view is None else view
        return gate.observe(now, gray, corners, ids, (gray.shape[1], gray.shape[0]), manual=manual)

    def test_initial_delay_then_steady_hold_accepts_once(self):
        gate = CalibrationCaptureGate()
        self.assertEqual(self.observe(gate, 10)["reason"], "starting")
        self.assertFalse(self.observe(gate, 14.9)["capture"])
        captured = self.observe(gate, 15)
        self.assertTrue(captured["capture"])
        self.assertEqual(captured["count"], 1)
        self.assertFalse(self.observe(gate, 16)["capture"])

    def test_motion_resets_hold_and_slow_drift_cannot_accumulate(self):
        gate = CalibrationCaptureGate(start_delay=0)
        for index in range(8):
            status = self.observe(gate, index * .25, view=board(offset=(120 + index * 3, 100)))
            self.assertFalse(status["capture"])
        held = board(offset=(141, 100))
        self.assertTrue(self.observe(gate, 3, view=held)["capture"])

    def test_missing_board_resets_stability(self):
        gate = CalibrationCaptureGate(start_delay=0)
        self.observe(gate, 0)
        self.observe(gate, .8)
        missing = gate.observe(.9, checker(), None, None, (640, 480))
        self.assertEqual(missing["reason"], "no_board")
        self.assertFalse(self.observe(gate, 1)["capture"])
        self.assertTrue(self.observe(gate, 2)["capture"])

    def test_blur_rejects_manual_capture_and_resets_hold(self):
        gate = CalibrationCaptureGate(start_delay=0)
        self.observe(gate, 0)
        blurred = np.full((480, 640), 128, np.uint8)
        self.assertEqual(self.observe(gate, .8, gray=blurred, manual=True)["reason"], "blur")
        self.assertFalse(self.observe(gate, 1)["capture"])
        self.assertTrue(self.observe(gate, 2)["capture"])

    def test_marker_subset_change_preserves_hold_but_is_not_new_pose(self):
        gate = CalibrationCaptureGate(start_delay=0)
        self.observe(gate, 0)
        subset = board(subset=np.arange(22))
        self.assertFalse(self.observe(gate, .9, view=subset)["capture"])
        self.assertTrue(self.observe(gate, 1, view=subset)["capture"])
        self.assertEqual(self.observe(gate, 3, view=board(subset=np.arange(6, 24)), manual=True)["reason"], "duplicate")

    def test_flashing_watermark_subsets_and_handheld_jitter_capture_one_view(self):
        gate = CalibrationCaptureGate(start_delay=0)
        gray = checker()
        captures = []
        jitter = [(2, -2), (-2, 2), (1, 1), (-1, -1), (0, 0)]
        for frame, now in enumerate(np.linspace(0, 2, 41)):
            dx, dy = jitter[frame % len(jitter)]
            subset = np.arange(24) if frame % 4 < 2 else np.r_[0:6, 18:24]
            status = self.observe(gate, now, gray=gray,
                view=board(scale=30, offset=(120 + dx, 100 + dy), subset=subset))
            if status["capture"]:
                captures.append(status)
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]["holdResets"], 0)
        self.assertGreaterEqual(captures[0]["validCorners"], 10)

    def test_disjoint_valid_subsets_use_agreeing_canonical_board_pose(self):
        gate = CalibrationCaptureGate(start_delay=0)
        for frame, now in enumerate(np.linspace(0, 1, 21)):
            subset = np.arange(12) if frame % 2 == 0 else np.arange(12, 24)
            status = self.observe(gate, now, view=board(subset=subset))
        self.assertTrue(status["capture"])
        self.assertEqual(status["holdResets"], 0)

    def test_isolated_bad_corner_blocks_capture_without_restarting_steady_hold(self):
        gate = CalibrationCaptureGate(start_delay=0)
        for frame, now in enumerate(np.linspace(0, 1, 21)):
            points, ids = board()
            bad = frame % len(ids)
            points[bad, 0] += (45, 30)
            status = self.observe(gate, now, view=(points, ids))
            self.assertFalse(status["capture"])
            self.assertEqual(status["reason"], "corner_outlier")
        self.assertEqual(status["grossCornerOutliers"], 1)
        self.assertTrue(self.observe(gate, 1.05)["capture"])
        self.assertEqual(status["holdResets"], 0)

    def test_real_radial_distortion_is_retained_for_full_and_partial_board_captures(self):
        ids = np.arange(24, dtype=np.int32)
        grid = np.column_stack((ids % 6, ids // 6))
        ideal = grid * [210, 180] + [260, 160]
        normalized = (ideal - [960, 540]) / 650
        for coefficient in (-.2, -.1, .1, .2):
            radius = np.sum(normalized ** 2, axis=1)
            points = normalized * (1 + coefficient * radius)[:, None] * 650 + [960, 540]
            for selected in (np.arange(24), np.arange(12)):
                with self.subTest(radial=coefficient, corners=len(selected)):
                    gate = CalibrationCaptureGate(start_delay=0)
                    status = self.observe(gate, 0, gray=checker(1920, 1080),
                        view=(points[selected].reshape(-1, 1, 2), ids[selected]), manual=True)
                    self.assertTrue(status["capture"], status)
                    self.assertEqual(status["validCorners"], len(selected))
                    self.assertEqual(status["grossCornerOutliers"], 0)
                    self.assertNotIn("inlierIds", status)

    def test_curved_board_subset_flicker_does_not_change_the_physical_pose(self):
        ids = np.arange(24, dtype=np.int32)
        grid = np.column_stack((ids % 6, ids // 6))
        normalized = (grid * [210, 180] + [260, 160] - [960, 540]) / 650
        points = normalized * (1 + .2 * np.sum(normalized ** 2, axis=1))[:, None] * 650 + [960, 540]
        gate = CalibrationCaptureGate(start_delay=0)
        gray = checker(1920, 1080)
        captures = 0
        for frame, now in enumerate(np.linspace(0, 2, 41)):
            selected = np.arange(24) if frame % 2 else np.arange(12)
            status = self.observe(gate, now, gray=gray,
                view=(points[selected].reshape(-1, 1, 2), ids[selected]))
            captures += status["capture"]
        self.assertEqual(captures, 1)

    def test_subset_change_does_not_hide_real_camera_motion(self):
        gate = CalibrationCaptureGate(start_delay=0)
        for frame, now in enumerate(np.linspace(0, 2, 41)):
            subset = np.arange(24) if frame % 2 else np.arange(12)
            status = self.observe(gate, now, view=board(offset=(120 + frame, 100), subset=subset))
            self.assertFalse(status["capture"])
        self.assertGreater(status["holdResets"], 0)
        self.assertEqual(status["lastHoldResetReason"], "motion")

    def test_hold_diagnostics_report_motion_and_quality_resets(self):
        gate = CalibrationCaptureGate(start_delay=0)
        self.observe(gate, 0)
        moving = self.observe(gate, .5, view=board(offset=(140, 100)))
        self.assertTrue(moving["holdReset"])
        self.assertEqual(moving["lastHoldResetReason"], "motion")
        self.assertEqual(moving["sharedCorners"], 24)
        self.assertGreater(moving["motionPixels"], moving["motionLimitPixels"])
        blurry = self.observe(gate, .6, gray=np.full((480, 640), 128, np.uint8))
        self.assertTrue(blurry["holdReset"])
        self.assertEqual(blurry["lastHoldResetReason"], "blur")
        self.assertEqual(blurry["holdResets"], 2)

    def test_cooldown_requires_another_view_and_time_to_reposition(self):
        gate = CalibrationCaptureGate(start_delay=0, hold_seconds=1, cooldown_seconds=2)
        self.observe(gate, 0)
        self.assertTrue(self.observe(gate, 1)["capture"])
        other = board(offset=(220, 100))
        self.assertEqual(self.observe(gate, 1.1, view=other)["reason"], "cooldown")
        self.assertFalse(self.observe(gate, 2.5, view=other)["capture"])
        self.assertTrue(self.observe(gate, 3, view=other)["capture"])

    def test_manual_mode_bypasses_delay_and_hold_but_preserves_quality_and_novelty(self):
        gate = CalibrationCaptureGate(automatic=False)
        self.assertEqual(self.observe(gate, 0)["reason"], "manual")
        self.assertFalse(self.observe(gate, 6)["capture"])
        self.assertTrue(self.observe(gate, 6, manual=True)["capture"])
        self.assertEqual(self.observe(gate, 7, manual=True)["reason"], "duplicate")
        gate.automatic = True
        other = board(offset=(220, 100))
        self.assertFalse(self.observe(gate, 8, view=other)["capture"])
        self.assertTrue(self.observe(gate, 9, view=other)["capture"])

    def test_jitter_near_previously_saved_view_is_not_a_new_capture(self):
        gate = CalibrationCaptureGate(start_delay=0)
        self.assertTrue(self.observe(gate, 0, manual=True)["capture"])
        for now, shift in enumerate((1, -2, 3), start=1):
            status = self.observe(gate, now, view=board(offset=(120 + shift, 100)), manual=True)
            self.assertFalse(status["capture"])
        self.assertEqual(gate.count, 1)

    def test_repeated_parallel_views_require_camera_tilt(self):
        gate = CalibrationCaptureGate(start_delay=0)
        gray = checker(1000, 700)
        positions = [(100, 100), (350, 100), (100, 350)]
        for now, position in enumerate(positions):
            self.assertTrue(self.observe(gate, now, gray=gray, view=board(offset=position), manual=True)["capture"])
        fourth = self.observe(gate, 3, gray=gray, view=board(offset=(350, 350)), manual=True)
        self.assertEqual(fourth["reason"], "angle")
        tilted = self.observe(gate, 4, gray=gray, view=board(offset=(350, 350), perspective=(.08, .02)), manual=True)
        self.assertTrue(tilted["capture"])

    def test_letterboxing_does_not_change_board_roi_quality(self):
        gate = CalibrationCaptureGate(start_delay=0)
        padded = np.zeros((1800, 3000), np.uint8)
        padded[:480, :640] = checker()
        status = self.observe(gate, 0, gray=padded, view=board(scale=30), manual=True)
        self.assertTrue(status["capture"])

    def test_too_few_duplicate_collinear_and_invalid_corners_are_rejected(self):
        for view in (board(subset=np.arange(5)),
                     (np.zeros((12, 1, 2)), np.arange(12).reshape(-1, 1)),
                     (board()[0], np.zeros((24, 1))),
                     (board()[0], np.full((24, 1), float("nan"))),
                     (board()[0] * float("nan"), board()[1])):
            with self.subTest(view=view):
                gate = CalibrationCaptureGate(start_delay=0)
                self.assertFalse(self.observe(gate, 0, view=view, manual=True)["capture"])

    def test_frame_size_change_and_clock_regression_are_rejected(self):
        gate = CalibrationCaptureGate()
        corners, ids = board()
        status = gate.observe(0, checker(), corners, ids, (1920, 1080))
        self.assertEqual(status["reason"], "frame_size")
        self.observe(gate, 1)
        with self.assertRaises(ValueError):
            self.observe(gate, .5)

    def test_target_is_bounded_and_does_not_capture_after_completion(self):
        gate = CalibrationCaptureGate(total=2)
        self.assertTrue(self.observe(gate, 0, manual=True)["capture"])
        last = self.observe(gate, 1, view=board(offset=(220, 100)), manual=True)
        self.assertTrue(last["complete"])
        self.assertFalse(self.observe(gate, 2, manual=True)["capture"])
        self.assertEqual(gate.count, 2)

    def test_sweep_selects_clear_varied_moving_views_without_steady_hold(self):
        gate = CalibrationCaptureGate(total=3, mode="sweep", start_delay=0)
        for now, position in ((0, (100, 100)), (.4, (210, 100)), (.8, (100, 210))):
            status = self.observe(gate, now, view=board(offset=position))
            self.assertTrue(status["capture"], status)
        self.assertTrue(status["complete"])
        self.assertEqual(status["holdResets"], 0)

    def test_sweep_rejects_blur_duplicates_and_isolated_misplaced_corners(self):
        gate = CalibrationCaptureGate(mode="sweep", start_delay=0)
        self.assertTrue(self.observe(gate, 0)["capture"])
        self.assertEqual(self.observe(gate, .4)["reason"], "duplicate")
        shifted = board(offset=(220, 100))
        blurred = np.full((480, 640), 128, np.uint8)
        self.assertEqual(self.observe(gate, .8, gray=blurred, view=shifted, manual=True)["reason"], "blur")
        points, ids = shifted
        points[10, 0] += (45, 30)
        self.assertEqual(self.observe(gate, 1.2, view=(points, ids))["reason"], "corner_outlier")
        self.assertEqual(gate.count, 1)
        self.assertTrue(self.observe(gate, 1.6, view=board(offset=(220, 100)))["capture"])

    def test_sweep_still_requires_countdown_spacing_and_varied_tilts(self):
        gate = CalibrationCaptureGate(mode="sweep", start_delay=1)
        self.assertEqual(self.observe(gate, 0)["reason"], "starting")
        self.assertTrue(self.observe(gate, 1)["capture"])
        self.assertEqual(self.observe(gate, 1.1, view=board(offset=(220, 100)))["reason"], "cooldown")
        self.assertTrue(self.observe(gate, 1.4, view=board(offset=(220, 100)))["capture"])
        gray = checker(1000, 700)
        self.assertTrue(self.observe(gate, 1.8, gray=gray, view=board(offset=(100, 350)))["capture"])
        self.assertEqual(self.observe(gate, 2.2, gray=gray, view=board(offset=(350, 350)))["reason"], "angle")
        self.assertTrue(self.observe(gate, 2.6, gray=gray,
            view=board(offset=(350, 350), perspective=(.08, .02)))["capture"])


class CalibrationViewArchiveTests(unittest.TestCase):
    def test_selected_full_size_frames_and_exact_corners_are_bounded_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = CalibrationViewArchive("fixture-profile", total=2, root=directory)
            image = np.dstack([checker()] * 3)
            points, ids = board()
            for i in range(2):
                self.assertTrue(archive.record(image, points + i, ids, capture_seconds=100 + i,
                    state={"captureMode": "sweep", "sharpness": 100.},
                    source_metadata={"seq": i, "format": {"lens": "wide"}}))
            self.assertFalse(archive.record(image, points, ids, capture_seconds=103, state={}))
            self.assertTrue(archive.finish("saved"))
            manifest = json.loads((archive.path / "manifest.json").read_text())
            self.assertEqual(manifest["outcome"], "saved")
            self.assertEqual(len(manifest["selectedViews"]), 2)
            self.assertEqual(len(list(archive.path.glob("*.jpg"))), 2)
            saved = json.loads((archive.path / "view-02.json").read_text())
            np.testing.assert_array_equal(saved["charucoCorners"], (points + 1).reshape(-1, 2))
            self.assertEqual(saved["charucoIds"], ids.ravel().tolist())
            self.assertEqual(saved["source"]["seq"], 1)
            self.assertEqual(saved["imageSize"], [640, 480])
            self.assertEqual(cv2.imread(str(archive.path / "view-02.jpg")).shape, image.shape)

    def test_image_write_failure_is_reported_in_manifest_not_called_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = CalibrationViewArchive("fixture-profile", root=directory)
            points, ids = board()
            with patch("app_paths.atomic_write_bytes", side_effect=OSError("disk full")):
                self.assertFalse(archive.record(checker(), points, ids, capture_seconds=1., state={}))
            self.assertFalse(archive.finish("failed"))
            manifest = json.loads((archive.path / "manifest.json").read_text())
            self.assertIn("disk full", manifest["error"])
            self.assertEqual(manifest["selectedViews"], [])


if __name__ == "__main__":
    unittest.main()
