"""Board geometry and window routing checked without opening any windows."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from calibration_visuals import (BOARD_WINDOW_NAME, PREVIEW_WINDOW_NAME, choose_preview_display,
                                 close_calibration_windows, render_board_status,
                                 render_calibration_preview, render_charuco_board,
                                 show_calibration_preview, show_charuco_board, position_own_macos_window,
                                 calibration_diagnostic_lines, calibration_mode_label)


class CalibrationVisualTests(unittest.TestCase):
    def setUp(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.board = cv2.aruco.CharucoBoard((7, 5), 0.03, 0.015, dictionary)
        self.board_display = {"x": -1920, "y": 0, "width": 1920, "height": 1080}
        self.preview_display = {"x": 0, "y": 0, "width": 1440, "height": 900}

    def cv(self):
        return Mock(WINDOW_NORMAL=cv2.WINDOW_NORMAL, WND_PROP_FULLSCREEN=cv2.WND_PROP_FULLSCREEN,
                    WINDOW_FULLSCREEN=cv2.WINDOW_FULLSCREEN)

    def test_board_has_exact_square_geometry_and_white_border_on_both_orientations(self):
        with patch.object(cv2, "namedWindow", side_effect=AssertionError("No GUI in image rendering")):
            for width, height in ((1920, 1080), (1080, 1920), (1280, 720), (320, 240)):
                with self.subTest(width=width, height=height):
                    image = render_charuco_board(self.board, width, height)
                    self.assertEqual(image.shape, (height, width))
                    self.assertEqual(image.dtype, np.uint8)
                    self.assertEqual(set(np.unique(image)), {0, 255})
                    ys, xs = np.where(image == 0)
                    cell = int(min(width * 0.85 / 7, height * 0.85 / 5))
                    self.assertEqual(xs.max() - xs.min() + 1, 7 * cell)
                    self.assertEqual(ys.max() - ys.min() + 1, 5 * cell)
                    self.assertTrue(np.all(image[:ys.min()] == 255))
                    self.assertTrue(np.all(image[ys.max() + 1:] == 255))
                    self.assertTrue(np.all(image[:, :xs.min()] == 255))
                    self.assertTrue(np.all(image[:, xs.max() + 1:] == 255))

    def test_generated_image_detects_the_shared_board_and_keeps_equal_axis_scale(self):
        image = render_charuco_board(self.board, 1920, 1080)
        corners, ids, _marker_corners, marker_ids = cv2.aruco.CharucoDetector(self.board).detectBoard(image)
        self.assertIsNotNone(ids)
        self.assertEqual(len(ids), 24)
        self.assertEqual(set(marker_ids.flatten()), set(self.board.getIds().flatten()))
        points = self.board.getChessboardCorners()[ids.flatten(), :2]
        pixels = corners.reshape(-1, 2)
        scale_x = np.polyfit(points[:, 0], pixels[:, 0], 1)[0]
        scale_y = np.polyfit(points[:, 1], pixels[:, 1], 1)[0]
        self.assertAlmostEqual(scale_x / scale_y, 1, places=3)

    def test_invalid_or_excessive_display_sizes_fail_before_allocation(self):
        for width, height in ((0, 1080), (-1, 1080), (True, 1080), (1920.5, 1080),
                              (16385, 1080), (16000, 16000), (60, 40)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                render_charuco_board(self.board, width, height)
        for fraction in (0, 1, float("nan"), True):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                render_charuco_board(self.board, 1920, 1080, board_fraction=fraction)

    def test_show_board_uses_the_requested_screen_origin_and_fullscreen_window(self):
        cv = self.cv()
        name = show_charuco_board(self.board, self.board_display, cv=cv)
        self.assertEqual(name, BOARD_WINDOW_NAME)
        cv.namedWindow.assert_called_once_with(name, cv.WINDOW_NORMAL)
        cv.moveWindow.assert_called_once_with(name, -1920, 0)
        cv.resizeWindow.assert_called_once_with(name, 1920, 1080)
        cv.setWindowProperty.assert_called_once_with(name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        self.assertEqual(cv.imshow.call_args.args[1].shape, (1080, 1920))
        self.assertEqual([call[0] for call in cv.mock_calls], ["namedWindow", "imshow", "waitKey",
                         "resizeWindow", "moveWindow", "waitKey", "setWindowProperty", "imshow", "waitKey"])

    def test_failed_board_creation_closes_its_own_window(self):
        cv = self.cv()
        cv.setWindowProperty.side_effect = RuntimeError("Fullscreen unavailable")
        with self.assertRaisesRegex(RuntimeError, "Fullscreen unavailable"):
            show_charuco_board(self.board, self.board_display, cv=cv)
        cv.destroyWindow.assert_called_once_with(BOARD_WINDOW_NAME)
        cv.destroyAllWindows.assert_not_called()

    def test_preview_uses_a_nonoverlapping_display_and_keeps_camera_aspect(self):
        self.assertEqual(choose_preview_display(self.board_display,
            [self.board_display, self.preview_display]), self.preview_display)
        frame = np.zeros((1920, 1080, 3), np.uint8)
        cv = self.cv()
        name = show_calibration_preview(frame, self.preview_display, board_display=self.board_display, cv=cv)
        self.assertEqual(name, PREVIEW_WINDOW_NAME)
        _name, width, height = cv.resizeWindow.call_args.args
        self.assertAlmostEqual(width / height, 1080 / 1920, places=2)
        _name, x, y = cv.moveWindow.call_args.args
        self.assertGreaterEqual(x, self.preview_display["x"])
        self.assertGreaterEqual(y, self.preview_display["y"])
        self.assertLessEqual(x + width, self.preview_display["x"] + self.preview_display["width"])
        self.assertLessEqual(y + height, self.preview_display["y"] + self.preview_display["height"])

    def test_single_or_overlapping_monitor_never_creates_a_preview_over_the_board(self):
        self.assertIsNone(choose_preview_display(self.board_display, [self.board_display]))
        frame = np.zeros((480, 640, 3), np.uint8)
        overlap = {**self.board_display, "x": -1000}
        for display in (None, self.board_display, overlap):
            with self.subTest(display=display):
                cv = self.cv()
                self.assertIsNone(show_calibration_preview(frame, display, board_display=self.board_display, cv=cv))
                cv.namedWindow.assert_not_called()
                cv.imshow.assert_not_called()

    def test_window_cleanup_destroys_only_owned_names_and_tolerates_closed_windows(self):
        cv = self.cv()
        cv.destroyWindow.side_effect = [RuntimeError("already closed"), None]
        close_calibration_windows(BOARD_WINDOW_NAME, PREVIEW_WINDOW_NAME, BOARD_WINDOW_NAME, None, cv=cv)
        self.assertEqual([call.args[0] for call in cv.destroyWindow.call_args_list],
                         [BOARD_WINDOW_NAME, PREVIEW_WINDOW_NAME])
        cv.destroyAllWindows.assert_not_called()

    def test_preview_annotation_does_not_change_the_calibration_frame(self):
        frame = np.full((1080, 1920, 3), 127, np.uint8)
        original = frame.copy()
        result = render_calibration_preview(frame, captured=3, required=10, detected_corners=24,
            auto_capture=True, prompt="Tilt the phone slightly, then hold still.", countdown=2.5)
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(result.shape, (540, 960, 3))
        self.assertGreater(np.count_nonzero(result[:140] > 200), 100)
        np.testing.assert_array_equal(result[-100:-28], 127)

    def test_tv_status_changes_only_the_footer_and_keeps_the_board_detectable(self):
        base = render_charuco_board(self.board, 1920, 1080)
        original = base.copy()
        pattern_bottom = np.flatnonzero(np.any(base < 128, axis=1))[-1] + 1
        for seconds in (5.0, 2.5, 0.0):
            image = render_board_status(base, "Move or tilt the phone, then hold still.",
                count=3, total=10, auto_capture=True, countdown=seconds)
            np.testing.assert_array_equal(image[:pattern_bottom + 10], base[:pattern_bottom + 10])
            np.testing.assert_array_equal(base, original)
            self.assertTrue(np.any(image[pattern_bottom + 10:] != base[pattern_bottom + 10:]))
            _corners, ids, _markers, marker_ids = cv2.aruco.CharucoDetector(self.board).detectBoard(image)
            self.assertEqual(len(ids), 24)
            self.assertEqual(set(marker_ids.flatten()), set(self.board.getIds().flatten()))

    def test_small_tv_status_prioritizes_count_without_covering_pattern(self):
        base = render_charuco_board(self.board, 320, 240)
        image = render_board_status(base, "Tilt phone", count=1, total=10, auto_capture=False)
        pattern_bottom = np.flatnonzero(np.any(base < 128, axis=1))[-1] + 1
        np.testing.assert_array_equal(image[:pattern_bottom + 4], base[:pattern_bottom + 4])
        self.assertTrue(np.any(image[pattern_bottom + 4:] != base[pattern_bottom + 4:]))

    def test_own_macos_windows_use_primary_height_for_all_monitor_arrangements(self):
        window = Mock()
        window.title.return_value = BOARD_WINDOW_NAME
        window.isVisible.return_value = True
        screen = Mock()
        screen.frame.return_value = SimpleNamespace(size=SimpleNamespace(height=900))
        appkit = SimpleNamespace(NSScreen=Mock(), NSApplication=Mock())
        appkit.NSScreen.screens.return_value = [screen]
        appkit.NSApplication.sharedApplication.return_value.windows.return_value = [window]
        for x, y in ((0, 0), (-1920, 0), (0, -1080), (0, 900), (-1920, -1080)):
            with self.subTest(x=x, y=y):
                position_own_macos_window(BOARD_WINDOW_NAME, x, y, appkit=appkit, cv=self.cv())
                window.setFrameTopLeftPoint_.assert_called_with((float(x), float(900 - y)))

    def test_own_window_lookup_ignores_hidden_or_other_titles_and_pumps_bounded_events(self):
        hidden, other, current = Mock(), Mock(), Mock()
        hidden.title.return_value = BOARD_WINDOW_NAME
        hidden.isVisible.return_value = False
        other.title.return_value = "Unrelated Sarween view"
        other.isVisible.return_value = True
        current.title.return_value = BOARD_WINDOW_NAME
        current.isVisible.return_value = True
        screen = Mock()
        screen.frame.return_value = SimpleNamespace(size=SimpleNamespace(height=900))
        appkit = SimpleNamespace(NSScreen=Mock(), NSApplication=Mock())
        appkit.NSScreen.screens.return_value = [screen]
        application = appkit.NSApplication.sharedApplication.return_value
        application.windows.side_effect = [[hidden, other], [hidden, other, current]]
        cv = self.cv()
        position_own_macos_window(BOARD_WINDOW_NAME, 0, 900, appkit=appkit, cv=cv)
        cv.waitKey.assert_called_once_with(1)
        current.setFrameTopLeftPoint_.assert_called_once_with((0.0, 0.0))
        hidden.setFrameTopLeftPoint_.assert_not_called()
        other.setFrameTopLeftPoint_.assert_not_called()
        application.windows.side_effect = None
        application.windows.return_value = [hidden, other]
        cv.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "selected display"):
            position_own_macos_window(BOARD_WINDOW_NAME, 0, 0, appkit=appkit, cv=cv)
        self.assertEqual(cv.waitKey.call_count, 5)

    def test_show_board_can_use_injected_own_window_positioner_without_cv_move(self):
        cv = self.cv()
        positioner = Mock()
        show_charuco_board(self.board, self.board_display, cv=cv, native_positioner=positioner)
        positioner.assert_called_once_with(BOARD_WINDOW_NAME, -1920, 0)
        cv.moveWindow.assert_not_called()
        first_show_index = [call[0] for call in cv.mock_calls].index("imshow")
        self.assertLess(first_show_index, [call[0] for call in cv.mock_calls].index("setWindowProperty"))

    def test_diagnostics_explain_the_blocker_and_report_corner_and_stability_metrics(self):
        for reason, title in (("no_board", "No board detected"), ("corners", "10 corners needed"),
                              ("blur", "Too blurry"), ("holding", "Hold steady"),
                              ("duplicate", "Move to a new angle")):
            with self.subTest(reason=reason):
                lines = calibration_diagnostic_lines(8, diagnostics={"reason": reason,
                    "validCorners": 7, "poseCorners": 6, "sharedCorners": 6, "stableProgress": 0.65,
                    "motionRatio": 0.009, "holdResets": 3, "sharpness": 42.2})
                self.assertEqual(lines[0], title)
                self.assertEqual(lines[1], "Calibration corners: 7/10 needed | detected: 8")
                self.assertIn("Steady: 65%", lines[2])
                self.assertIn("Motion: 0.9%", lines[2])
                self.assertIn("Shared: 6", lines[2])
                self.assertIn("Pose model: 6", lines[2])
                self.assertIn("Hold resets: 3", lines[2])
                self.assertEqual(lines[3], "Sharpness: 42")

    def test_diagnostic_tv_footer_keeps_every_pattern_pixel_and_corner(self):
        base = render_charuco_board(self.board, 1920, 1080)
        pattern_bottom = np.flatnonzero(np.any(base < 128, axis=1))[-1] + 1
        annotated = render_board_status(base, "Hold the camera still.", count=0, total=10,
            detected_corners=24, diagnostics={"reason": "holding", "stableProgress": .8,
                "validCorners": 24, "poseCorners": 22, "sharedCorners": 20, "motionRatio": .005, "holdResets": 2},
            countdown=.2)
        np.testing.assert_array_equal(annotated[:pattern_bottom + 10], base[:pattern_bottom + 10])
        _corners, ids, _markers, marker_ids = cv2.aruco.CharucoDetector(self.board).detectBoard(annotated)
        self.assertEqual(len(ids), 24)
        self.assertEqual(set(marker_ids.flatten()), set(self.board.getIds().flatten()))

    def test_capture_mode_labels_follow_explicit_mode(self):
        self.assertEqual(calibration_mode_label(False, {"captureMode": "timed"}), "TIMED (5s)")
        self.assertEqual(calibration_mode_label(True, {"captureMode": "manual"}), "MANUAL")
        self.assertEqual(calibration_mode_label(False, {"captureMode": "auto"}), "AUTO")
        self.assertEqual(calibration_mode_label(True), "AUTO")
        lines = calibration_diagnostic_lines(24, diagnostics={"captureMode": "timed", "reason": "manual",
                                                              "validCorners": 24, "poseCorners": 6})
        self.assertEqual(lines[0], "Timer running")
        self.assertEqual(lines[1], "Calibration corners: 24/10 needed")
        self.assertIn("Pose model: 6", lines[2])

    def test_timed_renderers_show_next_capture_and_timer_control(self):
        base = render_charuco_board(self.board, 1920, 1080)
        state = {"captureMode": "timed", "reason": "manual", "validCorners": 24}
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            render_calibration_preview(base, captured=0, required=10, detected_corners=24,
                diagnostics=state, countdown=4.2)
            labels = [call.args[1] for call in text.call_args_list]
            self.assertTrue(any("TIMED (5s)" in label for label in labels))
            self.assertIn("Next capture in 4.2s", labels)
            self.assertTrue(any("T 5s timer" in label and "C/SPACE now" in label for label in labels))
            self.assertFalse(any("hold still" in label for label in labels))
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            render_board_status(base, "Ready", count=0, total=10, detected_corners=24,
                diagnostics=state, countdown=4.2)
            labels = [call.args[1] for call in text.call_args_list]
            self.assertTrue(any("TIMED (5s)" in label and "Next capture in 4.2s" in label for label in labels))


if __name__ == "__main__":
    unittest.main()
