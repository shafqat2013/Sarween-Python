"""Mac preview regression without a Tk root, native windows or Qt pumping."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import tk_tracking_windows as windows


class TkTrackingWindowTests(unittest.TestCase):
    def setUp(self):
        self.root = Mock()
        self.native = Mock()
        self.canvas = Mock()
        self.canvas.create_image.return_value = "one-image-item"
        self.top = patch.object(windows.tk, "Toplevel", return_value=self.native).start()
        self.can = patch.object(windows.tk, "Canvas", return_value=self.canvas).start()
        self.photo = patch.object(windows.ImageTk, "PhotoImage", side_effect=lambda **kw: kw["image"]).start()
        self.addCleanup(patch.stopall)
        for name in ("imshow", "waitKey", "namedWindow", "resizeWindow",
                     "destroyWindow", "destroyAllWindows"):
            patch.object(windows.cv2, name, side_effect=AssertionError("HighGUI reached on Mac")).start()

    def test_mac_preview_updates_one_image_and_keeps_source_pixels(self):
        backend = windows.create_tracking_windows(self.root, platform="darwin")
        self.assertIsInstance(backend, windows.TkTrackingWindows)
        frame = np.zeros((240, 480, 3), np.uint8)
        frame[:] = (0, 0, 255)
        before = frame.copy()
        backend.ensure_window("Camera", 640, 480)
        backend.imshow("Camera", frame)
        backend.imshow("Camera", frame)
        self.assertEqual(self.top.call_count, 1)
        self.canvas.create_image.assert_called_once()
        self.assertEqual(self.canvas.itemconfigure.call_count, 2)
        self.assertEqual(self.photo.call_args.kwargs["image"].size, (640, 320))
        self.assertEqual(self.photo.call_args.kwargs["image"].getpixel((0, 0)), (255, 0, 0))
        self.assertIs(self.photo.call_args.kwargs["master"], self.root)
        np.testing.assert_array_equal(frame, before)

    def test_resize_records_dimensions_without_recursive_geometry_work(self):
        backend = windows.TkTrackingWindows(self.root)
        backend.ensure_window("Camera", 640, 480)
        callback = self.canvas.bind.call_args.args[1]
        self.photo.reset_mock()
        callback(SimpleNamespace(width=300, height=300))
        self.photo.assert_not_called()
        self.root.update.assert_not_called()
        backend.imshow("Camera", np.zeros((100, 200), np.uint8))
        self.assertEqual(self.photo.call_args.kwargs["image"].size, (300, 150))
        self.canvas.coords.assert_called_with("one-image-item", 150, 150)

    def test_keyboard_is_tk_owned_and_quit_survives_intermediate_polls(self):
        backend = windows.TkTrackingWindows(self.root)
        backend._key(SimpleNamespace(char="", keysym="Escape"))
        self.assertEqual(backend.waitKey(1), 27)
        self.assertEqual(backend.waitKey(1), -1)
        backend._key(SimpleNamespace(char="q", keysym="q"))
        self.assertEqual(backend.waitKey(1), ord("q"))
        self.assertEqual(backend.waitKey(1), ord("q"))
        self.assertEqual(self.root.update.call_count, 4)
        backend.destroyAllWindows()
        self.assertEqual(backend.waitKey(1), -1)

    def test_user_close_does_not_destroy_shared_root_or_respawn_until_toggle(self):
        backend = windows.TkTrackingWindows(self.root)
        backend.ensure_window("Camera")
        close = self.native.protocol.call_args.args[1]
        close()
        backend.imshow("Camera", np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(self.top.call_count, 1)
        self.root.destroy.assert_not_called()
        backend.destroyWindow("Camera")
        backend.imshow("Camera", np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(self.top.call_count, 2)
        backend.destroyAllWindows()
        self.root.destroy.assert_not_called()

    def test_headless_backend_never_constructs_tk_or_calls_highgui(self):
        backend = windows.create_tracking_windows(enabled=False, platform="darwin")
        backend.ensure_window("Camera")
        backend.imshow("Camera", np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(backend.waitKey(1), -1)
        backend.destroyWindow("Camera")
        backend.destroyAllWindows()
        self.top.assert_not_called()
        self.photo.assert_not_called()

    def test_destroyed_application_root_requests_clean_session_exit(self):
        backend = windows.TkTrackingWindows(self.root)
        self.root.update.side_effect = windows.tk.TclError("application destroyed")
        self.assertEqual(backend.waitKey(1), ord("q"))
        self.assertFalse(backend._pumping)

    def test_non_mac_factory_preserves_opencv_backend(self):
        self.assertIsInstance(windows.create_tracking_windows(platform="linux"),
                              windows.OpenCvTrackingWindows)

    def test_shared_homography_preview_uses_supplied_backend_without_highgui(self):
        import cv_core
        backend = Mock()
        frame = np.zeros((120, 160, 3), np.uint8)
        with patch.object(cv_core, "ARUCO_DET", None), \
             patch.object(cv_core, "ensure_window", side_effect=AssertionError("HighGUI helper reached")):
            cv_core.show_homography_view(frame, None, 160, 120, 16, 12,
                                        display_backend=backend)
        backend.ensure_window.assert_called_once()
        self.assertEqual(backend.imshow.call_args.args[1].shape, (120, 160, 3))

    def test_live_tracking_contains_no_direct_highgui_calls(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "v3_tracking.py").read_text())
        session = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "begin_session")
        forbidden = {"imshow", "waitKey", "namedWindow", "resizeWindow",
                     "destroyWindow", "destroyAllWindows", "getWindowProperty"}
        for node in ast.walk(session):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            self.assertFalse(isinstance(node.func.value, ast.Name)
                             and node.func.value.id == "cv2" and node.func.attr in forbidden)
        homography = [node for node in ast.walk(session) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "show_homography_view"]
        self.assertTrue(homography)
        self.assertTrue(all(any(kw.arg == "display_backend" for kw in call.keywords)
                            for call in homography))


if __name__ == "__main__":
    unittest.main()
