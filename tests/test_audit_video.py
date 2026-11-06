"""Independent audit rendering from generated arrays or saved files only."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from audit_video import AuditVideo


ROOT = Path(__file__).resolve().parents[1]


class Frames:
    def __init__(self, frames, *, opened=True, fps=12):
        self.frames, self.position, self.released = frames, 0, False
        self.opened, self.fps = opened, fps

    def isOpened(self):
        return self.opened

    def get(self, key):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
                cv2.CAP_PROP_POS_FRAMES: self.position}.get(key, 0)

    def set(self, key, value):
        if key != cv2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        return True

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position].copy()
        self.position += 1
        return True, frame

    def release(self):
        self.released = True


def scene():
    return {"sceneId": "test", "width": 320, "height": 180, "gridSize": 40, "gridType": 1}


def view(shift=0):
    return {"sceneId": "test", "registration": {"left": 0, "top": 0, "right": 319, "bottom": 179},
        "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": shift, "ty": 0},
        "gridOriginX": 0, "gridOriginY": 0, "gridSize": 40}


def timeline(mode="viewport"):
    return {"markerMode": mode, "warpWidth": 320, "warpHeight": 180,
            "events": [{"frame": 0, "sceneInfo": scene(), "viewTransform": view()}]}


def marker_frame(ids=(10, 11, 12, 13), shift=0):
    board = np.full((220, 360, 3), 245, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    for marker_id, (x, y) in zip(ids, ((20, 20), (300, 20), (300, 160), (20, 160))):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 40)
        board[y:y + 40, x:x + 40] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    cv2.circle(board, (180, 110), 12, (0, 0, 230), -1)
    source = np.float32([[0, 0], [359, 0], [359, 219], [0, 219]])
    destination = np.float32([[90 + shift, 50], [550 + shift, 70],
                              [520 + shift, 400], [70 + shift, 380]])
    return cv2.warpPerspective(board, cv2.getPerspectiveTransform(source, destination),
                               (640, 480), borderValue=(45, 45, 45))


class AuditVideoTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(0)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "saved video.mp4"
        self.path.write_bytes(b"fixture is supplied by the mocked file decoder")

    def reader(self, frames, data=None, **kwargs):
        capture = Frames(frames, **kwargs)
        with patch("audit_video.cv2.VideoCapture", return_value=capture) as factory:
            reader = AuditVideo(self.path, data)
        factory.assert_called_once_with(str(self.path.resolve()))
        self.addCleanup(reader.close)
        return reader, capture

    def test_file_only_capture_never_accepts_device_index_or_missing_path(self):
        with patch("audit_video.cv2.VideoCapture") as capture:
            for value in (0, 1, True, "0", self.path.parent):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    AuditVideo(value)
            capture.assert_not_called()

    def test_invalid_or_unreadable_video_releases_capture(self):
        for capture in (Frames([], opened=False), Frames([], fps=12),
                        Frames([marker_frame()], fps=float("nan"))):
            with self.subTest(opened=capture.opened, fps=capture.fps), \
                 patch("audit_video.cv2.VideoCapture", return_value=capture), \
                 self.assertRaises(ValueError):
                AuditVideo(self.path)
            self.assertTrue(capture.released)

    def test_perspective_board_crop_grid_and_cell_focus_preserve_raw_pixels(self):
        source = marker_frame()
        reader, _capture = self.reader([source], timeline())
        raw = reader.image.copy()
        board, hint = reader.display(board=True, grid=False)
        self.assertEqual(board.shape, (180, 320, 3))
        self.assertEqual(hint, "Board view")
        self.assertGreater(int(board[90, 160, 2]), 180)
        self.assertLess(int(board[90, 160, 0]), 50)
        self.assertEqual(reader.cell_center("B2"), (60.0, 60.0))
        gridded, hint = reader.display(board=True, grid=True)
        self.assertFalse(np.array_equal(gridded, board))
        np.testing.assert_array_equal(reader.image, raw)
        self.assertIsNone(reader.cell_center("A0"))
        self.assertIsNone(reader.cell_center("ZZ999"))
        full, hint = reader.display(board=False, grid=False)
        np.testing.assert_array_equal(full, source)
        self.assertEqual(hint, "Full camera view")
        focus = reader.cell_center("B2")
        expected = cv2.perspectiveTransform(np.float64([[[60, 60]]]), np.linalg.inv(reader._h))[0, 0]
        np.testing.assert_allclose(focus, expected)

    def test_metadata_changes_at_current_frame_drive_grid_coordinates(self):
        data = timeline()
        data["events"].append({"frame": 1, "sceneInfo": scene(), "viewTransform": view(40)})
        reader, _capture = self.reader([marker_frame(), marker_frame()], data)
        reader.display()
        self.assertEqual(reader.cell_center("B2"), (60.0, 60.0))
        reader.seek(1)
        reader.display()
        self.assertEqual(reader.cell_center("B2"), (20.0, 60.0))

    def test_missing_geometry_never_invents_grid_labels(self):
        data = {"markerMode": "viewport", "warpWidth": 320, "warpHeight": 180,
                "gridCols": 8, "gridRows": 5}
        reader, _capture = self.reader([marker_frame()], data)
        plain, _hint = reader.display(grid=False)
        image, hint = reader.display(grid=True)
        np.testing.assert_array_equal(image, plain)
        self.assertIn("grid unavailable", hint)
        self.assertIsNone(reader.cell_center("A1"))

    def test_legacy_ids_use_only_explicit_known_uniform_grid(self):
        data = {"markerMode": "legacy", "warpWidth": 320, "warpHeight": 180,
                "gridCols": 4, "gridRows": 2}
        reader, _capture = self.reader([marker_frame((0, 1, 2, 3))], data)
        reader.display()
        self.assertEqual(reader.cell_center("A1"), (40.0, 45.0))
        self.assertEqual(reader.cell_center("D2"), (280.0, 135.0))
        self.assertIsNone(reader.cell_center("E2"))

    def test_corners_covered_hold_previous_view_and_refresh_when_visible(self):
        first = marker_frame()
        covered = np.full_like(first, (40, 80, 120))
        reader, _capture = self.reader([first, covered, marker_frame(shift=20)], timeline())
        first_h = reader._h.copy()
        reader.seek(1)
        image, hint = reader.display(grid=False)
        self.assertEqual(hint, "Corners covered · using last board view")
        self.assertEqual(image.shape, (180, 320, 3))
        np.testing.assert_allclose(reader._h, first_h)
        reader.seek(2)
        self.assertFalse(np.allclose(reader._h, first_h))
        reader.seek(1)
        # A later camera position cannot supply coordinates to an earlier frame.
        np.testing.assert_allclose(reader._h, first_h)

    def test_no_registration_returns_full_camera_and_no_cell_coordinates(self):
        source = np.full((240, 320, 3), 120, np.uint8)
        reader, _capture = self.reader([source], timeline())
        image, hint = reader.display()
        self.assertEqual(hint, "Full camera view")
        np.testing.assert_array_equal(image, source)
        self.assertIsNone(reader.cell_center("A1"))

    def test_verified_checkpoint_reads_only_h_and_uses_held_hint(self):
        path = self.path.parent / "checkpoint.npz"
        h = cv2.getPerspectiveTransform(np.float32([[20, 20], [300, 20], [300, 220], [20, 220]]),
            np.float32([[0, 0], [319, 0], [319, 179], [0, 179]]))
        # Object predictions cannot be read with allow_pickle=False. They are
        # intentionally irrelevant to this inspector and must never be loaded.
        np.savez_compressed(path, H_saved=h, tracking=np.array({"prediction": "ignored"}, dtype=object))
        data = timeline()
        data["checkpoints"] = [{"path": path.name, "frame": 0,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
        reader, _capture = self.reader([np.full((240, 320, 3), 120, np.uint8)], data)
        display, hint = reader.display(grid=False)
        self.assertEqual(display.shape, (180, 320, 3))
        self.assertEqual(hint, "Corners covered · using last board view")

    def test_bad_checksum_and_traversal_checkpoint_fall_back_without_coordinates(self):
        existing = self.path.parent / "wrong-checksum.npz"
        np.savez_compressed(existing, H_saved=np.eye(3))
        for entry in ({"path": "../outside.npz", "sha256": "a" * 64},
                      {"path": "missing.npz", "sha256": "a" * 64},
                      {"path": existing.name, "sha256": "a" * 64}):
            data = timeline()
            data["checkpoints"] = [entry]
            with self.subTest(entry=entry), patch("audit_video.np.load") as loader:
                reader, _capture = self.reader([np.full((240, 320, 3), 120, np.uint8)], data)
                self.assertIsNone(reader.cell_center("A1"))
                self.assertEqual(reader.display()[1], "Full camera view")
                loader.assert_not_called()

    def test_verified_but_invalid_checkpoint_geometry_does_not_invent_board(self):
        for matrix in (np.zeros((3, 3)), np.array([[1, 0, 2000], [0, 1, 0], [0, 0, 1]]),
                       np.eye(3) * complex(1, 2)):
            path = self.path.parent / "invalid-h.npz"
            np.savez_compressed(path, H_saved=matrix)
            data = timeline()
            data["checkpoints"] = [{"path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]
            with self.subTest(matrix=matrix):
                reader, _capture = self.reader([np.full((240, 320, 3), 120, np.uint8)], data)
                self.assertEqual(reader.display()[1], "Full camera view")
                self.assertIsNone(reader.cell_center("A1"))

    def test_invalid_viewport_transform_never_draws_grid_or_supplies_focus(self):
        data = timeline()
        data["events"][0]["viewTransform"]["clientToCanvasTransform"]["a"] = float("nan")
        reader, _capture = self.reader([marker_frame()], data)
        plain, _hint = reader.display(grid=False)
        image, hint = reader.display(grid=True)
        np.testing.assert_array_equal(image, plain)
        self.assertIn("grid unavailable", hint)
        self.assertIsNone(reader.cell_center("A1"))

    def test_seeking_clamps_and_rendering_does_not_redetect_or_use_native_windows(self):
        reader, capture = self.reader([marker_frame(), marker_frame(shift=20)], timeline())
        detector = Mock(wraps=reader._detector)
        reader._detector = detector
        with patch("audit_video.cv2.imshow", side_effect=AssertionError("Native UI forbidden")), \
             patch("audit_video.cv2.waitKey", side_effect=AssertionError("Native UI forbidden")):
            reader.display()
            reader.display(grid=False)
            reader.display(board=False)
            detector.detectMarkers.assert_not_called()
            reader.seek(500)
            self.assertEqual(reader.frame, 1)
            self.assertEqual(detector.detectMarkers.call_count, 1)
            reader.seek(-10)
            self.assertEqual(reader.frame, 0)
        reader.close()
        self.assertTrue(capture.released)
        with self.assertRaises(ValueError):
            reader.seek(0)

    @unittest.skipUnless((ROOT / "demo/tabletop.mp4").is_file(), "Saved demo unavailable")
    def test_actual_saved_demo_file_reads_and_seeks_without_gui(self):
        with patch("audit_video.cv2.imshow", side_effect=AssertionError("Native UI forbidden")), \
             patch("audit_video.cv2.waitKey", side_effect=AssertionError("Native UI forbidden")):
            reader = AuditVideo(ROOT / "demo/tabletop.mp4")
            try:
                self.assertGreater(reader.fps, 0)
                self.assertGreater(reader.count, 1)
                self.assertGreater(reader.duration, 0)
                reader.seek(min(reader.count - 1, 5))
                image, _hint = reader.display(grid=False)
                self.assertEqual(image.ndim, 3)
                self.assertEqual(reader.frame, min(reader.count - 1, 5))
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
