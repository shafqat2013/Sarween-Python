"""Calibration orchestration with simulated capture and presentation boundaries."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import calibration
from hardware_profiles import HardwareProfiles


MATRIX = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
DISTORTION = np.zeros((1, 5))
DISPLAYS = [{"index": 0, "primary": True, "x": 0, "y": 0, "width": 1280, "height": 800},
            {"index": 1, "primary": False, "x": -640, "y": -480, "width": 640, "height": 480}]


class AcceptedViews:
    """Pose quality is exercised separately by test_calibration_capture."""
    def __init__(self, total, *, automatic=True, **_options):
        self.total, self.automatic = total, automatic
        self.count = 0
        self.frames = []

    def observe(self, _now, gray, corners, ids, _size, *, manual=False):
        self.frames.append((gray.copy(), manual))
        accepted = corners is not None and ids is not None and (self.automatic or manual)
        self.count += int(accepted)
        return {"capture": accepted, "reason": "captured" if accepted else "no_board",
                "message": "Captured" if accepted else "Show the board"}


class CameraCalibrationFlowTests(unittest.TestCase):
    @contextmanager
    def flow(self, *, frames=3, keys=None, marker_ids=None):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            profile = store.upsert_profile("camera", {"name": "iPhone", "lens": "Wide", "zoom": "1x"})
            capture = Mock()
            capture.isOpened.return_value = True
            capture.read.return_value = (True, np.full((240, 320, 3), 80, np.uint8))
            corners = np.array([[[40 + col * 35, 40 + row * 40]] for row in range(4) for col in range(6)], np.float32)
            ids = np.arange(24, dtype=np.int32).reshape(-1, 1)
            gates = []

            def gate(*args, **kwargs):
                value = AcceptedViews(*args, **kwargs)
                gates.append(value)
                return value

            values = {"store": store, "profile": profile, "capture": capture, "gates": gates}
            stack.enter_context(patch("hardware_profiles.HardwareProfiles", return_value=store))
            stack.enter_context(patch.object(calibration, "NUM_FRAMES", frames))
            stack.enter_context(patch.object(calibration, "CalibrationCaptureGate", side_effect=gate))
            stack.enter_context(patch.object(calibration.cv2, "VideoCapture", return_value=capture))
            values["board"] = stack.enter_context(patch.object(calibration, "display_charuco", return_value="CharucoBoard"))
            values["preview"] = stack.enter_context(patch.object(calibration, "show_calibration_preview", return_value="Camera calibration"))
            values["close"] = stack.enter_context(patch.object(calibration, "close_calibration_windows"))
            values["diagnostics"] = stack.enter_context(patch.object(calibration, "_save_calibration_diagnostics"))
            stack.enter_context(patch.object(calibration.cv2, "imshow"))
            values["keys"] = stack.enter_context(patch.object(calibration.cv2, "waitKey",
                side_effect=keys if keys is not None else [255] * frames))
            values["interpolate"] = stack.enter_context(patch.object(calibration.cv2.aruco, "interpolateCornersCharuco",
                return_value=(24, corners, ids)))
            stack.enter_context(patch.object(calibration.cv2.aruco, "detectMarkers",
                return_value=([], np.array([[0], [1]]) if marker_ids is None else marker_ids, [])))
            stack.enter_context(patch.object(calibration.cv2.aruco, "drawDetectedMarkers"))
            values["solve"] = stack.enter_context(patch.object(calibration.cv2.aruco, "calibrateCameraCharuco",
                return_value=(0.2, MATRIX, DISTORTION, [], [])))
            yield values

    def run_flow(self, value, **kwargs):
        return calibration.calibrate_webcam(2, profile_id=value["profile"]["id"],
            display=DISPLAYS[1], displays=DISPLAYS, **kwargs)

    def test_generated_board_and_solver_share_geometry_and_auto_capture_needs_no_keys(self):
        with self.flow() as value:
            result = self.run_flow(value)
            self.assertTrue(result["bindingVerified"])
            self.assertEqual(result["calibration"]["imageSize"], [320, 240])
            self.assertEqual(value["capture"].read.call_count, 3)
            board_call = value["board"].call_args
            self.assertEqual(board_call.args[0], DISPLAYS[1])
            board = board_call.kwargs["board"]
            self.assertEqual(board.getChessboardSize(), (7, 5))
            self.assertIs(value["solve"].call_args.args[2], board)
            self.assertEqual(len(value["solve"].call_args.args[0]), 3)
            self.assertTrue(all(not manual and np.all(gray == 80) for gray, manual in value["gates"][0].frames))
            self.assertEqual(value["preview"].call_args.args[1], DISPLAYS[0])
            self.assertTrue(all(call.args == (1,) for call in value["keys"].call_args_list))
            value["capture"].release.assert_called_once()
            value["close"].assert_called_once_with("CharucoBoard", "Camera calibration", cv=calibration.cv2)

    def test_cancel_closes_owned_windows_and_never_solves_or_saves(self):
        with self.flow(keys=[ord("q")]) as value:
            self.assertIsNone(self.run_flow(value))
            value["solve"].assert_not_called()
            self.assertIsNone(value["store"].get_profile("camera", value["profile"]["id"])["calibration"])
            value["capture"].release.assert_called_once()
            value["close"].assert_called_once()

    def test_stability_pose_subset_does_not_remove_lens_distortion_constraints(self):
        with self.flow(frames=1) as value:
            def observe(_gate, *_args, **_kwargs):
                return {"capture": True, "message": "Captured", "reason": "captured",
                        "poseCornerIds": list(range(12))}
            with patch.object(AcceptedViews, "observe", observe):
                self.run_flow(value)
            self.assertEqual(len(value["solve"].call_args.args[0][0]), 24)
            self.assertEqual(len(value["solve"].call_args.args[1][0]), 24)

    def test_timed_capture_waits_five_seconds_and_needs_no_per_view_keypress(self):
        with self.flow(frames=1, keys=[ord("t"), 255, 255]) as value:
            with patch.object(calibration.time, "monotonic", side_effect=[0.0, 4.9, 5.0]):
                result = self.run_flow(value)
            self.assertTrue(result["bindingVerified"])
            self.assertEqual(value["capture"].read.call_count, 3)
            self.assertEqual([manual for _gray, manual in value["gates"][0].frames], [False, False, True])

    def test_duplicate_marker_ids_never_reach_the_interpolator_or_solver(self):
        with self.flow(keys=[255, ord("q")], marker_ids=np.array([[10], [10]])) as value:
            self.assertIsNone(self.run_flow(value))
            value["interpolate"].assert_not_called()
            value["solve"].assert_not_called()
            self.assertEqual(value["gates"][0].count, 0)

    def test_feed_size_change_cancels_without_replacing_calibration(self):
        with self.flow() as value:
            value["capture"].read.side_effect = [(True, np.zeros((240, 320, 3), np.uint8)),
                                                (True, np.zeros((480, 640, 3), np.uint8))]
            with self.assertRaisesRegex(ValueError, "resolution changed"):
                self.run_flow(value)
            value["solve"].assert_not_called()
            value["capture"].release.assert_called_once()

    def test_presentation_failure_releases_camera_and_preserves_profile(self):
        with self.flow() as value:
            value["board"].side_effect = RuntimeError("Display disconnected")
            with self.assertRaisesRegex(RuntimeError, "Display disconnected"):
                self.run_flow(value)
            value["capture"].release.assert_called_once()
            self.assertIsNone(value["store"].get_profile("camera", value["profile"]["id"])["calibration"])

    def test_setup_uses_positioned_inventory_without_running_system_profiler(self):
        import setup
        with patch.object(setup, "detect_displays_backup", return_value=DISPLAYS), \
             patch.object(setup, "detect_webcams_backup", return_value=[{"index": 2}]), \
             patch.object(setup.subprocess, "run", side_effect=AssertionError("Unexpected profiler")):
            displays, cameras = setup.detect_setup()
        self.assertEqual(displays[1]["y"], -480)
        self.assertEqual(displays[1]["x"], -640)
        self.assertEqual(cameras, [{"index": 2}])


if __name__ == "__main__":
    unittest.main()
