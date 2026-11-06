import json
from pathlib import Path
import tempfile
import unittest

from calibration_diagnostics import CalibrationDiagnostics


class CalibrationDiagnosticsTests(unittest.TestCase):
    def test_partial_corner_failure_is_saved_without_camera_images(self):
        diagnostics = CalibrationDiagnostics("phone", "iPhone", total=10)
        for count in (5, 7, 9):
            diagnostics.observe({"reason": "corners", "message": "Show more board", "stableProgress": 0,
                                 "holdResets": 3, "sharedCorners": 4, "motionRatio": 0.05},
                                detected_corners=count, frame_size=(1920, 1080), automatic=True)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "last_attempt.json"
            diagnostics.save("cancelled", path=target)
            data = json.loads(target.read_text())
            self.assertEqual(list(Path(directory).iterdir()), [target])
        self.assertEqual(data["acceptedViews"], 0)
        self.assertEqual(data["maximumDetectedCorners"], 9)
        self.assertEqual(data["reasonCounts"], {"corners": 3})
        self.assertIn("Too few", data["summary"])
        self.assertEqual(data["events"][-1]["sharedCorners"], 4)

    def test_startup_does_not_hide_the_real_blocker_and_history_is_bounded(self):
        diagnostics = CalibrationDiagnostics("phone", "iPhone")
        for _ in range(300):
            diagnostics.observe({"reason": "starting"}, detected_corners=24, frame_size=(640, 480), automatic=True)
        diagnostics.observe({"reason": "blur", "message": "Too blurry"}, detected_corners=24,
                            frame_size=(640, 480), automatic=True)
        self.assertIn("blurry", diagnostics.summary())
        self.assertEqual(len(diagnostics.snapshot("cancelled")["events"]), 240)

    def test_completion_and_errors_have_clear_summaries(self):
        diagnostics = CalibrationDiagnostics("phone", "iPhone", total=1)
        diagnostics.observe({"reason": "captured", "capture": True}, detected_corners=20,
                            frame_size=(640, 480), automatic=True)
        self.assertIn("completed", diagnostics.summary())
        diagnostics.error = "Camera profile changed during calibration."
        self.assertIn("profile changed", diagnostics.summary())

    def test_cancelled_timer_wait_is_not_reported_as_a_manual_capture_request(self):
        diagnostics = CalibrationDiagnostics("phone", "iPhone")
        diagnostics.observe({"reason": "timed_wait", "captureMode": "timed"}, detected_corners=24,
                            frame_size=(640, 480), automatic=False)
        self.assertIn("timer was still counting down", diagnostics.summary())
        self.assertEqual(diagnostics.snapshot("cancelled")["events"][0]["captureMode"], "timed")


if __name__ == "__main__":
    unittest.main()
