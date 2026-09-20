import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import foundryoutput as foundry
import tracking_regression as regression
from guided_capture import GuidedCapture, regression_case


def capture_payload():
    participants = [{"miniId": "red10", "tokenId": "red-token", "route": "A", "ringColor": "red"}]
    targets = [{**participants[0], "row": 2, "column": index, "cell": f"{chr(65 + index)}3", "step": index, "index": index} for index in range(6)]
    return {"sessionId": "test-capture", "sceneId": "capture-scene", "participants": participants, "targets": targets}


class GuidedCaptureTest(unittest.TestCase):
    def test_confirmation_uses_plan_not_detection_output(self):
        capture = GuidedCapture(capture_payload(), "capture-scene")
        capture.prompt(0, 10)
        event = capture.confirm(0, 20)
        self.assertEqual(event["cell"], "A3")
        self.assertEqual(event["promptFrame"], 10)
        self.assertEqual(event["source"], "userConfirmation")
        with self.assertRaises(ValueError):
            capture.confirm(0, 21)

    def test_wrong_scene_and_duplicate_token_are_rejected(self):
        with self.assertRaises(ValueError):
            GuidedCapture(capture_payload(), "other-scene")
        payload = capture_payload()
        payload["participants"].append({"miniId": "blue", "tokenId": "red-token"})
        with self.assertRaises(ValueError):
            GuidedCapture(payload, "capture-scene")

    def test_exported_case_uses_confirmation_window(self):
        data = {"fps": 10, "groundTruth": [{"mini": "red10", "cell": "A3", "promptFrame": 10, "frame": 20}]}
        case = regression_case(data, "capture.mp4")["cases"][0]
        expected = case["expectations"][0]
        self.assertEqual(expected["between"], [1.0, 5.0])
        event = regression.MovementEvent("red10", None, "A3", 30, 3, None, "r2c0", 0, 1)
        self.assertTrue(regression._event_matches_expectation(event, expected, 2))
        event.time_seconds = 6
        self.assertFalse(regression._event_matches_expectation(event, expected, 2))

    def test_recording_confirmation_and_case_export(self):
        globals_to_restore = {key: getattr(foundry, key) for key in (
            "SCENE_ID", "_capture_session", "_capture_ready", "_capture_status",
            "_tracking_output_paused", "_camera_lock_state", "_capture_commands",
        )}
        try:
            from collections import deque
            foundry.SCENE_ID = "capture-scene"
            foundry._capture_session = None
            foundry._capture_ready = False
            foundry._capture_commands = deque()
            foundry.update_camera_lock(True, [])
            with tempfile.TemporaryDirectory() as directory, patch.object(foundry, "queue_control") as send:
                video = str(Path(directory) / "capture.mp4")
                foundry.handle_capture_control({"action": "start", **capture_payload()})
                self.assertTrue(foundry.tracking_output_paused())
                self.assertEqual(foundry.pop_capture_command()["action"], "start")
                foundry.start_timeline_recording(video, fps=10, frame_width=100, frame_height=100, marker_mode="viewport", warp_width=100, warp_height=100, grid_cols=10, grid_rows=10)
                foundry.ready_guided_capture(video, {"red10": {"lab": [20, 30, 20]}})
                foundry.record_timeline_frame()
                common = {"sessionId": "test-capture", "index": 0}
                foundry.handle_capture_control({"action": "prompt", **common})
                foundry.record_tracking_event("red10", "r9c9")
                foundry.update_camera_lock(False, [10])
                foundry.handle_capture_control({"action": "confirm", **common})
                self.assertEqual(send.call_args[0][0]["state"], "error")
                foundry.update_camera_lock(True, [])
                foundry.handle_capture_control({"action": "confirm", **common})
                foundry.finish_guided_capture("userStopped")
                sidecar = foundry.stop_timeline_recording()
                data = json.loads(sidecar.read_text())
                self.assertEqual(data["groundTruth"][0]["cell"], "A3")
                self.assertEqual(data["trackingEvents"][0]["cell"], "r9c9")
                self.assertFalse(data["captureComplete"])
                self.assertTrue(Path(directory, "capture.case.json").exists())
                self.assertTrue(Path(directory, "capture.profiles.json").exists())
                self.assertTrue(foundry.tracking_output_paused())
                foundry.handle_capture_control({"action": "resume"})
                self.assertFalse(foundry.tracking_output_paused())
        finally:
            foundry.stop_timeline_recording()
            for key, value in globals_to_restore.items():
                setattr(foundry, key, value)


if __name__ == "__main__":
    unittest.main()
