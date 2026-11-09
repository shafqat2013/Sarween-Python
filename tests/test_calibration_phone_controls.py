"""Calibration phone requests and truthful feedback, without camera or sockets."""
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import Mock, patch
import unittest

import numpy as np

import calibration
import camera_sources as sources
from tests import test_camera_calibration_flow as models


DEVICE = "calibration-control-fixture-phone"
SOURCE = {"type": sources.PHONE_SOURCE, "deviceId": DEVICE}
INDEX = sources.phone_camera_index(DEVICE)
METADATA = {"deviceId": DEVICE, "sessionId": "fixture-session", "seq": 41,
            "connectionGeneration": 1,
            "format": {"width": 320, "height": 240, "lens": "wide", "cameraPosition": "back", "zoom": 1}}


class CalibrationPhoneControlTests(unittest.TestCase):
    @contextmanager
    def flow(self, **kwargs):
        with models.CameraCalibrationFlowTests().flow(**kwargs) as value:
            profile = sources.create_phone_profile(value["store"], SOURCE, metadata=METADATA)
            capture = value["capture"]
            capture.last_frame_metadata = deepcopy(METADATA)
            capture.take_requests.return_value = []
            capture.send_feedback.return_value = True
            value["phone_profile"] = profile
            with patch.object(sources, "source_for_index", return_value=SOURCE), \
                 patch.object(sources, "open_camera_capture", return_value=capture):
                yield value

    def run_flow(self, value, **kwargs):
        return calibration.calibrate_webcam(INDEX, profile_id=value["phone_profile"]["id"],
            camera_source=SOURCE, display=models.DISPLAYS[1], displays=models.DISPLAYS, **kwargs)

    def test_phone_cancel_before_read_preserves_profile_and_clears_feedback_before_release(self):
        with self.flow() as value:
            capture = value["capture"]
            capture.take_requests.return_value = [{"action": "calibration.cancel", "requestId": "cancel-1"}]
            events = Mock()
            events.attach_mock(capture.send_feedback, "feedback")
            events.attach_mock(capture.release, "release")
            before = deepcopy(value["store"].get_profile("camera", value["phone_profile"]["id"]))
            self.assertIsNone(self.run_flow(value))
            capture.read.assert_not_called()
            value["solve"].assert_not_called()
            self.assertEqual(value["store"].get_profile("camera", before["id"]), before)
            self.assertEqual([call[0] for call in events.mock_calls][-2:], ["feedback", "release"])
            terminal = capture.send_feedback.call_args.args[0]["calibration"]
            self.assertFalse(terminal["active"])
            self.assertEqual(terminal["reason"], "cancelled")

    def test_phone_capture_selects_one_eligible_manual_view_and_keeps_actual_lens_binding(self):
        with self.flow(frames=1) as value:
            value["capture"].take_requests.return_value = [{"action": "calibration.capture", "requestId": "capture-1"}]
            saved = self.run_flow(value, auto_capture=False)
            self.assertTrue(saved["bindingVerified"])
            self.assertEqual(sources.phone_profile_lens(saved), "wide")
            self.assertEqual([manual for _gray, manual in value["gates"][0].frames], [True])
            self.assertEqual(value["archive"].return_value.record.call_args.kwargs["source_metadata"], METADATA)
            self.assertTrue(any(call.args[0].get("sourceSeq") == 41 for call in value["capture"].send_feedback.call_args_list))

    def test_early_finish_does_not_solve_and_charuco_ids_never_claim_tv_lock_or_cpl_measurement(self):
        with self.flow(keys=[255, ord("q")], marker_ids=np.array([[10], [11], [12], [13]])) as value:
            capture = value["capture"]
            capture.take_requests.side_effect = [[{"action": "calibration.finish"}, {"action": "cpl.reference"}], []]
            self.assertIsNone(self.run_flow(value, auto_capture=False))
            value["solve"].assert_not_called()
            feedback = [call.args[0] for call in capture.send_feedback.call_args_list]
            needed = [payload for payload in feedback if payload["calibration"]["reason"] == "need_more_views"]
            self.assertEqual(len(needed), 1)
            self.assertFalse(needed[0]["calibration"]["canFinish"])
            self.assertIn("0/3", needed[0]["calibration"]["message"])
            for payload in feedback:
                self.assertEqual(payload["markers"]["ids"], [])
                self.assertEqual(payload["markers"]["count"], 0)
                self.assertFalse(payload["markers"]["locked"])
                self.assertEqual(payload["cpl"]["state"], "unavailable")
                self.assertIn("Finish camera calibration", payload["cpl"]["message"])

    def test_phone_cancel_remains_available_while_waiting_for_first_frame(self):
        with self.flow(keys=[255]) as value:
            capture = value["capture"]
            capture.read.return_value = (False, None)
            capture.status = {"message": "Waiting for a fresh iPhone frame"}
            capture.take_requests.side_effect = [[], [{"action": "calibration.cancel"}]]
            self.assertIsNone(self.run_flow(value))
            capture.read.assert_called_once()
            value["solve"].assert_not_called()
            capture.release.assert_called_once()
            reasons = [call.args[0]["calibration"]["reason"] for call in capture.send_feedback.call_args_list]
            self.assertIn("waiting_camera", reasons)
            self.assertEqual(reasons[-1], "cancelled")

    def test_solver_failure_never_announces_saved_profile_to_phone(self):
        with self.flow(frames=1) as value:
            value["solve"].side_effect = ValueError("No valid calibration solution")
            with self.assertRaisesRegex(ValueError, "No valid calibration"):
                self.run_flow(value)
            profile = value["store"].get_profile("camera", value["phone_profile"]["id"])
            self.assertIsNone(profile["calibration"])
            messages = [call.args[0]["calibration"]["message"] for call in value["capture"].send_feedback.call_args_list]
            self.assertFalse(any("profile saved" in message.lower() for message in messages))
            self.assertFalse(value["capture"].send_feedback.call_args.args[0]["calibration"]["active"])


if __name__ == "__main__":
    unittest.main()
