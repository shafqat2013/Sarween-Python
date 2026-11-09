"""Phone startup ownership and source binding; no devices, UI or sockets."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest

import camera_sources as sources
import main
import v3_tracking
from hardware_profiles import HardwareProfiles
from tests import test_session_startup as models
from tests import test_camera_calibration_flow as calibration_models


DEVICE = "test-iphone-usb-serial"
INDEX = sources.phone_camera_index(DEVICE)
SOURCE = {"type": sources.PHONE_SOURCE, "deviceId": DEVICE}
WIDE_METADATA = {"deviceId": DEVICE, "format": {"lens": "wide", "width": 1080, "height": 1920}}


class SourceTests(unittest.TestCase):
    def test_discovery_is_passive_and_negative_handle_never_reaches_opencv(self):
        with patch("phone_usb.list_phone_cameras", return_value=[
                {"id": DEVICE, "transport": "usb", "label": "Phone"},
                {"id": "wifi-device", "transport": "network"}]), \
             patch.object(sources.cv2, "VideoCapture", side_effect=AssertionError("camera opened")), \
             patch("phone_camera.PhoneCameraCapture") as capture:
            rows = sources.phone_camera_inventory()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["index"], INDEX)
            capture.assert_not_called()
            selection = {"webcam_index": INDEX, "camera_source": SOURCE}
            sources.open_camera_capture(INDEX, selection=selection)
            capture.assert_called_once_with(DEVICE)
            with self.assertRaises(ValueError):
                sources.open_camera_capture(-1)

    def test_saved_identity_is_strong_and_does_not_switch_to_other_phone(self):
        selection = {"webcam_index": INDEX, "camera_source": SOURCE}
        self.assertEqual(sources.source_for_index(INDEX, selection=selection, inventory=[]), SOURCE)
        with self.assertRaises(ValueError):
            sources.source_for_index(INDEX, selection={"webcam_index": INDEX,
                "camera_source": {"type": sources.PHONE_SOURCE, "deviceId": "other-phone"}}, inventory=[])

    def test_new_phone_profile_does_not_copy_or_select_obs_calibration(self):
        with TemporaryDirectory() as directory:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            obs = store.upsert_profile("camera", {"name": "OBS", "sourceHint": "OBS"}, select=True)
            phone = sources.create_phone_profile(store, SOURCE, metadata=WIDE_METADATA)
            self.assertEqual(store.active_profile("camera")["id"], obs["id"])
            self.assertIsNone(phone["calibration"])
            self.assertTrue(sources.profile_matches_phone(phone, SOURCE))
            self.assertFalse(sources.profile_matches_phone(obs, SOURCE))
            self.assertFalse(sources.profile_matches_phone(phone, {**SOURCE, "deviceId": "other"}))
            self.assertNotEqual(sources.create_phone_profile(store, SOURCE, metadata=WIDE_METADATA)["name"], phone["name"])

    def test_stream_key_changes_on_reconnect_but_not_each_frame(self):
        data = {"deviceId": DEVICE, "sessionId": "one", "connectionGeneration": 1,
                "format": {"width": 1080, "height": 1920, "pixelFormat": "jpeg"}, "seq": 1}
        self.assertEqual(sources.stable_stream_key(data), sources.stable_stream_key({**data, "seq": 2,
            "captureTimeNs": 12345, "hostReceiveAgeSeconds": .1}))
        self.assertNotEqual(sources.stable_stream_key(data), sources.stable_stream_key({**data, "sessionId": "two"}))
        self.assertNotEqual(sources.stable_stream_key(data), sources.stable_stream_key({**data,
            "format": {**data["format"], "zoom": 2}}))


class PhoneStartupTests(unittest.TestCase):
    setUp = models.SessionStartupTests.setUp
    model = models.SessionStartupTests.model

    def phone_model(self):
        ui = self.model(mode="self_hosted")
        ui.selector.draft.update(webcam_index=INDEX, camera_source=SOURCE)
        ui.selector.draft["camera_profile_id"] = None
        capture = Mock()
        capture.status = {"fresh": False, "message": "Open Sarween Camera on your iPhone"}
        self.factory = self.stack.enter_context(patch.object(sources, "open_camera_capture", return_value=capture))
        return ui, capture

    def test_wait_stays_cancellable_then_hands_same_reader_to_runtime_once(self):
        ui, capture = self.phone_model()
        ui.begin_connection()
        self.assertEqual(ui.phase, "camera")
        self.assertFalse(ui.cancel_button.disabled)
        self.assertIn("Open Sarween", ui.message.get())
        for _ in range(80):
            self.scheduler.next()
        ui.on_ready.assert_not_called()
        self.factory.assert_called_once()
        capture.read.assert_not_called()
        capture.status = {"fresh": True, "message": "Receiving"}
        self.scheduler.until(lambda: ui.on_ready.called)
        self.assertIs(ui.take_camera_capture(), capture)
        self.assertIsNone(ui.take_camera_capture())
        ui.dispose()
        capture.release.assert_not_called()

    def test_cancel_or_edit_closes_waiting_reader(self):
        for action in ("cancel", "edit_hardware"):
            ui, capture = self.phone_model()
            ui.begin_connection()
            getattr(ui, action)()
            capture.release.assert_called_once()
            self.assertFalse(self.scheduler.pending)
            ui.on_ready.assert_not_called()

    def test_revoked_access_while_phone_waits_releases_capture(self):
        ui, capture = self.phone_model()
        ui.begin_connection()
        ui.access.allowed = False
        self.scheduler.next()
        capture.release.assert_called_once()
        ui.on_ready.assert_not_called()
        self.assertTrue(ui._closed)

    def test_main_transfers_phone_capture_once_and_releases_if_runner_fails(self):
        ui, capture = self.phone_model()
        ui.selection_data = ui.selector.draft
        ui.service.access = ui.access
        ui._camera_capture = capture
        with patch.object(main.s, "initialize"), \
             patch.object(main.s, "selected_mode", main.s.MODE_SELF_HOSTED), \
             patch.object(main.c, "calibrate"), \
             patch("v3_tracking.begin_session", side_effect=RuntimeError("constructor failed")) as begin, \
             patch("v3_tracking.close_pending_recording"):
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                main.run_live_session(ui.service, startup=ui)
        self.assertIs(begin.call_args.kwargs["camera_capture"], capture)
        capture.release.assert_called_once()
        self.assertIsNone(ui.take_camera_capture())


class PhoneCalibrationTests(unittest.TestCase):
    flow = calibration_models.CameraCalibrationFlowTests.flow

    def test_obs_profile_is_rejected_before_phone_camera_open(self):
        with self.flow() as value, \
             patch.object(sources, "source_for_index", return_value=SOURCE), \
             patch.object(sources, "open_camera_capture") as capture:
            with self.assertRaisesRegex(ValueError, "Sarween Camera USB profile"):
                calibration_models.calibration.calibrate_webcam(INDEX, profile_id=value["profile"]["id"],
                    display=calibration_models.DISPLAYS[1], displays=calibration_models.DISPLAYS)
            capture.assert_not_called()

    def test_waiting_phone_calibration_can_be_cancelled_without_saving(self):
        with self.flow(keys=[ord('q')]) as value:
            profile = sources.create_phone_profile(value["store"], SOURCE, metadata=WIDE_METADATA)
            capture = value["capture"]
            capture.read.return_value = False, None
            capture.status = {"message": "Waiting for camera"}
            with patch.object(sources, "source_for_index", return_value=SOURCE), \
                 patch.object(sources, "open_camera_capture", return_value=capture):
                result = calibration_models.calibration.calibrate_webcam(INDEX, profile_id=profile["id"],
                    display=calibration_models.DISPLAYS[1], displays=calibration_models.DISPLAYS)
            self.assertIsNone(result)
            value["solve"].assert_not_called()
            capture.release.assert_called_once()
            self.assertIsNone(value["store"].get_profile("camera", profile["id"])["calibration"])

    def test_reconnect_during_calibration_aborts_before_saving_mixed_views(self):
        with self.flow() as value:
            profile = sources.create_phone_profile(value["store"], SOURCE, metadata=WIDE_METADATA)
            capture = value["capture"]
            image = capture.read.return_value[1]
            sessions = iter(("one", "two"))
            def read():
                capture.last_frame_metadata = {"deviceId": DEVICE, "sessionId": next(sessions),
                    "connectionGeneration": 1, "format": {"width": 320, "height": 240, "lens": "wide"}}
                return True, image
            capture.read.side_effect = read
            with patch.object(sources, "source_for_index", return_value=SOURCE), \
                 patch.object(sources, "open_camera_capture", return_value=capture):
                with self.assertRaisesRegex(ValueError, "stream restarted"):
                    calibration_models.calibration.calibrate_webcam(INDEX, profile_id=profile["id"],
                        display=calibration_models.DISPLAYS[1], displays=calibration_models.DISPLAYS)
            value["solve"].assert_not_called()
            capture.release.assert_called_once()
            self.assertIsNone(value["store"].get_profile("camera", profile["id"])["calibration"])


if __name__ == "__main__":
    unittest.main()
