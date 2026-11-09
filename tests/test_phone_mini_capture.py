"""One-off mini/background captures respect phone selection and calibration."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import mini_tracking as minis
from camera_sources import PHONE_SOURCE, phone_camera_index
from hardware_profiles import HardwareProfiles


DEVICE = "mini-capture-phone"
INDEX = phone_camera_index(DEVICE)
FRAME = np.full((24, 32, 3), 80, np.uint8)


class PhoneMiniCaptureTests(unittest.TestCase):
    def selection(self, profile_id=None):
        return {"webcam_index": INDEX, "camera_profile_id": profile_id,
                "camera_source": {"type": PHONE_SOURCE, "deviceId": DEVICE}}

    def test_all_three_capture_entries_route_phone_factory_and_skip_native_warmups(self):
        for function in (minis.capture_background, minis.capture_background_full, minis.capture_mini):
            with self.subTest(function=function.__name__), ExitStack() as stack:
                capture = Mock()
                capture.read.return_value = True, FRAME.copy()
                stack.enter_context(patch.object(minis.s, "load_last_selection", return_value=self.selection()))
                opened = stack.enter_context(patch.object(minis, "open_camera_capture", return_value=capture))
                native = stack.enter_context(patch.object(minis.cv2, "VideoCapture", side_effect=AssertionError("Reserved phone index reached OpenCV")))
                stack.enter_context(patch.object(minis, "get_camera_params", return_value=(None, None)))
                stack.enter_context(patch.object(minis, "save_mini_from_frame_and_contour", return_value="saved"))
                sleep = stack.enter_context(patch.object(minis.time, "sleep"))
                self.assertIsNotNone(function(camera_index=INDEX))
                opened.assert_called_once()
                self.assertEqual(opened.call_args.args, (INDEX,))
                capture.read.assert_called_once()
                capture.release.assert_called_once()
                native.assert_not_called()
                sleep.assert_not_called()

    def test_failed_phone_read_releases_and_reports_explicit_retry(self):
        capture = Mock()
        capture.read.return_value = False, None
        with patch.object(minis.s, "load_last_selection", return_value=self.selection()), \
             patch.object(minis, "open_camera_capture", return_value=capture), \
             patch("builtins.print") as printed:
            self.assertIsNone(minis.capture_background_full())
        capture.release.assert_called_once()
        self.assertTrue(any("retry capture" in str(call) for call in printed.call_args_list))

    def test_native_source_still_uses_shared_opencv_factory_and_original_warmup(self):
        capture = Mock()
        capture.read.return_value = True, FRAME.copy()
        with patch.object(minis.s, "load_last_selection", return_value={"webcam_index": 2}), \
             patch.object(minis.cv2, "VideoCapture", return_value=capture) as opened, \
             patch.object(minis, "get_camera_params", return_value=(None, None)), \
             patch.object(minis.time, "sleep") as sleep:
            self.assertIsNotNone(minis.capture_background_full())
        opened.assert_called_once_with(2)
        self.assertEqual(capture.read.call_count, 11)
        capture.release.assert_called_once()
        sleep.assert_called_once_with(1)

    def test_phone_one_off_calibration_requires_same_usb_serial_and_source_type(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            for number, (hint, serial, expected) in enumerate((
                ("OBS", DEVICE, False), ("Sarween Camera USB", "different-phone", False),
                ("Sarween Camera USB", DEVICE, True))):
                profile = store.upsert_profile("camera", {"name": f"Source {number}",
                    "sourceHint": hint, "deviceLabel": serial, "lens": "Back wide"}, select=True)
                matrix = np.array([[100., 0, 16], [0, 100., 12], [0, 0, 1]])
                store.save_camera_calibration(profile["id"], matrix, np.zeros(5), (32, 24), rms=.1)
                with self.subTest(hint=hint, serial=serial), \
                     patch.object(minis.s, "load_last_selection", return_value=self.selection(profile["id"])), \
                     patch("hardware_profiles.HardwareProfiles", return_value=store):
                    result = minis.get_camera_params((32, 24), camera_index=INDEX,
                        source_metadata={"deviceId": DEVICE, "format": {"lens": "wide", "width": 32, "height": 24}})
                    self.assertEqual(result[0] is not None, expected)

    def test_native_one_off_rejects_usb_profile_but_preserves_confirmed_obs_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            for hint, expected in (("Sarween Camera USB", False), ("OBS", True)):
                profile = store.upsert_profile("camera", {"name": hint, "sourceHint": hint,
                    "deviceLabel": DEVICE}, select=True)
                matrix = np.array([[100., 0, 16], [0, 100., 12], [0, 0, 1]])
                store.save_camera_calibration(profile["id"], matrix, np.zeros(5), (32, 24), rms=.1)
                selection = {"webcam_index": 2, "camera_profile_id": profile["id"]}
                with self.subTest(hint=hint), \
                     patch.object(minis.s, "load_last_selection", return_value=selection), \
                     patch("hardware_profiles.HardwareProfiles", return_value=store):
                    result = minis.get_camera_params((32, 24), camera_index=2)
                    self.assertEqual(result[0] is not None, expected)


if __name__ == "__main__":
    unittest.main()
