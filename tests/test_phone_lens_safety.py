"""Physical-lens boundaries exercised with saved profiles and fake USB frames.

Equal pixel dimensions do not imply equal optics. These tests deliberately keep
the image size fixed while changing physical lenses; no phone or camera opens.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import calibration
import camera_sources as sources
import cv_core as core
from camera_profile_runtime import CameraCalibrationChanged, CameraCorrection
from hardware_profiles import HardwareProfiles
from tests.test_phone_cv_core import PhoneCapture
from tests import test_camera_calibration_flow as calibration_models


DEVICE = "independent-lens-test-phone"
SOURCE = {"type": sources.PHONE_SOURCE, "deviceId": DEVICE}
INDEX = sources.phone_camera_index(DEVICE)
IMAGE = np.full((24, 32, 3), 80, np.uint8)
MATRIX = np.array([[100., 0., 16.], [0., 100., 12.], [0., 0., 1.]])


def metadata(lens="wide", *, session="wide-session", generation=1, seq=1):
    return {"source": sources.PHONE_SOURCE, "transport": "usb", "deviceId": DEVICE,
            "sourceId": "fixture:" + lens, "sessionId": session,
            "connectionGeneration": generation, "seq": seq, "captureTimeNs": 1000 + seq,
            "hostReceivedMonotonic": 100., "hostReceiveAgeSeconds": 0.,
            "connected": True, "fresh": True,
            "format": {"width": 32, "height": 24, "fps": 15, "pixelFormat": "jpeg",
                       "lens": lens, "cameraPosition": "back", "zoom": 1}}


class PhysicalLensSafetyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = HardwareProfiles(Path(directory.name) / "profiles.json")

    def profile(self, lens="Back wide"):
        value = self.store.upsert_profile("camera", {"name": "Fixture " + lens,
            "sourceHint": sources.PHONE_PROFILE_HINT, "deviceLabel": DEVICE,
            "lens": lens, "zoom": "1x"}, select=True)
        return self.store.save_camera_calibration(value["id"], MATRIX, np.zeros(5), (32, 24), rms=.1)

    def core_session(self, capture, profile):
        stack = ExitStack()
        self.addCleanup(stack.close)
        selection = {"webcam_index": INDEX, "mode": "self_hosted",
                     "camera_profile_id": profile["id"], "camera_source": SOURCE}
        stack.enter_context(patch.dict("os.environ", {"TRUESIGHT_SOURCE": ""}))
        stack.enter_context(patch.object(core.s, "load_last_selection", return_value=selection))
        stack.enter_context(patch.object(core, "CameraCorrection", side_effect=lambda **kwargs:
            CameraCorrection(profiles=self.store, **kwargs)))
        stack.enter_context(patch.object(core.fo, "set_grid_params"))
        stack.enter_context(patch.object(core.cv2, "VideoCapture", side_effect=AssertionError("Native camera opened")))
        session = core.CVCoreSession(camera_index=INDEX, marker_mode="legacy",
            warp_w=32, warp_h=24, grid_w=8, grid_h=6, camera_capture=capture)
        self.addCleanup(session.close)
        return session

    def test_legacy_back_wide_calibration_is_wide_only(self):
        profile = self.profile()
        correction = CameraCorrection(profiles=self.store, profile_id=profile["id"])
        matrix, _ = correction.resolve((32, 24), source_metadata=metadata())
        np.testing.assert_array_equal(matrix, MATRIX)
        for lens in ("ultra-wide", "telephoto"):
            with self.subTest(lens=lens):
                self.assertIsNotNone(sources.phone_profile_mismatch_reason(profile, SOURCE, metadata(lens)))
                self.assertEqual(correction.resolve((32, 24), source_metadata=metadata(lens)), (None, None))
                self.assertIsNone(correction.matrix)

    def test_cached_matrix_cannot_survive_missing_lens_or_other_phone(self):
        profile = self.profile()
        correction = CameraCorrection(profiles=self.store, profile_id=profile["id"])
        missing_lens = metadata()
        del missing_lens["format"]["lens"]
        for bad in (None, missing_lens, {**metadata(), "deviceId": "another-phone"}):
            with self.subTest(metadata=bad):
                self.assertIsNotNone(correction.resolve((32, 24), source_metadata=metadata())[0])
                correction.applied = True
                self.assertEqual(correction.resolve((32, 24), source_metadata=bad), (None, None))
                self.assertFalse(correction.applied)

    def test_fresh_ultra_and_tele_profiles_are_blank_and_leave_active_wide_unchanged(self):
        wide = self.profile()
        for lens in ("ultra-wide", "telephoto"):
            with self.subTest(lens=lens):
                created = sources.create_phone_profile(self.store, SOURCE, metadata=metadata(lens))
                self.assertEqual(sources.phone_profile_lens(created), lens)
                self.assertIsNone(created["calibration"])
                self.assertFalse(created["bindingVerified"])
                self.assertEqual(self.store.active_profile("camera")["id"], wide["id"])
                self.assertTrue(sources.profile_compatible_with_source(created, SOURCE, metadata=metadata(lens)))
                self.assertFalse(sources.profile_compatible_with_source(created, SOURCE, metadata=metadata()))

    def test_first_same_size_wrong_lens_rejects_before_undistort_and_releases(self):
        profile = self.profile()
        for lens in ("ultra-wide", "telephoto"):
            capture = PhoneCapture([(metadata(lens), IMAGE)])
            with self.subTest(lens=lens), patch.object(core.cv2, "undistort") as undistort:
                with self.assertRaises(CameraCalibrationChanged):
                    self.core_session(capture, profile)
                undistort.assert_not_called()
                self.assertEqual(capture.release_count, 1)

    def test_live_same_size_new_session_lens_change_rejects_before_next_correction(self):
        profile = self.profile()
        capture = PhoneCapture([(metadata(), IMAGE),
            (metadata("ultra-wide", session="ultra-session", generation=2, seq=1), IMAGE)])
        with patch.object(core.cv2, "undistort", side_effect=lambda image, *_: image.copy()) as undistort:
            session = self.core_session(capture, profile)
            self.assertEqual(undistort.call_count, 1)
            with self.assertRaises(CameraCalibrationChanged):
                next(session.frames())
            self.assertEqual(undistort.call_count, 1)
            self.assertIsNone(session.cam_mtx)
            self.assertFalse(session.camera_calibration_metadata()["applied"])

    def test_matching_ultra_profile_corrects_first_frame(self):
        profile = self.profile("Back ultra-wide")
        capture = PhoneCapture([(metadata("ultra-wide"), IMAGE)])
        with patch.object(core.cv2, "undistort", side_effect=lambda image, *_: image.copy()) as undistort:
            session = self.core_session(capture, profile)
            undistort.assert_called_once()
            self.assertTrue(session.camera_calibration_metadata()["applied"])

    def test_save_boundary_rejects_wrong_lens_in_source_and_packaged_calibration(self):
        path = Path(__file__).resolve().parents[1] / "app_bundle_overrides" / "calibration.py"
        spec = importlib.util.spec_from_file_location("independent_bundle_lens_calibration", path)
        packaged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(packaged)
        profile = self.profile()
        before = deepcopy(self.store.get_profile("camera", profile["id"]))
        for module in (calibration, packaged):
            with self.subTest(module=module.__name__):
                with self.assertRaisesRegex(ValueError, "Ultra Wide"):
                    module.save_camera_calibration_result(profile["id"], MATRIX * 2, np.zeros(5),
                        (32, 24), profiles=self.store, source_metadata=metadata("ultra-wide"))
                self.assertEqual(self.store.get_profile("camera", profile["id"]), before)

    def test_calibration_rejects_wrong_physical_lens_before_collecting_or_solving(self):
        flow = calibration_models.CameraCalibrationFlowTests().flow
        with flow() as value:
            profile = value["store"].upsert_profile("camera", {"name": "USB Wide",
                "sourceHint": sources.PHONE_PROFILE_HINT, "deviceLabel": DEVICE,
                "lens": "Back wide", "zoom": "1x"})
            capture = value["capture"]
            capture.last_frame_metadata = metadata("ultra-wide")
            before = deepcopy(value["store"].get_profile("camera", profile["id"]))
            with patch.object(sources, "source_for_index", return_value=SOURCE), \
                 patch.object(sources, "open_camera_capture", return_value=capture):
                with self.assertRaisesRegex(ValueError, "Ultra Wide"):
                    calibration.calibrate_webcam(INDEX, profile_id=profile["id"],
                        display=calibration_models.DISPLAYS[1], displays=calibration_models.DISPLAYS,
                        camera_source=SOURCE)
            self.assertEqual(value["gates"][0].frames, [])
            value["solve"].assert_not_called()
            capture.release.assert_called_once()
            self.assertEqual(value["store"].get_profile("camera", profile["id"]), before)


if __name__ == "__main__":
    unittest.main()
