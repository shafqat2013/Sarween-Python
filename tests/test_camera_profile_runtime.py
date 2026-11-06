"""Profile-bound lens correction and calibration saving without camera or Tk."""

from pathlib import Path
from contextlib import ExitStack, contextmanager
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from camera_profile_runtime import CameraCorrection, camera_params, legacy_offline_camera_params
from hardware_profiles import HardwareProfiles


MATRIX = np.array([[200.0, 0, 160], [0, 200.0, 120], [0, 0, 1]])
DISTORTION = np.array([[0.1, 0, 0, 0, 0]])


class CameraProfileRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = HardwareProfiles(Path(self.directory.name) / "profiles.json")

    def camera(self, name="Phone", calibrated=True):
        profile = self.store.upsert_profile("camera", {"name": name, "sourceHint": "OBS",
            "lens": "Wide", "zoom": "1x", "transformNotes": "Portrait; no crop"}, select=True)
        if calibrated:
            profile = self.store.save_camera_calibration(profile["id"], MATRIX, DISTORTION, (320, 240), rms=0.2)
        return profile

    @contextmanager
    def calibration_presentation(self, calibration):
        """Keep the real calibration runner isolated from native windows."""
        display = {"index": 1, "x": 1280, "y": 0, "width": 640, "height": 480}
        with ExitStack() as stack:
            stack.enter_context(patch.object(calibration, "_calibration_displays", return_value=(display, None)))
            stack.enter_context(patch.object(calibration, "display_charuco", return_value="CharucoBoard"))
            stack.enter_context(patch.object(calibration, "show_calibration_preview"))
            stack.enter_context(patch.object(calibration, "close_calibration_windows"))
            stack.enter_context(patch.object(calibration, "_save_calibration_diagnostics"))
            yield

    def session(self, correction):
        import cv_core
        session = cv_core.CVCoreSession.__new__(cv_core.CVCoreSession)
        session.source_frames_undistorted = False
        session._camera_correction = correction
        session._legacy_offline_calibration = False
        session._legacy_correction_applied = False
        session.cam_mtx = session.cam_dist = None
        session.camera_calibration_status = "Waiting"
        return session

    def test_calibration_requires_explicit_selection_and_actual_frame_dimensions(self):
        profile = self.camera()
        correction = CameraCorrection(profiles=self.store)
        matrix, distortion = correction.resolve((320, 240))
        np.testing.assert_array_equal(matrix, MATRIX)
        np.testing.assert_array_equal(distortion.ravel(), DISTORTION.ravel())
        self.assertEqual(correction.snapshot()["profileId"], profile["id"])
        self.assertEqual(correction.resolve((1920, 1080)), (None, None))
        self.assertIn("320", correction.reason)
        self.store.select_profile("camera", None)
        self.assertEqual(camera_params((320, 240), profiles=self.store), (None, None))
        self.assertEqual(camera_params(profiles=self.store), (None, None))

    def test_camera_choice_is_frozen_for_the_session_and_refreshed_next_session(self):
        first = self.camera("First")
        current = CameraCorrection(profiles=self.store)
        second = self.camera("Second")
        self.assertEqual(current.profile_id, first["id"])
        self.assertIsNotNone(current.resolve((320, 240))[0])
        self.assertEqual(CameraCorrection(profiles=self.store).profile_id, second["id"])
        no_profile = CameraCorrection(profiles=self.store, profile_id=None)
        self.assertEqual(no_profile.resolve((320, 240)), (None, None))

    def test_missing_and_unverified_profiles_never_use_legacy_globals(self):
        profile = self.camera(calibrated=False)
        self.assertEqual(CameraCorrection(profiles=self.store).resolve((320, 240)), (None, None))
        self.store.save_camera_calibration(profile["id"], MATRIX, DISTORTION, (320, 240), binding_verified=False)
        correction = CameraCorrection(profiles=self.store)
        self.assertEqual(correction.resolve((320, 240)), (None, None))
        self.assertIn("unverified", correction.reason)
        with patch("numpy.load", side_effect=AssertionError("Legacy file read")):
            self.assertEqual(correction.resolve((640, 480)), (None, None))

    def test_same_size_cache_revalidates_profile_changes_and_deletion(self):
        profile = self.camera()
        correction = CameraCorrection(profiles=self.store)
        self.assertIsNotNone(correction.resolve((320, 240))[0])
        self.store.upsert_profile("camera", {"lens": "Telephoto"}, profile_id=profile["id"])
        self.assertEqual(correction.resolve((320, 240)), (None, None))
        self.assertIn("unverified", correction.reason)
        replacement = MATRIX.copy()
        replacement[0, 0] = 210
        self.store.save_camera_calibration(profile["id"], replacement, DISTORTION, (320, 240))
        self.assertEqual(correction.resolve((320, 240))[0][0, 0], 210)
        self.store.delete_profile("camera", profile["id"])
        self.assertEqual(correction.resolve((320, 240)), (None, None))
        self.assertIn("missing", correction.reason)

    def test_live_geometry_change_requires_new_registration_before_processing(self):
        import cv_core
        profile = self.camera()
        session = self.session(CameraCorrection(profiles=self.store))
        frame = np.zeros((240, 320, 3), np.uint8)
        with patch.object(cv_core.cv2, "undistort", return_value=frame):
            session._maybe_undistort(frame)
        session.BG_cam = {"bgr": frame}
        self.store.upsert_profile("camera", {"transformNotes": "Crop changed"}, profile_id=profile["id"])
        with patch.object(cv_core.cv2, "undistort") as undistort:
            with self.assertRaisesRegex(ValueError, "Restart the live session"):
                session._maybe_undistort(frame)
            undistort.assert_not_called()

    def test_live_pipeline_applies_verified_profile_and_reports_snapshot(self):
        import cv_core
        self.camera()
        session = self.session(CameraCorrection(profiles=self.store))
        frame = np.zeros((240, 320, 3), np.uint8)
        result = np.ones_like(frame)
        with patch.object(cv_core.cv2, "undistort", return_value=result) as undistort:
            self.assertIs(session._maybe_undistort(frame), result)
        np.testing.assert_array_equal(undistort.call_args.args[1], MATRIX)
        self.assertTrue(session.camera_calibration_metadata()["applied"])
        self.assertEqual(session.camera_calibration_metadata()["frameSize"], [320, 240])
        with patch.object(cv_core.cv2, "undistort") as undistort:
            wrong_size = np.zeros((480, 640, 3), np.uint8)
            self.assertIs(session._maybe_undistort(wrong_size), wrong_size)
            undistort.assert_not_called()
        self.assertFalse(session.camera_calibration_metadata()["applied"])

    def test_processed_recording_is_not_corrected_again(self):
        import cv_core
        correction = Mock()
        session = self.session(correction)
        session.source_frames_undistorted = True
        frame = np.zeros((240, 320, 3), np.uint8)
        with patch.object(cv_core.cv2, "undistort") as undistort:
            self.assertIs(session._maybe_undistort(frame), frame)
            undistort.assert_not_called()
        correction.resolve.assert_not_called()

    def test_failed_undistortion_never_falls_back_to_different_geometry(self):
        import cv_core
        from camera_profile_runtime import CameraCalibrationChanged
        self.camera()
        session = self.session(CameraCorrection(profiles=self.store))
        frame = np.zeros((240, 320, 3), np.uint8)
        with patch.object(cv_core.cv2, "undistort", side_effect=ValueError("invalid correction")):
            with self.assertRaisesRegex(CameraCalibrationChanged, "Lens correction failed"):
                session._maybe_undistort(frame)
        self.assertFalse(session.camera_calibration_metadata()["applied"])

    def test_live_session_only_uses_profile_for_the_confirmed_camera_selection(self):
        import cv_core
        frame = np.zeros((240, 320, 3), np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = lambda: (True, frame.copy())
        correction = Mock(reason="No calibration", profile_id=None, applied=False)
        correction.resolve.return_value = (None, None)
        selection = {"webcam_index": 2, "camera_profile_id": "confirmed-profile", "mode": "foundry"}
        for camera_index, expected_id in ((2, "confirmed-profile"), (0, None)):
            with self.subTest(camera_index=camera_index), \
                 patch.object(cv_core.s, "load_last_selection", return_value=selection), \
                 patch.object(cv_core.cv2, "VideoCapture", return_value=capture), \
                 patch.object(cv_core, "CameraCorrection", return_value=correction) as resolver, \
                 patch.object(cv_core, "legacy_offline_camera_params") as legacy, \
                 patch.object(cv_core.time, "sleep"), patch.object(cv_core.fo, "set_grid_params"):
                session = cv_core.CVCoreSession(camera_index=camera_index,
                    grid_w=20, grid_h=12, warp_w=800, warp_h=480)
                resolver.assert_called_once_with(profile_id=expected_id)
                legacy.assert_not_called()
                session.close()

    def test_only_raw_file_replay_uses_archived_legacy_arrays(self):
        import cv_core
        frame = np.zeros((240, 320, 3), np.uint8)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = lambda: (True, frame.copy())
        capture.get.return_value = 30.0
        for processed in (False, True):
            with self.subTest(processed=processed), \
                 patch.object(cv_core.s, "load_last_selection", return_value={}), \
                 patch.object(cv_core.cv2, "VideoCapture", return_value=capture), \
                 patch.object(cv_core, "CameraCorrection") as profile, \
                 patch.object(cv_core, "legacy_offline_camera_params", return_value=(MATRIX, DISTORTION)) as legacy, \
                 patch.object(cv_core.cv2, "undistort", side_effect=lambda value, *_args: value), \
                 patch.object(cv_core.fo, "set_grid_params"):
                session = cv_core.CVCoreSession(source_path="historical.mp4", source_frames_undistorted=processed,
                    grid_w=20, grid_h=12, warp_w=800, warp_h=480)
                self.assertEqual(legacy.call_count, 0 if processed else 1)
                profile.assert_not_called()
                session.close()

    def test_legacy_reader_is_separate_and_rejects_invalid_arrays(self):
        root = Path(self.directory.name)
        np.save(root / "camera_matrix.npy", MATRIX)
        np.save(root / "dist_coeffs.npy", DISTORTION)
        with patch("app_paths.data_path", side_effect=lambda name: root / name):
            np.testing.assert_array_equal(legacy_offline_camera_params()[0], MATRIX)
            np.save(root / "dist_coeffs.npy", np.array([float("nan")] * 5))
            self.assertEqual(legacy_offline_camera_params(), (None, None))

    def test_saving_new_calibration_changes_named_profile_only(self):
        from calibration import save_camera_calibration_result
        profile = self.camera(calibrated=False)
        saved = save_camera_calibration_result(profile["id"], MATRIX, DISTORTION, (320, 240),
                                               rms=0.25, profiles=self.store)
        self.assertEqual(saved["id"], profile["id"])
        self.assertEqual(saved["calibration"]["imageSize"], [320, 240])
        self.assertTrue(saved["bindingVerified"])
        self.assertFalse((Path(self.directory.name) / "camera_matrix.npy").exists())
        with self.assertRaisesRegex(ValueError, "Choose"):
            save_camera_calibration_result(None, MATRIX, DISTORTION, (320, 240), profiles=self.store)

    def test_calibration_requires_profile_before_opening_any_camera(self):
        import calibration
        with patch("hardware_profiles.HardwareProfiles", return_value=self.store), \
             patch.object(calibration.cv2, "VideoCapture") as camera:
            with self.assertRaisesRegex(ValueError, "choose"):
                calibration.calibrate_webcam(0)
            camera.assert_not_called()

    def test_calibration_requires_access_before_opening_any_camera(self):
        import calibration
        access = Mock()
        access.require.side_effect = PermissionError("Sign in")
        with patch.object(calibration.cv2, "VideoCapture") as camera:
            with self.assertRaisesRegex(PermissionError, "Sign in"):
                calibration.calibrate_webcam(0, access=access)
            camera.assert_not_called()

    def test_cancelled_calibration_releases_camera_without_replacing_profile(self):
        import calibration
        profile = self.camera(calibrated=False)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        with self.calibration_presentation(calibration), \
             patch("hardware_profiles.HardwareProfiles", return_value=self.store), \
             patch.object(calibration.cv2, "VideoCapture", return_value=capture), \
             patch.object(calibration, "_ensure_window"), \
             patch.object(calibration.cv2, "imshow"), \
             patch.object(calibration.cv2, "waitKey", return_value=ord("q")), \
             patch.object(calibration.cv2, "destroyAllWindows"):
            self.assertIsNone(calibration.calibrate_webcam(2, profile_id=profile["id"]))
        capture.release.assert_called_once()
        self.assertIsNone(self.store.get_profile("camera", profile["id"])["calibration"])

    def test_calibration_feed_failure_is_bounded_and_releases_camera(self):
        import calibration
        profile = self.camera(calibrated=False)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with self.calibration_presentation(calibration), \
             patch("hardware_profiles.HardwareProfiles", return_value=self.store), \
             patch.object(calibration.cv2, "VideoCapture", return_value=capture), \
             patch.object(calibration.cv2, "destroyAllWindows"), patch.object(calibration.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "feed stopped"):
                calibration.calibrate_webcam(2, profile_id=profile["id"])
        capture.release.assert_called_once()
        self.assertEqual(capture.read.call_count, 10)
        self.assertIsNone(self.store.get_profile("camera", profile["id"])["calibration"])

    def test_revocation_during_solve_does_not_save_a_verified_profile(self):
        import calibration
        profile = self.camera(calibrated=False)
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        access = Mock()
        access.require.side_effect = [None, None, PermissionError("Access revoked")]
        gate = Mock(automatic=True)
        gate.observe.return_value = {"capture": True, "message": "Captured", "reason": "captured"}
        with self.calibration_presentation(calibration), \
             patch("hardware_profiles.HardwareProfiles", return_value=self.store), \
             patch.object(calibration, "CalibrationCaptureGate", return_value=gate), \
             patch.object(calibration, "NUM_FRAMES", 1), \
             patch.object(calibration.cv2, "VideoCapture", return_value=capture), \
             patch.object(calibration, "_ensure_window"), \
             patch.object(calibration.cv2, "imshow"), \
             patch.object(calibration.cv2, "waitKey", return_value=ord("c")), \
             patch.object(calibration.cv2, "destroyAllWindows"), \
             patch.object(calibration.cv2.aruco, "detectMarkers", return_value=([], np.array([[1]]), [])), \
             patch.object(calibration.cv2.aruco, "drawDetectedMarkers"), \
             patch.object(calibration.cv2.aruco, "interpolateCornersCharuco", return_value=(4, np.zeros((4, 1, 2)), np.arange(4))), \
             patch.object(calibration.cv2.aruco, "calibrateCameraCharuco", return_value=(0.2, MATRIX, DISTORTION, [], [])), \
             patch.object(calibration, "save_camera_calibration_result") as save:
            with self.assertRaisesRegex(PermissionError, "Access revoked"):
                calibration.calibrate_webcam(2, profile_id=profile["id"], access=access)
            save.assert_not_called()
        capture.release.assert_called_once()
        self.assertIsNone(self.store.get_profile("camera", profile["id"])["calibration"])
    def test_hardware_selection_preserves_explicit_no_profile_and_other_settings(self):
        import setup
        config = Path(self.directory.name) / "hardware.json"
        with patch.object(setup, "CONFIG_FILE", str(config)):
            setup.save_last_selection(webcam_index=2, mode="foundry", camera_profile_id="profile-one")
            setup.save_last_selection(display_index=1)
            self.assertEqual(setup.load_last_selection()["camera_profile_id"], "profile-one")
            setup.save_last_selection(camera_profile_id=None)
            selected = setup.load_last_selection()
            self.assertIsNone(selected["camera_profile_id"])
            self.assertEqual(selected["webcam_index"], 2)
            self.assertEqual(selected["mode"], "foundry")

    def test_one_shot_mini_capture_checks_the_selected_camera_pairing(self):
        import mini_tracking
        selection = {"webcam_index": 2, "camera_profile_id": "confirmed"}
        for index, expected in ((None, "confirmed"), (2, "confirmed"), (0, None)):
            with self.subTest(index=index), \
                 patch.object(mini_tracking.s, "load_last_selection", return_value=selection), \
                 patch("camera_profile_runtime.camera_params", return_value=(None, None)) as params:
                mini_tracking.get_camera_params((320, 240), camera_index=index)
                params.assert_called_once_with((320, 240), profile_id=expected)

    def test_camera_profile_pairing_requires_confirmation_for_every_selected_profile(self):
        from setup import camera_profile_pairing_confirmed
        self.assertTrue(camera_profile_pairing_confirmed(None, False))
        self.assertTrue(camera_profile_pairing_confirmed("camera-one", True))
        self.assertFalse(camera_profile_pairing_confirmed("camera-one", False))
        self.assertFalse(camera_profile_pairing_confirmed("camera-one", None))


if __name__ == "__main__":
    unittest.main()
