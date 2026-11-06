"""Phone source handoff, calibration binding and loss holds without devices."""
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cv_core as core
from camera_profile_runtime import CameraCalibrationChanged, CameraCorrection
from camera_sources import PHONE_SOURCE, phone_camera_index
from hardware_profiles import HardwareProfiles


DEVICE = "test-usb-serial"
INDEX = phone_camera_index(DEVICE)
IMAGE = np.full((24, 32, 3), 80, np.uint8)


def meta(session="one", generation=1, seq=0):
    return {"source": PHONE_SOURCE, "transport": "usb", "deviceId": DEVICE,
            "sessionId": session, "connectionGeneration": generation, "seq": seq,
            "captureTimeNs": 9_000_000_000_000_000 + seq,
            "hostReceivedMonotonic": 100. + seq / 15,
            "hostReceiveAgeSeconds": .001 * seq, "connected": True, "fresh": True,
            "format": {"width": 32, "height": 24, "fps": 15, "pixelFormat": "jpeg"}}


class PhoneCapture:
    def __init__(self, rows, placeholder_size=(32, 24)):
        self.rows = iter(rows)
        self.read_count = 0
        self.release_count = 0
        self.last_frame_metadata = None
        self.status = {"connected": False, "fresh": False, "message": "Phone reconnecting"}
        self.placeholder_size = placeholder_size

    def read(self):
        self.read_count += 1
        row = next(self.rows, None)
        if row is None:
            self.status = {"connected": False, "fresh": False, "message": "Phone reconnecting"}
            return False, None
        metadata, image = row
        self.last_frame_metadata = deepcopy(metadata)
        self.status = deepcopy(metadata)
        return True, image.copy()

    def isOpened(self):
        return not self.release_count

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: self.placeholder_size[0],
                cv2.CAP_PROP_FRAME_HEIGHT: self.placeholder_size[1], cv2.CAP_PROP_FPS: 15}.get(prop, 0)

    def release(self):
        self.release_count += 1


class PhoneCoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = HardwareProfiles(Path(self.directory.name) / "profiles.json")

    def profile(self, *, source="Sarween Camera USB", serial=DEVICE):
        profile = self.store.upsert_profile("camera", {"name": f"Test phone {len(self.store.list_profiles('camera'))}", "sourceHint": source,
            "deviceLabel": serial, "lens": "wide", "zoom": "1x"}, select=True)
        matrix = np.array([[100., 0, 16], [0, 100., 12], [0, 0, 1]])
        return self.store.save_camera_calibration(profile["id"], matrix, np.zeros(5), (32, 24), rms=.1)

    def create(self, capture, *, profile=None, inject=True):
        selection = {"webcam_index": INDEX, "mode": "self_hosted", "camera_profile_id": profile["id"] if profile else None,
                     "camera_source": {"type": PHONE_SOURCE, "deviceId": DEVICE}}
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict("os.environ", {"TRUESIGHT_SOURCE": ""}))
        stack.enter_context(patch.object(core.s, "load_last_selection", return_value=selection))
        def correction(**kwargs):
            kwargs.setdefault("profiles", self.store)
            return CameraCorrection(**kwargs)
        stack.enter_context(patch.object(core, "CameraCorrection", side_effect=correction))
        stack.enter_context(patch.object(core.fo, "set_grid_params"))
        native = stack.enter_context(patch.object(core.cv2, "VideoCapture", side_effect=AssertionError("Native camera opened")))
        factory = stack.enter_context(patch.object(core, "open_camera_capture", return_value=capture))
        sleep = stack.enter_context(patch.object(core.time, "sleep"))
        session = core.CVCoreSession(camera_index=INDEX, marker_mode="legacy", warp_w=32, warp_h=24,
                                      grid_w=8, grid_h=6, camera_capture=capture if inject else None)
        self.addCleanup(session.close)
        return session, factory, native, sleep

    def lock_pipeline(self, session, stack):
        def lock(cam):
            session.last_mask_cam = np.full(cam.shape[:2], 255, np.uint8)
            session.last_mask_warp = np.full((24, 32), 255, np.uint8)
            session.H_saved = np.array([[1., 0, 1], [0, 1., 1], [0, 0, 1]], np.float32)
            session.last_marker_count = 4
        stack.enter_context(patch.object(session, "_update_lock_if_due", side_effect=lock))
        return stack.enter_context(patch.object(session, "_compute_shared_masks", side_effect=lambda cam: {
            "warp_blur": np.zeros((24, 32), np.uint8), "warp_bgr": cam.copy(), "bg_warp_u8": None,
            "motion_warp": None, "motion_cam": None, "shadowfree_cam": None, "final_mask_cam": None,
            "raw_motion_ratio": 0., "largest_motion_area": 0.}))

    def test_injected_capture_has_one_background_read_no_reopen_or_warmup(self):
        capture = PhoneCapture([(meta(), IMAGE)])
        session, factory, native, sleep = self.create(capture)
        self.assertIs(session.cap, capture)
        self.assertEqual(capture.read_count, 1)
        factory.assert_not_called()
        native.assert_not_called()
        sleep.assert_not_called()

    def test_reserved_phone_index_routes_through_factory_not_opencv(self):
        capture = PhoneCapture([(meta(), IMAGE)])
        _, factory, native, _ = self.create(capture, inject=False)
        self.assertEqual(factory.call_args.args, (INDEX,))
        native.assert_not_called()

    def test_obs_or_other_phone_profile_is_never_applied_at_same_resolution(self):
        for source, serial in (("OBS", DEVICE), ("Sarween Camera USB", "other-phone")):
            with self.subTest(source=source, serial=serial):
                profile = self.profile(source=source, serial=serial)
                capture = PhoneCapture([(meta(), IMAGE)])
                with patch.object(core.cv2, "undistort") as undistort:
                    session, *_ = self.create(capture, profile=profile)
                    undistort.assert_not_called()
                snapshot = session.camera_calibration_metadata()
                self.assertIsNone(snapshot["profileId"])
                self.assertFalse(snapshot["applied"])
                self.assertIn("OBS", snapshot["reason"])

    def test_matching_phone_profile_applies_and_remains_separate_from_stream_clocks(self):
        profile = self.profile()
        capture = PhoneCapture([(meta(), IMAGE)])
        with patch.object(core.cv2, "undistort", side_effect=lambda image, *_: image) as undistort:
            session, *_ = self.create(capture, profile=profile)
        undistort.assert_called_once()
        snapshot = session.camera_calibration_metadata()
        self.assertEqual(snapshot["profileId"], profile["id"])
        self.assertTrue(snapshot["applied"])
        self.assertNotIn("captureTimeNs", snapshot)
        self.assertNotIn("hostReceiveAgeSeconds", snapshot)

    def test_usb_profile_is_disabled_on_native_obs_but_manual_native_profile_still_applies(self):
        for hint, expected in (("Sarween Camera USB", False), ("OBS", True)):
            profile = self.profile(source=hint)
            capture = PhoneCapture([(meta(), IMAGE)])
            selection = {"webcam_index": 2, "camera_profile_id": profile["id"], "mode": "self_hosted"}
            def correction(**kwargs):
                kwargs.setdefault("profiles", self.store)
                return CameraCorrection(**kwargs)
            with self.subTest(hint=hint), \
                 patch.dict("os.environ", {"TRUESIGHT_SOURCE": ""}), \
                 patch.object(core.s, "load_last_selection", return_value=selection), \
                 patch.object(core, "CameraCorrection", side_effect=correction), \
                 patch.object(core.fo, "set_grid_params"), \
                 patch.object(core.cv2, "undistort", side_effect=lambda image, *_: image) as undistort:
                session = core.CVCoreSession(camera_index=2, marker_mode="legacy", warp_w=32, warp_h=24,
                    grid_w=8, grid_h=6, camera_capture=capture)
                self.addCleanup(session.close)
                self.assertEqual(undistort.call_count, int(expected))
                snapshot = session.camera_calibration_metadata()
                self.assertEqual(snapshot["applied"], expected)
                if not expected:
                    self.assertIsNone(snapshot["profileId"])
                    self.assertIn("direct iPhone", snapshot["reason"])

    def test_losses_yield_diagnostic_holds_and_reconnect_rebinds_once_without_capture_clocks(self):
        capture = PhoneCapture([(meta(seq=0), IMAGE), (meta(seq=1), IMAGE), None, None,
            (meta("two", 2, 0), IMAGE), (meta("two", 2, 1), IMAGE)])
        session, *_ = self.create(capture)
        session._recorder = Mock()
        session.recording_state_provider = lambda: {}
        with ExitStack() as stack:
            shared = self.lock_pipeline(session, stack)
            timeline = stack.enter_context(patch.object(core.fo, "record_timeline_frame"))
            stack.enter_context(patch.object(core.fo, "record_checkpoint"))
            invalid = stack.enter_context(patch.object(session, "_invalidate_registration", wraps=session._invalidate_registration))
            generator = session.frames()
            bundles = [next(generator) for _ in range(5)]
            generator.close()
        self.assertEqual([bundle.locked for bundle in bundles], [True, False, False, True, True])
        self.assertEqual([bundle.camera_source_metadata["freshFrame"] for bundle in bundles], [True, False, False, True, True])
        for bundle in bundles[1:3]:
            self.assertIsNone(bundle.warp_bgr)
            self.assertIsNone(bundle.motion_warp)
            self.assertIsNone(bundle.mask_cam)
        self.assertEqual(shared.call_count, 3)
        self.assertEqual(session._recorder.write.call_count, 3)
        self.assertEqual(timeline.call_count, 3)
        self.assertEqual(invalid.call_count, 2)  # one gap and one new session
        self.assertNotEqual(bundles[0].registration_metadata, bundles[3].registration_metadata)
        self.assertEqual(bundles[3].registration_metadata, bundles[4].registration_metadata)
        stream = bundles[3].registration_metadata["cameraStream"]
        self.assertEqual((stream["sessionId"], stream["connectionGeneration"]), ("two", 2))
        self.assertNotIn("seq", stream)
        self.assertNotIn("hostReceiveAgeSeconds", stream)
        self.assertNotEqual(session.frame_time, bundles[4].camera_source_metadata["captureTimeNs"])

    def test_connection_loss_during_handoff_keeps_ui_alive_and_first_real_geometry_can_arrive(self):
        capture = PhoneCapture([None, None, (meta(), IMAGE)], placeholder_size=(64, 48))
        session, *_ = self.create(capture)
        with ExitStack() as stack:
            self.lock_pipeline(session, stack)
            generator = session.frames()
            hold = next(generator)
            fresh = next(generator)
            generator.close()
        self.assertFalse(hold.locked)
        self.assertEqual(hold.cam_bgr.shape, (48, 64, 3))
        self.assertTrue(fresh.locked)
        self.assertEqual(fresh.cam_bgr.shape, (24, 32, 3))

    def test_later_phone_resolution_change_requires_session_restart(self):
        capture = PhoneCapture([(meta(), IMAGE), (meta(seq=1), np.zeros((48, 64, 3), np.uint8))])
        session, *_ = self.create(capture)
        with self.assertRaisesRegex(CameraCalibrationChanged, "Restart the live session"):
            next(session.frames())


if __name__ == "__main__":
    unittest.main()
