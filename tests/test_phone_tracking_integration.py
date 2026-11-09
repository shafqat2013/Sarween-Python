"""Exercise the real live-loop handoff/hold gates with inert services and UI."""
from contextlib import ExitStack
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import observer_session
import v3_tracking as tracking


class PhoneTrackingIntegrationTests(unittest.TestCase):
    def exercise(self, *, observation_only, mode="foundry"):
        pixels = np.zeros((24, 32, 3), np.uint8)
        stream = {"deviceId": "usb-phone", "sessionId": "two", "connectionGeneration": 2}
        source = {**stream, "captureTimeNs": 900_000, "hostReceiveAgeSeconds": .02, "seq": 17}
        registration = {"mode": "screen", "cameraStream": deepcopy(stream)}
        bundles = [SimpleNamespace(frame_idx=index + 1, locked=fresh,
            last_missing_ids=[] if fresh else [10, 11, 12, 13], last_marker_count=4 if fresh else 0,
            warp_bgr=pixels if fresh else None, cam_bgr=pixels,
            warp_w=32, warp_h=24, grid_w=8, grid_h=6,
            registration_metadata=deepcopy(registration),
            camera_source_metadata={**source, "freshFrame": fresh, "message": "Phone reconnecting"})
            for index, fresh in enumerate((False, True))]
        session = SimpleNamespace(warp_w=32, warp_h=24, grid_w=8, grid_h=6,
            is_recording=True, _record_path="in-memory.mp4", frame_time=0, lock_miss_streak=0,
            _watermark_current_exclusion_warp=None, close=Mock(),
            camera_calibration_metadata=Mock(return_value={"profileId": "phone-profile"}))
        def frames():
            for index, bundle in enumerate(bundles):
                session.frame_time = 100. + index
                yield bundle
        session.frames = frames
        panel = Mock()
        panel.pump.return_value = True
        panel.pop_actions.side_effect = [{"recapture_bg": True, "calibrate_minis": True,
                                          "calibrate_band": {"sample_lab": [1, 2, 3]}}, {}]
        panel.get_toggles.return_value = {}
        panel.get_motion_thresh.return_value = 20
        engine = Mock()
        engine.prev_state = engine.cell_hist = engine.last_emitted = {}
        engine.last_seen = engine.last_physical_xy = {}
        engine.step.return_value = SimpleNamespace(reason="held")
        observer = Mock(closed=False)
        observer.engine = None
        observer.binding_generation = None
        observer.last_processed_timestamp = None
        observer.is_recording = False
        observer.journal = None
        observer.update.return_value = None
        panel.get_confirmed_tracking_assignments.return_value = {}
        service = Mock()
        service.get_mini_assignments.return_value = ({}, {})
        service.get_scene_params.return_value = {"sceneId": "test", "gridCols": 8, "gridRows": 6}
        service.get_delivery_status.return_value = {"connected": True}
        service.pop_capture_command.return_value = None
        service.guided_capture_active.return_value = False
        service.get_view_transform_payload.return_value = {"sceneId": "test"}
        service.get_rendered_reference_status.return_value = {}
        service.get_scene_visual_change.return_value = {"revision": 0}
        callback, capture = Mock(), object()
        with ExitStack() as stack:
            replacements = [(tracking, "initialize_user_data", Mock()),
                (tracking.s, "load_last_selection", Mock(return_value={"mode": mode})),
                (tracking, "load_profiles_with_curves", Mock(return_value={})),
                (tracking.ml, "load_synced_library", Mock(return_value={})),
                (tracking.ml, "library_rows", Mock(return_value=[])),
                (tracking, "fo", service), (tracking, "_pending_recorder", None),
                (tracking, "ControlPanel", Mock(return_value=panel)),
                (tracking, "TrackingEngine", Mock(return_value=engine)),
                (tracking, "create_tracking_windows", Mock(return_value=Mock())),
                (tracking, "tracking_context", Mock(return_value={})),
                (observer_session, "LiveObserverController", Mock(return_value=observer))]
            for module, name, value in replacements:
                stack.enter_context(patch.object(module, name, value))
            core_factory = stack.enter_context(patch.object(tracking.core, "CVCoreSession", return_value=session))
            stack.enter_context(patch("tk_camera_preview._get_root", return_value=None))
            result = tracking.begin_session(callback, camera_index=-100001, show_windows=False,
                                             observation_only=observation_only, camera_capture=capture)
        core_factory.assert_called_once_with(camera_index=-100001, camera_capture=capture,
                                             tracking_pixels_only=mode == "foundry" or observation_only)
        self.assertIsNone(result)
        session.close.assert_called_once()
        callback.assert_not_called()
        self.assertEqual(panel.pump.call_count, 2, "A failed phone read must keep pumping the UI")
        return session, service, observer, engine, bundles, source

    def test_observer_skips_stale_capture_clocks_and_preserves_fresh_source_metadata(self):
        _, service, observer, engine, bundles, source = self.exercise(observation_only=True)
        self.assertEqual(service.record_tracking_input.call_count, 1)
        self.assertEqual(observer.update.call_count, 1)
        second = observer.update.call_args
        self.assertTrue(second.kwargs["locked"])
        self.assertEqual(second.kwargs["metadata"]["camera_source"]["seq"], source["seq"])
        self.assertEqual(second.kwargs["metadata"]["camera_source"]["captureTimeNs"], source["captureTimeNs"])
        self.assertNotIn("seq", second.kwargs["calibration"])
        self.assertNotIn("hostReceiveAgeSeconds", second.kwargs["registration"]["cameraStream"])
        engine.step.assert_not_called()

    def test_legacy_tracker_never_measures_stale_preview_or_records_its_input_clock(self):
        _, service, observer, engine, bundles, _ = self.exercise(observation_only=False, mode="self_hosted")
        self.assertEqual(service.record_tracking_input.call_count, 2)  # fresh input and tracked input
        self.assertTrue(service.record_tracking_input.call_args.kwargs["tracked"])
        engine.step.assert_called_once()
        self.assertIs(engine.step.call_args.args[0], bundles[1])
        observer.update.assert_not_called()

    def test_normal_foundry_session_also_skips_stale_frames_and_bypasses_legacy_tracker(self):
        _, service, observer, engine, _bundles, _source = self.exercise(observation_only=False)
        self.assertEqual(observer.update.call_count, 1)
        self.assertEqual(service.record_tracking_input.call_count, 1)
        engine.step.assert_not_called()

    def test_native_and_old_bundle_defaults_remain_fresh(self):
        for value in (SimpleNamespace(), SimpleNamespace(camera_source_metadata=None),
                      SimpleNamespace(camera_source_metadata={}),
                      SimpleNamespace(camera_source_metadata={"freshFrame": True})):
            self.assertTrue(tracking._fresh_camera_frame(value))
        self.assertFalse(tracking._fresh_camera_frame(SimpleNamespace(camera_source_metadata={"freshFrame": False})))


if __name__ == "__main__":
    unittest.main()
