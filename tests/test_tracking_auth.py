import contextlib
import unittest
from unittest.mock import Mock, patch

from auth_provider import AccessDenied


class TrackingAuthTests(unittest.TestCase):
    def test_revocation_closes_a_pending_foundry_setup_wait(self):
        import calibration
        access = Mock(allowed=False)
        access.require.side_effect = AccessDenied("Revoked")
        with patch("tk_camera_preview.TkFoundryWait") as wait_window:
            with self.assertRaises(AccessDenied):
                calibration._foundry_wait_for_scene_grid(access)
            wait_window.return_value.close.assert_called_once()

    def test_denied_session_never_opens_camera(self):
        import v3_tracking as live
        access = Mock()
        access.require.side_effect = AccessDenied("Revoked")
        with patch.object(live.core, "CVCoreSession") as camera:
            with self.assertRaises(AccessDenied):
                live.begin_session(Mock(), access=access)
            camera.assert_not_called()

    def test_revocation_stops_before_processing_or_moving_and_closes_recording(self):
        import v3_tracking as live
        session = Mock()
        session.frames.return_value = iter([Mock(), Mock()])
        session.is_recording = True
        session.warp_w, session.warp_h, session.grid_w, session.grid_h = 800, 480, 20, 12
        panel = Mock()
        panel.pump.return_value = True
        access = Mock(allowed=False, status=Mock(return_value="Access revoked"))
        callback = Mock()
        with contextlib.ExitStack() as stack:
            for target, name, value in (
                (live, "initialize_user_data", Mock()), (live.s, "load_last_selection", Mock(return_value={})),
                (live, "load_profiles_with_curves", Mock(return_value={})),
                (live.ml, "load_synced_library", Mock(return_value={"minis": {}})),
                (live.core, "CVCoreSession", Mock(return_value=session)),
                (live, "ControlPanel", Mock(return_value=panel)),
                (live.fo, "get_mini_assignments", Mock(return_value=({}, {}))),
                (live.fo, "guided_capture_active", Mock(return_value=True)),
                (live.fo, "finish_guided_capture", Mock()),
                (live.fo, "update_camera_lock", Mock()),
                (live.fo, "record_tracking_event", Mock()),
            ):
                stack.enter_context(patch.object(target, name, value))
            # Avoid creating native windows during a headless tracking test.
            stack.enter_context(patch("tk_camera_preview._get_root", return_value=None))
            live.begin_session(callback, show_windows=False, access=access)
            callback.assert_not_called()
            live.fo.record_tracking_event.assert_not_called()
            live.fo.finish_guided_capture.assert_called_once()
            session.close.assert_called_once_with()
            panel.root.destroy.assert_called_once()

    def test_relay_refuses_start_without_access(self):
        from foundry_service import FoundryService
        access = Mock()
        access.require.side_effect = AccessDenied("Sign in")
        relay = FoundryService()
        relay.access = access
        with self.assertRaises(AccessDenied):
            relay.start()
        self.assertFalse(relay.running)
