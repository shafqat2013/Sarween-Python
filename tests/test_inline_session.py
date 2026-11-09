"""Inline home/startup/runtime lifecycle with no native UI or hardware."""

import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app_window import AppWindow
import main
from session_startup import StartupPending
import v3_tracking


SCENE = {"sceneId": "current-scene", "sceneW": 1000, "sceneH": 800, "gridPx": 50}


class Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class StartupDouble:
    def __init__(self, parent, **options):
        self.parent, self.options = parent, options
        self.selection_data = {"display_index": 1, "webcam_index": 4,
                               "mode": "foundry", "camera_profile_id": "phone"}
        self.displays = [{"index": 1, "width": 1920, "height": 1080}]
        self.webcams = [{"index": 4, "name": "Selected camera"}]
        self.scene_info = {"sceneId": "stale-scene"}
        self.runtime_parent = options["runtime_parent"]
        self.runtime_started = Mock(side_effect=options["on_runtime"])
        self.resume_wait = Mock()
        self.show_error = Mock()
        self.dispose = Mock()
        self.cancel = Mock(side_effect=lambda: options["on_cancel"](self))


def home_double():
    """Real AppWindow methods, with only physical widgets replaced."""
    app = AppWindow.__new__(AppWindow)
    app.root = Mock(name="existing_main_root")
    app.root.winfo_width.return_value = 860
    app.root.winfo_height.return_value = 620
    app.inline_startup = True
    app.closed = app.busy = app._close_requested = app._observer_startup = False
    app._startup = app._runtime_panel = None
    app._poll_id = None
    app.replay = None
    app.auth = Mock(allowed=True)
    app.auth.status.return_value = "Approved"
    app.service = Mock(access=app.auth)
    app.run_session = Mock()
    app.run_observer = Mock()
    app.status, app.diagnostic = Value(), Value()
    for name in ("tabs", "session_tab", "session_intro", "runtime_host", "home", "login",
                 "vtt_controls", "start_button", "observer_button"):
        setattr(app, name, Mock(name=name))
    app._access_changed = Mock()
    app.refresh_library = Mock()
    app._startup_factory = Mock(side_effect=StartupDouble)
    return app


class InlineMainTests(unittest.TestCase):
    def startup(self):
        return StartupDouble(Mock(), runtime_parent=Mock(name="existing_runtime_frame"),
                             on_runtime=Mock(), on_cancel=Mock())

    def test_preselected_inventory_fresh_scene_and_runtime_frame_reach_runner(self):
        startup = self.startup()
        service = Mock(access=Mock())
        service.startup_status.return_value = {"phase": "ready", "scene_info": SCENE}
        with patch.object(main.s, "initialize") as initialize, \
             patch.object(main.s, "selected_mode", main.s.MODE_FOUNDRY), \
             patch.object(main.c, "calibrate") as calibrate, \
             patch.object(v3_tracking, "begin_session", return_value=None) as begin, \
             patch.object(v3_tracking, "close_pending_recording") as cleanup:
            main.run_live_session(service, startup=startup)
        initialize.assert_called_once_with(access=service.access, selection=startup.selection_data,
                                            displays=startup.displays, webcams=startup.webcams)
        calibrate.assert_called_once_with(access=service.access, scene_info=SCENE)
        self.assertIs(startup.scene_info, SCENE)
        begin.assert_called_once_with(main.on_mini_moved, access=service.access,
                                      foundry_service=service, observation_only=False,
                                      control_parent=startup.runtime_parent,
                                      control_started=startup.runtime_started)
        cleanup.assert_called_once()

    def test_readiness_race_never_calibrates_or_enters_camera_before_ready(self):
        for phase in ("starting_relay", "waiting_module", "waiting_scene"):
            with self.subTest(phase=phase):
                startup = self.startup()
                service = Mock(access=Mock())
                service.startup_status.return_value = {"phase": phase, "scene_info": None}
                with patch.object(main.s, "initialize"), \
                     patch.object(main.s, "selected_mode", main.s.MODE_FOUNDRY), \
                     patch.object(main.c, "calibrate") as calibrate, \
                     patch.object(v3_tracking, "begin_session") as begin, \
                     patch.object(v3_tracking, "close_pending_recording") as cleanup:
                    with self.assertRaises(StartupPending):
                        main.run_live_session(service, startup=startup)
                calibrate.assert_not_called()
                begin.assert_not_called()
                cleanup.assert_not_called()

    def test_inline_observer_does_not_release_app_owned_guard_on_pending_or_error(self):
        for error in (StartupPending(), RuntimeError("Camera unavailable")):
            with self.subTest(error=type(error).__name__), \
                 patch.object(main.fo, "set_observation_only") as guard, \
                 patch.object(main, "run_live_session", side_effect=error):
                with self.assertRaises(type(error)):
                    main.run_observer_session(Mock(), startup=self.startup())
                self.assertEqual([item.args[0] for item in guard.call_args_list], [True])


class InlineHomeLifecycleTests(unittest.TestCase):
    def test_pending_race_resumes_same_setup_without_error_or_second_start_click(self):
        app = home_double()
        app.run_observer = main.run_observer_session
        app.service.startup_status.side_effect = [
            {"phase": "waiting_module", "scene_info": None},
            {"phase": "ready", "scene_info": SCENE},
        ]
        with patch.object(main.fo, "set_observation_only") as guard, \
             patch.object(main.s, "initialize"), \
             patch.object(main.s, "selected_mode", main.s.MODE_FOUNDRY), \
             patch.object(main.c, "calibrate") as calibrate, \
             patch.object(v3_tracking, "begin_session", return_value=None) as begin, \
             patch.object(v3_tracking, "close_pending_recording"):
            app.start_session(observer=True)
            # Foundry startup must remain able to unpause before the runner.
            guard.assert_not_called()
            startup = app._startup
            draft = startup.selection_data
            startup.options["on_ready"](startup)
            startup.resume_wait.assert_called_once()
            startup.show_error.assert_not_called()
            startup.dispose.assert_not_called()
            app.auth.usage_event.assert_not_called()
            calibrate.assert_not_called()
            begin.assert_not_called()
            self.assertTrue(app.busy)
            self.assertIs(app._startup, startup)
            self.assertIs(startup.selection_data, draft)
            # A duplicate click cannot create another setup or start the camera.
            app.start_session(observer=True)
            self.assertEqual(app._startup_factory.call_count, 1)
            # This callback represents the automatic next ready poll, not Retry.
            startup.options["on_ready"](startup)
            self.assertFalse(app.busy)
            self.assertIsNone(app._startup)
            self.assertEqual(begin.call_count, 1)
            startup.dispose.assert_called_once()
            self.assertFalse(guard.call_args_list[-1].args[0])
        app.root.withdraw.assert_not_called()
        app.root.destroy.assert_not_called()
        self.assertIs(app._startup_factory.call_args.kwargs["runtime_parent"], app.runtime_host)

    def test_camera_failure_keeps_draft_and_releases_guard_before_setup_retry(self):
        app = home_double()
        app.run_observer.side_effect = RuntimeError("Selected camera is unavailable")
        with patch.object(main.fo, "set_observation_only") as guard, \
             contextlib.redirect_stderr(io.StringIO()):
            app.start_session(observer=True)
            startup = app._startup
            draft = startup.selection_data
            startup.options["on_ready"](startup)
            self.assertIs(app._startup, startup)
            self.assertIs(startup.selection_data, draft)
            self.assertTrue(app.busy)
            startup.show_error.assert_called_once_with("Selected camera is unavailable")
            startup.dispose.assert_not_called()
            self.assertEqual([call.args[0] for call in guard.call_args_list], [True, False])
            self.assertIn("needs attention", app.status.get())
            self.assertIn("camera is unavailable", app.diagnostic.get())
            startup.cancel()
            startup.dispose.assert_called_once()
            self.assertFalse(app.busy)
            self.assertFalse(guard.call_args_list[-1].args[0])
        app.root.withdraw.assert_not_called()
        app.root.destroy.assert_not_called()

    def test_close_requests_runtime_stop_then_waits_for_runner_cleanup(self):
        app = home_double()
        panel = Mock(name="embedded_control_panel")
        order = []
        panel._on_close.side_effect = lambda: order.append("stop_requested")
        app.root.destroy.side_effect = lambda: order.append("root_destroyed")

        def run(_service, *, startup):
            startup.runtime_started(panel)
            app.close()
            self.assertTrue(app.busy)
            self.assertFalse(app.closed)
            app.root.destroy.assert_not_called()
            app.service.stop.assert_not_called()
            order.append("runner_cleanup")

        app.run_session.side_effect = run
        app.start_session()
        startup = app._startup
        startup.options["on_ready"](startup)
        self.assertEqual(order, ["stop_requested", "runner_cleanup", "root_destroyed"])
        self.assertTrue(app.closed)
        self.assertFalse(app.busy)
        startup.dispose.assert_called_once()
        app.auth.close.assert_called_once()
        app.service.stop.assert_called_once()
        app.vtt_controls.close.assert_called_once()
        app.root.withdraw.assert_not_called()

    def test_late_callback_from_disposed_setup_cannot_run_camera(self):
        app = home_double()
        app.start_session()
        startup = app._startup
        startup.cancel()
        startup.options["on_ready"](startup)
        app.run_session.assert_not_called()
        self.assertFalse(app.busy)

    def test_setup_construction_error_releases_observer_guard(self):
        app = home_double()
        app._startup_factory.side_effect = RuntimeError("Hardware inventory unavailable")
        with patch.object(main.fo, "set_observation_only") as guard:
            app.start_session(observer=True)
        self.assertFalse(app.busy)
        self.assertIsNone(app._startup)
        guard.assert_not_called()
        self.assertIn("Hardware inventory unavailable", app.diagnostic.get())


if __name__ == "__main__":
    unittest.main()
