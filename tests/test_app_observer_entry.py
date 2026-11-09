"""Observer entry-point routing plus optional native Sarween home layout.

Native tests create only Sarween's own widgets with inert service/connection
panels. They never open a camera, request Accessibility or start a relay.
Run those explicitly with SARWEEN_GUI_TESTS=1.
"""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import app_window


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def entry_model():
    app = app_window.AppWindow.__new__(app_window.AppWindow)
    app.root = Mock()
    app.service = Mock()
    app.auth = SimpleNamespace(allowed=True, status=lambda: "Sign in required", usage_event=Mock())
    app.run_session = Mock()
    app.run_observer = Mock()
    app.closed = app.busy = False
    app.replay = None
    app.start_button = Mock()
    app.observer_button = Mock()
    app.recording_button = Mock()
    app.demo_button = Mock()
    app.status = Value()
    app.diagnostic = Value()
    app.tabs = Mock()
    app.diagnostics_tab = object()
    app.refresh_library = Mock()
    app._access_changed = Mock()
    return app


class ObserverEntryTests(unittest.TestCase):
    def test_observer_button_routes_to_observer_and_never_the_legacy_runner(self):
        app = entry_model()
        app.start_session(observer=True)
        app.run_observer.assert_called_once_with(app.service)
        app.run_session.assert_not_called()
        self.assertFalse(app.busy)
        app.root.withdraw.assert_called_once_with()
        app.root.deiconify.assert_called_once_with()
        app.root.lift.assert_called_once_with()
        self.assertEqual(app.status.get(), "Session stopped - camera off")

    def test_legacy_entry_remains_separate(self):
        app = entry_model()
        app.start_session()
        app.run_session.assert_called_once_with(app.service)
        app.run_observer.assert_not_called()

    def test_authorization_blocks_observer_before_any_camera_runner(self):
        app = entry_model()
        app.auth.allowed = False
        app.start_session(observer=True)
        app.run_observer.assert_not_called()
        app.run_session.assert_not_called()
        app.root.withdraw.assert_not_called()
        self.assertEqual(app.status.get(), "Sign in required")

    def test_busy_closed_and_nested_starts_cannot_open_another_observer(self):
        for field in ("busy", "closed"):
            app = entry_model()
            setattr(app, field, True)
            app.start_session(observer=True)
            app.run_observer.assert_not_called()
        app = entry_model()
        app.run_observer.side_effect = lambda _service: app.start_session(observer=True)
        app.start_session(observer=True)
        app.run_observer.assert_called_once()

    def test_cancel_restores_home_and_does_not_switch_to_legacy_tracking(self):
        app = entry_model()
        app.run_observer.side_effect = SystemExit
        app.start_session(observer=True)
        self.assertFalse(app.busy)
        self.assertEqual(app.status.get(), "Setup or session cancelled - camera off")
        app.root.deiconify.assert_called_once_with()
        app._access_changed.assert_called_once_with()
        app.run_session.assert_not_called()

    def test_runner_failure_is_reported_and_home_is_restored(self):
        app = entry_model()
        app.run_observer.side_effect = ValueError("No locked camera frame")
        with patch("traceback.print_exc"):
            app.start_session(observer=True)
        self.assertFalse(app.busy)
        self.assertIn("could not continue", app.status.get())
        self.assertEqual(app.diagnostic.get(), "No locked camera frame")
        app.tabs.select.assert_called_once_with(app.diagnostics_tab)
        app.root.deiconify.assert_called_once_with()
        app.auth.usage_event.assert_called_once_with("live_errors")

    def test_unavailable_observer_does_not_fall_back_to_legacy_tracking(self):
        app = entry_model()
        app.run_observer = None
        app.start_session(observer=True)
        self.assertIn("unavailable", app.status.get())
        app.run_session.assert_not_called()
        app.root.withdraw.assert_not_called()


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native own-Sarween UI check is opt-in")
class NativeObserverEntryLayoutTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from tkinter import ttk

        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = tk.Tk()
        def cleanup_root():
            try:
                self.root.destroy()
            except tk.TclError:
                pass
        self.addCleanup(cleanup_root)

        class NativeAuth:
            def __init__(self, root, parent, *, account_parent=None, changed=None):
                self.allowed = True
                self.account = ttk.Frame(account_parent, padding=10)
                self.account.pack(fill="x", pady=(0, 12))
                ttk.Label(self.account, text="Developer mode — login bypassed", font=("Helvetica", 12, "bold")).pack(anchor="w")
                ttk.Label(self.account, text="Restart without --dev-no-login to use normal sign-in.").pack(anchor="w")
                self.email_entry = ttk.Entry(parent)
                self.email_entry.pack()

            def status(self):
                return "Sign in required"

            def require(self):
                if not self.allowed:
                    raise PermissionError(self.status())

            def close(self):
                self.allowed = False

        class NativeVtt(ttk.Frame):
            def __init__(self, parent, **_kwargs):
                super().__init__(parent)
                self.connection = tk.StringVar(self, "Foundry: disconnected")
                self.relay = tk.StringVar(self, "Local relay: stopped")
                self.scene = tk.StringVar(self, "Scene: unavailable")
                self.connect_button = ttk.Button(self, text="Start relay")
                self.grid_controls = Mock()
                self.display_controller = Mock()
                self.display_profiles = SimpleNamespace(refresh_profiles=lambda: None)

            def set_enabled(self, _allowed):
                pass

            def set_mini_rows(self, _rows):
                pass

            def close(self):
                pass

        class NativeHardware(ttk.Frame):
            def __init__(self, parent, **_kwargs):
                super().__init__(parent)

            def set_enabled(self, _allowed):
                pass

        self.stack.enter_context(patch("vtt_controls.VTTControls", NativeVtt))
        self.stack.enter_context(patch("hardware_profiles_ui.HardwareProfilesPanel", NativeHardware))
        self.stack.enter_context(patch.object(app_window, "saved_roster", return_value=({"minis": {}}, [])))
        self.stack.enter_context(patch.object(app_window, "data_dir", return_value=Path(self.directory)))
        self.stack.enter_context(patch.object(app_window.AppWindow, "_poll", lambda self: None))
        self.service = SimpleNamespace(running=False, error=None, stop=Mock())
        self.legacy = Mock()
        self.observer = Mock()
        self.app = app_window.AppWindow(self.root, service=self.service, run_session=self.legacy,
                                        run_observer=self.observer, auth_panel_factory=NativeAuth,
                                        inline_startup=False)
        self.rows = []

    def widgets_fit(self, size):
        self.root.geometry(size)
        self.root.update_idletasks()
        self.root.update()
        width, height = self.root.winfo_width(), self.root.winfo_height()
        for name in ("start_button", "observer_button", "recording_button", "demo_button"):
            widget = getattr(self.app, name)
            x = widget.winfo_rootx() - self.root.winfo_rootx()
            y = widget.winfo_rooty() - self.root.winfo_rooty()
            row = {"window": [width, height], "widget": name,
                   "bounds": [x, y, widget.winfo_width(), widget.winfo_height()],
                   "requested_width": widget.winfo_reqwidth(), "mapped": bool(widget.winfo_ismapped())}
            self.rows.append(row)
            self.assertTrue(row["mapped"], row)
            self.assertGreaterEqual(x, 0, row)
            self.assertGreaterEqual(y, 0, row)
            self.assertLessEqual(x + widget.winfo_width(), width, row)
            self.assertLessEqual(y + widget.winfo_height(), height, row)
            self.assertGreaterEqual(widget.winfo_width(), widget.winfo_reqwidth(), row)

    def test_real_home_header_fits_default_and_minimum_and_entry_obeys_auth(self):
        try:
            self.root.update()
            self.assertEqual(self.app.start_button.cget("text"), "Start session")
            for name in ("observer_button", "recording_button", "demo_button"):
                self.assertFalse(getattr(self.app, name).winfo_ismapped())
            self.app.tabs.select(self.app.diagnostics_tab)
            self.widgets_fit("860x620")
            self.widgets_fit("700x460")
            self.app.observer_button.invoke()
            self.observer.assert_called_once_with(self.service)
            self.legacy.assert_not_called()
            self.root.update()
            self.assertFalse(self.app.busy)
            self.assertTrue(self.root.winfo_ismapped())
            self.app.auth.allowed = False
            self.app._access_changed()
            self.root.update()
            self.assertTrue(self.app.observer_button.instate(["disabled"]))
            self.app.start_session(observer=True)
            self.assertEqual(self.observer.call_count, 1)
            self.app.auth.allowed = True
            self.app._access_changed()
            self.root.update()
            self.app.run_observer = Mock(side_effect=SystemExit)
            self.app.observer_button.invoke()
            self.assertFalse(self.app.busy)
            self.assertTrue(self.root.winfo_ismapped())
            self.assertIn("cancelled", self.app.status.get())
        finally:
            folder = Path(__file__).resolve().parents[1] / "artifacts/poc_single_mover_20261105/home-ui-smoke"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "metrics.json").write_text(json.dumps(self.rows, indent=2) + "\n")


if __name__ == "__main__":
    unittest.main()
