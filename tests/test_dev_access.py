"""Developer access is explicit, temporary, and unavailable in packaged apps."""

import contextlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class DeveloperAccessTests(unittest.TestCase):
    def make_panel(self):
        import dev_access

        with contextlib.ExitStack() as stack:
            for widget in ("Frame", "Label"):
                stack.enter_context(patch.object(dev_access.ttk, widget))
            # Local testing must work while the remote login service is down and
            # must not read, replace, or report the owner's real login session.
            for target in ("auth_ui.AuthPanel", "auth_storage.KeychainStore",
                           "auth_provider.SupabaseProvider", "usage_metrics.UsageMetrics"):
                stack.enter_context(patch(target, side_effect=AssertionError(target)))
            panel = dev_access.DeveloperAuthPanel(Mock(), Mock(), account_parent=Mock())
            panel.usage_state("tracking", True)
            panel.usage_event("live_errors")
        return panel

    def test_source_access_needs_no_login_service_and_ends_on_close(self):
        panel = self.make_panel()
        self.assertTrue(panel.allowed)
        self.assertIn("Developer mode", panel.status())
        self.assertIn("login bypassed", panel.status())
        panel.require()
        panel.close()
        self.assertFalse(panel.allowed)
        with self.assertRaises(PermissionError):
            panel.require()

    def test_packaged_runtime_cannot_create_developer_access(self):
        from dev_access import DeveloperAuthPanel

        for marker, value in (("frozen", True), ("_MEIPASS", "/fake/bundle")):
            with self.subTest(marker=marker), patch.dict(sys.__dict__, {marker: value}):
                with self.assertRaises((ValueError, PermissionError)):
                    DeveloperAuthPanel(Mock(), Mock())

    def test_packaged_marker_invalidates_an_existing_developer_gate(self):
        panel = self.make_panel()
        for marker, value in (("frozen", True), ("_MEIPASS", "/fake/bundle")):
            with self.subTest(marker=marker), patch.dict(sys.__dict__, {marker: value}):
                self.assertFalse(panel.allowed)
                with self.assertRaises(PermissionError):
                    panel.require()
        panel.close()


class DeveloperLaunchTests(unittest.TestCase):
    def launch(self, **kwargs):
        import app_window
        import main

        root, app = Mock(), Mock()
        with patch.object(main, "initialize_user_data"), \
             patch.object(main.s, "_get_persistent_root", return_value=root), \
             patch.object(main.s, "initialize") as setup, \
             patch.object(main.c, "calibrate") as calibrate, \
             patch.object(main, "FoundryService") as service, \
             patch.object(app_window, "AppWindow", return_value=app) as window:
            main.main(**kwargs)
            setup.assert_not_called()
            calibrate.assert_not_called()
            service.return_value.start.assert_not_called()
            root.mainloop.assert_called_once()
            app.close.assert_called_once()
        return window.call_args

    def test_normal_launch_keeps_the_real_login_panel(self):
        call = self.launch()
        self.assertIsNone(call.kwargs.get("auth_panel_factory"))

    def test_explicit_developer_launch_uses_only_the_local_gate(self):
        from dev_access import DeveloperAuthPanel

        call = self.launch(dev_no_login=True)
        self.assertIs(call.kwargs["auth_panel_factory"], DeveloperAuthPanel)

    def test_packaged_developer_launch_is_rejected_before_user_data_or_windows(self):
        import main

        for marker, value in (("frozen", True), ("_MEIPASS", "/fake/bundle")):
            with self.subTest(marker=marker), \
                 patch.dict(sys.__dict__, {marker: value}), \
                 patch.object(main, "initialize_user_data") as initialize, \
                 patch.object(main.s, "_get_persistent_root") as root:
                with self.assertRaises(ValueError):
                    main.main(dev_no_login=True)
                initialize.assert_not_called()
                root.assert_not_called()


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class DeveloperHomeUITests(unittest.TestCase):
    def test_developer_home_is_ready_without_login_and_displays_its_mode(self):
        import tkinter as tk
        from tkinter import ttk
        import app_window
        from dev_access import DeveloperAuthPanel
        from foundry_service import FoundryService

        root = tk.Tk()
        root.withdraw()
        service = FoundryService()
        runner = Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app_window, "data_path", side_effect=lambda name: Path(directory) / name), \
             patch("auth_ui.AuthPanel", side_effect=AssertionError("Developer launch must not sign in")), \
             patch("cv2.VideoCapture", side_effect=AssertionError("Home must not open a camera")):
            # This checks developer access with the legacy synchronous runner.
            app = app_window.AppWindow(root, service=service, run_session=runner, inline_startup=False,
                                       auth_panel_factory=DeveloperAuthPanel)
            try:
                root.update()
                self.assertTrue(app.home.winfo_ismapped())
                self.assertFalse(app.login.winfo_ismapped())
                self.assertIs(service.access, app.auth)
                self.assertNotIn("disabled", app.start_button.state())
                self.assertNotIn("disabled", app.connect_button.state())
                self.assertFalse(service.running)

                def visible_labels(parent):
                    for child in parent.winfo_children():
                        if isinstance(child, ttk.Label) and child.winfo_viewable():
                            yield str(child.cget("text"))
                        yield from visible_labels(child)

                labels = "\n".join(visible_labels(app.home))
                self.assertIn("Developer mode", labels)
                self.assertIn("login bypassed", labels)
                app.start_session()
                runner.assert_called_once_with(service)
                self.assertEqual(list(Path(directory).iterdir()), [])
            finally:
                app.close()
            self.assertFalse(app.auth.allowed)


if __name__ == "__main__":
    unittest.main()
