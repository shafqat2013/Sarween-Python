import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch



@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class AuthUITests(unittest.TestCase):
    def spin(self, root, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            root.update()
            if predicate():
                return
            time.sleep(.01)
        self.assertTrue(predicate())

    def test_email_code_login_logout_and_all_home_actions_are_gated(self):
        from test_alpha_access import AccessTests
        import tkinter as tk
        from auth_ui import AuthPanel
        from app_window import AppWindow
        from foundry_service import FoundryService
        case = AccessTests()
        case.setUp()
        root = tk.Tk()
        service = FoundryService()
        runner = Mock()
        with tempfile.TemporaryDirectory() as directory, \
             patch("app_window.data_path", side_effect=lambda name: Path(directory) / name):
            app = AppWindow(root, service=service, run_session=runner,
                auth_panel_factory=lambda root, parent, **kw: AuthPanel(root, parent, factory=lambda: case.access, **kw))
            try:
                self.spin(root, lambda: not app.auth.pending)
                self.assertFalse(app.auth.allowed)
                self.assertTrue(app.login.winfo_ismapped())
                self.assertFalse(app.home.winfo_ismapped())
                self.assertFalse(app.tabs.winfo_viewable())
                self.assertFalse(app.auth.recovery.winfo_ismapped())
                for button in (app.start_button, app.demo_button, app.recording_button, app.connect_button):
                    self.assertIn("disabled", button.state())
                app.start_session()
                runner.assert_not_called()
                app.open_demo()
                self.assertIsNone(app.replay)
                app.auth.email.set("tester@example.com")
                app.auth.send()
                self.spin(root, lambda: not app.auth.pending)
                self.assertIn("If this address", app.auth.message.get())
                app.auth.code.set("12345678")
                app.auth.verify()
                self.spin(root, lambda: app.auth.allowed and not app.auth.pending and app.home.winfo_ismapped())
                self.assertEqual(app.auth.code.get(), "")
                self.assertNotIn("disabled", app.start_button.state())
                self.assertFalse(app.auth.fields.winfo_manager())
                root.update()
                self.assertTrue(app.home.winfo_ismapped())
                self.assertFalse(app.login.winfo_ismapped())
                self.assertTrue(app.auth.logout_button.winfo_viewable())
                app.start_session()
                runner.assert_called_once()
                app.auth.logout()
                self.assertFalse(app.auth.allowed)
                self.spin(root, lambda: not app.auth.pending)
                self.assertIsNone(case.store.value)
                self.assertIn("disabled", app.start_button.state())
                self.assertTrue(app.auth.fields.winfo_manager())
                self.assertTrue(app.login.winfo_ismapped())
                self.assertFalse(app.home.winfo_ismapped())
                for width, height in ((860, 620), (700, 460)):
                    root.geometry(f"{width}x{height}")
                    root.update()
                    for widget in (app.auth.frame, app.auth.email_entry, app.auth.verify_button,
                                   app.auth.disclosure, app.auth.login_message):
                        self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(), root.winfo_rootx() + root.winfo_width())
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(), root.winfo_rooty() + root.winfo_height())
            finally:
                app.close()

    def test_saved_login_offline_expiry_and_revocation_switch_the_entire_screen(self):
        from test_alpha_access import AccessTests
        from auth_provider import AccessDenied, Unavailable
        from alpha_access import OFFLINE_SECONDS
        from auth_ui import AuthPanel
        from app_window import AppWindow
        from foundry_service import FoundryService
        import tkinter as tk
        case = AccessTests()
        case.setUp()
        case.login()
        case.access = case.manager()
        root = tk.Tk()
        with tempfile.TemporaryDirectory() as directory, \
             patch("app_window.data_path", side_effect=lambda name: Path(directory) / name):
            app = AppWindow(root, service=FoundryService(), run_session=Mock(),
                auth_panel_factory=lambda root, parent, **kw: AuthPanel(root, parent, factory=lambda: case.access, **kw))
            try:
                self.spin(root, lambda: app.auth.allowed and not app.auth.pending and app.home.winfo_ismapped())
                self.assertTrue(app.home.winfo_ismapped())
                case.provider.permission.side_effect = Unavailable()
                case.provider.refresh.side_effect = Unavailable()
                app.auth.retry()
                self.spin(root, lambda: not app.auth.pending)
                self.assertTrue(app.home.winfo_ismapped())
                self.assertIn("Offline access", app.auth.message.get())
                replay = app.replay = Mock()
                case.advance(OFFLINE_SECONDS + 1)
                self.spin(root, lambda: app.login.winfo_ismapped() and app.auth.recovery.winfo_ismapped()
                          and not app.auth.pending)
                replay.close.assert_called_once()
                self.assertFalse(app.home.winfo_ismapped())
                self.assertIsNone(app.replay)
                self.assertTrue(app.auth.recovery.winfo_ismapped())
                case.provider.permission.side_effect = lambda _token, device: case.permission(device)
                case.provider.refresh.side_effect = None
                case.provider.refresh.return_value = {**case.session, "expires_at": case.now + 3600}
                app.auth.retry()
                self.spin(root, lambda: app.auth.allowed and not app.auth.pending and app.home.winfo_ismapped())
                self.assertTrue(app.home.winfo_ismapped())
                case.provider.permission.side_effect = AccessDenied("Revoked")
                app.auth.retry()
                self.spin(root, lambda: not app.auth.allowed and not app.auth.pending)
                self.assertTrue(app.login.winfo_ismapped())
                self.assertFalse(app.home.winfo_ismapped())
                self.assertIn("Revoked", app.auth.message.get())
            finally:
                app.close()

    def test_missing_configuration_stays_locked_without_hardware_or_keychain(self):
        import tkinter as tk
        from auth_ui import AuthPanel
        root = tk.Tk()
        try:
            with patch("auth_storage.KeychainStore") as keychain:
                panel = AuthPanel(root, root, factory=Mock(side_effect=ValueError("unconfigured")))
                self.spin(root, lambda: not panel.pending)
                self.assertFalse(panel.allowed)
                self.assertIn("not configured", panel.message.get())
                keychain.assert_not_called()
                panel.close()
        finally:
            root.destroy()
