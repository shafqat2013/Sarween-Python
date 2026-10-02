import asyncio
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import app_window
from foundry_service import FoundryService


class HomeDataTests(unittest.TestCase):
    def test_saved_roster_is_read_only_and_distinguishes_unverified_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "combo_profiles.json"
            profile.write_text(json.dumps({"red10": {"lab": [60, 160, 150]}}))
            with patch.object(app_window, "data_path", side_effect=lambda name: root / name):
                library, rows = app_window.saved_roster()
            red = next(row for row in rows if row["id"] == "red10")
            self.assertEqual(red["sampleCount"], 0)
            self.assertEqual(red["scanStatus"], "Ready")
            self.assertFalse(library["minis"]["red10"]["samples"][0]["verified"])
            self.assertEqual(list(root.iterdir()), [profile])

    def test_invalid_saved_data_is_not_replaced_with_empty_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "mini_library.json"
            profile.write_text("broken json")
            with patch.object(app_window, "data_path", side_effect=lambda name: root / name):
                with self.assertRaises(ValueError):
                    app_window.saved_roster()
            self.assertEqual(profile.read_text(), "broken json")

    def test_main_does_not_start_hardware_or_server(self):
        import main
        root, app = Mock(), Mock()
        with patch.object(main, "initialize_user_data"), \
             patch.object(main.s, "_get_persistent_root", return_value=root), \
             patch.object(main.s, "initialize") as setup, \
             patch.object(main.c, "calibrate") as calibrate, \
             patch.object(main, "FoundryService") as service, \
             patch.object(app_window, "AppWindow", return_value=app):
            main.main()
            setup.assert_not_called()
            calibrate.assert_not_called()
            service.return_value.start.assert_not_called()
            root.mainloop.assert_called_once()
            app.close.assert_called_once()

    def test_cancelled_brightness_scan_finishes_carried_recording(self):
        import v3_tracking as tracking
        recorder = Mock()
        with patch.object(tracking, "_pending_recorder", recorder), \
             patch.object(tracking, "_pending_record_path", "test.mp4"), \
             patch.object(tracking.fo, "stop_timeline_recording") as stop:
            tracking.close_pending_recording()
            tracking.close_pending_recording()
            recorder.release.assert_called_once()
            stop.assert_called_once()
            self.assertIsNone(tracking._pending_recorder)


class FoundryServiceTests(unittest.TestCase):
    def until(self, predicate):
        deadline = time.monotonic() + 2
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_explicit_start_idempotence_stop_and_restart(self):
        async def serve(*, stop_event, ready_event):
            ready_event.set()
            while not stop_event.is_set():
                await asyncio.sleep(.005)
        service = FoundryService(serve)
        self.assertEqual(service.state, "Stopped")
        try:
            service.start()
            self.until(lambda: service.state == "Listening")
            thread = service._thread
            service.start()
            self.assertIs(service._thread, thread)
            service.stop()
            self.assertFalse(thread.is_alive())
            self.assertEqual(service.state, "Stopped")
            service.start()
            self.until(lambda: service.state == "Listening")
            self.assertIsNot(service._thread, thread)
        finally:
            service.stop()

    def test_bind_failure_is_visible_and_stops_thread(self):
        async def failed(**_kwargs):
            raise OSError("Address already in use")
        service = FoundryService(failed)
        try:
            service.start()
            self.until(lambda: service.state == "Failed")
            self.assertIn("already in use", service.error)
        finally:
            service.stop()
        self.assertFalse(service.running)

    def test_real_relay_coroutine_cleans_up_loop_on_stop(self):
        import foundryoutput as fo
        class Server:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *_args):
                self.closed = True
        server = Server()
        with patch.object(fo.websockets, "serve", return_value=server), \
             patch.object(fo, "_load_mapping"), patch.object(fo, "_loop", None):
            service = FoundryService(fo.main)
            try:
                service.start()
                self.until(lambda: service.state == "Listening")
            finally:
                service.stop()
            self.assertTrue(server.closed)
            self.assertIsNone(fo._loop)


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class HomeUITests(unittest.TestCase):
    def test_local_relay_closes_a_connected_client_and_releases_port(self):
        import foundryoutput as fo
        import websockets
        original_serve = websockets.serve
        address = []

        @contextlib.asynccontextmanager
        async def ephemeral_server(handler, host, _port, **kwargs):
            async with original_serve(handler, host, 0, **kwargs) as server:
                address.append(server.sockets[0].getsockname()[1])
                yield server

        service = FoundryService(fo.main)
        with patch.object(fo.websockets, "serve", side_effect=ephemeral_server), \
             patch.object(fo, "_load_mapping"):
            try:
                service.start()
                deadline = time.monotonic() + 3
                while service.state == "Starting" and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertEqual(service.state, "Listening", service.error)

                async def connect_then_stop():
                    async with websockets.connect(f"ws://127.0.0.1:{address[0]}") as client:
                        await asyncio.to_thread(service.stop)
                        await asyncio.wait_for(client.wait_closed(), timeout=2)

                asyncio.run(connect_then_stop())
                self.assertFalse(service.running)
                self.assertIsNone(fo._active_socket)
                self.assertIsNone(fo._loop)
            finally:
                service.stop()

    def test_offline_library_layout_cancel_error_and_close(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        service = FoundryService()
        session = Mock(side_effect=SystemExit(0))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(app_window, "data_path", side_effect=lambda name: Path(directory) / name), \
             patch("cv2.VideoCapture", side_effect=AssertionError("Home must not open a camera")), \
             patch.object(service, "start", side_effect=AssertionError("Home must not bind a port")):
            # These existing tests exercise the home/session lifecycle with an
            # approved identity. Real authentication gates have their own UI tests.
            auth = Mock(allowed=True, status=Mock(return_value="Approved"))
            app = app_window.AppWindow(root, service=service, run_session=session,
                                       auth_panel_factory=lambda *args, **kwargs: auth)
            try:
                root.update()
                self.assertFalse(service.running)
                self.assertEqual(len(app.tree.get_children()), 5)
                self.assertEqual(app.tabs.tab(app.tabs.select(), "text"), "Mini Library")
                self.assertIn("unverified", app.detail.get())
                self.assertEqual(list(Path(directory).iterdir()), [])
                for width, height in ((860, 620), (700, 460)):
                    root.geometry(f"{width}x{height}")
                    root.update()
                    for widget in (app.start_button, app.tabs, app.tree):
                        self.assertLessEqual(widget.winfo_rootx() + widget.winfo_width(),
                                             root.winfo_rootx() + root.winfo_width())
                        self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                             root.winfo_rooty() + root.winfo_height())
                app.start_session()
                root.update()
                self.assertIn("cancelled", app.status.get())
                self.assertEqual(root.state(), "normal")
                self.assertNotIn("disabled", app.start_button.state())
                session.side_effect = RuntimeError("Camera unavailable")
                with contextlib.redirect_stderr(io.StringIO()):
                    app.start_session()
                root.update()
                self.assertIn("Camera unavailable", app.diagnostic.get())
                self.assertEqual(root.state(), "normal")
                self.assertEqual(app.tabs.tab(app.tabs.select(), "text"), "Diagnostics")
                session.side_effect = None
                app.start_session()
                self.assertIn("Session stopped", app.status.get())
            finally:
                app.close()
                app.close()
            self.assertFalse(service.running)

    def test_setup_handles_empty_devices_in_source_and_bundle(self):
        import runpy
        import tkinter as tk
        from tkinter import ttk
        import setup
        from ui_preview import descendants
        root = tk.Tk()
        root.withdraw()
        try:
            bundled = runpy.run_path(str(Path(__file__).resolve().parents[1] / "app_bundle_overrides/setup.py"))
            for select in (setup.unified_selection_window, bundled["unified_selection_window"]):
                def check_and_cancel():
                    buttons = {widget.cget("text"): widget for widget in descendants(root)
                               if isinstance(widget, ttk.Button)}
                    self.assertIn("disabled", buttons["Select"].state())
                    self.assertIn("disabled", buttons["Preview Webcam"].state())
                    buttons["Cancel"].invoke()
                with patch.dict(select.__globals__, {"_get_persistent_root": lambda: root}):
                    root.after(50, check_and_cancel)
                    result = select([], [], default_mode="foundry")
                self.assertIsNone(result)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
