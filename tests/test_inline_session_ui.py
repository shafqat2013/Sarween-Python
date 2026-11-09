"""Own-Sarween native geometry, with fake inventory and a fake connection."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

import tests.test_app_observer_entry as entry_tests


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native own-Sarween UI is opt-in")
class NativeInlineSessionTests(unittest.TestCase):
    def setUp(self):
        entry_tests.NativeObserverEntryLayoutTests.setUp(self)
        self.app.inline_startup = True
        from session_startup import SessionStartupUI
        self.metrics = []
        self.displays = [{"index": 2, "name": "TV", "width": 1920, "height": 1080}]
        self.cameras = [{"index": 7, "model": "OBS Virtual Camera", "resolution": "Not opened"}]
        defaults = {"display_index": 2, "webcam_index": 7, "mode": "foundry"}
        self.stack.enter_context(patch("setup.load_last_selection", return_value=defaults))
        self.stack.enter_context(patch("setup._active_camera_profile_id", return_value=None))
        self.stack.enter_context(patch("cv2.VideoCapture", side_effect=AssertionError("UI test opened camera")))
        self.stack.enter_context(patch("foundryoutput.request_scene_info"))
        self.stack.enter_context(patch("foundryoutput.set_observation_only"))
        def foundry_step(parent, **kwargs):
            from guided_foundry_setup import GuidedFoundrySetup
            from foundry_startup import DEFAULTS
            from hardware_profiles import HardwareProfiles
            controller = Mock()
            controller.defaults.return_value = {**DEFAULTS, "worldId": "testing", "username": "Gamemaster"}
            controller.worlds.return_value = [{"id": "testing", "name": "Testing"}]
            guide = GuidedFoundrySetup(parent, controller=controller,
                store=HardwareProfiles(Path(self.directory) / "profiles.json"), **kwargs)
            # Render real controls; asynchronous launch/ack has separate tests.
            guide.coordinator.validate = lambda choices: choices
            guide.coordinator.begin = lambda choices: True
            guide.coordinator.poll = lambda: {"phase": "ready", "ready": True,
                "message": "Foundry ready", "error": None, "warning": ""}
            return guide
        self.app._startup_factory = lambda parent, **kwargs: SessionStartupUI(parent,
            inventory_loader=lambda: (self.displays, self.cameras),
            foundry_setup_factory=foundry_step, **kwargs)

        class Service:
            running = False
            error = ""
            phase = "starting_relay"
            def start(inner):
                inner.running = True
            def stop(inner):
                inner.running = False
            def startup_status(inner):
                return {"phase": inner.phase, "message": inner.phase.replace("_", " "),
                        "error": "", "scene_info": {"sceneId": "test", "sceneW": 1920,
                            "sceneH": 1080, "gridPx": 40} if inner.phase == "ready" else None}
        self.service = self.app.service = Service()

    def until(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.01)
        self.assertTrue(predicate())

    def assert_visible_inside_root(self, name, widget):
        self.root.update_idletasks()
        x = widget.winfo_rootx() - self.root.winfo_rootx()
        y = widget.winfo_rooty() - self.root.winfo_rooty()
        width, height = widget.winfo_width(), widget.winfo_height()
        row = {"widget": name, "window": [self.root.winfo_width(), self.root.winfo_height()],
               "bounds": [x, y, width, height], "mapped": bool(widget.winfo_ismapped())}
        self.metrics.append(row)
        self.assertTrue(row["mapped"], row)
        self.assertGreater(width, 2, row)
        self.assertGreaterEqual(x, 0, row)
        self.assertGreaterEqual(y, 0, row)
        self.assertLessEqual(x + width, self.root.winfo_width(), row)
        self.assertLessEqual(y + height, self.root.winfo_height(), row)

    def assert_one_window(self):
        import tkinter as tk
        def visit(widget):
            for child in widget.winfo_children():
                self.assertNotIsInstance(child, tk.Toplevel)
                visit(child)
        self.assertTrue(self.root.winfo_ismapped())
        visit(self.root)

    def test_setup_connection_runtime_and_cancel_share_existing_window(self):
        from control_panel import ControlPanel
        try:
            for size in ("860x620", "700x460"):
                self.root.geometry(size)
                self.root.update()
                self.service.phase = "starting_relay"
                calls = []

                def run(_service, *, startup):
                    calls.append(startup)
                    panel = ControlPanel(mode="foundry", tk_root=self.root,
                                         parent=startup.runtime_parent, access=self.app.auth,
                                         foundry_service=self.service)
                    panel.show()
                    startup.runtime_started(panel)
                    self.root.update()
                    self.assert_one_window()
                    self.assertIs(panel.root.winfo_toplevel(), self.root)
                    self.assert_visible_inside_root("runtime-tabs", panel.tabs)
                    panel._on_close()
                    self.assertTrue(panel.pop_actions()["exit"])
                    self.assertTrue(self.root.winfo_ismapped())
                    panel.root.destroy()

                self.app.run_observer = run
                with patch.object(self.root, "withdraw", wraps=self.root.withdraw) as withdraw:
                    self.app.start_session(observer=True)
                    self.until(lambda: self.app._startup.phase == "hardware")
                    ui = self.app._startup
                    self.assert_one_window()
                    for name in ("continue_button", "cancel_button", "refresh_button"):
                        self.assert_visible_inside_root(name, getattr(ui, name))
                    self.assertTrue(ui.selector.ready)
                    ui.continue_button.invoke()
                    self.assertEqual(ui.phase, "foundry_setup")
                    self.assert_one_window()
                    self.assertGreaterEqual(ui.selector.master._scroll_canvas.winfo_height(), 150)
                    self.assertFalse(ui.refresh_button.winfo_ismapped())
                    for name in ("continue_button", "cancel_button", "edit_button"):
                        self.assert_visible_inside_root("foundry-" + name, getattr(ui, name))
                    self.assertEqual(ui.foundry_setup.world_choice.get(), "Testing (testing)")
                    ui.continue_button.invoke()
                    self.assertEqual(calls, [])
                    for phase, heading in (("waiting_module", "Waiting for the Foundry module"),
                                           ("waiting_scene", "Waiting for the Foundry scene")):
                        self.service.phase = phase
                        self.until(lambda: ui.heading.get() == heading)
                        self.assert_one_window()
                        self.assert_visible_inside_root(phase + "-cancel", ui.cancel_button)
                        self.assertEqual(calls, [])
                    self.service.phase = "ready"
                    self.until(lambda: not self.app.busy)
                    self.assertEqual(len(calls), 1)
                    self.assert_one_window()
                    withdraw.assert_not_called()
            self.app.start_session()
            self.until(lambda: self.app._startup.phase == "hardware")
            self.app._startup.cancel_button.invoke()
            self.assertFalse(self.app.busy)
            self.assert_one_window()
        finally:
            out = Path(__file__).resolve().parents[1] / "artifacts/startup-ui-20261105"
            out.mkdir(exist_ok=True, parents=True)
            (out / "metrics.json").write_text(json.dumps(self.metrics, indent=2) + "\n")
            self.app.close()
