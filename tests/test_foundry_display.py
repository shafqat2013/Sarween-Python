"""Headless display-controller tests; no macOS window API is called."""

import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from foundry_display import (FOUNDRY_BUNDLE_ID, FoundryDisplayController,
                             MacFoundryDisplayBackend, screen_coordinates)


DISPLAYS = [{"id": "1", "name": "Mac", "primary": True, "x": 0, "y": 0,
             "width": 1200, "height": 800},
            {"id": "2", "name": "TV", "primary": False, "x": -1920, "y": -280,
             "width": 1920, "height": 1080}]


class Backend:
    supported = True

    def __init__(self):
        self.permission = True
        self.keys = [(12, 100)]
        self.moves = []
        self.scans = 0
        self.block = None
        self.entered = threading.Event()

    def trusted(self):
        return self.permission

    def displays(self):
        return [dict(item) for item in DISPLAYS]

    def windows(self):
        self.scans += 1
        return list(self.keys)

    def place(self, key, display, cancelled):
        self.entered.set()
        if self.block is not None:
            self.block.wait(1)
        if cancelled():
            return False
        self.moves.append((key, dict(display)))
        return True


class DisplayControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "display.json"
        self.backend = Backend()
        self.controller = FoundryDisplayController(backend=self.backend, path=self.path,
                                                   poll_interval=0.01)

    def tearDown(self):
        if self.backend.block:
            self.backend.block.set()
        self.controller.close()
        self.directory.cleanup()

    def until(self, condition):
        deadline = time.monotonic() + 2
        while not condition():
            if time.monotonic() >= deadline:
                self.fail("Display worker did not reach expected state")
            time.sleep(0.005)

    def test_default_is_idle_and_prefers_external_display(self):
        state = self.controller.status()
        self.assertEqual(state["selectedDisplay"], "2")
        self.assertFalse(state["auto"])
        self.assertIsNone(self.controller._thread)
        self.assertEqual(self.backend.scans, 0)
        self.assertFalse(self.path.exists())

    def test_explicit_placement_runs_off_the_calling_thread(self):
        self.assertTrue(self.controller.place())
        self.until(lambda: not self.controller.status()["busy"])
        self.assertEqual(self.backend.moves, [(None, DISPLAYS[1])])
        self.assertIn("full screen on TV", self.controller.status()["message"])
        self.assertIsNot(self.controller._thread, threading.current_thread())

    def test_permission_denial_never_moves_or_prompts(self):
        self.backend.permission = False
        state = self.controller.status()
        self.assertFalse(state["permission"])
        self.assertIn("Privacy & Security", state["message"])
        self.assertIn("does not request permission automatically", state["message"])
        self.assertFalse(self.controller.place())
        self.assertIsNone(self.controller._thread)
        self.assertEqual(self.backend.moves, [])

    def test_selected_display_and_explicit_auto_preference_are_persisted(self):
        self.controller.configure(display_id="1", auto=True)
        self.assertEqual(json.loads(self.path.read_text()), {"displayId": "1", "auto": True})
        other = FoundryDisplayController(backend=self.backend, path=self.path)
        try:
            self.assertTrue(other._auto)
            self.assertEqual(other._selected, "1")
            self.assertIsNone(other._thread)
        finally:
            other.close()

    def test_auto_only_moves_new_windows_once(self):
        self.controller.configure(auto=True)
        self.until(lambda: self.backend.scans > 1)
        self.assertEqual(self.backend.moves, [])
        self.backend.keys.append((13, 101))
        self.until(lambda: len(self.backend.moves) == 1)
        self.assertEqual(self.backend.moves[0], ((13, 101), DISPLAYS[1]))
        previous = self.backend.scans
        self.until(lambda: self.backend.scans > previous + 2)
        self.assertEqual(len(self.backend.moves), 1)
        self.controller.configure(auto=False)
        self.backend.keys.append((14, 102))
        time.sleep(0.04)
        self.assertEqual(len(self.backend.moves), 1)

    def test_auth_revocation_cancels_in_flight_placement(self):
        self.backend.block = threading.Event()
        self.assertTrue(self.controller.place())
        self.assertTrue(self.backend.entered.wait(1))
        self.controller.set_enabled(False)
        self.backend.block.set()
        self.until(lambda: not self.controller.status()["busy"])
        self.assertEqual(self.backend.moves, [])
        self.assertFalse(self.controller.place())

    def test_changing_target_cancels_old_display_request(self):
        self.backend.block = threading.Event()
        self.assertTrue(self.controller.place())
        self.assertTrue(self.backend.entered.wait(1))
        self.controller.configure(display_id="1")
        self.backend.block.set()
        self.until(lambda: not self.controller.status()["busy"])
        self.assertEqual(self.backend.moves, [])

    def test_invalid_or_missing_display_is_not_silently_replaced(self):
        self.controller.status()
        with self.assertRaises(ValueError):
            self.controller.configure(display_id="missing")
        self.controller._selected = "disconnected-tv"
        self.assertFalse(self.controller.place())
        self.assertIn("saved display is disconnected", self.controller.status()["message"])
        self.assertEqual(self.backend.moves, [])

    def test_backend_failure_and_shutdown_are_visible_and_safe(self):
        self.backend.place = lambda *_args: (_ for _ in ()).throw(RuntimeError("Open Foundry first"))
        self.assertTrue(self.controller.place())
        self.until(lambda: not self.controller.status()["busy"])
        self.assertEqual(self.controller.status()["message"], "Open Foundry first")
        self.controller.close()
        self.assertFalse(self.controller.place())

    def test_close_cancels_worker_without_waiting_for_an_app_response(self):
        self.backend.block = threading.Event()
        self.assertTrue(self.controller.place())
        self.assertTrue(self.backend.entered.wait(1))
        start = time.monotonic()
        self.controller.close()
        self.assertLess(time.monotonic() - start, 0.5)
        self.backend.block.set()
        self.until(lambda: not self.controller._thread.is_alive())
        self.assertEqual(self.backend.moves, [])

    def test_permission_or_config_failure_does_not_enable_auto(self):
        with patch("foundry_display.atomic_write_json", side_effect=OSError("read-only")):
            with self.assertRaises(OSError):
                self.controller.configure(auto=True)
        self.assertFalse(self.controller.status()["auto"])
        with self.assertRaises(ValueError):
            self.controller.configure(auto="yes")


class NativeBackendContractTests(unittest.TestCase):
    def test_coordinates_cover_left_above_and_below_displays(self):
        self.assertEqual(screen_coordinates((-1920, 0, 1920, 1080), 800),
                         {"x": -1920, "y": -280, "width": 1920, "height": 1080})
        self.assertEqual(screen_coordinates((0, 800, 1000, 600), 800)["y"], -600)
        self.assertEqual(screen_coordinates((0, -600, 1000, 600), 800)["y"], 800)

    def test_only_the_native_foundry_bundle_is_considered(self):
        def app(bundle, pid, terminated=False):
            return SimpleNamespace(bundleIdentifier=lambda: bundle,
                processIdentifier=lambda: pid, isTerminated=lambda: terminated)
        workspace = SimpleNamespace(runningApplications=lambda: [
            app("com.apple.Safari", 100), app(FOUNDRY_BUNDLE_ID, 10),
            app(FOUNDRY_BUNDLE_ID, 11), app(FOUNDRY_BUNDLE_ID, 12, True)])
        with patch.dict("sys.modules", {"AppKit": SimpleNamespace(NSWorkspace=
                SimpleNamespace(sharedWorkspace=lambda: workspace))}):
            result = MacFoundryDisplayBackend()._applications()
        self.assertEqual([item.processIdentifier() for item in result], [11, 10])

    def test_native_placement_verifies_target_before_entering_fullscreen(self):
        calls = []
        backend = MacFoundryDisplayBackend()
        full = {"value": False}
        ax = SimpleNamespace(true=True, false=False, settable=lambda *_args: True,
            boolean=lambda _window, name: name == "AXFullScreen" and full["value"],
            point=lambda _window: (-1920, -280))
        def set_attribute(_window, name, value):
            calls.append((name, value))
            if name == "AXFullScreen":
                full["value"] = value
        ax.set = set_attribute
        ax.geometry = lambda _window, name, _kind, _value: calls.append((name, None))
        backend._accessibility = ax
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        self.assertTrue(backend.place((12, 100), DISPLAYS[1], lambda: False))
        self.assertEqual([name for name, _ in calls],
                         ["AXPosition", "AXSize", "AXPosition", "AXFullScreen"])
        calls.clear()
        full["value"] = False
        ax.point = lambda _window: (0, 0)
        with self.assertRaisesRegex(RuntimeError, "did not move"):
            backend.place((12, 100), DISPLAYS[1], lambda: False)
        self.assertNotIn(("AXFullScreen", True), calls)

    def test_existing_fullscreen_exits_and_transient_frame_errors_retry(self):
        backend = MacFoundryDisplayBackend()
        full, calls = {"value": True}, []
        ax = SimpleNamespace(true=True, false=False, settable=lambda *_args: True,
            boolean=lambda _window, name: name == "AXFullScreen" and full["value"],
            point=lambda _window: (-1920, -280))
        def set_attribute(_window, name, value):
            calls.append((name, value))
            if name == "AXFullScreen":
                full["value"] = value
        fail_once = {"value": True}
        def geometry(_window, name, _kind, _value):
            calls.append((name, None))
            if fail_once["value"]:
                fail_once["value"] = False
                raise RuntimeError("Window is still transitioning")
        ax.set, ax.geometry = set_attribute, geometry
        backend._accessibility = ax
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        with patch("foundry_display.time.sleep"):
            self.assertTrue(backend.place((12, 100), DISPLAYS[1], lambda: False))
        self.assertEqual(calls[0], ("AXFullScreen", False))
        self.assertEqual(calls[-1], ("AXFullScreen", True))
        self.assertEqual(sum(name == "AXPosition" for name, _value in calls), 3)

    def test_cancellation_between_frame_updates_prevents_fullscreen(self):
        backend = MacFoundryDisplayBackend()
        cancelled, calls = {"value": False}, []
        ax = SimpleNamespace(true=True, false=False, settable=lambda *_args: True,
            boolean=lambda *_args: False, point=lambda _window: (-1920, -280))
        def geometry(_window, name, _kind, _value):
            calls.append(name)
            cancelled["value"] = True
        ax.set = lambda _window, name, _value: calls.append(name)
        ax.geometry = geometry
        backend._accessibility = ax
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        self.assertFalse(backend.place((12, 100), DISPLAYS[1], lambda: cancelled["value"]))
        self.assertEqual(calls, ["AXPosition"])

    def test_missing_permission_is_checked_before_any_application_lookup(self):
        backend = MacFoundryDisplayBackend()
        backend.trusted = lambda: False
        backend._applications = lambda: self.fail("No application enumeration expected")
        with self.assertRaises(PermissionError):
            backend.windows()


if __name__ == "__main__":
    unittest.main()
