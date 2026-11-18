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
        self.fullscreen_changes = []
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

    def set_fullscreen(self, key, enabled, cancelled):
        self.entered.set()
        if self.block is not None:
            self.block.wait(1)
        if cancelled():
            return False
        self.fullscreen_changes.append((key, enabled))
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

    def test_absolute_fullscreen_actions_work_without_a_connected_saved_tv(self):
        self.controller._selected = "disconnected-tv"
        for enabled in (True, False, False):
            self.assertTrue(self.controller.set_fullscreen(enabled))
            self.until(lambda: not self.controller.status()["busy"])
            state = self.controller.status()
            self.assertEqual(state["fullscreen"], enabled)
            self.assertTrue(state["fullscreenSupported"])
            self.assertIn("fullscreen" if enabled else "windowed", state["fullscreenMessage"])
            self.assertNotIn("disconnected", state["message"])
        self.assertEqual(self.backend.fullscreen_changes, [(None, True), (None, False), (None, False)])
        self.assertEqual(self.backend.moves, [])
        self.assertFalse(self.path.exists())

    def test_fullscreen_rejects_permission_disabled_closed_and_invalid_requests(self):
        for value in (None, 1, "true"):
            with self.assertRaises(ValueError):
                self.controller.set_fullscreen(value)
        self.backend.permission = False
        self.assertFalse(self.controller.set_fullscreen(True))
        self.backend.permission = True
        self.controller._last_refresh = float("-inf")
        self.controller.set_enabled(False)
        self.assertFalse(self.controller.set_fullscreen(True))
        self.controller.set_enabled(True)
        self.controller.close()
        self.assertFalse(self.controller.set_fullscreen(False))
        self.assertEqual(self.backend.fullscreen_changes, [])

    def test_fullscreen_busy_and_cancellation_never_issue_a_second_request(self):
        self.backend.block = threading.Event()
        self.assertTrue(self.controller.set_fullscreen(True))
        self.assertTrue(self.backend.entered.wait(1))
        self.assertFalse(self.controller.set_fullscreen(False))
        self.assertFalse(self.controller.place())
        self.controller.set_enabled(False)
        self.backend.block.set()
        self.until(lambda: not self.controller.status()["busy"])
        self.assertEqual(self.backend.fullscreen_changes, [])
        self.assertIsNone(self.controller.status()["fullscreen"])
        self.assertIn("cancelled", self.controller.status()["fullscreenMessage"])

    def test_fullscreen_failure_does_not_claim_state_or_change_placement(self):
        self.backend.set_fullscreen = lambda *_args: (_ for _ in ()).throw(RuntimeError("No fullscreen attribute"))
        self.assertTrue(self.controller.set_fullscreen(False))
        self.until(lambda: not self.controller.status()["busy"])
        self.assertIsNone(self.controller.status()["fullscreen"])
        self.assertEqual(self.controller.status()["fullscreenMessage"], "No fullscreen attribute")
        self.assertEqual(self.backend.moves, [])

    def test_unsupported_fullscreen_controls_explain_the_disabled_action(self):
        self.backend.set_fullscreen = None
        self.assertFalse(self.controller.status()["fullscreenSupported"])
        self.assertIn("unavailable", self.controller.status()["fullscreenMessage"])
        self.assertFalse(self.controller.set_fullscreen(True))

    def test_auto_placement_does_not_undo_manual_fullscreen_identity_change(self):
        self.controller.configure(auto=True)
        self.until(lambda: self.backend.scans > 1)
        original = self.backend.set_fullscreen
        def change_identity(key, enabled, cancelled):
            result = original(key, enabled, cancelled)
            self.backend.keys = [(12, 101)]
            return result
        self.backend.set_fullscreen = change_identity
        before = self.backend.scans
        self.assertTrue(self.controller.set_fullscreen(False))
        self.until(lambda: not self.controller.status()["busy"] and self.backend.scans > before + 2)
        self.assertEqual(self.backend.fullscreen_changes, [(None, False)])
        self.assertEqual(self.backend.moves, [])
        self.assertTrue(self.controller.status()["auto"])
        self.assertTrue(json.loads(self.path.read_text())["auto"])
        self.backend.keys.append((13, 102))
        self.until(lambda: len(self.backend.moves) == 1)
        self.assertEqual(self.backend.moves, [((13, 102), DISPLAYS[1])])


class NativeBackendContractTests(unittest.TestCase):
    def fullscreen_backend(self, initial=False, *, writable=True, button=False):
        backend = MacFoundryDisplayBackend()
        state, calls = {"value": initial}, []
        def set_attribute(_window, name, value):
            calls.append((name, value))
            state["value"] = value
        def press(_window, cancelled):
            if cancelled():
                return False
            calls.append(("AXFullScreenButton", "AXPress"))
            if button:
                state["value"] = not state["value"]
            return button
        ax = SimpleNamespace(true=True, false=False, optional_boolean=lambda *_args: state["value"],
            boolean=lambda *_args: state["value"], settable=lambda *_args: writable,
            set=set_attribute, press_fullscreen_button=press,
            geometry=lambda *_args: self.fail("Fullscreen must not move or resize Foundry"))
        backend._accessibility = ax
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        return backend, state, calls

    def test_standalone_fullscreen_is_absolute_idempotent_and_never_moves(self):
        backend, state, calls = self.fullscreen_backend()
        self.assertTrue(backend.set_fullscreen(None, True, lambda: False))
        self.assertTrue(state["value"])
        self.assertTrue(backend.set_fullscreen(None, True, lambda: False))
        self.assertEqual(calls, [("AXFullScreen", True)])
        self.assertTrue(backend.set_fullscreen(None, False, lambda: False))
        self.assertFalse(state["value"])
        self.assertEqual(calls[-1], ("AXFullScreen", False))

    def test_explicit_fullscreen_button_fallback_requires_known_opposite_state(self):
        backend, state, calls = self.fullscreen_backend(writable=False, button=True)
        for enabled in (True, True, False):
            self.assertTrue(backend.set_fullscreen(None, enabled, lambda: False))
        self.assertEqual(calls, [("AXFullScreenButton", "AXPress"), ("AXFullScreenButton", "AXPress")])
        state["value"] = None
        with self.assertRaisesRegex(RuntimeError, "does not expose"):
            backend.set_fullscreen(None, False, lambda: False)
        self.assertEqual(len(calls), 2)

    def test_unavailable_control_missing_window_and_cancel_do_not_succeed(self):
        backend, _state, calls = self.fullscreen_backend(writable=False)
        with self.assertRaisesRegex(RuntimeError, "cannot change"):
            backend.set_fullscreen(None, True, lambda: False)
        self.assertEqual(calls, [("AXFullScreenButton", "AXPress")])
        calls.clear()
        self.assertFalse(backend.set_fullscreen(None, True, lambda: True))
        self.assertEqual(calls, [])
        backend._visit_windows = lambda callback: None
        with self.assertRaisesRegex(RuntimeError, "Open the native Foundry"):
            backend.set_fullscreen(None, True, lambda: False)

    def test_fullscreen_readback_timeout_does_not_report_success(self):
        backend, _state, calls = self.fullscreen_backend()
        backend.ax.set = lambda *_args: calls.append("requested")
        with patch("foundry_display.time.monotonic", side_effect=[0, 5]):
            with self.assertRaisesRegex(RuntimeError, "has not confirmed"):
                backend.set_fullscreen(None, True, lambda: False)
        self.assertEqual(calls, ["requested"])

    def test_fullscreen_cancel_after_write_does_not_confirm_or_move(self):
        backend, _state, _calls = self.fullscreen_backend()
        with patch.object(backend.ax, "set", return_value=None):
            checks = iter((False, False, True))
            self.assertFalse(backend.set_fullscreen(None, True, lambda: next(checks)))

    def test_cancellation_during_capability_lookup_prevents_the_write(self):
        backend, _state, calls = self.fullscreen_backend()
        cancelled = {"value": False}
        def settable(*_args):
            cancelled["value"] = True
            return True
        backend.ax.settable = settable
        self.assertFalse(backend.set_fullscreen(None, True, lambda: cancelled["value"]))
        self.assertEqual(calls, [])

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

    def test_fullscreen_frame_becomes_settable_after_exit_and_transition(self):
        backend = MacFoundryDisplayBackend()
        full, calls, transitions = {"value": True}, [], {"remaining": 1}

        def settable(_window, name):
            calls.append(("check", name, full["value"]))
            if name == "AXFullScreen":
                return True
            if full["value"]:
                return False
            if transitions["remaining"]:
                transitions["remaining"] -= 1
                return False
            return True

        def set_attribute(_window, name, value):
            calls.append(("set", name, value))
            if name == "AXFullScreen":
                full["value"] = value

        ax = SimpleNamespace(true=True, false=False, settable=settable, set=set_attribute,
            boolean=lambda _window, name: name == "AXFullScreen" and full["value"],
            point=lambda _window: (-1920, -280),
            geometry=lambda _window, name, _kind, _value: calls.append(("geometry", name)))
        backend._accessibility = ax
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        with patch("foundry_display.time.sleep") as sleep:
            self.assertTrue(backend.place((12, 100), DISPLAYS[1], lambda: False))
        # Geometry checks must never reject the still-full-screen window.
        self.assertNotIn(("check", "AXPosition", True), calls)
        self.assertNotIn(("check", "AXSize", True), calls)
        self.assertEqual(calls[1], ("set", "AXFullScreen", False))
        sleep.assert_called_once_with(0.1)
        self.assertEqual(calls[-1], ("set", "AXFullScreen", True))

    def test_fullscreen_transition_cancellation_does_not_move_window(self):
        backend = MacFoundryDisplayBackend()
        full, cancelled, moves = {"value": True}, {"value": False}, []
        def set_attribute(_window, name, value):
            if name == "AXFullScreen":
                full["value"] = value
                cancelled["value"] = True
        backend._accessibility = SimpleNamespace(true=True, false=False,
            settable=lambda _window, name: name == "AXFullScreen" or not full["value"],
            set=set_attribute,
            boolean=lambda _window, name: name == "AXFullScreen" and full["value"],
            geometry=lambda *_args: moves.append(_args))
        backend._visit_windows = lambda callback: callback((12, 100), "window")
        self.assertFalse(backend.place((12, 100), DISPLAYS[1], lambda: cancelled["value"]))
        self.assertEqual(moves, [])

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
