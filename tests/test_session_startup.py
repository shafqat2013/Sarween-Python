"""Queued startup lifecycle tests: no Tk windows, devices or relay sockets."""
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
import heapq
import queue
import unittest
from unittest.mock import Mock, patch

import session_startup as startup


DISPLAYS = [{"index": 2, "name": "TV", "width": 1920, "height": 1080}]
CAMERAS = [{"index": 7, "model": "OBS Virtual Camera", "resolution": "Not opened"}]
SELECTION = {"display_index": 2, "webcam_index": 7, "camera_profile_id": "profile",
             "camera_profile_confirmed": True, "mode": "foundry", "map_path": None}
SCENE = {"sceneId": "scene", "sceneW": 1920, "sceneH": 1080, "gridPx": 40}


class Scheduler:
    """Model Tk after/after_cancel without invoking callbacks synchronously."""
    def __init__(self):
        self.now, self.serial = 0.0, 0
        self.queue, self.pending = [], {}

    def after(self, milliseconds, callback):
        self.serial += 1
        token = f"timer-{self.serial}"
        self.pending[token] = callback
        heapq.heappush(self.queue, (self.now + milliseconds / 1000, self.serial, token))
        return token

    def after_cancel(self, token):
        self.pending.pop(token, None)

    def next(self):
        while self.queue:
            when, _serial, token = heapq.heappop(self.queue)
            callback = self.pending.pop(token, None)
            if callback is not None:
                self.now = when
                callback()
                return callback
        raise AssertionError("No scheduled callback")

    def until(self, condition, limit=40):
        for _ in range(limit):
            if condition():
                return
            self.next()
        raise AssertionError("Startup condition did not become true")


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Button:
    def __init__(self):
        self.disabled, self.options = False, {}

    def state(self, values=None):
        if values is not None:
            if "disabled" in values:
                self.disabled = True
            if "!disabled" in values:
                self.disabled = False
        return ("disabled",) if self.disabled else ()

    def configure(self, **options):
        self.options.update(options)

    def pack(self, **_options):
        self.packed = True

    def pack_forget(self):
        self.packed = False


class Progress:
    def __init__(self):
        self.running = False

    def start(self, _interval):
        self.running = True

    def stop(self):
        self.running = False


class Access:
    allowed = True

    def require(self):
        if not self.allowed:
            raise PermissionError("Sign in required")


class Selector:
    def __init__(self):
        self.enabled = True
        self.ready = True
        self.draft = deepcopy(SELECTION)
        self.inventories = []

    def set_enabled(self, enabled):
        self.enabled = bool(enabled)

    def selection(self):
        if not self.ready:
            raise ValueError("Choose a camera")
        return deepcopy(self.draft)

    def set_inventory(self, displays, cameras, defaults=None):
        self.inventories.append(deepcopy((displays, cameras, defaults)))


class Service:
    def __init__(self, running=False):
        self.running = running
        self.phase = "starting_relay"
        self.error = ""
        self.start_count = self.stop_count = 0
        self.start_error = None

    def start(self):
        self.start_count += 1
        if self.start_error:
            raise self.start_error
        self.running = True

    def stop(self):
        self.stop_count += 1
        self.running = False

    def startup_status(self):
        return {"phase": self.phase, "message": f"Progress: {self.phase}",
                "scene_info": deepcopy(SCENE) if self.phase == "ready" else None,
                "error": self.error}


class SessionStartupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.scheduler = Scheduler()
        self.workers = deque()
        owner = self

        class DeferredThread:
            def __init__(self, *, target, **_kwargs):
                self.target = target

            def start(self):
                owner.workers.append(self.target)

        self.stack.enter_context(patch.object(startup.threading, "Thread", DeferredThread))
        self.stack.enter_context(patch.object(startup.time, "monotonic", side_effect=lambda: self.scheduler.now))
        self.camera = self.stack.enter_context(patch("cv2.VideoCapture", side_effect=AssertionError("No camera")))
        self.toplevel = self.stack.enter_context(patch("tkinter.Toplevel", side_effect=AssertionError("No extra window")))
        self.stack.enter_context(patch.object(startup.hardware, "load_last_selection", return_value={
            "display_index": 2, "webcam_index": 7, "mode": "foundry"}))
        self.stack.enter_context(patch.object(startup.hardware, "_active_camera_profile_id", return_value="profile"))
        self.request_scene = self.stack.enter_context(patch("foundryoutput.request_scene_info"))

    def model(self, *, running=False, mode="foundry"):
        ui = startup.SessionStartupUI.__new__(startup.SessionStartupUI)
        ui.service, ui.access = Service(running), Access()
        ui.runtime_parent = object()
        ui.on_ready, ui.on_cancel, ui.on_runtime = Mock(), Mock(), Mock()
        ui.observer = True
        ui.inventory_loader = Mock(return_value=(deepcopy(DISPLAYS), deepcopy(CAMERAS)))
        ui.displays, ui.webcams = deepcopy(DISPLAYS), deepcopy(CAMERAS)
        ui.selection_data = ui.scene_info = None
        ui._closed = ui._launching = ui._owns_relay = False
        ui._after_id = None
        ui._inventory_generation = 0
        ui._inventory_results = queue.Queue()
        ui._last_scene_request = float("-inf")
        ui.phase = "hardware"
        ui.heading, ui.message = Value(), Value()
        ui.progress, ui.selector = Progress(), Selector()
        ui.selector.draft["mode"] = mode
        for name in ("refresh_button", "continue_button", "cancel_button", "edit_button"):
            setattr(ui, name, Button())
        ui.after, ui.after_cancel = self.scheduler.after, self.scheduler.after_cancel
        ui.destroy, ui.update_idletasks = Mock(), Mock()
        return ui

    def finish_inventory(self, ui):
        ui.refresh_inventory()
        self.assertEqual(ui.phase, "inventory")
        self.assertTrue(ui.progress.running)
        ui.on_ready.assert_not_called()
        self.workers.popleft()()
        self.scheduler.until(lambda: ui.phase == "hardware")

    def test_delayed_relay_module_and_scene_wait_inline_then_launch_exactly_once(self):
        ui = self.model()
        self.finish_inventory(ui)
        self.assertFalse(ui.continue_button.disabled)
        ui.on_ready.assert_not_called()
        ui.begin_connection()
        self.assertEqual(ui.phase, "connection")
        self.assertTrue(ui.progress.running)
        self.assertIn("starting_relay", ui.message.get())
        self.assertTrue(ui.continue_button.disabled)
        # A cold relay or a world not opened yet can wait beyond the old10s cap.
        for _ in range(80):
            self.scheduler.next()
        self.assertGreater(self.scheduler.now, 10)
        ui.on_ready.assert_not_called()
        ui.service.phase = "waiting_module"
        self.scheduler.next()
        self.assertEqual(ui.heading.get(), "Waiting for the Foundry module")
        self.assertEqual(ui.phase, "connection")
        ui.service.phase = "waiting_scene"
        self.scheduler.next()
        self.assertEqual(ui.heading.get(), "Waiting for the Foundry scene")
        self.request_scene.assert_called_once()
        ui.on_ready.assert_not_called()
        ui.service.phase = "ready"
        self.scheduler.next()
        self.assertEqual(ui.phase, "launching")
        self.assertEqual(ui.scene_info, SCENE)
        self.assertTrue(ui.cancel_button.disabled)
        ui.on_ready.assert_not_called()  # let launching text paint first
        ui._queue_launch()  # repeated readiness cannot schedule a second launch
        self.assertEqual(len(self.scheduler.pending), 1)
        self.scheduler.next()
        ui.on_ready.assert_called_once_with(ui)
        self.assertFalse(self.scheduler.pending)
        self.assertEqual(ui.selection_data, SELECTION)
        self.camera.assert_not_called()
        self.toplevel.assert_not_called()

    def test_inventory_worker_and_poll_only_update_widgets_on_poll(self):
        ui = self.model()
        ui.refresh_inventory()
        ui.inventory_loader.assert_not_called()
        self.assertFalse(ui.selector.enabled)
        self.assertTrue(ui.continue_button.disabled)
        self.scheduler.next()  # still waiting for worker
        self.assertEqual(ui.phase, "inventory")
        self.assertFalse(ui.selector.inventories)
        self.workers.popleft()()
        self.assertFalse(ui.selector.inventories)
        self.scheduler.next()
        self.assertEqual(ui.phase, "hardware")
        self.assertEqual(ui.selector.inventories[0][0:2], (DISPLAYS, CAMERAS))
        self.assertTrue(ui.selector.enabled)
        self.assertFalse(ui.progress.running)
        ui.on_ready.assert_not_called()

    def test_default_inventory_reads_displays_on_ui_then_passive_cameras_in_worker(self):
        ui = self.model()
        ui.inventory_loader = None
        operations = []
        ui.update_idletasks = Mock(side_effect=lambda: operations.append("paint"))
        with patch.object(startup.hardware, "detect_displays_backup",
                side_effect=lambda: operations.append("displays") or DISPLAYS) as displays, \
             patch.object(startup.hardware, "detect_webcams_passive",
                side_effect=lambda: operations.append("cameras") or CAMERAS) as cameras:
            ui.refresh_inventory()
            self.assertEqual(operations, ["paint", "displays"])
            self.assertTrue(ui.progress.running)
            displays.assert_called_once_with()
            cameras.assert_not_called()
            self.workers.popleft()()
            self.assertEqual(operations, ["paint", "displays", "cameras"])
            self.assertFalse(ui.selector.inventories)
            self.scheduler.next()
        self.assertEqual(ui.phase, "hardware")
        self.assertEqual((ui.displays, ui.webcams), (DISPLAYS, CAMERAS))
        ui.on_ready.assert_not_called()

    def test_cancel_inventory_ignores_late_worker_and_already_queued_poll(self):
        ui = self.model()
        ui.refresh_inventory()
        late_poll = self.scheduler.pending[ui._after_id]
        ui.cancel()
        self.workers.popleft()()
        late_poll()
        self.assertTrue(ui._closed)
        self.assertFalse(ui.selector.inventories)
        self.assertFalse(self.scheduler.pending)
        ui.on_cancel.assert_called_once_with(ui)
        ui.on_ready.assert_not_called()
        ui.destroy.assert_called_once()
        ui.cancel()
        ui.on_cancel.assert_called_once()

    def test_revoked_inventory_or_connection_or_launch_cancels_before_camera(self):
        for phase in ("inventory", "connection", "launching"):
            with self.subTest(phase=phase):
                ui = self.model()
                if phase == "inventory":
                    ui.refresh_inventory()
                    self.workers.popleft()()
                else:
                    ui.service.phase = "ready" if phase == "launching" else "waiting_module"
                    ui.begin_connection()
                ui.access.allowed = False
                self.scheduler.next()
                self.assertTrue(ui._closed)
                self.assertFalse(self.scheduler.pending)
                ui.on_ready.assert_not_called()
                ui.on_cancel.assert_called_once_with(ui)
                if phase != "inventory":
                    self.assertEqual(ui.service.stop_count, 1)
        self.camera.assert_not_called()

    def test_connection_failure_retry_preserves_hardware_draft(self):
        ui = self.model()
        ui.selector.draft["webcam_index"] = 11
        ui.service.phase, ui.service.error = "failed", "Address already in use"
        original = deepcopy(ui.selector.draft)
        ui.begin_connection()
        self.assertEqual(ui.phase, "failed")
        self.assertEqual(ui.continue_button.options["text"], "Retry")
        self.assertIn("already in use", ui.message.get())
        self.assertFalse(ui.progress.running)
        self.assertFalse(self.scheduler.pending)
        ui.on_ready.assert_not_called()
        self.assertEqual(ui.selector.draft, original)
        ui.service.phase, ui.service.error = "waiting_module", ""
        ui.begin_connection()
        self.assertEqual(ui.phase, "connection")
        self.assertEqual(ui.selection_data, original)
        ui.service.phase = "ready"
        self.scheduler.until(lambda: ui.on_ready.called)
        ui.on_ready.assert_called_once_with(ui)
        self.assertEqual(ui.inventory_loader.call_count, 0)

    def test_start_exception_is_visible_retryable_and_keeps_draft(self):
        ui = self.model()
        ui.service.start_error = OSError("Socket unavailable")
        ui.begin_connection()
        self.assertEqual(ui.phase, "failed")
        self.assertEqual(ui.selector.draft, SELECTION)
        self.assertFalse(ui.continue_button.disabled)
        self.assertEqual(ui.message.get(), "Socket unavailable")
        ui.service.start_error = None
        ui.service.phase = "ready"
        ui.begin_connection()
        self.scheduler.next()
        ui.on_ready.assert_called_once_with(ui)

    def test_cancel_preserves_preexisting_relay_and_stops_newly_owned_relay(self):
        for already_running in (False, True):
            with self.subTest(already_running=already_running):
                ui = self.model(running=already_running)
                ui.service.phase = "waiting_module"
                ui.begin_connection()
                self.assertEqual(ui._owns_relay, not already_running)
                late_poll = self.scheduler.pending[ui._after_id]
                ui.cancel()
                late_poll()
                self.assertEqual(ui.service.stop_count, 0 if already_running else 1)
                self.assertEqual(ui.service.running, already_running)
                self.assertFalse(self.scheduler.pending)
                ui.on_ready.assert_not_called()

    def test_cancel_after_ready_before_launch_ignores_queued_launch(self):
        ui = self.model()
        ui.service.phase = "ready"
        ui.begin_connection()
        late_launch = self.scheduler.pending[ui._after_id]
        self.assertEqual(ui.phase, "launching")
        ui.cancel()
        late_launch()
        ui.on_ready.assert_not_called()
        self.assertEqual(ui.service.stop_count, 1)
        ui.on_cancel.assert_called_once_with(ui)

    def test_self_hosted_launch_requires_continue_and_never_starts_relay(self):
        ui = self.model(mode="self_hosted")
        self.finish_inventory(ui)
        ui.on_ready.assert_not_called()
        self.assertEqual(ui.service.start_count, 0)
        ui.begin_connection()
        self.assertEqual(ui.phase, "launching")
        ui.on_ready.assert_not_called()
        self.scheduler.next()
        ui.on_ready.assert_called_once_with(ui)
        self.assertEqual(ui.service.start_count, 0)
        self.request_scene.assert_not_called()

    def test_change_hardware_cancels_wait_and_rechecks_current_draft(self):
        ui = self.model()
        ui.service.phase = "waiting_module"
        ui.begin_connection()
        ui.edit_hardware()
        self.assertEqual(ui.phase, "hardware")
        self.assertTrue(ui.selector.enabled)
        self.assertFalse(ui.refresh_button.disabled)
        self.assertFalse(ui.continue_button.disabled)
        self.assertFalse(self.scheduler.pending)
        ui.selector.ready = False
        ui._selection_changed()
        self.assertTrue(ui.continue_button.disabled)
        ui.begin_connection()
        self.assertEqual(ui.phase, "hardware")
        self.assertIn("Choose a camera", ui.message.get())
        ui.on_ready.assert_not_called()
        ui.selector.ready = True
        ui.selector.draft["webcam_index"] = 12
        ui.service.phase = "ready"
        ui.begin_connection()
        self.scheduler.next()
        self.assertEqual(ui.selection_data["webcam_index"], 12)
        ui.on_ready.assert_called_once_with(ui)

    def test_failed_readiness_recheck_resumes_wait_and_launches_on_later_ready(self):
        ui = self.model()
        ui.service.phase = "ready"
        ui.begin_connection()
        self.scheduler.next()
        ui.on_ready.assert_called_once()
        ui.service.phase = "waiting_module"
        ui.resume_wait()
        self.assertFalse(ui.cancel_button.disabled)
        self.assertFalse(ui.edit_button.disabled)
        self.scheduler.next()
        self.assertEqual(ui.phase, "connection")
        ui.service.phase = "ready"
        self.scheduler.until(lambda: ui.on_ready.call_count == 2)
        self.assertEqual(ui.on_ready.call_count, 2)

    def test_runtime_handoff_stops_progress_and_keeps_existing_root_owner(self):
        ui = self.model()
        ui.progress.start(12)
        panel = object()
        ui.runtime_started(panel)
        self.assertFalse(ui.progress.running)
        ui.on_runtime.assert_called_once_with(panel)
        ui.on_ready.assert_not_called()
        ui.on_cancel.assert_not_called()

    def test_stale_inventory_generation_cannot_replace_newer_refresh(self):
        ui = self.model()
        ui.refresh_inventory()
        old_worker = self.workers.popleft()
        ui.refresh_inventory()
        new_worker = self.workers.popleft()
        ui.inventory_loader.side_effect = [(DISPLAYS, [{**CAMERAS[0], "index": 9}]), (DISPLAYS, CAMERAS)]
        new_worker()
        old_worker()
        self.scheduler.next()
        self.assertEqual(ui.webcams[0]["index"], 9)
        self.assertEqual(len(ui.selector.inventories), 1)
        self.assertFalse(self.scheduler.pending)

    def test_inventory_error_stays_visible_and_can_be_refreshed(self):
        ui = self.model()
        ui.inventory_loader.side_effect = [RuntimeError("Native inventory unavailable"), (DISPLAYS, CAMERAS)]
        self.finish_inventory(ui)
        self.assertIn("Native inventory unavailable", ui.message.get())
        self.assertFalse(ui.refresh_button.disabled)
        self.assertTrue(ui.continue_button.disabled)
        self.assertFalse(ui.progress.running)
        self.finish_inventory(ui)
        self.assertEqual(ui.phase, "hardware")
        self.assertTrue(ui.selector.enabled)
        self.assertFalse(ui.continue_button.disabled)
        ui.on_ready.assert_not_called()

    def test_temporary_scene_request_error_cannot_strand_startup(self):
        ui = self.model()
        ui.service.phase = "waiting_scene"
        self.request_scene.side_effect = OSError("Module reconnecting")
        ui.begin_connection()
        self.assertTrue(ui.phase == "failed" or self.scheduler.pending)
        self.assertFalse(ui._closed)
        ui.on_ready.assert_not_called()
        self.request_scene.side_effect = None
        ui.service.phase = "ready"
        if ui.phase == "failed":
            ui.begin_connection()
        self.scheduler.until(lambda: ui.on_ready.called)
        ui.on_ready.assert_called_once_with(ui)

    def test_readiness_snapshot_error_is_visible_or_keeps_a_cancellable_poll(self):
        ui = self.model()
        ui.service.startup_status = Mock(side_effect=RuntimeError("Snapshot unavailable"))
        ui.begin_connection()
        self.assertTrue(ui.phase == "failed" or self.scheduler.pending)
        self.assertFalse(ui._closed)
        ui.on_ready.assert_not_called()


if __name__ == "__main__":
    unittest.main()
