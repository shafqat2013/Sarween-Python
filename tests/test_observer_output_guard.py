"""Observer entry must inhibit the old tracker and all delivery paths.

These exercise the real session runner using finite fake camera frames and a
non-Tk panel. No camera, native window, service or desktop access is involved.
"""

import asyncio
from collections import deque
from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

import foundryoutput as fo
from foundry_delivery import MoveOutbox
import observer_session
import v3_tracking as tracking


class GuardSetup:
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.context = ("observer-test", 1000, 800, 50, 0, 0, 1)
        self.box = MoveOutbox(ack_timeout=.01)
        self.box.set_context(self.context)
        for name, value in (("_delivery", self.box), ("_observation_only", False),
                            ("_tracking_output_paused", False), ("_capture_session", None),
                            ("_capture_ready", False), ("_capture_commands", deque()),
                            ("SCENE_ID", "observer-test"), ("_connection_ready", True),
                            ("_protocol_ready", True), ("MINI_TO_TOKEN", {"red": "red-token"})):
            self.stack.enter_context(patch.object(fo, name, value))
        self.stack.enter_context(patch.object(fo, "_scene_context", return_value=self.context))
        self.stack.enter_context(patch.object(fo, "_grid_to_pixels", return_value=(100, 100)))
        self.queue_control = self.stack.enter_context(patch.object(fo, "queue_control"))


class ObserverOutputGuardTests(GuardSetup, unittest.TestCase):
    def test_enter_and_exit_discard_old_intentions_and_active_calls_cannot_queue(self):
        fo.queue_cell_move("red", "r1c1")
        self.assertIn("red", self.box.snapshot())
        fo.set_observation_only(True)
        self.assertTrue(fo.tracking_output_paused())
        self.assertEqual(self.box.snapshot(), {})
        fo.queue_cell_move("red", "r2c2")
        fo.move_token_to_grid("red", "r3c3")
        self.assertEqual(self.box.snapshot(), {})
        self.assertEqual(fo.get_delivery_status()["message"], "Observer mode — Foundry movement off")
        # A stale producer bypassing the public API must not escape on exit.
        self.box.offer("red", "r4c4", self.context)
        fo.set_observation_only(False)
        self.assertFalse(fo.observation_only())
        self.assertEqual(self.box.snapshot(), {})

    def test_retry_is_inert_while_observation_is_active(self):
        fo.set_observation_only(True)
        self.box.offer("red", "r1c1", self.context)
        self.box.moves["red"].state = "failed"
        before = self.box.snapshot()
        with patch.object(self.box, "retry", wraps=self.box.retry) as retry:
            fo.retry_failed_moves()
            retry.assert_not_called()
        self.assertEqual(self.box.snapshot(), before)

    def test_guided_capture_finish_and_resume_cannot_release_observer_guard(self):
        fo._tracking_output_paused = True
        fo._capture_session = SimpleNamespace(session_id="capture-1")
        fo.set_observation_only(True)
        fo.finish_guided_capture("userStopped")
        self.assertTrue(fo.tracking_output_paused())
        fo.handle_capture_control({"action": "resume"})
        self.assertFalse(fo._tracking_output_paused)
        self.assertTrue(fo.observation_only())
        self.assertTrue(fo.tracking_output_paused())
        fo.queue_cell_move("red", "r1c1")
        self.assertEqual(self.box.snapshot(), {})

    def test_observer_does_not_clear_an_independent_guided_capture_pause_on_exit(self):
        fo._tracking_output_paused = True
        fo.set_observation_only(True)
        fo.set_observation_only(False)
        self.assertTrue(fo.tracking_output_paused())

    def test_queue_offer_and_observer_activation_cannot_repopulate_cleared_outbox(self):
        entered, release, activated = threading.Event(), threading.Event(), threading.Event()
        original_offer = self.box.offer
        errors = []

        def delayed_offer(*args, **kwargs):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test did not release the pending producer")
            return original_offer(*args, **kwargs)

        def produce():
            try:
                fo.queue_cell_move("red", "r1c1")
            except Exception as exc:
                errors.append(exc)

        def activate():
            fo.set_observation_only(True)
            activated.set()

        with patch.object(self.box, "offer", side_effect=delayed_offer):
            producer = threading.Thread(target=produce)
            transition = threading.Thread(target=activate)
            producer.start()
            self.assertTrue(entered.wait(2))
            transition.start()
            try:
                # With atomic guarding, activation waits for the older producer
                # and clears it afterward. Without it, the producer writes after
                # activation has cleared the outbox, reproducing the race.
                activated.wait(.1)
            finally:
                release.set()
                producer.join(2)
                transition.join(2)
        self.assertFalse(producer.is_alive())
        self.assertFalse(transition.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(fo.observation_only())
        self.assertEqual(self.box.snapshot(), {})


class ObserverSendGuardTests(GuardSetup, unittest.IsolatedAsyncioTestCase):
    async def run_one_iteration(self, socket):
        with patch.object(fo.asyncio, "sleep", AsyncMock(side_effect=asyncio.CancelledError)):
            with self.assertRaises(asyncio.CancelledError):
                await fo.send_loop(socket)

    async def test_latched_observer_cannot_prepare_or_send_even_a_directly_injected_old_intent(self):
        fo.set_observation_only(True)
        self.box.offer("red", "r1c1", self.context)
        socket = SimpleNamespace(send=AsyncMock())
        with patch.object(self.box, "prepare", wraps=self.box.prepare) as prepare:
            await self.run_one_iteration(socket)
            prepare.assert_not_called()
        socket.send.assert_not_called()

    async def test_observer_activated_during_preparation_blocks_the_prepared_payload(self):
        self.box.offer("red", "r1c1", self.context)
        original = self.box.prepare

        def prepare_then_activate(*args, **kwargs):
            result = original(*args, **kwargs)
            fo.set_observation_only(True)
            return result

        socket = SimpleNamespace(send=AsyncMock())
        with patch.object(self.box, "prepare", side_effect=prepare_then_activate):
            await self.run_one_iteration(socket)
        socket.send.assert_not_called()
        self.assertEqual(self.box.snapshot(), {})

    async def test_regular_delivery_still_sends_when_observer_is_off(self):
        self.box.offer("red", "r1c1", self.context)
        socket = SimpleNamespace(send=AsyncMock())
        await self.run_one_iteration(socket)
        socket.send.assert_awaited_once()
        self.assertEqual(json.loads(socket.send.call_args.args[0])["type"], "moveToken")

    async def test_already_queued_physical_tap_cannot_change_selection_in_observer_mode(self):
        fo.set_observation_only(True)
        socket = SimpleNamespace(send=AsyncMock())
        control_queue = SimpleNamespace(get=AsyncMock(side_effect=[
            {"type": "miniTap", "miniId": "red"}, asyncio.CancelledError()]))
        with patch.object(fo, "_ctrl_queue", control_queue):
            with self.assertRaises(asyncio.CancelledError):
                await fo.ctrl_loop(socket)
        socket.send.assert_not_called()

    async def test_already_queued_assignment_request_is_dropped_while_observing(self):
        fo.set_observation_only(True)
        socket = SimpleNamespace(send=AsyncMock())
        assignment_queue = SimpleNamespace(get=AsyncMock(side_effect=[
            ("red", "observer-test"), asyncio.CancelledError()]))
        with patch.object(fo, "_assign_queue", assignment_queue):
            with self.assertRaises(asyncio.CancelledError):
                await fo.assign_request_loop(socket)
        socket.send.assert_not_called()


class ObserverStartupGuardTests(GuardSetup, unittest.TestCase):
    def test_home_observer_wrapper_inhibits_before_setup_and_clears_on_success_or_failure(self):
        import main
        service = object()
        for failure in (None, RuntimeError("camera failed"), SystemExit("cancelled")):
            with self.subTest(failure=failure):
                fo.queue_cell_move("red", "r1c1")

                def start(actual_service, *, observer_only):
                    self.assertIs(actual_service, service)
                    self.assertTrue(observer_only)
                    self.assertTrue(fo.observation_only())
                    self.assertEqual(self.box.snapshot(), {})
                    # Public output remains inhibited during setup; any stale
                    # internal intent must also be discarded by final cleanup.
                    fo.queue_cell_move("red", "r2c2")
                    self.assertEqual(self.box.snapshot(), {})
                    self.box.offer("red", "r3c3", self.context)
                    if failure:
                        raise failure
                    return "session-return"

                with patch.object(main, "run_live_session", side_effect=start):
                    if failure:
                        with self.assertRaises(type(failure)):
                            main.run_observer_session(service)
                    else:
                        self.assertEqual(main.run_observer_session(service), "session-return")
                self.assertFalse(fo.observation_only())
                self.assertEqual(self.box.snapshot(), {})

    def test_real_startup_forwards_observer_only_through_setup_and_calibration(self):
        import main
        flags = []
        service = SimpleNamespace(access=SimpleNamespace(require=Mock()))
        with patch.object(main.s, "initialize", side_effect=lambda **_kw: flags.append(fo.observation_only())), \
             patch.object(main.s, "selected_mode", "self_hosted"), \
             patch.object(main.c, "calibrate", side_effect=lambda **_kw: flags.append(fo.observation_only())), \
             patch.object(tracking, "begin_session", return_value=None) as begin, \
             patch.object(tracking, "close_pending_recording") as finish:
            main.run_observer_session(service)
        self.assertEqual(flags, [True, True])
        self.assertTrue(begin.call_args.kwargs["observation_only"])
        self.assertIs(begin.call_args.kwargs["access"], service.access)
        finish.assert_called_once()
        self.assertFalse(fo.observation_only())


class ObserverSessionIntegrationTests(GuardSetup, unittest.TestCase):
    def run_fake_session(self, *, construction_error=False, closed=False, update_error=False, preenabled=False):
        pixels = np.zeros((90, 160, 3), np.uint8)
        flags_between_frames = []
        session = SimpleNamespace(is_recording=False, warp_w=160, warp_h=90,
                                  grid_w=16, grid_h=9, frame_time=0,
                                  lock_miss_streak=0, _watermark_current_exclusion_warp=np.zeros((90, 160), np.uint8),
                                  camera_calibration_metadata=Mock(return_value={}), close=Mock())

        def frames():
            for index in range(3):
                session.frame_time = index / 8
                yield SimpleNamespace(locked=True, last_missing_ids=[], last_marker_count=4,
                                      frame_idx=index, warp_bgr=pixels, cam_bgr=pixels)
                flags_between_frames.append(fo.observation_only())

        session.frames = frames
        panel = MagicMock()
        panel.pump.return_value = True
        panel.pop_actions.side_effect = [{"open_observer": not preenabled, "retry_moves": True},
                                       {"retry_moves": True}, {}]
        panel.get_toggles.return_value = {}
        panel.get_motion_thresh.return_value = 20
        old_engine = MagicMock()
        old_engine.prev_state = old_engine.cell_hist = old_engine.last_emitted = {}
        old_engine.last_physical_xy = old_engine.last_seen = {}
        old_engine.step.return_value = SimpleNamespace(reason=None, detections={}, lost_minis=[],
                                                       tapped_mini="red", moves=[])
        observer = MagicMock()
        observer.closed = closed
        observer.update.side_effect = RuntimeError("Simulated observer failure") if update_error else lambda *a, **kw: None
        callback = Mock()
        with ExitStack() as patchers:
            replacements = ((tracking, "initialize_user_data", Mock()),
                            (tracking.s, "load_last_selection", Mock(return_value={"mode": "foundry"})),
                            (tracking, "load_profiles_with_curves", Mock(return_value={})),
                            (tracking.ml, "load_synced_library", Mock(return_value={})),
                            (tracking.ml, "library_rows", Mock(return_value=[])),
                            (tracking.core, "CVCoreSession", Mock(return_value=session)),
                            (tracking, "ControlPanel", Mock(return_value=panel)),
                            (tracking, "TrackingEngine", Mock(return_value=old_engine)),
                            (tracking, "create_tracking_windows", Mock(return_value=MagicMock())),
                            (tracking, "recordings_dir", Mock(return_value=Path("/tmp/observer-guard-test"))),
                            (tracking, "tracking_context", Mock(return_value={})),
                            (fo, "get_mini_assignments", Mock(return_value=({}, {}))),
                            (fo, "set_grid_params", Mock()),
                            (fo, "update_camera_lock", Mock()),
                            (fo, "get_scene_params", Mock(return_value={"sceneId": "observer-test"})),
                            (fo, "pop_capture_command", Mock(return_value=None)),
                            (fo, "guided_capture_active", Mock(return_value=False)),
                            (fo, "get_latest_rendered_reference", Mock(return_value=None)))
            for module, name, value in replacements:
                patchers.enter_context(patch.object(module, name, value))
            patchers.enter_context(patch("tk_camera_preview._get_root", return_value=None))
            factory = patchers.enter_context(patch.object(observer_session, "LiveObserverController",
                side_effect=RuntimeError("Simulated missing window") if construction_error else None,
                return_value=observer))
            retry = patchers.enter_context(patch.object(fo, "retry_failed_moves"))
            result = tracking.begin_session(callback, show_windows=False, observation_only=preenabled)
            retry.assert_not_called()
        self.assertIsNone(result)
        factory.assert_called_once()
        old_engine.step.assert_not_called()
        callback.assert_not_called()
        session.close.assert_called_once()
        self.assertEqual(flags_between_frames, [True, True, True])
        self.assertFalse(fo.observation_only(), "The session's finalizer releases the guard after cleanup")
        controls = [call.args[0] for call in self.queue_control.call_args_list]
        self.assertFalse(any(row.get("type") in {"moveToken", "miniTap"} for row in controls))
        self.assertEqual(self.box.snapshot(), {})
        return observer, panel

    def test_open_observer_bypasses_the_old_detector_and_movement_callback(self):
        observer, _panel = self.run_fake_session()
        self.assertEqual(observer.update.call_count, 3)

    def test_window_construction_failure_keeps_old_tracker_and_output_disabled(self):
        observer, panel = self.run_fake_session(construction_error=True)
        observer.update.assert_not_called()
        self.assertTrue(any("remains off" in str(call) for call in panel.set_hint.call_args_list))

    def test_closing_observer_window_does_not_fall_back_to_the_old_tracker(self):
        observer, _panel = self.run_fake_session(closed=True)
        observer.update.assert_not_called()

    def test_observer_update_error_does_not_fall_back_to_the_old_tracker(self):
        observer, _panel = self.run_fake_session(update_error=True)
        self.assertEqual(observer.update.call_count, 3)

    def test_home_preenabled_observer_mode_opens_without_the_session_button_action(self):
        observer, _panel = self.run_fake_session(preenabled=True)
        self.assertEqual(observer.update.call_count, 3)


if __name__ == "__main__":
    unittest.main()
