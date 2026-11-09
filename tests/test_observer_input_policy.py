"""Owned watermark-free input avoids caption work; unknown cameras keep it."""
from copy import deepcopy
import tempfile
import unittest
from unittest.mock import patch

from observer_journal import replay_journal
from observer_session import LiveObserverController, watermark_mode
from tests.test_observer_session import FakeWindow, POSITIONS, image, view


class ObserverInputPolicyTests(unittest.TestCase):
    def test_only_owned_usb_source_opts_out_of_caption_recognition(self):
        for source in (None, {}, {"source": "opencv"}, {"source": "OBS Virtual Camera"},
                       {"source": "sarween-phone-usb"}, "unknown"):
            with self.subTest(source=source):
                self.assertEqual(watermark_mode({"metadata": {"camera_source": source}}), "board-template-v1")
        self.assertEqual(watermark_mode({"metadata": {"camera_source": {
            "source": "sarween-phone-usb", "transport": "usb"}}}),
                         "provided")

    def test_owned_source_policy_is_recorded_replayed_and_retained_on_recording_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            controller = LiveObserverController(None, folder, window_factory=FakeWindow,
                                                observation_only=False, mini_count=2)
            self.addCleanup(controller.close)
            mapping = {**view(), "sceneId": "input-policy-scene"}
            metadata = {"camera_source": {"source": "sarween-phone-usb", "transport": "usb"},
                        "scene_visual_revision": 0}
            def update(now):
                return controller.update(image(), None, now, view=mapping, locked=True,
                                         metadata=metadata, frame_index=round(now * 8))
            with patch("watermark_detection.BoardCaptionDetector", side_effect=AssertionError("Unneeded caption search")):
                update(1.)
                controller.window.choose(POSITIONS, controller.current[0], deepcopy(controller.current[1]))
                update(1.125)
                row = update(1.25)
                self.assertTrue(row["acquired"])
                self.assertEqual(controller.engine.watermark_mode, "provided")
                first = controller.journal
                controller.toggle_recording()
                self.assertEqual(first.manifest["initialization"]["watermark"]["mode"], "provided")
                self.assertTrue(replay_journal(first.folder)["verification_passed"])
                controller.toggle_recording()
                self.assertEqual(controller.engine.watermark_mode, "provided")
                self.assertEqual(controller.journal.manifest["initialization"]["watermark"]["mode"], "provided")

    def test_recording_restart_preserves_pending_command_provenance_and_replays_it(self):
        from tests.foundry_closed_loop_harness import ClosedLoopHarness
        for pending in (True, False):
            with self.subTest(pending=pending):
                harness = ClosedLoopHarness(artwork="disk")
                try:
                    harness.initialize()
                    harness.output.ack_delay = 2. if pending else 0.
                    harness.output.render_delay = 2.5 if pending else 0.
                    harness.move("mini1", (80, 160))
                    for _ in range(12):
                        row = harness.frame()
                        if row.get("events"):
                            break
                    self.assertEqual(len(harness.events), 1)
                    if not pending:
                        harness.frames(2)
                    harness.controller.toggle_recording()
                    journal = harness.controller.journal
                    self.assertTrue(harness.controller.is_recording)
                    harness.frames(24)
                    self.assertIsNotNone(harness.controller.engine)
                    self.assertFalse(harness.controller.engine.needs_reset)
                    harness.controller._stop_recording("restart_test")
                    report = replay_journal(journal.folder)
                    self.assertTrue(report["verification_passed"])
                finally:
                    harness.close()


if __name__ == "__main__":
    unittest.main()
