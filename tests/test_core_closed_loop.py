"""Repeated feedback cycles through camera/controller/bridge/real MoveOutbox.

Every mock redraw and acknowledgement is caused by a command emitted during
this test. These tests do not replay an earlier run's control trajectory.
"""
import math
import unittest

import cv2

from tests.foundry_closed_loop_harness import ClosedLoopHarness, POSITIONS


class CoreClosedLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def harness(self, **kwargs):
        harness = ClosedLoopHarness(**kwargs)
        self.addCleanup(harness.close)
        harness.initialize()
        return harness

    def assert_follows(self, harness, mini):
        command = next(row for row in reversed(harness.output.applied) if row["miniId"] == mini)
        actual = (command["x"] + 20, command["y"] + 20)
        self.assertLessEqual(math.dist(actual, harness.physical[mini]), 8)

    def test_three_complete_feedback_cycles_with_invisible_or_minimal_redraws(self):
        for artwork in ("hidden", "minimal"):
            with self.subTest(artwork=artwork):
                h = self.harness(artwork=artwork, ack_delay=.125, render_delay=.25)
                for count, (mini, destination) in enumerate(
                        (("mini1", (80, 160)), ("mini2", (200, 160)), ("mini1", (80, 210))), 1):
                    h.move(mini, destination)
                    row = h.frames(10)
                    self.assertFalse(row["needs_reset"], row["state"])
                    self.assertEqual(len(h.events), count, row["state"])
                    self.assert_follows(h, mini)
                    self.assertEqual(h.output.box.snapshot()[mini]["state"], "confirmed")
                self.assertEqual(len(h.output.offers), 9)
                self.assertEqual(len(h.output.applied), 9)
                self.assertEqual([row["mini"] for row in h.events], ["mini1", "mini2", "mini1"])

    def test_visible_digital_source_and_destination_disks_follow_two_complete_cycles(self):
        h = self.harness(artwork="disk", ack_delay=.25, render_delay=.5)
        for count, (mini, destination) in enumerate((("mini1", (80, 160)), ("mini2", (200, 160))), 1):
            h.move(mini, destination)
            row = h.frames(12)
            self.assertFalse(row["needs_reset"], row["state"])
            self.assertEqual(len(h.events), count, row["state"])
            self.assert_follows(h, mini)
        self.assertEqual(len(h.output.offers), 8)

    def test_another_mini_moves_while_first_ack_and_camera_redraw_are_pending(self):
        h = self.harness()
        h.output.ack_delay, h.output.render_delay = 2.5, 3.
        h.move("mini1", (80, 160))
        h.frames(5)
        self.assertEqual(len(h.events), 1)
        self.assertEqual(h.output.box.snapshot()["mini1"]["state"], "waiting_ack")
        h.move("mini2", (200, 160))
        h.frames(5)
        self.assertEqual([row["mini"] for row in h.events], ["mini1", "mini2"])
        self.assertEqual(h.output.box.snapshot()["mini1"]["state"], "waiting_ack")
        h.frames(32)
        self.assertEqual(len(h.events), 2)
        self.assert_follows(h, "mini1")
        self.assert_follows(h, "mini2")
        self.assertTrue(all(row["state"] == "confirmed" for row in h.output.box.snapshot().values()))
        self.assertGreater(len(h.output.sent), len(h.output.applied))
        self.assertEqual(len(h.output.applied), 8)  # Retries do not apply another physical move.

    def test_noop_ack_retry_and_duplicate_ack_do_not_create_additional_moves(self):
        h = self.harness()
        self.assertTrue(all(row["changed"] is False for row in h.output.applied))
        h.output.ack_delay = 1.5
        h.move("mini1", (80, 160))
        h.frames(28)
        self.assertEqual(len(h.events), 1)
        self.assertEqual(len(h.output.offers), 7)
        self.assertEqual(len(h.output.applied), 7)
        sent = [row for row in h.output.sent if row["miniId"] == "mini1"][1:]
        self.assertGreater(len(sent), 1)
        self.assertEqual(len({row["commandId"] for row in sent}), 1)
        self.assertTrue(any(not accepted for _payload, accepted in h.output.ack_results))

    def test_same_mini_can_move_again_before_its_earlier_ack_and_render(self):
        h = self.harness()
        h.output.ack_delay, h.output.render_delay = 2.5, 3.
        h.move("mini1", (80, 160))
        h.frames(5)
        old = h.output.box.snapshot()["mini1"]["command_id"]
        h.move("mini1", (80, 210))
        h.frames(5)
        self.assertEqual(len(h.events), 2)
        self.assertNotEqual(h.output.box.snapshot()["mini1"]["command_id"], old)
        h.frames(32)
        self.assertEqual(len(h.events), 2)
        self.assert_follows(h, "mini1")
        self.assertEqual(h.output.box.snapshot()["mini1"]["state"], "confirmed")
        self.assertTrue(any(payload["commandId"] == old and not accepted
                            for payload, accepted in h.output.ack_results))

    def test_assignment_confirmation_loss_gates_output_but_observes_current_physical_positions(self):
        h = self.harness()
        engine, generation = h.controller.engine, h.controller.binding_generation
        h.assignments.confirmed = {}
        h.frame()
        h.move("mini3", (320, 160))
        h.frames(8)
        self.assertIs(h.controller.engine, engine)
        self.assertEqual(h.controller.binding_generation, generation)
        self.assertEqual([row["mini"] for row in h.events], ["mini3"])
        self.assertEqual(len(h.output.offers), 6)
        self.assertTrue(h.output.observation_only())
        h.assignments.confirm()
        h.frames(8)
        self.assertTrue(h.bridge.aligned)
        self.assert_follows(h, "mini3")
        self.assertEqual(len(h.output.offers), 12)
        self.assertEqual(len(h.events), 1)

    def test_rejected_output_latches_delivery_but_keeps_observing_later_physical_moves(self):
        h = self.harness()
        engine, generation = h.controller.engine, h.controller.binding_generation
        h.output.fail_mini = "mini1"
        h.move("mini1", (80, 160))
        h.frames(8)
        self.assertIs(h.controller.engine, engine)
        self.assertIn("physical observations continue", h.bridge.message)
        h.move("mini2", (200, 160))
        h.frames(8)
        self.assertIs(h.controller.engine, engine)
        self.assertEqual(h.controller.binding_generation, generation)
        self.assertEqual([row["mini"] for row in h.events], ["mini1", "mini2"])
        self.assertEqual(len(h.output.applied), 6)
        self.assertEqual(h.output.box.snapshot(), {})
        self.assertTrue(h.output.observation_only())

    def test_forty_two_moves_across_six_minis_keep_feedback_and_command_history_bounded(self):
        h = self.harness(artwork="disk", ack_delay=.125, render_delay=.125)
        for index in range(42):
            mini = f"mini{index % 6 + 1}"
            x, y = POSITIONS[mini]
            alternate = (x, y + (50 if y < 180 else -50))
            destination = alternate if h.physical[mini] == POSITIONS[mini] else POSITIONS[mini]
            h.move(mini, destination)
            row = h.frames(6)
            self.assertFalse(row["needs_reset"], (index, row["state"]))
            self.assertEqual(len(h.events), index + 1, (index, row["state"]))
            self.assert_follows(h, mini)
        self.assertEqual(len(h.output.offers), 48)
        self.assertEqual(len(h.output.applied), 48)
        for mini, actual in h.physical.items():
            self.assertLessEqual(math.dist(h.controller.engine.mover.positions[mini], actual), 8)
            self.assert_follows(h, mini)
        self.assertLessEqual(len(h.controller.engine._known_display_commands), 32)
        self.assertLessEqual(len(h.bridge._commands), 32)
        self.assertLessEqual(len(h.output.visual["changes"]), 32)

    def test_glare_only_frames_and_unknown_change_never_emit_physical_moves(self):
        h = self.harness()
        for glare in (1, 2, 3, 4, 3, 2, 1, 0):
            h.frame(glare=glare)
        h.frames(10)
        self.assertEqual(h.events, [])
        self.assertEqual(len(h.output.offers), 6)
        h.output.notice("manualFogChange")
        row = h.frames(4)
        self.assertTrue(row["needs_reset"])
        self.assertEqual(h.events, [])
        self.assertEqual(len(h.output.offers), 6)
        self.assertTrue(h.output.observation_only())

    def test_reflection_can_hide_stationary_bases_then_restore_without_inventing_moves(self):
        h = self.harness()
        h.frames(6, glare=10)
        row = h.frames(10)
        self.assertFalse(row["needs_reset"], row["state"])
        self.assertEqual(h.events, [])
        self.assertEqual(len(h.output.offers), 6)
        h.move("mini1", (80, 160))
        h.frames(10)
        self.assertEqual([row["mini"] for row in h.events], ["mini1"])
        self.assert_follows(h, "mini1")


if __name__ == "__main__":
    unittest.main()
