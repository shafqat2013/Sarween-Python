import unittest

from foundry_delivery import MoveOutbox


class MoveOutboxTest(unittest.TestCase):
    def setUp(self):
        self.box = MoveOutbox(ack_timeout=1, max_attempts=3)
        self.context = ("scene", 1000, 800, 50, 0, 0, 1)
        self.box.set_context(self.context)
        self.box.offer("red", "B2", self.context)

    def send(self, now=0, cell="B2", token="token", xy=(50, 50)):
        return self.box.prepare("red", cell, token, xy, now)

    def ack(self, command, **values):
        return {**command, "type": "tokenMoveApplied", **values}

    def test_move_remains_until_exact_acknowledgement(self):
        command = self.send()
        self.assertEqual(self.box.snapshot()["red"]["state"], "waiting_ack")
        self.assertFalse(self.box.reply(self.ack(command, commandId="stale"), 0.1))
        self.assertFalse(self.box.reply(self.ack(command, sceneId="other"), 0.1))
        self.assertTrue(self.box.reply(self.ack(command), 0.1))
        self.assertEqual(self.box.snapshot()["red"]["state"], "confirmed")
        self.assertIsNone(self.send(10))

    def test_latest_destination_replaces_pending_path_and_ignores_old_ack(self):
        old = self.send()
        self.box.offer("red", "C3", self.context)
        self.box.offer("red", "D4", self.context)
        self.assertIsNone(self.send())
        new = self.send(cell="D4", xy=(150, 150))
        self.assertNotEqual(new["commandId"], old["commandId"])
        self.assertFalse(self.box.reply(self.ack(old), 0.1))
        self.assertEqual(len(self.box.snapshot()), 1)

    def test_assignment_wait_retains_position_without_prompt_spam(self):
        self.assertTrue(self.box.waiting_for_assignment("red"))
        self.assertFalse(self.box.waiting_for_assignment("red"))
        self.assertEqual(self.send()["cell"], "B2")

    def test_movement_while_unassigned_does_not_repeat_prompt(self):
        self.assertTrue(self.box.waiting_for_assignment("red"))
        self.box.offer("red", "C3", self.context)
        self.assertFalse(self.box.waiting_for_assignment("red"))
        self.send(cell="C3", token="assigned")
        self.assertTrue(self.box.waiting_for_assignment("red"), "A deleted assignment must prompt again")

    def test_timeout_retries_have_same_id_and_stop(self):
        first = self.send()
        self.assertIsNone(self.send(0.5))
        self.assertEqual(self.send(1)["commandId"], first["commandId"])
        self.assertEqual(self.send(2)["commandId"], first["commandId"])
        self.assertIsNone(self.send(3))
        self.assertIsNone(self.send(100))
        self.assertEqual(self.box.snapshot()["red"]["state"], "failed")
        self.box.retry()
        self.assertIsNotNone(self.send(101))

    def test_rejection_retries_and_wrong_applied_position_is_not_success(self):
        command = self.send()
        self.box.reply(self.ack(command, x=0), 0.1)
        self.assertEqual(self.box.snapshot()["red"]["state"], "retry")
        self.assertIsNone(self.send(0.2))
        command = self.send(1)
        self.box.reply({**command, "type": "tokenMoveError", "error": "Rejected"}, 1.1)
        command = self.send(3)
        self.box.reply({**command, "type": "tokenMoveError", "error": "Rejected"}, 3.1)
        self.assertEqual(self.box.snapshot()["red"]["state"], "failed")

    def test_reconnect_reconciles_even_previously_confirmed_position(self):
        command = self.send()
        self.box.reply(self.ack(command), 0.1)
        self.box.retry(reconnect=True)
        self.assertNotEqual(self.send(10)["commandId"], command["commandId"])
        self.assertFalse(self.box.reply(self.ack(command), 10.1))

    def test_rebinding_invalidates_old_token_ack(self):
        old = self.send()
        new = self.send(token="replacement")
        self.assertNotEqual(old["commandId"], new["commandId"])
        self.assertFalse(self.box.reply(self.ack(old), 0.1))

    def test_scene_or_geometry_change_discards_prior_intents(self):
        self.box.set_context(("other", *self.context[1:]))
        self.assertFalse(self.box.offer("red", "B2", self.context))
        self.assertEqual(self.box.snapshot(), {})
        self.box.set_context(self.context)
        self.assertEqual(self.box.snapshot(), {})

    def test_identical_scene_info_keeps_intents(self):
        self.box.set_context(self.context)
        self.assertEqual(self.send()["cell"], "B2")

    def test_capture_clear_drops_old_intents(self):
        self.box.clear()
        self.assertIsNone(self.send())

    def test_viewport_source_survives_main_callback_delivery_and_retry(self):
        from unittest.mock import patch
        import foundryoutput as fo
        from main import on_mini_moved
        with patch.object(fo, "_delivery", self.box), \
             patch.object(fo, "_scene_context", return_value=self.context), \
             patch.object(fo, "_grid_to_pixels", return_value=(100, 100)), \
             patch.object(fo, "tracking_output_paused", return_value=False):
            on_mini_moved("red", "C3", source="viewportTransform")
        command = self.send(cell="C3", xy=(100, 100))
        self.assertEqual(command["source"], "viewportTransform")
        self.assertEqual(self.send(1, cell="C3", xy=(100, 100))["source"], "viewportTransform")


if __name__ == "__main__":
    unittest.main()
