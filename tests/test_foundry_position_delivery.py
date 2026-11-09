"""Measured base-center delivery through the normal acknowledged Foundry outbox."""
import asyncio
from collections import deque
import math
import unittest
from unittest.mock import patch

import foundryoutput as fo
from foundry_delivery import MoveOutbox
from tests import test_foundry_delivery_integration as integration


def token_state(*, width=1, height=1):
    return {"sceneId": "test-scene", "ready": True,
            "scene": {"gridSize": 50},
            "tokens": [{"tokenId": "piece-token", "tokenName": "Player", "width": width, "height": height}]}


class PositionOutboxTests(unittest.TestCase):
    def setUp(self):
        self.box = MoveOutbox()
        self.context = ("scene", 1000, 800, 50, 0, 0, 1)
        self.box.set_context(self.context)

    def test_position_offer_id_is_available_before_send_and_stays_for_ack(self):
        self.assertTrue(self.box.offer_position("mini1", (173.5, 287.25), self.context, generation="binding1"))
        offered = self.box.snapshot()["mini1"]
        command = self.box.prepare_position("mini1", (173.5, 287.25), "token", (124, 262), 0, generation="binding1")
        self.assertEqual(command["commandId"], offered["command_id"])
        self.assertEqual(command["center"], [173.5, 287.25])
        self.assertEqual(command["bindingGeneration"], "binding1")
        self.assertIsNone(command["cell"])
        self.assertEqual(command["targetKind"], "position")
        self.assertTrue(self.box.reply({**command, "type": "tokenMoveApplied", "changed": False}, .1))
        state = self.box.snapshot()["mini1"]
        self.assertEqual(state["state"], "confirmed")
        self.assertIs(state["applied_changed"], False)

    def test_same_center_in_new_binding_generation_cannot_accept_prior_ack(self):
        self.box.offer_position("mini1", (100, 100), self.context, generation="old")
        old = self.box.prepare_position("mini1", (100, 100), "token", (75, 75), 0, generation="old")
        self.box.offer_position("mini1", (100, 100), self.context, generation="new")
        new = self.box.prepare_position("mini1", (100, 100), "token", (75, 75), .1, generation="new")
        self.assertNotEqual(old["commandId"], new["commandId"])
        self.assertFalse(self.box.reply({**old, "type": "tokenMoveApplied"}, .2))
        self.assertIsNone(self.box.prepare_position("mini1", (100, 100), "token", (75, 75), .2, generation="old"))

    def test_pinned_position_cannot_rebind_or_retry_after_assignment_changes(self):
        self.box.offer_position("mini1", (100, 100), self.context, generation="one", expected_token_id="token")
        offered = self.box.snapshot()["mini1"]["command_id"]
        self.assertIsNone(self.box.prepare_position("mini1", (100, 100), "replacement", (75, 75), 0, generation="one"))
        state = self.box.snapshot()["mini1"]
        self.assertTrue(state["binding_invalid"])
        self.assertEqual(state["command_id"], offered)
        self.box.retry(reconnect=True)
        self.assertIsNone(self.box.prepare_position("mini1", (100, 100), "token", (75, 75), .1, generation="one"))
        self.assertEqual(self.box.snapshot()["mini1"]["state"], "failed")
        self.box.offer_position("mini1", (100, 100), self.context, generation="two", expected_token_id="replacement")
        self.assertIsNotNone(self.box.prepare_position("mini1", (100, 100), "replacement", (75, 75), .2, generation="two"))

    def test_position_and_cell_destinations_do_not_alias_or_reuse_an_old_ack(self):
        self.box.offer("mini1", "B2", self.context)
        cell = self.box.prepare("mini1", "B2", "token", (50, 50), 0)
        self.box.offer_position("mini1", (75, 75), self.context)
        point = self.box.prepare_position("mini1", (75, 75), "token", (50, 50), .1)
        self.assertFalse(self.box.reply({**cell, "type": "tokenMoveApplied"}, .2))
        self.assertNotEqual(cell["commandId"], point["commandId"])
        self.box.set_context(("another-scene", *self.context[1:]))
        self.assertEqual(self.box.snapshot(), {})

    def test_invalid_centers_and_generations_never_enter_outbox(self):
        for center in ((float("nan"), 1), (1, math.inf), (True, 1), (1,), "1,2"):
            with self.subTest(center=center), self.assertRaises(ValueError):
                self.box.offer_position("mini1", center, self.context)
        with self.assertRaises(ValueError):
            self.box.offer_position("mini1", (1, 1), self.context, generation=True)
        self.assertEqual(self.box.snapshot(), {})

    def test_expected_scene_rejects_a_new_context_with_a_reused_token_id(self):
        next_context = ("new-scene", *self.context[1:])
        self.box.set_context(next_context)
        self.assertFalse(self.box.offer_position("mini1", (100, 100), next_context,
            generation="one", expected_token_id="same-token", expected_scene_id="scene"))
        self.assertEqual(self.box.snapshot(), {})
        self.assertTrue(self.box.offer_position("mini1", (100, 100), next_context,
            generation="two", expected_token_id="same-token", expected_scene_id="new-scene"))


class PositionDeliveryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = integration.DeliveryIntegrationTest.asyncSetUp
    asyncTearDown = integration.DeliveryIntegrationTest.asyncTearDown
    until = integration.DeliveryIntegrationTest.until
    connect = integration.DeliveryIntegrationTest.connect
    moves = integration.DeliveryIntegrationTest.moves

    async def position_connection(self, *, width=1, height=1):
        self.patches.extend([patch.object(fo, "_observation_only", False),
                             patch.object(fo, "_view_transform", None),
                             patch.object(fo, "_vtt_state", {})])
        for p in self.patches[-3:]:
            p.start()
        socket, _handler = await self.connect()
        fo.MINI_TO_TOKEN["mini1"] = "piece-token"
        fo._vtt_state["data"] = token_state(width=width, height=height)
        return socket

    async def test_current_token_size_converts_base_center_without_grid_snapping(self):
        socket = await self.position_connection(width=2, height=1)
        self.assertTrue(fo.queue_position_move("mini1", 173.5, 287.25, generation="one"))
        offered = fo._delivery.snapshot()["mini1"]["command_id"]
        await self.until(lambda: self.moves(socket))
        command = self.moves(socket)[0]
        self.assertEqual((command["x"], command["y"]), (124, 262))
        self.assertEqual(command["commandId"], offered)
        self.assertEqual(command["center"], [173.5, 287.25])
        self.assertEqual(command["source"], "base-tracker")
        await socket.incoming.put({**command, "type": "tokenMoveApplied", "changed": True})
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "confirmed")
        self.assertIs(fo.get_delivery_status()["moves"]["mini1"]["applied_changed"], True)

    async def test_missing_current_geometry_waits_then_sends_without_another_detection(self):
        socket = await self.position_connection()
        fo._vtt_state["data"] = {}
        fo.queue_position_move("mini1", 200, 300)
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "geometry")
        self.assertEqual(self.moves(socket), [])
        self.assertIn("token geometry", fo.get_delivery_status()["message"])
        fo._vtt_state["data"] = token_state()
        await self.until(lambda: self.moves(socket))
        self.assertEqual((self.moves(socket)[0]["x"], self.moves(socket)[0]["y"]), (175, 275))

    async def test_position_ids_never_use_legacy_prefix_or_default_token_fallback(self):
        socket = await self.position_connection()
        fo.MINI_TO_TOKEN["red"] = "piece-token"
        fo.queue_position_move("red_suffix", 200, 300)
        await self.until(lambda: any(m["type"] == "assignMini" and m["miniId"] == "red_suffix" for m in socket.sent))
        self.assertEqual(self.moves(socket), [])

    async def test_stale_grid_metadata_and_overflowing_token_size_hold_output(self):
        socket = await self.position_connection()
        fo._vtt_state["data"]["scene"]["gridSize"] = 25
        fo.queue_position_move("mini1", 200, 300)
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "geometry")
        self.assertEqual(self.moves(socket), [])
        fo._vtt_state["data"] = token_state(width=1e308)
        await asyncio.sleep(.12)
        self.assertEqual(fo._delivery.snapshot()["mini1"]["state"], "geometry")
        self.assertEqual(self.moves(socket), [])
        fo._vtt_state["data"] = token_state()
        await self.until(lambda: self.moves(socket))

    async def test_pinned_assignment_change_cannot_dispatch_retained_confirmed_intent(self):
        socket = await self.position_connection()
        self.assertFalse(fo.queue_position_move("mini1", 200, 300, expected_token_id="replacement"))
        self.assertTrue(fo.queue_position_move("mini1", 200, 300, generation="one", expected_token_id="piece-token"))
        offered = fo._delivery.snapshot()["mini1"]["command_id"]
        await self.until(lambda: self.moves(socket))
        original = self.moves(socket)[0]
        self.assertEqual(original["commandId"], offered)
        await socket.incoming.put({**original, "type": "tokenMoveApplied", "changed": True})
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "confirmed")
        fo._vtt_state["data"]["tokens"].append({"tokenId": "replacement", "width": 1, "height": 1})
        fo.MINI_TO_TOKEN["mini1"] = "replacement"
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["binding_invalid"])
        self.assertEqual(len(self.moves(socket)), 1)
        fo.MINI_TO_TOKEN["mini1"] = "piece-token"
        fo.retry_failed_moves()
        await asyncio.sleep(.12)
        self.assertEqual(len(self.moves(socket)), 1)
        self.assertEqual(fo._delivery.snapshot()["mini1"]["state"], "failed")
        fo.MINI_TO_TOKEN["mini1"] = "replacement"
        self.assertTrue(fo.queue_position_move("mini1", 200, 300, generation="two", expected_token_id="replacement"))
        await self.until(lambda: len(self.moves(socket)) == 2)
        self.assertEqual(self.moves(socket)[-1]["tokenId"], "replacement")

    async def test_assignment_change_before_first_dispatch_revokes_pinned_offer(self):
        socket = await self.position_connection()
        self.assertTrue(fo.queue_position_move("mini1", 200, 300, expected_token_id="piece-token"))
        fo.MINI_TO_TOKEN["mini1"] = "replacement"
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["binding_invalid"])
        self.assertEqual(self.moves(socket), [])

    async def test_transient_assignment_change_revokes_pin_before_it_can_be_restored(self):
        socket = await self.position_connection()
        fo.queue_position_move("mini1", 200, 300, expected_token_id="piece-token")
        await self.until(lambda: self.moves(socket))
        await socket.incoming.put({"type": "assignMiniResult", "sceneId": "test-scene", "miniId": "mini1", "tokenId": "replacement"})
        await socket.incoming.put({"type": "assignMiniResult", "sceneId": "test-scene", "miniId": "mini1", "tokenId": "piece-token"})
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["binding_invalid"])
        await self.until(lambda: fo.MINI_TO_TOKEN["mini1"] == "piece-token")
        fo.retry_failed_moves()
        await asyncio.sleep(.12)
        self.assertEqual(len(self.moves(socket)), 1)

    async def test_confirmed_position_does_not_resend_when_token_footprint_changes(self):
        socket = await self.position_connection()
        fo.queue_position_move("mini1", 200, 300, generation="one", expected_token_id="piece-token")
        await self.until(lambda: self.moves(socket))
        original = self.moves(socket)[0]
        await socket.incoming.put({**original, "type": "tokenMoveApplied", "changed": True})
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "confirmed")
        fo._vtt_state["data"] = token_state(width=2)
        await asyncio.sleep(.22)
        self.assertEqual(len(self.moves(socket)), 1)
        self.assertEqual(fo._delivery.snapshot()["mini1"]["command_id"], original["commandId"])
        fo.queue_position_move("mini1", 250, 300, generation="one", expected_token_id="piece-token")
        await self.until(lambda: len(self.moves(socket)) == 2)
        self.assertEqual((self.moves(socket)[-1]["x"], self.moves(socket)[-1]["y"]), (200, 275))

    async def test_expected_scene_refuses_stale_pose_even_when_token_binding_is_identical(self):
        socket = await self.position_connection()
        self.assertFalse(fo.queue_position_move("mini1", 200, 300,
            expected_scene_id="old-scene", expected_token_id="piece-token"))
        self.assertEqual(fo._delivery.snapshot(), {})
        self.assertTrue(fo.queue_position_move("mini1", 200, 300,
            expected_scene_id="test-scene", expected_token_id="piece-token"))
        await self.until(lambda: self.moves(socket))
        self.assertEqual(self.moves(socket)[0]["sceneId"], "test-scene")

    async def test_scene_change_between_initial_check_and_outbox_offer_is_rejected(self):
        socket = await self.position_connection()
        original_context = fo._scene_context
        def race_context():
            context = ("new-scene", *original_context()[1:])
            fo._delivery.set_context(context)
            return context
        with patch.object(fo, "_scene_context", side_effect=race_context):
            self.assertFalse(fo.queue_position_move("mini1", 200, 300,
                expected_scene_id="test-scene", expected_token_id="piece-token"))
        self.assertEqual(fo._delivery.snapshot(), {})
        await asyncio.sleep(.12)
        self.assertEqual(self.moves(socket), [])

    async def test_observer_guard_blocks_position_output_and_clear_drops_stale_intent(self):
        socket = await self.position_connection()
        fo.set_observation_only(True)
        self.assertFalse(fo.queue_position_move("mini1", 200, 300))
        self.assertEqual(fo._delivery.snapshot(), {})
        fo.set_observation_only(False)
        self.assertTrue(fo.queue_position_move("mini1", 200, 300))
        fo.clear_pending_moves()
        await asyncio.sleep(.12)
        self.assertEqual(self.moves(socket), [])

    async def test_outside_scene_and_token_footprint_never_send(self):
        socket = await self.position_connection()
        with self.assertRaises(ValueError):
            fo.queue_position_move("mini1", -1, 300)
        fo.queue_position_move("mini1", 5, 300)
        await self.until(lambda: fo._delivery.snapshot()["mini1"]["state"] == "failed")
        self.assertIn("outside", fo._delivery.snapshot()["mini1"]["error"])
        self.assertEqual(self.moves(socket), [])

    async def test_canvas_padding_is_included_in_absolute_scene_bounds(self):
        socket = await self.position_connection()
        fo._view_transform = {"sceneId": "test-scene", "gridOriginX": 100, "gridOriginY": 50}
        fo.queue_position_move("mini1", 125, 75)
        await self.until(lambda: self.moves(socket))
        self.assertEqual((self.moves(socket)[0]["x"], self.moves(socket)[0]["y"]), (100, 50))
        with self.assertRaises(ValueError):
            fo.queue_position_move("mini1", 50, 75)


class SceneVisualHistoryTests(unittest.TestCase):
    def test_snapshot_preserves_unknown_changes_between_correlated_own_moves(self):
        with patch.object(fo, "_scene_visual_revision", 0), patch.object(fo, "_scene_visual_reason", "initial"), \
             patch.object(fo, "_scene_visual_changes", deque(maxlen=32)), patch.object(fo, "SCENE_ID", "scene"):
            fo.mark_scene_visual_changed("tokenUpdated", {"sceneId": "scene", "tokenIds": ["manual"]})
            fo.mark_scene_visual_changed("sarweenTokenMove", {"sceneId": "scene", "reasons": ["sarweenTokenMove"],
                "tokenIds": ["piece"], "commandIds": ["confirmed-command"]})
            value = fo.get_scene_visual_change()
            self.assertEqual(value["revision"], 2)
            self.assertEqual(value["commandIds"], ["confirmed-command"])
            self.assertEqual([row["reason"] for row in value["changes"]], ["tokenUpdated", "sarweenTokenMove"])
            value["changes"][0]["reasons"].append("mutated")
            self.assertNotIn("mutated", fo.get_scene_visual_change()["changes"][0]["reasons"])
            for _ in range(40):
                fo.mark_scene_visual_changed("unknown")
            self.assertEqual(len(fo.get_scene_visual_change()["changes"]), 32)
