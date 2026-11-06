"""Token creation uses explicit requests and stable retry IDs, without a real world."""
import asyncio
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import foundryoutput as fo
from token_creation import TokenCreationController, blocked_reason, parse_cell, validate_token_request
from token_creation_controls import TokenCreationControls
import tests.test_vtt_bridge as bridge_fixture


def ready(**changes):
    return {"connected": True, "supported": True, "blocked": False, "pending": False,
            "sceneId": "scene-a", "world": {"id": "testing", "name": "Testing"}, "paused": False,
            "user": {"id": "gm", "name": "Gamemaster", "isGM": True},
            "capabilities": ["token.create", "lifecycle.setPaused"], "capture": {},
            "actors": [{"actorId": "goblin", "actorName": "Goblin", "type": "npc"}], **changes}


def draft(**changes):
    return {"actorId": "goblin", "name": "Scout", "placement": "cell", "row": 3, "col": 2, **changes}


class TokenCreationTests(unittest.TestCase):
    def controller(self):
        request = Mock(side_effect=["request-1", "request-2", "request-3"])
        controller = TokenCreationController(request, new_id=Mock(side_effect=["creation_00000001", "creation_00000002"]))
        return controller, request

    def test_cells_are_explicit_zero_based_and_invalid_values_are_rejected(self):
        self.assertEqual(parse_cell(" A1 "), (0, 0))
        self.assertEqual(parse_cell("AN14"), (13, 39))
        for value in ("A0", "0A", "A-1", "A1 trailing", "ZZZZ999999", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cell(value)
        valid = {**draft(), "creationId": "creation_00000001"}
        self.assertEqual(validate_token_request(valid), valid)
        for changes in ({"row": True}, {"col": -1}, {"name": "bad\nname"}, {"creationId": "short"},
                        {"actorId": ""}, {"extra": True}, {"placement": "viewport"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_token_request({**valid, **changes})

    def test_polling_never_sends_and_retry_after_timeout_reuses_creation_id(self):
        controller, request = self.controller()
        for _ in range(4):
            controller.refresh(ready())
        request.assert_not_called()
        self.assertTrue(controller.create(draft(), ready()))
        first = deepcopy(request.call_args.args[1])
        self.assertFalse(controller.create(draft(), ready(pending=True)))
        controller.refresh(ready(lastResult={"requestId": "request-1", "success": False,
                                           "message": "No acknowledgement"}))
        self.assertTrue(controller.create(draft(), ready()))
        self.assertEqual(request.call_args.args[1]["creationId"], first["creationId"])
        self.assertEqual(request.call_count, 2)

    def test_disconnect_preserves_uncertain_operation_and_reconnect_retry_id(self):
        controller, request = self.controller()
        controller.create(draft(), ready())
        first_id = request.call_args.args[1]["creationId"]
        controller.refresh({"connected": False, "supported": False})
        self.assertIsNone(controller.pending_request)
        self.assertIn("not confirmed", controller.message)
        self.assertTrue(controller.create(draft(), ready()))
        self.assertEqual(request.call_args.args[1]["creationId"], first_id)

    def test_success_allows_new_explicit_creation_but_never_creates_automatically(self):
        controller, request = self.controller()
        controller.create(draft(), ready())
        result = {"requestId": "request-1", "success": True, "message": "Created Scout",
                  "result": {"tokenId": "created", "creationId": "creation_00000001", "name": "Scout"}}
        controller.refresh(ready(lastResult=result))
        self.assertEqual(request.call_count, 1)
        self.assertEqual(controller.last_created["tokenId"], "created")
        self.assertTrue(controller.create(draft(), ready(lastResult=result)))
        self.assertEqual(request.call_args.args[1]["creationId"], "creation_00000002")

    def test_changed_world_or_scene_cannot_reuse_an_old_creation_scope(self):
        controller, request = self.controller()
        controller.create(draft(), ready())
        changed = ready(sceneId="scene-b")
        controller.refresh(changed)
        self.assertIsNone(controller.pending_request)
        self.assertTrue(controller.create(draft(), changed))
        self.assertEqual(request.call_args.args[1]["creationId"], "creation_00000002")

    def test_actor_capture_observer_capability_and_pending_guards_never_request(self):
        for changes in ({"actors": []}, {"outputPaused": True}, {"capture": {"phase": "Saving"}},
                        {"connected": False}, {"supported": False}, {"blocked": True},
                        {"pending": True}, {"capabilities": []}, {"sceneId": None}):
            controller, request = self.controller()
            with self.subTest(changes=changes):
                self.assertFalse(controller.create(draft(), ready(**changes)))
                request.assert_not_called()


class Value:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class TokenCreationControlTests(unittest.TestCase):
    def controls(self):
        ui = TokenCreationControls.__new__(TokenCreationControls)
        ui._require_access, ui._read_status = Mock(), Mock(return_value=ready())
        ui._can_create = Mock(return_value=True)
        ui._enabled = True
        ui._status, ui._actors = {}, {}
        ui._selected_actor_id = None
        ui.controller = TokenCreationController(Mock(return_value="request-1"), new_id=lambda: "creation_00000001")
        ui.actor, ui.name, ui.placement, ui.cell, ui.message = Value(), Value("Scout"), Value("Grid cell"), Value("C4"), Value()
        for name in ("actor_combo", "name_entry", "placement_combo", "cell_entry", "create_button"):
            setattr(ui, name, Mock())
        ui.refresh(ready())
        ui.actor.set("Goblin (goblin)")
        ui._update_enabled()
        return ui

    def test_actor_name_and_explicit_cell_are_sent_only_after_create(self):
        ui = self.controls()
        ui.controller.request.assert_not_called()
        self.assertTrue(ui.create())
        ui.controller.request.assert_called_once_with("token.create", {**draft(), "creationId": "creation_00000001"})
        self.assertEqual(ui.create_button.configure.call_args.kwargs["state"], "disabled")
        self.assertIn("Creating", ui.message.get())

    def test_auth_or_display_busy_blocks_creation_even_after_button_was_enabled(self):
        ui = self.controls()
        ui._require_access.side_effect = PermissionError("Sign in")
        self.assertFalse(ui.create())
        ui.controller.request.assert_not_called()
        ui._require_access.side_effect = None
        ui._can_create.return_value = False
        self.assertFalse(ui.create())
        ui.controller.request.assert_not_called()

    def test_actor_choices_preserve_id_across_rename_and_disconnect_but_not_deleted_actor(self):
        ui = self.controls()
        ui.refresh(ready(actors=[{"actorId": "goblin", "actorName": "Renamed"}]))
        self.assertEqual(ui.actor.get(), "Renamed (goblin)")
        ui.refresh({"connected": False})
        self.assertEqual(ui.actor.get(), "")
        ui.refresh(ready())
        self.assertEqual(ui.actor.get(), "Goblin (goblin)")
        ui.refresh(ready(actors=[]))
        self.assertEqual(ui.actor.get(), "")
        self.assertIsNone(ui._selected_actor_id)
        self.assertEqual(ui.create_button.configure.call_args.kwargs["state"], "disabled")


class TokenCreationBridgeTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only in-memory fixtures, not another test class's test methods.
    asyncSetUp = bridge_fixture.VTTBridgeTests.asyncSetUp
    asyncTearDown = bridge_fixture.VTTBridgeTests.asyncTearDown
    until = bridge_fixture.VTTBridgeTests.until
    state = bridge_fixture.VTTBridgeTests.state
    connect = bridge_fixture.VTTBridgeTests.connect
    commands = bridge_fixture.VTTBridgeTests.commands
    reply = bridge_fixture.VTTBridgeTests.reply

    async def test_new_values_are_validated_before_queue_and_pause_is_absolute_bool(self):
        socket, _ = await self.connect()
        for action, payload in (("lifecycle.setPaused", {}), ("lifecycle.setPaused", {"paused": 0}),
                                ("lifecycle.setPaused", {"paused": False, "extra": True}),
                                ("token.create", {**draft(), "creationId": "short"})):
            with self.subTest(action=action, payload=payload), self.assertRaises(ValueError):
                fo.request_vtt_command(action, payload)
        self.assertEqual(self.commands(socket), [])

    async def test_observer_and_capture_pause_block_manual_mutations_with_clear_message(self):
        socket, _ = await self.connect()
        for flag in ("_observation_only", "_tracking_output_paused"):
            with patch.object(fo, flag, True):
                self.assertTrue(fo.get_vtt_status()["outputPaused"])
                for action, payload in (("lifecycle.setPaused", {"paused": False}),
                                        ("token.create", {**draft(), "creationId": "creation_00000001"})):
                    with self.assertRaisesRegex(ValueError, "Stop the mini observer"):
                        fo.request_vtt_command(action, payload)
        self.assertEqual(self.commands(socket), [])

    async def test_observer_enabled_before_queued_callback_discards_manual_mutation(self):
        socket, _ = await self.connect()
        with patch.object(fo, "_observation_only", False):
            self.assertTrue(fo.request_vtt_command("lifecycle.setPaused", {"paused": False}))
            fo._observation_only = True
            await self.until(lambda: not fo.get_vtt_status()["pending"])
            self.assertIn("blocks", fo.get_vtt_status()["lastResult"]["message"])
            self.assertEqual(self.commands(socket), [])

    async def test_remote_capture_state_blocks_manual_mutations_before_queue(self):
        socket, _ = await self.connect()
        fo._receive_vtt(self.state(capture={"phase": "Saving", "recording": False}))
        with patch.object(fo, "_observation_only", False), patch.object(fo, "_tracking_output_paused", False):
            with self.assertRaisesRegex(ValueError, "Finish recording"):
                fo.request_vtt_command("lifecycle.setPaused", {"paused": False})
            with self.assertRaisesRegex(ValueError, "Finish recording"):
                fo.request_vtt_command("token.create", {**draft(), "creationId": "creation_00000001"})
        self.assertEqual(self.commands(socket), [])

    async def test_observer_enabled_at_actual_send_discards_already_queued_mutation(self):
        socket, _ = await self.connect()

        class SwitchingQueue(asyncio.Queue):
            async def get(self):
                message = await super().get()
                fo._observation_only = True
                return message

        messages = SwitchingQueue()
        message = {"type": "vttCommand", "requestId": "queued-pause", "sceneId": "scene-a",
                   "action": "lifecycle.setPaused", "values": {"paused": False}}
        messages.put_nowait(message)
        with patch.object(fo, "_observation_only", False), patch.object(fo, "_ctrl_queue", messages):
            fo._vtt_state["pending"] = {"requestId": "queued-pause", "sceneId": "scene-a",
                "action": "lifecycle.setPaused", "deadline": fo.time.monotonic() + 10}
            sender = asyncio.create_task(fo.ctrl_loop(socket))
            try:
                await self.until(lambda: not fo.get_vtt_status()["pending"])
                self.assertIn("blocks", fo.get_vtt_status()["lastResult"]["message"])
                self.assertEqual(self.commands(socket), [])
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)

    async def test_creation_result_and_identity_are_retained_as_detached_state(self):
        socket, _ = await self.connect()
        request_id = fo.request_vtt_command("token.create", {**draft(), "creationId": "creation_00000001"}, return_request_id=True)
        await self.until(lambda: self.commands(socket))
        state = self.state(**{key: value for key, value in ready().items() if key in {"world", "user", "paused", "actors"}})
        result = {"tokenId": "created", "creationId": "creation_00000001", "name": "Scout"}
        fo._receive_vtt(self.reply(request_id, state=state, result=result))
        status = fo.get_vtt_status()
        self.assertEqual(status["world"]["id"], "testing")
        self.assertEqual(status["user"]["name"], "Gamemaster")
        self.assertFalse(status["paused"])
        self.assertEqual(status["lastResult"]["result"], result)
        status["lastResult"]["result"]["tokenId"] = "mutated"
        self.assertEqual(fo.get_vtt_status()["lastResult"]["result"]["tokenId"], "created")

    async def test_invalid_actor_identity_or_pause_state_cannot_replace_valid_state(self):
        await self.connect()
        for changes in ({"actors": ["not-an-actor"]}, {"paused": 1}, {"world": "wrong"},
                        {"user": {"id": "gm", "name": "GM", "isGM": "yes"}}):
            fo._receive_vtt(self.state(**changes))
            status = fo.get_vtt_status()
            self.assertNotIn("actors", status)
            self.assertNotIn("paused", status)
