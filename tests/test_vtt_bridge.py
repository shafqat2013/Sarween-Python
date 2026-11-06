"""Scene-scoped desktop VTT commands over an in-memory Foundry connection."""

import asyncio
import json
import unittest
from unittest.mock import patch

import foundryoutput as fo
from foundry_delivery import MoveOutbox


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return json.dumps(await self.incoming.get())


class VTTBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        replacements = {
            "_delivery": MoveOutbox(), "_active_socket": None, "_connection_ready": False,
            "_protocol_ready": False, "_connection_error": "", "_loop": asyncio.get_running_loop(),
            "_ctrl_queue": None, "_assign_queue": None, "_capture_session": None,
            "_tracking_output_paused": False, "_physical_grid_state": {}, "_vtt_state": {},
            "SCENE_ID": "scene-a", "SCENE_W": 1000, "SCENE_H": 800, "GRID_PX": 50,
            "GRID_TYPE": 1, "SHIFT_X": 0, "SHIFT_Y": 0, "_grid_cols": 20, "_grid_rows": 16,
            "MINI_TO_TOKEN": {}, "SCENE_TOKEN_NAMES": {},
        }
        self.patches = [patch.object(fo, name, value) for name, value in replacements.items()]
        for item in self.patches:
            item.start()
        self.handlers = []

    async def asyncTearDown(self):
        for handler in self.handlers:
            handler.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)
        for item in reversed(self.patches):
            item.stop()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.002)
        await asyncio.wait_for(wait(), 2)

    def state(self, **overrides):
        return {"type": "vttState", "supported": True, "ready": True, "sceneId": "scene-a",
                "capabilities": sorted(fo._vtt_actions), "settings": {"viewportMarkersEnabled": True},
                "scene": {"id": "scene-a", "name": "Test map"}, "tokens": [{"tokenId": "red", "tokenName": "Red"}],
                "minis": [{"miniId": "red10", "route": "A", "tokenId": "red"}],
                "capture": {"state": "idle", "recording": False}, "movement": None,
                "assignment": None, **overrides}

    async def connect(self, supported=True):
        socket = Socket()
        task = asyncio.create_task(fo.handler(socket))
        self.handlers.append(task)
        await socket.incoming.put({"type": "hello", "protocolVersion": 2,
                                   "physicalGridControls": True, "vttControls": supported})
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "scene-a", "width": 1000,
                                   "height": 800, "gridSize": 50, "gridType": 1})
        await socket.incoming.put({"type": "physicalGridState", "supported": True, "ready": True,
            "sceneId": "scene-a", "settings": {"targetGridInches": 1.0, "displayDiagonalInches": 54.0,
                                               "physicalGridCalibration": 1.0}})
        await socket.incoming.put(self.state())
        await self.until(lambda: fo._connection_ready)
        if supported:
            await self.until(lambda: not fo.get_vtt_status()["blocked"])
        return socket, task

    def commands(self, socket):
        return [msg for msg in socket.sent if msg["type"] == "vttCommand"]

    def reply(self, request_id, **overrides):
        return {"type": "vttResult", "requestId": request_id, "success": True,
                "message": "Applied", "state": self.state(), **overrides}

    async def test_command_copies_draft_and_confirms_result(self):
        socket, _ = await self.connect()
        draft = {"settings": {"viewportMarkerSize": 144}}
        self.assertTrue(fo.request_vtt_command("settings.set", draft))
        draft["settings"]["viewportMarkerSize"] = 240
        self.assertTrue(fo.get_vtt_status()["pending"])
        self.assertFalse(fo.request_vtt_command("movement.reset"))
        await self.until(lambda: self.commands(socket))
        command = self.commands(socket)[0]
        self.assertEqual(command["values"]["settings"]["viewportMarkerSize"], 144)
        self.assertEqual(command["sceneId"], "scene-a")
        await socket.incoming.put(self.reply(command["requestId"]))
        await self.until(lambda: not fo.get_vtt_status()["pending"])
        self.assertEqual(fo.get_vtt_status()["message"], "Applied")

    async def test_unavailable_connection_or_module_never_queues(self):
        self.assertFalse(fo.request_vtt_command("markers.set", {"enabled": False}))
        socket, _ = await self.connect(supported=False)
        self.assertFalse(fo.request_vtt_command("markers.set", {"enabled": False}))
        self.assertIn("Reload Foundry", fo.get_vtt_status()["message"])
        self.assertEqual(self.commands(socket), [])

    async def test_profile_apply_can_track_its_exact_acknowledgement(self):
        socket, _ = await self.connect()
        request_id = fo.request_vtt_command("settings.set", {"settings": {"displayDiagonalInches": 54}},
                                           return_request_id=True)
        self.assertIsInstance(request_id, str)
        pending = fo.get_vtt_status()["pendingRequest"]
        self.assertEqual(pending, {"requestId": request_id, "action": "settings.set", "sceneId": "scene-a"})
        fo._receive_vtt(self.reply("other-request"))
        self.assertIsNone(fo.get_vtt_status()["lastResult"])
        fo._receive_vtt(self.reply(request_id, success=False, message="Not applied"))
        result = fo.get_vtt_status()["lastResult"]
        self.assertEqual(result, {"requestId": request_id, "action": "settings.set", "sceneId": "scene-a",
                                  "success": False, "message": "Not applied"})
        result["success"] = True
        self.assertFalse(fo.get_vtt_status()["lastResult"]["success"])
        retry = fo.request_vtt_command("settings.set", return_request_id=True)
        self.assertNotEqual(retry, request_id)
        fo._receive_vtt(self.reply(retry))
        self.assertTrue(fo.get_vtt_status()["lastResult"]["success"])
        fo.set_scene_params(scene_id="scene-b")
        self.assertIsNone(fo.get_vtt_status()["lastResult"])

    async def test_unknown_actions_and_invalid_json_are_rejected(self):
        socket, _ = await self.connect()
        for action, values in (("arbitrary.code", {}), ([], {}), ("settings.set", []),
                               ("settings.set", {"value": float("nan")}),
                               ("settings.set", {"value": object()})):
            with self.subTest(action=action), self.assertRaises(ValueError):
                fo.request_vtt_command(action, values)
        self.assertEqual(self.commands(socket), [])

    async def test_missing_capability_is_not_silently_sent(self):
        await self.connect()
        fo._receive_vtt(self.state(capabilities=["markers.set"]))
        with self.assertRaisesRegex(ValueError, "does not support"):
            fo.request_vtt_command("capture.start")

    async def test_capture_controls_remain_available_during_recording(self):
        socket, _ = await self.connect()
        fo._receive_vtt(self.state(capture={"state": "recording", "recording": True,
                                          "phase": "placement", "waiting": False}))
        with patch.object(fo, "_capture_session", object()):
            self.assertFalse(fo.get_vtt_status()["blocked"])
            self.assertTrue(fo.request_vtt_command("capture.confirm"))
            self.assertTrue(fo.get_physical_grid_status()["blocked"])
        await self.until(lambda: self.commands(socket))
        self.assertEqual(self.commands(socket)[0]["action"], "capture.confirm")

    async def test_grid_and_vtt_edits_do_not_overlap(self):
        await self.connect()
        self.assertTrue(fo.request_physical_grid())
        self.assertTrue(fo.get_vtt_status()["blocked"])
        self.assertFalse(fo.request_vtt_command("settings.set"))
        fo._physical_grid_state["pending"] = None
        self.assertTrue(fo.request_vtt_command("settings.set"))
        self.assertTrue(fo.get_physical_grid_status()["blocked"])
        self.assertFalse(fo.request_physical_grid())

    async def test_scene_change_drops_queued_command(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_vtt_command("movement.undo"))
        fo.set_scene_params(scene_id="scene-b")
        await asyncio.sleep(0.02)
        self.assertEqual(self.commands(socket), [])
        self.assertFalse(fo.get_vtt_status()["pending"])
        self.assertTrue(fo.get_vtt_status()["blocked"])
        fo._receive_vtt(self.state(sceneId="scene-b"))
        self.assertFalse(fo.get_vtt_status()["blocked"])

    async def test_reconnect_discards_old_commands_and_results(self):
        first, task = await self.connect()
        self.assertTrue(fo.request_vtt_command("lifecycle.reload"))
        await self.until(lambda: self.commands(first))
        previous_id = self.commands(first)[0]["requestId"]
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(fo.get_vtt_status()["pending"])
        second, _ = await self.connect()
        fo._receive_vtt(self.reply(previous_id, message="obsolete"))
        self.assertEqual(self.commands(second), [])
        self.assertNotEqual(fo.get_vtt_status()["message"], "obsolete")

    async def test_timeout_allows_retry_without_accepting_old_ack(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_vtt_command("movement.reset"))
        await self.until(lambda: self.commands(socket))
        previous_id = self.commands(socket)[0]["requestId"]
        with patch.object(fo.time, "monotonic", return_value=fo._vtt_state["pending"]["deadline"] + 1):
            self.assertFalse(fo.get_vtt_status()["pending"])
        self.assertIn("did not confirm", fo.get_vtt_status()["message"])
        self.assertTrue(fo.request_vtt_command("movement.reset"))
        fo._receive_vtt(self.reply(previous_id))
        self.assertTrue(fo.get_vtt_status()["pending"])

    async def test_remote_updates_are_deep_copies_and_bad_state_is_ignored(self):
        await self.connect()
        fo._receive_vtt(self.state(assignment={"miniId": "blue"}, movement={"miniId": "red10", "usedFeet": 10}))
        snapshot = fo.get_vtt_status()
        self.assertEqual(snapshot["assignment"]["miniId"], "blue")
        snapshot["movement"]["usedFeet"] = 100
        self.assertEqual(fo.get_vtt_status()["movement"]["usedFeet"], 10)
        fo._receive_vtt(self.state(tokens=["bad"]))
        fo._receive_vtt(self.state(sceneId="other"))
        self.assertEqual(fo.get_vtt_status()["movement"]["usedFeet"], 10)

    async def test_error_result_and_invalid_ack_are_visible(self):
        await self.connect()
        for state, expected in ((self.state(), "No selected token"), ({}, "scene changed"),
                                (self.state(capture=[]), "invalid VTT")):
            self.assertTrue(fo.request_vtt_command("movement.undo"))
            request_id = fo._vtt_state["pending"]["requestId"]
            fo._receive_vtt(self.reply(request_id, success=False, message="No selected token", state=state))
            self.assertFalse(fo.get_vtt_status()["pending"])
            self.assertIn(expected, fo.get_vtt_status()["message"])


if __name__ == "__main__":
    unittest.main()
