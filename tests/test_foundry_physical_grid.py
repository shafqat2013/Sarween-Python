"""Desktop-to-display protocol tests without a camera, GUI, or network server."""

import asyncio
import json
import unittest
from unittest.mock import patch

import foundryoutput as fo
from foundry_delivery import MoveOutbox


SETTINGS = {"targetGridInches": 1.0, "displayDiagonalInches": 55.0,
            "physicalGridCalibration": 1.0}


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return json.dumps(await self.incoming.get())


class PhysicalGridBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        replacements = {
            "_delivery": MoveOutbox(), "_active_socket": None,
            "_connection_ready": False, "_protocol_ready": False, "_connection_error": "",
            "_loop": asyncio.get_running_loop(), "_ctrl_queue": None, "_assign_queue": None,
            "_capture_session": None, "_tracking_output_paused": False,
            "_physical_grid_state": {}, "SCENE_ID": "scene-a", "SCENE_W": 1000,
            "SCENE_H": 800, "GRID_PX": 50, "GRID_TYPE": 1, "SHIFT_X": 0,
            "SHIFT_Y": 0, "_grid_cols": 20, "_grid_rows": 16,
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
        return {"type": "physicalGridState", "sceneId": "scene-a", "supported": True,
                "ready": True, "settings": dict(SETTINGS), **overrides}

    async def connect(self, supported=True):
        socket = Socket()
        task = asyncio.create_task(fo.handler(socket))
        self.handlers.append(task)
        await socket.incoming.put({"type": "hello", "protocolVersion": 2,
                                   "physicalGridControls": supported})
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "scene-a",
                                   "width": 1000, "height": 800, "gridSize": 50, "gridType": 1})
        await socket.incoming.put(self.state())
        await self.until(lambda: fo._connection_ready)
        if supported:
            await self.until(lambda: not fo.get_physical_grid_status()["blocked"])
        return socket, task

    def edits(self, socket):
        return [msg for msg in socket.sent if msg["type"] == "physicalGridSet"]

    async def test_roundtrip_sets_pending_then_acknowledges_current_scene(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_physical_grid({"targetGridInches": 1.25}))
        self.assertTrue(fo.get_physical_grid_status()["pending"])
        self.assertFalse(fo.request_physical_grid({"targetGridInches": 1.5}))
        await self.until(lambda: self.edits(socket))
        request = self.edits(socket)[0]
        self.assertEqual(request["sceneId"], "scene-a")
        self.assertEqual(request["settings"], {"targetGridInches": 1.25})
        self.assertTrue(request["apply"])
        await socket.incoming.put(self.state(type="physicalGridResult", requestId=request["requestId"],
            success=True, message="Applied", settings={**SETTINGS, "targetGridInches": 1.25}))
        await self.until(lambda: not fo.get_physical_grid_status()["pending"])
        status = fo.get_physical_grid_status()
        self.assertEqual(status["settings"]["targetGridInches"], 1.25)
        self.assertEqual(status["message"], "Applied")

    async def test_disconnected_unsupported_and_capture_cannot_queue(self):
        self.assertFalse(fo.request_physical_grid())
        socket, task = await self.connect(supported=False)
        self.assertFalse(fo.request_physical_grid())
        self.assertIn("updated Sarween module", fo.get_physical_grid_status()["message"])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        socket, _ = await self.connect()
        with patch.object(fo, "_capture_session", object()):
            self.assertTrue(fo.get_physical_grid_status()["blocked"])
            self.assertFalse(fo.request_physical_grid())
        self.assertEqual(self.edits(socket), [])

    async def test_invalid_requests_never_queue(self):
        socket, _ = await self.connect()
        for settings in ([], {"extra": 1}, {"targetGridInches": True},
                         {"targetGridInches": float("nan")}, {"displayDiagonalInches": 151},
                         {"physicalGridCalibration": 0.49}, {"targetGridInches": "1"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                fo.request_physical_grid(settings)
        with self.assertRaises(ValueError):
            fo.request_physical_grid(apply="yes")
        self.assertFalse(fo.get_physical_grid_status()["pending"])
        self.assertEqual(self.edits(socket), [])

    async def test_queued_command_dropped_if_capture_starts(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_physical_grid())
        with patch.object(fo, "_capture_session", object()):
            await asyncio.sleep(0.02)
        self.assertEqual(self.edits(socket), [])
        self.assertFalse(fo.get_physical_grid_status()["pending"])

    async def test_scene_change_drops_queued_edit_and_invalidates_settings(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_physical_grid())
        fo.set_scene_params(scene_id="scene-b")
        await asyncio.sleep(0.02)
        self.assertEqual(self.edits(socket), [])
        status = fo.get_physical_grid_status()
        self.assertFalse(status["pending"])
        self.assertTrue(status["blocked"])
        self.assertFalse(fo.request_physical_grid())

    async def test_disconnect_clears_edit_and_reconnect_does_not_replay(self):
        first, task = await self.connect()
        self.assertTrue(fo.request_physical_grid({"targetGridInches": 1.25}))
        await self.until(lambda: self.edits(first))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(fo.get_physical_grid_status()["pending"])
        second, _ = await self.connect()
        self.assertEqual(self.edits(second), [])
        self.assertEqual(fo.get_physical_grid_status()["settings"], SETTINGS)

    async def test_timeout_ignores_late_ack_and_allows_explicit_retry(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_physical_grid())
        await self.until(lambda: self.edits(socket))
        request = self.edits(socket)[0]
        with patch.object(fo.time, "monotonic", return_value=fo._physical_grid_state["pending"]["deadline"] + 1):
            status = fo.get_physical_grid_status()
        self.assertFalse(status["pending"])
        self.assertIn("timed out", status["message"])
        fo._receive_physical_grid(self.state(type="physicalGridResult", requestId=request["requestId"],
            success=True, settings={**SETTINGS, "targetGridInches": 2.0}))
        self.assertEqual(fo.get_physical_grid_status()["settings"], SETTINGS)
        self.assertTrue(fo.request_physical_grid())

    async def test_error_ack_and_invalid_reply_are_visible(self):
        socket, _ = await self.connect()
        for settings, expected in ((SETTINGS, "No permission"), ({}, "invalid TV grid")):
            self.assertTrue(fo.request_physical_grid())
            request_id = fo._physical_grid_state["pending"]["requestId"]
            fo._receive_physical_grid(self.state(type="physicalGridResult", requestId=request_id,
                success=False, message="No permission", settings=settings))
            status = fo.get_physical_grid_status()
            self.assertFalse(status["pending"])
            self.assertIn(expected, status["message"])

    async def test_remote_updates_copy_settings_and_ignore_wrong_scene_or_id(self):
        socket, _ = await self.connect()
        fo._receive_physical_grid(self.state(settings={**SETTINGS, "displayDiagonalInches": 65.0}))
        status = fo.get_physical_grid_status()
        self.assertEqual(status["settings"]["displayDiagonalInches"], 65.0)
        status["settings"]["displayDiagonalInches"] = 100
        self.assertEqual(fo.get_physical_grid_status()["settings"]["displayDiagonalInches"], 65.0)
        fo._receive_physical_grid(self.state(sceneId="other"))
        self.assertEqual(fo.get_physical_grid_status()["settings"]["displayDiagonalInches"], 65.0)
        self.assertTrue(fo.request_physical_grid())
        fo._receive_physical_grid(self.state(type="physicalGridResult", requestId="obsolete", success=True))
        self.assertTrue(fo.get_physical_grid_status()["pending"])


if __name__ == "__main__":
    unittest.main()
