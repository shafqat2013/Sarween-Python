import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import foundryoutput as fo
from foundry_delivery import MoveOutbox


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = None

    async def send(self, raw):
        self.sent.append(json.loads(raw))

    async def recv(self):
        return json.dumps(await self.incoming.get())

    async def close(self, **kwargs):
        self.closed = kwargs


class DeliveryIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        replacements = {
            "_delivery": MoveOutbox(ack_timeout=0.2), "_active_socket": None,
            "_connection_ready": False, "_protocol_ready": False, "_connection_error": "",
            "_loop": asyncio.get_running_loop(), "_assign_queue": None, "_ctrl_queue": None,
            "_tracking_output_paused": False, "_capture_session": None,
            "MINI_TO_TOKEN": {}, "SCENE_TOKEN_NAMES": {}, "SCENE_ID": "test-scene",
            "SCENE_W": 1000, "SCENE_H": 800, "GRID_PX": 50, "GRID_TYPE": 1,
            "SHIFT_X": 0, "SHIFT_Y": 0, "_grid_cols": 20, "_grid_rows": 16,
            "MAP_PATH": Path(self.directory.name) / "mapping.json",
        }
        self.patches = [patch.object(fo, name, value) for name, value in replacements.items()]
        for item in self.patches:
            item.start()
        fo._delivery.set_context(fo._scene_context())
        self.handlers = []

    async def asyncTearDown(self):
        for handler in self.handlers:
            handler.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(wait(), timeout=2)

    async def connect(self, protocol=2):
        socket = Socket()
        handler = asyncio.create_task(fo.handler(socket))
        self.handlers.append(handler)
        await socket.incoming.put({"type": "hello", "protocolVersion": protocol})
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "test-scene",
                                   "width": 1000, "height": 800, "gridSize": 50, "gridType": 1})
        await self.until(lambda: fo._active_socket is socket)
        await asyncio.sleep(0.01)
        return socket, handler

    def moves(self, socket):
        return [message for message in socket.sent if message["type"] == "moveToken"]

    async def test_assignment_recovers_latest_position_without_another_detection(self):
        socket, _ = await self.connect()
        fo.queue_cell_move("red10", "B2")
        await self.until(lambda: any(message["type"] == "assignMini" for message in socket.sent))
        fo.queue_cell_move("red10", "C3")
        await socket.incoming.put({"type": "assignMiniResult", "sceneId": "test-scene",
                                   "miniId": "red10", "tokenId": "red-token"})
        await self.until(lambda: self.moves(socket))
        move = self.moves(socket)[-1]
        self.assertEqual(move["cell"], "C3")
        await socket.incoming.put({**move, "type": "tokenMoveApplied"})
        await self.until(lambda: fo._delivery.snapshot()["red10"]["state"] == "confirmed")

    async def test_reconnect_waits_for_handshake_then_sends_only_latest(self):
        fo.MINI_TO_TOKEN["red10"] = "red-token"
        socket, handler = await self.connect()
        fo.queue_cell_move("red10", "B2")
        await self.until(lambda: self.moves(socket))
        handler.cancel()
        await asyncio.gather(handler, return_exceptions=True)
        fo.queue_cell_move("red10", "C3")
        fo.queue_cell_move("red10", "D4")
        next_socket, _ = await self.connect()
        await self.until(lambda: self.moves(next_socket))
        self.assertEqual([message["cell"] for message in self.moves(next_socket)], ["D4"])

    async def test_ack_transcript_uses_verified_outbox_cell_not_missing_reply_cell(self):
        fo.MINI_TO_TOKEN["red10"] = "red-token"
        socket, _ = await self.connect()
        fo.queue_cell_move("red10", "C3")
        await self.until(lambda: self.moves(socket))
        reply = {**self.moves(socket)[-1], "type": "tokenMoveApplied"}
        reply.pop("cell")
        with patch.object(fo, "record_delivery_event") as record:
            await socket.incoming.put(reply)
            await self.until(lambda: record.called)
            self.assertEqual(record.call_args.args[0]["cell"], "C3")
            self.assertEqual(record.call_args.args[0]["type"], "tokenMoveApplied")

    async def test_wrong_position_is_not_logged_as_confirmed(self):
        fo.MINI_TO_TOKEN["red10"] = "red-token"
        socket, _ = await self.connect()
        fo.queue_cell_move("red10", "C3")
        await self.until(lambda: self.moves(socket))
        with patch.object(fo, "record_delivery_event") as record:
            await socket.incoming.put({**self.moves(socket)[-1], "type": "tokenMoveApplied", "x": -1})
            await self.until(lambda: record.called)
            self.assertEqual(record.call_args.args[0]["type"], "tokenMoveError")

    async def test_rejection_is_visible_and_token_replacement_recovers(self):
        fo.MINI_TO_TOKEN["red10"] = "deleted"
        socket, _ = await self.connect()
        fo.queue_cell_move("red10", "B2")
        await self.until(lambda: self.moves(socket))
        await socket.incoming.put({**self.moves(socket)[0], "type": "tokenMoveError",
                                   "code": "tokenMissing", "error": "Deleted token"})
        await self.until(lambda: "red10" not in fo.MINI_TO_TOKEN)
        await self.until(lambda: any(message["type"] == "assignMini" for message in socket.sent))
        self.assertIn("Assignment needed", fo.get_delivery_status()["message"])
        await socket.incoming.put({"type": "assignMiniResult", "sceneId": "test-scene",
                                   "miniId": "red10", "tokenId": "replacement"})
        await self.until(lambda: any(move["tokenId"] == "replacement" for move in self.moves(socket)))

    async def test_old_module_cannot_send_unacknowledged_moves(self):
        fo.MINI_TO_TOKEN["red10"] = "red-token"
        socket, _ = await self.connect(protocol=None)
        fo.queue_cell_move("red10", "B2")
        await asyncio.sleep(0.12)
        self.assertEqual(self.moves(socket), [])
        self.assertIn("Reload Foundry", fo.get_delivery_status()["message"])

    async def test_second_display_is_rejected_and_first_survives(self):
        first, _ = await self.connect()
        second = Socket()
        await fo.handler(second)
        self.assertEqual(second.closed["code"], 1013)
        self.assertIs(fo._active_socket, first)


if __name__ == "__main__":
    unittest.main()
