"""Manual pause commands retain acknowledged state and existing safety gates."""
import unittest
from unittest.mock import patch

import foundryoutput as fo
from tests import test_vtt_bridge as fixture


class ManualPauseBridgeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixture.VTTBridgeTests.asyncSetUp
    asyncTearDown = fixture.VTTBridgeTests.asyncTearDown
    until = fixture.VTTBridgeTests.until
    state = fixture.VTTBridgeTests.state
    connect = fixture.VTTBridgeTests.connect
    commands = fixture.VTTBridgeTests.commands
    reply = fixture.VTTBridgeTests.reply

    async def test_absolute_pause_and_unpause_keep_readback_until_acknowledged(self):
        socket, _ = await self.connect()
        fo._receive_vtt(self.state(paused=False))
        for desired in (True, False):
            request_id = fo.request_vtt_command("lifecycle.setPaused", {"paused": desired},
                                               return_request_id=True)
            self.assertIsInstance(request_id, str)
            self.assertIs(fo.get_vtt_status()["paused"], not desired)
            self.assertTrue(fo.get_vtt_status()["pending"])
            await self.until(lambda: any(message["requestId"] == request_id
                                        for message in self.commands(socket)))
            command = self.commands(socket)[-1]
            self.assertEqual(command["action"], "lifecycle.setPaused")
            self.assertEqual(command["values"], {"paused": desired})
            await socket.incoming.put(self.reply(request_id, state=self.state(paused=desired)))
            await self.until(lambda: not fo.get_vtt_status()["pending"])
            self.assertIs(fo.get_vtt_status()["paused"], desired)

    async def test_external_pause_changes_update_state_without_completing_another_request(self):
        await self.connect()
        fo._receive_vtt(self.state(paused=False))
        request_id = fo.request_vtt_command("settings.set", {"settings": {"debug": True}},
                                           return_request_id=True)
        fo._receive_vtt(self.state(paused=True))
        self.assertIs(fo.get_vtt_status()["paused"], True)
        self.assertTrue(fo.get_vtt_status()["pending"])
        self.assertEqual(fo.get_vtt_status()["pendingRequest"]["requestId"], request_id)
        fo._receive_vtt(self.state(paused=False))
        self.assertIs(fo.get_vtt_status()["paused"], False)
        self.assertTrue(fo.get_vtt_status()["pending"])

    async def test_malformed_pause_values_never_queue(self):
        socket, _ = await self.connect()
        for values in ({}, {"paused": 1}, {"paused": "false"}, {"paused": None},
                       {"paused": False, "toggle": True}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "absolute"):
                fo.request_vtt_command("lifecycle.setPaused", values)
        self.assertEqual(self.commands(socket), [])

    async def test_capture_and_observer_modes_block_manual_pause(self):
        socket, _ = await self.connect()
        for phase in ("Starting", "placement", "Saving", "Restoring"):
            fo._receive_vtt(self.state(capture={"active": True, "phase": phase}))
            with self.subTest(phase=phase), self.assertRaisesRegex(ValueError, "Finish recording"):
                fo.request_vtt_command("lifecycle.setPaused", {"paused": True})
        fo._receive_vtt(self.state())
        with patch.object(fo, "_observation_only", True):
            with self.assertRaisesRegex(ValueError, "mini observer"):
                fo.request_vtt_command("lifecycle.setPaused", {"paused": True})
        self.assertEqual(self.commands(socket), [])

    async def test_invalid_pause_state_does_not_overwrite_confirmed_state(self):
        await self.connect()
        fo._receive_vtt(self.state(paused=True))
        for invalid in ("false", 0, None):
            fo._receive_vtt(self.state(paused=invalid))
            self.assertIs(fo.get_vtt_status()["paused"], True)


if __name__ == "__main__":
    unittest.main()
