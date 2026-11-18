"""Visibility changes use the existing scene-scoped, confirmed Foundry relay."""

import asyncio
import unittest
from unittest.mock import patch

import foundryoutput as fo
from tests import test_vtt_bridge as bridge_fixture


SCENE_VALUES = {"tokenVision": True, "fogExploration": True, "globalIllumination": False}
TOKEN_VALUES = {"tokenId": "red", "enabled": True, "range": 30}


class VisionFogBridgeTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = bridge_fixture.VTTBridgeTests.asyncSetUp
    asyncTearDown = bridge_fixture.VTTBridgeTests.asyncTearDown
    until = bridge_fixture.VTTBridgeTests.until
    state = bridge_fixture.VTTBridgeTests.state
    connect = bridge_fixture.VTTBridgeTests.connect
    commands = bridge_fixture.VTTBridgeTests.commands
    reply = bridge_fixture.VTTBridgeTests.reply

    async def test_settings_and_view_selection_wait_for_confirmed_scene_state(self):
        socket, _ = await self.connect()
        before = {"tokenVision": False, "fogExploration": False,
                  "globalIllumination": True, "distanceUnits": "ft", "selectedTokenId": None}
        fo._receive_vtt(self.state(vision=before))
        for index, (action, values) in enumerate((
            ("vision.scene.set", SCENE_VALUES),
            ("vision.token.set", TOKEN_VALUES),
            ("vision.token.select", {"tokenId": "red"}),
        )):
            with self.subTest(action=action):
                request_id = fo.request_vtt_command(action, values, return_request_id=True)
                self.assertIsInstance(request_id, str)
                self.assertEqual(fo.get_vtt_status()["vision"], before)
                await self.until(lambda: len(self.commands(socket)) == index + 1)
                command = self.commands(socket)[-1]
                self.assertEqual(command["sceneId"], "scene-a")
                self.assertEqual(command["action"], action)
                self.assertEqual(command["values"], values)
                confirmed = {**before, **SCENE_VALUES, "selectedTokenId": "red"}
                fo._receive_vtt(self.reply(request_id, state=self.state(
                    vision=confirmed, tokens=[{"tokenId": "red", "tokenName": "Red",
                                               "vision": {"enabled": True, "range": 30}}])))
                self.assertEqual(fo.get_vtt_status()["vision"], confirmed)
                self.assertTrue(fo.get_vtt_status()["lastResult"]["success"])
                before = confirmed
        # Clearing the controlled view is explicit, not an empty token ID.
        self.assertTrue(fo.request_vtt_command("vision.token.select", {"tokenId": None}))

    async def test_invalid_or_incomplete_visibility_changes_never_enter_queue(self):
        socket, _ = await self.connect()
        invalid = [
            ("vision.scene.set", {}),
            ("vision.scene.set", {**SCENE_VALUES, "tokenVision": 1}),
            ("vision.scene.set", {**SCENE_VALUES, "unexpected": False}),
            ("vision.token.set", {**TOKEN_VALUES, "tokenId": None}),
            ("vision.token.set", {**TOKEN_VALUES, "tokenId": " red "}),
            ("vision.token.set", {**TOKEN_VALUES, "enabled": "true"}),
            ("vision.token.set", {"tokenId": "red", "enabled": True}),
            ("vision.token.select", {"tokenId": ""}),
            ("vision.token.select", {"tokenId": True}),
            ("vision.token.select", {"tokenId": "red", "extra": True}),
        ]
        invalid += [("vision.token.set", {**TOKEN_VALUES, "range": value})
                    for value in (True, "30", -1, float("nan"), float("inf"))]
        for action, values in invalid:
            with self.subTest(action=action, values=values), self.assertRaises(ValueError):
                fo.request_vtt_command(action, values)
        await asyncio.sleep(0)
        self.assertEqual(self.commands(socket), [])

    async def test_observer_capture_and_old_module_cannot_change_visibility(self):
        socket, _ = await self.connect()
        actions = (("vision.scene.set", SCENE_VALUES), ("vision.token.set", TOKEN_VALUES),
                   ("vision.token.select", {"tokenId": "red"}))
        for flag in ("_observation_only", "_tracking_output_paused"):
            with patch.object(fo, flag, True):
                for action, values in actions:
                    with self.subTest(flag=flag, action=action), self.assertRaisesRegex(ValueError, "Stop the mini observer"):
                        fo.request_vtt_command(action, values)
        for capture in ({"active": True}, {"recording": True}, {"phase": "Saving"}):
            fo._receive_vtt(self.state(capture=capture))
            for action, values in actions:
                with self.subTest(capture=capture, action=action), self.assertRaisesRegex(ValueError, "Finish recording"):
                    fo.request_vtt_command(action, values)
        fo._receive_vtt(self.state(capabilities=["settings.set"]))
        for action, values in actions:
            with self.assertRaisesRegex(ValueError, "does not support"):
                fo.request_vtt_command(action, values)
        self.assertEqual(self.commands(socket), [])

    async def test_new_observer_or_scene_change_cancels_a_queued_visibility_change(self):
        socket, _ = await self.connect()
        self.assertTrue(fo.request_vtt_command("vision.scene.set", SCENE_VALUES))
        with patch.object(fo, "_observation_only", True):
            await asyncio.sleep(0.02)
        self.assertEqual(self.commands(socket), [])
        self.assertFalse(fo.get_vtt_status()["pending"])
        self.assertIn("Stop the mini observer", fo.get_vtt_status()["message"])
        self.assertTrue(fo.request_vtt_command("vision.token.set", TOKEN_VALUES))
        fo.set_scene_params(scene_id="scene-b")
        await asyncio.sleep(0.02)
        self.assertEqual(self.commands(socket), [])


if __name__ == "__main__":
    unittest.main()
