"""Main-panel startup readiness without networks, cameras or native windows."""

import asyncio
import json
import unittest
from unittest.mock import Mock, PropertyMock, patch

import calibration
import foundryoutput as fo
from foundry_delivery import MoveOutbox
from foundry_service import FoundryService, validated_scene_info


SCENE = {"sceneId": "scene", "width": 1000, "height": 800, "gridSize": 50}


class SceneValidationTests(unittest.TestCase):
    def test_complete_snapshot_is_normalized_without_unrelated_objects(self):
        scene = validated_scene_info({**SCENE, "unrelated": object()})
        self.assertEqual(scene, {"sceneId": "scene", "sceneW": 1000,
                                 "sceneH": 800, "gridPx": 50})
        self.assertEqual(json.loads(json.dumps(scene)), scene)

    def test_missing_identity_or_dimensions_cannot_use_defaults(self):
        for invalid in (None, {}, {**SCENE, "sceneId": ""},
                        {key: value for key, value in SCENE.items() if key != "width"},
                        {**SCENE, "height": 0}, {**SCENE, "gridSize": True},
                        {**SCENE, "width": float("nan")},
                        {**SCENE, "gridSize": float("inf")},
                        {**SCENE, "error": "No active scene"}):
            with self.subTest(invalid=invalid):
                self.assertIsNone(validated_scene_info(invalid))


class ServiceStartupTests(unittest.TestCase):
    def status(self, connection, state="Listening", error=""):
        service = FoundryService()
        service.error = error
        with patch.object(FoundryService, "state", new_callable=PropertyMock, return_value=state), \
             patch.object(fo, "get_connection_status", return_value=connection):
            return service.startup_status()

    def test_delayed_module_and_scene_progress_without_failing_or_restarting(self):
        stages = (
            ({"connected": False}, "waiting_module"),
            ({"connected": True, "handshake_ready": False}, "waiting_module"),
            ({"connected": True, "handshake_ready": True, "scene_ready": False}, "waiting_scene"),
            ({"connected": True, "handshake_ready": True, "scene_ready": True,
              "scene_info": SCENE}, "ready"),
        )
        for connection, phase in stages:
            result = self.status(connection)
            self.assertEqual(result["phase"], phase)
            self.assertEqual(result["error"], "")
            self.assertTrue(result["message"])
            json.dumps(result)
        self.assertEqual(self.status(stages[-1][0])["scene_info"], validated_scene_info(SCENE))

    def test_current_handshake_and_valid_scene_are_both_required(self):
        for status in ({"connected": False, "handshake_ready": True, "scene_ready": True,
                        "scene_info": SCENE},
                       {"connected": True, "handshake_ready": False, "scene_ready": True,
                        "scene_info": SCENE},
                       {"connected": True, "handshake_ready": True, "scene_ready": True,
                        "scene_info": {"sceneId": "scene"}}):
            result = self.status(status)
            self.assertNotEqual(result["phase"], "ready")
            self.assertIsNone(result["scene_info"])

    def test_relay_start_is_progress_but_actual_failure_remains_visible(self):
        self.assertEqual(self.status({}, state="Starting")["phase"], "starting_relay")
        result = self.status({}, state="Failed", error="Address already in use")
        self.assertEqual(result["phase"], "failed")
        self.assertEqual(result["error"], "Address already in use")

    def test_module_update_message_is_recoverable_waiting_state(self):
        result = self.status({"connected": True, "handshake_ready": False,
                              "error": "Reload Foundry: updated Sarween module required"})
        self.assertEqual(result["phase"], "waiting_module")
        self.assertIn("Reload Foundry", result["message"])
        self.assertEqual(result["error"], "")


class CalibrationStartupTests(unittest.TestCase):
    def test_prevalidated_scene_bypasses_every_wait_window(self):
        access = Mock()
        with patch.object(calibration, "_get_mode", return_value="foundry"), \
             patch.object(calibration, "_foundry_wait_for_scene_grid") as wait, \
             patch.object(calibration.s, "grid_cols", None, create=True), \
             patch.object(calibration.s, "grid_rows", None, create=True):
            result = calibration.prepare_map_asset(access, scene_info=SCENE)
            self.assertEqual(result[3:5], (20, 16))
            wait.assert_not_called()
            access.require.assert_called_once()
            self.assertEqual((calibration.s.grid_cols, calibration.s.grid_rows), (20, 16))

    def test_bad_explicit_snapshot_errors_without_falling_back_to_a_popup(self):
        with patch.object(calibration, "_get_mode", return_value="foundry"), \
             patch.object(calibration, "_foundry_wait_for_scene_grid") as wait:
            with self.assertRaisesRegex(ValueError, "valid dimensions"):
                calibration.prepare_map_asset(scene_info={})
            wait.assert_not_called()

    def test_legacy_none_path_and_calibrate_forwarding_are_preserved(self):
        access = Mock()
        with patch.object(calibration, "_get_mode", return_value="foundry"), \
             patch.object(calibration, "_foundry_wait_for_scene_grid", return_value=(20, 16, SCENE)) as wait, \
             patch.object(calibration.s, "grid_cols", None, create=True), \
             patch.object(calibration.s, "grid_rows", None, create=True):
            calibration.prepare_map_asset(access)
            wait.assert_called_once_with(access)
        with patch.object(calibration, "initialize_user_data"), \
             patch.object(calibration, "_get_mode", return_value="foundry"), \
             patch.object(calibration, "prepare_map_asset") as prepare:
            calibration.calibrate(access, scene_info=SCENE)
            prepare.assert_called_once_with(access, scene_info=SCENE)

    def test_explicit_scene_does_not_bypass_access(self):
        access = Mock()
        access.require.side_effect = PermissionError("Sign in required")
        with patch.object(calibration, "_get_mode", return_value="foundry"), \
             patch.object(calibration, "_foundry_wait_for_scene_grid") as wait:
            with self.assertRaises(PermissionError):
                calibration.prepare_map_asset(access, scene_info=SCENE)
            wait.assert_not_called()


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()

    async def recv(self):
        return json.dumps(await self.incoming.get())

    async def send(self, _raw):
        pass


class ConnectionFreshnessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        replacements = {
            "_active_socket": None, "_connection_ready": False, "_protocol_ready": False,
            "_connection_error": "", "_connection_generation": 10,
            "_scene_info_generation": 10, "_connection_scene_info": validated_scene_info(SCENE),
            "_loop": asyncio.get_running_loop(), "_assign_queue": None, "_ctrl_queue": None,
            "_delivery": MoveOutbox(), "_capture_session": None,
            "_physical_grid_state": {}, "_vtt_state": {},
            "SCENE_ID": "scene", "SCENE_W": 1000, "SCENE_H": 800,
            "GRID_PX": 50, "SHIFT_X": 0, "SHIFT_Y": 0,
            "SCENE_BACKGROUND": None, "_grid_cols": 20, "_grid_rows": 16,
        }
        self.patches = [patch.object(fo, key, value) for key, value in replacements.items()]
        for item in self.patches:
            item.start()
        self.handlers = []

    async def asyncTearDown(self):
        for task in self.handlers:
            task.cancel()
        await asyncio.gather(*self.handlers, return_exceptions=True)
        for item in reversed(self.patches):
            item.stop()

    async def until(self, predicate):
        async def wait():
            while not predicate():
                await asyncio.sleep(.001)
        await asyncio.wait_for(wait(), 2)

    async def connect(self):
        socket = Socket()
        task = asyncio.create_task(fo.handler(socket))
        self.handlers.append(task)
        await self.until(lambda: fo._active_socket is socket)
        return socket, task

    async def test_new_connection_cannot_accept_cached_scene_and_getter_is_read_only(self):
        socket, _task = await self.connect()
        status = fo.get_connection_status()
        self.assertEqual(status["generation"], 11)
        self.assertFalse(status["scene_ready"])
        self.assertIsNone(status["scene_info"])
        await socket.incoming.put({"type": "hello", "protocolVersion": 2})
        await self.until(lambda: fo.get_connection_status()["handshake_ready"])
        self.assertFalse(fo.get_connection_status()["scene_ready"])
        # Existing scene params remain cached, but this message is incomplete.
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "scene"})
        await self.until(lambda: fo._connection_ready)
        self.assertFalse(fo.get_connection_status()["scene_ready"])
        await socket.incoming.put({"type": "sceneInfo", **SCENE})
        await self.until(lambda: fo.get_connection_status()["scene_ready"])
        with patch.object(fo, "get_vtt_status", side_effect=AssertionError("No command expiration")), \
             patch.object(fo, "get_physical_grid_status", side_effect=AssertionError("No command expiration")), \
             patch.object(fo, "queue_control", side_effect=AssertionError("No commands")):
            snapshot = fo.get_connection_status()
        self.assertEqual(snapshot["scene_info"], validated_scene_info(SCENE))
        json.dumps(snapshot)

    async def test_disconnect_reconnect_and_hello_each_need_fresh_scene(self):
        socket, task = await self.connect()
        await socket.incoming.put({"type": "hello", "protocolVersion": 2})
        await socket.incoming.put({"type": "sceneInfo", **SCENE})
        await self.until(lambda: fo.get_connection_status()["scene_ready"])
        await socket.incoming.put({"type": "hello", "protocolVersion": 2})
        await self.until(lambda: not fo.get_connection_status()["scene_ready"])
        await socket.incoming.put({"type": "sceneInfo", **SCENE})
        await self.until(lambda: fo.get_connection_status()["scene_ready"])
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertFalse(fo.get_connection_status()["connected"])
        self.assertIsNone(fo.get_connection_status()["scene_info"])
        socket, _task = await self.connect()
        self.assertEqual(fo.get_connection_status()["generation"], 12)
        await socket.incoming.put({"type": "hello", "protocolVersion": 2})
        await self.until(lambda: fo.get_connection_status()["handshake_ready"])
        self.assertFalse(fo.get_connection_status()["scene_ready"])


if __name__ == "__main__":
    unittest.main()
