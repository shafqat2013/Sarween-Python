import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from foundry_startup import (DEFAULTS, FoundryStartupController, NativeFoundryLauncher,
                            installed_worlds, normalize_preferences)


class Backend:
    def __init__(self, pids=(), steps=None):
        self.pids = list(pids)
        self.steps = iter(steps or [{"phase": "joined"}])
        self.launches = []
        self.logins = []

    def running_pids(self):
        return self.pids

    def launch(self, config, port):
        self.launches.append((dict(config), port))
        return 100

    def alive(self):
        return True

    def prepare_login(self, port, config, password):
        self.logins.append((port, config["worldId"], password))
        return next(self.steps, {"phase": "joined"})


class StartupTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        world = self.root / "Data/worlds/testing"
        world.mkdir(parents=True)
        (world / "world.json").write_text(json.dumps({"id": "testing", "title": "Testing"}))
        self.config = {**DEFAULTS, "dataPath": str(self.root)}

    def controller(self, *, backend=None, statuses=None, **kwargs):
        self.current = {"connected": True, "supported": True, "blocked": False,
                        "pending": False, "world": {"id": "testing"}, "paused": True,
                        "user": {"name": "Gamemaster", "isGM": True}}
        def request(action, values):
            self.commands.append((action, values))
            self.current.update(paused=False, lastResult={"requestId": "request-1", "success": True})
            return "request-1"
        self.commands = []
        self.backend = backend or Backend()
        c = FoundryStartupController(backend=self.backend, path=self.root / "preferences.json",
            read_status=lambda: dict(self.current), request=request, pause=lambda _: time.sleep(.001),
            connection_grace=.01, **kwargs)
        c._new_debug_port = lambda: 45123
        return c

    def wait(self, controller):
        deadline = time.monotonic() + 2
        while controller.status()["busy"] and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(controller.status()["busy"])
        return controller.status()

    def test_connected_world_skips_launch_and_unpauses_absolutely(self):
        c = self.controller()
        self.assertTrue(c.start(self.config))
        self.assertEqual(self.wait(c)["phase"], "ready")
        self.assertEqual(self.backend.launches, [])
        self.assertEqual(self.commands, [("lifecycle.setPaused", {"paused": False})])

    def test_cold_start_waits_for_game_then_module_and_does_not_persist_password(self):
        c = self.controller(backend=Backend(steps=[{"phase": "loading"}, {"phase": "joined"}]))
        self.current["connected"] = False
        original = self.backend.prepare_login
        def login(*args):
            result = original(*args)
            if result["phase"] == "joined":
                self.current["connected"] = True
            return result
        self.backend.prepare_login = login
        self.assertTrue(c.start(self.config, password="not-a-saved-secret"))
        self.assertEqual(self.wait(c)["phase"], "ready")
        self.assertEqual(len(self.backend.launches), 1)
        self.assertGreater(self.backend.launches[0][1], 0)
        saved = (self.root / "preferences.json").read_text()
        self.assertNotIn("not-a-saved-secret", saved)
        self.assertNotIn("password", saved.lower())

    def test_existing_unmanaged_native_app_is_not_duplicated(self):
        c = self.controller(backend=Backend(pids=[17]))
        self.current["connected"] = False
        c.start(self.config)
        status = self.wait(c)
        self.assertEqual(status["phase"], "failed")
        self.assertIn("Close it once", status["message"])
        self.assertFalse(self.backend.launches)

    def test_different_connected_world_is_not_replaced_or_unpaused(self):
        c = self.controller()
        self.current["world"]["id"] = "another-world"
        c.start(self.config)
        self.assertEqual(self.wait(c)["phase"], "failed")
        self.assertFalse(self.backend.launches)
        self.assertFalse(self.commands)

    def test_already_open_world_gets_time_to_reconnect_to_the_new_relay(self):
        c = self.controller(backend=Backend(pids=[17]))
        calls = []
        def snapshot():
            calls.append(True)
            return {**self.current, "connected": len(calls) >= 3}
        c._read_status = snapshot
        c.start(self.config)
        self.assertEqual(self.wait(c)["phase"], "ready")
        self.assertFalse(self.backend.launches)
        self.assertEqual(len(self.commands), 1)

    def test_failed_unpause_confirmation_is_not_reported_ready(self):
        c = self.controller()
        def reject(action, values):
            self.current["lastResult"] = {"requestId": "bad", "success": False, "message": "Observer mode is active"}
            return "bad"
        c._request = reject
        c.start(self.config)
        status = self.wait(c)
        self.assertEqual(status["phase"], "failed")
        self.assertIn("Observer", status["message"])

    def test_matching_world_with_different_user_is_not_reused(self):
        c = self.controller()
        self.current["user"]["name"] = "Another Gamemaster"
        c.start(self.config)
        self.assertEqual(self.wait(c)["phase"], "failed")
        self.assertFalse(self.commands)

    def test_cancel_during_module_status_read_never_sends_unpause(self):
        c = self.controller()
        calls = []
        def snapshot():
            calls.append(True)
            if len(calls) > 1:
                c.cancel()
            return dict(self.current)
        c._read_status = snapshot
        c.start(self.config)
        self.assertEqual(self.wait(c)["phase"], "cancelled")
        self.assertFalse(self.commands)

    def test_cancellation_stops_waiting_without_closing_or_relaunching_foundry(self):
        c = self.controller(backend=Backend(steps=[{"phase": "loading"}] * 1000))
        self.current["connected"] = False
        c.start(self.config)
        c.cancel()
        self.assertEqual(self.wait(c)["phase"], "cancelled")
        self.assertLessEqual(len(self.backend.launches), 1)
        self.assertFalse(self.commands)

    def test_window_placement_is_a_one_shot_main_thread_handoff(self):
        c = self.controller()
        c.start({**self.config, "placeWindow": True})
        self.assertEqual(self.wait(c)["phase"], "ready")
        self.assertTrue(c.take_placement_request())
        self.assertFalse(c.take_placement_request())

    def test_installed_world_ids_are_verified_against_manifest_folder(self):
        bad = self.root / "Data/worlds/mismatch"
        bad.mkdir()
        (bad / "world.json").write_text('{"id":"../../other","title":"Bad"}')
        self.assertEqual(installed_worlds(self.root), [{"id": "testing", "name": "Testing"}])

    def test_remote_urls_credentials_and_invalid_options_are_rejected(self):
        for values in ({"baseUrl": "https://example.com:30000"},
                       {"baseUrl": "http://user:secret@localhost:30000"},
                       {"baseUrl": "http://127.0.0.1:30000/../other"},
                       {"worldId": "testing; something"}, {"unpause": 1},
                       {"password": "must-not-save"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalize_preferences(values)

    def test_login_channel_targets_only_expected_local_origin(self):
        same = NativeFoundryLauncher._same_foundry_origin
        self.assertTrue(same("http://127.0.0.1:30000/join", DEFAULTS["baseUrl"]))
        self.assertFalse(same("http://example.com:30000/join", DEFAULTS["baseUrl"]))
        self.assertFalse(same("http://localhost:30001/join", DEFAULTS["baseUrl"]))
        self.assertFalse(same("http://localhost:30000/other", DEFAULTS["baseUrl"]))

    def test_no_credential_in_native_launch_arguments(self):
        app = self.root / "Foundry.app"
        (app / "Contents/MacOS").mkdir(parents=True)
        import plistlib
        (app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleIdentifier": "com.electron.foundryvtt", "CFBundleExecutable": "Foundry"}))
        executable = app / "Contents/MacOS/Foundry"
        executable.write_text("not actually run")
        executable.chmod(0o700)
        backend = NativeFoundryLauncher()
        backend.supported = True
        with patch("foundry_startup.subprocess.Popen") as run:
            backend.launch({**self.config, "appPath": str(app)}, 45123)
        args = run.call_args.args[0]
        self.assertIn("--world=testing", args)
        self.assertIn("--remote-debugging-address=127.0.0.1", args)
        self.assertFalse(any("password" in arg for arg in args))
        self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
