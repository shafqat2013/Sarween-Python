"""Session setup transaction boundaries without Tk, cameras or native windows."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from guided_foundry_setup import GuidedFoundryCoordinator, load_setup_preferences


TV = {"id": "12", "name": "AirPlay TV", "primary": False}
PROFILE = {"id": "profile", "name": "54-inch TV", "settings": {
    "displayDiagonalInches": 54, "targetGridInches": 1,
    "physicalGridCalibration": 1}, "targetDisplayId": "obsolete"}
STARTUP = {"worldId": "testing", "username": "Gamemaster", "unpause": True, "placeWindow": True}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.store = Mock()
        self.store.get_profile.return_value = deepcopy(PROFILE)
        self.store.active_profile.return_value = deepcopy(PROFILE)
        self.controller = Mock()
        self.controller.start.return_value = True
        self.controller.status.return_value = {"phase": "unpausing", "message": "Unpausing…"}
        self.controller.take_placement_request.return_value = True
        self.display = Mock()
        self.display.status.return_value = {"displays": [TV], "busy": False,
                                             "message": "Foundry is full screen on AirPlay TV."}
        self.display.place.return_value = True
        self.remote = {"connected": True, "supported": True, "blocked": False,
                       "pending": False, "sceneId": "scene", "capabilities": ["settings.set"]}
        self.request = Mock(return_value="request-1")
        self.access = Mock()
        self.clock = Mock(return_value=10)
        self.flow = GuidedFoundryCoordinator(service=self.service, store=self.store,
            controller=self.controller, display_controller=self.display,
            require_access=self.access, read_status=lambda: deepcopy(self.remote),
            request=self.request, clock=self.clock)
        self.choices = {"startup": STARTUP, "display": TV, "profileId": "profile",
                        "printed": False, "gridLabels": True, "password": "temporary"}

    def test_saved_overlay_defaults_are_local_and_validated(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "session_setup.json"
            self.assertEqual(load_setup_preferences(path), {})
            path.write_text('{"gridLabelsVisible":true,"password":"never load"}')
            self.assertEqual(load_setup_preferences(path), {"gridLabelsVisible": True})
            path.write_text('{"gridLabelsVisible":"false"}')
            self.assertEqual(load_setup_preferences(path), {})
            path.write_text('broken JSON')
            self.assertEqual(load_setup_preferences(path), {})

    def advance_to_settings(self):
        self.assertTrue(self.flow.begin(self.choices))
        self.controller.status.return_value = {"phase": "ready", "message": "Foundry ready"}
        self.assertEqual(self.flow.poll()["phase"], "placement")
        self.assertEqual(self.flow.poll()["phase"], "settings")

    def acknowledge(self, **changes):
        self.remote["lastResult"] = {"requestId": "request-1", "sceneId": "scene",
                                     "success": True, **changes}
        return self.flow.poll()

    def test_waits_for_unpause_before_placement_or_settings(self):
        self.assertTrue(self.flow.begin(self.choices))
        for _ in range(3):
            self.assertEqual(self.flow.poll()["phase"], "foundry")
        self.request.assert_not_called()
        self.display.place.assert_not_called()
        self.store.select_profile.assert_not_called()
        self.service.start.assert_called_once_with()

    def test_one_combined_settings_command_and_matching_ack(self):
        self.advance_to_settings()
        self.request.assert_called_once_with("settings.set", {"settings": {
            **PROFILE["settings"], "viewportMarkersEnabled": True, "gridLabelsVisible": True}})
        self.store.select_profile.assert_not_called()
        self.assertFalse(self.flow.poll()["ready"])
        self.assertTrue(self.acknowledge()["ready"])
        self.store.select_profile.assert_called_once_with("display", "profile")
        # Profile's obsolete preferred display must never override selected TV.
        self.display.configure.assert_called_once_with(display_id="12", auto=False)

    def test_same_scene_ack_but_grid_still_busy_does_not_finish(self):
        self.advance_to_settings()
        self.remote["blocked"] = True
        self.assertFalse(self.acknowledge()["ready"])
        self.remote["blocked"] = False
        self.assertTrue(self.flow.poll()["ready"])

    def test_scene_change_never_selects_or_mutates_profile(self):
        self.advance_to_settings()
        self.remote["sceneId"] = "other-scene"
        status = self.acknowledge(sceneId="other-scene")
        self.assertEqual(status["phase"], "failed")
        self.store.select_profile.assert_not_called()
        self.store.set_printed_markers_enabled.assert_not_called()

    def test_failed_ack_stays_in_setup(self):
        self.advance_to_settings()
        status = self.acknowledge(success=False, message="Permission denied")
        self.assertEqual(status["phase"], "failed")
        self.assertEqual(status["error"], "Permission denied")
        self.store.select_profile.assert_not_called()

    def test_profile_changed_while_request_pending_needs_review(self):
        self.advance_to_settings()
        self.store.get_profile.return_value = {**PROFILE, "name": "Changed"}
        self.assertEqual(self.acknowledge()["phase"], "failed")
        self.store.select_profile.assert_not_called()

    def test_airplay_id_reconnect_resolves_unique_same_name(self):
        self.choices["display"] = {**TV, "id": "old"}
        self.assertTrue(self.flow.begin(self.choices))
        self.display.configure.assert_called_once_with(display_id="12", auto=False)

    def test_placement_off_also_disables_saved_automatic_window_watcher(self):
        self.choices["startup"] = {**STARTUP, "placeWindow": False}
        self.assertTrue(self.flow.begin(self.choices))
        self.display.configure.assert_called_once_with(auto=False)

    def test_ambiguous_name_or_reused_id_fails_before_mutations(self):
        self.choices["display"] = {**TV, "id": "old"}
        self.display.status.return_value = {"displays": [TV, {**TV, "id": "13"}]}
        self.assertFalse(self.flow.begin(self.choices))
        self.display.configure.assert_not_called()
        self.controller.start.assert_not_called()
        self.service.start.assert_not_called()

    def test_no_permission_to_place_retains_actionable_warning(self):
        self.display.place.return_value = False
        self.display.status.return_value.update(message="Allow Sarween in Accessibility, or place Foundry manually.")
        self.assertTrue(self.flow.begin(self.choices))
        self.controller.status.return_value = {"phase": "ready", "message": "Ready"}
        status = self.flow.poll()
        self.assertEqual(status["phase"], "placement_attention")
        self.assertFalse(status["ready"])
        self.assertIn("manually", status["warning"])
        self.request.assert_not_called()
        self.assertTrue(self.flow.confirm_placement())
        self.flow.poll()
        self.assertTrue(self.acknowledge()["ready"])

    def test_placement_worker_failure_is_not_reported_as_success(self):
        self.advance_to_settings()
        # Placement result was successful above; a separate run checks failure.
        self.flow = GuidedFoundryCoordinator(service=self.service, store=self.store,
            controller=self.controller, display_controller=self.display,
            read_status=lambda: self.remote, request=self.request, clock=self.clock)
        self.flow.begin(self.choices)
        self.flow.poll()
        self.display.status.return_value["message"] = "Window cannot be moved through Accessibility."
        status = self.flow.poll()
        self.assertEqual(status["phase"], "placement_attention")
        self.assertIn("cannot be moved", status["warning"])
        self.request.reset_mock()
        self.assertEqual(self.flow.poll()["phase"], "placement_attention")
        self.request.assert_not_called()
        self.assertTrue(self.flow.confirm_placement())
        self.flow.poll()
        self.assertTrue(self.acknowledge()["ready"])

    def test_printed_mode_requires_alignment_without_changing_previous_mode(self):
        self.choices["printed"] = True
        self.assertFalse(self.flow.begin(self.choices))
        self.assertIn("registered display profile", self.flow.poll()["message"])
        self.service.start.assert_not_called()
        self.store.set_printed_markers_enabled.assert_not_called()

    def test_printed_choice_hides_screen_markers_and_changes_flag_only_after_ack(self):
        profile = {**PROFILE, "printedMarkerLayout": {"existing": "layout"}}
        self.store.get_profile.return_value = profile
        self.choices["printed"] = True
        with patch("printed_markers.validate_layout", return_value=profile["printedMarkerLayout"]):
            self.advance_to_settings()
        settings = self.request.call_args.args[1]["settings"]
        self.assertFalse(settings["viewportMarkersEnabled"])
        self.store.set_printed_markers_enabled.assert_not_called()
        self.assertTrue(self.acknowledge()["ready"])
        self.store.set_printed_markers_enabled.assert_called_once_with("profile", True, expected_profile=profile)

    def test_timeout_is_visible_and_never_selects_profile(self):
        self.advance_to_settings()
        self.clock.return_value = 100
        self.assertEqual(self.flow.poll()["phase"], "failed")
        self.store.select_profile.assert_not_called()

    def test_cancel_does_not_close_foundry_or_start_another_step(self):
        self.flow.begin(self.choices)
        self.flow.cancel()
        self.controller.cancel.assert_called_once_with()
        self.assertEqual(self.flow.poll()["phase"], "cancelled")
        self.display.place.assert_not_called()
        self.request.assert_not_called()

    def test_real_profile_printed_mode_round_trip_preserves_registered_layout(self):
        from hardware_profiles import HardwareProfiles
        from printed_markers import guide_aligned_layout
        with tempfile.TemporaryDirectory() as directory:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            layout = guide_aligned_layout({"viewportWidth": 1920, "viewportHeight": 1080,
                "registration": {"left": 22, "top": 22, "right": 1898, "bottom": 1058}}, 72, 10)
            profile = store.upsert_profile("display", {"name": "Actual TV", "settings": PROFILE["settings"],
                "printedMarkerLayout": layout, "printedMarkersEnabled": True}, select=True)
            self.flow.store = store
            # Keep the existing grid but switch runtime registration to screen
            # markers: active paper mode must change along with the overlays.
            self.choices["profileId"] = None
            self.choices["printed"] = False
            self.advance_to_settings()
            settings = self.request.call_args.args[1]["settings"]
            self.assertNotIn("displayDiagonalInches", settings)
            self.assertTrue(self.acknowledge()["ready"])
            after = store.active_profile("display")
            self.assertFalse(after["printedMarkersEnabled"])
            self.assertEqual(after["printedMarkerLayout"], profile["printedMarkerLayout"])
            self.assertEqual(after["settings"], profile["settings"])
            # Enable the same verified paper registration without redoing it.
            self.choices["profileId"] = profile["id"]
            self.choices["printed"] = True
            self.advance_to_settings()
            self.assertTrue(self.acknowledge()["ready"])
            restored = HardwareProfiles(store.path).active_profile("display")
            self.assertTrue(restored["printedMarkersEnabled"])
            self.assertEqual(restored["printedMarkerLayout"], profile["printedMarkerLayout"])

    def test_disconnected_cached_ack_cannot_commit_profile(self):
        self.advance_to_settings()
        self.remote["connected"] = False
        self.assertEqual(self.acknowledge()["phase"], "failed")
        self.store.select_profile.assert_not_called()

    def test_cancel_during_placement_invalidates_only_its_pending_job(self):
        self.flow.begin(self.choices)
        self.controller.status.return_value = {"phase": "ready", "message": "Ready"}
        self.assertEqual(self.flow.poll()["phase"], "placement")
        self.flow.cancel()
        self.assertEqual(self.display.set_enabled.call_args_list, [unittest.mock.call(False), unittest.mock.call(True)])
        self.request.assert_not_called()

    def test_access_revocation_stops_poll_before_actions(self):
        self.flow.begin(self.choices)
        self.access.side_effect = PermissionError("Session access expired")
        self.assertEqual(self.flow.poll()["phase"], "failed")
        self.request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
