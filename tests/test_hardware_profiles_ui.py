"""Headless checks for profile editing, selection, and acknowledged display apply."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from control_panel import ControlPanel, format_camera_calibration_status
from hardware_profiles import HardwareProfiles
from hardware_profiles_ui import (CameraProfilePicker, DisplayProfilePicker,
                                  HardwareProfilesPanel, CAMERA_FIELDS, camera_calibration_status,
                                  camera_form_values, display_settings, marker_total_text, DISPLAY_SETTINGS)


class Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def ready(**overrides):
    return {"connected": True, "supported": True, "pending": False, "blocked": False,
            "sceneId": "scene", "capabilities": ["settings.set"], "capture": {}, "message": "Ready",
            **overrides}


class HardwareProfilesUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = HardwareProfiles(Path(self.directory.name) / "profiles.json")

    def display_picker(self):
        existing = self.store.list_profiles("display")
        profile = self.store.upsert_profile("display", {"name": "54 inch TV",
            "settings": {"displayDiagonalInches": 54.0, "targetGridInches": 1.0}, "targetDisplayId": "tv"},
            profile_id=existing[0]["id"] if existing else None)
        picker = DisplayProfilePicker.__new__(DisplayProfilePicker)
        picker.store = self.store
        picker._require_access = Mock()
        picker._read_status = Mock(return_value=ready())
        picker._request = Mock(return_value="request-one")
        picker._display = Mock()
        picker._can_apply = None
        picker._enabled = True
        picker._profiles = {}
        picker._labels = {}
        picker._pending = None
        picker._last_status = {}
        picker._last_profiles = float("-inf")
        picker._selected_id = profile["id"]
        picker.choice = Variable()
        picker.message = Variable()
        picker.combo = Mock()
        picker.apply_button = Mock()
        picker.refresh(ready())
        return picker, profile

    def test_display_profile_is_not_activated_until_its_exact_successful_reply(self):
        picker, profile = self.display_picker()
        picker._read_status.side_effect = [ready(), ready(pending=True,
            pendingRequest={"requestId": "request-one", "action": "settings.set", "sceneId": "scene"})]
        self.assertTrue(picker.apply_profile())
        picker._request.assert_called_once_with("settings.set", {"settings": {
            "displayDiagonalInches": 54.0, "targetGridInches": 1.0}})
        self.assertIsNone(self.store.active_profile("display"))
        picker.refresh(ready(lastResult={"requestId": "request-one", "action": "settings.set",
            "sceneId": "scene", "success": True, "message": "Applied"}))
        self.assertEqual(self.store.active_profile("display")["id"], profile["id"])
        self.assertEqual(picker.message.get(), "Applied 54 inch TV.")
        picker._display.configure.assert_called_once_with(display_id="tv")
        picker._display.place.assert_not_called()

    def test_legacy_profile_applies_normalized_marker_fields_without_mutating_saved_profile(self):
        picker, profile = self.display_picker()
        profile = self.store.upsert_profile("display", {"settings": {"viewportMarkerSize": 120}},
                                            profile_id=profile["id"])
        before = self.store.path.read_bytes()
        picker._read_status.side_effect = [ready(), ready(pending=True,
            pendingRequest={"requestId": "request-one", "action": "settings.set", "sceneId": "scene"})]
        self.assertTrue(picker.apply_profile())
        settings = picker._request.call_args.args[1]["settings"]
        self.assertEqual(settings["viewportMarkerPatternSize"], 80)
        self.assertEqual(settings["viewportMarkerMargin"], 12)
        self.assertNotIn("viewportMarkerSize", settings)
        self.assertEqual(self.store.path.read_bytes(), before)
        picker.refresh(ready(lastResult={"requestId": "request-one", "action": "settings.set",
            "sceneId": "scene", "success": True}))
        self.assertEqual(picker.message.get(), "Applied 54 inch TV.")
        self.assertEqual(self.store.active_profile("display")["settings"]["viewportMarkerSize"], 120)

    def test_marker_controls_normalize_legacy_values_and_keep_pattern_independent(self):
        self.assertEqual(display_settings({"viewportMarkerSize": 72}),
                         {"viewportMarkerPatternSize": 48, "viewportMarkerMargin": 12})
        self.assertEqual(display_settings({"viewportMarkerPatternSize": "40", "viewportMarkerMargin": "0"}),
                         {"viewportMarkerPatternSize": 40, "viewportMarkerMargin": 0})
        self.assertEqual(display_settings({"viewportMarkerPatternSize": 240, "viewportMarkerMargin": 12,
            "viewportMarkerSize": 264}), {"viewportMarkerPatternSize": 240, "viewportMarkerMargin": 12})
        self.assertEqual(display_settings({"viewportMarkerMargin": "6"}), {"viewportMarkerMargin": 6})
        self.assertIn("60 pixels", marker_total_text({"viewportMarkerPatternSize": 48, "viewportMarkerMargin": 6}))
        self.assertIn("48 pixels", marker_total_text({"viewportMarkerPatternSize": 48, "viewportMarkerMargin": 0}))
        self.assertEqual(marker_total_text({"viewportMarkerPatternSize": "bad", "viewportMarkerMargin": 6}),
                         "Total marker width: —")
        for margin in (-1, 13, float("nan")):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                display_settings({"viewportMarkerMargin": margin})

    def test_loading_legacy_display_editor_shows_canonical_fields_without_writing(self):
        profile = self.store.upsert_profile("display", {"name": "Legacy TV", "settings": {"viewportMarkerSize": 72}})
        before = self.store.path.read_bytes()
        panel = HardwareProfilesPanel.__new__(HardwareProfilesPanel)
        panel.display_values = {key: Variable() for key in ("name", *DISPLAY_SETTINGS)}
        panel.display_choice = Variable(profile["name"])
        panel.include_display = Variable(False)
        panel._target_display_id = None
        panel._display_profiles = {profile["name"]: profile}
        panel.load_display()
        self.assertEqual(panel.display_values["viewportMarkerPatternSize"].get(), 48)
        self.assertEqual(panel.display_values["viewportMarkerMargin"].get(), 12)
        self.assertNotIn("viewportMarkerSize", panel._display_saved_keys)
        self.assertEqual(self.store.path.read_bytes(), before)

    def test_failed_wrong_scene_or_unrelated_reply_never_activates_profile(self):
        for result in ({"requestId": "request-one", "sceneId": "scene", "success": False},
                       {"requestId": "request-one", "sceneId": "other-scene", "success": True},
                       {"requestId": "other-request", "sceneId": "scene", "success": True}):
            with self.subTest(result=result):
                picker, _profile = self.display_picker()
                picker._read_status.side_effect = [ready(), ready(pending=True,
                    pendingRequest={"requestId": "request-one"})]
                self.assertTrue(picker.apply_profile())
                picker.refresh(ready(lastResult=result))
                self.assertIsNone(self.store.active_profile("display"))
                picker._display.configure.assert_not_called()

    def test_disconnected_capture_busy_or_auth_denied_profile_cannot_send(self):
        for state in (ready(connected=False), ready(pending=True), ready(blocked=True),
                      ready(capture={"phase": "Saving"}), ready(capture={"active": True})):
            with self.subTest(state=state):
                picker, _profile = self.display_picker()
                picker._read_status.return_value = state
                self.assertFalse(picker.apply_profile())
                picker._request.assert_not_called()
        picker, _profile = self.display_picker()
        picker._can_apply = Mock(return_value=False)
        self.assertFalse(picker.apply_profile())
        picker._request.assert_not_called()
        picker._can_apply = None
        picker._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(picker.apply_profile())
        self.assertEqual(picker.message.get(), "Sign in first")

    def test_lost_connection_during_apply_keeps_the_previous_active_selection(self):
        picker, _profile = self.display_picker()
        picker._read_status.side_effect = [ready(), ready(pending=True,
            pendingRequest={"requestId": "request-one"})]
        self.assertTrue(picker.apply_profile())
        picker.refresh(ready(connected=False, message="Connect Foundry to use VTT controls."))
        self.assertIsNone(self.store.active_profile("display"))
        self.assertIsNone(picker._pending)
        self.assertIn("Connect Foundry", picker.message.get())

    def test_profile_changed_or_deleted_during_apply_is_not_marked_active(self):
        for delete in (False, True):
            with self.subTest(delete=delete):
                picker, profile = self.display_picker()
                picker._read_status.side_effect = [ready(), ready(pending=True,
                    pendingRequest={"requestId": "request-one"})]
                self.assertTrue(picker.apply_profile())
                if delete:
                    self.store.delete_profile("display", profile["id"])
                else:
                    self.store.upsert_profile("display", {"settings": {"targetGridInches": 1.25}}, profile_id=profile["id"])
                picker.refresh(ready(lastResult={"requestId": "request-one", "action": "settings.set",
                    "sceneId": "scene", "success": True}))
                self.assertIsNone(self.store.active_profile("display"))
                self.assertIn("profile changed", picker.message.get())
                picker._display.configure.assert_not_called()

    def test_camera_setup_selection_can_be_cancelled_without_persisting(self):
        profile = self.store.upsert_profile("camera", {"name": "iPhone", "deviceLabel": "OBS Virtual Camera"}, select=True)
        picker = CameraProfilePicker.__new__(CameraProfilePicker)
        picker.store = self.store
        picker._changed = Mock()
        picker._persist = False
        picker._profiles = {}
        picker._selected_id = profile["id"]
        picker.choice = Variable()
        picker.status = Variable()
        picker.combo = Mock()
        picker.refresh()
        picker.select_id(None)
        self.assertIsNone(picker.selected_id())
        self.assertEqual(self.store.active_profile("camera")["id"], profile["id"])
        picker._changed.assert_called_once_with(None)
        picker.select_id(profile["id"])
        self.assertEqual(picker.selected_profile()["deviceLabel"], "OBS Virtual Camera")

    def test_partial_display_and_blank_camera_metadata_do_not_invent_feed_details(self):
        self.assertEqual(display_settings({"displayDiagonalInches": 54}), {"displayDiagonalInches": 54.0})
        values = camera_form_values({"name": "iPhone"})
        self.assertIsNone(values["resolution"])
        self.assertEqual(values["deviceLabel"], "")
        self.assertEqual(values["lens"], "")
        self.assertNotIn("calibration", values)
        self.assertNotIn("bindingVerified", values)
        with self.assertRaises(ValueError):
            camera_form_values({"name": "iPhone", "width": "1920"})
        with self.assertRaises(ValueError):
            display_settings({"viewportMarkerSize": float("nan")})
        with self.assertRaises(ValueError):
            display_settings({"displayDiagonalInches": 54}, complete=True)

    def test_manager_refreshes_external_calibration_metadata_but_preserves_drafts(self):
        profile = self.store.upsert_profile("camera", {"name": "Camera", "resolution": [640, 480]})
        panel = HardwareProfilesPanel.__new__(HardwareProfilesPanel)
        panel.store = self.store
        panel._enabled = True
        panel._require_access = Mock()
        panel._last_refresh = float("-inf")
        panel._camera_id = profile["id"]
        panel._display_id = None
        panel._camera_profiles = {profile["name"]: profile}
        panel._display_profiles = {}
        panel.camera_values = {key: Variable() for key in (*[key for key, _label in CAMERA_FIELDS], "width", "height")}
        panel.camera_choice = Variable("Camera")
        panel.camera_status = Variable()
        panel.message = Variable()
        panel.camera_combo = Mock()
        panel.display_combo = Mock()
        panel.load_camera()
        self.store.upsert_profile("camera", {"resolution": [1920, 1080]}, profile_id=profile["id"])
        panel.refresh(force=True)
        self.assertEqual(panel.camera_values["width"].get(), 1920)
        panel.camera_values["lens"].set("Draft lens")
        self.store.upsert_profile("camera", {"resolution": [1080, 1920]}, profile_id=profile["id"])
        panel.refresh(force=True)
        self.assertEqual(panel.camera_values["lens"].get(), "Draft lens")
        self.assertEqual(panel.camera_values["width"].get(), 1920)
        panel.use_camera()
        self.assertIsNone(self.store.active_profile("camera"))
        self.assertIn("Save these profile changes", panel.message.get())

    def test_calibration_profile_status_distinguishes_missing_unverified_and_saved(self):
        self.assertIn("No camera profile", camera_calibration_status(None))
        self.assertIn("No calibration", camera_calibration_status({"calibration": None}))
        calibration = {"imageSize": [1080, 1920]}
        self.assertIn("Unverified", camera_calibration_status({"calibration": calibration, "bindingVerified": False}))
        saved = camera_calibration_status({"calibration": calibration, "bindingVerified": True})
        self.assertIn("1080 × 1920", saved)
        self.assertNotIn("applied", saved)

    def test_live_calibration_label_reports_actual_result_and_caches_repeated_frames(self):
        text = format_camera_calibration_status({"profileName": "iPhone", "applied": False,
            "reason": "size mismatch", "frameSize": [1920, 1080]})
        self.assertIn("lens correction off: size mismatch", text)
        self.assertIn("1920 × 1080", text)
        self.assertIn("lens correction on", format_camera_calibration_status({"profileName": "iPhone", "applied": True}))
        self.assertIn("no profile", format_camera_calibration_status(None))
        panel = ControlPanel.__new__(ControlPanel)
        panel.var_camera_calibration = Mock()
        metadata = {"profileName": "iPhone", "applied": False, "reason": "size mismatch"}
        panel.set_camera_calibration_status(metadata)
        panel.set_camera_calibration_status(dict(metadata))
        panel.var_camera_calibration.set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
