"""Printed marker profile persistence and explicit controls without devices/apps."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from hardware_profiles import HardwareProfiles
from printed_marker_controls import PrintedMarkerControls


def layout():
    corners = {}
    for marker, x, y in ((20, -.06, -.06), (21, 1.01, -.06), (22, 1.01, 1.01), (23, -.06, 1.01)):
        corners[str(marker)] = [[x, y], [x+.04, y], [x+.04, y+.04], [x, y+.04]]
    return {"schemaVersion": 1, "dictionary": "DICT_6X6_250",
            "markerIds": {"TL": 20, "TR": 21, "BR": 22, "BL": 23},
            "markerCorners": corners, "viewportAspectRatio": 16/9}


class PrintedMarkerStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "profiles.json"
        self.store = HardwareProfiles(self.path)
        self.validate = Mock(side_effect=deepcopy)
        self.patcher = patch.dict("sys.modules", {"printed_markers": SimpleNamespace(validate_layout=self.validate)})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def display(self):
        return self.store.upsert_profile("display", {"name": "54 inch TV", "settings": {
            "displayDiagonalInches": 54, "targetGridInches": 1}}, select=True)

    def test_defaults_are_off_and_calibration_round_trips_without_losing_grid(self):
        display = self.display()
        self.assertIsNone(display["printedMarkerLayout"])
        self.assertFalse(display["printedMarkersEnabled"])
        saved = self.store.upsert_profile("display", {"printedMarkerLayout": layout(),
            "printedMarkersEnabled": True}, profile_id=display["id"])
        loaded = HardwareProfiles(self.path).active_profile("display")
        self.assertEqual(loaded, saved)
        self.assertEqual(loaded["settings"]["displayDiagonalInches"], 54)
        self.assertEqual(loaded["printedMarkerLayout"], layout())
        self.assertTrue(loaded["printedMarkersEnabled"])
        self.validate.assert_called()
        renamed = self.store.rename_profile("display", display["id"], "TV renamed")
        self.assertEqual(renamed["printedMarkerLayout"], layout())
        saved["printedMarkerLayout"]["markerCorners"]["20"][0][0] = 999
        self.assertEqual(self.store.active_profile("display")["printedMarkerLayout"], layout())

    def test_legacy_profile_reads_without_migration_or_new_file_write(self):
        display = self.display()
        raw = json.loads(self.path.read_text())
        raw["displays"][0].pop("printedMarkerLayout")
        raw["displays"][0].pop("printedMarkersEnabled")
        self.path.write_text(json.dumps(raw))
        before = self.path.read_bytes()
        current = HardwareProfiles(self.path).active_profile("display")
        self.assertEqual(current["id"], display["id"])
        self.assertFalse(current.get("printedMarkersEnabled", False))
        self.assertIsNone(current.get("printedMarkerLayout"))
        self.assertEqual(self.path.read_bytes(), before)
        self.validate.assert_not_called()

    def test_invalid_flags_layouts_and_enable_without_registration_never_write(self):
        display = self.display()
        before = self.path.read_bytes()
        for values in ({"printedMarkersEnabled": 1}, {"printedMarkersEnabled": "false"},
                       {"printedMarkerLayout": []}, {"printedMarkersEnabled": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.store.upsert_profile("display", values, profile_id=display["id"])
            self.assertEqual(self.path.read_bytes(), before)
        self.validate.side_effect = ValueError("Bad marker geometry")
        with self.assertRaisesRegex(ValueError, "Bad marker geometry"):
            self.store.upsert_profile("display", {"printedMarkerLayout": layout()}, profile_id=display["id"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_saved_printed_fields_are_rejected_on_read_without_overwrite(self):
        self.display()
        for field, value in (("printedMarkerLayout", "broken"), ("printedMarkersEnabled", 1),
                             ("printedMarkersEnabled", True)):
            raw = json.loads(self.path.read_text())
            raw["displays"][0][field] = value
            self.path.write_text(json.dumps(raw))
            before = self.path.read_bytes()
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                HardwareProfiles(self.path).active_profile("display")
            self.assertEqual(self.path.read_bytes(), before)
            raw["displays"][0][field] = None if field == "printedMarkerLayout" else False
            self.path.write_text(json.dumps(raw))

    def test_layout_validator_runs_for_saved_layout_before_use(self):
        display = self.display()
        self.store.upsert_profile("display", {"printedMarkerLayout": layout()}, profile_id=display["id"])
        before = self.path.read_bytes()
        self.validate.side_effect = ValueError("Unsupported layout schema")
        with self.assertRaisesRegex(ValueError, "Unsupported layout schema"):
            HardwareProfiles(self.path).list_profiles("display")
        self.assertEqual(self.path.read_bytes(), before)

    def test_camera_profiles_reject_printed_display_fields(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self.store.upsert_profile("camera", {"name": "Phone", "printedMarkersEnabled": False})

    def test_atomic_calibration_save_requires_same_active_unchanged_profile(self):
        display = self.display()
        saved = self.store.save_printed_marker_layout(display["id"], layout(), expected_profile=display)
        self.assertTrue(saved["printedMarkersEnabled"])
        other = self.store.upsert_profile("display", {"name": "Other", "settings": {
            "displayDiagonalInches": 32}}, select=True)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "active display profile changed"):
            self.store.save_printed_marker_layout(display["id"], layout(), expected_profile=saved)
        self.assertEqual(self.path.read_bytes(), before)
        self.store.select_profile("display", display["id"])
        HardwareProfiles(self.path).rename_profile("display", display["id"], "Renamed in another process")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "display profile changed"):
            self.store.save_printed_marker_layout(display["id"], layout(), expected_profile=saved)
        self.assertEqual(self.path.read_bytes(), before)

    def test_atomic_mode_change_cannot_write_a_switched_or_edited_profile(self):
        display = self.display()
        saved = self.store.save_printed_marker_layout(display["id"], layout(), expected_profile=display)
        disabled = self.store.set_printed_markers_enabled(display["id"], False, expected_profile=saved)
        self.assertFalse(disabled["printedMarkersEnabled"])
        self.assertEqual(disabled["printedMarkerLayout"], saved["printedMarkerLayout"])
        self.store.select_profile("display", None)
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "active display profile changed"):
            self.store.set_printed_markers_enabled(display["id"], True, expected_profile=disabled)
        self.assertEqual(self.path.read_bytes(), before)
        self.store.select_profile("display", display["id"])
        HardwareProfiles(self.path).rename_profile("display", display["id"], "Edited externally")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "display profile changed"):
            self.store.set_printed_markers_enabled(display["id"], True, expected_profile=disabled)
        self.assertEqual(self.path.read_bytes(), before)


class Value:
    def __init__(self, value=""):
        self.value = value
    def get(self):
        return self.value
    def set(self, value):
        self.value = value


class PrintedMarkerRealSchemaTests(unittest.TestCase):
    def test_real_layout_canonicalization_round_trips_and_rejects_saved_corruption(self):
        from printed_markers import validate_layout
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profiles.json"
            store = HardwareProfiles(path)
            display = store.upsert_profile("display", {"name": "TV", "settings": {
                "displayDiagonalInches": 54}}, select=True)
            supplied = {**layout(), "calibrationMethod": "guide-aligned-user-declared"}
            saved = store.save_printed_marker_layout(display["id"], supplied, expected_profile=display)
            self.assertEqual(saved["printedMarkerLayout"], validate_layout(supplied))
            self.assertEqual(HardwareProfiles(path).active_profile("display"), saved)
            raw = json.loads(path.read_text())
            raw["displays"][0]["printedMarkerLayout"]["markerIds"]["TL"] = 99
            path.write_text(json.dumps(raw))
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "DICT_6X6_250"):
                HardwareProfiles(path).active_profile("display")
            self.assertEqual(path.read_bytes(), before)
class Controller:
    def __init__(self, profile):
        self.value = {"available": True, "mode": "screen", "message": "Ready to register",
                      "pending": False, "calibrated": False, "profileName": profile["name"],
                      "profileId": profile["id"]}
        self.request_calibration = Mock(side_effect=self.request)
        self.cancel_calibration = Mock(side_effect=self.cancel)
        self.use_printed = Mock(side_effect=self.use)
        self.view_context = Mock(return_value={"ready": True, "view": {"viewportWidth": 1920,
            "viewportHeight": 1080}, "settings": {"viewportMarkerSize": 120, "viewportMarkerInset": 8}})
    def status(self):
        return deepcopy(self.value)
    def request(self, **_kwargs):
        self.value.update(pending=True, message="Aligning…")
    def cancel(self):
        self.value.update(pending=False, message="Cancelled")
    def use(self, enabled):
        self.value.update(mode="printed" if enabled else "screen", message="Mode applied")


class PrintedMarkerControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = HardwareProfiles(Path(self.temp.name) / "profiles.json")
        self.display = self.store.upsert_profile("display", {"name": "TV", "settings": {
            "displayDiagonalInches": 54}}, select=True)
        self.controller = Controller(self.display)
        self.controls = PrintedMarkerControls.__new__(PrintedMarkerControls)
        ui = self.controls
        ui.store, ui._controller = self.store, self.controller
        ui._require_access, ui._can_change = Mock(), Mock(return_value=True)
        ui._open_sheet = Mock()
        ui._sheet_parameters = Mock(return_value={"diagonal_inches": 54, "viewport_width": 1920,
            "viewport_height": 1080, "marker_size_px": 120, "marker_inset_px": 8})
        ui._sheet_builder = Mock(side_effect=lambda path, **_options: {"path": str(path)})
        ui._enabled, ui._dirty, ui._syncing = True, False, False
        ui._profile_id, ui._last_status, ui._notice = None, {}, ""
        ui.use_paper, ui.profile, ui.message = Value(False), Value(), Value()
        for name in ("sheet_button", "register_button", "bezel_button", "cancel_button", "use_check", "apply_button"):
            setattr(ui, name, Mock())
        ui.refresh()
        camera_patch = patch("cv2.VideoCapture", side_effect=AssertionError("No camera"))
        self.camera = camera_patch.start()
        self.addCleanup(camera_patch.stop)
        window_patch = patch("tkinter.Toplevel", side_effect=AssertionError("No window"))
        self.window = window_patch.start()
        self.addCleanup(window_patch.stop)

    def test_poll_only_reads_and_registration_has_explicit_guide_and_bezel_paths(self):
        ui = self.controls
        for _ in range(3):
            ui.refresh()
        self.controller.request_calibration.assert_not_called()
        self.controller.use_printed.assert_not_called()
        self.assertTrue(ui.register())
        self.controller.request_calibration.assert_called_once_with(aligned_to_guides=True)
        self.assertEqual(ui.register_button.configure.call_args.kwargs["state"], "disabled")
        self.assertEqual(ui.cancel_button.configure.call_args.kwargs["state"], "normal")
        self.assertTrue(ui.cancel_registration())
        self.controller.cancel_calibration.assert_called_once()
        self.assertTrue(ui.register_bezel())
        self.assertEqual(self.controller.request_calibration.call_args.args, ())
        self.assertEqual(self.controller.request_calibration.call_args.kwargs, {})
        self.camera.assert_not_called()
        self.window.assert_not_called()

    def test_missing_profile_and_unavailable_or_busy_disable_registration(self):
        ui = self.controls
        for changes in ({"available": False}, {"pending": True}):
            self.controller.value.update({"available": True, "pending": False, **changes})
            ui.refresh()
            self.assertFalse(ui.register())
        self.controller.value.update(available=True, pending=False)
        ui._can_change.return_value = False
        ui.refresh()
        self.assertFalse(ui.register())
        ui._can_change.return_value = True
        self.store.select_profile("display", None)
        ui.refresh()
        self.assertIn("Save a display profile", ui.message.get())
        self.assertFalse(ui.register())
        self.controller.request_calibration.assert_not_called()

    def test_auth_revoked_after_enabled_is_checked_before_every_action(self):
        ui = self.controls
        ui._require_access.side_effect = PermissionError("Sign in")
        self.assertFalse(ui.register())
        self.assertFalse(ui.register_bezel())
        self.assertFalse(ui.apply_mode())
        self.assertFalse(ui.cancel_registration())
        self.assertFalse(ui.open_printable_sheet())
        self.controller.request_calibration.assert_not_called()
        self.controller.cancel_calibration.assert_not_called()
        self.controller.use_printed.assert_not_called()
        ui._open_sheet.assert_not_called()
        self.assertEqual(ui.message.get(), "Sign in")

    def test_toggle_is_only_a_draft_until_apply_and_pending_disables_it(self):
        ui = self.controls
        self.controller.value["calibrated"] = True
        ui.refresh()
        ui.use_paper.set(True)
        ui._edited()
        ui.refresh()
        self.controller.use_printed.assert_not_called()
        self.assertTrue(ui.use_paper.get())
        self.assertTrue(ui.apply_mode())
        self.controller.use_printed.assert_called_once_with(True)
        self.assertFalse(ui._dirty)
        self.assertTrue(ui.use_paper.get())
        self.controller.value.update(pending=True)
        ui.refresh()
        self.assertEqual(ui.apply_button.configure.call_args.kwargs["state"], "disabled")

    def test_cannot_enable_uncalibrated_layout_but_can_return_to_screen_mode(self):
        ui = self.controls
        ui.use_paper.set(True)
        ui._edited()
        self.assertFalse(ui.apply_mode())
        self.assertIn("Register", ui.message.get())
        self.controller.use_printed.assert_not_called()
        self.controller.value.update(mode="printed", calibrated=False)
        ui._dirty = False
        ui.refresh()
        ui.use_paper.set(False)
        ui._edited()
        self.assertTrue(ui.apply_mode())
        self.controller.use_printed.assert_called_once_with(False)

    def test_profile_change_invalidates_draft_and_does_not_apply_to_new_profile(self):
        ui = self.controls
        self.controller.value["calibrated"] = True
        ui.refresh()
        ui.use_paper.set(True)
        ui._edited()
        replacement = self.store.upsert_profile("display", {"name": "Other TV", "settings": {
            "displayDiagonalInches": 32}}, select=True)
        self.controller.value.update(profileId=replacement["id"], profileName=replacement["name"])
        self.assertFalse(ui.apply_mode())
        self.assertIn("profile changed", ui.message.get())
        self.assertFalse(ui._dirty)
        self.controller.use_printed.assert_not_called()

    def test_stale_backend_profile_cannot_register_current_profile(self):
        ui = self.controls
        self.controller.value["profileId"] = "previous-profile"
        ui.refresh()
        self.assertIn("updating", ui.message.get())
        self.assertFalse(ui.register())
        self.controller.request_calibration.assert_not_called()

    def test_backend_error_remains_visible_and_draft_is_preserved(self):
        ui = self.controls
        self.controller.value["calibrated"] = True
        ui.refresh()
        ui.use_paper.set(True)
        ui._edited()
        self.controller.use_printed.side_effect = OSError("Could not save profile")
        self.assertFalse(ui.apply_mode())
        ui.refresh()
        self.assertEqual(ui.message.get(), "Could not save profile")
        self.assertTrue(ui.use_paper.get())
        self.assertTrue(ui._dirty)

    def test_matching_sheet_is_created_from_profile_and_live_context_only_after_explicit_action(self):
        ui = self.controls
        sheet = Path(self.temp.name) / "sheet.pdf"
        sheet.write_bytes(b"fake PDF")
        with patch("printed_marker_controls.data_path", return_value=sheet) as output:
            ui.refresh()
            ui._open_sheet.assert_not_called()
            ui._sheet_builder.assert_not_called()
            self.assertTrue(ui.open_printable_sheet())
        output.assert_called_once_with("Printouts/sarween-printed-markers-a4.pdf")
        ui._sheet_parameters.assert_called_once_with(self.display, self.controller.view_context.return_value)
        ui._sheet_builder.assert_called_once_with(sheet, **ui._sheet_parameters.return_value)
        ui._open_sheet.assert_called_once_with(sheet.resolve())

    def test_matching_sheet_requires_display_profile_and_current_foundry_view(self):
        ui = self.controls
        self.controller.view_context.return_value = {"ready": False}
        self.assertFalse(ui.open_printable_sheet())
        self.assertIn("full screen", ui.message.get())
        self.store.select_profile("display", None)
        self.assertFalse(ui.open_printable_sheet())
        self.assertIn("display profile", ui.message.get())
        ui._sheet_builder.assert_not_called()
        ui._open_sheet.assert_not_called()

    def test_matching_sheet_action_uses_real_parameter_validation_and_pdf_writer(self):
        ui = self.controls
        ui._sheet_builder = ui._sheet_parameters = None
        self.controller.view_context.return_value["view"]["registration"] = {
            "left": 28, "top": 28, "right": 1892, "bottom": 1052}
        sheet = Path(self.temp.name) / "real-sheet.pdf"
        with patch("printed_marker_controls.data_path", return_value=sheet):
            self.assertTrue(ui.open_printable_sheet())
        self.assertTrue(sheet.read_bytes().startswith(b"%PDF-"))
        ui._open_sheet.assert_called_once_with(sheet.resolve())

    def test_profile_change_during_generation_does_not_open_wrong_size_sheet(self):
        ui = self.controls
        sheet = Path(self.temp.name) / "sheet.pdf"
        sheet.write_bytes(b"fixture")

        def build(path, **_options):
            self.store.rename_profile("display", self.display["id"], "Changed TV")
            return {"path": str(path)}

        ui._sheet_builder.side_effect = build
        with patch("printed_marker_controls.data_path", return_value=sheet):
            self.assertFalse(ui.open_printable_sheet())
        self.assertIn("profile changed", ui.message.get())
        ui._open_sheet.assert_not_called()
