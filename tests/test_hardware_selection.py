"""Embedded hardware setup without cameras, native windows or relay sockets."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import setup
from hardware_profiles import HardwareProfiles


DISPLAYS = [{"index": 2, "name": "TV", "width": 1920, "height": 1080, "x": 1440, "y": 0},
            {"index": 0, "name": "Mac", "width": 1440, "height": 900, "x": 0, "y": 0}]
CAMERAS = [{"index": 7, "model": "OBS Virtual Camera", "resolution": "Not opened"},
           {"index": 1, "model": "Built in", "resolution": "Not opened"}]


class Variable:
    def __init__(self, _master=None, value="", **_kwargs):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, parent=None, **options):
        self.children, self.options, self.bindings = [], options, {}
        self.value = ""
        if parent is not None:
            parent.children.append(self)

    def configure(self, **options):
        self.options.update(options)

    def columnconfigure(self, *_args, **_kwargs):
        pass

    def grid(self, **options):
        self.layout = options

    def pack(self, **options):
        self.layout = options

    def bind(self, event, callback):
        self.bindings[event] = callback

    def set(self, value):
        self.value = value
        if self.options.get("textvariable") is not None:
            self.options["textvariable"].set(value)

    def get(self):
        variable = self.options.get("textvariable")
        return variable.get() if variable is not None else self.value

    def current(self, position=None):
        values = self.options.get("values", ())
        if position is not None:
            self.set(values[position])
        try:
            return list(values).index(self.get())
        except ValueError:
            return -1

    def update_idletasks(self):
        pass


class HardwareSelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.store = HardwareProfiles(self.folder / "profiles.json")
        self.profile = self.store.upsert_profile("camera", {"name": "iPhone", "lens": "wide"}, select=True)
        self.map_path = str(self.folder / "table.png")
        Path(self.map_path).write_bytes(b"fixture map")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        # Retain the real frame base class while replacing its Tcl operations.
        frame = setup.ttk.Frame
        for name in ("__init__", "columnconfigure", "grid", "pack", "bind", "update_idletasks"):
            self.stack.enter_context(patch.object(frame, name, getattr(Widget, name)))
        self.stack.enter_context(patch.multiple(setup.ttk, Frame=Widget, Label=Widget,
            Combobox=Widget, Checkbutton=Widget, Button=Widget))
        self.stack.enter_context(patch.multiple(setup.tk, StringVar=Variable, BooleanVar=Variable,
            Tk=Mock(side_effect=AssertionError("No native root")),
            Toplevel=Mock(side_effect=AssertionError("No Setup window"))))
        self.camera = self.stack.enter_context(patch.object(setup.cv2, "VideoCapture",
            side_effect=AssertionError("No implicit camera open")))
        self.scan = self.stack.enter_context(patch.object(setup, "detect_setup",
            side_effect=AssertionError("No implicit inventory scan")))
        self.stack.enter_context(patch.object(setup, "CONFIG_FILE", str(self.folder / "hardware.json")))
        self.stack.enter_context(patch.object(setup, "_list_map_files", return_value=[self.map_path]))
        self.stack.enter_context(patch.object(setup, "_profile_store", return_value=self.store))

    def panel(self, **kwargs):
        panel = setup.HardwareSelectionPanel(Widget(), store=self.store, **kwargs)
        panel.set_inventory(DISPLAYS, CAMERAS)
        return panel

    def test_embed_has_no_hardware_side_effects_and_named_profile_requires_confirmation(self):
        changed = Mock()
        panel = self.panel(changed=changed)
        self.assertFalse(panel.ready)
        self.assertIn("Confirm", panel.status.get())
        panel.profile_confirmed.set(True)
        panel._update_state()
        self.assertTrue(panel.ready)
        selection = panel.selection()
        self.assertEqual(selection["display_index"], 2)
        self.assertEqual(selection["webcam_index"], 7)
        self.assertEqual(selection["mode"], "foundry")
        self.assertIsNone(selection["map_path"])
        self.assertTrue(selection["camera_profile_confirmed"])
        self.assertFalse((self.folder / "hardware.json").exists())
        self.assertEqual(self.store.active_profile("camera")["id"], self.profile["id"])
        self.camera.assert_not_called()
        self.scan.assert_not_called()
        self.assertGreater(changed.call_count, 0)

    def test_camera_change_clears_profile_without_changing_saved_active_profile(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        panel.webcam_combo.current(1)
        panel._camera_changed()
        self.assertIsNone(panel.profile_picker.selected_id())
        self.assertFalse(panel.profile_confirmed.get())
        self.assertTrue(panel.ready)
        self.assertEqual(panel.selection()["webcam_index"], 1)
        self.assertEqual(self.store.active_profile("camera")["id"], self.profile["id"])

    def test_inventory_reorder_preserves_choice_but_missing_choice_does_not_switch_camera(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        panel.set_inventory(list(reversed(DISPLAYS)), list(reversed(CAMERAS)))
        self.assertTrue(panel.ready)
        self.assertEqual(panel.selection()["webcam_index"], 7)
        panel.set_inventory(DISPLAYS, [CAMERAS[1]])
        self.assertFalse(panel.ready)
        self.assertIn("camera", panel.status.get())
        self.assertFalse(panel.profile_confirmed.get())
        self.assertEqual(panel.webcam_combo.current(), -1)

    def test_profile_metadata_changed_externally_invalidates_confirmation(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        self.store.upsert_profile("camera", {"lens": "telephoto"}, profile_id=self.profile["id"])
        panel.refresh_profiles()
        self.assertFalse(panel.profile_confirmed.get())
        self.assertFalse(panel.ready)

    def test_changed_native_device_at_same_index_clears_confirmation(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        panel.set_inventory(DISPLAYS, [{**CAMERAS[0], "model": "Different phone"}, CAMERAS[1]])
        self.assertFalse(panel.profile_confirmed.get())
        self.assertFalse(panel.ready)

    def test_deleted_default_profile_clears_and_new_saved_profile_can_be_chosen(self):
        panel = self.panel()
        self.store.delete_profile("camera", self.profile["id"])
        panel.set_inventory(DISPLAYS, CAMERAS, defaults={"camera_profile_id": self.profile["id"]})
        self.assertIsNone(panel.profile_picker.selected_id())
        other = self.store.upsert_profile("camera", {"name": "New camera"})
        panel.set_inventory(DISPLAYS, CAMERAS, defaults={"camera_profile_id": other["id"]})
        self.assertEqual(panel.profile_picker.selected_id(), other["id"])
        self.assertFalse(panel.ready)

    def test_self_hosted_requires_a_real_available_map_foundry_does_not(self):
        panel = self.panel(defaults={"camera_profile_id": None, "mode": "self_hosted"})
        self.assertTrue(panel.ready)
        self.assertEqual(panel.selection()["map_path"], self.map_path)
        Path(self.map_path).unlink()
        panel._update_state()
        self.assertFalse(panel.ready)
        self.assertIn("map", panel.status.get())
        panel.mode_combo.set("Foundry")
        panel._update_state()
        self.assertTrue(panel.ready)
        self.assertEqual(panel.map_combo.options["state"], "disabled")

    def test_passive_fallback_note_is_visible_without_claiming_open_camera(self):
        panel = self.panel(defaults={"camera_profile_id": None})
        fallback = {**CAMERAS[0], "availability": "unverified", "inventory_note": "Preview saved camera first."}
        panel.set_inventory(DISPLAYS, [fallback])
        self.assertTrue(panel.ready)
        self.assertEqual(panel.status.get(), "Preview saved camera first.")

    def test_disabled_or_revoked_actions_cannot_open_camera(self):
        access = Mock()
        panel = self.panel(access=access)
        with patch.object(setup, "preview_webcam_device") as preview, \
             patch("calibration.calibrate_webcam") as calibrate:
            panel.set_enabled(False)
            self.assertFalse(panel.preview_camera())
            self.assertFalse(panel.calibrate_camera())
            panel.set_enabled(True)
            access.require.side_effect = PermissionError("Access revoked")
            self.assertFalse(panel.preview_camera())
            self.assertFalse(panel.calibrate_camera())
            preview.assert_not_called()
            calibrate.assert_not_called()
        self.assertIn("revoked", panel.status.get())

    def test_explicit_preview_and_calibration_use_selected_indices_and_access(self):
        access = Mock()
        panel = self.panel(access=access)
        with patch.object(setup, "preview_webcam_device") as preview, \
             patch("calibration.calibrate_webcam", return_value={"calibration": "saved"}) as calibrate:
            self.assertTrue(panel.preview_camera())
            preview.assert_called_once_with(7)
            self.assertTrue(panel.calibrate_camera())
            calibrate.assert_called_once_with(7, profile_id=self.profile["id"], access=access,
                display=DISPLAYS[0], displays=DISPLAYS)
        self.assertTrue(panel.profile_confirmed.get())
        self.assertTrue(panel.ready)

    def test_invalid_selection_does_not_write_or_change_globals(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        valid = panel.selection()
        for changes in ({"webcam_index": 99}, {"display_index": True}, {"mode": "invalid"},
                        {"camera_profile_confirmed": False}, {"camera_profile_id": "gone"},
                        {"mode": "self_hosted", "map_path": "/not/a/map"}):
            with self.subTest(changes=changes), patch.object(setup, "selected_mode", "old"):
                with self.assertRaises(ValueError):
                    setup.apply_selection({**valid, **changes}, DISPLAYS, CAMERAS)
                self.assertEqual(setup.selected_mode, "old")
                self.assertFalse((self.folder / "hardware.json").exists())
                self.assertEqual(self.store.active_profile("camera")["id"], self.profile["id"])

    def test_apply_and_explicit_initialize_bypass_modal_and_clear_stale_map(self):
        panel = self.panel()
        panel.profile_confirmed.set(True)
        setup.save_last_selection(mode="self_hosted", map_path=self.map_path)
        with patch.object(setup, "initialize_user_data"), \
             patch.object(setup, "unified_selection_window", side_effect=AssertionError("No modal")), \
             patch.multiple(setup, selected_display=None, selected_webcam=None,
                            selected_mode=None, selected_map_path=None):
            result = setup.initialize(selection=panel.selection(), displays=DISPLAYS, webcams=CAMERAS)
            self.assertEqual(result, (DISPLAYS[0], CAMERAS[0], "foundry"))
            self.assertIsNone(setup.selected_map_path)
            saved = setup.load_last_selection()
            self.assertIsNone(saved["map_path"])
            self.assertNotIn("camera_profile_confirmed", saved)
            self.assertEqual(saved["camera_profile_id"], self.profile["id"])
        self.camera.assert_not_called()
        self.scan.assert_not_called()

    def test_explicit_initialize_requires_supplied_inventory_and_permission(self):
        with patch.object(setup, "initialize_user_data"):
            with self.assertRaisesRegex(ValueError, "inventory"):
                setup.initialize(selection={"mode": "foundry"})
            access = Mock()
            access.require.side_effect = PermissionError("Sign in")
            with self.assertRaisesRegex(PermissionError, "Sign in"):
                setup.initialize(access, selection={}, displays=DISPLAYS, webcams=CAMERAS)
        self.camera.assert_not_called()
        self.scan.assert_not_called()

    def test_apply_self_hosted_map_and_failed_save_preserve_prior_active_profile(self):
        panel = self.panel(defaults={"camera_profile_id": None, "mode": "self_hosted"})
        with patch.multiple(setup, selected_display=None, selected_webcam=None,
                            selected_mode="prior", selected_map_path=None):
            with patch.object(setup, "save_last_selection", side_effect=OSError("Disk full")):
                with self.assertRaisesRegex(OSError, "Disk full"):
                    setup.apply_selection(panel.selection(), DISPLAYS, CAMERAS)
            self.assertEqual(setup.selected_mode, "prior")
            self.assertEqual(self.store.active_profile("camera")["id"], self.profile["id"])
            setup.apply_selection(panel.selection(), DISPLAYS, CAMERAS)
            self.assertEqual(setup.selected_map_path, self.map_path)
            self.assertEqual(setup.load_last_selection()["map_path"], self.map_path)
            self.assertIsNone(self.store.active_profile("camera"))


class PassiveInventoryTests(unittest.TestCase):
    def setUp(self):
        # These native-inventory cases deliberately have no attached phone.
        phone_inventory = patch.object(setup, "phone_camera_inventory", return_value=[])
        phone_inventory.start()
        self.addCleanup(phone_inventory.stop)

    def test_native_inventory_matches_opencv_video_muxed_and_unique_id_order(self):
        video = [Mock(uniqueID=Mock(return_value="20")), Mock(uniqueID=Mock(return_value="05"))]
        muxed = [Mock(uniqueID=Mock(return_value="10"))]
        devices = Mock(devicesWithMediaType_=Mock(side_effect=[video, muxed]))
        bundle = Mock(load=Mock(return_value=True))
        foundation = SimpleNamespace(NSBundle=Mock(bundleWithPath_=Mock(return_value=bundle)))
        objc = SimpleNamespace(lookUpClass=Mock(return_value=devices))
        with patch.dict("sys.modules", {"objc": objc, "Foundation": foundation}), \
             patch.object(setup.cv2, "VideoCapture", side_effect=AssertionError("No probes")):
            self.assertEqual(setup._avfoundation_video_devices(), [video[1], muxed[0], video[0]])
        self.assertEqual(devices.devicesWithMediaType_.call_args_list, [unittest.mock.call("vide"),
                                                                      unittest.mock.call("muxx")])
        objc.lookUpClass.assert_called_once_with("AVCaptureDevice")

    def test_passive_display_failure_does_not_run_profiler_or_probe_cameras(self):
        with patch.object(setup, "detect_displays_backup", side_effect=RuntimeError("No display API")), \
             patch.object(setup, "detect_webcams_passive", return_value=CAMERAS), \
             patch.object(setup.cv2, "VideoCapture", side_effect=AssertionError("No probes")), \
             patch.object(setup.subprocess, "run", side_effect=AssertionError("No subprocess")):
            self.assertEqual(setup.detect_setup(passive=True), ([], CAMERAS))

    def test_native_passive_enumeration_uses_names_without_opening_or_profiler(self):
        devices = [Mock(localizedName=Mock(return_value="OBS Virtual Camera")),
                   Mock(localizedName=Mock(return_value="MacBook Camera"))]
        with patch.object(setup.sys, "platform", "darwin"), \
             patch.object(setup, "_avfoundation_video_devices", return_value=devices), \
             patch.object(setup, "detect_displays_backup", return_value=DISPLAYS), \
             patch.object(setup.cv2, "VideoCapture", side_effect=AssertionError("No probes")), \
             patch.object(setup.subprocess, "run", side_effect=AssertionError("No profiler")):
            displays, cameras = setup.detect_setup(passive=True)
        self.assertEqual(displays, DISPLAYS)
        self.assertEqual([camera["model"] for camera in cameras], ["OBS Virtual Camera", "MacBook Camera"])
        self.assertEqual([camera["index"] for camera in cameras], [0, 1])
        self.assertTrue(all(camera["resolution"] == "Not opened" for camera in cameras))

    def test_missing_framework_only_offers_unverified_saved_index(self):
        with patch.object(setup.sys, "platform", "darwin"), \
             patch.object(setup, "_avfoundation_video_devices", side_effect=ImportError("No objc")), \
             patch.object(setup, "load_last_selection", return_value={"webcam_index": 4}), \
             patch.object(setup.cv2, "VideoCapture", side_effect=AssertionError("No probes")):
            cameras = setup.detect_webcams_passive()
        self.assertEqual(len(cameras), 1)
        self.assertEqual(cameras[0]["index"], 4)
        self.assertEqual(cameras[0]["availability"], "unverified")
        self.assertIn("inventory is unavailable", cameras[0]["inventory_note"])

    def test_no_native_devices_is_empty_without_guessing_or_using_stale_saved_index(self):
        with patch.object(setup.sys, "platform", "darwin"), \
             patch.object(setup, "_avfoundation_video_devices", return_value=[]), \
             patch.object(setup, "load_last_selection", return_value={"webcam_index": 4}):
            self.assertEqual(setup.detect_webcams_passive(), [])

    def test_non_mac_without_saved_index_does_not_guess_camera_zero(self):
        with patch.object(setup.sys, "platform", "linux"), \
             patch.object(setup, "load_last_selection", return_value={}):
            self.assertEqual(setup.detect_webcams_passive(), [])

    def test_legacy_inventory_still_uses_active_backend_by_default(self):
        with patch.object(setup, "detect_displays_backup", return_value=DISPLAYS), \
             patch.object(setup, "detect_webcams_backup", return_value=CAMERAS) as active, \
             patch.object(setup, "detect_webcams_passive") as passive:
            self.assertEqual(setup.detect_setup(), (DISPLAYS, CAMERAS))
        active.assert_called_once()
        passive.assert_not_called()


if __name__ == "__main__":
    unittest.main()
