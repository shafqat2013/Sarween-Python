"""Phone setup contracts for source and bundle, without hardware or Tk windows."""
import ast
from contextlib import ExitStack, contextmanager
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import setup
from camera_sources import PHONE_SOURCE, phone_camera_inventory
from hardware_profiles import HardwareProfiles
from tests.test_hardware_selection import DISPLAYS, Variable, Widget


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("sarween_bundle_setup_phone_test", ROOT / "app_bundle_overrides/setup.py")
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)
PHONE = phone_camera_inventory(devices=[{"id": "test-usb-iphone", "transport": "usb", "label": "Test iPhone"}])[0]
SOURCE = PHONE["camera_source"]
NATIVE = {"index": 7, "model": "OBS Virtual Camera", "resolution": "Not opened"}


@contextmanager
def environment(module):
    with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
        store = HardwareProfiles(Path(folder) / "profiles.json")
        original = store.upsert_profile("camera", {"name": "Existing OBS profile", "sourceHint": "OBS"}, select=True)
        store.save_camera_calibration(original["id"], [[1200, 0, 540], [0, 1200, 960], [0, 0, 1]],
                                      [0, 0, 0, 0], [1080, 1920], rms=.2)
        original = store.get_profile("camera", original["id"])
        frame = module.ttk.Frame
        for name in ("__init__", "columnconfigure", "grid", "pack", "bind", "update_idletasks"):
            stack.enter_context(patch.object(frame, name, getattr(Widget, name)))
        stack.enter_context(patch.multiple(module.ttk, Frame=Widget, Label=Widget,
            Combobox=Widget, Checkbutton=Widget, Button=Widget))
        stack.enter_context(patch.multiple(module.tk, StringVar=Variable, BooleanVar=Variable,
            Tk=Mock(side_effect=AssertionError("No native root")),
            Toplevel=Mock(side_effect=AssertionError("No native windows"))))
        stack.enter_context(patch.object(module, "CONFIG_FILE", str(Path(folder) / "hardware.json")))
        stack.enter_context(patch.object(module, "_profile_store", return_value=store))
        stack.enter_context(patch.object(module, "_list_map_files", return_value=[]))
        stack.enter_context(patch.object(module, "phone_camera_inventory", return_value=[PHONE]))
        stack.enter_context(patch.object(module, "open_camera_capture", side_effect=AssertionError("No implicit camera open")))
        stack.enter_context(patch.object(module.cv2, "VideoCapture", side_effect=AssertionError("No native camera open")))
        stack.enter_context(patch.multiple(module, selected_display=None, selected_webcam=None,
                                          selected_mode=None, selected_map_path=None))
        yield store, original


def selection(index, profile_id=None, **extra):
    return {"display_index": DISPLAYS[0]["index"], "webcam_index": index,
            "mode": "foundry", "map_path": None, "camera_profile_id": profile_id,
            "camera_profile_confirmed": True, **extra}


class PhoneSetupTests(unittest.TestCase):
    def test_selection_uses_inventory_identity_and_native_shape_remains_compatible(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                forged = {"type": PHONE_SOURCE, "deviceId": "another-phone"}
                chosen = module._validate_selection(selection(PHONE["index"], camera_source=forged),
                    DISPLAYS, [PHONE, NATIVE], store=store)
                self.assertEqual(chosen["camera_source"], SOURCE)
                native = module._validate_selection(selection(7, camera_source=SOURCE), DISPLAYS, [PHONE, NATIVE], store=store)
                self.assertNotIn("camera_source", native)
                with self.assertRaises(ValueError):
                    module._validate_selection(selection(-1), DISPLAYS, [{**NATIVE, "index": -1}], store=store)

    def test_phone_refuses_obs_calibration_even_after_manual_confirmation(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, original):
                with self.assertRaisesRegex(ValueError, "different feed"):
                    module._validate_selection(selection(PHONE["index"], original["id"]), DISPLAYS, [PHONE], store=store)
                self.assertEqual(store.active_profile("camera")["id"], original["id"])

    def test_native_input_refuses_phone_profile_even_after_manual_confirmation(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                phone_profile = module.create_phone_profile(store, SOURCE)
                with self.assertRaisesRegex(ValueError, "direct USB"):
                    module._validate_selection(selection(7, phone_profile["id"]), DISPLAYS, [NATIVE], store=store)
                panel = module.HardwareSelectionPanel(Widget(), store=store)
                panel.set_inventory(DISPLAYS, [NATIVE])
                panel.profile_picker.select_id(phone_profile["id"], persist=False, notify=False)
                panel.profile_confirmed.set(True)
                panel._update_state()
                self.assertFalse(panel.ready)
                self.assertEqual(panel.calibrate_button.options["state"], "disabled")
                with patch("calibration.calibrate_webcam") as calibrate:
                    self.assertFalse(panel.calibrate_camera())
                    calibrate.assert_not_called()

    def test_calibration_entry_point_cannot_overwrite_phone_profile_from_native_pixels(self):
        import calibration
        spec = importlib.util.spec_from_file_location("sarween_bundle_calibration_phone_test",
            ROOT / "app_bundle_overrides/calibration.py")
        bundle_calibration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle_calibration)
        for module in (calibration, bundle_calibration):
            with self.subTest(module=module.__name__), environment(setup) as (store, _):
                profile = setup.create_phone_profile(store, SOURCE)
                before = store.get_profile("camera", profile["id"])
                with patch("hardware_profiles.HardwareProfiles", return_value=store), \
                     patch("camera_sources.open_camera_capture", side_effect=AssertionError("No camera may open")) as capture:
                    with self.assertRaisesRegex(ValueError, "separate profiles"):
                        module.calibrate_webcam(7, profile_id=profile["id"], camera_source={"type": "opencv", "index": 7})
                    capture.assert_not_called()
                self.assertEqual(store.get_profile("camera", profile["id"]), before)

    def test_phone_profile_button_drafts_new_uncalibrated_profile_without_applying_it(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, original):
                panel = module.HardwareSelectionPanel(Widget(), store=store)
                panel.set_inventory(DISPLAYS, [PHONE, NATIVE])
                self.assertEqual(panel.create_phone_profile_button.options["state"], "normal")
                self.assertNotIn(str(PHONE["index"]), panel.webcam_combo.options["values"][0])
                self.assertFalse(panel.ready)
                created = panel.create_iphone_profile()
                self.assertNotEqual(created["id"], original["id"])
                self.assertIsNone(created.get("calibration"))
                self.assertFalse(created["bindingVerified"])
                self.assertEqual(created["deviceLabel"], SOURCE["deviceId"])
                self.assertEqual(panel.profile_picker.selected_id(), created["id"])
                self.assertEqual(store.active_profile("camera")["id"], original["id"])
                self.assertFalse(panel.profile_confirmed.get())
                panel.profile_confirmed.set(True)
                module.apply_selection(panel.selection(), DISPLAYS, [PHONE, NATIVE])
                self.assertEqual(store.active_profile("camera")["id"], created["id"])
                self.assertEqual(module.load_last_selection()["camera_source"], SOURCE)
                panel.webcam_combo.current(1)
                panel._camera_changed()
                self.assertEqual(panel.create_phone_profile_button.options["state"], "disabled")
                self.assertFalse(panel.create_iphone_profile())

    def test_profile_creation_respects_revoked_access(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                access = Mock()
                panel = module.HardwareSelectionPanel(Widget(), store=store, access=access)
                panel.set_inventory(DISPLAYS, [PHONE])
                before = store.list_profiles("camera")
                access.require.side_effect = PermissionError("Access revoked")
                self.assertFalse(panel.create_iphone_profile())
                self.assertEqual(store.list_profiles("camera"), before)

    def test_phone_calibration_is_gated_and_passes_structured_source_explicitly(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                panel = module.HardwareSelectionPanel(Widget(), store=store)
                panel.set_inventory(DISPLAYS, [PHONE])
                with patch("calibration.calibrate_webcam", return_value={"saved": True}) as calibrate:
                    self.assertFalse(panel.calibrate_camera())
                    calibrate.assert_not_called()
                    created = panel.create_iphone_profile()
                    self.assertTrue(panel.calibrate_camera())
                    calibrate.assert_called_once_with(PHONE["index"], profile_id=created["id"], access=None,
                        display=DISPLAYS[0], displays=DISPLAYS, camera_source=SOURCE)

    def test_phone_source_survives_partial_saves_and_native_selection_clears_it(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module):
                module.save_last_selection(webcam_index=PHONE["index"], camera_source=SOURCE)
                module.save_last_selection(display_index=0)
                self.assertEqual(module.load_last_selection()["camera_source"], SOURCE)
                before = Path(module.CONFIG_FILE).read_bytes()
                with self.assertRaises(ValueError):
                    module.save_last_selection(camera_source={"type": PHONE_SOURCE, "deviceId": "wrong-phone"})
                self.assertEqual(Path(module.CONFIG_FILE).read_bytes(), before)
                module.save_last_selection(webcam_index=7)
                self.assertNotIn("camera_source", module.load_last_selection())

    def test_phone_waiting_preview_is_cancellable_and_releases_capture(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module), ExitStack() as stack:
                cap = Mock(isOpened=Mock(return_value=True), read=Mock(return_value=(False, None)),
                           status={"message": "Open Sarween Camera on your iPhone"})
                factory = stack.enter_context(patch.object(module, "open_camera_capture", return_value=cap))
                for name in ("namedWindow", "resizeWindow", "moveWindow", "putText", "imshow", "destroyWindow"):
                    stack.enter_context(patch.object(module.cv2, name))
                stack.enter_context(patch.object(module.cv2, "waitKey", return_value=27))
                module.preview_webcam_device(PHONE["index"], camera_source=SOURCE)
                factory.assert_called_once_with(PHONE["index"], source=SOURCE)
                cap.release.assert_called_once()

    def test_shared_effective_definitions_match_without_removing_bundle_specific_behavior(self):
        def effective(path):
            result = {}
            for node in ast.parse(path.read_text()).body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    result[node.name] = node
            return result
        source, bundle = effective(ROOT / "setup.py"), effective(ROOT / "app_bundle_overrides/setup.py")
        for name in ("save_last_selection", "_phone_inventory", "detect_webcams_backup", "detect_webcams_passive",
                     "_preview_phone_camera", "_validate_selection", "HardwareSelectionPanel", "apply_selection"):
            with self.subTest(definition=name):
                self.assertEqual(ast.dump(source[name]), ast.dump(bundle[name]))
        self.assertIn("_prime_camera_permission", bundle)
        self.assertIn("app lifetime", ast.get_docstring(bundle["_get_persistent_root"]))
        self.assertIn("destroyAllWindows", ast.unparse(bundle["preview_webcam_device"]))
        self.assertIn("< 3", ast.unparse(bundle["preview_webcam_device"]))
        self.assertIn("destroyWindow", ast.unparse(source["preview_webcam_device"]))


class PhoneInventoryTests(unittest.TestCase):
    def test_passive_inventory_appends_usb_phone_without_opening_any_camera(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), \
                 patch.object(module.sys, "platform", "darwin"), \
                 patch.object(module, "_avfoundation_video_devices", return_value=[Mock(localizedName=Mock(return_value="OBS"))]), \
                 patch.object(module, "phone_camera_inventory", return_value=[PHONE]), \
                 patch.object(module.cv2, "VideoCapture", side_effect=AssertionError("No camera probes")), \
                 patch.object(module, "open_camera_capture", side_effect=AssertionError("No phone app connection")):
                rows = module.detect_webcams_passive()
                self.assertEqual([row["index"] for row in rows], [0, PHONE["index"]])
                self.assertTrue(all(row["availability"] == "not_opened" for row in rows))

    def test_usb_service_failure_preserves_native_choices(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), \
                 patch.object(module.sys, "platform", "darwin"), \
                 patch.object(module, "_avfoundation_video_devices", return_value=[Mock(localizedName=Mock(return_value="OBS"))]), \
                 patch.object(module, "phone_camera_inventory", side_effect=OSError("No daemon")), \
                 patch.object(module.cv2, "VideoCapture", side_effect=AssertionError("No camera probes")):
                self.assertEqual([row["model"] for row in module.detect_webcams_passive()], ["OBS"])

    def test_backup_probes_native_indices_only_then_appends_phone(self):
        for module in (setup, BUNDLE):
            opened = []
            def capture(index):
                opened.append(index)
                return Mock(isOpened=Mock(return_value=index == 2), get=Mock(return_value=720))
            with self.subTest(module=module.__name__), \
                 patch.object(module.cv2, "VideoCapture", side_effect=capture), \
                 patch.object(module, "phone_camera_inventory", return_value=[PHONE]), \
                 patch.object(module, "open_camera_capture", side_effect=AssertionError("No phone app connection")):
                rows = module.detect_webcams_backup()
                self.assertEqual(opened, list(range(6)))
                self.assertEqual([row["index"] for row in rows], [2, PHONE["index"]])


if __name__ == "__main__":
    unittest.main()
