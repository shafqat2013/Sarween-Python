"""Bundle overrides must support the same inline startup entry points."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


def load_override(name):
    path = Path(__file__).resolve().parents[1] / "app_bundle_overrides" / (name + ".py")
    spec = importlib.util.spec_from_file_location("inline_bundle_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InlineBundleStartupTests(unittest.TestCase):
    def test_bundle_applies_explicit_hardware_without_a_popup_or_camera_probe(self):
        module = load_override("setup")
        store = Mock()
        store.active_profile.return_value = None
        displays = [{"index": 2, "width": 1920, "height": 1080}]
        cameras = [{"index": 7, "model": "OBS", "resolution": "Not opened"}]
        selection = {"display_index": 2, "webcam_index": 7, "mode": "foundry",
                     "map_path": None, "camera_profile_id": None}
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(module, "CONFIG_FILE", str(Path(directory) / "hardware.json")), \
                patch.object(module, "initialize_user_data"), \
                patch.object(module, "_profile_store", return_value=store), \
                patch.object(module, "detect_setup", side_effect=AssertionError("discovery")), \
                patch.object(module, "unified_selection_window", side_effect=AssertionError("popup")), \
                patch.object(module.cv2, "VideoCapture", side_effect=AssertionError("camera")):
            chosen = module.initialize(selection=selection, displays=displays, webcams=cameras)
            self.assertEqual(chosen, (displays[0], cameras[0], "foundry"))
            saved = json.loads(Path(module.CONFIG_FILE).read_text())
            self.assertIsNone(saved["map_path"])
            self.assertEqual(saved["webcam_index"], 7)
            self.assertTrue(callable(module.HardwareSelectionPanel))
            self.assertTrue(callable(module.detect_webcams_passive))

    def test_bundle_uses_ready_scene_without_the_legacy_foundry_wait_window(self):
        module = load_override("calibration")
        with patch.object(module, "initialize_user_data"), \
                patch.object(module, "_get_mode", return_value="foundry"), \
                patch.object(module, "_foundry_wait_for_scene_grid", side_effect=AssertionError("popup")), \
                patch.object(module.s, "grid_cols", None, create=True), \
                patch.object(module.s, "grid_rows", None, create=True):
            module.calibrate(scene_info={"sceneId": "test", "sceneW": 1920,
                                         "sceneH": 1080, "gridPx": 40})
            self.assertEqual((module.s.grid_cols, module.s.grid_rows), (48, 27))
