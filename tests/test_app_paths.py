import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import app_paths as paths


ROOT = Path(__file__).resolve().parents[1]


class UserDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.legacy = self.base / "old project"
        self.legacy.mkdir()
        self.data = self.base / "Application Support" / "Sarween"
        override = patch.dict(os.environ, {"SARWEEN_DATA_DIR": str(self.data)})
        override.start()
        self.addCleanup(override.stop)

    def source(self, name, value):
        target = self.legacy / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return target

    def migrate(self):
        return paths.initialize_user_data([self.legacy])

    def test_path_lookup_and_dry_run_do_not_write(self):
        self.source("combo_profiles.json", {"red": {"lab": [1, 2, 3]}})
        self.assertEqual(paths.data_path("combo_profiles.json"), self.data / "combo_profiles.json")
        plan = paths.migration_plan([self.legacy])
        self.assertEqual(len(plan["files"]), 1)
        self.assertFalse(self.data.exists())
        with self.assertRaises(ValueError):
            paths.data_path("../outside")
        with patch.dict(os.environ, {"SARWEEN_DATA_DIR": "relative"}):
            with self.assertRaises(ValueError):
                paths.data_dir()

    def test_mac_default_is_independent_of_cwd_and_bundle(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, "platform", "darwin"), \
             patch.object(Path, "home", return_value=self.base), \
             patch.object(sys, "_MEIPASS", str(self.legacy), create=True):
            self.assertEqual(paths.data_dir(), self.base / "Library/Application Support/Sarween")
            self.assertEqual(paths.resource_path("maps/demo.jpg"), str(self.legacy / "maps/demo.jpg"))
        self.assertFalse((self.base / "Library").exists())

    def test_copies_byte_for_byte_preserves_sources_and_records_hashes(self):
        self.source("combo_profiles.json", {"red": {"lab": [1, 2, 3]}})
        self.source("combo_profiles.json.bak", b"original backup")
        self.source("camera_matrix.npy", b"opaque matrix")
        self.source("mini_captures/nested/scan.png", b"scan image")
        self.source("sarween_rec_old.mp4", b"recordings stay in place")
        originals = {p.relative_to(self.legacy): p.read_bytes() for p in self.legacy.rglob("*") if p.is_file()}
        result = self.migrate()
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["files"]), 4)
        for entry in result["files"]:
            self.assertEqual((self.data / entry["path"]).read_bytes(), originals[Path(entry["path"])])
            self.assertEqual(entry["sha256"], entry["destinationSha256"])
            self.assertEqual(entry["status"], "copied")
        for name, value in originals.items():
            self.assertEqual((self.legacy / name).read_bytes(), value)
        self.assertFalse((self.data / "sarween_rec_old.mp4").exists())

    def test_existing_user_data_wins_and_updates_never_reimport(self):
        self.source("combo_profiles.json", {"old": 1})
        self.source("mini_library.json", {"minis": {}})
        paths.atomic_write_json(self.data / "combo_profiles.json", {"new": 2})
        result = self.migrate()
        record = next(e for e in result["files"] if e["path"] == "combo_profiles.json")
        self.assertEqual(record["status"], "preserved-existing")
        self.assertEqual(json.loads((self.data / "combo_profiles.json").read_text()), {"new": 2})
        (self.data / "mini_library.json").unlink()
        self.source("combo_profiles.json", {"later-old-copy": 3})
        self.assertEqual(self.migrate(), result)
        self.assertFalse((self.data / "mini_library.json").exists())
        self.assertEqual(json.loads((self.data / "combo_profiles.json").read_text()), {"new": 2})

    def test_restart_after_copy_failure_preserves_complete_files(self):
        self.source("combo_profiles.json", {"red": 1})
        self.source("mini_library.json", {"minis": {}})
        link = os.link
        count = 0
        def interrupted(source, destination):
            nonlocal count
            count += 1
            if count == 2:
                raise OSError("simulated disk failure")
            return link(source, destination)
        with patch.object(paths.os, "link", side_effect=interrupted):
            with self.assertRaises(OSError):
                self.migrate()
        self.assertFalse((self.data / paths.MIGRATION_FILE).exists())
        self.assertEqual(list(self.data.glob("*.tmp")), [])
        self.assertEqual(json.loads((self.data / "combo_profiles.json").read_text()), {"red": 1})
        self.assertTrue(self.migrate()["complete"])
        self.assertTrue((self.data / "mini_library.json").exists())

    def test_corrupt_input_is_not_imported_or_replaced(self):
        self.source("combo_profiles.json", b"invalid JSON")
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertFalse((self.data / paths.MIGRATION_FILE).exists())
        self.assertFalse((self.data / "combo_profiles.json").exists())

    def test_corrupt_destination_stops_without_touching_it(self):
        self.source("combo_profiles.json", {"old": 1})
        self.data.mkdir(parents=True)
        target = self.data / "combo_profiles.json"
        target.write_text("invalid JSON")
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertEqual(target.read_text(), "invalid JSON")

    def test_isolated_override_does_not_import_checkout(self):
        with patch.object(paths, "resource_root", return_value=self.legacy):
            self.source("combo_profiles.json", {"old": 1})
            result = paths.initialize_user_data()
        self.assertEqual(result["files"], [])
        self.assertFalse((self.data / "combo_profiles.json").exists())

    def test_same_directory_and_symlinked_sources_are_rejected(self):
        with self.assertRaises(ValueError):
            paths.migration_plan([self.legacy], destination=self.legacy)
        external = self.base / "external.json"
        external.write_text("{}")
        (self.legacy / "combo_profiles.json").symlink_to(external)
        with self.assertRaises(ValueError):
            self.migrate()
        self.assertEqual(external.read_text(), "{}")

    def test_symlinked_capture_destination_is_not_followed(self):
        self.source("mini_captures/image.png", b"capture")
        self.data.mkdir(parents=True)
        external = self.base / "external"
        external.mkdir()
        (self.data / "mini_captures").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlinked migration destination"):
            self.migrate()
        self.assertEqual(list(external.iterdir()), [])

    def test_source_change_during_copy_is_detected(self):
        original = self.source("combo_profiles.json", {"old": 1})
        entry = paths.migration_plan([self.legacy])["files"][0]
        original.write_text('{"new": 2}')
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            paths._copy_new(entry, self.data / "combo_profiles.json")
        self.assertFalse((self.data / "combo_profiles.json").exists())

    def test_atomic_saves_keep_backup_and_survive_failed_replace(self):
        target = self.data / "combo_profiles.json"
        paths.atomic_write_json(target, {"red": 1})
        paths.atomic_write_json(target, {"red": 2})
        self.assertEqual(json.loads(target.with_suffix(".json.bak").read_text()), {"red": 1})
        with patch.object(paths.os, "replace", side_effect=OSError("failed write")):
            with self.assertRaises(OSError):
                paths.atomic_write_json(target, {"red": 3})
        self.assertEqual(json.loads(target.read_text()), {"red": 2})
        self.assertEqual(list(self.data.glob("*.tmp")), [])
        with self.assertRaises(ValueError):
            paths.atomic_write_json(target, {"invalid": float("nan")})

    def test_csv_absolute_and_relative_capture_paths_remain_usable(self):
        asset = self.source("mini_captures/a_hist.npy", b"histogram")
        self.migrate()
        copied = self.data / "mini_captures/a_hist.npy"
        asset.unlink()
        db = self.data / "mini_database.csv"
        self.assertEqual(paths.capture_asset_path(str(asset), db), str(copied))
        self.assertEqual(paths.capture_asset_path("mini_captures/a_hist.npy", db), str(copied))
        self.assertEqual(copied.read_bytes(), b"histogram")
        outside = self.base / "custom_hist.npy"
        self.assertEqual(paths.capture_asset_path(str(outside), db), str(outside))

    def test_saved_relative_map_paths_resolve_without_changing_config(self):
        self.source("maps/custom.png", b"map")
        original = self.source("hardware_config.json", {"map_path": "maps/custom.png"})
        self.migrate()
        self.assertEqual(Path(paths.saved_map_path("maps/custom.png")), (self.legacy / "maps/custom.png").resolve())
        self.assertEqual((self.data / "hardware_config.json").read_bytes(), original.read_bytes())

    def test_imports_do_not_create_data_or_open_hardware_and_saves_use_data_dir(self):
        code = '''import pathlib, os
import setup, calibration, cv_core, mini_tracking, mini_calibration, mini_library
import foundryoutput, v3_tracking, ui_preview
from app_paths import data_path
root = pathlib.Path(os.environ["SARWEEN_DATA_DIR"])
assert not root.exists(), "imports must not migrate/write"
assert setup.CONFIG_FILE == str(root / "hardware_config.json")
assert foundryoutput.MAP_PATH == root / "mini_token_map.json"
assert mini_library.LIBRARY_PATH == root / "mini_library.json"
assert v3_tracking._PROFILES_PATH == mini_calibration._PROFILES_PATH == root / "combo_profiles.json"
assert mini_tracking.DB_CSV == str(root / "mini_database.csv")
assert str(root) in calibration.BLENDED_OUTPUT_PATH
setup.save_last_selection(mode="foundry", webcam_index=2)
mini_library.save_library(mini_library.default_library())
foundryoutput.MINI_TO_TOKEN = {"red": "example"}
foundryoutput._save_mapping()
v3_tracking._save_profiles({"red": {"lab": [1, 2, 3]}})
assert {p.name for p in root.iterdir()} == {"hardware_config.json", "mini_library.json", "mini_token_map.json", "combo_profiles.json"}
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_migration_works_with_read_only_old_bundle_and_new_bundle_resources(self):
        profile = self.source("combo_profiles.json", {"red": {"lab": [1, 2, 3]}})
        profile.chmod(0o444)
        self.legacy.chmod(0o555)
        self.addCleanup(self.legacy.chmod, 0o755)
        with patch.object(sys, "_MEIPASS", str(self.legacy), create=True), \
             patch.object(sys, "frozen", True, create=True), \
             patch.dict(os.environ, {}, clear=True), patch.object(paths, "data_dir", return_value=self.data):
            result = paths.initialize_user_data()
            self.assertTrue(result["complete"])
            new = self.base / "new bundle"
            new.mkdir()
            with patch.object(sys, "_MEIPASS", str(new)):
                self.assertEqual(paths.initialize_user_data(), result)
                self.assertEqual(json.loads(paths.data_path("combo_profiles.json").read_text()), {"red": {"lab": [1, 2, 3]}})

    def test_frozen_setup_uses_shared_paths_instead_of_bundle_state(self):
        code = '''import os, pathlib, runpy, sys
from unittest.mock import Mock, patch
root = pathlib.Path(os.environ["SARWEEN_DATA_DIR"])
sys.frozen = True
sys._MEIPASS = str(root.parent / "application-resources")
with patch("tkinter.Tk", return_value=Mock()):
    setup = runpy.run_path("app_bundle_overrides/setup.py")
assert setup["CONFIG_FILE"] == str(root / "hardware_config.json")
assert setup["MAPS_DIR"] == str(pathlib.Path(sys._MEIPASS) / "maps")
assert not root.exists()
setup["save_last_selection"](mode="foundry", webcam_index=3)
assert (root / "hardware_config.json").is_file()
assert not pathlib.Path(sys._MEIPASS).exists()
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_live_startup_stops_before_services_when_migration_fails(self):
        import main
        with patch.object(main, "initialize_user_data", side_effect=ValueError("bad saved data")), \
             patch.object(main, "start_foundry_server_in_background") as server, \
             patch.object(main.s, "initialize") as setup:
            with self.assertRaisesRegex(ValueError, "bad saved data"):
                main.main()
            server.assert_not_called()
            setup.assert_not_called()

    def test_setup_reads_migrated_selection_at_runtime_not_import_time(self):
        import setup
        with patch.object(setup, "initialize_user_data"), \
             patch.object(setup, "detect_setup", return_value=([], [])), \
             patch.object(setup, "load_last_selection", return_value={"webcam_index": 3, "mode": "foundry"}), \
             patch.object(setup, "unified_selection_window", return_value=None) as window:
            with self.assertRaises(SystemExit):
                setup.initialize()
            self.assertEqual(window.call_args.kwargs["default_webcam_device_index"], 3)
            self.assertEqual(window.call_args.kwargs["default_mode"], "foundry")


if __name__ == "__main__":
    unittest.main()
