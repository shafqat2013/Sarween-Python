from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import UUID

import hardware_profiles as profiles

try:
    import numpy as np
except ImportError:
    np = None


MATRIX = [[900, 0, 640], [0, 900, 360], [0, 0, 1]]
DISTORTION = [0.1, -0.02, 0, 0, 0]
DISPLAY_SETTINGS = {"targetGridInches": 1, "displayDiagonalInches": 54, "physicalGridCalibration": 1.03,
                    "viewportMarkersEnabled": True, "viewportMarkerSize": 120,
                    "viewportMarkerInset": 8, "physicalGridAutoApply": True}


class HardwareProfileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "user data"
        self.path = self.root / "hardware_profiles.json"
        self.store = profiles.HardwareProfiles(self.path)

    def camera(self, name="iPhone through OBS", **values):
        return self.store.upsert_profile("camera", {"name": name, "sourceHint": "USB → OBS Virtual Camera",
                                                   "lens": "Wide", "zoom": "1.0x", **values})

    def calibrated_camera(self, name="iPhone through OBS", *, select=True):
        camera = self.camera(name)
        camera = self.store.save_camera_calibration(camera["id"], MATRIX, DISTORTION, [1280, 720], rms=0.24)
        if select:
            self.store.select_profile("camera", camera["id"])
        return camera

    def test_import_lookup_and_missing_legacy_import_do_not_create_files(self):
        with patch.dict(os.environ, {"SARWEEN_DATA_DIR": str(self.root)}):
            default = profiles.HardwareProfiles()
            self.assertEqual(default.path, self.path)
            self.assertEqual(default.snapshot()["generation"], 0)
            self.assertEqual(default.list_profiles("camera"), [])
            self.assertIsNone(default.active_profile("display"))
            self.assertIsNone(default.import_legacy_camera())
            self.assertIsNone(default.resolve_camera_calibration([1280, 720])[0])
        self.assertFalse(self.root.exists())

    def test_namespaces_stable_ids_rename_selection_and_explicit_no_calibration(self):
        camera = self.calibrated_camera("Tabletop")
        display = self.store.upsert_profile("display", {"name": "Tabletop", "settings": DISPLAY_SETTINGS}, select=True)
        self.assertEqual(str(UUID(camera["id"])), camera["id"])
        renamed = self.store.rename_profile("camera", camera["id"], "iPhone Wide 1x")
        self.assertEqual(renamed["id"], camera["id"])
        self.assertEqual(renamed["createdAt"], camera["createdAt"])
        self.assertTrue(renamed["bindingVerified"])
        self.assertEqual(self.store.active_profile("display")["id"], display["id"])
        self.assertIsNotNone(self.store.resolve_camera_calibration([1280, 720])[0])
        self.store.select_profile("camera", None)
        calibration, status = self.store.resolve_camera_calibration([1280, 720])
        self.assertIsNone(calibration)
        self.assertIn("No camera calibration selected", status)
        self.assertEqual(self.store.get_profile("camera", camera["id"])["calibration"]["imageSize"], [1280, 720])

    def test_duplicate_names_and_unknown_fields_are_rejected_without_replacing_profiles(self):
        camera = self.camera("Wide")
        for values in ({"name": " wide "}, {"name": "Wide\nOther"}, {"name": ""}, {"name": "Test", "unknown": 1}):
            with self.assertRaises(ValueError):
                self.store.upsert_profile("camera", values)
        with self.assertRaises(ValueError):
            self.store.upsert_profile("camera", {"name": "Missing"}, profile_id="does-not-exist")
        self.assertEqual(self.store.list_profiles("camera"), [camera])

    def test_display_profiles_preserve_54_inches_and_merge_only_explicit_settings(self):
        display = self.store.upsert_profile("display", {"name": "54-inch TV", "settings": {"displayDiagonalInches": 54}})
        self.assertEqual(display["settings"], {"displayDiagonalInches": 54})
        updated = self.store.upsert_profile("display", {"settings": {"targetGridInches": 1, "viewportMarkerSize": 160},
                                                        "targetDisplayId": "external-tv"}, profile_id=display["id"])
        self.assertEqual(updated["settings"], {"displayDiagonalInches": 54, "targetGridInches": 1, "viewportMarkerSize": 160})
        self.store.select_profile("display", display["id"])
        self.assertEqual(self.store.active_profile("display")["targetDisplayId"], "external-tv")
        self.assertEqual(self.store.active_profile("display")["settings"]["displayDiagonalInches"], 54)
        self.assertEqual(self.store.list_profiles("camera"), [])
        for invalid in ({}, {"viewportMarkerSize": 71}, {"physicalGridAutoApply": 1},
                        {"displayDiagonalInches": "54"}, {"unknown": 1}, {"targetGridInches": float("nan")}):
            with self.assertRaises(ValueError):
                self.store.upsert_profile("display", {"name": "Invalid", "settings": invalid})

    def test_calibration_requires_explicit_selection_verified_binding_and_exact_final_feed(self):
        camera = self.calibrated_camera(select=False)
        self.assertIsNone(self.store.resolve_camera_calibration([1280, 720])[0])
        self.store.select_profile("camera", camera["id"])
        calibration, status = self.store.resolve_camera_calibration([1280, 720])
        self.assertEqual(calibration["cameraMatrix"], MATRIX)
        self.assertEqual(calibration["distCoeffs"], DISTORTION)
        self.assertIn("Using camera calibration", status)
        for size in ([720, 1280], [1920, 1080], [0, 720], [1280.0, 720], [True, 720]):
            result, status = self.store.resolve_camera_calibration(size)
            self.assertIsNone(result)
            self.assertTrue(status)
        other = self.camera("Uncalibrated webcam")
        self.store.select_profile("camera", other["id"])
        self.assertIsNone(self.store.resolve_camera_calibration([1280, 720])[0])
        self.assertIsNotNone(self.store.resolve_camera_calibration([1280, 720], profile_id=camera["id"])[0])
        self.assertIn("missing", self.store.resolve_camera_calibration([1280, 720], profile_id="missing")[1])

    def test_camera_binding_changes_invalidate_calibration_but_rename_does_not(self):
        camera = self.calibrated_camera()
        self.store.rename_profile("camera", camera["id"], "Known camera")
        self.assertIsNotNone(self.store.resolve_camera_calibration([1280, 720])[0])
        for values in ({"lens": "Ultra Wide"}, {"zoom": "2x"}, {"transformNotes": "Rotated in OBS"},
                       {"resolution": [720, 1280]}, {"deviceLabel": "Other camera"}):
            updated = self.store.upsert_profile("camera", values, profile_id=camera["id"])
            self.assertFalse(updated["bindingVerified"])
            self.assertIsNotNone(updated["calibration"])
            self.assertIn("unverified", self.store.resolve_camera_calibration([1280, 720])[1])
            self.store.save_camera_calibration(camera["id"], MATRIX, DISTORTION, [1280, 720])

    def test_calibration_commit_rejects_changed_optics_but_allows_renaming(self):
        camera = self.camera()
        self.store.upsert_profile("camera", {"lens": "Telephoto"}, profile_id=camera["id"])
        changed = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "changed during calibration"):
            self.store.save_camera_calibration(camera["id"], MATRIX, DISTORTION, [1280, 720],
                                               expected_binding=camera)
        self.assertEqual(self.path.read_bytes(), changed)
        self.assertIsNone(self.store.get_profile("camera", camera["id"])["calibration"])
        current = self.store.get_profile("camera", camera["id"])
        self.store.rename_profile("camera", camera["id"], "Renamed camera")
        saved = self.store.save_camera_calibration(camera["id"], MATRIX, DISTORTION, [1280, 720],
                                                   expected_binding=current)
        self.assertEqual(saved["name"], "Renamed camera")
        self.assertTrue(saved["bindingVerified"])

    def test_invalid_calibration_cannot_replace_a_valid_one(self):
        camera = self.calibrated_camera()
        previous = self.path.read_bytes()
        for matrix, distortion, size, rms in (
            ([[1]], DISTORTION, [1280, 720], 0.1),
            ([[-1, 0, 640], [0, 1, 360], [0, 0, 1]], DISTORTION, [1280, 720], 0.1),
            (MATRIX, [0, 0, 0], [1280, 720], 0.1),
            (MATRIX, [0, 0, float("nan"), 0, 0], [1280, 720], 0.1),
            (MATRIX, DISTORTION, [0, 720], 0.1),
            (MATRIX, DISTORTION, [1280, 720], -1),
        ):
            with self.assertRaises(ValueError):
                self.store.save_camera_calibration(camera["id"], matrix, distortion, size, rms=rms)
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertIsNotNone(self.store.resolve_camera_calibration([1280, 720])[0])

    def test_metadata_editor_cannot_reverify_old_calibration_or_bypass_binding_invalidation(self):
        camera = self.calibrated_camera()
        self.store.upsert_profile("camera", {"lens": "Telephoto"}, profile_id=camera["id"])
        for values in ({"bindingVerified": True}, {"calibration": camera["calibration"]},
                       {"lens": "Wide", "calibration": camera["calibration"], "bindingVerified": True}):
            with self.assertRaises(ValueError):
                self.store.upsert_profile("camera", values, profile_id=camera["id"])
        self.assertIn("unverified", self.store.resolve_camera_calibration([1280, 720])[1])
        self.store.save_camera_calibration(camera["id"], MATRIX, DISTORTION, [1280, 720], rms=0.2)
        self.assertIsNotNone(self.store.resolve_camera_calibration([1280, 720])[0])

    def test_file_changes_refresh_cached_profiles_and_returned_objects_cannot_mutate_cache(self):
        camera = self.calibrated_camera()
        revision = self.store.revision()
        self.assertEqual(self.store.revision(), revision)
        with patch.object(self.store, "_read_file", wraps=self.store._read_file) as read:
            for _ in range(20):
                calibration, _status = self.store.resolve_camera_calibration([1280, 720])
                calibration["cameraMatrix"][0][0] = 1
            self.assertEqual(read.call_count, 0)
        self.assertEqual(self.store.resolve_camera_calibration([1280, 720])[0]["cameraMatrix"][0][0], 900)
        other = profiles.HardwareProfiles(self.path)
        other.rename_profile("camera", camera["id"], "Renamed elsewhere")
        self.assertNotEqual(self.store.revision(), revision)
        self.assertEqual(self.store.active_profile("camera")["name"], "Renamed elsewhere")
        value = json.loads(self.path.read_text())
        value["generation"] += 1
        value["cameras"][0]["calibration"]["distCoeffs"] = [0]
        self.path.write_text(json.dumps(value))
        calibration, status = self.store.resolve_camera_calibration([1280, 720])
        self.assertIsNone(calibration)
        self.assertIn("coefficients", status)

    def test_corrupt_registry_is_reported_and_never_overwritten(self):
        self.root.mkdir()
        self.path.write_bytes(b"not valid JSON")
        calibration, status = self.store.resolve_camera_calibration([1280, 720])
        self.assertIsNone(calibration)
        self.assertIn("unavailable", status)
        with self.assertRaises(ValueError):
            self.camera()
        self.assertEqual(self.path.read_bytes(), b"not valid JSON")

    def test_atomic_write_failure_keeps_previous_registry_and_active_selection(self):
        camera = self.calibrated_camera()
        previous = self.path.read_bytes()
        with patch.object(profiles, "atomic_write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.rename_profile("camera", camera["id"], "Lost rename")
        self.assertEqual(self.path.read_bytes(), previous)
        self.assertEqual(self.store.active_profile("camera")["name"], camera["name"])

    def test_parallel_instances_do_not_lose_profile_writes(self):
        def create(index):
            store = profiles.HardwareProfiles(self.path)
            return store.upsert_profile("camera", {"name": f"Camera {index}"})["id"]
        with ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(create, range(24)))
        self.assertEqual(len(set(ids)), 24)
        self.assertEqual(len(self.store.list_profiles("camera")), 24)
        self.assertEqual(self.store.snapshot()["generation"], 24)

    def test_separate_processes_share_the_atomic_registry_lock(self):
        script = "from hardware_profiles import HardwareProfiles; import sys; HardwareProfiles(sys.argv[1]).upsert_profile('camera', {'name': sys.argv[2]})"
        processes = [subprocess.Popen([sys.executable, "-c", script, str(self.path), f"Process {index}"],
                                     cwd=Path(__file__).resolve().parents[1], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                     for index in range(4)]
        for process in processes:
            _output, error = process.communicate(timeout=15)
            self.assertEqual(process.returncode, 0, error.decode())
        self.assertEqual(len(self.store.list_profiles("camera")), 4)
        self.assertEqual(self.store.snapshot()["generation"], 4)

    def test_deleting_the_active_profile_clears_selection_without_falling_back(self):
        selected = self.calibrated_camera()
        self.calibrated_camera("Other", select=False)
        self.store.delete_profile("camera", selected["id"])
        self.assertIsNone(self.store.active_profile("camera"))
        self.assertIsNone(self.store.resolve_camera_calibration([1280, 720])[0])
        self.assertEqual(len(self.store.list_profiles("camera")), 1)

    @unittest.skipIf(np is None, "NumPy is needed only for explicit legacy NPY import")
    def test_legacy_import_is_unverified_inactive_idempotent_and_preserves_original_files(self):
        self.root.mkdir()
        matrix_path, distortion_path = self.root / "camera_matrix.npy", self.root / "dist_coeffs.npy"
        np.save(matrix_path, np.array(MATRIX, dtype=float))
        np.save(distortion_path, np.array([DISTORTION], dtype=float))
        original = matrix_path.read_bytes(), distortion_path.read_bytes()
        self.assertEqual(self.store.list_profiles("camera"), [])
        imported = self.store.import_legacy_camera(matrix_path, distortion_path)
        self.assertEqual(imported["name"], "Legacy camera (unverified)")
        self.assertFalse(imported["bindingVerified"])
        self.assertIsNone(imported["calibration"]["imageSize"])
        self.assertIsNone(self.store.active_profile("camera"))
        generation = self.store.snapshot()["generation"]
        self.assertEqual(self.store.import_legacy_camera(matrix_path, distortion_path)["id"], imported["id"])
        self.assertEqual(self.store.snapshot()["generation"], generation)
        self.store.select_profile("camera", imported["id"])
        self.assertIn("unverified", self.store.resolve_camera_calibration([1280, 720])[1])
        self.assertEqual((matrix_path.read_bytes(), distortion_path.read_bytes()), original)


if __name__ == "__main__":
    unittest.main()
