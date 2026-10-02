import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import cv_core
import mini_calibration
import mini_library
import ring_calibration as ring
import v3_tracking


class RingCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((160, 160, 3), 240, dtype=np.uint8)
        cv2.circle(self.image, (80, 80), 24, (0, 0, 255), -1)
        cv2.circle(self.image, (80, 80), 16, (60, 60, 130), -1)

    def test_requires_explicit_ring_point_not_largest_motion_blob(self):
        bundle = SimpleNamespace(locked=True, warp_bgr=self.image)
        self.assertIsNone(v3_tracking.calibrate_from_bundle(bundle, "red", {}))
        profile = v3_tracking.calibrate_from_bundle(bundle, "red", {}, sample_xy=(101, 80))
        expected = ring.sample_ring_patch(self.image, (101, 80))
        self.assertEqual(profile["lab"], list(expected))
        self.assertGreater(np.linalg.norm(np.array(profile["lab"]) - ring.sample_ring_patch(self.image, (80, 80))), 50)

    def test_mixed_and_edge_pixels_are_rejected(self):
        self.assertIsNone(ring.sample_ring_patch(self.image, (0, 80)))
        self.assertIsNone(ring.sample_ring_patch(self.image, (104, 80)))
        self.assertIsNone(ring.sample_ring_patch(self.image, None))

    def test_scan_preserves_curve_thresholds_and_source_object(self):
        existing = {"lab": [30, 30, 20], "lab_curve": [[30, 30, 20], None],
                    "brightness_steps": [255, 0], "presence_lab_dist": 12}
        updated = ring.profile_with_sample(existing, [40, 40, 30])
        self.assertEqual(updated["lab_curve"], [[30, 30, 20], None, [40, 40, 30]])
        self.assertEqual(updated["brightness_steps"], [255, 0, None])
        self.assertEqual(updated["presence_lab_dist"], 12)
        self.assertEqual(len(existing["lab_curve"]), 2)
        self.assertEqual(ring.profile_with_sample(updated, [40, 40, 30]), updated)

    def test_explicit_replacement_only_changes_colors(self):
        old = {"lab": [30, 30, 20], "presence_lab_dist": 12, "expected_diameter_squares": 1.1}
        new = ring.profile_with_sample(old, [40, 40, 30], replace_colors=True)
        self.assertEqual(new["lab_curve"], [[40, 40, 30]])
        self.assertEqual(new["presence_lab_dist"], 12)
        self.assertEqual(new["expected_diameter_squares"], 1.1)

    def test_atomic_save_and_partial_scan_preserve_other_minis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            initial = {"red": {"lab": [20, 30, 40]}, "blue": {"lab": [40, 10, -30]}}
            ring.save_profiles(initial, path)
            updated = ring.merge_profiles({"red": {"lab": [21, 31, 41]}}, path)
            self.assertEqual(updated["blue"], initial["blue"])
            self.assertEqual(json.loads(path.with_suffix(".json.bak").read_text()), initial)
            with patch.object(ring.os, "replace", side_effect=OSError("disk error")):
                with self.assertRaises(OSError):
                    ring.save_profiles({}, path)
            self.assertEqual(json.loads(path.read_text()), updated)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_corrupt_existing_profile_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text("not-json")
            with self.assertRaises(ValueError):
                ring.merge_profiles({"red": {}}, path)
            self.assertEqual(path.read_text(), "not-json")

    def test_brightness_scan_session_uses_its_own_explicit_registration(self):
        with patch.object(mini_calibration.core, "CVCoreSession") as session:
            mini_calibration._calibration_session(2, 1920, 1080)
        options = session.call_args.kwargs
        self.assertEqual(options["marker_mode"], "legacy")
        self.assertEqual((options["warp_w"], options["warp_h"]), (1920, 1080))
        image = mini_calibration._generate_calibration_image(1920, 1080)
        homography, count, _, _, _ = cv_core.solve_H_from_markers(image, 1920, 1080, cv_core.LEGACY_CORNER_IDS)
        self.assertEqual(count, 4)
        self.assertIsNotNone(homography)

    def test_labeler_freezes_blob_identity_and_remembers_clicked_ring(self):
        contour = np.array([[[50, 50]], [[110, 50]], [[110, 110]], [[50, 110]]], dtype=np.int32)
        blob = (contour, 80, 80, 3600, 0.8)
        labeler = mini_calibration.BlobLabeler()
        labeler.set_blobs([blob])
        labeler.on_mouse(cv2.EVENT_LBUTTONDOWN, 101, 80, None, None)
        labeler.set_blobs([])
        self.assertEqual(len(labeler.blobs), 1)
        self.assertEqual(labeler.sample_points[0], (101, 80))
        for letter in "red":
            labeler.handle_key(ord(letter))
        labeler.handle_key(13)
        second = (contour + 100, 180, 180, 3600, 0.8)
        labeler.set_blobs([second])
        self.assertEqual(len(labeler.blobs), 2)
        self.assertEqual(labeler.labels, {0: "red"})
        self.assertEqual(labeler.sample_points[0], (101, 80))

    def test_import_preserves_brightness_indices_and_requires_confirmation(self):
        library = mini_library.default_library()
        mini_library.sync_profile_samples(library, {"red10": {
            "lab_curve": [None, [20, 30, 40]], "brightness_steps": [255, 100]}})
        sample = library["minis"]["red10"]["samples"][0]
        self.assertFalse(sample["verified"])
        self.assertEqual(sample["conditions"]["display"], "display-brightness:100")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "library.json"
            mini_library.save_library(library, path)
            self.assertEqual(mini_library.add_verified_sample("red10", [20, 30, 40], path=path), "verified")
            self.assertTrue(mini_library.load_library(path)["minis"]["red10"]["samples"][0]["verified"])


if __name__ == "__main__":
    unittest.main()
