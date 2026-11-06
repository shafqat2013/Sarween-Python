import json
import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cv_core
import mini_calibration
import mini_library
import ring_calibration as ring
import ring_scan_dialog as picker
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

    def test_base_sample_has_independent_curve_threshold_and_metadata(self):
        old = {"lab": [30, 39, 26], "lab_curve": [[30, 39, 26]],
               "brightness_steps": [255], "presence_lab_dist": 12,
               "max_lab_dist": 40, "scan_source": "manual-ring-sample",
               "expected_diameter_squares": 1.1}
        original = copy.deepcopy(old)
        updated = ring.profile_with_sample(old, [65, 2, -4], sample_kind="base_appearance")
        appearance = updated.pop("base_appearance")
        self.assertEqual(updated, original)
        self.assertEqual(old, original)
        self.assertEqual(appearance, {"lab": [65, 2, -4], "lab_curve": [[65, 2, -4]],
            "presence_lab_dist": 18, "scan_source": "manual-base-appearance-sample"})

    def test_append_replace_and_duplicate_affect_only_the_selected_cue(self):
        old = {"lab": [30, 39, 26], "lab_curve": [[30, 39, 26]], "max_lab_dist": 35,
               "base_appearance": {"lab": [60, 2, -4], "lab_curve": [[60, 2, -4]],
                                   "presence_lab_dist": 12, "condition": "night"}}
        original = copy.deepcopy(old)
        appended = ring.profile_with_sample(old, [65, 3, -2], sample_kind="base_appearance")
        self.assertEqual(appended["base_appearance"]["lab_curve"], [[60, 2, -4], [65, 3, -2]])
        self.assertEqual(appended["base_appearance"]["presence_lab_dist"], 12)
        self.assertEqual(appended["base_appearance"]["condition"], "night")
        self.assertEqual(ring.profile_with_sample(appended, [65, 3, -2],
                         sample_kind="base_appearance"), appended)
        replaced = ring.profile_with_sample(appended, [70, 4, -1],
            sample_kind="base_appearance", replace_colors=True)
        self.assertEqual(replaced["base_appearance"]["lab_curve"], [[70, 4, -1]])
        self.assertEqual(replaced["lab_curve"], old["lab_curve"])
        self.assertEqual(replaced["max_lab_dist"], 35)
        identity_replaced = ring.profile_with_sample(appended, [40, 40, 30], replace_colors=True)
        self.assertEqual(identity_replaced["base_appearance"], appended["base_appearance"])
        self.assertEqual(identity_replaced["lab_curve"], [[40, 40, 30]])
        self.assertEqual(old, original)

    def test_base_cue_roundtrips_and_atomic_backup_preserves_prior_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            original = {"red10": {"lab": [30, 39, 26]}, "blue": {"lab": [45, 2, -30]}}
            ring.save_profiles(original, path)
            updated = {**original, "red10": ring.profile_with_sample(original["red10"],
                [65, 2, -4], sample_kind="base_appearance", base_diameter_squares=1.25)}
            ring.save_profiles(updated, path)
            self.assertEqual(mini_calibration.load_profiles_with_curves(path), updated)
            self.assertEqual(updated["red10"]["base_appearance"]["diameter_squares"], 1.25)
            self.assertEqual(json.loads(path.with_suffix(".json.bak").read_text()), original)
            self.assertEqual(updated["blue"], original["blue"])

    def test_invalid_kind_or_base_sample_never_changes_identity(self):
        old = {"lab": [30, 39, 26], "lab_curve": [[30, 39, 26]]}
        original = copy.deepcopy(old)
        for kind in (None, "", "Ring color", "base", "unknown", [], {}, True):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "Choose Ring color"):
                ring.profile_with_sample(old, [65, 2, -4], sample_kind=kind)
        for lab in ([1, 2], [float("nan"), 2, 3], [1, float("inf"), 3]):
            with self.subTest(lab=lab), self.assertRaisesRegex(ValueError, "finite Lab"):
                ring.profile_with_sample(old, lab, sample_kind="base_appearance")
        with self.assertRaisesRegex(ValueError, "base appearance must be"):
            ring.profile_with_sample({**old, "base_appearance": []}, [65, 2, -4],
                                     sample_kind="base_appearance")
        self.assertEqual(old, original)

    def test_base_cue_requires_an_existing_finite_identity_sample(self):
        for old in ({}, {"lab_curve": [None]}, {"lab": [float("nan"), 0, 0]},
                    {"base_appearance": {"lab": [65, 2, -4]}}):
            with self.subTest(old=old), self.assertRaisesRegex(ValueError, "Save a ring color sample first"):
                ring.profile_with_sample(old, [65, 2, -4], sample_kind="base_appearance")
        curve_only = {"lab_curve": [[30, 39, 26]]}
        saved = ring.profile_with_sample(curve_only, [65, 2, -4], sample_kind="base_appearance")
        self.assertEqual(saved["lab_curve"], curve_only["lab_curve"])
        self.assertNotIn("lab", saved)

    def test_base_diameter_is_optional_bounded_and_independent_of_identity_geometry(self):
        old = {"lab": [30, 39, 26], "expected_diameter_squares": 2,
               "base_appearance": {"lab": [60, 2, -4], "lab_curve": [[60, 2, -4]],
                                   "diameter_squares": 1.25}}
        original = copy.deepcopy(old)
        for replace_colors in (False, True):
            saved = ring.profile_with_sample(old, [65, 3, -2], sample_kind="base_appearance",
                                             replace_colors=replace_colors)
            self.assertEqual(saved["base_appearance"]["diameter_squares"], 1.25)
        for value in (.1, 1.5, 6, "1.75"):
            saved = ring.profile_with_sample(old, [65, 3, -2], sample_kind="base_appearance",
                                             base_diameter_squares=value)
            self.assertEqual(saved["base_appearance"]["diameter_squares"], float(value))
            self.assertEqual(saved["expected_diameter_squares"], 2)
            self.assertEqual(saved["lab"], old["lab"])
        duplicate = ring.profile_with_sample(old, [60, 2, -4], sample_kind="base_appearance",
                                             base_diameter_squares=1.5)
        self.assertEqual(duplicate["base_appearance"]["lab_curve"], [[60, 2, -4]])
        self.assertEqual(duplicate["base_appearance"]["diameter_squares"], 1.5)
        with self.assertRaisesRegex(ValueError, "only to Base appearance"):
            ring.profile_with_sample(old, [40, 40, 30], base_diameter_squares=1.5)
        self.assertEqual(old, original)

    def test_invalid_base_diameters_are_rejected_without_profile_mutation(self):
        old = {"lab": [30, 39, 26], "base_appearance": {"lab": [60, 2, -4], "diameter_squares": 1.25}}
        original = copy.deepcopy(old)
        for value in (0, -.1, 6.01, float("nan"), float("inf"), True, False,
                      np.bool_(True), "unknown", "", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Base diameter"):
                ring.profile_with_sample(old, [65, 3, -2], sample_kind="base_appearance",
                                         base_diameter_squares=value)
        self.assertEqual(old, original)

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


class PickerVariable:
    def __init__(self, *, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class PickerWidget:
    def __init__(self, kind, parent=None, **options):
        self.kind, self.options = kind, options
        self.parent, self.packed = parent, False
        self.bindings, self.children = {}, []
        self.destroyed = False
        if isinstance(parent, PickerWidget):
            parent.children.append(self)

    def pack(self, **options):
        self.packed = True

    def pack_forget(self):
        self.packed = False

    def title(self, text):
        self.title_text = text

    def configure(self, **options):
        self.options.update(options)

    def bind(self, name, callback):
        self.bindings[name] = callback

    def create_image(self, *args, **kwargs):
        pass

    def create_rectangle(self, *args, **kwargs):
        pass

    def delete(self, *args):
        pass

    def destroy(self):
        self.destroyed = True

    def invoke(self):
        if self.options.get("state") == "disabled":
            return
        if self.kind == "Radiobutton":
            self.options["variable"].set(self.options["value"])
        return self.options["command"]()


class RingPickerHeadlessTests(unittest.TestCase):
    def setUp(self):
        self.widgets, self.samples = [], []
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)

        def factory(kind):
            def make(parent=None, **options):
                widget = PickerWidget(kind, parent, **options)
                self.widgets.append(widget)
                return widget
            return make

        self.patches.enter_context(patch.multiple(picker.tk, Toplevel=factory("Toplevel"),
            Canvas=factory("Canvas"), StringVar=PickerVariable, BooleanVar=PickerVariable))
        self.patches.enter_context(patch.multiple(picker.ttk,
            **{name: factory(name) for name in ("Frame", "Label", "Button", "Checkbutton", "Radiobutton", "Entry")}))
        self.patches.enter_context(patch.object(picker.ImageTk, "PhotoImage", return_value=object()))
        parent = Mock()
        parent.winfo_screenwidth.return_value = 1440
        parent.winfo_screenheight.return_value = 900
        self.image = np.full((160, 160, 3), (20, 30, 180), np.uint8)
        self.window = picker.show_ring_scan(parent, "red10", self.image, self.samples.append)
        self.view = next(widget for widget in self.widgets if widget.kind == "Canvas")
        self.use = next(widget for widget in self.widgets if widget.options.get("text") == "Use ring sample")
        self.replace = next(widget for widget in self.widgets if widget.kind == "Checkbutton")
        self.diameter = next(widget for widget in self.widgets if widget.kind == "Entry")

    def choose(self):
        self.view.bindings["<Button-1>"](SimpleNamespace(x=80, y=80))

    def mode(self, label):
        return next(widget for widget in self.widgets if widget.kind == "Radiobutton"
                    and widget.options["text"] == label)

    def test_default_picker_confirms_identity_and_unchecked_append(self):
        original = self.image.copy()
        self.use.invoke()
        self.assertEqual(self.samples, [])
        self.choose()
        self.use.invoke()
        self.assertEqual(self.samples[0]["sample_kind"], "identity")
        self.assertFalse(self.samples[0]["replace_colors"])
        self.assertEqual(self.samples[0]["name"], "red10")
        self.assertNotIn("base_diameter_squares", self.samples[0])
        self.assertFalse(self.diameter.parent.packed)
        np.testing.assert_array_equal(self.image, original)
        self.assertTrue(self.window.destroyed)

    def test_changing_mode_requires_a_fresh_point_and_resets_replace_unchecked(self):
        self.choose()
        self.replace.options["variable"].set(True)
        self.mode("Base appearance").invoke()
        self.assertEqual(self.use.options["text"], "Use base sample")
        self.assertFalse(self.replace.options["variable"].get())
        self.use.invoke()
        self.assertEqual(self.samples, [])
        self.choose()
        self.use.invoke()
        self.assertEqual(self.samples[0]["sample_kind"], "base_appearance")
        self.assertFalse(self.samples[0]["replace_colors"])
        self.assertIsNone(self.samples[0]["base_diameter_squares"])

    def test_base_replacement_remains_explicit_and_invalid_mode_cannot_confirm(self):
        self.mode("Base appearance").invoke()
        self.choose()
        self.replace.options["variable"].set(True)
        self.use.invoke()
        self.assertEqual(self.samples[0]["sample_kind"], "base_appearance")
        self.assertTrue(self.samples[0]["replace_colors"])
        self.samples.clear()
        self.window.destroyed = False
        self.mode("Ring color").options["variable"].set("invalid")
        self.use.invoke()
        self.assertEqual(self.samples, [])
        self.assertFalse(self.window.destroyed)

    def test_optional_diameter_only_confirms_valid_base_mode_values(self):
        self.mode("Base appearance").invoke()
        self.assertTrue(self.diameter.parent.packed)
        self.choose()
        for invalid in ("0", "-1", "6.1", "nan", "inf", "true", "unknown"):
            self.diameter.options["textvariable"].set(invalid)
            self.use.invoke()
            self.assertEqual(self.samples, [])
            self.assertFalse(self.window.destroyed)
        self.diameter.options["textvariable"].set(" 1.75 ")
        self.use.invoke()
        self.assertEqual(self.samples[0]["base_diameter_squares"], 1.75)

    def test_hidden_diameter_draft_cannot_change_an_identity_sample(self):
        self.mode("Base appearance").invoke()
        self.diameter.options["textvariable"].set("1.75")
        self.mode("Ring color").invoke()
        self.assertFalse(self.diameter.parent.packed)
        self.choose()
        self.use.invoke()
        self.assertEqual(self.samples[0]["sample_kind"], "identity")
        self.assertNotIn("base_diameter_squares", self.samples[0])


if __name__ == "__main__":
    unittest.main()
