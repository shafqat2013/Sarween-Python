"""Queue/save/profile races with temporary stores and synthetic detections."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hardware_profiles import HardwareProfiles
from printed_marker_runtime import PrintedMarkerController
import printed_markers as pm
from tests.test_printed_markers import detected, layout, view


class PrintedControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = HardwareProfiles(Path(self.temp.name) / "profiles.json")
        self.profile = self.store.upsert_profile("display", {"name": "Test TV", "settings": {"targetGridInches": 1}}, select=True)
        self.context = {"ready": True, "view": view(), "settings": {
            "viewportMarkersEnabled": True, "viewportMarkerSize": 120, "viewportMarkerInset": 8}}
        self.controller = PrintedMarkerController(self.store, context_provider=lambda: self.context)
        self.controller.heartbeat(self)

    def frames(self, controller=None, *, paper=None, screen=False, count=3):
        controller = controller or self.controller
        c, ids = detected(paper or layout(), include_screen=screen)
        for _ in range(count):
            controller.consume_detections(c, ids, self.context, (1280, 720))

    def test_guide_registration_works_with_screen_ids_hidden_and_saves_explicit_provenance(self):
        self.controller.request_calibration(aligned_to_guides=True)
        paper = pm.guide_aligned_layout(view(), 120, 8)
        with patch("cv2.VideoCapture", side_effect=AssertionError("No camera may be opened here")), \
             patch("foundryoutput.queue_control", side_effect=AssertionError("No Foundry command here")):
            self.frames(paper=paper, count=2)
            self.assertTrue(self.controller.status()["pending"])
            self.frames(paper=paper, count=1)
        saved = self.store.active_profile("display")
        self.assertTrue(saved["printedMarkersEnabled"])
        self.assertEqual(saved["printedMarkerLayout"]["calibrationMethod"], "guide-aligned-user-declared")
        self.assertEqual(self.controller.status()["profileId"], self.profile["id"])
        self.assertEqual(self.controller.status()["mode"], "printed")
        self.assertFalse(self.controller.status()["pending"])

    def test_covision_requires_all_eight_and_accepts_rotated_bezel_cards(self):
        self.controller.request_calibration()
        self.frames(screen=False)
        self.assertFalse(self.controller.status()["calibrated"])
        self.assertTrue(self.controller.status()["pending"])
        self.frames(screen=True)
        self.assertEqual(self.store.active_profile("display")["printedMarkerLayout"]["calibrationMethod"], "co-visible")

    def test_declared_alignment_consumes_independent_pattern_and_margin_settings(self):
        self.context["view"] = view(margin=14)
        self.context["settings"].update(viewportMarkerPatternSize=48, viewportMarkerMargin=6)
        paper = pm.guide_aligned_layout(self.context["view"], inset=8, pattern_size=48, margin=6)
        self.controller.request_calibration(aligned_to_guides=True)
        self.frames(paper=paper)
        saved = self.store.active_profile("display")["printedMarkerLayout"]
        self.assertIsNotNone(saved)
        for marker_id, corners in paper["markerCorners"].items():
            self.assertEqual(saved["markerCorners"][marker_id], corners)

    def test_cancel_preserves_previous_profile_and_late_observations_cannot_save(self):
        self.controller.request_calibration()
        self.frames(screen=True, count=1)
        previous = self.store.snapshot()
        self.controller.cancel_calibration()
        self.frames(screen=True)
        self.assertEqual(self.store.snapshot(), previous)
        self.assertFalse(self.controller.status()["pending"])

    def test_profile_switch_or_edit_cancels_instead_of_saving_into_wrong_tv(self):
        for action in ("switch", "rename"):
            with self.subTest(action=action):
                self.store.select_profile("display", self.profile["id"])
                self.controller.request_calibration()
                self.frames(screen=True, count=1)
                if action == "switch":
                    self.store.upsert_profile("display", {"name": "Other TV", "settings": {"targetGridInches": 1}}, select=True)
                else:
                    self.store.rename_profile("display", self.profile["id"], "Renamed TV")
                previous = self.store.snapshot()
                self.frames(screen=True)
                self.assertEqual(self.store.snapshot(), previous)
                self.assertFalse(self.controller.status()["pending"])

    def test_save_failure_is_visible_and_does_not_switch_marker_mode(self):
        self.controller.request_calibration()
        with patch.object(self.store, "save_printed_marker_layout", side_effect=OSError("Disk unavailable")):
            self.frames(screen=True)
        self.assertIn("Disk unavailable", self.controller.status()["message"])
        self.assertEqual(self.controller.status()["mode"], "screen")
        self.assertIsNone(self.store.active_profile("display")["printedMarkerLayout"])

    def test_no_active_profile_and_invalid_enable_are_rejected_without_files(self):
        self.store.select_profile("display", None)
        with self.assertRaises(ValueError):
            self.controller.request_calibration()
        with self.assertRaises(ValueError):
            self.controller.use_printed(True)
        self.assertFalse(self.controller.status()["available"])
        self.store.select_profile("display", self.profile["id"])
        with self.assertRaises(ValueError):
            self.controller.use_printed(True)
        with self.assertRaises(ValueError):
            self.controller.use_printed(1)

    def test_stale_guide_settings_or_absent_connection_do_not_complete_registration(self):
        paper = pm.guide_aligned_layout(view(), 120, 8)
        self.controller.request_calibration(aligned_to_guides=True)
        for change in ({"ready": False}, {"settings": {"viewportMarkersEnabled": False}},
                       {"settings": {"viewportMarkersEnabled": True, "viewportMarkerSize": 160, "viewportMarkerInset": 8}}):
            old = deepcopy(self.context)
            self.context.update(change)
            self.frames(paper=paper)
            self.assertFalse(self.controller.status()["calibrated"])
            self.context = old
        self.frames(paper=paper)
        self.assertTrue(self.controller.status()["calibrated"])

    def test_mode_changes_preserve_layout_and_idempotent_choice_preserves_revision(self):
        self.controller.request_calibration()
        self.frames(screen=True)
        saved = self.store.active_profile("display")["printedMarkerLayout"]
        before = self.controller.configuration()["token"]["revision"]
        self.controller.use_printed(True)
        self.assertEqual(before, self.controller.configuration()["token"]["revision"])
        self.controller.use_printed(False)
        self.assertEqual(self.controller.status()["mode"], "screen")
        self.assertEqual(saved, self.store.active_profile("display")["printedMarkerLayout"])
        self.assertGreater(self.controller.configuration()["token"]["revision"], before)

    def test_atomic_mode_change_rejects_a_profile_edited_after_the_initial_read(self):
        self.controller.request_calibration()
        self.frames(screen=True)
        original = self.store.set_printed_markers_enabled
        def edit_then_save(*args, **kwargs):
            self.store.rename_profile("display", self.profile["id"], "Changed elsewhere")
            return original(*args, **kwargs)
        with patch.object(self.store, "set_printed_markers_enabled", side_effect=edit_then_save):
            with self.assertRaises(ValueError):
                self.controller.use_printed(False)
        self.assertTrue(self.store.active_profile("display")["printedMarkersEnabled"])

    def test_real_context_requires_current_handshake_and_matching_scene(self):
        c = PrintedMarkerController(self.store)
        with patch("foundryoutput.get_vtt_status", return_value={"connected": False, "sceneId": "scene"}), \
             patch("foundryoutput.get_view_transform_payload", return_value=view()):
            self.assertFalse(c.view_context()["ready"])
        with patch("foundryoutput.get_vtt_status", return_value={"connected": True, "sceneId": "other"}), \
             patch("foundryoutput.get_view_transform_payload", return_value=view()):
            self.assertFalse(c.view_context()["ready"])

    def test_camera_off_expiry_and_source_switch_cancel_pending_registration(self):
        clock = [0.]
        controller = PrintedMarkerController(self.store, clock=lambda: clock[0])
        self.assertFalse(controller.status()["available"])
        with self.assertRaisesRegex(ValueError, "live camera"):
            controller.request_calibration()
        with self.assertRaisesRegex(ValueError, "live camera"):
            controller.use_printed(False)
        source = object()
        controller.heartbeat(source)
        controller.request_calibration()
        self.assertTrue(controller.status()["pending"])
        clock[0] = 4.
        self.assertFalse(controller.status()["available"])
        self.assertFalse(controller.status()["pending"])
        controller.heartbeat(source)
        controller.request_calibration()
        controller.heartbeat(object())
        self.assertFalse(controller.status()["pending"])


if __name__ == "__main__":
    unittest.main()
