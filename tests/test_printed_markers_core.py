"""CVCore integration with generated marker observations; no devices or UI."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np

import cv_core as core
from hardware_profiles import HardwareProfiles
from printed_marker_runtime import PrintedMarkerController
import printed_markers as pm
from tests.test_printed_markers import camera_transform, detected, layout, view


class PrintedCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = HardwareProfiles(Path(self.temp.name) / "profiles.json")
        self.profile = self.store.upsert_profile("display", {"name": "Temporary TV", "settings": {"targetGridInches": 1},
            "printedMarkerLayout": layout(), "printedMarkersEnabled": True}, select=True)
        self.context = {"ready": True, "view": view(), "settings": {
            "viewportMarkersEnabled": True, "viewportMarkerSize": 120, "viewportMarkerInset": 8}}
        self.controller = PrintedMarkerController(self.store, context_provider=lambda: self.context)
        self.session = core.CVCoreSession.__new__(core.CVCoreSession)
        values = dict(marker_mode="viewport", _printed_marker_controller=self.controller,
            _screen_corner_ids=dict(pm.SCREEN_IDS), corner_ids=dict(pm.SCREEN_IDS), required_ids=set(pm.SCREEN_IDS.values()),
            warp_w=1280, warp_h=720, grid_w=48, grid_h=27, frame_idx=1, _frame_phase_offset=0,
            aruco_every_n=1, aruco_every_n_fast=1, last_marker_count=0, last_mask_cam=None,
            last_mask_warp=None, H_saved=np.eye(3, dtype=np.float32), lock_miss_streak=0, lock_drop_after=10,
            _registration_token=None, _registration_revision=0, _recorder=None,
            BG_warp_f32=np.ones((720, 1280), np.float32), _bg_seeded=True,
            _last_shadowfree_cam=np.ones((4, 4), np.uint8), _last_final_mask_cam=np.ones((4, 4), np.uint8))
        self.session.__dict__.update(values)
        self.controller.heartbeat(self.session)
        self.image = np.zeros((1080, 1920, 3), np.uint8)

    def update(self, *, included=(20, 21, 22, 23), H=None, duplicate=False, paper=None):
        corners, ids = detected(paper, included=included, H=H)
        if duplicate:
            corners.append(corners[0])
            ids = np.vstack([ids, ids[0]])
        with patch.object(core, "detect_markers", return_value=(corners, ids, [])):
            self.session._update_lock_if_due(self.image)

    def test_printed_acquisition_uses_lcd_mask_and_carries_detached_metadata(self):
        self.update()
        self.assertIsNotNone(self.session.last_mask_cam)
        self.assertEqual(self.session.last_marker_count, 4)
        self.assertEqual(self.session.corner_ids, pm.PRINTED_IDS)
        self.assertFalse(self.session._bg_seeded)
        self.assertIsNone(self.session.BG_warp_f32)
        self.assertIsNone(self.session._last_shadowfree_cam)
        metadata = self.session.registration_metadata_snapshot()
        self.assertEqual(metadata["mode"], "printed")
        self.assertEqual(metadata["profileId"], self.profile["id"])
        metadata["mode"] = "changed"
        self.assertEqual(self.session.registration_metadata["mode"], "printed")

    def test_three_matching_cards_hold_and_two_immediately_remove_unsafe_lock(self):
        self.update()
        H = self.session.H_saved.copy()
        self.update(included=(20, 21, 22))
        self.assertIsNotNone(self.session.last_mask_cam)
        np.testing.assert_array_equal(self.session.H_saved, H)
        self.update(included=(20, 21))
        self.assertIsNone(self.session.last_mask_cam)
        self.assertIn("unverified", self.session.lock_lost_reason)
        self.update(included=(20, 21, 22))
        self.assertIsNone(self.session.last_mask_cam)
        self.update()
        self.assertIsNotNone(self.session.last_mask_cam)

    def test_duplicate_or_moved_camera_with_only_three_cards_clears_lock(self):
        self.update()
        self.update(duplicate=True)
        self.assertIsNone(self.session.last_mask_cam)
        self.assertIn("Duplicate", self.session.lock_lost_reason)
        self.update()
        self.update(included=(20, 21, 22), H=camera_transform((30, 0)))
        self.assertIsNone(self.session.last_mask_cam)

    def test_mode_layout_and_viewport_changes_invalidate_background_and_geometry_revision(self):
        self.update()
        first = self.session.registration_metadata_snapshot()
        self.context["view"] = view(1280, 720, 20)
        self.session.BG_warp_f32 = np.ones((720, 1280), np.float32)
        self.session._bg_seeded = True
        self.update()
        self.assertIsNotNone(self.session.last_mask_cam)
        self.assertGreater(self.session.registration_metadata["revision"], first["revision"])
        self.assertIsNone(self.session.BG_warp_f32)
        self.context["view"] = view(1024, 768)
        self.update()
        self.assertIsNone(self.session.last_mask_cam)
        self.assertIn("aspect changed", self.session.lock_lost_reason)
        self.controller.use_printed(False)
        with patch.object(core, "solve_H_from_markers", return_value=(None, 0, [], None, None)):
            self.session._update_lock_if_due(self.image)
        self.assertEqual(self.session.registration_metadata["mode"], "screen")
        self.assertEqual(self.session.required_ids, set(pm.SCREEN_IDS.values()))
        self.assertIsNone(self.session.last_mask_cam)

    def test_pending_guide_request_finishes_from_four_paper_cards_without_screen_markers(self):
        self.controller.use_printed(False)
        self.controller.request_calibration(aligned_to_guides=True)
        guide = pm.guide_aligned_layout(view(), 120, 8)
        for _ in range(3):
            self.update(paper=guide)
        self.assertFalse(self.controller.status()["pending"])
        self.assertEqual(self.session.registration_metadata["mode"], "printed")
        self.assertEqual(self.session.registration_metadata["calibrationMethod"], "guide-aligned-user-declared")
        self.assertIsNotNone(self.session.last_mask_cam)

    def test_recorded_repeated_loss_writes_one_checkpoint_and_rebuilds_background_once(self):
        self.update()
        session = self.session
        session._recorder = Mock()
        session._checkpoint_kind = None
        session.recording_state_provider = lambda: {}
        session.before_frame_callback = None
        session._source_is_file = True
        session.loop_video = False
        session.replay_frame_clock = None
        session.cap = Mock()
        session.cap.read.side_effect = [(True, self.image)] * 6 + [(False, None)]
        c, ids = detected(included=(20, 21))
        original_blur = core.cv2.GaussianBlur
        with patch.object(session, "_maybe_undistort", side_effect=lambda frame: frame), \
             patch.object(core, "detect_markers", return_value=(c, ids, [])), \
             patch.object(core.fo, "get_scene_params", return_value={"gridCols": 48, "gridRows": 27}), \
             patch.object(core.fo, "record_timeline_frame"), \
             patch.object(core.fo, "record_checkpoint") as checkpoint, \
             patch.object(core.cv2, "GaussianBlur", wraps=original_blur) as blur:
            frames = list(session.frames())
            self.assertEqual(len(frames), 6)
            self.assertTrue(all(not frame.locked for frame in frames))
            self.assertEqual(checkpoint.call_count, 1)
            self.assertEqual(blur.call_count, 1)
            self.assertIsNone(session._checkpoint_kind)
            # An actual mode change still needs its own checkpoint even though
            # the preceding run already removed the unsafe geometry.
            self.controller.use_printed(False)
            session.cap.read.side_effect = [(True, self.image)] * 2 + [(False, None)]
            list(session.frames())
            self.assertEqual(checkpoint.call_count, 2)
            self.assertEqual(blur.call_count, 2)


if __name__ == "__main__":
    unittest.main()
