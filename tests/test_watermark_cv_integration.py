"""Watermark/background integration without a camera, display or native window."""

import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cv_core as core
import recording_state as state


class WhiteGlyphDetector:
    """Deterministic synthetic caption detector; other colors remain foreground."""
    def __init__(self):
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        glyph = np.all(image == 255, axis=2).astype(np.uint8) * 255
        return SimpleNamespace(mask=glyph, exclusion_mask=cv2.dilate(
            glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))))


def image(caption=None):
    result = np.full((180, 320, 3), 50, np.uint8)
    if caption is not None:
        x, y = caption
        result[y:y + 20, x:x + 20] = 255
    return result


def session(background):
    result = core.CVCoreSession.__new__(core.CVCoreSession)
    for name in state.CORE_SCALARS:
        setattr(result, name, 0)
    result.warp_w, result.warp_h = background.shape[1], background.shape[0]
    result.grid_w, result.grid_h = 16, 9
    result.H_saved = result._H_inv = np.eye(3, dtype=np.float32)
    result.last_mask_cam = np.full(background.shape[:2], 255, np.uint8)
    result.last_mask_warp = result.last_mask_cam.copy()
    result.BG_cam = {"bgr": background.copy(), "blur": cv2.GaussianBlur(
        cv2.cvtColor(background, cv2.COLOR_BGR2GRAY), (21, 21), 0)}
    result.BG_warp_f32 = result.BG_cam["blur"].astype(np.float32)
    result.marker_mode = "legacy"
    result._bg_seeded = True
    result.warp_motion_thresh = 30
    result.fog_change_ratio = .08
    result.bg_alpha_fast, result.bg_alpha_slow = .15, .02
    result._heal_acc = np.zeros(background.shape[:2], np.uint16)
    result._heal_ms = 1500
    result._fps_estimate = 30
    result._source_is_file = True
    result.frame_time = 1.0
    result.replay_frame_clock = 1.0
    result._SHADOW_EVERY_N = 1
    result._last_shadowfree_cam = result._last_final_mask_cam = None
    result._watermark_detector = WhiteGlyphDetector()
    result.reset_watermark_background()
    return result


def compute(session, frame):
    session.frame_time += 1 / 30
    # Shadow computation is independent of this change. Make it deliberately
    # permissive so the test proves all output paths reapply caption exclusions.
    with patch.object(core.mt, "shadow_free_mask", return_value=np.full(frame.shape[:2], 255, np.uint8)):
        return session._compute_shared_masks(frame)


class WatermarkCvIntegrationTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(0)

    def test_current_and_background_letters_do_not_become_motion(self):
        saved = image((30, 70))
        current = image((250, 70))
        sess = session(saved)
        result = compute(sess, current)
        self.assertEqual(result["raw_motion_ratio"], 0)
        self.assertEqual(result["largest_motion_area"], 0)
        self.assertEqual(cv2.countNonZero(result["motion_warp"]), 0)
        self.assertEqual(result["watermark_mask_warp"][80, 40], 0)
        self.assertEqual(result["watermark_mask_warp"][80, 260], 255)
        self.assertEqual(result["watermark_exclusion_warp"][80, 40], 255)
        np.testing.assert_array_equal(current, image((250, 70)))
        # Two scans initially (current and saved background), one thereafter.
        self.assertEqual(sess._watermark_detector.calls, 2)
        compute(sess, current)
        self.assertEqual(sess._watermark_detector.calls, 3)

    def test_clear_pixels_relearn_after_three_observations_but_hidden_pixels_do_not(self):
        sess = session(image((30, 70)))
        current = image((250, 70))
        compute(sess, current)
        compute(sess, current)
        self.assertEqual(sess.BG_cam["bgr"][80, 40, 0], 255)
        compute(sess, current)
        self.assertEqual(sess.BG_cam["bgr"][80, 40, 0], 50)
        self.assertEqual(sess._watermark_bg_invalid_cam[80, 40], 0)
        # There was never a clean observation of the current caption.
        self.assertEqual(sess.BG_cam["bgr"][80, 260, 0], 50)
        self.assertLess(sess.BG_warp_f32[80, 260], 51)
        result = compute(sess, image())
        self.assertEqual(result["raw_motion_ratio"], 0)

    def test_moving_foreground_is_not_immediately_learned_when_caption_clears(self):
        sess = session(image((30, 70)))
        for value in (80, 140, 200, 90):
            current = image()
            current[55:105, 15:65] = value
            compute(sess, current)
        self.assertEqual(sess._watermark_bg_invalid_cam[80, 40], 255)
        self.assertEqual(sess.BG_cam["bgr"][80, 40, 0], 255)

    def test_real_foreground_and_observed_area_denominator_survive(self):
        sess = session(image((30, 70)))
        current = image((250, 70))
        current[115:170, 110:195] = 180
        result = compute(sess, current)
        observed = cv2.bitwise_and(sess.last_mask_warp,
                                   cv2.bitwise_not(result["watermark_exclusion_warp"]))
        self.assertGreater(result["raw_motion_ratio"], .035)
        self.assertGreater(result["largest_motion_area"], 1000)
        self.assertEqual(result["motion_warp"][145, 150], 255)
        self.assertAlmostEqual(result["raw_motion_ratio"],
            cv2.countNonZero(result["motion_warp"]) / cv2.countNonZero(observed))

    def test_cached_shadow_and_final_masks_also_exclude_new_caption(self):
        sess = session(image())
        compute(sess, image())
        sess._SHADOW_EVERY_N = 100
        sess._last_shadowfree_cam = np.full((180, 320), 255, np.uint8)
        sess._last_final_mask_cam = sess._last_shadowfree_cam.copy()
        result = compute(sess, image((250, 70)))
        for key in ("shadowfree_cam", "final_mask_cam", "motion_cam"):
            self.assertEqual(result[key][80, 260], 0)
        self.assertEqual(result["final_mask_cam"][145, 150], 255)

    def test_replacing_background_rescans_it_and_reset_discards_old_validity(self):
        sess = session(image())
        compute(sess, image())
        sess.BG_cam = session(image((30, 70))).BG_cam
        sess.BG_warp_f32 = sess.BG_cam["blur"].astype(np.float32)
        result = compute(sess, image())
        self.assertEqual(result["raw_motion_ratio"], 0)
        self.assertEqual(sess._watermark_bg_invalid_cam[80, 40], 255)
        self.assertEqual(sess._watermark_detector.calls, 4)
        sess.reset_watermark_background()
        self.assertFalse(sess._watermark_background_known)
        self.assertIsNone(sess._watermark_bg_invalid_warp)

    def test_new_homography_reprojects_current_letters_without_moving_saved_ghost(self):
        sess = session(image((30, 70)))
        compute(sess, image((250, 70)))
        old_invalid = sess._watermark_bg_invalid_warp.copy()
        sess.H_saved = np.array([[1, 0, 18], [0, 1, 0], [0, 0, 1]], np.float32)
        sess._H_inv = np.linalg.inv(sess.H_saved).astype(np.float32)
        # Existing BG_warp pixels remain in their original array coordinates;
        # only the current camera caption follows the new H.
        sess._prepare_watermark_masks(image((250, 70)))
        np.testing.assert_array_equal(sess._watermark_bg_invalid_warp, old_invalid)
        self.assertEqual(sess._watermark_mask_warp[80, 278], 255)
        self.assertEqual(sess._watermark_mask_warp[80, 252], 0)
        self.assertEqual(sess._watermark_detector.calls, 3)

    def test_checkpoint_roundtrip_and_old_archive_bootstrap(self):
        sess = session(image((30, 70)))
        compute(sess, image((250, 70)))
        compute(sess, image((250, 70)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            entry = {"path": path.name, "frame": 0,
                     "sha256": state.save_checkpoint(path, sess)}
            restored = session(image())
            state.restore_checkpoint(Path(directory) / "timeline.json", entry, restored, Mock())
            self.assertTrue(restored._watermark_background_known)
            for name in state.WATERMARK_ARRAYS:
                np.testing.assert_array_equal(getattr(restored, name), getattr(sess, name))
            compute(restored, image((250, 70)))
            self.assertEqual(restored._watermark_detector.calls, 1)
            self.assertEqual(restored.BG_cam["bgr"][80, 40, 0], 50)

            # Actual previous recordings have neither the optional scalar nor
            # arrays. They must detect the caption already baked into BG_cam.
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name].copy() for name in archive.files
                          if name not in state.WATERMARK_ARRAYS}
            metadata = json.loads(arrays["metadata"].tobytes())
            metadata.pop("_watermark_background_known")
            arrays["metadata"] = np.frombuffer(json.dumps(metadata).encode(), np.uint8)
            with path.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            legacy = session(image())
            state.restore_checkpoint(Path(directory) / "timeline.json", entry, legacy, Mock())
            self.assertFalse(legacy._watermark_background_known)
            result = compute(legacy, image((250, 70)))
            self.assertEqual(legacy._watermark_detector.calls, 2)
            self.assertEqual(result["raw_motion_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
