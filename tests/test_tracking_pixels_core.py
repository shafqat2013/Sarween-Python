"""Registered pixel work stays independent of legacy motion and source guesses."""
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

import cv_core as core


class TrackingPixelsTests(unittest.TestCase):
    def session(self, *, phone=False, metadata=None, recorded=False):
        session = core.CVCoreSession.__new__(core.CVCoreSession)
        session.warp_w, session.warp_h = 80, 60
        session.H_saved = np.array([[.5, 0, 2], [0, .5, 1], [0, 0, 1]], np.float32)
        session.last_mask_warp = np.full((60, 80), 255, np.uint8)
        session._source_is_phone = phone
        session._source_is_file = recorded
        session._camera_source_metadata = metadata
        session._watermark_detector = Mock()
        return session

    def test_owned_phone_clears_stale_coverage_and_retains_exact_color_warp(self):
        session = self.session(phone=True, metadata={"source": "sarween-phone-usb", "transport": "usb"})
        session._watermark_current_exclusion_warp = np.full((60, 80), 255, np.uint8)
        pixels = np.arange(120 * 160 * 3, dtype=np.uint8).reshape(120, 160, 3)
        original = pixels.copy()
        with patch.object(session, "_compute_shared_masks", side_effect=AssertionError("Legacy pipeline called")), \
                patch.object(core.mt, "shadow_free_mask", side_effect=AssertionError("Legacy shadow called")), \
                patch.object(session, "_prepare_watermark_masks", side_effect=AssertionError("Legacy masks called")):
            result = session._compute_tracking_pixels(pixels)
        np.testing.assert_array_equal(result["warp_bgr"], cv2.warpPerspective(pixels, session.H_saved, (80, 60)))
        np.testing.assert_array_equal(pixels, original)
        self.assertEqual(result["watermark_coverage"], 0)
        self.assertEqual(result["watermark_excluded_fraction"], 0)
        self.assertFalse(np.any(result["watermark_exclusion_warp"]))
        self.assertFalse(np.any(session._watermark_current_exclusion_warp))
        self.assertIsNone(result["bg_warp_u8"])
        session._watermark_detector.detect.assert_not_called()

    def test_native_unknown_and_recorded_sources_retain_current_caption_exclusion(self):
        for phone, metadata, recorded in (
                (False, {"source": "sarween-phone-usb", "transport": "usb"}, False),
                (True, None, False),
                (True, {"source": "sarween-phone-usb", "transport": "wireless"}, False),
                (True, {"source": "sarween-phone-usb", "transport": "usb"}, True)):
            with self.subTest(phone=phone, metadata=metadata, recorded=recorded):
                session = self.session(phone=phone, metadata=metadata, recorded=recorded)
                pixels = np.zeros((120, 160, 3), np.uint8)
                glyph = np.zeros(pixels.shape[:2], np.uint8)
                glyph[40:48, 60:68] = 255
                exclusion = cv2.dilate(glyph, np.ones((7, 7), np.uint8))
                session._watermark_detector.detect.return_value = Mock(mask=glyph, exclusion_mask=exclusion)
                result = session._compute_tracking_pixels(pixels)
                session._watermark_detector.detect.assert_called_once_with(pixels)
                self.assertGreater(result["watermark_excluded_fraction"], 0)
                expected = session._watermark_project_exclusion(exclusion)
                np.testing.assert_array_equal(result["watermark_exclusion_warp"], expected)

    def test_legacy_phone_watermark_api_also_skips_only_the_verified_owned_source(self):
        session = self.session(phone=True, metadata={"source": "sarween-phone-usb", "transport": "usb"})
        glyph, exclusion = session._watermark_observation(np.ones((120, 160, 3), np.uint8))
        self.assertEqual(glyph.shape, (120, 160))
        self.assertFalse(np.any(glyph))
        self.assertFalse(np.any(exclusion))
        session._watermark_detector.detect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
