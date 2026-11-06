"""Caption matching checks; all inputs are generated images or saved video."""

from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from watermark_detection import BoardCaptionDetector, ObsWatermarkDetector


RECORDING = (Path.home() / "Library/Application Support/Sarween/Recordings"
             / "sarween_rec_20261103_160008.mp4")
LIVE003 = (Path.home() / "Library/Application Support/Sarween/Recordings/Observer"
           / "observer_20261105_145550_b6ef8a/recording_003")


class WatermarkDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = ObsWatermarkDetector()

    def caption_frame(self, x=740, y=450, size=(1080, 1920), alpha=1.0):
        height, width = size
        frame = np.full((height, width, 3), (30, 55, 65), np.uint8)
        template = cv2.resize(self.detector.templates[0], None,
                              fx=height / 1080, fy=height / 1080,
                              interpolation=cv2.INTER_NEAREST)
        th, tw = template.shape
        region = frame[y:y + th, x:x + tw]
        region[template > 0] = np.round(
            region[template > 0] * (1 - alpha) + 255 * alpha).astype(np.uint8)
        return frame, template

    def test_translated_known_caption_produces_only_local_exclusion(self):
        for x, y in ((40, 20), (740, 450), (1550, 820)):
            with self.subTest(x=x, y=y):
                frame, glyph = self.caption_frame(x, y)
                original = frame.copy()
                result = self.detector.detect(frame)
                self.assertTrue(result.detected)
                self.assertGreater(result.confidence, .7)
                self.assertLessEqual(abs(result.bbox[0] - x), 2)
                self.assertLessEqual(abs(result.bbox[1] - y), 2)
                covered = result.mask[y:y + glyph.shape[0], x:x + glyph.shape[1]]
                self.assertGreater(np.mean(covered[glyph > 0] > 0), .98)
                self.assertGreater(cv2.countNonZero(result.exclusion_mask), cv2.countNonZero(result.mask))
                self.assertEqual(result.mask[500, 1500], 0)
                np.testing.assert_array_equal(frame, original)

    def test_resized_feed_and_translucent_lettering(self):
        frame, _ = self.caption_frame(180, 140, size=(720, 1280), alpha=.75)
        result = self.detector.detect(frame)
        self.assertTrue(result.detected)
        self.assertEqual(result.mask.shape, frame.shape[:2])
        self.assertLessEqual(abs(result.bbox[0] - 180), 2)

    def test_empty_glare_grid_and_other_white_words_are_not_the_caption(self):
        images = [np.zeros((1080, 1920, 3), np.uint8),
                  np.full((1080, 1920, 3), 255, np.uint8)]
        image = np.full((1080, 1920, 3), 50, np.uint8)
        for x in range(0, 1920, 45):
            cv2.line(image, (x, 0), (x, 1079), (220, 220, 220), 1)
        for y in range(0, 1080, 45):
            cv2.line(image, (0, y), (1919, y), (220, 220, 220), 1)
        cv2.circle(image, (850, 400), 80, (255, 255, 255), -1)
        for index, text in enumerate(("CAMERA SETTINGS", "FOR PLAYERS", "STUDIO MODE")):
            cv2.putText(image, text, (500, 650 + index * 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 4)
        images.append(image)
        for frame in images:
            result = self.detector.detect(frame)
            self.assertFalse(result.detected)
            self.assertEqual(cv2.countNonZero(result.mask), 0)
            self.assertEqual(cv2.countNonZero(result.exclusion_mask), 0)

    def test_two_matching_lines_are_insufficient(self):
        frame, _ = self.caption_frame()
        frame[510:575, 740:990] = (30, 55, 65)
        result = self.detector.detect(frame)
        self.assertFalse(result.detected)
        self.assertEqual(cv2.countNonZero(result.mask), 0)

    def test_absent_observation_does_not_reuse_a_stale_caption(self):
        frame, _ = self.caption_frame()
        self.assertTrue(self.detector.detect(frame).detected)
        result = self.detector.detect(np.zeros_like(frame))
        self.assertFalse(result.detected)
        self.assertEqual(cv2.countNonZero(result.mask), 0)

    def test_incomplete_off_frame_caption_is_not_invented(self):
        # The current OBS feed keeps the complete caption inside its frame.
        # Partial captions need separate support, rather than an invented box.
        frame = np.full((1080, 1920, 3), 40, np.uint8)
        partial = self.detector.templates[0][:, :120]
        frame[450:630, 1800:1920][partial > 0] = 255
        result = self.detector.detect(frame)
        self.assertFalse(result.detected)
        self.assertEqual(cv2.countNonZero(result.exclusion_mask), 0)

    @unittest.skipUnless(RECORDING.is_file(), "Local manual-test recording is not distributed")
    def test_real_saved_caption_on_table_screen_and_caption_absence(self):
        cap = cv2.VideoCapture(str(RECORDING))
        try:
            for index, expected in ((0, True), (55, False), (220, True),
                                    (431, False), (537, True), (605, False), (825, True)):
                with self.subTest(frame=index):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, image = cap.read()
                    self.assertTrue(ok)
                    result = self.detector.detect(image)
                    self.assertEqual(result.detected, expected)
        finally:
            cap.release()

    def test_invalid_inputs_are_rejected(self):
        for frame in (None, np.zeros((100, 100), np.uint8),
                      np.zeros((100, 100, 3), np.float32)):
            with self.assertRaises(ValueError):
                self.detector.detect(frame)


class BoardCaptionDetectionTests(unittest.TestCase):
    def setUp(self):
        self.detector = BoardCaptionDetector()

    def scene(self, *, clipped=False, alpha=1., two_lines=False):
        image = np.full((720, 1280, 3), (30, 55, 65), np.uint8)
        glyph = self.detector.templates[0].copy()
        if two_lines:
            glyph[125:] = 0
        # A rotated, stretched, mildly projective camera overlay, independent
        # of any miniature, map square or manually reviewed destination.
        transform = np.array(((.12, 2.4, 150), (-2.0, .20, 430 if clipped else 610),
                              (.0001, .0004, 1.)), np.float64)
        projected = cv2.warpPerspective(glyph, transform, (1280, 720))
        blend = projected.astype(float)[:, :, None] / 255 * alpha
        image = np.rint(image * (1 - blend) + 255 * blend).astype(np.uint8)
        return image, projected

    def test_projective_rotated_caption_has_local_glyph_masks(self):
        image, glyphs = self.scene()
        before = image.copy()
        result = self.detector.detect(image)
        self.assertTrue(result.detected)
        self.assertGreater(np.mean(result.mask[glyphs > 200] > 0), .98)
        self.assertGreater(np.count_nonzero(result.exclusion_mask), np.count_nonzero(result.mask))
        self.assertEqual(result.exclusion_mask[600, 1100], 0)
        self.assertLess(np.mean(result.exclusion_mask > 0), .30)
        self.assertEqual(result.provenance["method"], "known-caption-homography-v1")
        self.assertEqual(len(result.provenance["line_correlations"]), 3)
        np.testing.assert_array_equal(image, before)

    def test_faded_and_partly_clipped_caption_remains_supported(self):
        image, glyphs = self.scene(clipped=True, alpha=.40)
        result = self.detector.detect(image)
        self.assertTrue(result.detected)
        self.assertLess(result.provenance["visible_fraction"], 1)
        self.assertGreater(np.mean(result.mask[glyphs > 200] > 0), .96)

    def test_two_lines_glare_grid_and_skin_do_not_authorize_masks(self):
        partial, _ = self.scene(two_lines=True)
        clutter = np.full((720, 1280, 3), (45, 65, 85), np.uint8)
        for x in range(20, 1280, 35):
            for y in range(20, 720, 35):
                cv2.putText(clutter, "AB", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            .2, (235, 235, 235), 1)
        cv2.circle(clutter, (650, 100), 100, (255, 255, 255), -1)
        cv2.ellipse(clutter, (1000, 320), (70, 300), 30, 0, 360, (90, 150, 220), -1)
        for image in (partial, clutter, np.zeros_like(clutter), np.full_like(clutter, 255)):
            with self.subTest(kind=int(image.mean())):
                result = self.detector.detect(image)
                self.assertFalse(result.detected)
                self.assertEqual(np.count_nonzero(result.exclusion_mask), 0)

    def test_no_stale_mask_and_no_global_rng_dependency(self):
        image, _ = self.scene()
        first = self.detector.detect(image)
        self.assertTrue(first.detected)
        cv2.setRNGSeed(2847)
        second = self.detector.detect(image)
        np.testing.assert_array_equal(first.exclusion_mask, second.exclusion_mask)
        self.assertEqual(first.provenance, second.provenance)
        absent = self.detector.detect(np.zeros_like(image))
        self.assertFalse(absent.detected)
        self.assertEqual(np.count_nonzero(absent.exclusion_mask), 0)

    def test_masks_and_registration_repeat_in_fresh_processes(self):
        image, _ = self.scene(clipped=True)
        code = ("import cv2,hashlib,json,sys; "
                "from watermark_detection import BoardCaptionDetector; "
                "r=BoardCaptionDetector().detect(cv2.imread(sys.argv[1])); "
                "print(json.dumps({'mask':hashlib.sha256(r.exclusion_mask.tobytes()).hexdigest(),"
                "'provenance':r.provenance},sort_keys=True))")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "caption.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            results = [subprocess.check_output([sys.executable, "-c", code, str(path)],
                       cwd=Path(__file__).resolve().parents[1], text=True) for _ in range(2)]
        self.assertEqual(results[0], results[1])
        self.assertTrue(json.loads(results[0])["provenance"]["detected"])

    @unittest.skipUnless((LIVE003 / "inputs.jsonl").is_file(), "Local live journal is not distributed")
    def test_actual_unmasked_caption_transition_is_removed_but_hand_motion_remains(self):
        inputs = {row["frame_index"]: row for row in
                  map(json.loads, (LIVE003 / "inputs.jsonl").read_text().splitlines())}
        observations, images = {}, {}
        for frame in (1913, 1914, 1916, 1917, 1927, 1935):
            image = cv2.imread(str(LIVE003 / inputs[frame]["board"]))
            images[frame] = image
            observations[frame] = self.detector.detect(image)
        for frame in (1914, 1916, 1935):
            self.assertTrue(observations[frame].detected, frame)
        for frame in (1913, 1917, 1927):
            self.assertFalse(observations[frame].detected, frame)
        def remaining(previous, current):
            delta = np.max(cv2.absdiff(cv2.GaussianBlur(images[previous], (5, 5), 0),
                                      cv2.GaussianBlur(images[current], (5, 5), 0)), axis=2) > 35
            valid = ((observations[previous].exclusion_mask == 0)
                     & (observations[current].exclusion_mask == 0))
            return np.count_nonzero(delta), np.count_nonzero(delta & valid)
        before, after = remaining(1913, 1914)
        self.assertGreater(before, 60000)
        self.assertLess(after, 1000)
        # Caption disappearance plus a real entering hand must still hold.
        _before, after = remaining(1916, 1917)
        self.assertGreater(after, 15000)

    def test_invalid_board_inputs(self):
        for image in (None, np.zeros((100, 100), np.uint8),
                      np.zeros((100, 100, 3), np.float32)):
            with self.assertRaises(ValueError):
                self.detector.detect(image)


if __name__ == "__main__":
    unittest.main()
