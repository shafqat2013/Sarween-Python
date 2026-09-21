import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import foundryoutput as fo
from rendered_reference import candidate_evidence, difference_mask, load_references, matching_view
from v3_tracking import ComboDetection


class RenderedReferenceTest(unittest.TestCase):
    def test_difference_mask_handles_display_gain_and_detects_physical_ring(self):
        rng = np.random.default_rng(3)
        reference = rng.integers(35, 170, size=(360, 640, 3), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (41, 41), 0)
        camera = np.clip(reference.astype(np.float32) * 1.12 + 8, 0, 255).astype(np.uint8)
        cv2.circle(camera, (310, 180), 16, (20, 20, 235), -1)
        mask = difference_mask(camera, reference)
        ring = mask[166:194, 296:324]
        background = mask[:100, :100]
        self.assertGreater(np.count_nonzero(ring), 250)
        self.assertLess(np.count_nonzero(background), 50)

    def test_reference_is_stored_at_camera_frame_and_validated(self):
        fo.stop_timeline_recording()
        old_scene = fo.SCENE_ID
        fo.SCENE_ID = "reference-test"
        try:
            with tempfile.TemporaryDirectory() as temp:
                video = Path(temp) / "test.mp4"
                timeline = fo.start_timeline_recording(
                    video, fps=10, frame_width=1920, frame_height=1080,
                    marker_mode="viewport", warp_width=1280, warp_height=720,
                    grid_cols=48, grid_rows=27,
                )
                image = np.zeros((360, 640, 3), dtype=np.uint8)
                success, jpg = cv2.imencode(".jpg", image)
                self.assertTrue(success)
                payload = {
                    "sceneId": "reference-test",
                    "image": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii"),
                }
                self.assertFalse(fo.record_rendered_reference({**payload, "sceneId": "other"}))
                fo.record_timeline_frame()
                self.assertTrue(fo.record_rendered_reference(payload))
                self.assertFalse(fo.record_rendered_reference(payload))
                data = json.loads(timeline.read_text(encoding="utf-8"))
                self.assertEqual(data["referenceFrames"][0]["frame"], 1)
                self.assertEqual(len(load_references(timeline)), 1)
                self.assertFalse(matching_view(None, fo.get_view_transform_payload()))
                fo.stop_timeline_recording()
        finally:
            fo.stop_timeline_recording()
            fo.SCENE_ID = old_scene

    def test_candidate_evidence_uses_warp_to_snapshot_scale(self):
        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[80:100, 100:120] = 255
        detection = ComboDetection(
            mini_id="red10", cx=220, cy=180, lab_dist=1,
            contour_area=200, circularity=0.9, score=0.8,
            bbox=(200, 160, 40, 40), sampled_lab=(50, 30, 20),
        )
        rows = candidate_evidence(mask, {"red10": detection}, (1280, 720))
        self.assertEqual(rows[0]["mini"], "red10")
        self.assertGreater(rows[0]["candidateChangedPercent"], 80)


if __name__ == "__main__":
    unittest.main()
