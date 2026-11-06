import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import foundryoutput as fo
from reference_health import (CAPTURE_METHOD, capture_provenance, image_quality,
                              matching_view, reference_status, validated_view)
from rendered_reference import compare_video, load_references, reference_for_camera


def view(tx=0):
    return {"type": "viewTransform", "sceneId": "ref-health", "viewportWidth": 1920,
            "viewportHeight": 1080, "registration": {"left": 28, "top": 28, "right": 1892, "bottom": 1052},
            "canvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": tx, "ty": 0},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": -tx, "ty": 0},
            "gridOriginX": 0, "gridOriginY": 0, "gridSize": 50}


def capture(epoch=999900):
    return {"method": CAPTURE_METHOD, "sessionId": "session-1", "sequence": 1,
            "capturedAtEpochMs": epoch, "durationMs": 4, "sourceWidth": 1920, "sourceHeight": 1080}


def patterned_image():
    image = np.full((360, 640, 3), 40, np.uint8)
    cv2.rectangle(image, (50, 50), (500, 250), (80, 160, 200), -1)
    return image


def payload(image=None, **overrides):
    success, jpg = cv2.imencode(".jpg", patterned_image() if image is None else image)
    assert success
    return {"sceneId": "ref-health", "viewTransform": view(), "capture": capture(),
            "image": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii"), **overrides}


class ReferenceHealthTest(unittest.TestCase):
    def test_black_and_white_are_retained_as_low_information_not_proven_failed_exports(self):
        for value in (0, 1, 255):
            quality = image_quality(np.full((360, 640, 3), value, np.uint8))
            self.assertFalse(quality["informative"])
            self.assertEqual(quality["status"], "low_information")
            self.assertIn("legitimate fog", quality["note"])
        self.assertTrue(image_quality(patterned_image())["informative"])

    def test_geometry_validation_copies_source_and_rejects_bad_bounds_and_matrix(self):
        original = view()
        copied = validated_view(original, "ref-health")
        original["registration"]["left"] = 99
        self.assertEqual(copied["registration"]["left"], 28)
        invalid = view(); invalid["registration"]["right"] = 2000
        with self.assertRaises(ValueError): validated_view(invalid, "ref-health")
        invalid = view(); invalid["clientToCanvasTransform"]["a"] = 0
        with self.assertRaises(ValueError): validated_view(invalid, "ref-health")
        invalid = view(); invalid["gridSize"] = float("nan")
        with self.assertRaises(ValueError): validated_view(invalid, "ref-health")
        self.assertFalse(matching_view(view(), view(50)))

    def test_provenance_requires_explicit_target_clock_and_bounds(self):
        self.assertTrue(capture_provenance(capture(), received_epoch_ms=1000000)["verified"])
        self.assertFalse(capture_provenance(None, received_epoch_ms=1000000)["verified"])
        for bad in ({**capture(), "method": "default_framebuffer"}, capture(epoch=1001000),
                    capture(epoch=0), {**capture(), "sourceWidth": 10**9}):
            self.assertFalse(capture_provenance(bad, received_epoch_ms=1000000)["verified"])

    def entry(self):
        return {"viewTransform": view(), "geometryValid": True,
                "quality": image_quality(patterned_image()), "receivedAtMonotonic": 100,
                "provenance": {"verified": True, "ageAtReceiptSeconds": .1}}

    def test_staleness_includes_age_before_receipt_and_does_not_certify_alignment(self):
        entry = self.entry()
        fresh = reference_status(entry, current_view=view(), now_monotonic=100.3)
        self.assertEqual(fresh["state"], "available")
        self.assertAlmostEqual(fresh["captureAgeSeconds"], .4)
        self.assertFalse(fresh["displayAlignmentVerified"])
        entry["provenance"]["ageAtReceiptSeconds"] = 1
        stale = reference_status(entry, current_view=view(), now_monotonic=100.01)
        self.assertEqual(stale["state"], "stale")
        self.assertFalse(stale["usableForComparison"])

    def test_geometry_loss_legacy_and_low_information_are_exposed(self):
        entry = self.entry()
        self.assertEqual(reference_status(entry, current_view=view(1), now_monotonic=100)["state"], "geometry_changed")
        entry["quality"] = image_quality(np.zeros((360, 640, 3), np.uint8))
        self.assertEqual(reference_status(entry, current_view=view(), now_monotonic=100)["state"], "low_information")
        entry["provenance"] = {"verified": False}
        self.assertEqual(reference_status(entry, current_view=view(), now_monotonic=100)["state"], "unverified_capture")

    def test_reference_selection_is_past_only_recent_and_geometry_matched(self):
        entry = {**self.entry(), "sceneId": "ref-health", "frame": 10}
        references = {10: (entry, patterned_image())}
        self.assertIsNone(reference_for_camera(references, 99.8, view()))
        self.assertIsNone(reference_for_camera(references, 101, view()))
        self.assertIsNone(reference_for_camera(references, 100, view(1)))
        chosen = reference_for_camera(references, 100, view())
        self.assertIsNotNone(chosen)
        self.assertAlmostEqual(chosen[2], .1)
        entry["provenance"]["verified"] = False
        self.assertIsNone(reference_for_camera(references, 100, view()))


class ReferenceRecorderTest(unittest.TestCase):
    def setUp(self):
        self.scene, self.saved_view = fo.SCENE_ID, fo.get_view_transform_payload()
        self.health = copy.deepcopy(fo._rendered_reference_status_entry)
        self.latest = copy.deepcopy(fo._latest_rendered_reference)
        fo.SCENE_ID = "ref-health"
        fo.set_view_transform(view())

    def tearDown(self):
        fo.stop_timeline_recording()
        fo.SCENE_ID = self.scene
        if self.saved_view: fo.set_view_transform(self.saved_view)
        else: fo.clear_view_transform()
        fo._rendered_reference_status_entry = self.health
        fo._latest_rendered_reference = self.latest

    def test_recording_preserves_captured_geometry_when_receiver_view_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = fo.start_timeline_recording(Path(directory)/"capture.mp4", fps=10,
                frame_width=1920, frame_height=1080, marker_mode="viewport",
                warp_width=1280, warp_height=720, grid_cols=48, grid_rows=27)
            fo.record_timeline_frame(capture_time=99.8)
            fo.set_view_transform(view(300))
            with patch.object(fo.time, "perf_counter", return_value=100), patch.object(fo.time, "time", return_value=1000):
                self.assertTrue(fo.record_rendered_reference(payload()))
                status = fo.get_rendered_reference_status()
            entry = json.loads(path.read_text())["referenceFrames"][0]
            self.assertEqual(entry["viewTransform"], view())
            self.assertEqual(fo.get_view_transform_payload()["canvasTransform"]["tx"], 300)
            self.assertIsNone(entry["viewRevision"])
            self.assertEqual(entry["frame"], 0)
            self.assertEqual(entry["latestCameraCaptureTime"], 99.8)
            self.assertEqual(entry["capture"]["capturedAtEpochMs"], 999900)
            self.assertAlmostEqual(entry["provenance"]["ageAtReceiptSeconds"], .1)
            self.assertEqual(status["state"], "geometry_changed")
            fo.stop_timeline_recording()

    def test_health_updates_without_standard_video_recording(self):
        with patch.object(fo.time, "perf_counter", return_value=100), patch.object(fo.time, "time", return_value=1000):
            self.assertFalse(fo.record_rendered_reference(payload()))
            self.assertEqual(fo.get_rendered_reference_status()["state"], "available")

    def test_latest_jpeg_is_bounded_and_metadata_is_not_shared_with_caller(self):
        source = payload()
        with patch.object(fo.time, "perf_counter", return_value=100), patch.object(fo.time, "time", return_value=1000):
            fo.record_rendered_reference(source)
        latest = fo.get_latest_rendered_reference()
        self.assertIsInstance(latest["image"], bytes)
        self.assertLess(len(latest["image"]), 350000)
        self.assertEqual(latest["entry"]["capture"]["sequence"], 1)
        latest["entry"]["viewTransform"]["registration"]["left"] = 999
        source["capture"]["sequence"] = 999
        self.assertEqual(fo.get_latest_rendered_reference()["entry"]["viewTransform"]["registration"]["left"], 28)
        self.assertEqual(fo.get_latest_rendered_reference()["entry"]["capture"]["sequence"], 1)
        with patch.object(fo.time, "perf_counter", return_value=100), patch.object(fo.time, "time", return_value=1000):
            fo.record_rendered_reference(payload(np.zeros((360, 640, 3), np.uint8)))
        self.assertNotEqual(fo.get_latest_rendered_reference()["image"], latest["image"])

    def test_invalid_image_exposes_error_and_uniform_payload_is_not_claimed_usable(self):
        with patch.object(fo.time, "perf_counter", return_value=100), patch.object(fo.time, "time", return_value=1000):
            fo.record_rendered_reference(payload(np.zeros((360, 640, 3), np.uint8)))
            self.assertEqual(fo.get_rendered_reference_status()["state"], "low_information")
            self.assertFalse(fo.record_rendered_reference(payload(np.zeros((100, 100, 3), np.uint8))))
            self.assertEqual(fo.get_rendered_reference_status()["state"], "capture_error")

    def test_legacy_black_reference_cannot_produce_a_successful_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root/"black.jpg"), np.full((360, 640, 3), 1, np.uint8))
            timeline = root/"test.tracking.json"
            timeline.write_text(json.dumps({"referenceFrames": [{"frame": 0, "path": "black.jpg"}]}))
            self.assertEqual(len(load_references(timeline)), 1)
            with self.assertRaisesRegex(ValueError, "No usable rendered reference evidence"):
                compare_video(root/"test.mp4", timeline)


if __name__ == "__main__":
    unittest.main()
