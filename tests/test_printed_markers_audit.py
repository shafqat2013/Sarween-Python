"""Lightweight inspection uses recorded paper layouts without a hardware store."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audit_video import AuditVideo
from tests.test_audit_video import Frames
from tests.test_printed_markers import detected, view
from tests.test_printed_markers_recording import registration


class PrintedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "recorded.mp4"
        self.path.write_bytes(b"file decoder is injected")
        self.timeline = {"markerMode": "viewport", "warpWidth": 1280, "warpHeight": 720,
            "markerRegistration": registration(), "events": [{"frame": 0, "viewTransform": view(),
                "sceneInfo": {"sceneId": "scene", "width": 2400, "height": 1350, "gridSize": 50, "gridType": 1}}]}

    def reader(self, data=None, outputs=None):
        outputs = outputs or [(*detected(), [])]
        detector = Mock()
        detector.detectMarkers.side_effect = outputs
        frames = Frames([np.zeros((1080, 1920, 3), np.uint8)] * len(outputs))
        with patch("audit_video.cv2.VideoCapture", return_value=frames), \
             patch("audit_video.cv2.aruco.ArucoDetector", return_value=detector), \
             patch("hardware_profiles.HardwareProfiles", side_effect=AssertionError("No user profiles")):
            reader = AuditVideo(self.path, data or self.timeline)
        self.addCleanup(reader.close)
        return reader

    def test_header_paper_geometry_registers_and_partial_inspection_is_marked_held(self):
        outputs = [(*detected(), []), (*detected(included=(20, 21)), [])]
        reader = self.reader(outputs=outputs)
        image, hint = reader.display(grid=False)
        self.assertEqual(image.shape, (720, 1280, 3))
        self.assertEqual(hint, "Board view")
        reader.seek(1)
        self.assertIn("using last board view", reader.display(grid=False)[1])

    def test_layout_change_cannot_reuse_old_paper_anchor(self):
        data = deepcopy(self.timeline)
        changed = registration()
        for points in changed["printedMarkerLayout"]["markerCorners"].values():
            for point in points:
                point[0] += .02
        data["events"].append({"frame": 1, "viewTransform": view(), "cameraLock": {"markerRegistration": changed}})
        reader = self.reader(data, outputs=[(*detected(), []), ([], None, [])])
        reader.seek(1)
        self.assertIsNone(reader._h)

    def test_checkpoint_metadata_supplies_layout_without_tracking_state_access(self):
        data = deepcopy(self.timeline)
        data.pop("markerRegistration")
        checkpoint = self.path.parent / "state.npz"
        metadata = np.frombuffer(json.dumps({"markerRegistration": registration()}).encode(), np.uint8)
        np.savez_compressed(checkpoint, H_saved=np.eye(3), metadata=metadata,
                            tracking=np.array({"ignored": True}, dtype=object))
        data["checkpoints"] = [{"path": checkpoint.name, "frame": 0,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}]
        reader = self.reader(data)
        self.assertFalse(reader._held)
        self.assertEqual(reader.display(grid=False)[0].shape, (720, 1280, 3))

    def test_invalid_printed_layout_never_falls_back_to_visible_screen_ids(self):
        data = deepcopy(self.timeline)
        data["markerRegistration"]["printedMarkerLayout"]["dictionary"] = "wrong"
        c, ids = detected(include_screen=True)
        reader = self.reader(data, outputs=[(c, ids, [])])
        image, hint = reader.display(grid=False)
        self.assertEqual(image.shape, (1080, 1920, 3))
        self.assertIn("printed registration unavailable", hint)


if __name__ == "__main__":
    unittest.main()
