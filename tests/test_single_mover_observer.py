"""File-only checks for geometry, source clocks and genuine causal replay."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from single_mover_observer import BoardGeometry, load_inputs, observe, sha256


def view(width=400, height=240):
    return {"registration": {"left": 0, "top": 0, "right": width, "bottom": height},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
            "gridSize": 40, "gridOriginX": 0, "gridOriginY": 0}


class ObserverTests(unittest.TestCase):
    def test_continuous_position_near_border_does_not_invent_exact_cell(self):
        geometry = BoardGeometry(view(), (400, 240))
        self.assertEqual(geometry.describe((100, 100))["cell"], "C3")
        uncertain = geometry.describe((120, 100))
        self.assertIsNone(uncertain["cell"])
        self.assertEqual(uncertain["cells"], ["C3", "D3"])
        self.assertTrue(uncertain["boundary_uncertain"])
        self.assertEqual(geometry.describe((100, 120))["cells"], ["C3", "C4"])

    def test_grid_origin_and_view_transform_are_applied(self):
        transformed = view()
        transformed["clientToCanvasTransform"].update(a=2, d=2, tx=40, ty=40)
        transformed.update(gridOriginX=40, gridOriginY=40)
        self.assertEqual(BoardGeometry(transformed, (400, 240)).describe((50, 50))["cell"], "C3")

    def make_capture(self, folder):
        video = folder / "synthetic.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 2, (400, 240))
        self.assertTrue(writer.isOpened())
        for index in range(12):
            image = np.full((240, 400, 3), (150, 90, 40), np.uint8)
            center = (100, 100) if index < 4 else (260, 100)
            cv2.circle(image, center, 25, (35, 75, 180), -1)
            cv2.line(image, center, (center[0] - 43, center[1]), (35, 75, 180), 8)
            writer.write(image)
        writer.release()
        checkpoint = folder / "calibration.npz"
        np.savez(checkpoint, H_saved=np.eye(3, dtype=np.float32))
        timeline = video.with_suffix(".tracking.json")
        data = {"framesRecorded": 12, "framesUndistorted": True, "events": [{"viewTransform": view()}],
                "frameClocks": [{"frame": i, "captureTime": 100 + i / 8} for i in range(12)],
                "groundTruth": [{"mini": "wrong", "cell": "Z99"}],
                "trackingEvents": [{"mini": "wrong", "cell": "Z99"}]}
        timeline.write_text(json.dumps(data))
        initial = folder / "initial.json"
        initial.write_text(json.dumps({"kind": "declared-initial-bindings",
            "source": {"video_basename": video.name, "video_sha256": sha256(video),
                       "timeline_basename": timeline.name, "timeline_sha256": sha256(timeline),
                       "checkpoint": {"path": checkpoint.name, "sha256": sha256(checkpoint)}},
            "initial_frame": 0, "warp_size": [400, 240],
            "initialization_source": "synthetic-start-only", "initial_positions": {"piece": [100, 100]},
            "initial_cells": {"piece": "C3"}, "labels": {"piece": "Same target"}}))
        return video, initial

    def test_causal_replay_generates_move_without_ground_truth_or_encoded_clock(self):
        cv2.setNumThreads(0)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video, initial = self.make_capture(folder)
            report = observe(video, initial, folder / "result", write_preview=False)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["events"]), 1)
            event = report["events"][0]
            self.assertEqual((event["mini"], event["from_cell"], event["to_cell"]), ("piece", "C3", "G3"))
            self.assertLessEqual(event["confirmed_time"], 1.0)
            self.assertAlmostEqual(report["capture_duration_seconds"], 11 / 8)
            self.assertFalse(report["scope"]["llm_used_for_measurement"])
            self.assertTrue((folder / "result" / "movements.csv").is_file())

    def test_expected_endpoints_are_rejected_as_initialization_input(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video, initial = self.make_capture(folder)
            data = json.loads(initial.read_text())
            data["expected_moves"] = [{"to": "G3"}]
            initial.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "starting bindings only"):
                load_inputs(video, initial)

    def test_changed_source_fingerprint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            video, initial = self.make_capture(folder)
            with video.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "Video does not match"):
                load_inputs(video, initial)


if __name__ == "__main__":
    unittest.main()
