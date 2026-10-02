import json
from pathlib import Path
import unittest

import cv2
import numpy as np

from scripts.review_tracking_video import cell_at, geometry_at, registration, scene_matrix
from tracking_evaluation import validate_case


class VideoReviewTests(unittest.TestCase):
    def setUp(self):
        self.scene = {"sceneId": "a", "width": 500, "height": 400}
        self.view = {"sceneId": "a", "registration": {"left": 0, "top": 0, "right": 500, "bottom": 400},
                     "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
                     "gridSize": 50, "gridOriginX": 0, "gridOriginY": 0}

    def test_manual_point_maps_with_pan_zoom_and_grid_origin(self):
        self.assertEqual(cell_at(self.view, self.scene, 501, 401, 125, 75), "C2")
        self.view["clientToCanvasTransform"].update(a=2, d=2, tx=-50, ty=-50)
        self.view.update(gridOriginX=10, gridOriginY=10)
        self.assertEqual(cell_at(self.view, self.scene, 501, 401, 125, 75), "D2")
        self.assertIsNone(cell_at(self.view, self.scene, 501, 401, -1, 75))
        self.assertIsNone(cell_at(self.view, self.scene, 501, 401, 500, 75))

    def test_geometry_does_not_use_predictions_or_future_events(self):
        timeline = {"trackingEvents": [{"to": "WRONG99"}], "events": [
            {"frame": 0, "sceneInfo": self.scene, "viewTransform": self.view},
            {"frame": 5, "sceneInfo": {**self.scene, "sceneId": "b"}},
            {"frame": 8, "viewTransform": {**self.view, "sceneId": "b"}}]}
        self.assertEqual(geometry_at(timeline, 4), (self.scene, self.view))
        with self.assertRaises(ValueError):
            geometry_at(timeline, 5)
        self.assertEqual(geometry_at(timeline, 8)[1]["sceneId"], "b")

    def test_singular_transform_rejected(self):
        self.view["clientToCanvasTransform"].update(a=0, d=0)
        with self.assertRaises(ValueError):
            scene_matrix(self.view, 501, 401)

    def test_registration_uses_outer_oriented_corners_and_requires_all_four(self):
        class Detector:
            def detectMarkers(self, _):
                corners = [np.float32([[[x,y],[x+10,y],[x+10,y+10],[x,y+10]]])
                           for x,y in [(20,20),(470,20),(470,370),(20,370)]]
                return corners, np.array([[10],[11],[12],[13]]), None
        detector = Detector()
        matrix, seen = registration(np.zeros((401,501,3), np.uint8), detector, 501, 401)
        self.assertEqual(seen, [10,11,12,13])
        mapped = cv2.perspectiveTransform(np.float32([[[20,20],[480,20],[480,380],[20,380]]]), matrix)
        np.testing.assert_allclose(mapped, [[[0,0],[500,0],[500,400],[0,400]]], atol=1e-5)
        original = detector.detectMarkers
        detector.detectMarkers = lambda image: (original(image)[0][:3], np.array([[10],[11],[12]]), None)
        self.assertIsNone(registration(np.zeros((401,501,3), np.uint8), detector, 501, 401)[0])

    def test_reviewed_fixture_has_separate_provenance_and_all_placements(self):
        root = Path(__file__).parent / "fixtures"
        case = json.loads((root / "tracking_cases.reviewed.json").read_text())["cases"][0]
        validate_case(case)
        labels = json.loads((root / case["label_evidence"]).read_text())
        self.assertEqual(case["label_source"], "assistant-visually-reviewed")
        self.assertEqual([e["to"] for e in case["expectations"]], [p["cell"] for p in labels["placements"]])
        self.assertNotIn("ignore_before", case)
        self.assertFalse(case["allow_unexpected"])
        for expectation, placement in zip(case["expectations"][1:], labels["placements"][1:]):
            self.assertAlmostEqual(expectation["settled_at"], placement["clear_frame"] / labels["fps"], places=5)


if __name__ == "__main__":
    unittest.main()
