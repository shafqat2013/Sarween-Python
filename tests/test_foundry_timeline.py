import json
import tempfile
import unittest
from pathlib import Path

import foundryoutput as foundry
import tracking_regression


class FoundryTimelineTest(unittest.TestCase):
    def setUp(self):
        self.original_scene = {
            "scene_id": foundry.SCENE_ID,
            "scene_w": foundry.SCENE_W,
            "scene_h": foundry.SCENE_H,
            "grid_px": foundry.GRID_PX,
            "shift_x": foundry.SHIFT_X,
            "shift_y": foundry.SHIFT_Y,
            "grid_cols": foundry._grid_cols,
            "grid_rows": foundry._grid_rows,
        }
        foundry.stop_timeline_recording()
        foundry.clear_view_transform()
        foundry.set_test_sequence({"targets": []})
        foundry.set_scene_params("timeline-scene", 1000, 800, 50, 0, 0)

    def tearDown(self):
        foundry.stop_timeline_recording()
        foundry.clear_view_transform()
        foundry.set_test_sequence({"targets": []})
        foundry.SCENE_ID = self.original_scene["scene_id"]
        foundry.SCENE_W = self.original_scene["scene_w"]
        foundry.SCENE_H = self.original_scene["scene_h"]
        foundry.GRID_PX = self.original_scene["grid_px"]
        foundry.SHIFT_X = self.original_scene["shift_x"]
        foundry.SHIFT_Y = self.original_scene["shift_y"]
        foundry._grid_cols = self.original_scene["grid_cols"]
        foundry._grid_rows = self.original_scene["grid_rows"]

    @staticmethod
    def view_payload(tx=0):
        return {
            "type": "viewTransform",
            "sceneId": "timeline-scene",
            "viewportWidth": 1000,
            "viewportHeight": 800,
            "registration": {
                "left": 0,
                "top": 0,
                "right": 1000,
                "bottom": 800,
            },
            "canvasTransform": {
                "a": 1,
                "b": 0,
                "c": 0,
                "d": 1,
                "tx": -tx,
                "ty": 0,
            },
            "clientToCanvasTransform": {
                "a": 1,
                "b": 0,
                "c": 0,
                "d": 1,
                "tx": tx,
                "ty": 0,
            },
            "gridOriginX": 0,
            "gridOriginY": 0,
            "gridSize": 50,
        }

    def test_record_and_replay_frame_keyed_foundry_state(self):
        foundry.set_view_transform(self.view_payload())
        foundry.set_test_sequence(
            {
                "sceneId": "timeline-scene",
                "targets": [
                    {"step": 1, "row": 2, "column": 3, "cell": "D3"},
                    {"step": 2, "row": 5, "column": 7, "cell": "H6"},
                ],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "session.mp4"
            sidecar = foundry.start_timeline_recording(
                video,
                fps=15,
                frame_width=1920,
                frame_height=1080,
                marker_mode="viewport",
                warp_width=1280,
                warp_height=720,
                grid_cols=20,
                grid_rows=16,
            )
            foundry.record_timeline_frame()
            foundry.set_view_transform(self.view_payload(tx=100))
            foundry.record_timeline_frame()
            foundry.record_tracking_event("red10", "r2c3")
            foundry.mark_scene_visual_changed("tokenMove")
            foundry.record_timeline_frame()
            self.assertEqual(foundry.stop_timeline_recording(), sidecar)

            data = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertEqual(data["markerMode"], "viewport")
            self.assertTrue(data["framesUndistorted"])
            self.assertEqual(data["warpWidth"], 1280)
            self.assertEqual(data["framesRecorded"], 3)
            self.assertEqual([event["frame"] for event in data["events"]], [0, 1, 2])
            self.assertEqual(
                [target["cell"] for target in data["events"][0]["testSequence"]["targets"]],
                ["D3", "H6"],
            )
            self.assertEqual(
                data["trackingEvents"],
                [
                    {
                        "frame": 1,
                        "mini": "red10",
                        "cell": "r2c3",
                        "source": "detection",
                    }
                ],
            )

            foundry.clear_view_transform()
            replay = tracking_regression.FoundryTimelineReplay(sidecar)
            replay.apply_through(0)
            self.assertEqual(foundry.warp_to_grid_cell(25, 25, 1001, 801), (0, 0))
            replay.apply_through(1)
            self.assertEqual(foundry.warp_to_grid_cell(25, 25, 1001, 801), (2, 0))


if __name__ == "__main__":
    unittest.main()
