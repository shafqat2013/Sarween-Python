import unittest
from pathlib import Path

import foundryoutput as foundry


class FoundryViewTransformTest(unittest.TestCase):
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
        foundry.clear_view_transform()
        foundry.set_scene_params("test-scene", 1000, 800, 50, 0, 0)

    def tearDown(self):
        foundry.clear_view_transform()
        foundry.SCENE_ID = self.original_scene["scene_id"]
        foundry.SCENE_W = self.original_scene["scene_w"]
        foundry.SCENE_H = self.original_scene["scene_h"]
        foundry.GRID_PX = self.original_scene["grid_px"]
        foundry.SHIFT_X = self.original_scene["shift_x"]
        foundry.SHIFT_Y = self.original_scene["shift_y"]
        foundry._grid_cols = self.original_scene["grid_cols"]
        foundry._grid_rows = self.original_scene["grid_rows"]

    @staticmethod
    def payload(**overrides):
        payload = {
            "type": "viewTransform",
            "sceneId": "test-scene",
            "viewportWidth": 1200,
            "viewportHeight": 900,
            "registration": {
                "left": 100,
                "top": 50,
                "right": 1100,
                "bottom": 850,
            },
            "canvasTransform": {
                "a": 1,
                "b": 0,
                "c": 0,
                "d": 1,
                "tx": 0,
                "ty": 0,
            },
            "gridOriginX": 0,
            "gridOriginY": 0,
            "gridSize": 50,
        }
        payload.update(overrides)
        return payload

    def test_identity_transform_maps_warp_to_scene_cell(self):
        self.assertTrue(foundry.set_view_transform(self.payload()))
        self.assertEqual(foundry.warp_to_grid_cell(25, 25, 1001, 801), (2, 1))
        self.assertEqual(foundry.warp_grid_dimensions(1001, 801), (50.0, 50.0))

    def test_pan_and_zoom_map_stationary_physical_point(self):
        payload = self.payload()
        payload["canvasTransform"] = {
            "a": 2,
            "b": 0,
            "c": 0,
            "d": 2,
            "tx": -300,
            "ty": -100,
        }
        self.assertTrue(foundry.set_view_transform(payload))

        # Warp (650, 600) is browser (750, 650), which maps back to scene
        # (525, 375): the center of column 10, row 7.
        self.assertEqual(foundry.warp_to_grid_cell(650, 600, 1001, 801), (10, 7))
        self.assertEqual(foundry.warp_grid_dimensions(1001, 801), (100.0, 100.0))

    def test_foundry_client_conversion_takes_priority(self):
        payload = self.payload()
        payload["clientToCanvasTransform"] = {
            "a": 0.5,
            "b": 0,
            "c": 0,
            "d": 0.5,
            "tx": 150,
            "ty": 50,
        }
        self.assertTrue(foundry.set_view_transform(payload))
        self.assertEqual(foundry.warp_to_grid_cell(650, 600, 1001, 801), (10, 7))
        self.assertEqual(foundry.warp_grid_dimensions(1001, 801), (100.0, 100.0))

    def test_points_outside_scene_are_ignored(self):
        self.assertTrue(foundry.set_view_transform(self.payload(gridOriginX=200)))
        self.assertIsNone(foundry.warp_to_grid_cell(0, 0, 1001, 801))

    def test_revision_changes_only_when_transform_changes(self):
        start = foundry.get_view_transform_revision()
        payload = self.payload()
        self.assertTrue(foundry.set_view_transform(payload))
        first = foundry.get_view_transform_revision()
        self.assertEqual(first, start + 1)

        self.assertTrue(foundry.set_view_transform(payload))
        self.assertEqual(foundry.get_view_transform_revision(), first)

        payload["canvasTransform"] = {**payload["canvasTransform"], "tx": 1}
        self.assertTrue(foundry.set_view_transform(payload))
        self.assertEqual(foundry.get_view_transform_revision(), first + 1)

    def test_scene_visual_revision_increments(self):
        start = foundry.get_scene_visual_revision()
        revision = foundry.mark_scene_visual_changed("test")
        self.assertEqual(revision, start + 1)
        self.assertEqual(foundry.get_scene_visual_revision(), start + 1)

    def test_scene_switch_recalculates_grid_and_discards_old_view(self):
        foundry.set_view_transform(self.payload())
        foundry.set_scene_params("new-map", 2400, 1350, 50)
        self.assertEqual(foundry.get_scene_params()["gridCols"], 48)
        self.assertEqual(foundry.get_scene_params()["gridRows"], 27)
        self.assertIsNone(foundry.warp_to_grid_cell(25, 25, 1001, 801))
        foundry.set_view_transform(self.payload(sceneId="new-map"))
        self.assertEqual(foundry.warp_to_grid_cell(25, 25, 1001, 801), (2, 1))

    def test_token_moves_use_scene_padding_and_grid_origin(self):
        foundry.set_view_transform(self.payload(gridOriginX=250, gridOriginY=200))
        self.assertEqual(foundry._grid_to_pixels("C4"), (350, 350))


class ViewportMarkerAssetTest(unittest.TestCase):
    def test_assets_are_detectable_as_expected_ids(self):
        try:
            import cv2
        except ModuleNotFoundError:
            self.skipTest("OpenCV/cv2 is not installed in this Python environment")

        aruco = cv2.aruco
        detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_6X6_250),
            aruco.DetectorParameters(),
        )
        root = Path(__file__).resolve().parents[1] / "viewport_markers"
        for expected_id in (10, 11, 12, 13):
            image = cv2.imread(str(root / f"marker_{expected_id}.png"))
            self.assertIsNotNone(image)
            _, ids, _ = detector.detectMarkers(image)
            self.assertIsNotNone(ids)
            self.assertEqual(ids.flatten().tolist(), [expected_id])


if __name__ == "__main__":
    unittest.main()
