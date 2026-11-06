import json
import math
from pathlib import Path
import tempfile
import unittest

from printed_marker_sheet import build_sheet, dimensions, parameters_from_context


class SheetTest(unittest.TestCase):
    def test_54inch_fullscreen_marker_size_includes_quiet_zone(self):
        spec = dimensions(54, 1920, 1080, 120, 8)
        self.assertAlmostEqual(spec["plateMm"], 74.71581292)
        self.assertAlmostEqual(spec["blackSquareMm"], 49.810541947)
        self.assertAlmostEqual(spec["whiteBorderMm"] * 2 + spec["blackSquareMm"], spec["plateMm"])
        scaled = dimensions(54, 3840, 2160, 240, 16)
        self.assertAlmostEqual(scaled["plateMm"], spec["plateMm"])

    def test_current_context_requires_matching_settings_and_view(self):
        profile = {"settings": {"displayDiagonalInches": 54}}
        context = {"ready": True, "settings": {"viewportMarkerSize": 120, "viewportMarkerInset": 8},
                   "view": {"viewportWidth": 1920, "viewportHeight": 1080,
                            "registration": {"left": 28, "top": 28, "right": 1892, "bottom": 1052}}}
        args = parameters_from_context(profile, context)
        self.assertEqual(args["marker_size_px"], 120)
        context["settings"]["viewportMarkerSize"] = 160
        with self.assertRaises(ValueError):
            parameters_from_context(profile, context)
        with self.assertRaises(ValueError):
            parameters_from_context(profile, {"ready": False})
        with self.assertRaises(ValueError):
            parameters_from_context(None, context)

    def test_normal_runtime_without_reportlab_writes_four_vector_cards(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sheet.pdf"
            result = build_sheet(path, diagonal_inches=54, viewport_width=1920,
                                 viewport_height=1080, marker_size_px=120, engine="vector")
            data = path.read_bytes()
            self.assertTrue(data.startswith(b"%PDF-1.4"))
            self.assertIn(b"/Count 1", data)
            self.assertEqual(result["markerIds"], [20, 21, 22, 23])
            for identity in (20, 21, 22, 23):
                self.assertIn(str(identity).encode() + b" - ", data)
            self.assertTrue(data.endswith(b"%%EOF\n"))

    def test_large_markers_get_separate_pages_instead_of_shrinking(self):
        with tempfile.TemporaryDirectory() as folder:
            result = build_sheet(Path(folder)/"large.pdf", diagonal_inches=54, viewport_width=1920,
                                 viewport_height=1080, marker_size_px=240, engine="vector")
            self.assertEqual(result["pages"], 4)
            self.assertAlmostEqual(result["plateMm"], 149.43162584)

    def test_cells_are_real_opencv_dictionary_codes(self):
        import cv2
        import numpy as np
        path = Path(__file__).resolve().parents[1]/"assets/printed_markers/dictionary-cells.json"
        cells = json.loads(path.read_text())["markers"]
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
        for identity, values in cells.items():
            code = cv2.resize(np.array(values, np.uint8)*255, (240, 240), interpolation=cv2.INTER_NEAREST)
            board = np.full((400, 400), 255, np.uint8)
            board[80:320,80:320] = code
            _corners, ids, _rejected = detector.detectMarkers(board)
            self.assertEqual(ids.flatten().tolist(), [int(identity)])


if __name__ == "__main__":
    unittest.main()
