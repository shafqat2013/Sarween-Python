import unittest

from marker_geometry import marker_geometry, normalized_marker_settings


class MarkerGeometryTests(unittest.TestCase):
    def test_legacy_recording_keeps_exact_existing_geometry(self):
        self.assertEqual(marker_geometry({"viewportMarkerSize": 120}),
                         {"patternSize": 80, "margin": 20, "totalSize": 120})
        self.assertEqual(marker_geometry({"viewportMarkerSize": 72}),
                         {"patternSize": 48, "margin": 12, "totalSize": 72})

    def test_margin_changes_total_without_changing_pattern(self):
        for margin in (0, 3, 12):
            result = marker_geometry({"viewportMarkerPatternSize": 48, "viewportMarkerMargin": margin})
            self.assertEqual(result["patternSize"], 48)
            self.assertEqual(result["totalSize"], 48 + margin * 2)

    def test_small_patterns_and_maximum_new_total_are_accepted(self):
        self.assertEqual(marker_geometry({"viewportMarkerPatternSize": 1, "viewportMarkerMargin": 0})["totalSize"], 1)
        # New state includes the derived total as its backwards-compatible key.
        self.assertEqual(marker_geometry({"viewportMarkerPatternSize": 240, "viewportMarkerMargin": 12,
                                          "viewportMarkerSize": 264})["totalSize"], 264)
        self.assertEqual(normalized_marker_settings({"viewportMarkerPatternSize": 240,
            "viewportMarkerMargin": 12, "viewportMarkerSize": 264}), {
            "viewportMarkerPatternSize": 240, "viewportMarkerMargin": 12})

    def test_normalizing_a_profile_does_not_rewrite_the_original(self):
        original = {"viewportMarkerSize": 120, "targetGridInches": 1}
        self.assertEqual(normalized_marker_settings(original), {
            "viewportMarkerPatternSize": 80, "viewportMarkerMargin": 12, "targetGridInches": 1})
        self.assertEqual(original, {"viewportMarkerSize": 120, "targetGridInches": 1})
        self.assertEqual(normalized_marker_settings({"targetGridInches": 1}), {"targetGridInches": 1})
        self.assertEqual(normalized_marker_settings({"viewportMarkerMargin": 3}), {"viewportMarkerMargin": 3})
        self.assertEqual(normalized_marker_settings({"viewportMarkerPatternSize": 48}), {"viewportMarkerPatternSize": 48})

    def test_rejects_invalid_widths_margins_and_nonfinite_numbers(self):
        for width in (0, -1, 241, float("nan"), True):
            with self.subTest(width=width), self.assertRaises(ValueError):
                marker_geometry({"viewportMarkerPatternSize": width})
        for margin in (-1, 13, float("inf"), True):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                marker_geometry({"viewportMarkerPatternSize": 48, "viewportMarkerMargin": margin})
