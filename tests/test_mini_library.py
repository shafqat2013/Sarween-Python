import tempfile
import unittest
from pathlib import Path

import mini_library


class MiniLibraryTest(unittest.TestCase):
    def test_defaults_include_all_five_player_minis(self):
        library = mini_library.default_library()
        self.assertEqual(
            set(library["minis"]),
            {"red10", "blue", "yellow", "green", "white"},
        )

    def test_profile_points_migrate_once(self):
        library = mini_library.default_library()
        profiles = {
            "red10": {
                "lab": [20, 30, 40],
                "lab_curve": [[20, 30, 40], [60, 30, 40]],
                "brightness_steps": [30, 220],
            }
        }

        self.assertTrue(mini_library.sync_profile_samples(library, profiles))
        self.assertFalse(mini_library.sync_profile_samples(library, profiles))
        samples = library["minis"]["red10"]["samples"]
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[1]["conditions"]["display"], "display-brightness:220")

    def test_verified_samples_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "library.json"
            self.assertEqual(
                mini_library.add_verified_sample("blue", [50, 10, -35], path=path),
                "added",
            )
            self.assertEqual(
                mini_library.add_verified_sample("blue", [50.1, 10, -35], path=path),
                "duplicate",
            )
            loaded = mini_library.load_library(path)
            self.assertEqual(len(loaded["minis"]["blue"]["samples"]), 1)

    def test_rows_combine_scan_assignment_and_live_confidence(self):
        class Detection:
            score = 0.825

        rows = mini_library.library_rows(
            mini_library.default_library(),
            profiles={"red10": {"lab": [20, 30, 40]}},
            mappings={"red10": "token-1"},
            token_names={"token-1": "Red"},
            detections={"red10": Detection()},
            positions={"red10": "K14"},
        )
        red = next(row for row in rows if row["id"] == "red10")
        self.assertEqual(red["scanStatus"], "Ready")
        self.assertEqual(red["token"], "Red")
        self.assertAlmostEqual(red["confidence"], 0.825)
        self.assertEqual(red["position"], "K14")


if __name__ == "__main__":
    unittest.main()
