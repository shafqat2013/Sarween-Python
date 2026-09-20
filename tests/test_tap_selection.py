import unittest

from tap_selection import TapGestureDetector


class TapGestureDetectorTest(unittest.TestCase):
    def test_short_touch_on_stationary_mini_emits_tap(self):
        detector = TapGestureDetector(cooldown_seconds=0)
        positions = {"red10": (100.0, 100.0)}
        self.assertIsNone(detector.update(0.0, {"red10"}, positions, 50))
        self.assertIsNone(detector.update(0.2, {"red10"}, {}, 50))
        self.assertEqual(detector.update(0.3, set(), positions, 50), "red10")

    def test_pickup_and_relocation_is_not_a_tap(self):
        detector = TapGestureDetector(cooldown_seconds=0)
        self.assertIsNone(detector.update(0.0, {"red10"}, {"red10": (100, 100)}, 50))
        self.assertIsNone(detector.update(0.3, set(), {}, 50))
        self.assertIsNone(detector.update(0.6, set(), {"red10": (150, 100)}, 50))

    def test_long_contact_is_rejected(self):
        detector = TapGestureDetector(cooldown_seconds=0)
        positions = {"red10": (100, 100)}
        detector.update(0.0, {"red10"}, positions, 50)
        self.assertIsNone(detector.update(1.0, {"red10"}, positions, 50))
        self.assertIsNone(detector.update(1.1, set(), positions, 50))

    def test_ambiguous_contact_is_ignored(self):
        detector = TapGestureDetector(cooldown_seconds=0)
        positions = {"red10": (100, 100), "blue": (120, 100)}
        self.assertIsNone(detector.update(0.0, {"red10", "blue"}, positions, 50))
        self.assertIsNone(detector.update(0.2, set(), positions, 50))


if __name__ == "__main__":
    unittest.main()
