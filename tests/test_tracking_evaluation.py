import unittest

from tracking_evaluation import MovementEvent, evaluate_events, format_time, normalize_cell, validate_case


def event(mini="red", to="A1", at=1, source="detection", from_cell=None):
    return MovementEvent(mini, from_cell, to, round(at * 10), at, None, "r0c0", 0, 1, source)


def case(expectations, **kwargs):
    return {"video": "video.mp4", "expectations": expectations, **kwargs}


class TrackingEvaluationTest(unittest.TestCase):
    def test_overlapping_windows_use_both_events_in_order(self):
        result = evaluate_events(case([
            {"mini": "red", "to": "A1", "between": [0, 10]},
            {"mini": "red", "to": "A1", "between": [4, 6]},
        ]), [event(at=1), event(at=5)])
        self.assertTrue(result.ok)
        self.assertEqual([item["event_index"] for item in result.matches], [0, 1])

    def test_one_event_cannot_satisfy_two_expectations(self):
        expectation = {"mini": "red", "to": "A1", "at": 1}
        result = evaluate_events(case([expectation, expectation]), [event()])
        self.assertEqual(result.matched_expectations, 1)
        self.assertEqual(len(result.missing), 1)

    def test_minis_can_move_in_interleaved_order(self):
        result = evaluate_events(case([
            {"mini": "red", "to": "A1", "between": [0, 3]},
            {"mini": "blue", "to": "B1", "between": [0, 3]},
            {"mini": "red", "from": "A1", "to": "C1", "between": [3, 6]},
        ]), [event("blue", "B1", 1), event(at=2), event("red", "C1", 4, from_cell="A1")])
        self.assertTrue(result.ok)
        self.assertEqual(result.metrics["red"]["matched"], 2)
        self.assertEqual(result.metrics["blue"]["matched"], 1)

    def test_same_mini_cannot_match_reversed_route(self):
        result = evaluate_events(case([
            {"mini": "red", "to": "A1", "between": [0, 10]},
            {"mini": "red", "to": "B1", "between": [0, 10]},
        ]), [event(to="B1", at=1), event(to="A1", at=2)])
        self.assertEqual(result.matched_expectations, 1)

    def test_viewport_move_does_not_count_as_physical_detection(self):
        result = evaluate_events(case([{"mini": "red", "to": "A1", "at": 1}]), [event(source="viewportTransform")])
        self.assertEqual(len(result.missing), 1)
        self.assertEqual(len(result.unexpected), 1)

    def test_stationary_minis_and_unknown_id_false_positives(self):
        result = evaluate_events(case([], stationary=True), [event("blue")])
        self.assertFalse(result.ok)
        self.assertEqual(result.metrics["blue"]["extra"], 1)

    def test_wrong_origin_and_duplicate_emission_are_extra(self):
        result = evaluate_events(case([{"mini": "red", "from": "B1", "to": "A1", "at": 1}]),
                                 [event(from_cell="C1"), event(from_cell="B1"), event(from_cell="B1")])
        self.assertEqual(result.matched_expectations, 1)
        self.assertEqual(len(result.unexpected), 2)

    def test_scope_excludes_warmup_and_tail(self):
        result = evaluate_events(case([], stationary=True, ignore_before=2, ignore_after=3), [event(at=1), event(at=4)])
        self.assertTrue(result.ok)
        self.assertEqual(result.unexpected, [])

    def test_timing_error_is_not_claimed_to_be_latency(self):
        result = evaluate_events(case([{"mini": "red", "to": "A1", "at": 1}]), [event(at=1.5)])
        self.assertEqual(result.matches[0]["timing_error_seconds"], 0.5)
        self.assertIsNone(result.metrics["red"]["mean_response_seconds"])

    def test_verified_placement_time_allows_response_measurement(self):
        result = evaluate_events(case([{"mini": "red", "to": "A1", "between": [1, 3], "settled_at": 1.2}]), [event(at=1.5)])
        self.assertAlmostEqual(result.matches[0]["response_seconds"], 0.3)
        self.assertNotIn("timing_error_seconds", result.matches[0])

    def test_user_confirmation_is_not_exact_placement_time(self):
        result = evaluate_events(case([{"mini": "red", "to": "A1", "between": [1, 3], "confirmed_at": 2}]), [event(at=1.5)])
        self.assertEqual(result.matches[0]["confirmation_offset_seconds"], -0.5)
        self.assertNotIn("response_seconds", result.matches[0])

    def test_allow_unexpected_still_reports_extra_movements(self):
        result = evaluate_events(case([], stationary=True, allow_unexpected=True), [event()])
        self.assertTrue(result.ok)
        self.assertEqual(len(result.unexpected), 1)

    def test_invalid_expectations_fail_before_replay(self):
        bad = [{"mini": "red", "to": "A0", "at": 1}, {"mini": "red", "to": "A1"},
               {"mini": "red", "to": "A1", "between": [5, 1]},
               {"mini": "red", "to": "A1", "at": float("nan")}]
        for expected in bad:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                validate_case(case([expected]))
        with self.assertRaises(ValueError):
            validate_case(case([]))
        with self.assertRaisesRegex(ValueError, "must be true or false"):
            validate_case(case([], stationary=True, allow_unexpected="false"))

    def test_cell_normalization_and_millisecond_rollover(self):
        self.assertEqual(normalize_cell("r9c26"), "AA10")
        self.assertEqual(format_time(59.9999), "01:00.000")


if __name__ == "__main__":
    unittest.main()
