"""Independent state tests: accepted measurements, no ring colors or image labels."""

import math
import unittest

from single_mover_state import SingleMover


def candidate(x, y, **values):
    return {"cx": x, "cy": y, "radius": 12, "confidence": .9,
            "uncertainty_px": 1.2, **values}


class SingleMoverTests(unittest.TestCase):
    def make(self):
        return SingleMover({"one": (20, 20), "two": (80, 20)}, grid_px=25)

    def test_identical_bases_move_in_sequence_without_color_identity(self):
        state = self.make()
        self.assertEqual(state.update([candidate(20, 20), candidate(80, 20)], 0), [])
        self.assertEqual(state.update([candidate(80, 20)], .1), [])
        self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], .2), [])
        move, = state.update([candidate(45, 60), candidate(80, 20)], .4)
        self.assertEqual((move["mini"], move["from_xy"], move["to_xy"]),
                         ("one", (20., 20.), (45., 60.)))
        self.assertEqual(move["departure_time"], .1)
        self.assertEqual(move["first_visible_time"], .2)
        self.assertEqual(move["confirmed_time"], .4)
        self.assertEqual(state.positions["two"], (80, 20))
        self.assertEqual(state.update([candidate(45, 60), candidate(80, 20)], .5), [])
        self.assertEqual(state.update([candidate(45, 60), candidate(105, 60)], .6), [])
        move, = state.update([candidate(45, 60), candidate(105, 60)], .8)
        self.assertEqual(move["mini"], "two")
        self.assertEqual(move["from_xy"], (80, 20))

    def test_no_move_when_extra_candidate_appears_but_layout_remains(self):
        state = self.make()
        frame = [candidate(20, 20), candidate(80, 20), candidate(130, 90)]
        for timestamp in (0, .3, .6):
            self.assertEqual(state.update(frame, timestamp), [])
        self.assertEqual(state.status["reason"], "extra_candidates_with_full_layout")
        self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})

    def test_two_missing_anchors_and_one_candidate_do_not_choose_identity(self):
        state = self.make()
        for timestamp in (0, .3, .6):
            self.assertEqual(state.update([candidate(100, 90)], timestamp), [])
        self.assertEqual(state.status["reason"], "multiple_missing_anchors")
        self.assertIsNone(state.status["active_mini"])

    def test_two_movers_do_not_become_one_inferred_move(self):
        state = self.make()
        for timestamp in (0, .3, .6):
            self.assertEqual(state.update([candidate(45, 60), candidate(105, 60)], timestamp), [])
        self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})

    def test_competing_destinations_clear_stability_window(self):
        state = self.make()
        clear = [candidate(80, 20), candidate(45, 60)]
        state.update(clear, 0)
        state.update(clear + [candidate(130, 90)], .15)
        self.assertEqual(state.status["reason"], "competing_destinations")
        self.assertEqual(state.update(clear, .3), [])
        move, = state.update(clear, .5)
        self.assertEqual(move["first_visible_time"], .3)

    def test_obstructing_hand_cannot_be_destination_and_interrupts_settling(self):
        state = self.make()
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 0)
        self.assertEqual(state.update([candidate(150, 100)], .3, obstructed=True), [])
        self.assertEqual(state.positions["one"], (20, 20))
        self.assertEqual(state.update(frame, .4), [])
        move, = state.update(frame, .6)
        self.assertEqual(move["first_visible_time"], .4)

    def test_missing_measurement_interrupts_settling(self):
        state = self.make()
        state.update([candidate(80, 20), candidate(45, 60)], 0)
        state.update([candidate(80, 20)], .15)
        self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], .3), [])
        move, = state.update([candidate(80, 20), candidate(45, 60)], .5)
        self.assertEqual(move["first_visible_time"], .3)

    def test_candidate_jump_resets_settling_without_committing_intermediate_point(self):
        state = self.make()
        state.update([candidate(80, 20), candidate(45, 60)], 0)
        self.assertEqual(state.update([candidate(80, 20), candidate(70, 70)], .3), [])
        move, = state.update([candidate(80, 20), candidate(70, 70)], .5)
        self.assertEqual(move["to_xy"], (70, 70))
        self.assertEqual(move["first_visible_time"], .3)

    def test_tiny_jitter_does_not_drift_confirmed_layout(self):
        state = self.make()
        for timestamp, offset in enumerate((2, -3, 4, -2)):
            self.assertEqual(state.update([candidate(20 + offset, 20), candidate(80, 20)], timestamp), [])
        self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})

    def test_duplicate_measurements_cannot_be_assigned_twice(self):
        state = self.make()
        frame = [candidate(80, 20), candidate(80.5, 20.2), candidate(45, 60), candidate(45, 60)]
        state.update(frame, 0)
        move, = state.update(frame, .3)
        self.assertEqual(move["mini"], "one")
        self.assertEqual(state.update(frame, .6), [])

    def test_ambiguous_shared_measurement_cannot_satisfy_two_anchors(self):
        state = SingleMover({"a": (20, 20), "b": (35, 20)}, grid_px=25)
        for timestamp in (0, .3):
            self.assertEqual(state.update([candidate(27, 20)], timestamp), [])
        self.assertEqual(state.status["reason"], "ambiguous_anchor_match")

    def test_unavailable_stationary_anchor_is_preserved_without_global_freeze(self):
        state = self.make()
        frame = [candidate(45, 60)]
        state.update(frame, 0, unavailable={"two"})
        move, = state.update(frame, .3, unavailable={"two"})
        self.assertEqual(move["mini"], "one")
        self.assertEqual(state.positions["two"], (80, 20))
        self.assertEqual(state.status["unavailable"], ["two"])

    def test_unavailable_active_anchor_cannot_switch_identity(self):
        state = self.make()
        state.update([candidate(80, 20), candidate(45, 60)], 0)
        self.assertEqual(state.update([candidate(105, 60)], .3, unavailable={"one"}), [])
        self.assertEqual(state.status["reason"], "active_mini_unavailable")
        self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})

    def test_restored_origin_cancels_pending_move(self):
        state = self.make()
        state.update([candidate(80, 20), candidate(45, 60)], 0)
        self.assertEqual(state.update([candidate(20, 20), candidate(80, 20)], .3), [])
        self.assertIsNone(state.status["pending"])
        self.assertEqual(state.status["reason"], "layout_confirmed")

    def test_continuous_positions_and_uncertainty_are_not_snapped_to_cells(self):
        state = self.make()
        state.update([candidate(80, 20), candidate(49.6, 61.0, uncertainty_px=2.7)], 0)
        move, = state.update([candidate(80, 20), candidate(50.2, 60.8, uncertainty_px=2.7)], .3)
        self.assertAlmostEqual(move["to_xy"][0], 49.9)
        self.assertGreaterEqual(move["uncertainty_px"], 2.7)

    def test_weak_measurements_cannot_start_or_sustain_settling(self):
        state = self.make()
        for timestamp in (0, .3):
            self.assertEqual(state.update([candidate(80, 20), candidate(45, 60, confidence=.1)], timestamp), [])
        self.assertEqual(state.status["reason"], "destination_not_visible")

    def test_clock_validation_happens_before_state_mutation(self):
        state = self.make()
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 1)
        before = dict(state.status)
        for timestamp in (.5, math.nan, math.inf):
            with self.assertRaises(ValueError):
                state.update(frame, timestamp)
        self.assertEqual(state.status, before)
        self.assertEqual(state.update(frame, 1), [])
        move, = state.update(frame, 1.3)
        self.assertEqual(move["mini"], "one")

    def test_large_capture_gap_cannot_count_as_observed_settling(self):
        state = self.make()
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 0)
        self.assertEqual(state.update(frame, 2), [])
        move, = state.update(frame, 2.3)
        self.assertEqual(move["first_visible_time"], 2)

    def slow_layout(self, state):
        origins = [candidate(20, 20), candidate(80, 20)]
        for index in range(10):
            self.assertEqual(state.update(origins, index * .6), [])

    def test_regular_slow_capture_confirms_each_move_from_distinct_observations(self):
        state = self.make()
        self.slow_layout(state)
        self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], 6.0), [])
        first, = state.update([candidate(80, 20), candidate(45, 60)], 6.6)
        self.assertEqual((first["mini"], first["from_xy"], first["to_xy"], first["samples"]),
                         ("one", (20, 20), (45, 60), 2))
        self.assertEqual(state.update([candidate(45, 60), candidate(105, 60)], 7.2), [])
        second, = state.update([candidate(45, 60), candidate(105, 60)], 7.8)
        self.assertEqual((second["mini"], second["from_xy"], second["to_xy"]),
                         ("two", (80, 20), (105, 60)))

    def test_legacy_policy_reproduces_slow_capture_confirmation_starvation(self):
        state = SingleMover({"one": (20, 20), "two": (80, 20)}, 25,
                            timing_policy="fixed-gap-v1")
        self.slow_layout(state)
        for index in range(7):
            self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], 6 + index * .6), [])
            self.assertEqual(state.status["pending"]["samples"], 1)
        self.assertNotIn("timing", state.status)

    def test_a_sudden_gap_cannot_train_the_limit_that_checks_it(self):
        state = self.make()
        origins = [candidate(20, 20), candidate(80, 20)]
        for index in range(10):
            state.update(origins, index * .1)
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 1.0)
        self.assertEqual(state.update(frame, 1.8), [])
        move, = state.update(frame, 2.0)
        self.assertEqual(move["first_visible_time"], 1.8)

    def test_slow_cadence_does_not_bridge_a_real_outage(self):
        state = self.make()
        self.slow_layout(state)
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 6.0)
        self.assertEqual(state.update(frame, 7.3), [])
        move, = state.update(frame, 7.9)
        self.assertEqual(move["first_visible_time"], 7.3)
        self.assertEqual(move["samples"], 2)

    def test_repeated_stills_never_advance_slow_cadence_confirmation(self):
        state = self.make()
        self.slow_layout(state)
        frame = [candidate(80, 20), candidate(45, 60)]
        state.update(frame, 6.0)
        for _ in range(10):
            self.assertEqual(state.update(frame, 6.0), [])
            self.assertEqual(state.status["pending"]["samples"], 1)
        move, = state.update(frame, 6.6)
        self.assertEqual(move["samples"], 2)

    def test_sparse_stills_do_not_expand_the_hard_gap_bound(self):
        state = self.make()
        frame = [candidate(80, 20), candidate(45, 60)]
        for now in (0, 2, 4, 6, 8, 10):
            self.assertEqual(state.update(frame, now), [])
            self.assertLessEqual(state.maximum_observation_gap, 1)
            self.assertIsNone(state.status["timing"]["cadence_seconds"])

    def test_zero_settling_duration_still_needs_two_new_observations(self):
        state = SingleMover({"one": (20, 20), "two": (80, 20)}, 25, settle_seconds=0)
        frame = [candidate(80, 20), candidate(45, 60)]
        self.assertEqual(state.update(frame, 0), [])
        self.assertEqual(state.update(frame, 0), [])
        move, = state.update(frame, .1)
        self.assertEqual(move["samples"], 2)

    def test_slow_cadence_masking_and_obstructions_still_reset_destinations(self):
        for hidden, obstructed in (({"one"}, False), (set(), True)):
            with self.subTest(hidden=hidden, obstructed=obstructed):
                state = self.make()
                self.slow_layout(state)
                frame = [candidate(80, 20), candidate(45, 60)]
                state.update(frame, 6.0)
                self.assertEqual(state.update(frame, 6.6, unavailable=hidden, obstructed=obstructed), [])
                self.assertEqual(state.update(frame, 7.2), [])
                move, = state.update(frame, 7.8)
                self.assertEqual(move["first_visible_time"], 7.2)

    def test_invalid_input_is_rejected(self):
        for initial, grid, settle in (({}, 25, .2), ({"a": (0, 0)}, 0, .2),
                                      ({"a": (0, 0)}, 25, -.2),
                                      ({"a": (0, 0), "b": (0, 0)}, 25, .2)):
            with self.assertRaises(ValueError):
                SingleMover(initial, grid, settle)
        with self.assertRaisesRegex(ValueError, "timing policy"):
            SingleMover({"a": (0, 0)}, 25, timing_policy="unbounded")
        state = self.make()
        for item in ({"cx": math.nan, "cy": 0}, {"cx": 0},
                     candidate(0, 0, uncertainty_px=-1), candidate(0, 0, confidence=2)):
            with self.assertRaises(ValueError):
                state.update([item], 0)
        with self.assertRaises(ValueError):
            state.update([], 0, unavailable={"unknown"})

    def test_transient_stationary_dropout_preserves_fresh_destination_and_mover_continuity(self):
        state = self.make()
        state.update([candidate(20, 20), candidate(80, 20)], 0)
        state.update([candidate(80, 20)], .1)
        self.assertEqual(state.status["active_mini"], "one")
        state.update([candidate(45, 60)], .2)
        self.assertEqual(state.status["unresolved_departures"], ["one", "two"])
        self.assertIsNone(state.status["pending"]["mini"])
        self.assertEqual(state.status["active_mini"], "one")
        self.assertEqual(state.update([candidate(45, 60)], .3), [])
        move, = state.update([candidate(80, 20), candidate(45, 60)], .4)
        self.assertEqual(move["mini"], "one")
        self.assertEqual(move["first_visible_time"], .2)
        self.assertEqual(move["samples"], 3)
        self.assertEqual(state.positions["two"], (80, 20))
        self.assertEqual(state.status["unresolved_departures"], [])

    def test_anonymous_destination_gets_identity_only_after_positive_origin_return(self):
        state = self.make()
        state.update([candidate(20, 20), candidate(80, 20)], 0)
        state.update([candidate(45, 60)], .1)
        state.update([candidate(45, 60)], .3)
        self.assertIsNone(state.status["active_mini"])
        self.assertIsNone(state.status["pending"]["mini"])
        move, = state.update([candidate(80, 20), candidate(45, 60)], .4)
        self.assertEqual(move["mini"], "one")
        self.assertEqual(move["first_visible_time"], .1)

    def test_caption_cannot_remove_an_already_missing_identity_from_hypotheses(self):
        state = self.make()
        state.update([candidate(20, 20), candidate(80, 20)], 0)
        state.update([candidate(45, 60)], .1)
        for index in range(1, 26):
            self.assertEqual(state.update([candidate(45, 60)], .1 + index * .2,
                                          unavailable={"two"}), [])
            self.assertEqual(state.status["unresolved_departures"], ["one", "two"])
            self.assertEqual(state.last_seen["two"], 0)
            self.assertIsNone(state.status["pending"]["mini"])
        move, = state.update([candidate(80, 20), candidate(45, 60)], 5.3)
        self.assertEqual(move["mini"], "one")
        self.assertEqual(state.last_seen["two"], 5.3)

    def test_all_hidden_then_two_changed_origins_cannot_become_unique_by_masking_one(self):
        for obstructed in (False, True):
            with self.subTest(obstructed=obstructed):
                state = self.make()
                state.update([candidate(20, 20), candidate(80, 20)], 0)
                state.update([], .1, obstructed=obstructed, unavailable={"one", "two"})
                self.assertEqual(state.status["unresolved_departures"], ["one", "two"])
                self.assertEqual(state.update([candidate(45, 60), candidate(105, 60)], .2), [])
                for now in (.4, .6, .8):
                    self.assertEqual(state.update([candidate(45, 60)], now, unavailable={"two"}), [])
                self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})
                self.assertEqual(state.status["unresolved_departures"], ["one", "two"])

    def test_true_second_departure_never_inherits_the_first_mover_identity(self):
        state = self.make()
        state.update([candidate(20, 20), candidate(80, 20)], 0)
        state.update([candidate(80, 20), candidate(45, 60)], .1)
        self.assertEqual(state.status["active_mini"], "one")
        state.update([candidate(45, 60), candidate(105, 60)], .2)
        for now in (.4, .6, .8):
            self.assertEqual(state.update([candidate(45, 60)], now, unavailable={"two"}), [])
        self.assertEqual(state.positions, {"one": (20, 20), "two": (80, 20)})

    def test_unobserved_destination_time_cannot_be_reused_when_identity_resolves(self):
        state = self.make()
        state.update([candidate(45, 60)], 0)
        state.update([], .15)
        self.assertIsNone(state.status["pending"])
        self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], .3), [])
        self.assertEqual(state.update([candidate(80, 20), candidate(45, 60)], .4), [])
        move, = state.update([candidate(80, 20), candidate(45, 60)], .5)
        self.assertEqual(move["first_visible_time"], .3)

    def test_anonymous_destination_cache_is_bounded_and_never_expires_identity_ambiguity(self):
        state = self.make()
        for index in range(100):
            self.assertEqual(state.update([candidate(45, 60)], index / 50), [])
            self.assertLessEqual(len(state._pending["samples"]), state.destination_cache_max_samples)
            self.assertTrue(all(index / 50 - timestamp <= state.destination_cache_seconds
                                for timestamp, _sample in state._pending["samples"]))
        self.assertEqual(state.status["unresolved_departures"], ["one", "two"])
        move, = state.update([candidate(80, 20), candidate(45, 60)], 2)
        self.assertEqual(move["mini"], "one")
        self.assertEqual(move["first_visible_time"], 0)
        self.assertLessEqual(move["samples"], state.destination_cache_max_samples)

    def test_ambiguous_old_anchor_is_not_fresh_but_other_anchor_can_be_observed(self):
        state = self.make()
        state.update([candidate(20, 20), candidate(80, 20)], 0)
        state.update([candidate(20, 20), candidate(110, 60)], .1)
        self.assertEqual(state.status["unresolved_departures"], ["two"])
        state.update([candidate(20, 20), candidate(75, 20), candidate(85, 20)], .2)
        self.assertEqual(state.status["reason"], "ambiguous_anchor_match")
        self.assertEqual(state.status["unresolved_departures"], ["two"])
        self.assertEqual(state.last_seen["two"], 0)
        self.assertEqual(state.last_seen["one"], .2)

    def test_all_origins_returning_cancels_anonymous_destination_without_assigning_it(self):
        state = self.make()
        state.update([candidate(45, 60)], 0)
        state.update([candidate(45, 60)], .3)
        self.assertEqual(state.update([candidate(20, 20), candidate(80, 20), candidate(45, 60)], .4), [])
        self.assertEqual(state.status["unresolved_departures"], [])
        self.assertIsNone(state.status["pending"])
        self.assertEqual(state.status["reason"], "extra_candidates_with_full_layout")


if __name__ == "__main__":
    unittest.main()
