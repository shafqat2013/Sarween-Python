"""Independent scoring cannot hide missing moves or turn ambiguity into truth."""

import copy
import json
from pathlib import Path
import unittest

from single_mover_evaluation import evaluate_observer


SHA = "a" * 64


def fixtures():
    moves = [
        ("red", "AP13", "AK6", None),
        ("salmon", "AG22", "AM20", None),
        ("green", "W9", "W17", None),
        ("yellow", "Q8", None, ["AA8", "AB8"]),
        ("orange", "K8", "Q13", None),
        ("green", "W17", None, ["AB16", "AC16"]),
    ]
    expected, events = [], []
    for sequence, (mini, origin, destination, boundary) in enumerate(moves, 1):
        reference_time = sequence * 10.0
        expected.append({
            "sequence": sequence, "mini": mini, "from_cell": origin,
            "to_cell": destination, "destination_boundary_cells": boundary,
            "observed_movement_frame_window": [sequence * 100 - 20, sequence * 100],
            "visible_stationary_capture_seconds": reference_time,
            "visible_stationary_capture_seconds_interval": [reference_time - .13, reference_time],
            "timing_reference_status": "assistant-frame-verified",
        })
        events.append({
            "sequence": sequence, "mini": mini, "from_cell": origin,
            "to_cell": destination, "destination_cells": list(boundary) if boundary else [destination],
            "boundary_uncertain": bool(boundary), "frame": sequence * 100 + 3,
            "capture_seconds": reference_time + .3,
            "confirmed_time": reference_time + .3,
            "first_visible_time": reference_time + .1,
            "simulated_available_capture_seconds": reference_time + .35,
        })
    review = {
        "expected_moves": expected, "source": {"video_sha256": SHA},
        "stationary_minis": {"purple": "S17"},
        "scoring_policy": {"ordered_movement_identity_count": 6},
        "expected_frames": 650,
    }
    report = {"events": events, "metadata": {"source": {"video_sha256": SHA}},
              "complete": True, "frames_consumed": 650,
              "final_positions": {"purple": {"cell": "S17", "boundary_uncertain": False}}}
    return report, review


class SingleMoverEvaluationTest(unittest.TestCase):
    def test_correct_sequence_scores_four_exact_cells_and_two_abstentions(self):
        report, review = fixtures()
        result = evaluate_observer(report, review)
        self.assertTrue(result["overall_development_acceptance"])
        self.assertEqual(result["counts"]["exact_destination_passes"], 4)
        self.assertEqual(result["counts"]["boundary_abstentions"], 2)
        self.assertAlmostEqual(result["timing"]["median_response_seconds"], .3)
        self.assertAlmostEqual(result["timing"]["processing_adjusted_median_response_seconds"], .35)
        self.assertEqual(result["timing"]["eligible_sequences"], [1, 2, 3, 5])

    def test_either_neighboring_cell_alone_is_not_a_boundary_success(self):
        for chosen in ("AA8", "AB8"):
            with self.subTest(chosen=chosen):
                report, review = fixtures()
                event = report["events"][3]
                event.update(to_cell=chosen, destination_cells=[chosen], boundary_uncertain=False)
                result = evaluate_observer(report, review)
                self.assertFalse(result["criteria"]["boundary_abstentions"])
                self.assertFalse(result["overall_development_acceptance"])
                self.assertEqual(result["counts"]["exact_destination_passes"], 4)

    def test_boundary_requires_exact_pair_not_arbitrarily_broad_uncertainty(self):
        report, review = fixtures()
        report["events"][3]["destination_cells"].append("AC8")
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["boundary_abstentions"])

    def test_missing_first_move_cannot_be_hidden_by_best_matching_later_moves(self):
        report, review = fixtures()
        report["events"].pop(0)
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["move_count_exact"])
        self.assertFalse(result["criteria"]["ordered_identities"])
        self.assertEqual(result["per_move"][0]["observed"]["mini"], "salmon")
        self.assertIn("Expected movement was missed", result["per_move"][-1]["discrepancies"])

    def test_extra_move_is_reported_and_stationary_identity_must_never_move(self):
        report, review = fixtures()
        report["events"].append({"sequence": 7, "mini": "purple", "from_cell": "S17", "to_cell": "S17"})
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["no_extra_moves"])
        self.assertFalse(result["criteria"]["stationary_minis_preserved"])
        self.assertIn("Unexpected extra movement", result["per_move"][-1]["discrepancies"])

    def test_correct_cell_for_wrong_mini_still_fails_identity(self):
        report, review = fixtures()
        report["events"][0]["mini"] = "yellow"
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["ordered_identities"])
        self.assertFalse(result["movement_acceptance"])

    def test_wrong_origin_fails_even_if_identity_and_destination_match(self):
        report, review = fixtures()
        report["events"][0]["from_cell"] = "AP14"
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["starting_cells"])

    def test_partial_reference_cannot_reduce_six_required_moves_to_three(self):
        report, review = fixtures()
        report["events"] = report["events"][:3]
        review["expected_moves"] = review["expected_moves"][:3]
        result = evaluate_observer(report, review)
        self.assertTrue(result["criteria"]["move_count_exact"])
        self.assertFalse(result["criteria"]["reference_complete"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_provisional_reference_never_claims_timing_pass(self):
        report, review = fixtures()
        review["expected_moves"][0]["timing_reference_status"] = "provisional"
        result = evaluate_observer(report, review)
        self.assertTrue(result["movement_acceptance"])
        self.assertEqual(result["timing"]["status"], "unverified")
        self.assertIsNone(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])
        self.assertEqual(result["acceptance_status"], "incomplete")

    def test_internal_short_settle_time_cannot_hide_long_physical_response(self):
        report, review = fixtures()
        event = report["events"][0]
        event.update(confirmed_time=12.0, capture_seconds=12.0, first_visible_time=11.9,
                     simulated_available_capture_seconds=12.05)
        result = evaluate_observer(report, review)
        timing = result["per_move"][0]["timing"]
        self.assertAlmostEqual(timing["observer_settle_seconds"], .1)
        self.assertEqual(timing["response_seconds"], 2.0)
        self.assertFalse(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_confirmation_before_visibility_reference_requires_review(self):
        report, review = fixtures()
        report["events"][0]["confirmed_time"] = 9.9
        result = evaluate_observer(report, review)
        self.assertEqual(result["timing"]["status"], "early-confirmation-needs-review")
        self.assertIsNone(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_event_predating_actual_motion_fails_even_with_correct_future_endpoint(self):
        report, review = fixtures()
        report["events"][0]["frame"] = 10
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["events_after_movement_started"])
        self.assertFalse(result["movement_acceptance"])

    def test_wrong_recording_fails_and_missing_source_cannot_claim_full_acceptance(self):
        report, review = fixtures()
        report["metadata"]["source"]["video_sha256"] = "b" * 64
        result = evaluate_observer(report, review)
        self.assertEqual(result["source_validation"]["status"], "mismatch")
        self.assertFalse(result["overall_development_acceptance"])
        report["metadata"] = {}
        result = evaluate_observer(report, review)
        self.assertEqual(result["source_validation"]["status"], "unverified")
        self.assertFalse(result["overall_development_acceptance"])

    def test_missing_stationary_final_state_is_not_silently_assumed_correct(self):
        report, review = fixtures()
        report["final_positions"] = {}
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["stationary_minis_preserved"])

    def test_scoring_does_not_modify_observer_report_or_review(self):
        report, review = fixtures()
        before = copy.deepcopy((report, review))
        evaluate_observer(report, review)
        self.assertEqual((report, review), before)

    def test_nonfinite_confirmation_time_cannot_pass_timing(self):
        report, review = fixtures()
        report["events"][0].update(confirmed_time=float("nan"), capture_seconds=float("nan"))
        result = evaluate_observer(report, review)
        self.assertIsNone(result["per_move"][0]["timing"]["response_seconds"])
        self.assertIsNone(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_empty_reference_is_not_zero_out_of_zero_success(self):
        result = evaluate_observer({"events": []}, {"expected_moves": []})
        self.assertFalse(result["criteria"]["reference_complete"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_skipped_tail_cannot_pass_even_after_all_six_moves_are_emitted(self):
        report, review = fixtures()
        report.update(complete=False, frames_consumed=620)
        result = evaluate_observer(report, review)
        self.assertTrue(result["criteria"]["move_count_exact"])
        self.assertFalse(result["criteria"]["recording_complete"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_complete_flag_alone_cannot_override_known_frame_count(self):
        report, review = fixtures()
        report["frames_consumed"] = 649
        result = evaluate_observer(report, review)
        self.assertFalse(result["criteria"]["recording_complete"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_short_capture_delay_does_not_hide_a_processing_backlog(self):
        report, review = fixtures()
        for event in report["events"]:
            event["simulated_available_capture_seconds"] = event["confirmed_time"] + 10
        result = evaluate_observer(report, review)
        self.assertAlmostEqual(result["timing"]["median_response_seconds"], .3)
        self.assertAlmostEqual(result["timing"]["processing_adjusted_median_response_seconds"], 10.3)
        self.assertFalse(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_missing_inference_availability_leaves_performance_targets_unverified(self):
        report, review = fixtures()
        for event in report["events"]:
            event.pop("simulated_available_capture_seconds")
        result = evaluate_observer(report, review)
        self.assertTrue(result["movement_acceptance"])
        self.assertEqual(result["timing"]["capture_clock_status"], "measured")
        self.assertEqual(result["timing"]["status"], "unverified")
        self.assertIsNone(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_inference_cannot_be_available_before_the_frame_confirmation(self):
        report, review = fixtures()
        report["events"][0]["simulated_available_capture_seconds"] = 10.0
        result = evaluate_observer(report, review)
        self.assertEqual(result["per_move"][0]["timing"]["performance_status"], "invalid-inference-availability")
        self.assertIsNone(result["timing"]["targets_met"])
        self.assertFalse(result["overall_development_acceptance"])

    def test_development_initialization_file_contains_no_future_labels(self):
        path = Path(__file__).resolve().parents[1] / "experiments" / "single_mover_230811.initial.json"
        initial = json.loads(path.read_text())
        self.assertEqual(initial["initial_frame"], 0)
        self.assertEqual(len(initial["initial_positions"]), 6)
        self.assertNotIn("expected_moves", initial)
        self.assertNotIn("review", initial)
        forbidden = {"to_cell", "confirmed_time", "first_visible_time", "visible_stationary_capture_seconds"}

        def visit(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden.intersection(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(initial)


if __name__ == "__main__":
    unittest.main()
