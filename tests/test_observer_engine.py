"""Shared image-level observer behavior and replay determinism, without devices."""

from copy import deepcopy
import json
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

import cv2
import numpy as np

from observer_engine import BoardGeometry, ObserverEngine, draw_observer_frame


def geometry():
    return BoardGeometry({
        "registration": {"left": 0, "top": 0, "right": 400, "bottom": 240},
        "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
        "gridSize": 40, "gridOriginX": 0, "gridOriginY": 0}, (400, 240))


def frame(center=(100, 100)):
    image = np.full((240, 400, 3), (150, 90, 40), np.uint8)
    cv2.circle(image, center, 25, (35, 75, 180), -1)
    cv2.line(image, center, (center[0] - 43, center[1]), (35, 75, 180), 8)
    return image


class SharedObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def engine(self, **kwargs):
        return ObserverEngine(geometry(), {"piece": (100, 100)}, {"piece": "C3"}, **kwargs)

    def test_same_pixels_masks_and_clock_produce_identical_live_and_replay_decisions(self):
        live, replay = self.engine(), self.engine()
        live_rows, replay_rows = [], []
        for index in range(12):
            image = frame((100, 100) if index < 4 else (260, 100))
            mask = np.zeros(image.shape[:2], np.uint8)
            # A file adapter must preserve exact algorithm inputs. np.savez's
            # roundtrip is used here instead of lossy video recompression.
            import io
            serialized = io.BytesIO()
            np.savez_compressed(serialized, board=image, mask=mask)
            serialized.seek(0)
            with np.load(serialized, allow_pickle=False) as saved:
                replay_image, replay_mask = saved["board"].copy(), saved["mask"].copy()
            live_rows.append(live.process(image, mask, index / 8))
            replay_rows.append(replay.process(replay_image, replay_mask, index / 8))
        self.assertEqual(json.dumps(live_rows, sort_keys=True), json.dumps(replay_rows, sort_keys=True))
        self.assertEqual(len(live.events), 1)
        self.assertEqual((live.events[0]["from_cell"], live.events[0]["to_cell"]), ("C3", "G3"))
        self.assertTrue(all(not row["foundry_token_output"] for row in live_rows))

    def test_lock_loss_latches_hold_until_new_explicit_bindings(self):
        engine = self.engine()
        engine.process(frame(), None, 0)
        engine.process(frame((260, 100)), None, .125)
        lost = engine.process(frame((260, 100)), None, .25, locked=False)
        recovered = engine.process(frame((260, 100)), None, .5, locked=True)
        self.assertEqual(lost["state"]["reason"], "camera_lock_lost")
        self.assertEqual(recovered["state"]["reason"], "camera_lock_lost")
        self.assertEqual(recovered["events"], [])
        self.assertEqual(recovered["held"], ["piece"])
        self.assertEqual(recovered["positions"]["piece"], (100, 100))
        engine.reset({"piece": (260, 100)}, {"piece": "G3"})
        fresh = engine.process(frame((260, 100)), None, .75)
        self.assertFalse(fresh["needs_reset"])
        self.assertEqual(fresh["events"], [])
        self.assertEqual(fresh["positions"]["piece"], (260, 100))

    def test_changed_geometry_never_commits_a_pending_move(self):
        key = {"scene": "A", "view": 1}
        engine = self.engine(geometry_key=key)
        key["view"] = 99  # The engine owns its original token.
        engine.process(frame(), None, 0, geometry_key={"scene": "A", "view": 1})
        held = engine.process(frame((260, 100)), None, .5,
                              geometry_key={"scene": "A", "view": 2})
        self.assertEqual(held["state"]["reason"], "geometry_changed")
        self.assertEqual(held["events"], [])
        self.assertTrue(held["needs_reset"])

    def test_changed_input_dimensions_hold_instead_of_comparing_unrelated_pixels(self):
        engine = self.engine()
        engine.process(frame(), None, 0)
        held = engine.process(frame()[:, :300].copy(), None, .1)
        self.assertEqual(held["state"]["reason"], "board_size_changed")
        self.assertEqual(held["held"], ["piece"])

    def test_caption_coverage_preserves_anchor_and_does_not_report_departure(self):
        engine = self.engine()
        engine.process(frame(), None, 0)
        image = frame()
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[50:150, 50:150] = 255
        image[mask > 0] = 255
        for index in range(1, 5):
            row = engine.process(image, mask, index / 8)
            self.assertEqual(row["events"], [])
            self.assertEqual(row["positions"]["piece"], (100, 100))
            self.assertIn("piece", row["held"])

    def test_large_background_change_marks_all_positions_held_for_that_frame(self):
        engine = self.engine()
        engine.process(frame(), None, 0)
        changed = np.full((240, 400, 3), 255, np.uint8)
        row = engine.process(changed, None, .125)
        self.assertTrue(row["obstructed"])
        self.assertEqual(row["held"], ["piece"])
        self.assertEqual(row["events"], [])
        self.assertEqual(row["positions"]["piece"], (100, 100))
        # This verifies a broad-change hold only, not arbitrary fog robustness.

    def test_returned_records_and_initialization_do_not_alias_engine_state(self):
        engine = self.engine()
        row = engine.process(frame(), None, 0)
        original = deepcopy(engine.last_record)
        row["positions"]["piece"] = (300, 200)
        row["state"]["reason"] = "external mutation"
        row["candidates"].clear()
        self.assertEqual(engine.last_record, original)
        self.assertEqual(engine.mover.positions["piece"], (100, 100))
        manifest = engine.initialization()
        manifest["view"]["gridSize"] = 999
        manifest["initial_positions"]["piece"] = (1, 1)
        self.assertEqual(engine.geometry.grid, 40)
        self.assertEqual(engine.initial_positions["piece"], (100, 100))

    def test_repeated_backward_and_nonfinite_timestamps_are_rejected(self):
        engine = self.engine()
        engine.process(frame(), None, 1)
        for timestamp in (1, .99, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                engine.process(frame(), None, timestamp)
        self.assertEqual(engine.process(frame(), None, 1.1)["frame"], 1)

    def test_caption_geometry_error_is_rejected_without_advancing_clock(self):
        engine = self.engine()
        with self.assertRaisesRegex(ValueError, "mask must match"):
            engine.process(frame(), np.zeros((10, 10), np.uint8), 1)
        self.assertEqual(engine.process(frame(), None, 1)["frame"], 0)

    def test_preview_does_not_modify_measurement_image(self):
        engine = self.engine()
        image = frame()
        expected = image.copy()
        row = engine.process(image, None, 0)
        preview = draw_observer_frame(image, row, engine.geometry)
        self.assertEqual(preview.shape, (304, 400, 3))
        np.testing.assert_array_equal(image, expected)


class CaptionProcessingTests(unittest.TestCase):
    def test_recognized_caption_transitions_do_not_freeze_uncovered_bases(self):
        # Isolate the engine contract from caption recognition itself. Its
        # image matcher has separate positive/negative image tests.
        engine = ObserverEngine(geometry(), {"piece": (100, 100)},
                                watermark_mode="board-template-v1")
        caption_detector = Mock()
        engine._caption_detector = caption_detector
        zero = np.zeros((240, 400), np.uint8)
        caption_mask = zero.copy()
        caption_mask[:, 180:] = 255

        def observation(present):
            return SimpleNamespace(exclusion_mask=caption_mask if present else zero,
                                   bbox=(180, 0, 220, 240) if present else None,
                                   confidence=.95 if present else 0., detected=present,
                                   provenance={"method": "known-caption-fixture", "detected": present})

        caption_detector.detect.return_value = observation(False)
        engine.process(frame(), None, 0)
        self.assertTrue(engine.process(frame(), None, .25)["acquired"])
        with_caption = frame()
        with_caption[:, 180:] = 255
        original = with_caption.copy()
        caption_detector.detect.return_value = observation(True)
        appeared = engine.process(with_caption, None, .5)
        self.assertFalse(appeared["obstructed"])
        self.assertEqual(appeared["state"]["reason"], "layout_confirmed")
        self.assertEqual(appeared["events"], [])
        self.assertTrue(appeared["watermark"]["recognized"])
        self.assertEqual(appeared["watermark"]["provided_pixels"], 0)
        self.assertGreater(appeared["watermark"]["supplemental_pixels"], 0)
        np.testing.assert_array_equal(with_caption, original)

        caption_detector.detect.return_value = observation(False)
        disappeared = engine.process(frame(), None, .75)
        self.assertFalse(disappeared["obstructed"])
        self.assertEqual(disappeared["events"], [])

        # The recognizer must not turn unrelated large image changes into
        # evidence of a clear board. Only the recognized caption is excluded.
        unrelated = frame()
        unrelated[:, :170] = 255
        held = engine.process(unrelated, None, 1.)
        self.assertTrue(held["obstructed"])
        self.assertEqual(held["held"], ["piece"])
        self.assertEqual(held["events"], [])

    def test_unsupported_caption_mode_fails_without_opening_a_detector(self):
        with self.assertRaisesRegex(ValueError, "Unsupported observer watermark mode"):
            ObserverEngine(geometry(), {"piece": (100, 100)}, watermark_mode="unknown")


class AcquisitionTests(unittest.TestCase):
    """Starting clicks cannot substitute for observed physical base evidence."""

    def setUp(self):
        self.detector = Mock()
        self.detector.detect.return_value = []
        self.factory = Mock(return_value=self.detector)
        self.engine = ObserverEngine(geometry(), {"first": (100, 100), "second": (260, 100)},
                                     detector_factory=self.factory)

    @staticmethod
    def candidate(x, y=100, confidence=.9):
        return {"cx": x, "cy": y, "radius": 25, "confidence": confidence,
                "uncertainty_px": 2.5}

    @staticmethod
    def mask_at(x):
        mask = np.zeros((240, 400), np.uint8)
        mask[60:140, x - 40:x + 40] = 255
        return mask

    def process(self, now, candidates, mask=None, image=None):
        self.detector.detect.return_value = candidates
        return self.engine.process(frame() if image is None else image, mask, now)

    def test_manual_bindings_with_missing_bases_never_become_ready_or_infer_moves(self):
        for now in (0, .25, .5):
            row = self.process(now, [self.candidate(100)])
            self.assertFalse(row["acquired"])
            self.assertEqual(row["acquisition_visible_count"], 1)
            self.assertEqual(row["acquisition_required"], 2)
            self.assertEqual(row["acquisition_missing"], ["second"])
            self.assertEqual(row["state"]["phase"], "acquiring")
            self.assertEqual(row["state"]["reason"], "waiting_for_base_acquisition")
            self.assertEqual(row["held"], ["first", "second"])
            self.assertEqual(row["events"], [])
            self.assertEqual(row["reference_updates"], 0)
        self.assertEqual(self.engine.mover.last_seen, {"first": None, "second": None})

    def test_every_base_needs_repeated_consistent_measurements_spanning_settle_time(self):
        candidates = [self.candidate(100), self.candidate(260)]
        first = self.process(0, candidates)
        self.assertEqual(first["acquisition"]["visible_count"], 2)
        self.assertEqual(first["acquisition"]["ready_count"], 0)
        self.assertFalse(first["acquired"])
        self.assertFalse(self.process(.1, candidates)["acquired"])
        acquired = self.process(.25, candidates)
        self.assertTrue(acquired["acquired"])
        self.assertEqual(acquired["acquisition"]["ready_count"], 2)
        self.assertEqual(acquired["state"]["reason"], "layout_confirmed")
        self.assertEqual(acquired["events"], [])
        self.assertGreater(acquired["reference_updates"], 0)

    def test_caption_cycles_can_acquire_bases_from_recent_separate_visible_windows(self):
        self.process(0, [self.candidate(100)], self.mask_at(260))
        row = self.process(.25, [self.candidate(100)], self.mask_at(260))
        self.assertEqual(row["acquisition"]["ready_count"], 1)
        self.process(.5, [self.candidate(260)], self.mask_at(100))
        row = self.process(.75, [self.candidate(260)], self.mask_at(100))
        self.assertTrue(row["acquired"])
        self.assertEqual(row["acquisition"]["visible_count"], 1)
        self.assertEqual(row["held"], ["first"])
        self.assertEqual(row["events"], [])

    def test_old_masked_evidence_expires_instead_of_eventually_claiming_readiness(self):
        self.process(0, [self.candidate(100)], self.mask_at(260))
        self.process(.25, [self.candidate(100)], self.mask_at(260))
        self.process(5.4, [self.candidate(260)], self.mask_at(100))
        row = self.process(5.65, [self.candidate(260)], self.mask_at(100))
        self.assertFalse(row["acquired"])
        self.assertEqual(row["acquisition"]["ready_count"], 1)
        self.assertEqual(row["acquisition"]["pending"], ["first"])

    def test_visible_miss_clears_previous_proof_and_preacquisition_motion_is_not_a_move(self):
        self.process(0, [self.candidate(100)], self.mask_at(260))
        self.process(.25, [self.candidate(100)], self.mask_at(260))
        for now in (.5, .75, 1):
            row = self.process(now, [self.candidate(180), self.candidate(260)])
            self.assertFalse(row["acquired"])
            self.assertIn("first", row["acquisition"]["pending"])
            if now >= .75:
                self.assertEqual(row["acquisition"]["pending"], ["first"])
            self.assertEqual(row["events"], [])
            self.assertEqual(row["reference_updates"], 0)

    def test_broad_obstruction_restarts_all_acquisition_evidence(self):
        self.process(0, [self.candidate(100)], self.mask_at(260))
        self.process(.25, [self.candidate(100)], self.mask_at(260))
        changed = np.full((240, 400, 3), 255, np.uint8)
        row = self.process(.5, [self.candidate(100), self.candidate(260)], image=changed)
        self.assertTrue(row["obstructed"])
        self.assertFalse(row["acquired"])
        self.assertEqual(row["acquisition"]["ready_count"], 0)
        self.assertEqual(row["acquisition"]["visible_count"], 0)

    def test_duplicates_and_low_confidence_do_not_supply_missing_base_evidence(self):
        candidates = [self.candidate(100), self.candidate(101), self.candidate(260, confidence=.44)]
        first = self.process(0, candidates)
        self.assertEqual(first["acquisition"]["visible_count"], 1)
        self.assertEqual(first["acquisition"]["ready_count"], 0)
        self.assertFalse(self.process(.25, candidates)["acquired"])

    def test_one_measurement_near_two_clicks_cannot_acquire_two_identities(self):
        self.engine = ObserverEngine(geometry(), {"first": (100, 100), "second": (120, 100)},
                                     detector_factory=self.factory)
        for now in (0, .25):
            row = self.process(now, [self.candidate(110)])
            self.assertFalse(row["acquired"])
            self.assertEqual(row["acquisition"]["visible_count"], 0)

    def test_initialization_receives_first_pixels_and_mask_once_and_factory_survives_reset(self):
        mask = self.mask_at(260)
        image = frame()
        self.process(0, [], mask, image)
        self.process(.25, [], mask, image)
        self.detector.initialize.assert_called_once()
        args = self.detector.initialize.call_args
        np.testing.assert_array_equal(args.args[0], image)
        np.testing.assert_array_equal(args.kwargs["excluded_mask"], mask)
        self.assertEqual(args.args[1], [(100, 100), (260, 100)])
        self.engine.reset({"first": (100, 100), "second": (260, 100)})
        self.assertEqual(self.factory.call_count, 2)
        self.assertFalse(self.engine._acquired)

    def test_normal_single_mover_inference_starts_only_after_acquisition(self):
        baseline = [self.candidate(100), self.candidate(260)]
        self.process(0, baseline)
        self.process(.25, baseline)
        for now in (.4, .55, .7):
            self.process(now, [self.candidate(180), self.candidate(260)], image=frame((180, 100)))
        self.assertEqual(len(self.engine.events), 1)
        move = self.engine.events[0]
        self.assertEqual((move["mini"], move["from_cell"], move["to_cell"]), ("first", "C3", "E3"))


if __name__ == "__main__":
    unittest.main()
