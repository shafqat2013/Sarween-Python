"""Continuous physical evidence is independent of known token delivery.

Synthetic static scenes exercise association and false-move guards. They do not
claim that arbitrary glare, animation, or real-world identity is solved.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector
from observer_engine import BoardGeometry, ObserverEngine, CONTINUOUS_DISPLAY_POLICY
from observer_journal import ObserverJournal, replay_journal
from tests.test_observer_expected_recovery import POSITIONS, VIEW, command, notice


def scene(positions=None, *, white=False, artifact=False, glare=None, gradient=0):
    positions = positions or POSITIONS
    pixels = np.full((360, 400, 3), (255, 255, 255) if white else (150, 90, 40), np.uint8)
    if gradient:
        yy, xx = np.mgrid[:360, :400]
        amount = np.maximum(0, 45 - np.abs(xx - gradient) / 2).astype(np.uint8)
        pixels = np.minimum(255, pixels.astype(np.int16) + amount[..., None]).astype(np.uint8)
    if artifact:
        cv2.circle(pixels, (300, 180), 25, (35, 75, 180), -1)
    for center in positions.values():
        center = tuple(round(value) for value in center)
        cv2.circle(pixels, center, 30, (35, 75, 180), -1)
        cv2.circle(pixels, center, 24, (160, 160, 160), -1)
        cv2.line(pixels, center, (center[0] - 43, center[1]), (160, 160, 160), 8)
    if glare is not None:
        cv2.circle(pixels, glare, 38, (255, 255, 255), -1)
    return pixels


class ContinuousStaticMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def engine(self):
        return ObserverEngine(BoardGeometry(VIEW, (400, 360)), POSITIONS,
                              detector_factory=HybridRingDetector,
                              display_transition_policy=CONTINUOUS_DISPLAY_POLICY)

    def acquire(self, engine, *, white=False):
        for now in (0, .125, .25):
            row = engine.process(scene(white=white), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"], row)
        self.assertFalse(row["held"], row["state"])

    def move(self, engine, positions, first, *, revision=0, change=None, white=False,
             expected=(), confirmed=(), artifact=False):
        for index in range(5):
            row = engine.process(scene(positions, white=white, artifact=artifact), None,
                first + index * .125, scene_visual_revision=revision, scene_visual_change=change,
                expected_display_commands=expected if index == 0 else (), confirmed_command_ids=confirmed)
        return row

    def test_missing_ack_and_no_lcd_change_do_not_interrupt_next_physical_move(self):
        engine = self.engine()
        self.acquire(engine)
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        self.move(engine, positions, .375)
        self.assertEqual(len(engine.events), 1)
        engine.expect_display_change(**command())
        self.assertIsNone(engine._display_recovery)
        self.assertNotIn("baseline", engine._known_display_commands["move1"])
        positions["mini2"] = (200, 160)
        row = self.move(engine, positions, 1.0)
        self.assertEqual([event["mini"] for event in engine.events], ["mini1", "mini2"])
        self.assertFalse(row["held"])
        self.assertEqual(row["display_transition"]["pending_command_ids"], ["move1"])
        self.assertFalse(row["display_transition"]["physical_tracking_waits_for_render"])
        self.assertFalse(row["display_transition"]["render_alignment_verified"])

    def test_restored_command_provenance_before_acquisition_neither_acquires_nor_moves(self):
        engine = self.engine()
        self.assertTrue(engine.expect_display_change(**command()))
        self.assertFalse(engine._acquired)
        self.assertIsNone(engine._previous)
        self.assertEqual(engine.events, [])
        self.assertIsNone(engine._display_recovery)
        row = engine.process(scene(), None, 0, scene_visual_revision=0,
            expected_display_commands=[command()], confirmed_command_ids=["move1"])
        self.assertFalse(row["acquired"])
        self.assertEqual(row["events"], [])
        self.assertEqual(row["display_transition"]["confirmed_command_ids"], ["move1"])
        for now in (.125, .25):
            row = engine.process(scene(), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"])
        self.assertEqual(row["events"], [])
        row = engine.process(scene(), None, .375, scene_visual_revision=1,
                             scene_visual_change=notice())
        self.assertFalse(row["needs_reset"])
        self.assertFalse(row["held"])
        self.assertEqual(row["events"], [])
        self.assertEqual(row["display_transition"]["notified_command_ids"], ["move1"])

    def test_legacy_policies_still_reject_command_registration_before_acquisition(self):
        for policy in ("disabled", "hold-known-display-changes-v1",
                       "recover-expected-display-changes-v1", "recover-expected-display-changes-v2",
                       "recover-expected-display-changes-v3"):
            with self.subTest(policy=policy):
                engine = ObserverEngine(BoardGeometry(VIEW, (400, 360)), POSITIONS,
                                        display_transition_policy=policy)
                with self.assertRaises(ValueError):
                    engine.expect_display_change(**command())
                self.assertEqual(engine._known_display_commands, {})

    def test_command_registration_and_late_notice_preserve_an_in_progress_destination(self):
        engine = self.engine()
        self.acquire(engine)
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        first = engine.process(scene(positions), None, .375, scene_visual_revision=0)
        self.assertEqual(first["state"]["phase"], "settling")
        pending = deepcopy(engine.mover._pending)
        engine.expect_display_change(**command())
        self.assertEqual(engine.mover._pending, pending)
        row = engine.process(scene(positions), None, .625, scene_visual_revision=1,
                             scene_visual_change=notice())
        self.assertEqual(len(row["events"]), 1)
        self.assertEqual(row["events"][0]["mini"], "mini1")
        self.assertEqual(row["display_transition"]["notified_command_ids"], ["move1"])
        self.assertEqual(row["display_transition"]["confirmed_command_ids"], [])

    def test_acknowledged_invisible_redraw_is_metadata_not_a_physical_hold(self):
        engine = self.engine()
        self.acquire(engine)
        row = engine.process(scene(), None, .375, scene_visual_revision=1,
            expected_display_commands=[command()], confirmed_command_ids=["move1"], scene_visual_change=notice())
        self.assertFalse(row["held"])
        self.assertFalse(row["needs_reset"])
        self.assertEqual(row["display_transition"]["pending_command_ids"], [])
        self.assertEqual(row["display_transition"]["confirmed_command_ids"], ["move1"])
        self.assertEqual(row["display_transition"]["notified_command_ids"], ["move1"])

    def test_full_layout_learns_stationary_display_artifact_without_accepting_a_move(self):
        engine = self.engine()
        self.acquire(engine)
        for now in (.375, .5, .625):
            row = engine.process(scene(artifact=True), None, now, scene_visual_revision=1,
                expected_display_commands=[command()] if now == .375 else (),
                scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertEqual(row["events"], [])
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        self.move(engine, positions, .75, revision=1, artifact=True)
        self.assertEqual([event["mini"] for event in engine.events], ["mini1"])

    def test_genuinely_competing_changed_candidates_remain_uncertain(self):
        engine = self.engine()
        self.acquire(engine)
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        row = self.move(engine, positions, .375, artifact=True)
        self.assertEqual(engine.events, [])
        self.assertEqual(row["state"]["reason"], "competing_destinations")
        self.assertEqual(engine.mover.positions, POSITIONS)

    def test_colored_ring_departures_on_white_lcd_are_not_masked_as_glare(self):
        engine = self.engine()
        self.acquire(engine, white=True)
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        self.move(engine, positions, .375, white=True)
        positions["mini2"] = (200, 160)
        self.move(engine, positions, 1., white=True, expected=[command()],
                  revision=1, change=notice(), confirmed=["move1"])
        self.assertEqual([event["mini"] for event in engine.events], ["mini1", "mini2"])

    def test_moving_white_glare_and_gradient_without_physical_moves_emit_nothing(self):
        engine = self.engine()
        self.acquire(engine)
        for index, x in enumerate((80, 90, 160, 240, 300, 200, 80)):
            row = engine.process(scene(glare=(x, 80 if x < 100 else 180), gradient=x), None,
                                 .375 + index * .125, scene_visual_revision=0)
            self.assertEqual(row["events"], [])
        for now in (1.375, 1.5, 1.625):
            row = engine.process(scene(), None, now, scene_visual_revision=0)
        self.assertEqual(engine.events, [])
        self.assertEqual(engine.mover.positions, POSITIONS)
        self.assertFalse(row["held"])

    def test_unknown_revision_scene_mapping_lock_and_mask_guards_remain_strict(self):
        for kwargs in ({"scene_visual_revision": 1, "scene_visual_change": notice(commands=("foreign",))},
                {"scene_visual_revision": 1, "scene_visual_change": notice(reason="pan")},
                {"scene_visual_revision": 1, "scene_visual_change": notice(sceneId="other")},
                {"scene_visual_revision": None}, {"locked": False}, {"geometry_key": {"other": 1}}):
            with self.subTest(kwargs=kwargs):
                engine = self.engine()
                self.acquire(engine)
                engine.expect_display_change(**command())
                context = {"scene_visual_revision": 0, **kwargs}
                row = engine.process(scene(), None, .375, **context)
                self.assertTrue(row["needs_reset"])
                self.assertEqual(row["events"], [])

    def test_masked_origin_does_not_become_a_departure(self):
        engine = self.engine()
        self.acquire(engine)
        mask = np.zeros((360, 400), np.uint8)
        mask[40:120, 40:120] = 255
        positions = deepcopy(POSITIONS)
        positions["mini1"] = (80, 160)
        for now in (.375, .5, .625, .75):
            row = engine.process(scene(positions), mask, now, scene_visual_revision=0)
            self.assertEqual(row["events"], [])
        self.assertIn("mini1", row["unavailable"])
        self.assertEqual(engine.mover.positions, POSITIONS)

    def test_continuous_policy_and_command_inputs_replay_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine()
            journal = ObserverJournal(Path(directory) / "recording", engine.initialization())
            positions = deepcopy(POSITIONS)
            for index in range(13):
                now = index * .125
                metadata = {"scene_visual_revision": int(index >= 8)}
                if index == 4:
                    metadata["expected_display_commands"] = [command()]
                if index >= 5:
                    positions["mini1"] = (80, 160)
                if index >= 8:
                    metadata["scene_visual_change"] = notice()
                    positions["mini2"] = (200, 160)
                pixels = scene(positions)
                row = engine.process(pixels, None, now, **metadata)
                journal.append(pixels, None, now, row, metadata=metadata)
            journal.close()
            replay = replay_journal(journal.folder)
            self.assertTrue(replay["verification_passed"])
            self.assertTrue(replay["identical_decisions"])
            self.assertEqual([event["mini"] for event in replay["events"]], ["mini1", "mini2"])


if __name__ == "__main__":
    unittest.main()
