import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from tracking_engine import TrackingContext, TrackingEngine


def bundle(**changes):
    values = dict(locked=True, warp_bgr=np.zeros((200, 300, 3), np.uint8),
                  warp_w=300, warp_h=200, grid_w=6, grid_h=4, marker_mode="legacy",
                  final_mask_cam=None, H_use=None, mask_warp=None)
    return SimpleNamespace(**{**values, **changes})


def detection(x=25, y=25):
    return SimpleNamespace(cx=x, cy=y, lab_dist=3, score=.9)


class TrackingEngineTest(unittest.TestCase):
    def setUp(self):
        self.detections = {"red": detection()}
        self.detector = Mock(side_effect=lambda **kw: (self.detections, kw["prev_state"]))
        self.engine = TrackingEngine(self.detector)
        self.context = TrackingContext(0, ("scene",), 1, 50, lambda x, y: (int(x // 50), int(y // 50)))
        self.frame = bundle()
        self.now = 0

    def step(self, **changes):
        self.now += .1
        return self.engine.step(self.frame, {}, replace(self.context, now=self.now, **changes))

    def settle(self, **changes):
        moves = []
        for _ in range(4):
            moves.extend(self.step(**changes).moves)
        return moves

    def test_consensus_and_independent_mini_state(self):
        self.detections["blue"] = detection(125)
        for _ in range(3):
            self.assertEqual(self.step().moves, [])
        moves = self.step().moves
        self.assertEqual([(m.mini, m.raw_to) for m in moves], [("red", "r0c0"), ("blue", "r0c2")])
        self.detections["red"] = detection(225)
        self.assertEqual(self.step().moves, [])
        self.assertEqual(self.step().moves, [])
        self.assertEqual(self.step().moves, [])
        move, = self.step().moves
        self.assertEqual((move.mini, move.raw_from, move.raw_to), ("red", "r0c0", "r0c4"))
        self.assertEqual(self.engine.prev_state["red"]["last_xy"], (225, 25))
        self.assertEqual(self.step().moves, [])

    def test_selected_mini_loses_anchor_earlier(self):
        self.detections["blue"] = detection(125)
        self.settle()
        self.detections = {}
        self.now = 1.2
        step = self.step(selected_mini="red")
        self.assertEqual(step.lost_minis, ["red"])
        self.assertIsNone(self.engine.prev_state["red"]["last_xy"])
        self.assertIsNotNone(self.engine.prev_state["blue"]["last_xy"])
        self.assertNotIn("red", self.engine.cell_hist)
        self.now = 2.5
        self.assertEqual(self.step().lost_minis, ["blue"])

    def test_scene_change_while_unlocked_clears_old_positions(self):
        self.settle()
        self.frame.locked = False
        self.assertEqual(self.step(scene_key=("new-scene",), view_revision=2).reason, "unlocked")
        self.assertEqual(self.engine.last_physical_xy, {})
        self.frame.locked = True
        moves = self.settle(scene_key=("new-scene",), view_revision=2)
        self.assertEqual(len(moves), 1)
        self.assertIsNone(moves[0].raw_from)
        self.assertEqual(moves[0].source, "detection")

    def test_pause_resume_requires_new_confirmation(self):
        self.settle()
        self.assertEqual(self.step(paused=True).reason, "paused")
        self.assertEqual(self.engine.last_emitted, {})
        moves = self.settle()
        self.assertEqual(len(moves), 1)
        self.assertIsNone(moves[0].raw_from)

    def test_view_change_remaps_and_clears_stale_consensus(self):
        self.settle()
        moved = lambda x, y: (int(x // 50) + 1, int(y // 50))
        result = self.step(view_revision=2, to_cell=moved)
        move, = result.moves
        self.assertEqual((move.raw_from, move.raw_to, move.source), ("r0c0", "r0c1", "viewportTransform"))
        self.assertEqual(list(self.engine.cell_hist["red"]), ["r0c1"])
        self.assertEqual(self.settle(view_revision=2, to_cell=moved), [])

    def test_lock_loss_suppresses_output_and_clock_still_ages_anchor(self):
        self.settle()
        self.frame.locked = False
        self.now = 3
        self.assertEqual(self.step().moves, [])
        self.frame.locked = True
        self.detections = {}
        self.assertEqual(self.step().lost_minis, ["red"])

    def test_unmapped_and_outside_grid_never_emit(self):
        self.assertEqual(self.step(grid_px=None).reason, "unmapped")
        self.detector.assert_not_called()
        self.assertEqual(self.settle(to_cell=lambda x, y: None), [])

    def test_initial_positions_seed_legacy_anchor_without_emitting(self):
        self.engine.seed_positions({"red": "r0c0"}, self.frame)
        self.assertEqual(self.engine.prev_state["red"]["last_xy"], (25, 25))
        self.assertEqual(self.settle(), [])

    def test_viewport_seed_has_no_physical_anchor_until_confirmed(self):
        self.frame.marker_mode = "viewport"
        self.engine.seed_positions({"red": "r0c1"}, self.frame)
        self.assertIsNone(self.engine.prev_state["red"]["last_xy"])
        self.settle()
        self.assertEqual(self.engine.prev_state["red"]["last_xy"], (25, 25))

    def test_rescan_resets_only_changed_mini(self):
        self.detections["blue"] = detection(125)
        self.settle()
        self.engine.reset_mini("red")
        for state in (self.engine.prev_state, self.engine.cell_hist, self.engine.last_seen,
                      self.engine.last_emitted, self.engine.last_physical_xy):
            self.assertNotIn("red", state)
            self.assertIn("blue", state)

    def test_replacement_detector_state_preserves_dictionary_identity(self):
        state_alias = self.engine.prev_state
        self.detector.side_effect = lambda **kw: ({}, {"red": {"last_xy": None}})
        self.step()
        self.assertIs(state_alias, self.engine.prev_state)
        self.assertIn("red", state_alias)

    def test_clock_is_explicit_monotonic_and_finite(self):
        self.step()
        for now in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "clock"):
                self.engine.step(self.frame, {}, replace(self.context, now=now))

    def test_tap_recognition_uses_shared_clock_and_is_cancelled_on_lock_loss(self):
        self.settle()
        with patch("tracking_engine.contact_minis", return_value={"red"}):
            self.assertIsNone(self.step().tapped_mini)
        with patch("tracking_engine.contact_minis", return_value=set()):
            self.assertEqual(self.step().tapped_mini, "red")
        self.now = 2
        with patch("tracking_engine.contact_minis", return_value={"red"}):
            self.step()
        self.frame.locked = False
        self.step()
        self.frame.locked = True
        self.assertIsNone(self.step().tapped_mini)

    def test_constructor_validates_parameters(self):
        for kwargs in ({"consensus_k": 7}, {"consensus_n": 0}, {"lost_timeout": -1},
                       {"selected_lost_timeout": float("inf")}):
            with self.assertRaises(ValueError):
                TrackingEngine(self.detector, **kwargs)


if __name__ == "__main__":
    unittest.main()
