"""Core tracking to the real outbox, using synthetic camera pixels only."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2

from core_token_bridge import CoreTokenBridge
from foundry_delivery import MoveOutbox
from observer_engine import BoardGeometry
from observer_session import LiveObserverController
from tests.test_observer_expected_recovery import POSITIONS, VIEW, image
from tests.test_observer_session import FakeWindow


class Output:
    def __init__(self):
        self.box = MoveOutbox()
        self.context = (VIEW["sceneId"], 400, 360, 40, 0, 0, 1)
        self.box.set_context(self.context)
        self.paused = False
        self.offers = []
        self.fail_mini = None
        self.visual = {"revision": 0, "sceneId": VIEW["sceneId"], "changes": []}

    def set_observation_only(self, paused):
        self.paused = paused
        self.box.clear()

    def observation_only(self):
        return self.paused

    def clear_pending_moves(self):
        self.box.clear()

    def get_delivery_status(self):
        return {"moves": self.box.snapshot(), "message": "Test transport", "retryAvailable": False}

    def get_scene_visual_change(self):
        return deepcopy(self.visual)

    def queue_position_move(self, mini, x, y, **kwargs):
        if self.paused or mini == self.fail_mini:
            return False
        self.offers.append((mini, (x, y), kwargs))
        return self.box.offer_position(mini, (x, y), self.context, source=kwargs["source"],
            generation=kwargs["generation"], expected_token_id=kwargs.get("expected_token_id"),
            expected_scene_id=kwargs.get("expected_scene_id"))


class Assignments:
    def __init__(self):
        self.context = None
        self.confirmed = {}

    def set_tracking_minis(self, names, generation, scene, **kwargs):
        self.context = (tuple(names), generation, scene, kwargs)
        self.confirmed = {}

    def get_confirmed_tracking_assignments(self):
        return dict(self.confirmed)

    def confirm(self):
        self.confirmed = {mini: "token" + mini[4:] for mini in self.context[0]}


class Window(FakeWindow):
    def set_output_status(self, enabled, message):
        self.output_status = (enabled, message)


def result(**overrides):
    return {"positions": deepcopy(POSITIONS), "events": [], "held": [], "acquired": True,
            "needs_reset": False, "acquisition": {"visible_count": 6}, "state": {},
            "frame": 1, "capture_seconds": .6, **overrides}


class CoreTokenBridgeTests(unittest.TestCase):
    def setUp(self):
        self.output, self.assignments = Output(), Assignments()
        self.controller = SimpleNamespace(folder=Path("run"), binding_generation="generation1",
            engine=SimpleNamespace(geometry=BoardGeometry(VIEW, (400, 360)),
                                   mover=SimpleNamespace(positions=deepcopy(POSITIONS))),
            expect_display_change=Mock(return_value=True),
            cancel_expected_display_change=Mock(), reset=Mock(), window=Mock())
        self.bridge = CoreTokenBridge(self.output, self.assignments)

    def bind(self):
        self.bridge.before_frame(self.controller, VIEW)
        self.assignments.confirm()
        self.bridge.before_frame(self.controller, VIEW)

    def align(self):
        self.bind()
        self.bridge.after_frame(self.controller, result())
        self.assertTrue(self.bridge.aligned)

    def test_acquisition_and_explicit_current_assignments_gate_every_move(self):
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result())
        self.assertEqual(self.output.offers, [])
        self.assertTrue(self.output.paused)
        self.assignments.confirm()
        self.bridge.before_frame(self.controller, VIEW)
        for row in (result(acquired=False), result(held=["mini1"]),
                    result(acquisition={"visible_count": 5})):
            self.bridge.after_frame(self.controller, row)
        self.assertEqual(self.output.offers, [])
        self.bridge.after_frame(self.controller, result())
        self.assertEqual(len(self.output.offers), 6)
        self.assertEqual({row[2]["expected_token_id"] for row in self.output.offers},
                         {"token" + str(i) for i in range(1, 7)})
        self.assertEqual(self.controller.expect_display_change.call_count, 6)

    def test_boundary_uncertainty_does_not_snap_continuous_position(self):
        self.align()
        self.bridge.after_frame(self.controller, result(frame=2, capture_seconds=1.2,
            events=[{"mini": "mini1", "to_xy": (137.5, 119.75), "to_cell": None,
                     "to_cells": ["D3", "D4"]}]))
        self.assertEqual(self.output.offers[-1][0:2], ("mini1", (137.5, 119.75)))
        self.assertEqual(self.controller.cancel_expected_display_change.call_count, 1)

    def test_duplicate_result_does_not_reoffer_or_rearm_confirmed_event(self):
        self.align()
        row = result(frame=2, capture_seconds=1.2, events=[{"mini": "mini1", "to_xy": (150, 150)}])
        self.bridge.after_frame(self.controller, row)
        count = len(self.output.offers)
        self.controller.expect_display_change.return_value = False
        self.bridge.after_frame(self.controller, row)
        self.assertEqual(len(self.output.offers), count)
        self.assertFalse(self.output.paused)

    def test_ack_noop_and_retry_replacement_reach_recovery_as_exact_evidence(self):
        self.align()
        old = self.output.box.snapshot()["mini1"]["command_id"]
        command = self.output.box.prepare_position("mini1", POSITIONS["mini1"], "token1", (60, 60),
                                                   0, generation=self.bridge.context[1])
        self.assertEqual(command["commandId"], old)
        self.output.box.reply({**command, "type": "tokenMoveApplied", "changed": False}, .1)
        evidence = self.bridge.before_frame(self.controller, VIEW)
        self.assertEqual(evidence["confirmed_command_ids"], [old])
        self.assertEqual(evidence["unchanged_command_ids"], [old])
        move = self.output.box.moves["mini1"]
        move.state = "failed"
        self.output.box.retry()
        replacement = self.output.box.prepare_position("mini1", POSITIONS["mini1"], "token1", (60, 60),
                                                       1, generation=self.bridge.context[1])
        self.bridge.before_frame(self.controller, VIEW)
        self.assertNotEqual(replacement["commandId"], old)
        self.controller.cancel_expected_display_change.assert_called_with(old, replacement_id=replacement["commandId"])

    def test_scene_and_binding_generation_changes_discard_intents_and_confirmation(self):
        self.align()
        for change in ("scene", "generation"):
            with self.subTest(change=change):
                if change == "generation":
                    self.controller.binding_generation = "generation2"
                    view = VIEW
                else:
                    view = {**VIEW, "sceneId": "other"}
                self.bridge.before_frame(self.controller, view)
                self.assertEqual(self.output.box.snapshot(), {})
                self.assertEqual(self.bridge.mapping, {})
                self.assertFalse(self.bridge.aligned)

    def test_lost_assignments_keep_physical_ids_and_realign_current_positions_after_confirmation(self):
        self.align()
        engine = self.controller.engine
        self.assignments.confirmed = {}
        self.bridge.before_frame(self.controller, VIEW)
        self.controller.reset.assert_not_called()
        self.assertIs(self.controller.engine, engine)
        self.assertTrue(self.output.paused)
        self.assertEqual(self.output.box.snapshot(), {})
        self.assertEqual(self.controller.cancel_expected_display_change.call_count, 6)
        current = {**deepcopy(POSITIONS), "mini1": (150, 150)}
        self.controller.engine.mover.positions = current
        self.bridge.after_frame(self.controller, result(positions=current))
        self.assertEqual(len(self.output.offers), 6)
        self.assignments.confirm()
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result(positions=current))
        self.assertTrue(self.bridge.aligned)
        self.assertEqual(self.output.offers[-6][0:2], ("mini1", (150., 150.)))

    def test_changed_confirmed_mapping_does_not_reset_measured_physical_ids(self):
        self.align()
        self.assignments.confirmed["mini1"] = "replacement"
        self.bridge.before_frame(self.controller, VIEW)
        self.controller.reset.assert_not_called()
        self.assertFalse(self.bridge.aligned)
        self.assertEqual(self.output.box.snapshot(), {})
        self.bridge.after_frame(self.controller, result())
        self.assertEqual(self.output.offers[-6][2]["expected_token_id"], "replacement")

    def test_revoked_binding_does_not_reuse_stale_assignment_cache_or_reset_camera(self):
        self.align()
        state = self.output.box.snapshot()["mini1"]
        self.output.box.reject_position_binding("mini1", state["command_id"])
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result())
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result())
        self.assertFalse(self.bridge.aligned)
        self.assertTrue(self.output.paused)
        self.assertEqual(len(self.output.offers), 6)
        self.controller.reset.assert_not_called()
        self.assignments.confirmed = {}
        self.bridge.before_frame(self.controller, VIEW)
        self.assignments.confirm()
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result())
        self.assertTrue(self.bridge.aligned)

    def test_pending_delivery_does_not_block_another_confirmed_physical_event(self):
        self.align()
        state = self.output.box.snapshot()["mini1"]
        self.output.box.prepare_position("mini1", state["center"], "token1", (60, 60), 0,
                                         generation=self.bridge.context[1])
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result(frame=2, capture_seconds=1.2,
            events=[{"mini": "mini2", "to_xy": (200, 160)}]))
        self.assertEqual(self.output.offers[-1][0:2], ("mini2", (200, 160)))
        self.controller.reset.assert_not_called()
        self.assertFalse(self.output.paused)

    def test_partial_alignment_failure_latches_and_discards_all_pending_moves(self):
        self.bind()
        self.output.fail_mini = "mini3"
        self.bridge.after_frame(self.controller, result())
        self.assertTrue(self.output.paused)
        self.assertEqual(self.output.box.snapshot(), {})
        self.output.fail_mini = None
        self.assignments.confirm()
        self.bridge.before_frame(self.controller, VIEW)
        self.bridge.after_frame(self.controller, result())
        self.assertTrue(self.output.paused)
        self.assertIn("physical observations continue", self.bridge.message)
        self.controller.reset.assert_not_called()
        self.controller.binding_generation = "generation2"
        self.bind()
        self.bridge.after_frame(self.controller, result())
        self.assertTrue(self.bridge.aligned)

    def test_readonly_and_held_results_never_queue_moves(self):
        self.bridge = CoreTokenBridge(self.output, self.assignments, observation_only=True)
        self.bind()
        self.bridge.after_frame(self.controller, result())
        self.assertEqual(self.output.offers, [])
        self.assertTrue(self.output.paused)
        self.bridge = CoreTokenBridge(self.output, self.assignments)
        self.align()
        self.bridge.after_frame(self.controller, result(frame=2, held=["mini1"],
            events=[{"mini": "mini1", "to_xy": (150, 150)}]))
        self.assertEqual(len(self.output.offers), 6)


class CoreCameraPipelineTests(unittest.TestCase):
    def test_six_alignments_ack_redraw_recovery_and_next_move_use_same_pipeline(self):
        cv2.setNumThreads(0)
        with tempfile.TemporaryDirectory() as folder:
            controller = LiveObserverController(None, folder, window_factory=Window, observation_only=False)
            output, assignments = Output(), Assignments()
            bridge = CoreTokenBridge(output, assignments)
            self.addCleanup(controller.close)
            self.addCleanup(bridge.close)

            def frame(now, pixels):
                evidence = bridge.before_frame(controller, VIEW)
                row = controller.update(pixels, None, now, view=VIEW, locked=True,
                    frame_index=round(now * 10), metadata={"scene_visual_revision": output.visual["revision"],
                                                         **evidence})
                bridge.after_frame(controller, row)
                return row

            frame(0, image())
            controller.window.choose(POSITIONS, controller.current[0], controller.current[1])
            frame(.6, image())
            assignments.confirm()
            for now in (1.2, 1.8, 2.4):
                row = frame(now, image())
            self.assertTrue(row["acquired"])
            self.assertTrue(bridge.aligned)
            self.assertEqual(len(output.offers), 6)
            identities = []
            for mini, state in output.box.snapshot().items():
                command = output.box.prepare_position(mini, state["center"], assignments.confirmed[mini],
                    (int(state["center"][0]) - 20, int(state["center"][1]) - 20), 0,
                    generation=bridge.context[1])
                self.assertTrue(output.box.reply({**command, "type": "tokenMoveApplied", "changed": True}, .1))
                identities.append(command["commandId"])
            output.visual = {"revision": 1, "sceneId": VIEW["sceneId"], "reason": "sarweenTokenMove",
                "reasons": ["sarweenTokenMove"], "tokenIds": list(assignments.confirmed.values()),
                "commandIds": identities}
            for now in (3, 3.6, 4.2, 4.8):
                row = frame(now, image(redraw=True))
            self.assertFalse(row["needs_reset"])
            self.assertFalse(row["held"], row["state"])
            self.assertEqual(controller.engine.mover.positions, POSITIONS)
            for now in (5.4, 6, 6.6, 7.2, 7.8, 8.4):
                row = frame(now, image(redraw=True, moved=True))
            self.assertEqual(len(controller.engine.events), 1)
            self.assertEqual(output.offers[-1][0], "mini1")
            self.assertAlmostEqual(output.offers[-1][1][1], 160, delta=8)
            self.assertEqual(len(output.offers), 7)


if __name__ == "__main__":
    unittest.main()
