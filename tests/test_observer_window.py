"""Observer UI geometry and callback contracts without opening native windows."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import observer_window as ui


class Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value


def window_model(*, initialize=None, count=2):
    """Use the real event handlers with inert widgets and no Tk interpreter."""
    window = ui.ObserverWindow.__new__(ui.ObserverWindow)
    window.parent = Mock()
    window.window = Mock()
    window.on_initialize = initialize or Mock()
    window.on_reset = Mock()
    window.on_record = Mock()
    window.on_close = Mock()
    window.selection = ui.StartingPositions(count)
    window.closed = False
    window.initialized = False
    window._latest_board = None
    window._latest_context = None
    window._latest_result = {}
    window._fit = ui.fit_image(120, 80, 120, 80)
    window._photo = None
    window._canvas_size = (120, 80)
    window._recording = False
    window._record_path = ""
    window._event_keys = set()
    window._events = []
    window.canvas = Mock()
    window._image_item = "image"
    window.choose_button = Mock()
    window.undo_button = Mock()
    window.record_button = Mock()
    window.transcript = Mock()
    window.status = Value()
    window.record_status = Value()
    window._paint = Mock()
    return window


class ObserverSelectionTests(unittest.TestCase):
    def test_clicks_in_black_bars_are_rejected(self):
        fit = ui.fit_image(1280, 720, 800, 600)
        self.assertEqual((fit.display_width, fit.display_height, fit.top), (800, 450, 75))
        self.assertIsNone(fit.image_point(300, 74))
        self.assertIsNone(fit.image_point(300, 525))
        self.assertIsNone(fit.image_point(800, 100))
        self.assertEqual(fit.image_point(400, 300), (640, 360))

    def test_rounded_resize_coordinates_round_trip(self):
        fit = ui.fit_image(1279, 719, 643, 491)
        for point in ((0, 0), (600.25, 211.5), (1278.99, 718.99)):
            mapped = fit.image_point(*fit.canvas_point(*point))
            np.testing.assert_allclose(mapped, point)
        with self.assertRaises(ValueError):
            ui.fit_image(0, 720, 800, 600)

    def test_all_clicks_use_copied_pixels_and_opaque_context(self):
        image = np.zeros((80, 120, 3), np.uint8)
        mask = np.zeros((80, 120), np.uint8)
        context = {"timestamp": 4.5, "mask": mask, "geometry": {"revision": 3}}
        selection = ui.StartingPositions(2)
        selection.start(image, context)
        image[:] = 255
        mask[:] = 255
        context["geometry"]["revision"] = 9
        self.assertFalse(selection.frame.any())
        self.assertFalse(selection.context["mask"].any())
        self.assertEqual(selection.context["geometry"]["revision"], 3)
        self.assertTrue(selection.choose((20, 20)))
        self.assertTrue(selection.choose((90, 60)))
        self.assertEqual(selection.positions, {"mini1": (20.0, 20.0), "mini2": (90.0, 60.0)})
        self.assertTrue(selection.complete)

    def test_duplicate_outside_and_nonfinite_clicks_do_not_advance(self):
        selection = ui.StartingPositions(2)
        self.assertFalse(selection.choose((10, 10)))
        selection.start(np.zeros((80, 120, 3), np.uint8))
        self.assertTrue(selection.choose((20, 20)))
        for point in ((21, 22), (-1, 3), (120, 3), (20, 80), (float("nan"), 20)):
            self.assertFalse(selection.choose(point))
        self.assertEqual(len(selection.points), 1)
        self.assertTrue(selection.choose((90, 60)))
        self.assertFalse(selection.choose((70, 50)))
        self.assertTrue(selection.undo())
        self.assertTrue(selection.choose((70, 50)))
        self.assertEqual(selection.positions["mini2"], (70.0, 50.0))

    def test_invalid_count_or_image_cannot_start_binding(self):
        for value in (0, 13, 2.5, True):
            with self.assertRaises(ValueError):
                ui.StartingPositions(value)
        for image in (np.zeros((5, 5)), np.zeros((5, 5, 4), np.uint8), np.zeros((1, 2, 3), np.uint8)):
            with self.assertRaises(ValueError):
                ui.StartingPositions().start(image)


class ObserverCallbackTests(unittest.TestCase):
    def test_initialization_callback_gets_frozen_frame_despite_new_video(self):
        callback = Mock(return_value=True)
        window = window_model(initialize=callback)
        first = np.zeros((80, 120, 3), np.uint8)
        mask = np.zeros((80, 120), np.uint8)
        window.update_frame(first, context={"timestamp": 1.0, "mask": mask})
        first[:] = 99
        mask[:] = 99
        window.start_selection()
        window._choose(SimpleNamespace(x=20, y=20))
        window.update_frame(np.full((80, 120, 3), 200, np.uint8),
                            context={"timestamp": 3.0, "mask": np.ones_like(mask)},
                            status="Live status must not replace click instructions")
        self.assertIn("Mini 2", window.status.value)
        callback.assert_not_called()
        window._choose(SimpleNamespace(x=90, y=60))
        positions, frozen, context = callback.call_args.args
        self.assertEqual(positions, {"mini1": (20.0, 20.0), "mini2": (90.0, 60.0)})
        self.assertFalse(frozen.any())
        self.assertFalse(context["mask"].any())
        self.assertEqual(context["timestamp"], 1.0)
        self.assertTrue(window.initialized)
        self.assertFalse(window.selecting)
        self.assertIn("Checking that every base is visible", window.status.value)
        self.assertNotIn("Observer ready", window.status.value)
        window.on_reset.assert_called_once_with()
        window.parent.update.assert_not_called()

    def test_invalid_geometry_refusal_does_not_mark_observer_ready(self):
        window = window_model(initialize=Mock(return_value=False))
        window.update_frame(np.zeros((80, 120, 3), np.uint8))
        window.start_selection()
        window._choose(SimpleNamespace(x=20, y=20))
        window._choose(SimpleNamespace(x=90, y=60))
        self.assertFalse(window.initialized)
        self.assertTrue(window.selecting)
        self.assertIn("no longer valid", window.status.value)
        window.undo()
        self.assertEqual(len(window.selection.points), 1)

    def test_callback_exception_is_explained_and_recoverable(self):
        window = window_model(initialize=Mock(side_effect=ValueError("Marker lock lost")))
        window.update_frame(np.zeros((80, 120, 3), np.uint8))
        window.start_selection()
        window._choose(SimpleNamespace(x=20, y=20))
        window._choose(SimpleNamespace(x=90, y=60))
        self.assertIn("Marker lock lost", window.status.value)
        self.assertFalse(window.initialized)
        window.invalidate_initialization("Choose again after marker lock returns")
        self.assertFalse(window.selecting)
        self.assertIsNone(window.selection.frame)
        self.assertEqual(window.status.value, "Choose again after marker lock returns")

    def test_events_are_not_repeated_when_history_is_resent(self):
        window = window_model()
        event = {"mini": "mini2", "sequence": 1, "from_cell": "C3", "to_cell": None,
                 "destination_cells": ["D4", "E4"], "to_xy": [20, 20], "confirmed_time": 3.0}
        window._remember_events({"events": [event]})
        window._remember_events({"events": [event]})
        self.assertEqual(len(window._events), 1)
        self.assertIn("D4/E4 (uncertain)", window.transcript.insert.call_args.args[1])
        self.assertIn("?", ui.movement_text({"mini": "mini1", "to_cell": "A1"}))

    def test_recording_callback_and_saved_path_are_visible(self):
        window = window_model()
        window.initialized = True
        window._record()
        window.on_record.assert_called_once_with()
        window.set_recording(True, "/tmp/observer-inputs")
        self.assertIn("Recording observer inputs", window.record_status.value)
        window.set_recording(False, "/tmp/observer-inputs")
        self.assertIn("Observer recording folder", window.record_status.value)
        window.on_record.side_effect = ValueError("disk full")
        window._record()
        self.assertIn("disk full", window.status.value)

    def test_close_is_idempotent_and_does_not_close_application_root(self):
        window = window_model()
        window.close()
        window.close()
        window.window.destroy.assert_called_once_with()
        window.on_close.assert_called_once_with()
        window.parent.destroy.assert_not_called()

    def test_engine_held_list_only_holds_named_mini_and_cell_uncertainty_is_shown(self):
        window = window_model()
        # Exercise the actual renderer with a fake image owner and Tk canvas.
        del window._paint
        window.canvas.bbox.return_value = (0, 0, 40, 12)
        result = {"positions": {"mini1": (20, 20), "mini2": (90, 60)},
                  "current_cells": {"mini2": {"xy": (90, 60), "boundary_uncertain": True}},
                  "held": ["mini1"], "state": {"phase": "settling"}}
        with patch.object(ui.ImageTk, "PhotoImage", return_value=object()):
            window.update_frame(np.zeros((80, 120, 3), np.uint8), result)
        labels = [call.kwargs["text"] for call in window.canvas.create_text.call_args_list]
        self.assertEqual(labels, ["mini1 (held)", "mini2 (?)"])

    def test_unacquired_positions_are_all_held_even_if_no_missing_list_is_supplied(self):
        window = window_model()
        del window._paint
        window.canvas.bbox.return_value = (0, 0, 40, 12)
        result = {"positions": {"mini1": (20, 20), "mini2": (90, 60)},
                  "acquired": False, "held": [], "state": {"phase": "acquiring"}}
        with patch.object(ui.ImageTk, "PhotoImage", return_value=object()):
            window.update_frame(np.zeros((80, 120, 3), np.uint8), result)
        labels = [call.kwargs["text"] for call in window.canvas.create_text.call_args_list]
        self.assertEqual(labels, ["mini1 (held)", "mini2 (held)"])

    def test_labels_near_preview_edges_move_inward_without_moving_measured_base(self):
        window = window_model()
        del window._paint
        window.canvas.create_text.return_value = "edge-label"
        window.canvas.bbox.side_effect = [(116, -8, 161, 8), (72, 3, 117, 19)]
        with patch.object(ui.ImageTk, "PhotoImage", return_value=object()):
            window.update_frame(np.zeros((80, 120, 3), np.uint8),
                                {"positions": {"mini1": (115, 5)}, "state": {"phase": "settled"}})
        window.canvas.move.assert_called_once_with("edge-label", -44, 11)
        window.canvas.create_rectangle.assert_called_once_with(72, 3, 117, 19,
                                                                fill="#111111", outline="",
                                                                tags="observation")
        # Canvas coordinates for the physical center itself remain (115, 5).
        oval = window.canvas.create_oval.call_args.args
        self.assertEqual(((oval[0]+oval[2])/2, (oval[1]+oval[3])/2), (115, 5))


if __name__ == "__main__":
    unittest.main()
