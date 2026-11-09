"""Live adapter contracts with real image measurements and lossless replay.

No Tk interpreter, physical camera, Foundry service or desktop access is used.
"""

import ast
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import observer_session as session
from observer_journal import ObserverJournal, replay_journal


def view():
    return {"sceneId": "test-scene", "registration": {"left": 0, "top": 0, "right": 400, "bottom": 240},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
            "gridSize": 40, "gridOriginX": 0, "gridOriginY": 0}


POSITIONS = {"mini1": (100, 100), "mini2": (300, 180)}


def image(first=(100, 100)):
    pixels = np.full((240, 400, 3), (150, 90, 40), np.uint8)
    for center in (first, (300, 180)):
        cv2.circle(pixels, center, 30, (35, 75, 180), -1)
        cv2.circle(pixels, center, 24, (160, 160, 160), -1)
        cv2.line(pixels, center, (center[0] - 43, center[1]), (160, 160, 160), 8)
    return pixels


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


class FakeWindow:
    def __init__(self, parent, initialize, on_reset=None, on_record=None, on_close=None, mini_count=6):
        self.parent = parent
        self.initialize = initialize
        self.reset = on_reset
        self.record = on_record
        self.on_close = on_close
        self.mini_count = mini_count
        self.closed = self.initialized = False
        self.status = ""
        self.recording = False
        self.recording_path = ""
        self.recording_changes = []
        self.last_frame = None
        self.invalidations = []

    def set_recording(self, recording, path="", *, summary=None):
        self.recording = bool(recording)
        self.recording_path = str(path)
        self.recording_changes.append((recording, path))
        self.recording_summary = deepcopy(summary)

    def update_recording(self, summary):
        self.recording_summary = deepcopy(summary)

    def set_status(self, message):
        self.status = str(message)

    def update_frame(self, pixels, result=None, *, context=None, status=None):
        self.last_frame = (pixels.copy(), deepcopy(result), deepcopy(context))
        if status is not None:
            self.set_status(status)

    def invalidate_initialization(self, message):
        self.invalidations.append(str(message))
        self.initialized = False
        self.set_status(message)

    def choose(self, positions, pixels, context):
        accepted = self.initialize(positions, pixels, context)
        self.initialized = accepted is not False
        return accepted

    def close(self):
        if not self.closed:
            self.closed = True
            if self.on_close:
                self.on_close()


class LiveObserverSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller = session.LiveObserverController(None, self.root, window_factory=FakeWindow,
                                                         mini_count=2)
        self.addCleanup(self.controller.close)

    def update(self, timestamp, pixels=None, *, locked=True, mapping=None, mask=None, **kwargs):
        pixels = image() if pixels is None else pixels
        return self.controller.update(pixels, mask, timestamp, view=mapping or view(), locked=locked,
                                      frame_index=round(timestamp * 8), **kwargs)

    def initialize(self, timestamp=1):
        self.update(timestamp)
        board, context = self.controller.current
        self.controller.window.choose(POSITIONS, board.copy(), deepcopy(context))
        return self.controller.journal

    def test_initialization_records_frozen_input_before_newer_camera_frames_and_replays_exactly(self):
        first = image()
        self.update(1.0, first)
        frozen, context = (self.controller.current[0].copy(), deepcopy(self.controller.current[1]))
        self.update(1.125, image())  # Camera continues while the user clicks the frozen preview.
        self.controller.window.choose(POSITIONS, frozen, context)
        for index in range(2, 13):
            self.update(1 + index / 8, image((100, 100) if index < 5 else (220, 100)))
        journal = self.controller.journal
        self.controller.toggle_recording()
        inputs = lines(journal.folder / "inputs.jsonl")
        self.assertEqual(inputs[0]["timestamp"], 1.0)
        self.assertEqual(inputs[1]["timestamp"], 1.25)
        np.testing.assert_array_equal(cv2.imread(str(journal.folder / inputs[0]["board"])), frozen)
        replay = replay_journal(journal.folder)
        self.assertTrue(replay["verification_passed"])
        self.assertEqual(replay["frames"], len(inputs))
        self.assertEqual(len(replay["events"]), 1)
        self.assertEqual(replay["events"][0]["mini"], "mini1")
        self.assertEqual(replay["events"][0]["to_cell"], "F3")

    def test_known_display_change_latches_hold_and_records_replayable_evidence(self):
        self.update(1.0, metadata={"scene_visual_revision": 10})
        board, context = self.controller.current
        self.controller.window.choose(POSITIONS, board.copy(), deepcopy(context))
        self.update(1.125, metadata={"scene_visual_revision": 10})
        acquired = self.update(1.25, metadata={"scene_visual_revision": 10})
        self.assertTrue(acquired["acquired"])
        held = self.update(1.375, metadata={"scene_visual_revision": 11})
        self.assertTrue(held["needs_reset"])
        self.assertEqual(held["state"]["reason"], "display_changed_rebind_required")
        self.assertEqual(held["events"], [])
        self.assertEqual(held["positions"], acquired["positions"])
        again = self.update(1.5, metadata={"scene_visual_revision": 11})
        self.assertTrue(again["needs_reset"])
        self.assertFalse(self.controller.window.initialized)
        journal = self.controller.journal
        self.controller._stop_recording("policy_test")
        self.assertEqual(journal.manifest["initialization"]["display_transitions"]["policy"],
                         "hold-known-display-changes-v1")
        self.assertEqual(lines(journal.folder / "inputs.jsonl")[-1]["metadata"]["scene_visual_revision"], 11)
        replay = replay_journal(journal.folder)
        self.assertTrue(replay["verification_passed"])
        self.assertEqual(replay["events"], [])
        from observer_event_journal import read_event_journal
        records = read_event_journal(self.controller.movement_journal.path)["records"]
        self.assertTrue(any(record["type"] == "generation_ended"
                            and record["reason"] == "display_changed_rebind_required" for record in records))
        self.assertEqual(self.controller.movement_journal.status()["acceptedMoveCount"], 0)
        self.assertTrue(all(not item["foundry_token_output"] for item in lines(journal.folder / "decisions.jsonl")))

    def test_lock_loss_remains_held_after_reacquisition_until_explicit_rebinding(self):
        old = self.initialize()
        lost = self.update(1.125, image((220, 100)), locked=False)
        regained = self.update(1.25, image((220, 100)))
        self.assertEqual(lost["state"]["reason"], "camera_lock_lost")
        self.assertTrue(regained["needs_reset"])
        self.assertEqual(regained["events"], [])
        self.assertFalse(self.controller.window.initialized)
        self.assertTrue(self.controller.window.invalidations)
        pixels, context = self.controller.current
        self.controller.window.choose({"mini1": (220, 100), "mini2": (300, 180)}, pixels, context)
        self.assertFalse(self.controller.result["needs_reset"])
        self.assertTrue(old.closed)
        self.assertEqual(self.controller.result["events"], [])
        self.assertTrue(self.controller.window.initialized)

    def test_changed_view_and_calibration_hold_instead_of_emitting_movement(self):
        self.initialize()
        changed = view()
        changed["clientToCanvasTransform"]["tx"] = 25
        result = self.update(1.125, image((220, 100)), mapping=changed)
        self.assertEqual(result["state"]["reason"], "geometry_changed")
        self.assertEqual(result["events"], [])
        with self.assertRaisesRegex(ValueError, "geometry changed"):
            original_context = deepcopy(self.controller.current[1])
            original_context["geometry_key"] = session.geometry_key(view(), image().shape)
            self.controller.initialize(POSITIONS, image(), original_context)
        self.controller.reset()
        self.update(1.25, calibration={"profile": "wide", "revision": 1})
        pixels, context = self.controller.current
        self.controller.window.choose(POSITIONS, pixels, context)
        result = self.update(1.375, calibration={"profile": "wide", "revision": 2})
        self.assertEqual(result["state"]["reason"], "geometry_changed")

    def test_recording_restart_requires_settled_layout_and_makes_a_new_replayable_interval(self):
        first = self.initialize()
        self.update(1.125)
        self.controller.toggle_recording()
        self.assertTrue(first.closed)
        self.controller.result["held"] = ["mini1"]
        with self.assertRaisesRegex(ValueError, "visible and still"):
            self.controller.toggle_recording()
        self.update(1.25)
        self.controller.toggle_recording()
        second = self.controller.journal
        self.assertNotEqual(first.folder, second.folder)
        self.assertTrue(self.controller.is_recording)
        self.assertEqual(lines(second.folder / "inputs.jsonl")[0]["timestamp"], 1.25)
        self.update(1.375)
        self.controller.toggle_recording()
        self.assertTrue(replay_journal(first.folder)["verification_passed"])
        self.assertTrue(replay_journal(second.folder)["verification_passed"])

    def test_printed_marker_layout_change_requires_new_bindings_and_is_recorded(self):
        registration = {"mode": "printed", "profileId": "display-1", "layoutFingerprint": "layout-a", "revision": 1}
        self.update(1, registration=registration)
        pixels, context = self.controller.current
        self.controller.window.choose(POSITIONS, pixels, deepcopy(context))
        journal = self.controller.journal
        self.assertEqual(context["geometry_key"]["registration"], registration)
        self.assertEqual(lines(journal.folder / "inputs.jsonl")[0]["metadata"]["marker_registration"], registration)
        registration["layoutFingerprint"] = "layout-b"
        registration["revision"] = 2
        result = self.update(1.125, image((220, 100)), registration=registration)
        self.assertEqual(result["state"]["reason"], "geometry_changed")
        self.assertEqual(result["events"], [])
        self.controller.toggle_recording()
        self.assertTrue(replay_journal(journal.folder)["verification_passed"])

    def test_missing_camera_pixels_hold_and_are_never_labeled_fresh(self):
        self.initialize()
        result = self.controller.update(None, None, 1.125, view=view(), locked=True, frame_index=9)
        self.assertTrue(result["needs_reset"])
        self.assertFalse(self.controller.current[1]["metadata"]["fresh_board_pixels"])
        self.assertFalse(self.controller.current[1]["locked"])

    def test_latest_exports_are_decodable_locatable_and_marked_inactive_after_close(self):
        index = json.loads((self.root / "latest.json").read_text())
        self.assertTrue(index["active"])
        self.assertEqual(Path(index["folder"]), self.controller.folder.resolve())
        self.initialize()
        self.controller.close()
        index = json.loads((self.root / "latest.json").read_text())
        latest = json.loads(Path(index["status"]).read_text())
        self.assertFalse(index["active"])
        self.assertFalse(latest["active"])
        self.assertFalse(latest["recording"])
        self.assertFalse(latest["foundry_token_output"])
        self.assertFalse(latest["fog_validated"])
        self.assertEqual(cv2.imread(index["image"]).shape[:2], (304, 400))
        self.assertTrue(self.controller.window.closed)
        self.assertTrue(self.controller.journal.closed)
        self.controller.close()  # Closing window/session may reach the same handler twice.

    def test_input_budget_stops_recording_visibly_without_stopping_observation(self):
        original = ObserverJournal
        with patch.object(session, "ObserverJournal", side_effect=lambda *a, **kw: original(*a, **kw, max_seconds=.2)):
            journal = self.initialize()
        self.update(1.125)
        result = self.update(1.25)
        self.assertFalse(self.controller.is_recording)
        self.assertFalse(self.controller.window.recording)
        self.assertEqual(journal.manifest["end_reason"], "recordingLimit")
        self.assertEqual(self.controller.window.recording_summary["end_reason"], "recordingLimit")
        self.assertEqual(self.controller.window.recording_summary["limit_reached"], "seconds")
        self.assertIn("limit", self.controller.error.lower())
        self.assertIsNotNone(self.controller.engine)
        self.assertFalse(result["needs_reset"])
        self.assertTrue(replay_journal(journal.folder)["verification_passed"])

    def test_journal_write_failure_is_visible_and_observation_can_continue(self):
        journal = self.initialize()
        with patch.object(journal, "append", side_effect=OSError("disk full")):
            self.update(1.125)
        self.assertFalse(self.controller.is_recording)
        self.assertIn("disk full", self.controller.error)
        self.assertEqual(journal.manifest["end_reason"], "recordingError")
        self.assertIn("disk full", self.controller.window.recording_summary["error"])
        self.assertIsNotNone(self.controller.engine)
        self.assertIsNotNone(self.update(1.25))

    def test_preview_export_failure_is_bounded_and_does_not_stop_observing(self):
        self.initialize()
        self.controller._last_publish = float("-inf")
        with patch.object(session.cv2, "imencode", side_effect=OSError("preview export unavailable")):
            self.update(1.125)
        self.assertFalse(self.controller.closed)
        self.assertIsNotNone(self.controller.engine)
        self.assertIn("preview export unavailable", self.controller.error)
        self.assertIsNotNone(self.update(1.25))

    def test_nonadvancing_capture_clock_latches_a_visible_hold_without_losing_positions(self):
        self.initialize()
        result = self.update(1.0)
        self.assertTrue(result["needs_reset"])
        self.assertEqual(result["state"]["reason"], "observer_error")
        self.assertEqual(set(result["held"]), set(POSITIONS))
        self.assertEqual(result["positions"], POSITIONS)
        self.assertEqual(result["events"], [])
        self.assertIn("clock did not advance", self.controller.error)
        self.assertIsNone(self.controller.engine)
        self.assertFalse(self.controller.window.initialized)

    def test_real_reference_bytes_and_capture_provenance_are_saved_separately_from_camera_inputs(self):
        ok, encoded = cv2.imencode(".jpg", image())
        self.assertTrue(ok)
        raw = encoded.tobytes()
        entry = {"capture": {"sessionId": "reference-session", "sequence": 2,
                             "capturedAtEpochMs": 123456},
                 "receivedAtMonotonic": 1.25, "viewTransform": view(),
                 "quality": {"status": "nonuniform"}}
        payload = {"image": raw, "entry": entry}
        self.controller.ingest_reference(payload)
        self.controller.ingest_reference(payload)
        reference_folder = self.controller.folder / "references"
        self.assertEqual(len(list(reference_folder.glob("*.jpg"))), 1)
        saved = next(reference_folder.glob("*.jpg"))
        self.assertEqual(saved.read_bytes(), raw)
        info = json.loads(saved.with_suffix(".json").read_text())
        self.assertEqual(info["capture"], entry["capture"])
        self.assertEqual(info["viewTransform"], entry["viewTransform"])
        self.assertIn("not alignment verified", info["camera_pairing"])
        self.update(1.5, metadata={"reference_status": {"message": "Capture not alignment verified"}})
        metadata = self.controller.current[1]["metadata"]
        self.assertEqual(metadata["reference_image"]["sha256"], info["sha256"])
        self.assertEqual(metadata["reference_status"]["message"], "Capture not alignment verified")
        entry["capture"]["sequence"] = 999
        self.assertEqual(metadata["reference_image"]["capture"]["sequence"], 2)

    def test_preview_throttling_retains_events_emitted_between_ui_refreshes(self):
        self.initialize()
        self.controller._last_preview = self.controller._last_publish = 99.0
        with patch.object(session.time, "perf_counter", return_value=100):
            self.update(1.125)
            self.update(1.25)  # Acquire both stationary bases before moving.
        # The following camera frames are processed during one UI refresh
        # interval. A later refresh must include their completed event history.
        with patch.object(session.time, "perf_counter", return_value=100.001):
            for index in range(3, 9):
                self.update(1 + index / 8, image((220, 100)))
        self.assertEqual(len(self.controller.engine.events), 1)
        with patch.object(session.time, "perf_counter", return_value=100.2):
            self.update(2.25, image((220, 100)))
        shown = self.controller.window.last_frame[1]
        self.assertEqual(len(shown["events"]), 1)
        self.assertEqual(shown["events"][0]["mini"], "mini1")

    def test_manifest_failure_during_close_is_bounded_and_remains_visible(self):
        journal = self.initialize()
        with patch("observer_journal._save_json", side_effect=OSError("manifest unavailable")):
            self.controller.close()
        self.assertTrue(self.controller.closed)
        self.assertTrue(self.controller.window.closed)
        self.assertTrue(journal.closed)
        self.assertIn("manifest unavailable", self.controller.error)
        self.assertFalse(json.loads((self.root / "latest.json").read_text())["active"])
        self.assertFalse(json.loads((journal.folder / "manifest.json").read_text())["complete"])

    def test_manifest_failure_during_reset_does_not_silently_clear_the_save_error(self):
        journal = self.initialize()
        with patch("observer_journal._save_json", side_effect=OSError("manifest unavailable")):
            self.controller.reset()
        self.assertIsNone(self.controller.engine)
        self.assertIsNone(self.controller.result)
        self.assertTrue(journal.closed)
        self.assertFalse(self.controller.window.recording)
        self.assertIn("manifest unavailable", self.controller.error)

    def test_close_marks_location_inactive_even_when_final_preview_encoding_fails(self):
        self.initialize()
        with patch.object(session.cv2, "imencode", side_effect=OSError("preview export unavailable")):
            self.controller.close()
        self.assertTrue(self.controller.closed)
        self.assertTrue(self.controller.window.closed)
        self.assertFalse(json.loads((self.root / "latest.json").read_text())["active"])

    def test_recording_start_failure_does_not_leave_a_running_half_initialized_observer(self):
        self.update(1)
        pixels, context = self.controller.current
        with patch.object(session, "ObserverJournal", side_effect=OSError("observer recorder unavailable")):
            with self.assertRaisesRegex(OSError, "recorder unavailable"):
                self.controller.window.choose(POSITIONS, pixels, context)
        self.assertIsNone(self.controller.engine)
        self.assertIsNone(self.controller.result)
        self.assertFalse(self.controller.window.initialized)
        self.assertFalse(self.controller.is_recording)
        self.assertIn("recorder unavailable", self.controller.error)
        # Once storage recovers the same explicit frozen input can start cleanly.
        self.controller.window.choose(POSITIONS, pixels, context)
        self.assertTrue(self.controller.window.initialized)
        self.assertTrue(self.controller.is_recording)

    def test_invalid_frozen_mask_closes_the_new_journal_without_claiming_success(self):
        self.update(1)
        pixels, context = self.controller.current
        broken = deepcopy(context)
        broken["mask"] = np.zeros((3, 4), np.uint8)
        with self.assertRaisesRegex(ValueError, "mask must match"):
            self.controller.window.choose(POSITIONS, pixels, broken)
        self.assertIsNone(self.controller.engine)
        self.assertIsNone(self.controller.result)
        self.assertFalse(self.controller.window.initialized)
        self.assertTrue(self.controller.journal.closed)
        manifest = json.loads((self.controller.journal.folder / "manifest.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["frames"], 0)
        self.assertEqual(manifest["end_reason"], "initializationError")

    def test_adapter_has_no_foundry_imports_or_token_sender_calls(self):
        tree = ast.parse(Path(session.__file__).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports.extend(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names)
        self.assertFalse(any(name and ("foundry" in name or name == "v3_tracking") for name in imports))
        prohibited = {"move_token_to_grid", "queue_control", "sendToPython", "request_vtt_command"}
        called = {node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
                  for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and isinstance(node.func, (ast.Name, ast.Attribute))}
        self.assertFalse(called & prohibited)


if __name__ == "__main__":
    unittest.main()
