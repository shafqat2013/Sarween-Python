"""Accepted-move durability tests; no physical camera, Tk or Foundry access."""
from copy import deepcopy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from observer_event_journal import ObserverEventJournal, read_event_journal
import observer_session as session
from tests.test_observer_session import FakeWindow, POSITIONS, view


def move(sequence=1, *, mini="mini1", uncertain=False):
    return {"sequence": sequence, "mini": mini, "from_xy": [100, 100], "to_xy": [220, 100],
            "from_cell": "C3", "to_cell": None if uncertain else "F3",
            "destination_cells": ["E3", "F3"] if uncertain else ["F3"],
            "boundary_uncertain": uncertain, "uncertainty_px": 6 if uncertain else 2.5,
            "confirmed_time": 10 + sequence, "departure_time": 9 + sequence,
            "first_visible_time": 9.8 + sequence, "samples": 3, "frame": sequence * 10,
            "capture_seconds": 10 + sequence}


class EventJournalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "movement-events.jsonl"
        self.log = ObserverEventJournal(self.path, "run-one", recent_limit=3)
        self.generation = self.log.begin_generation({"initial_positions": POSITIONS}, reason="initial_bindings")

    def movements(self):
        return [record for record in read_event_journal(self.path)["records"] if record["type"] == "move"]

    def append(self, event, generation=None):
        return self.log.append_events(generation or self.generation, [event], reason="single_move_confirmed")

    def test_same_event_is_durable_once_and_retains_uncertain_cells_and_reason(self):
        event = move(uncertain=True)
        self.assertTrue(self.append(event))
        self.assertTrue(self.append(deepcopy(event)))
        saved, = self.movements()
        self.assertEqual(saved["event"], event)
        self.assertEqual(saved["reason"], "single_move_confirmed")
        self.assertFalse(saved["foundry_token_output"])
        self.assertEqual(self.log.status()["savedMoveCount"], 1)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.append({**event, "to_cell": "G3"})

    def test_engine_sequence_restart_uses_new_generation_and_run_movement_number(self):
        self.append(move())
        self.log.end_generation("bindings_reset")
        second = self.log.begin_generation({"initial_positions": POSITIONS}, reason="new_bindings")
        self.append(move(mini="mini2"), second)
        saved = self.movements()
        self.assertEqual([record["movementSequence"] for record in saved], [1, 2])
        self.assertEqual([record["generation"] for record in saved], [1, 2])
        self.assertEqual([record["event"]["sequence"] for record in saved], [1, 1])
        self.assertEqual(len({record["recordId"] for record in saved}), 2)
        with self.log.csv_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["generation"] for row in rows], ["1", "2"])
        self.assertEqual([row["engine_sequence"] for row in rows], ["1", "1"])
        self.assertEqual([row["movement_sequence"] for row in rows], ["1", "2"])

    def test_recent_history_is_bounded_but_authoritative_file_keeps_every_move(self):
        for sequence in range(1, 9):
            self.append(move(sequence))
        self.assertEqual([record["movementSequence"] for record in self.log.recent_events()], [6, 7, 8])
        self.assertTrue(all(record["saved"] for record in self.log.recent_events()))
        self.assertEqual(len(self.movements()), 8)
        with self.log.csv_path.open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 8)

    def test_exports_preserve_uncertainty_utc_and_original_capture_clock(self):
        event = move(uncertain=True)
        self.append(event)
        saved, = self.movements()
        with self.log.csv_path.open(newline="") as stream:
            row, = list(csv.DictReader(stream))
        self.assertEqual(row["run_id"], "run-one")
        self.assertEqual(row["event_id"], saved["recordId"])
        self.assertEqual(row["created_at_utc"], saved["createdAt"])
        self.assertEqual(float(row["capture_seconds"]), event["capture_seconds"])
        self.assertEqual(int(row["frame"]), event["frame"])
        self.assertEqual(row["to_cell"], "")
        self.assertEqual(json.loads(row["destination_cells"]), ["E3", "F3"])
        self.assertEqual(row["boundary_uncertain"], "True")
        text = self.log.text_path.read_text()
        self.assertIn("E3/F3 (uncertain)", text)
        self.assertIn(saved["createdAt"], text)
        self.assertIn("NOT a video-relative timestamp", text)
        self.assertIn("Missed moves are not reconstructed", text)
        self.assertTrue(self.log.status()["exports"]["current"])

    def test_derived_export_failure_does_not_mark_durable_jsonl_unsaved(self):
        with patch.object(self.log, "_write_export", side_effect=OSError("CSV folder unavailable")) as export:
            self.assertTrue(self.append(move()))
            self.assertEqual(self.log.status()["savedMoveCount"], 1)
            self.assertEqual(self.log.status()["pendingRecords"], 0)
            self.assertEqual(self.log.status()["error"], "")
            self.assertIn("durable JSONL is unchanged", self.log.status()["exportError"])
            self.assertFalse(self.log.status()["exports"]["current"])
            self.assertEqual(len(self.movements()), 1)
            for _ in range(5):
                self.assertTrue(self.log.flush())
            self.assertEqual(export.call_count, 1)
        self.assertTrue(self.log.export_saved_moves())
        self.assertEqual(self.log.status()["exportError"], "")
        self.assertTrue(self.log.status()["exports"]["current"])

    def test_exports_are_written_for_saved_moves_and_close_not_empty_frame_flushes(self):
        with patch.object(self.log, "_write_export", wraps=self.log._write_export) as export:
            self.append(move())
            self.assertEqual(export.call_count, 2)
            for _ in range(5):
                self.log.flush()
            self.assertEqual(export.call_count, 2)
            self.log.close()
            self.assertEqual(export.call_count, 4)
        self.assertTrue(self.log.status()["exports"]["runClosed"])

    def test_empty_closed_run_exports_no_invented_movement(self):
        self.assertTrue(self.log.close())
        with self.log.csv_path.open(newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        self.assertIn("Saved moves: 0; run closed: True", self.log.text_path.read_text())

    def test_fsync_failure_after_complete_write_retries_without_duplicate_record(self):
        with patch("observer_event_journal.os.fsync", side_effect=OSError("Disk sync unavailable")):
            self.assertFalse(self.append(move()))
        self.assertEqual(self.log.status()["pendingMoves"], 1)
        self.assertEqual(self.log.status()["savedMoveCount"], 0)
        self.assertIn("not fully saved", self.log.status()["error"])
        self.assertTrue(self.log.flush())
        self.assertEqual(len(self.movements()), 1)
        self.assertEqual(self.log.status()["pendingRecords"], 0)
        self.assertEqual(self.log.status()["savedMoveCount"], 1)
        self.assertEqual(self.log.status()["error"], "")

    def test_partial_write_is_retained_and_reported_then_pending_event_is_recovered(self):
        def partial(record):
            raw = json.dumps(record).encode()
            with self.path.open("ab") as stream:
                stream.write(raw[:27])
            raise OSError("Disk filled partway through record")
        with patch.object(self.log, "_append_record", side_effect=partial):
            self.assertFalse(self.append(move()))
        self.assertTrue(self.log.flush())
        self.assertEqual(len(self.movements()), 1)
        self.assertTrue(self.log.status()["integrityWarnings"])
        self.assertTrue(read_event_journal(self.path)["warnings"])

    def test_changed_on_disk_pending_move_is_not_falsely_marked_saved_after_fsync_failure(self):
        with patch("observer_event_journal.os.fsync", side_effect=OSError("Sync failed")):
            self.assertFalse(self.append(move()))
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        records[-1]["event"]["to_cell"] = "WRONG"
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.assertFalse(self.log.flush())
        self.assertIn("Conflicting", self.log.status()["error"])
        self.assertEqual(self.log.status()["pendingMoves"], 1)
        self.assertEqual(self.log.status()["savedMoveCount"], 0)
        self.assertFalse(self.log.recent_events()[0]["saved"])

    def test_changed_pending_lifecycle_record_is_also_rejected_on_rescan(self):
        with patch("observer_event_journal.os.fsync", side_effect=OSError("Sync failed")):
            self.log.end_generation("camera_lock_lost")
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        records[-1]["reason"] = "changed_outside_the_writer"
        self.path.write_text("".join(json.dumps(record) + "\n" for record in records))
        self.assertFalse(self.log.flush())
        self.assertIn("Conflicting", self.log.status()["error"])
        self.assertEqual(self.log.status()["pendingRecords"], 1)

    def test_complete_json_with_missing_newline_does_not_duplicate_on_retry(self):
        def no_newline(record):
            with self.path.open("ab") as stream:
                stream.write(json.dumps(record).encode())
            raise OSError("Interrupted before newline")
        with patch.object(self.log, "_append_record", side_effect=no_newline):
            self.assertFalse(self.append(move()))
        self.assertTrue(self.log.flush())
        self.assertEqual(len(self.movements()), 1)
        raw_records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len([record for record in raw_records if record["type"] == "move"]), 1)

    def test_reopen_recovers_ids_and_closed_run_stays_closed(self):
        self.append(move())
        reopened = ObserverEventJournal(self.path, "run-one")
        reopened.append_events(self.generation, [move()], reason="repeated_delivery")
        self.assertEqual(len(self.movements()), 1)
        reopened.close()
        closed = ObserverEventJournal(self.path, "run-one")
        self.assertTrue(closed.status()["closed"])
        self.assertEqual(closed.status()["acceptedMoveCount"], 1)
        with self.assertRaises(RuntimeError):
            closed.begin_generation({}, reason="must_use_new_run")
        records = read_event_journal(self.path)["records"]
        self.assertEqual(sum(record["type"] == "run_started" for record in records), 1)
        self.assertEqual(sum(record["type"] == "run_closed" for record in records), 1)

    def test_close_failure_is_visible_and_later_close_retries_without_duplicate_marker(self):
        self.append(move())
        with patch.object(self.log, "_append_record", side_effect=OSError("Disk unavailable")):
            self.assertFalse(self.log.close())
        self.assertFalse(self.log.status()["closed"])
        self.assertTrue(self.log.status()["error"])
        self.assertTrue(self.log.close())
        self.assertTrue(self.log.status()["closed"])
        self.assertEqual(sum(record["type"] == "run_closed" for record in read_event_journal(self.path)["records"]), 1)


class ScriptedEngine:
    """Test acceptance source; image recognition is intentionally outside this test."""
    def __init__(self, geometry, positions, cells=None, **_options):
        self.geometry = geometry
        self.mover = SimpleNamespace(positions=deepcopy(positions))
        self.confirmed_cells = cells or {mini: geometry.describe(xy)["cell"] for mini, xy in positions.items()}
        self.events, self.next_events = [], []
        self.needs_reset = False

    def initialization(self):
        return {"kind": "scripted-test-acceptance-source", "initial_positions": self.mover.positions,
                "foundry_token_output": False}

    def process(self, _board, _mask, timestamp, **_context):
        emitted, self.next_events = self.next_events, []
        self.events.extend(deepcopy(emitted))
        return {"capture_seconds": timestamp, "positions": deepcopy(self.mover.positions),
                "current_cells": {mini: self.geometry.describe(xy) for mini, xy in self.mover.positions.items()},
                "events": deepcopy(emitted), "move_count": len(self.events), "held": [], "needs_reset": False,
                "state": {"phase": "confirmed" if emitted else "settled",
                          "reason": "single_move_confirmed" if emitted else "layout_confirmed"},
                "acquired": True, "acquisition": {"required": 2, "visible_count": 2, "ready_count": 2},
                "foundry_token_output": False}


class ControllerEventDurabilityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.factory = patch.object(session, "ObserverEngine", ScriptedEngine)
        self.factory.start()
        self.addCleanup(self.factory.stop)
        drawing = patch.object(session, "draw_observer_frame", side_effect=lambda board, *_: board)
        drawing.start()
        self.addCleanup(drawing.stop)
        self.controller = session.LiveObserverController(None, self.directory.name, window_factory=FakeWindow)
        self.addCleanup(self.controller.close)
        self.pixels = np.zeros((240, 400, 3), np.uint8)
        self.update(1.0)
        board, context = self.controller.current
        self.controller.window.choose(POSITIONS, board, deepcopy(context))

    def update(self, timestamp):
        return self.controller.update(self.pixels, None, timestamp, view=view(), locked=True, frame_index=round(timestamp * 10))

    def saved(self):
        return [record for record in read_event_journal(self.controller.movement_journal.path)["records"] if record["type"] == "move"]

    def test_between_tick_event_survives_in_file_and_next_latest_export(self):
        with patch.object(session.time, "perf_counter", return_value=100):
            self.controller._last_publish = float("-inf")
            self.update(1.1)
        self.controller.engine.next_events = [move(uncertain=True)]
        with patch.object(session.time, "perf_counter", return_value=100.01):
            self.update(1.2)
            self.update(1.3)
        self.assertEqual(len(self.saved()), 1)
        with patch.object(session.time, "perf_counter", return_value=101.1):
            self.update(1.4)
        latest = json.loads((self.controller.folder / "latest.json").read_text())
        self.assertEqual(latest["observation"]["events"], [])
        self.assertEqual(latest["movement_events"][0]["event"], move(uncertain=True))
        self.assertEqual(latest["movement_event_journal"]["savedMoveCount"], 1)
        self.assertFalse(latest["foundry_token_output"])

    def test_frame_recording_limit_does_not_stop_movement_history(self):
        self.controller.journal.max_seconds = .2
        self.controller.engine.next_events = [move()]
        self.update(1.3)
        self.assertFalse(self.controller.is_recording)
        self.controller.engine.next_events = [move(2)]
        self.update(1.6)
        self.assertEqual([record["event"]["sequence"] for record in self.saved()], [1, 2])
        self.assertEqual(self.controller.journal.frames, 1)
        self.assertEqual(self.controller.journal.manifest["end_reason"], "recordingLimit")

    def test_recording_restart_and_rebind_get_new_event_generations(self):
        self.controller.engine.next_events = [move()]
        self.update(1.2)
        self.controller.toggle_recording()  # Stop input recording; observer continues.
        self.controller.toggle_recording()  # Recreates the engine with sequence reset.
        self.controller.engine.next_events = [move()]
        self.update(1.4)
        self.controller.reset()
        board, context = self.controller.current
        self.controller.window.choose(POSITIONS, board, deepcopy(context))
        self.controller.engine.next_events = [move()]
        self.update(1.6)
        saved = self.saved()
        self.assertEqual([record["generation"] for record in saved], [1, 2, 3])
        self.assertEqual([record["movementSequence"] for record in saved], [1, 2, 3])
        self.assertEqual(len({record["recordId"] for record in saved}), 3)

    def test_save_error_is_visible_pending_event_recovers_and_close_is_auditable(self):
        self.controller.engine.next_events = [move()]
        with patch.object(self.controller.movement_journal, "_append_record", side_effect=OSError("Disk full")):
            self.update(1.2)
            self.assertIn("not fully saved", self.controller.window.status)
            self.assertEqual(self.controller.movement_journal.status()["pendingMoves"], 1)
        self.update(1.4)
        self.controller.close()
        latest = json.loads((self.controller.folder / "latest.json").read_text())
        self.assertFalse(latest["active"])
        self.assertTrue(latest["movement_event_journal"]["closed"])
        self.assertEqual(latest["movement_event_journal"]["pendingMoves"], 0)
        self.assertEqual(len(self.saved()), 1)
        second = session.LiveObserverController(None, self.directory.name, window_factory=FakeWindow)
        self.addCleanup(second.close)
        self.assertNotEqual(second.movement_journal.run_id, self.controller.movement_journal.run_id)
        self.assertEqual(second.movement_journal.status()["acceptedMoveCount"], 0)

    def test_failed_preview_encoding_cannot_discard_an_accepted_move(self):
        self.controller.toggle_recording()
        self.controller._last_publish = float("-inf")
        self.controller.engine.next_events = [move()]
        with patch.object(session.cv2, "imencode", return_value=(False, None)):
            self.update(1.2)
        self.assertEqual(len(self.saved()), 1)
        self.assertIn("preview", self.controller.error.lower())
        self.controller._last_publish = float("-inf")
        self.update(1.4)
        latest = json.loads((self.controller.folder / "latest.json").read_text())
        self.assertEqual(latest["movement_events"][0]["event"], move())


if __name__ == "__main__":
    unittest.main()
