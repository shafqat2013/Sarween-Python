"""Audit records need only saved bytes and JSON, never Tk, CV or live hardware."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_catalog as audit
from replay_review import ReviewSession


class AuditCatalogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.video = self.videos / "sarween_rec_20261103_194320.mp4"
        self.video.write_bytes(b"saved media bytes; no decoder should be opened")
        self.reviews = self.root / "reviews"
        self.manifest = self.root / "audit_sources.json"
        self.manifest.write_text(json.dumps({"schemaVersion": 1, "sources": [], "unavailable": []}))
        self.patch = patch.object(audit, "SOURCES_FILE", self.manifest)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def source(self, events=None, *, counterfactual=False, case_report=False):
        events = events if events is not None else [{"mini": "red10", "time_seconds": 2.5,
            "from_cell": "r0c0", "to_cell": "r1c1", "source": "detection"}]
        path = self.root / "events.json"
        fingerprint = hashlib.sha256(self.video.read_bytes()).hexdigest()
        if case_report:
            value = {"cases": [{"video": "different.mp4", "events": []},
                {"video": str(self.video), "events": events, "label_source": "assistant-visually-reviewed",
                 "diagnostics": {"inputs": {"video": {"sha256": fingerprint}}}}]}
            path.write_text(json.dumps(value))
        else:
            path.write_text(json.dumps(events))
        inputs = self.root / "inputs.json"
        inputs.write_text(json.dumps({"video": str(self.video), "videoSha256": fingerprint,
                                     "initialReseedCounterfactual": counterfactual}))
        entry = {"video": self.video.name, "path": "events.json", "label": "Replay · refreshed background"
                 if counterfactual else "Archived replay", "note": "Predictions require review."}
        if not case_report:
            entry["inputs"] = "inputs.json"
        self.manifest.write_text(json.dumps({"schemaVersion": 1, "sources": [entry], "unavailable": []}))
        return path

    def record(self):
        return audit.AuditRecord(self.video, folder=self.reviews)

    def test_discovery_deduplicates_roots_and_excludes_generated_or_demo_copies(self):
        second = self.videos / "sarween_rec_20260420_155413.mp4"
        second.write_bytes(b"old video")
        alias = self.videos / "alias.mp4"
        alias.symlink_to(self.video)
        for name in ("demo", "build", "deprecated"):
            folder = self.root / name
            folder.mkdir()
            (folder / "copy.mp4").write_bytes(b"generated")
        (self.videos / "not-video.json").write_text("{}")
        paths = audit.discover_videos([self.videos, self.videos, self.root / "demo", self.root / "build", self.root / "deprecated"])
        self.assertEqual(paths, [self.video.resolve(), second.resolve()])

    def test_curated_source_normalizes_cells_preserves_time_and_does_not_write_on_open(self):
        self.source(counterfactual=True)
        record = self.record()
        self.assertEqual(record.rows[0]["from_cell"], "A1")
        self.assertEqual(record.rows[0]["to_cell"], "B2")
        self.assertEqual(record.rows[0]["time_seconds"], 2.5)
        self.assertEqual(record.rows[0]["status"], "Unreviewed")
        self.assertIn("background was refreshed", record.source_note)
        self.assertEqual(record.rows[0]["origin"], "Replay")
        self.assertFalse(record.review.path.exists())
        self.assertEqual(record.review.data["labels"], [])
        self.assertFalse(record.complete)

    def test_report_selection_is_by_video_not_case_index_and_does_not_import_assistant_truth(self):
        self.source(case_report=True)
        record = self.record()
        self.assertEqual(len(record.rows), 1)
        self.assertEqual(record.rows[0]["to_cell"], "B2")
        self.assertEqual(record.review.data["labels"], [])
        self.assertEqual(record.rows[0]["status"], "Unreviewed")

    def test_saved_analysis_and_human_review_win_without_reset_or_reanalysis(self):
        self.source()
        record = self.record()
        row_id = record.rows[0]["id"]
        record.decide(row_id, True, duration=20, grid=(10, 10))
        record.decide(row_id, True, duration=20, grid=(10, 10))
        self.assertEqual(len(record.review.data["labels"]), 1)
        record.set_position(7.5)
        record.save_notes("Keep my notes")
        record.mark_complete(True)
        before = record.review.path.read_bytes()
        (self.root / "events.json").write_text("malformed external source changed later")
        reopened = self.record()
        self.assertEqual(reopened.review.path.read_bytes(), before)
        self.assertEqual(reopened.position_seconds, 7.5)
        self.assertEqual(reopened.notes, "Keep my notes")
        self.assertTrue(reopened.complete)
        self.assertEqual(reopened.rows[0]["status"], "Confirmed")
        self.assertEqual(audit.review_status(self.video, self.reviews), "Reviewed")

    def test_import_preserves_existing_labels_decisions_and_completion(self):
        self.source()
        review = ReviewSession(self.video, folder=self.reviews)
        review.data.update(complete=True, decisions={"old-source": "rejected"},
            labels=[{"id": "manual-old", "mini": "blue", "time_seconds": 4,
                     "to_cell": "D4", "provenance": "manual-video-review"}])
        review.save()
        before = review.path.read_bytes()
        record = self.record()
        self.assertTrue(record.complete)
        self.assertEqual(record.review.data["decisions"], {"old-source": "rejected"})
        self.assertEqual(record.review.data["labels"], review.data["labels"])
        self.assertEqual(review.path.read_bytes(), before)
        self.assertEqual([r["kind"] for r in record.rows], ["prediction", "manual"])

    def test_wrong_missed_remove_and_resume_are_atomic_canonical_mutations(self):
        self.source()
        record = self.record()
        predicted = record.rows[0]["id"]
        record.decide(predicted, True, duration=20)
        self.assertEqual(len(record.rows), 1, "confirmed predictions must not appear twice")
        record.decide(predicted, False, duration=20)
        self.assertEqual(record.rows[0]["status"], "Wrong")
        self.assertEqual(record.review.data["labels"], [])
        label = record.add_missed(5, "Blue", "c3", "B2", duration=20, grid=(10, 10), notes="hand was clear")
        self.assertEqual(record.rows[1]["to_cell"], "C3")
        self.assertEqual(record.rows[1]["status"], "Confirmed")
        self.assertEqual(record.rows[1]["kind"], "manual")
        record.set_position(6)
        self.assertFalse(list(record.review.path.parent.glob("movements.*")), "playhead saving must not regenerate exports")
        reopened = self.record()
        self.assertEqual(len(reopened.rows), 2)
        self.assertEqual(reopened.position_seconds, 6)
        reopened.remove_label(label["id"])
        self.assertEqual(len(reopened.rows), 1)
        with self.assertRaisesRegex(ValueError, "manually added"):
            reopened.remove_label(predicted)

    def test_validation_and_failed_save_do_not_change_saved_human_data(self):
        self.source()
        record = self.record()
        record.set_position(2)
        before = copy.deepcopy(record.review.data)
        saved = record.review.path.read_bytes()
        for value in (-1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                record.set_position(value)
        with self.assertRaises(ValueError):
            record.add_missed(30, "Red", "A1", duration=20)
        with self.assertRaises(ValueError):
            record.add_missed(5, "Red", "ZZ99", duration=20, grid=(10, 10))
        with patch.object(audit, "atomic_write_json", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                record.decide(record.rows[0]["id"], True, duration=20)
        self.assertEqual(record.review.data, before)
        self.assertEqual(record.review.path.read_bytes(), saved)

    def test_missing_transcript_is_manual_reviewable_but_malformed_source_is_not_stationary(self):
        record = self.record()
        self.assertEqual(record.rows, [])
        self.assertEqual(record.source_label, "No saved transcript")
        self.assertEqual(audit.review_status(self.video, self.reviews), "To review")
        self.source()
        for content in ("broken", "{}", '[{"mini":"Red","time_seconds":null,"to_cell":"A1"}]'):
            (self.root / "events.json").write_text(content)
            with self.assertRaises(ValueError):
                self.record()
        self.source()
        inputs = self.root / "inputs.json"
        data = json.loads(inputs.read_text())
        data["videoSha256"] = "0" * 64
        inputs.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.record()

    def test_live_rows_are_labeled_live_and_normalized(self):
        self.video.with_suffix(".tracking.json").write_text(json.dumps({"fps": 10,
            "trackingEvents": [{"frame": 15, "mini": "Red", "cell": "r2c3"}]}))
        record = self.record()
        self.assertTrue(record.source_label.startswith("Live"))
        self.assertEqual(record.rows[0]["time_seconds"], 1.5)
        self.assertEqual(record.rows[0]["to_cell"], "D3")
        self.assertEqual(record.rows[0]["origin"], "Live")

    def test_unplayable_flag_cannot_mark_complete_and_repaired_file_does_not_inherit_it(self):
        self.manifest.write_text(json.dumps({"schemaVersion": 1, "sources": [], "unavailable": [{
            "video": self.video.name, "videoSha256": hashlib.sha256(self.video.read_bytes()).hexdigest(),
            "reason": "Incomplete recording: missing MP4 index"}]}))
        record = self.record()
        self.assertIn("missing MP4 index", record.unavailable_reason)
        self.assertEqual(audit.review_status(self.video, self.reviews), "Unavailable")
        with self.assertRaises(ValueError):
            record.mark_complete(True)
        self.assertFalse(record.review.path.exists())
        self.video.write_bytes(b"repaired recording")
        self.assertEqual(audit.review_status(self.video, self.reviews), "To review")

    def test_exports_include_source_provenance_and_all_review_states(self):
        self.source(counterfactual=True)
        record = self.record()
        record.decide(record.rows[0]["id"], False, duration=20)
        record.add_missed(4, "Blue", "C3", duration=20)
        record.save_notes("Independent audit")
        base = record.export(self.root / "exports")
        output = json.loads(Path(str(base) + ".json").read_text())
        self.assertEqual([row["status"] for row in output["movements"]], ["Wrong", "Confirmed"])
        self.assertIn("background was refreshed", output["transcript_source"]["note"])
        self.assertIn("background was refreshed", Path(str(base) + ".txt").read_text())
        self.assertIn("Replay · refreshed background", Path(str(base) + ".csv").read_text())

    def test_restore_is_atomic_canonical_only_and_copies_the_snapshot(self):
        self.source()
        record = self.record()
        before = copy.deepcopy(record.review.data)
        record.decide(record.rows[0]["id"], True, duration=20)
        before.update(position_seconds=7, notes="New notes survive Undo")
        with patch.object(record.review, "save", side_effect=AssertionError("must not regenerate exports")):
            record.restore(before)
        self.assertEqual(record.rows[0]["status"], "Unreviewed")
        self.assertEqual(self.record().position_seconds, 7)
        self.assertEqual(self.record().notes, "New notes survive Undo")
        before["notes"] = "Do not retain caller references"
        self.assertEqual(record.notes, "New notes survive Undo")
        self.assertFalse(list(record.review.path.parent.glob("movements.*")))

    def test_restore_rejects_other_video_and_invalid_review_without_mutating(self):
        self.source()
        record = self.record()
        record.set_position(2)
        before = copy.deepcopy(record.review.data)
        saved = record.review.path.read_bytes()
        wrong_identity = copy.deepcopy(before)
        wrong_identity["video"]["path"] = "/another/video.mp4"
        invalid = [wrong_identity, {**before, "labels": {}}, {**before, "complete": "yes"},
                   {**before, "position_seconds": float("nan")}, {**before, "notes": 12},
                   {**before, "decisions": {"event": "maybe"}},
                   {**before, "labels": [{"id": "label", "mini": "Red", "to_cell": "", "time_seconds": 2}]}]
        for snapshot in invalid:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                record.restore(snapshot)
        with patch.object(audit, "atomic_write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                record.restore({**before, "notes": "Cannot save"})
        self.assertEqual(record.review.data, before)
        self.assertEqual(record.review.path.read_bytes(), saved)

    def test_reset_decision_preserves_unrelated_manual_labels_and_progress(self):
        self.source()
        record = self.record()
        predicted = record.rows[0]["id"]
        record.decide(predicted, True, duration=20)
        missed = record.add_missed(8, "Blue", "D4", duration=20)
        record.save_notes("Keep these")
        record.set_position(10)
        record.mark_complete(True)
        before = copy.deepcopy(record.review.data)
        with patch.object(audit, "atomic_write_json", side_effect=OSError("no space")):
            with self.assertRaises(OSError):
                record.reset_decision(predicted)
        self.assertEqual(record.review.data, before)
        record.reset_decision(predicted)
        self.assertFalse(record.complete)
        self.assertEqual(record.rows[0]["status"], "Unreviewed")
        self.assertEqual([label["id"] for label in record.review.data["labels"]], [missed["id"]])
        self.assertEqual(record.notes, "Keep these")
        self.assertEqual(record.position_seconds, 10)
        with self.assertRaisesRegex(ValueError, "predicted"):
            record.reset_decision(missed["id"])

    def test_export_preserves_multiline_notes_and_frozen_provenance_with_safe_csv_cells(self):
        self.source(counterfactual=True)
        record = self.record()
        dangerous_note = '=HYPERLINK("https://invalid.example", "untrusted")\nSecond line, with comma'
        record.add_missed(4, "Blue", "C3", duration=20, notes=dangerous_note)
        record.save_notes("My video notes\nDo not treat replay predictions as physical truth.")
        frozen_source = copy.deepcopy(record.review.data["audit_source"])
        # The review freezes the original source; changing the external file
        # later must not rewrite what an exported prediction claims to use.
        (self.root / "inputs.json").write_text(json.dumps({"initialReseedCounterfactual": False}))
        reopened = self.record()
        base = reopened.export(self.root / "exports")
        output = json.loads(Path(str(base) + ".json").read_text())
        self.assertEqual(output["audit_source"], frozen_source)
        self.assertTrue(output["audit_source"]["inputs"]["initialReseedCounterfactual"])
        self.assertEqual(output["video_identity"], record.review.identity)
        self.assertEqual(output["notes"], record.notes)
        self.assertEqual(output["movements"][1]["notes"], dangerous_note)
        with Path(str(base) + ".csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["Row notes"], "")
        self.assertEqual(rows[1]["Row notes"], "'" + dangerous_note)
        text = Path(str(base) + ".txt").read_text()
        self.assertIn("Video notes:\n" + record.notes, text)
        self.assertIn('Note: =HYPERLINK("https://invalid.example", "untrusted")', text)
        self.assertIn("Second line, with comma", text)
        self.assertIn("background was refreshed", text)

    def test_audit_csv_escapes_each_formula_prefix_in_movement_notes(self):
        record = self.record()
        values = ["=1+1", "+1+1", "-1+1", "@SUM(1)", "\t=1+1", "\r=1+1", "ordinary note"]
        for index, value in enumerate(values):
            record.add_missed(index, "Red", "A1", duration=20, notes=value)
        base = record.export(self.root / "exports")
        with Path(str(base) + ".csv").open(newline="", encoding="utf-8") as stream:
            notes = [row["Row notes"] for row in csv.DictReader(stream)]
        self.assertEqual(notes, ["'" + value for value in values[:-1]] + values[-1:])


if __name__ == "__main__":
    unittest.main()
