"""Exercise review-controller persistence without a Tk interpreter or decoder."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import audit_catalog
from audit_catalog import AuditRecord
import audit_videos
from audit_videos import AuditWindow


class Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self):
        self.disabled = False
        self.options = {}

    def configure(self, **options):
        self.options.update(options)

    def state(self, flags):
        if "disabled" in flags:
            self.disabled = True
        if "!disabled" in flags:
            self.disabled = False


class Tree(Widget):
    def __init__(self):
        super().__init__()
        self.rows = {}
        self.selected = ()

    def get_children(self):
        return tuple(self.rows)

    def delete(self, *items):
        for item in items:
            self.rows.pop(item, None)
        self.selected = tuple(item for item in self.selected if item in self.rows)

    def insert(self, _parent, _position, *, iid, **values):
        self.rows[iid] = values

    def selection_set(self, iid):
        self.selected = (iid,)

    def selection(self):
        return self.selected

    def see(self, iid):
        if iid not in self.rows:
            raise KeyError(iid)


class SavedReader:
    fps, count, duration = 10, 201, 20.1

    def __init__(self, video, timeline=None):
        self.video, self.timeline = Path(video), timeline
        self.frame, self.closed = 0, False

    def seek(self, frame):
        self.frame = max(0, min(self.count - 1, int(frame)))

    def close(self):
        self.closed = True


def controller():
    """Only controller collaborators, never root = Tk() or a native image."""
    window = AuditWindow.__new__(AuditWindow)
    window.root = Mock()
    window.state = {}
    window.extra_folders, window.extra_videos = [], []
    window.paths, window.visible = [], []
    window.record = window.reader = None
    window.playing = window.closed = False
    window.photo = window._pixels = None
    window._row_map, window._undo = {}, []
    window._selecting = window._scrubbing = False
    window._segment_end = window._save_after = window._tick_id = None
    window._last_saved = 0
    window._image_item = 1
    window.canvas = Mock()
    for name in ("progress", "title", "source", "source_note", "notes", "view_hint",
                 "clock", "status", "move_hint", "zoom", "seek_value"):
        setattr(window, name, Variable())
    window.filter = Variable("To review")
    for name in ("play_button", "seek_scale", "correct_button", "wrong_button",
                 "undo_button", "done_button"):
        setattr(window, name, Widget())
    window.queue, window.tree = Tree(), Tree()
    window.render = Mock()
    window.save_state = Mock()
    window.refresh_queue = Mock()
    return window


class AuditWindowControllerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.folder = Path(directory.name).resolve()
        self.reviews = self.folder / "reviews"
        self.manifest = self.folder / "sources.json"
        self.manifest.write_text(json.dumps({"schemaVersion": 1, "sources": [], "unavailable": []}))
        self.addCleanup(patch.stopall)
        patch.object(audit_catalog, "SOURCES_FILE", self.manifest).start()
        patch.object(audit_videos.messagebox, "showerror").start()
        patch.object(audit_videos, "AuditRecord", side_effect=self.record).start()
        self.reader_factory = patch.object(audit_videos, "AuditVideo", side_effect=SavedReader).start()

    def video(self, name, *, event=True):
        path = self.folder / (name + ".mp4")
        path.write_bytes(b"Saved file bytes, deliberately not a real video")
        path.with_suffix(".tracking.json").write_text(json.dumps({"fps": 10,
            "gridCols": 20, "gridRows": 20, "trackingEvents":
                [{"frame": 75, "mini": "red10", "cell": "r2c3"}] if event else []}))
        return path

    def record(self, path):
        return AuditRecord(path, folder=self.reviews)

    def test_switch_saves_outgoing_work_and_resumes_incoming_without_jumping_to_move(self):
        first, second = self.video("first"), self.video("second")
        incoming = self.record(second)
        incoming.set_position(12.3)
        incoming.save_notes("Earlier notes")
        window = controller()
        window.open_video(first)
        outgoing, reader = window.record, window.reader
        reader.frame = 62
        window.notes.set("Unsaved typing")

        window.open_video(second)

        saved = self.record(first)
        self.assertEqual(saved.position_seconds, 6.2)
        self.assertEqual(saved.notes, "Unsaved typing")
        self.assertTrue(reader.closed)
        self.assertIsNot(window.record, outgoing)
        self.assertEqual(window.record.video, second)
        self.assertEqual(window.reader.frame, 123, "programmatic move selection must preserve resume position")
        self.assertEqual(window.notes.get(), "Earlier notes")
        self.assertFalse(window.playing)

    def test_resume_restores_selected_move_independently_of_saved_playhead(self):
        video = self.video("resume-selection")
        timeline_path = video.with_suffix(".tracking.json")
        timeline = json.loads(timeline_path.read_text())
        timeline["trackingEvents"].append({"frame": 150, "mini": "red10", "cell": "r4c5"})
        timeline_path.write_text(json.dumps(timeline))
        record = self.record(video)
        selected_id = record.rows[1]["id"]
        record.set_position(9.3)
        window = controller()
        window.state["selected_rows"] = {str(video): selected_id}

        window.open_video(video)

        self.assertEqual(window.selected_row()["id"], selected_id)
        self.assertEqual(window.reader.frame, 93, "restoring the highlighted move must not seek to it")
        self.assertFalse(window.playing)
        # An intentional user selection still jumps just before the move.
        window.select_move()
        self.assertEqual(window.reader.frame, 130)

    def test_failed_note_save_blocks_switch_without_losing_typed_text_or_reader(self):
        first, second = self.video("first"), self.video("second")
        window = controller()
        window.open_video(first)
        record, reader = window.record, window.reader
        window.notes.set("Do not lose this")
        with patch.object(record, "save_notes", side_effect=OSError("disk full")):
            window.open_video(second)
        self.assertIs(window.record, record)
        self.assertIs(window.reader, reader)
        self.assertFalse(reader.closed)
        self.assertEqual(window.notes.get(), "Do not lose this")
        self.assertIn("disk full", window.status.get())
        self.assertEqual(self.reader_factory.call_count, 1)

    def test_failed_progress_save_blocks_close_and_successful_retry_persists(self):
        video = self.video("first")
        window = controller()
        window.open_video(video)
        reader = window.reader
        reader.frame = 81
        window.notes.set("Kept at close")
        with patch.object(window.record, "set_position", side_effect=OSError("read only")):
            window.close()
        self.assertFalse(window.closed)
        self.assertFalse(reader.closed)
        window.root.destroy.assert_not_called()
        self.assertEqual(self.record(video).notes, "Kept at close")
        window.close()
        self.assertTrue(window.closed)
        self.assertTrue(reader.closed)
        window.root.destroy.assert_called_once()
        self.assertEqual(self.record(video).position_seconds, 8.1)

    def test_undo_restores_review_decision_but_keeps_later_playhead_and_notes(self):
        video = self.video("first")
        window = controller()
        window.open_video(video)
        row_id = window.record.rows[0]["id"]
        window.decide(True)
        self.assertEqual(window.record.rows[0]["status"], "Confirmed")
        window.record.set_position(13.2)
        window.record.save_notes("Written after checking")

        window.undo()

        reopened = self.record(video)
        self.assertEqual(reopened.rows[0]["status"], "Unreviewed")
        self.assertNotIn(row_id, reopened.review.data["decisions"])
        self.assertEqual(reopened.review.data["labels"], [])
        self.assertEqual(reopened.position_seconds, 13.2)
        self.assertEqual(reopened.notes, "Written after checking")

    def test_failed_undo_keeps_saved_decision_and_snapshot_for_retry(self):
        video = self.video("first")
        window = controller()
        window.open_video(video)
        window.decide(True)
        before = copy.deepcopy(window.record.review.data)
        saved = window.record.review.path.read_bytes()
        with patch.object(window.record, "restore", side_effect=OSError("disk full")):
            window.undo()
        self.assertEqual(window.record.review.data, before)
        self.assertEqual(window.record.review.path.read_bytes(), saved)
        self.assertEqual(len(window._undo), 1)
        self.assertIn("disk full", window.status.get())
        window.undo()
        self.assertEqual(window.record.rows[0]["status"], "Unreviewed")
        self.assertEqual(window._undo, [])

    def test_failed_mutation_keeps_canonical_review_and_undo_stack_unchanged(self):
        video = self.video("first")
        window = controller()
        window.open_video(video)
        window.record.set_position(1)
        before = copy.deepcopy(window.record.review.data)
        saved = window.record.review.path.read_bytes()
        with patch.object(audit_catalog, "atomic_write_json", side_effect=OSError("no space")):
            window.decide(True)
        self.assertEqual(window.record.review.data, before)
        self.assertEqual(window.record.review.path.read_bytes(), saved)
        self.assertEqual(window._undo, [])
        audit_videos.messagebox.showerror.assert_called_once()

    def test_completion_requires_all_predictions_checked_then_explicit_done(self):
        video = self.video("first")
        window = controller()
        window.filter.set("All recordings")
        window.open_video(video)
        window.finish_video()
        self.assertFalse(window.record.complete)
        self.assertIn("remaining", window.status.get())
        window.decide(False)
        self.assertFalse(window.record.complete)
        window.finish_video()
        self.assertTrue(self.record(video).complete)

    def test_checking_moves_advances_selection_without_interrupting_full_video_playback(self):
        video = self.video("continuous-review")
        timeline_path = video.with_suffix(".tracking.json")
        timeline = json.loads(timeline_path.read_text())
        timeline["trackingEvents"] = [
            {"frame": 75, "mini": "red10", "cell": "r2c3"},
            {"frame": 100, "mini": "red10", "cell": "r2c4"},
            {"frame": 150, "mini": "red10", "cell": "r2c5"}]
        timeline_path.write_text(json.dumps(timeline))
        window = controller()
        window.open_video(video)
        window.reader.frame = 42
        window.playing = True
        window.play_button.configure(text="Pause")
        for accepted, expected_cell in ((True, "E3"), (False, "F3")):
            with self.subTest(accepted=accepted):
                window.decide(accepted)
                self.assertTrue(window.playing, "checking a move must not pause the full-video review")
                self.assertEqual(window.reader.frame, 42, "do not skip gaps where missed movements may occur")
                self.assertEqual(window.play_button.options["text"], "Pause")
                self.assertEqual(window.selected_row()["to_cell"], expected_cell)
                self.assertEqual(window.selected_row()["status"], "Unreviewed")
        self.assertEqual([row["status"] for row in window.record.rows], ["Confirmed", "Wrong", "Unreviewed"])

    def test_completion_does_not_advance_when_notes_cannot_be_saved(self):
        video = self.video("empty", event=False)
        window = controller()
        window.filter.set("All recordings")
        window.open_video(video)
        window.notes.set("Cannot discard these notes")
        with patch.object(window.record, "save_notes", side_effect=OSError("disk full")):
            window.finish_video()
        self.assertFalse(window.record.complete)
        self.assertEqual(window.notes.get(), "Cannot discard these notes")
        self.assertEqual(window._undo, [])
        self.assertIn("disk full", window.status.get())

    def test_empty_transcript_needs_explicit_completion_and_empty_queue_clears_old_view(self):
        video = self.video("empty", event=False)
        window = controller()
        window.filter.set("All recordings")
        window.open_video(video)
        self.assertFalse(window.record.complete)
        self.assertEqual(window.tree.rows, {})
        self.assertIn("missed", window.move_hint.get())
        window.finish_video()
        self.assertTrue(self.record(video).complete)
        old_reader = window.reader
        window.open_video(None)
        self.assertIsNone(window.record)
        self.assertIsNone(window.reader)
        self.assertTrue(old_reader.closed)
        self.assertEqual(window._row_map, {})
        self.assertEqual(window.notes.get(), "")
        self.assertTrue(window.done_button.disabled)

    def test_unavailable_recording_does_not_open_decoder_or_mark_complete(self):
        video = self.video("crashed")
        self.manifest.write_text(json.dumps({"schemaVersion": 1, "sources": [], "unavailable": [{
            "video": video.name, "videoSha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "reason": "Missing MP4 index"}]}))
        window = controller()
        window.open_video(video)
        self.reader_factory.assert_not_called()
        window.finish_video()
        self.assertFalse(window.record.complete)
        self.assertTrue(window.done_button.disabled)
        self.assertIn("Missing MP4 index", window.view_hint.get())

    def test_invalid_transcript_clears_previous_rows_notes_and_reader(self):
        first, broken = self.video("first"), self.video("broken")
        broken.with_suffix(".tracking.json").write_text("not json")
        window = controller()
        window.open_video(first)
        window.notes.set("Earlier review")
        old_reader = window.reader
        window.open_video(broken)
        self.assertTrue(old_reader.closed)
        self.assertIsNone(window.reader)
        self.assertIsNone(window.record)
        self.assertEqual(window._row_map, {})
        self.assertEqual(window.notes.get(), "")
        self.assertTrue(window.done_button.disabled)
        self.assertIn("Could not open", window.view_hint.get())

    def test_manual_rows_cannot_be_reclassified_as_predictions(self):
        video = self.video("manual", event=False)
        record = self.record(video)
        record.add_missed(4, "Red", "B3", duration=20)
        window = controller()
        window.open_video(video)
        self.assertTrue(window.correct_button.disabled)
        self.assertTrue(window.wrong_button.disabled)
        window.decide(False)
        self.assertEqual(window.record.rows[0]["status"], "Confirmed")
        self.assertEqual(window._undo, [])

    def test_folder_queue_keeps_default_videos_deduplicates_and_filters_unavailable(self):
        older, newest, middle = (self.video(name) for name in
            ("sarween_rec_20260921_120000", "sarween_rec_20261103_194320", "sarween_rec_20261103_164259"))
        window = controller()
        window.extra_folders = [str(self.folder)]
        window.extra_videos = [str(older)]
        statuses = {older: "To review", newest: "Reviewed", middle: "Unavailable"}
        def discovered(folders=None):
            return [older] if folders is None else [older, newest, middle]
        with patch.object(audit_videos, "discover_videos", side_effect=discovered), \
                patch.object(audit_videos, "review_status", side_effect=statuses.get):
            AuditWindow.refresh_queue(window)
            self.assertEqual(window.paths, [newest, middle, older])
            self.assertEqual(window.visible, [older])
            self.assertEqual(window.progress.get(), "1 of 2 reviewed · 1 unavailable")
            window.filter.set("All recordings")
            AuditWindow.refresh_queue(window)
            self.assertEqual(window.visible, [newest, middle, older])
            window.filter.set("Reviewed")
            AuditWindow.refresh_queue(window)
            self.assertEqual(window.visible, [newest])

    def test_filter_and_last_video_are_persisted_with_extra_folders(self):
        video = self.video("first")
        window = controller()
        window.record = self.record(video)
        window.filter.set("Reviewed")
        window.extra_folders = [str(self.folder)]
        with patch.object(audit_videos, "atomic_write_json") as write:
            AuditWindow.save_state(window)
        stored = write.call_args.args[1]
        self.assertEqual(stored["filter"], "Reviewed")
        self.assertEqual(stored["last_video"], str(video))
        self.assertEqual(stored["folders"], [str(self.folder)])

    def test_grid_metadata_is_validated_before_label_api(self):
        video = self.video("first")
        window = controller()
        window.open_video(video)
        for cols, rows in ((True, 10), (1, False), (0, 10), (-1, 10), (10001, 10),
                           (1.5, 10), (None, 10), (float("inf"), 10), ({}, 10)):
            with self.subTest(cols=cols, rows=rows):
                window.record.review.timeline.update(gridCols=cols, gridRows=rows)
                self.assertIsNone(window.grid())
        window.record.review.timeline.update(gridCols=32, gridRows=18)
        self.assertEqual(window.grid(), (32, 18))

    def test_typing_and_system_shortcuts_do_not_trigger_review_actions(self):
        window = controller()
        action = Mock()
        for class_name in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox"):
            event = SimpleNamespace(widget=SimpleNamespace(winfo_class=lambda: class_name), state=0)
            self.assertIsNone(window.shortcut(event, action))
        tree = SimpleNamespace(winfo_class=lambda: "Treeview")
        for state in (4, 8, 12):
            self.assertIsNone(window.shortcut(SimpleNamespace(widget=tree, state=state), action))
        action.assert_not_called()
        self.assertEqual(window.shortcut(SimpleNamespace(widget=tree, state=0), action), "break")
        action.assert_called_once()


if __name__ == "__main__":
    unittest.main()
