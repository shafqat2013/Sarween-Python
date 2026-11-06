"""Audit playback/zoom controller checks without constructing Tk or native UI."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from audit_videos import AuditWindow


class Variable:
    def __init__(self, value):
        self.value = value
        self.callback = None

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        if self.callback:
            self.callback(str(value))


class Reader:
    def __init__(self, frame=20):
        self.frame, self.fps, self.count, self.duration = frame, 10, 100, 10
        self.calls = []
        self.failure = None
        self.image = np.full((100, 200, 3), (30, 60, 90), np.uint8)
        self.cell_center = Mock(return_value=(150, 50))
        self.closed = False

    def seek(self, value):
        if self.failure:
            raise ValueError(self.failure)
        self.frame = max(0, min(self.count - 1, int(value)))
        self.calls.append(self.frame)
        return self.image

    def display(self, board=True, grid=True):
        return self.image.copy(), "Board view" if board else "Full camera view"

    def close(self):
        self.closed = True


class AuditPlaybackTests(unittest.TestCase):
    def window(self):
        window = AuditWindow.__new__(AuditWindow)
        window.root, window.canvas = Mock(), Mock()
        window.canvas.winfo_width.return_value = 400
        window.canvas.winfo_height.return_value = 300
        window.canvas.cget.return_value = "0 0 800 400"
        window.reader = Reader()
        window.record = None
        window.state = {}
        window._row_map = {}
        window.tree = Mock()
        window.tree.selection.return_value = ()
        window.closed = False
        window.playing = False
        window._scrubbing = False
        window._segment_end = None
        window._clock_origin, window._clock_started, window._last_saved = 20, 10, 10
        window._image_item = 1
        window._undo = []
        window._pixels = window.photo = None
        for name, value in (("speed", "1×"), ("zoom", "Fit"), ("board", True),
                            ("grid_labels", True), ("seek_value", 0), ("clock", ""),
                            ("status", ""), ("view_hint", ""), ("title", ""),
                            ("source", ""), ("source_note", ""), ("notes", "")):
            setattr(window, name, Variable(value))
        window.play_button, window.seek_scale, window.done_button = Mock(), Mock(), Mock()
        window.correct_button, window.wrong_button = Mock(), Mock()
        window.save_progress, window.flush_notes = Mock(return_value=True), Mock(return_value=True)
        window.save_state, window.refresh_queue, window.refresh_rows = Mock(), Mock(), Mock()
        window.render = Mock(return_value=True)
        window.selected_row = Mock(return_value=None)
        return window

    def test_wall_clock_playback_skips_to_due_frame_without_pausing(self):
        window = self.window()
        window.playing = True
        window.seek_value.callback = window.scrub
        with patch("audit_videos.time.monotonic", return_value=10.5):
            window.tick()
        self.assertEqual(window.reader.frame, 25)
        self.assertTrue(window.playing)
        self.assertEqual(window.clock.get(), "00:02.500 / 00:10.000")
        window.root.after.assert_called_once_with(40, window.tick)

    def test_speed_change_resets_origin_to_current_frame(self):
        window = self.window()
        window.reader.frame = 35
        window.speed.set("2×")
        with patch("audit_videos.time.monotonic", return_value=20):
            window._reset_clock()
        window.playing = True
        with patch("audit_videos.time.monotonic", return_value=21):
            window.tick()
        self.assertEqual(window.reader.frame, 55)

    def test_watch_segment_stops_at_its_end_and_saves_position(self):
        window = self.window()
        window.selected_row.return_value = {"time_seconds": 3, "to_cell": "B2"}
        with patch("audit_videos.time.monotonic", return_value=10):
            window.watch_move()
        self.assertEqual(window.reader.frame, 10)
        self.assertEqual(window._segment_end, 50)
        with patch("audit_videos.time.monotonic", return_value=20):
            window.tick()
        self.assertEqual(window.reader.frame, 50)
        self.assertFalse(window.playing)
        self.assertIsNone(window._segment_end)
        window.save_progress.assert_called()

    def test_manual_scrub_and_step_pause_and_clamp(self):
        window = self.window()
        window.playing = True
        window.scrub("44.2")
        self.assertEqual(window.reader.frame, 44)
        self.assertFalse(window.playing)
        window.step(-100)
        self.assertEqual(window.reader.frame, 0)
        window.step_frame(1000)
        self.assertEqual(window.reader.frame, 99)

    def test_play_at_end_rewinds_before_starting_clock(self):
        window = self.window()
        window.reader.frame = 99
        with patch("audit_videos.time.monotonic", return_value=30):
            window.toggle_play()
        self.assertEqual(window.reader.frame, 0)
        self.assertEqual(window._clock_origin, 0)
        self.assertTrue(window.playing)

    def test_decode_error_pauses_without_overwriting_old_frame(self):
        window = self.window()
        window.playing = True
        window.reader.failure = "Frame cannot be decoded"
        result = window.seek(45)
        self.assertFalse(result)
        self.assertEqual(window.reader.frame, 20)
        self.assertFalse(window.playing)
        self.assertFalse(window._scrubbing)
        self.assertIn("cannot be decoded", window.status.get())

    def test_failed_watch_seek_must_not_restart_old_segment(self):
        window = self.window()
        window.reader.failure = "Frame cannot be decoded"
        window.selected_row.return_value = {"time_seconds": 3, "to_cell": "B2"}
        window.watch_move()
        self.assertFalse(window.playing)
        self.assertEqual(window.reader.frame, 20)

    def test_failed_eof_rewind_must_not_start_playback(self):
        window = self.window()
        window.reader.frame = 99
        window.reader.failure = "Frame cannot be decoded"
        window.toggle_play()
        self.assertFalse(window.playing)

    def test_tick_decode_failure_keeps_timer_alive_but_pauses_playback(self):
        window = self.window()
        window.playing = True
        window.reader.failure = "Frame cannot be decoded"
        with patch("audit_videos.time.monotonic", return_value=10.5):
            window.tick()
        self.assertFalse(window.playing)
        self.assertEqual(window.reader.frame, 20)
        window.root.after.assert_called_once_with(40, window.tick)

    def test_render_scales_bgr_to_rgb_and_keeps_source_pixels(self):
        window = self.window()
        window.zoom.set("2×")
        original = window.reader.image.copy()
        with patch("PIL.ImageTk.PhotoImage", side_effect=lambda image, **_kw: SimpleNamespace(size=image.size)) as photo:
            AuditWindow.render(window)
        self.assertEqual(window.photo.size, (800, 400))
        self.assertEqual(photo.call_args.kwargs["master"], window.root)
        pixel = photo.call_args.args[0].getpixel((0, 0))
        self.assertEqual(pixel, (90, 60, 30))
        self.assertEqual(window._draw_scale, 4)
        self.assertEqual(window._draw_offset, (0, 0))
        window.canvas.configure.assert_called_with(scrollregion=(0, 0, 800, 400))
        np.testing.assert_array_equal(window.reader.image, original)

    def test_focus_uses_display_coordinates_and_canvas_zoom(self):
        window = self.window()
        window.zoom.set("2×")
        window._draw_offset, window._draw_scale = (0, 0), 4
        window.focus_cell("D2")
        window.canvas.xview_moveto.assert_called_once_with(.5)
        window.canvas.yview_moveto.assert_called_once_with(.125)

    def test_render_failure_clears_stale_picture_and_returns_failure(self):
        window = self.window()
        window.photo = object()
        window.reader.display = Mock(side_effect=ValueError("Display cannot be rendered"))
        with patch("PIL.ImageTk.PhotoImage") as photo:
            result = AuditWindow.render(window)
        self.assertFalse(result)
        self.assertIsNone(window.photo)
        self.assertIn("cannot be rendered", window.view_hint.get())
        window.canvas.itemconfigure.assert_called_once_with(window._image_item, image="")
        photo.assert_not_called()

    def test_decode_failure_can_be_retried_for_same_saved_video(self):
        window = self.window()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.mp4"
            path.write_bytes(b"saved-file fixture")
            stat = path.stat()
            record = SimpleNamespace(video=path.resolve(), title=path.name, source_label="Live",
                source_note="Recorded", notes="", unavailable_reason="", position_seconds=0,
                review=SimpleNamespace(timeline={}, identity={"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}))
            window.record, window.reader = record, None
            with patch("audit_videos.AuditRecord", return_value=record), \
                 patch("audit_videos.AuditVideo", return_value=Reader()) as video:
                window.open_video(path)
            video.assert_called_once_with(path, timeline={})
            self.assertIsNotNone(window.reader)

    def test_replaced_video_at_same_path_reloads_existing_reader(self):
        window = self.window()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.mp4"
            path.write_bytes(b"repaired saved-file fixture")
            stat = path.stat()
            current_record = SimpleNamespace(video=path.resolve(), title=path.name, source_label="Live",
                source_note="Recorded", notes="", unavailable_reason="", position_seconds=0,
                review=SimpleNamespace(timeline={}, identity={"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}))
            old_record = SimpleNamespace(video=path.resolve(),
                review=SimpleNamespace(identity={"bytes": 1, "mtime_ns": 1}))
            old_reader = window.reader
            window.record = old_record
            with patch("audit_videos.AuditRecord", return_value=current_record), \
                 patch("audit_videos.AuditVideo", return_value=Reader()) as video:
                window.open_video(path)
            self.assertTrue(old_reader.closed)
            video.assert_called_once_with(path, timeline={})

    def test_failed_resume_does_not_claim_successful_video_open(self):
        window = self.window()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.mp4"
            path.write_bytes(b"saved-file fixture")
            record = SimpleNamespace(video=path.resolve(), title=path.name, source_label="Live",
                source_note="Recorded", notes="", unavailable_reason="", position_seconds=3,
                review=SimpleNamespace(timeline={}))
            failed_reader = Reader(frame=0)
            failed_reader.failure = "Resume frame cannot be decoded"
            with patch("audit_videos.AuditRecord", return_value=record), \
                 patch("audit_videos.AuditVideo", return_value=failed_reader):
                window.open_video(path)
            self.assertIn("cannot be decoded", window.status.get())
            window.save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
