import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from movement_transcript import export_rows, review_case, timeline_rows, validate_label
from replay_jobs import ReplayJob
from replay_review import ReviewSession

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo" / "five_minis.mp4"


class ReviewTests(unittest.TestCase):
    def test_validation_and_review_gate(self):
        row = validate_label({"mini": "Red", "time_seconds": "02:23", "to_cell": "B21"}, duration=200, grid=(48, 27))
        self.assertEqual(row["time_seconds"], 143)
        for bad in ({**row, "mini": ""}, {**row, "time_seconds": -1}, {**row, "to_cell": "ZZ90"}):
            with self.assertRaises(ValueError):
                validate_label(bad, duration=200, grid=(48, 27))
        with self.assertRaises(ValueError):
            review_case(DEMO, [row], duration=200, options={})
        case = review_case(DEMO, [row], duration=200, options={}, complete=True)["cases"][0]
        self.assertFalse(case["allow_unexpected"])
        self.assertEqual(case["label_source"], "user-reviewed-video")

    def test_annotations_survive_new_analysis_and_reopening(self):
        with tempfile.TemporaryDirectory() as directory:
            review = ReviewSession(DEMO, folder=directory)
            result = {"events": [{"mini": "Red", "time_seconds": 2, "from_cell": None, "to_cell": "C4"}], "diagnostics": {}}
            review.accept_analysis(result, {})
            prediction = review.rows("Detected")[0]
            review.decide(prediction, True, duration=24, grid=(20, 12))
            review.decide(prediction, True, duration=24, grid=(20, 12))
            self.assertEqual(len(review.rows("Reviewed")), 1)
            review.accept_analysis({"events": [], "diagnostics": {}}, {})
            self.assertEqual(len(review.rows("Reviewed")), 1)
            reopened = ReviewSession(DEMO, folder=directory)
            self.assertEqual(reopened.rows("Reviewed"), review.rows("Reviewed"))
            self.assertEqual(len(list(review.path.parent.glob("analysis-*.json"))), 2)

    def test_transcript_tracks_and_formula_safety(self):
        data = {"fps": 10, "trackingEvents": [{"frame": 23, "mini": "=1+1", "cell": "r0c0"}],
                "deliveryEvents": [{"frame": 25, "miniId": "Red", "cell": "r0c0", "type": "tokenMoveApplied"}]}
        rows = timeline_rows(data) + timeline_rows(data, "foundry")
        self.assertEqual(rows[0]["time_seconds"], 2.3)
        self.assertEqual(rows[1]["decision"], "confirmed")
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "session"
            export_rows(rows, base)
            self.assertIn("'=1+1", base.with_suffix(".csv").read_text())
            self.assertIn("not an exhaustive", base.with_suffix(".txt").read_text())


class WorkerLifecycleTests(unittest.TestCase):
    def command(self, _request, _response):
        return [sys.executable, "-c", "import time; time.sleep(30)"]

    def test_cancel_reaps_owned_process_and_cleans_temporary_files(self):
        job = ReplayJob(DEMO, command_factory=self.command)
        folder = Path(job.directory.name)
        job.cancel()
        self.assertIsNotNone(job.process.returncode)
        self.assertFalse(folder.exists())
        job.cancel()
        self.assertTrue(job.poll())

    def test_timeout_and_invalid_time_limit(self):
        job = ReplayJob(DEMO, timeout=.02, command_factory=self.command)
        time.sleep(.04)
        self.assertTrue(job.poll())
        self.assertIn("time limit", job.error)
        self.assertIsNotNone(job.process.returncode)
        for value in (0, -1, float("nan"), 901):
            with self.assertRaises(ValueError):
                ReplayJob(DEMO, timeout=value)

    def test_worker_enforces_own_deadline_without_parent_polling(self):
        result = subprocess.run([sys.executable, "-c",
            "from tracking_regression import start_worker_guard; import time; start_worker_guard(.05); time.sleep(5)"],
            cwd=ROOT, timeout=8)
        self.assertEqual(result.returncode, 124)


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class ReplayUITests(unittest.TestCase):
    def test_opening_saved_review_does_not_reanalyze_or_reset_decisions(self):
        import tkinter as tk
        from replay_window import ReplayWindow
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SARWEEN_DATA_DIR=directory):
            review = ReviewSession(DEMO)
            review.accept_analysis({"events": [], "diagnostics": {"map_states": []}}, {})
            review.data.update(complete=True, decisions={"existing-event": "confirmed"})
            review.save()
            before = review.path.read_bytes()
            root = tk.Tk()
            root.withdraw()
            window = ReplayWindow(root, DEMO)
            try:
                root.update_idletasks()
                self.assertFalse(window.auto_analyze)
                self.assertIsNone(window._demo_start_id)
                self.assertIsNone(window.job)
                self.assertTrue(window.complete.get())
                self.assertEqual(window.review.data["decisions"], {"existing-event": "confirmed"})
                self.assertEqual(review.path.read_bytes(), before)
            finally:
                window.close()
                root.destroy()

    def test_real_example_uses_video_detections_and_seeks_both_views(self):
        import tkinter as tk
        from replay_window import ReplayWindow
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SARWEEN_DATA_DIR=directory):
            root = tk.Tk()
            root.withdraw()
            window = ReplayWindow(root, ROOT / "demo" / "tabletop.mp4", demo=True)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    root.update()
                    if window.job and getattr(window.job, "delivered", False):
                        break
                    time.sleep(.02)
                self.assertTrue(window.demo_ready, window.status.get())
                self.assertFalse(window.review.timeline["synthetic"])
                self.assertEqual([r["to_cell"] for r in window.review.rows("Detected")],
                                 ["J7", "Y14", "AL20", "AK7", "K21"])
                self.assertTrue(window.video_canvas.winfo_viewable())
                self.assertTrue(window.map.winfo_viewable())
                for frame, cell in ((40, "J7"), (80, "Y14"), (115, "AL20"), (155, "AK7"),
                                    (185, "K21"), (40, "J7")):
                    window.seek(frame)
                    self.assertEqual(window.frame, frame)
                    self.assertEqual(window.map.state["positions"]["Red mini"]["cell"], cell)
                window.seek(0)
                self.assertEqual(window.map.state["positions"], {})
            finally:
                window.close()
                root.destroy()

    def test_own_video_without_metadata_opens_with_setup_and_no_invented_tracking(self):
        import shutil
        import tkinter as tk
        from replay_window import ReplayWindow
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SARWEEN_DATA_DIR=directory):
            video = Path(directory) / "my-table.mp4"
            shutil.copy2(ROOT / "demo" / "tabletop.mp4", video)
            root = tk.Tk()
            root.withdraw()
            window = ReplayWindow(root, video)
            try:
                root.update_idletasks()
                self.assertFalse(window.auto_analyze)
                self.assertTrue(window.tools_visible)
                self.assertIn("ArUco", window.status.get())
                self.assertIsNone(window.job)
                window.seek(80)
                self.assertEqual(window.map.state["positions"], {})
                self.assertTrue(window.video_canvas.winfo_viewable())
                self.assertTrue(window.map.winfo_viewable())
                window.toggle_play()
                self.assertTrue(window.playing)
            finally:
                window.close()
                root.destroy()

    def test_demo_analysis_playback_map_review_and_close(self):
        import tkinter as tk
        from replay_window import ReplayWindow
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, SARWEEN_DATA_DIR=directory):
            root = tk.Tk()
            root.withdraw()
            window = ReplayWindow(root, DEMO, demo=True)
            try:
                root.update_idletasks()
                self.assertFalse(window.playing)
                self.assertIn("disabled", window.play_button.state())
                self.assertTrue(window.video_canvas.winfo_viewable())
                self.assertTrue(window.map.winfo_viewable())
                self.assertFalse(window.tree.winfo_viewable())
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    root.update()
                    if window.job and getattr(window.job, "delivered", False):
                        break
                    time.sleep(.02)
                self.assertIsNotNone(window.job)
                self.assertIsNone(window.job.error, window.job.log)
                self.assertEqual(len(window.review.rows("Detected")), 11)
                self.assertTrue(window.playing)
                self.assertTrue(window.demo_ready)
                self.assertNotIn("disabled", window.play_button.state())
                window.seek(220)
                root.update()
                self.assertEqual(len(window.map.state["positions"]), 5)
                self.assertEqual(window.map.state["positions"]["Red"]["cell"], "E9")
                self.assertGreater(len(window.map.canvas.find_all()), 30)
                for width, height in ((1100, 640), (780, 460)):
                    window.root.geometry(f"{width}x{height}")
                    root.update()
                    self.assertGreater(window.video_canvas.winfo_height(), 50)
                    self.assertGreaterEqual(window.map.winfo_rootx(),
                                            window.video_canvas.winfo_rootx() + window.video_canvas.winfo_width())
                    self.assertTrue(window.video_canvas.winfo_viewable())
                    self.assertTrue(window.map.winfo_viewable())
                window.seek(30)
                self.assertEqual(window.map.state["positions"]["Red"]["cell"], "C4")
                window.seek(60)
                self.assertEqual(window.map.state["positions"]["Red"]["cell"], "C9")
                window.toggle_review_tools()
                root.update()
                self.assertTrue(window.tree.winfo_viewable())
                self.assertLess(window.tree.winfo_rooty() + window.tree.winfo_height(), window.root.winfo_rooty() + window.root.winfo_height())
                row = window.review.rows("Detected")[0]
                window.tree.selection_set(row["id"])
                window.decide(True)
                self.assertEqual(len(window.review.rows("Reviewed")), 1)
                window.toggle_play()
                root.update()
                window.seek(0)
                window.analyze()
                worker = window.job.process
                window.close()
                self.assertIsNotNone(worker.returncode)
                self.assertFalse(window.cap.isOpened())
            finally:
                window.close()
                root.destroy()
