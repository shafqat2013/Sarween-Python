import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import cv_core as core
import foundryoutput as fo
import tracking_regression as regression
import v3_tracking as tracking
from recording_state import checkpoint_path, restore_checkpoint, save_checkpoint
from tracking_engine import TrackingEngine

ROOT = Path(__file__).resolve().parents[1]


class RecordingStateTests(unittest.TestCase):
    def test_mid_session_roundtrip_with_background_and_profile_changes(self):
        cv2.setNumThreads(0)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("SCENE_ID", "SCENE_W", "SCENE_H", "GRID_PX", "GRID_TYPE", "SHIFT_X", "SHIFT_Y",
                         "SCENE_BACKGROUND", "_grid_cols", "_grid_rows", "_view_transform",
                         "_view_transform_revision", "_scene_visual_revision", "_scene_visual_reason",
                         "_selected_mini_id", "_tracking_output_paused"):
                stack.enter_context(patch.object(fo, name, copy.deepcopy(getattr(fo, name))))
            fo._selected_mini_id = None
            fo._tracking_output_paused = False
            fo.set_scene_params("synthetic-demo", 800, 480, 40, 0, 0, grid_type=1)
            fo.clear_view_transform()
            stack.enter_context(patch.object(fo, "queue_control"))
            profiles = json.loads((ROOT / "demo/five_minis.profiles.json").read_text())
            engine = TrackingEngine(tracking.detect_minis)
            session = core.CVCoreSession(source_path=str(ROOT / "demo/five_minis.mp4"),
                grid_w=20, grid_h=12, warp_w=800, warp_h=480, marker_mode="legacy", source_frames_undistorted=True)
            session.before_frame_callback = lambda frame: setattr(session, "replay_frame_clock", 1000 + frame * .11)
            session.recording_state_provider = lambda: {"engine": engine.snapshot(), "profiles": profiles}
            video = Path(directory) / "roundtrip.mp4"
            try:
                for bundle in session.frames():
                    now = 1000 + bundle.frame_idx * .11 + .02
                    if session.is_recording:
                        fo.record_tracking_input(now)
                    if bundle.frame_idx == 55:
                        engine.reset_mini("Blue")
                        fo.record_profile_change(profiles, "Blue")
                        continue
                    if bundle.frame_idx == 45:
                        gray = cv2.cvtColor(bundle.cam_bgr, cv2.COLOR_BGR2GRAY)
                        session.BG_cam = {"bgr": bundle.cam_bgr.copy(), "blur": cv2.GaussianBlur(gray, (21, 21), 0)}
                        session.BG_warp_f32 = core.warp_gray_blur(bundle.cam_bgr, session.H_saved, 800, 480).astype(np.float32)
                        session._checkpoint_kind = "background"
                    if session.is_recording:
                        fo.record_tracking_input(now, tracked=True)
                    step = engine.step(bundle, profiles, tracking.tracking_context(bundle, now=now))
                    for move in step.moves:
                        fo.record_tracking_event(move.mini, move.raw_to, source=move.source)
                    if bundle.frame_idx == 30:
                        self.assertTrue(session.start_recording(str(video), fps=10))
                    if bundle.frame_idx == 85:
                        break
            finally:
                session.close()
            timeline = json.loads(video.with_suffix(".tracking.json").read_text())
            self.assertEqual(timeline["framesRecorded"], 55)
            self.assertEqual([item["kind"] for item in timeline["checkpoints"]], ["initial", "background"])
            self.assertEqual(timeline["profileChanges"][0]["mini"], "Blue")
            self.assertTrue(video.with_suffix(".movements.csv").is_file())
            self.assertTrue(video.with_suffix(".profiles.json").is_file())
            expected = [(item["mini"], item["cell"], item["frame"]) for item in timeline["trackingEvents"]]
            diagnostics = {}
            output = regression._run_video(video, diagnostics=diagnostics)
            actual = [(item.mini, item.raw_to, item.frame_idx - 1) for item in output]
            self.assertEqual(actual, expected)
            self.assertGreater(len(actual), 2)
            self.assertEqual(diagnostics["processed_frames"], 55)
            self.assertEqual(diagnostics["clock_source"], "recorded_monotonic")
            entry = timeline["checkpoints"][0]
            bad = {**entry, "path": "../outside.npz"}
            with self.assertRaises(ValueError):
                checkpoint_path(video.with_suffix(".tracking.json"), bad)
            bad = {**entry, "sha256": "0" * 64}
            with self.assertRaises(ValueError):
                checkpoint_path(video.with_suffix(".tracking.json"), bad)

    def test_engine_snapshot_is_deep_and_restores_consensus(self):
        from collections import deque
        engine = TrackingEngine(lambda **kwargs: ({}, {}))
        engine.prev_state["Red"] = {"last_xy": (3, 4), "history": [1, 2]}
        engine.cell_hist["Red"] = deque(["r0c0"] * 3, maxlen=6)
        engine.last_emitted["Red"] = "r0c0"
        engine._last_time = 125.5
        state = engine.snapshot()
        state["prev_state"]["Red"]["history"].append(3)
        self.assertEqual(engine.prev_state["Red"]["history"], [1, 2])
        restored = TrackingEngine(lambda **kwargs: ({}, {}))
        restored.restore(json.loads(json.dumps(state)))
        self.assertEqual(restored.snapshot(), state)
        self.assertEqual(restored.cell_hist["Red"].maxlen, 6)
