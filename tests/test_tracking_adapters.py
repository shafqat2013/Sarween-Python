"""Exercise the real live UI loop and replay loop with identical frame inputs."""

import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import foundryoutput as fo
import tracking_regression as replay
import v3_tracking as live


def snapshot(frame, scene="one", offset=0, paused=False):
    return {"frame": frame,
            "sceneInfo": {"sceneId": scene, "width": 1000, "height": 800, "gridSize": 50, "gridType": 1},
            "trackingControls": {"selectedMini": "red", "paused": paused},
            "viewTransform": {"sceneId": scene, "viewportWidth": 1000, "viewportHeight": 800,
                              "registration": {"left": 0, "top": 0, "right": 1000, "bottom": 800},
                              "canvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": -offset, "ty": 0},
                              "gridOriginX": 0, "gridOriginY": 0, "gridSize": 50}}


class TrackingAdaptersTest(unittest.TestCase):
    def test_live_and_replay_match_through_selection_pan_scene_and_pause(self):
        timeline_data = {"schemaVersion": 1, "markerMode": "viewport", "events": [
            snapshot(0), snapshot(19, offset=50), snapshot(22, scene="two"),
            snapshot(27, scene="two", paused=True), snapshot(28, scene="two")
        ]}
        profiles = {"red": {}, "blue": {}}
        geometry = dict(warp_w=1001, warp_h=801, grid_w=20, grid_h=16, marker_mode="viewport")
        frames = [SimpleNamespace(**geometry, frame_idx=i, locked=i not in (22, 28),
                                  warp_bgr=np.zeros((1, 1, 3), np.uint8), final_mask_cam=None,
                                  H_use=None, last_missing_ids=[], last_marker_count=4) for i in range(1, 33)]
        with tempfile.TemporaryDirectory() as directory:
            timeline_path = Path(directory) / "session.tracking.json"
            timeline_path.write_text(json.dumps(timeline_data))
            results, states = [], []
            original_context = live.tracking_context
            for adapter in ("live", "replay"):
                with self.subTest(adapter=adapter), ExitStack() as stack:
                    # Foundry owns mutable globals; keep each adapter and other
                    # tests isolated without touching a server or saved settings.
                    names = ("SCENE_ID", "SCENE_W", "SCENE_H", "GRID_PX", "GRID_TYPE", "SHIFT_X", "SHIFT_Y",
                             "SCENE_BACKGROUND", "_grid_cols", "_grid_rows", "_view_transform",
                             "_view_transform_revision", "_scene_visual_revision", "_scene_visual_reason")
                    for name in names:
                        stack.enter_context(patch.object(fo, name, copy.deepcopy(getattr(fo, name))))
                    timeline = replay.FoundryTimelineReplay(timeline_path)
                    timeline.apply_through(0)
                    session = SimpleNamespace(**geometry, is_recording=False, index=0, close=Mock())
                    session.cap = Mock(get=lambda prop: session.index + 1)
                    def iterate(callback):
                        for frame in frames:
                            session.index = frame.frame_idx
                            callback(frame.frame_idx)
                            yield frame
                    def create_session(**kwargs):
                        callback = kwargs.get("before_frame_callback") or timeline.apply_through
                        session.frames = lambda: iterate(callback)
                        return session
                    observations = []
                    def detect(**kwargs):
                        state = kwargs["prev_state"]
                        observations.append((session.index, copy.deepcopy(state)))
                        detections = {}
                        for name, x in (("red", 25 if session.index <= 7 else 225), ("blue", 125)):
                            detections[name] = (None if name == "red" and 8 <= session.index <= 16 else
                                                SimpleNamespace(cx=x, cy=25, lab_dist=3, score=.9))
                        return detections, {name: dict(state.get(name, {})) for name in profiles}
                    stack.enter_context(patch.object(live.core, "CVCoreSession", side_effect=create_session))
                    stack.enter_context(patch.object(live, "detect_minis", side_effect=detect))
                    stack.enter_context(patch.object(fo, "set_grid_params"))
                    if adapter == "live":
                        stack.enter_context(patch.object(live, "initialize_user_data"))
                        panel = Mock()
                        panel.pop_actions.return_value = {}
                        panel.get_toggles.return_value = {}
                        panel.get_motion_thresh.return_value = 18
                        stack.enter_context(patch.object(live, "ControlPanel", return_value=panel))
                        stack.enter_context(patch("tk_camera_preview._get_root", return_value=None))
                        stack.enter_context(patch.object(live, "_pending_recorder", None))
                        stack.enter_context(patch.object(live.s, "load_last_selection", return_value={}))
                        stack.enter_context(patch.object(live, "load_profiles_with_curves", return_value=profiles))
                        stack.enter_context(patch.object(live.ml, "load_synced_library", return_value={}))
                        stack.enter_context(patch.object(live.ml, "library_rows", return_value=[]))
                        stack.enter_context(patch.object(live.cv2, "destroyAllWindows"))
                        stack.enter_context(patch.object(fo, "pop_capture_command", return_value=None))
                        stack.enter_context(patch.object(fo, "guided_capture_active", return_value=False))
                        stack.enter_context(patch.object(fo, "update_camera_lock"))
                        stack.enter_context(patch.object(fo, "get_selected_mini", side_effect=lambda: timeline.selected_mini))
                        stack.enter_context(patch.object(fo, "tracking_output_paused", side_effect=lambda: timeline.prediction_paused))
                        stack.enter_context(patch.object(live, "tracking_context", side_effect=lambda b, **kw:
                                                       original_context(b, **{**kw, "now": session.index / 10})))
                        recorded = stack.enter_context(patch.object(fo, "record_tracking_event"))
                        callback = Mock()
                        live.begin_session(callback, show_windows=False)
                        panel.root.destroy.assert_called_once()
                        events = [(args.args[0], args.args[1], args.kwargs["source"]) for args in recorded.call_args_list]
                        self.assertEqual(callback.call_count, len(events))
                        self.assertEqual([(call.args[0], call.args[1], call.kwargs["source"])
                                          for call in callback.call_args_list], events)
                    else:
                        stack.enter_context(patch.object(replay, "_video_metadata", return_value=(10, 33)))
                        stack.enter_context(patch.object(replay, "load_profiles_with_curves", return_value=profiles))
                        diagnostics = {}
                        output = replay._run_video(Path(directory) / "session.mp4", timeline_path=timeline_path,
                                                   diagnostics=diagnostics)
                        events = [(event.mini, event.raw_to, event.source) for event in output]
                        self.assertEqual(diagnostics["controls_source"], "timeline")
                        self.assertEqual(diagnostics["paused_frames"], 1)
                    session.close.assert_called_once()
                    results.append(events)
                    states.append(observations)
            self.assertEqual(results[0], results[1])
            self.assertEqual(states[0], states[1])
            self.assertGreaterEqual(len(results[0]), 6)
            self.assertIn(("blue", "r0c3", "viewportTransform"), results[0])
            at_16 = dict(states[0])[16]
            self.assertIsNone(at_16["red"]["last_xy"])
            self.assertIsNotNone(at_16["blue"]["last_xy"])
            self.assertEqual(dict(states[0])[23], {})
            self.assertEqual(dict(states[0])[29], {})


if __name__ == "__main__":
    unittest.main()
