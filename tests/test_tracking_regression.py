import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TrackingRegressionTest(unittest.TestCase):
    def _load_runner(self):
        try:
            import tracking_regression
            return tracking_regression
        except ModuleNotFoundError as exc:
            if exc.name == "cv2":
                self.skipTest("OpenCV/cv2 is not installed in this Python environment")
            raise

    def test_video_expectations(self):
        if os.environ.get("SARWEEN_RUN_VIDEO_TESTS") != "1" and not os.environ.get("SARWEEN_TRACKING_CASES"):
            self.skipTest("Video replay is opt-in; run scripts/run_tracking_regression.sh")
        tracking_regression = self._load_runner()
        cases_path = Path(
            os.environ.get(
                "SARWEEN_TRACKING_CASES",
                tracking_regression.DEFAULT_CASES_PATH,
            )
        ).expanduser()
        if not cases_path.exists():
            self.skipTest(f"No tracking regression case file found at {cases_path}")

        cases = tracking_regression.load_cases(cases_path)
        if not cases:
            self.skipTest(f"No tracking regression cases defined in {cases_path}")

        failures = []
        for result in tracking_regression.check_cases(cases_path):
            failures.extend(f"{result.name}: {failure}" for failure in result.failures)

        if failures:
            self.fail("\n".join(failures))

    def test_missing_local_video_is_skipped(self):
        tracking_regression = self._load_runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            cases_path = Path(temp_dir) / "cases.json"
            cases_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "name": "portable-local-footage",
                                "video": "not-checked-in.mp4",
                                "expectations": [{"mini": "red", "to": "A1", "at": 0}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            results = tracking_regression.check_cases(cases_path)

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].ok)
        self.assertIn("Missing local video", results[0].skipped or "")

    def test_timeout_becomes_failure_and_next_case_runs(self):
        runner = self._load_runner()
        case = {"name": "slow", "video": "local.mp4", "stationary": True}
        success = runner.CaseResult("next", "local.mp4", [], [], 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps({"cases": [case, {**case, "name": "next"}]}))
            with patch.object(runner, "check_case", side_effect=[RuntimeError("worker stopped"), success]):
                results = runner.check_cases(path)
        self.assertFalse(results[0].ok)
        self.assertTrue(results[1].ok)

    def test_worker_timeout_is_reported(self):
        import subprocess
        runner = self._load_runner()
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.touch()
            with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 1)):
                with self.assertRaisesRegex(RuntimeError, "worker stopped"):
                    runner.run_video(video, timeout_seconds=1)

    def test_real_timeout_reaps_worker(self):
        runner = self._load_runner()
        original_popen = runner.subprocess.Popen
        children = []
        def start(*args, **kwargs):
            child = original_popen(*args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.touch()
            with patch.object(runner.subprocess, "Popen", side_effect=start):
                with self.assertRaisesRegex(RuntimeError, "worker stopped"):
                    runner.run_video(video, timeout_seconds=0.001)
        self.assertEqual(len(children), 1)
        self.assertIsNotNone(children[0].returncode)

    def test_missing_mini_profile_fails_before_video_processing(self):
        runner = self._load_runner()
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.touch()
            with patch.object(runner, "run_video") as replay:
                with self.assertRaisesRegex(ValueError, "Profiles missing for minis: blue"):
                    runner.check_case({"video": str(video), "profiles": str(runner.ROOT / "tests/fixtures/red10_baseline.profiles.json"),
                                       "expectations": [{"mini": "blue", "to": "A1", "at": 1}]},
                                      cases_base_dir=Path(directory))
                replay.assert_not_called()

    def test_all_missing_videos_and_empty_suite_return_nonzero(self):
        runner = self._load_runner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            for cases in ([], [{"name": "missing", "video": "missing.mp4", "stationary": True}]):
                path.write_text(json.dumps({"cases": cases}))
                code = runner.main(["check", str(path), "--report-dir", str(Path(directory) / "report")])
                self.assertEqual(code, 2)

    def test_stationary_case_without_marker_lock_cannot_pass(self):
        runner = self._load_runner()
        def replay(*args, diagnostics, **kwargs):
            diagnostics.update(processed_frames=10, locked_frames=0)
            return []
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "sample.mp4"
            video.touch()
            with patch.object(runner, "run_video", side_effect=replay):
                result = runner.check_case({"video": str(video), "stationary": True,
                                           "profiles": str(runner.ROOT / "tests/fixtures/red10_baseline.profiles.json")},
                                          cases_base_dir=Path(directory))
        self.assertFalse(result.ok)
        self.assertIn("No frames had marker lock", result.failures[0])

    def test_real_worker_reaches_eof_and_preserves_parent_foundry_state(self):
        runner = self._load_runner()
        import cv2
        import numpy as np
        import foundryoutput as foundry
        old_scene = foundry.SCENE_ID
        try:
            foundry.SCENE_ID = "parent-state-must-survive"
            with tempfile.TemporaryDirectory() as directory:
                video = Path(directory) / "blank.avi"
                writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 64))
                self.assertTrue(writer.isOpened())
                for _ in range(4):
                    writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
                writer.release()
                diagnostics = {}
                events = runner.run_video(video, timeout_seconds=20, frames_undistorted=True, diagnostics=diagnostics,
                                          profiles_path=runner.ROOT / "tests/fixtures/red10_baseline.profiles.json")
            self.assertEqual(events, [])
            self.assertEqual(diagnostics["processed_frames"], 3)
            self.assertEqual(diagnostics["threads"], 1)
            self.assertEqual(foundry.SCENE_ID, "parent-state-must-survive")
        finally:
            foundry.SCENE_ID = old_scene

    def test_premature_decoder_end_fails_and_closes_session(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        runner = self._load_runner()
        geometry = dict(grid_w=23, grid_h=16, warp_w=1280, warp_h=720, marker_mode="legacy")
        bundles = [SimpleNamespace(frame_idx=i, locked=False, warp_bgr=None, last_missing_ids=[0],
                                   **geometry) for i in (1, 2)]
        session = SimpleNamespace(grid_w=23, grid_h=16, warp_w=1280, warp_h=720,
                                  marker_mode="legacy", frames=lambda: iter(bundles), close=Mock(),
                                  cap=Mock(get=Mock(side_effect=[2, 3])))
        with patch.object(runner, "_video_metadata", return_value=(10, 5)), \
             patch.object(runner.core, "CVCoreSession", return_value=session):
            with self.assertRaisesRegex(RuntimeError, "Video decoding ended early"):
                runner._run_video(Path("missing-test-video.avi"),
                                  profiles_path=runner.ROOT / "tests/fixtures/red10_baseline.profiles.json")
        session.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
