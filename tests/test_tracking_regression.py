import json
import os
import tempfile
import unittest
from pathlib import Path


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
                                "expectations": [{"mini": "red", "to": "A1"}],
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


if __name__ == "__main__":
    unittest.main()
