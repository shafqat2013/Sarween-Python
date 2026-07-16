import os
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


if __name__ == "__main__":
    unittest.main()
