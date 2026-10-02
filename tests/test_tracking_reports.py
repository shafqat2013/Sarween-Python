import json
import tempfile
import unittest
from pathlib import Path

from tracking_evaluation import evaluate_events
from tracking_reports import compare_reports, write_report


class TrackingReportsTest(unittest.TestCase):
    def test_report_contains_failures_provenance_and_comparison(self):
        case = {"name": "red-control", "video": "local.mp4", "label_source": "legacy-unverified",
                "expectations": [{"mini": "red", "to": "A1", "at": 1}]}
        result = evaluate_events(case, [])
        result.diagnostics = {"inputs": {"profiles": {"sha256": "new"}}, "case": case,
                              "replay_limitations": ["No recorded frame clock"],
                              "pause_policy": "evaluate_guided_capture"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, markdown = write_report([result], root / "baseline", cases_path=root / "cases.json")
            data = json.loads(report.read_text())
            self.assertEqual(data["cases"][0]["label_source"], "legacy-unverified")
            self.assertIn("Missing expected movement", markdown.read_text())
            self.assertIn("Replay limitation: No recorded frame clock", markdown.read_text())
            self.assertIn("live prediction pauses are intentionally ignored", markdown.read_text())
            result.diagnostics["inputs"]["profiles"]["sha256"] = "changed"
            current, _ = write_report([result], root / "current", cases_path=root / "cases.json", baseline_path=report)
            comparison = json.loads(current.read_text())["comparison"][0]
            self.assertEqual(comparison["changed_inputs"], ["profiles"])
            self.assertEqual(comparison["matched_delta"], 0)

    def test_absent_cases_are_not_reported_as_improvements(self):
        self.assertEqual(compare_reports({"cases": []}, {"cases": [{"name": "old"}]}), [])


if __name__ == "__main__":
    unittest.main()
