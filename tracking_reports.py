"""Portable replay reports and comparisons between tracking experiments."""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from tracking_evaluation import format_time


def _status(result):
    if result.get("skipped"):
        return "SKIP"
    return "FAIL" if result["failures"] else "PASS"


def compare_reports(current, baseline):
    previous = {item["name"]: item for item in baseline["cases"]}
    comparisons = []
    for item in current["cases"]:
        old = previous.get(item["name"])
        if old is None:
            continue
        old_diag, new_diag = old.get("diagnostics", {}), item.get("diagnostics", {})
        changed = []
        for key in sorted(set(old_diag.get("inputs", {})) | set(new_diag.get("inputs", {}))):
            a = old_diag.get("inputs", {}).get(key, {}).get("sha256")
            b = new_diag.get("inputs", {}).get(key, {}).get("sha256")
            if a != b:
                changed.append(key)
        if old_diag.get("case") != new_diag.get("case"):
            changed.append("case definition")
        old_code, new_code = old_diag.get("code_sha256", {}), new_diag.get("code_sha256", {})
        changed_code = [key for key in sorted(set(old_code) | set(new_code))
                        if old_code.get(key) != new_code.get(key)] if old_code and new_code else None
        comparisons.append({
            "name": item["name"], "previous_status": _status(old), "status": _status(item),
            "matched_delta": item["matched_expectations"] - old["matched_expectations"],
            "extra_delta": len(item.get("unexpected", [])) - len(old.get("unexpected", [])),
            "changed_inputs": changed,
            "changed_code": changed_code,
        })
    return comparisons


def _text(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_report(results, directory: Path, *, cases_path: Path, baseline_path=None):
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except (OSError, subprocess.SubprocessError):
            return "unknown"
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(cases_path), "git_revision": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"), "cases": [asdict(result) for result in results],
    }
    if baseline_path:
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        report["comparison"] = compare_reports(report, baseline)
    lines = ["# Tracking Replay Report", "", f"Generated: {report['created_at']}", "",
             "These results measure agreement with the supplied labels. Unverified labels are historical baselines, not proof of tracking accuracy.", "",
             "| Case | Result | Labels | Matched | Missed | Extra | Lock | Elapsed |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for item in report["cases"]:
        diag = item["diagnostics"]
        frames = diag.get("processed_frames", 0)
        lock = f"{100 * diag.get('locked_frames', 0) / frames:.1f}%" if frames else "n/a"
        lines.append(f"| {_text(item['name'])} | {_status(item)} | {_text(item['label_source'])} | {item['matched_expectations']}/{item['total_expectations']} | {len(item['missing'])} | {len(item['unexpected'])} | {lock} | {diag.get('elapsed_seconds', 0):.1f}s |")
    if report.get("comparison"):
        lines.extend(["", "## Changes From Baseline", "", "| Case | Matched Change | Extra Change | Input Changes | Code Changes |", "| --- | ---: | ---: | --- | --- |"])
        for item in report["comparison"]:
            code = ", ".join(item["changed_code"]) or "none" if item["changed_code"] is not None else "not recorded in baseline"
            lines.append(f"| {_text(item['name'])} | {item['matched_delta']:+d} | {item['extra_delta']:+d} | {_text(', '.join(item['changed_inputs']) or 'none')} | {_text(code)} |")
    for item in report["cases"]:
        lines.extend(["", f"## {_text(item['name'])}", ""])
        if item["skipped"]:
            lines.append(item["skipped"])
            continue
        lines.extend([f"Video: {item['video']}", ""])
        for limitation in item["diagnostics"].get("replay_limitations", []):
            lines.append(f"- Replay limitation: {_text(limitation)}")
        if item["diagnostics"].get("pause_policy") == "evaluate_guided_capture":
            lines.append("- Guided-capture evaluation: live prediction pauses are intentionally ignored.")
        if item["failures"]:
            lines.extend(f"- {_text(failure)}" for failure in item["failures"])
            lines.append("")
        lines.extend(["| Time | Mini | From | To | Outcome | Timing Error |", "| --- | --- | --- | --- | --- | --- |"])
        matches = {match["event_index"]: match for match in item["matches"]}
        extras = {extra["event_index"] for extra in item["unexpected"]}
        for i, event in enumerate(item["events"]):
            match = matches.get(i)
            outcome = "matched" if match else "extra" if i in extras else "outside scoring window"
            error = f"{match['timing_error_seconds']:+.3f}s" if match and "timing_error_seconds" in match else "n/a"
            lines.append(f"| {format_time(event['time_seconds'])} | {_text(event['mini'])} | {_text(event['from_cell'] or '?')} | {_text(event['to_cell'])} | {outcome} | {error} |")
        lines.extend(["", "Timing error is relative to the supplied timestamp. Actual response time is available in JSON only when a manually verified `settled_at` timestamp is supplied.", ""])
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
