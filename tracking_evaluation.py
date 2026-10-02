"""Movement scoring without a camera, Foundry connection, or OpenCV."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Optional


@dataclass
class MovementEvent:
    mini: str
    from_cell: Optional[str]
    to_cell: str
    frame_idx: int
    time_seconds: float
    raw_from: Optional[str]
    raw_to: str
    lab_dist: float
    score: float
    source: str = "detection"


@dataclass
class CaseResult:
    name: str
    video: str
    events: list[MovementEvent]
    failures: list[str]
    matched_expectations: int
    total_expectations: int
    skipped: Optional[str] = None
    label_source: str = "unverified"
    matches: list[dict] = field(default_factory=list)
    missing: list[dict] = field(default_factory=list)
    unexpected: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failures


def parse_time_seconds(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    parts = str(value).strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time value: {value!r}")
    seconds = 0.0
    for part in parts:
        number = float(part)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Time must be finite and nonnegative: {value!r}")
        seconds = seconds * 60 + number
    return seconds


def format_time(seconds: float) -> str:
    total_ms = round(max(0.0, seconds) * 1000)
    whole, millis = divmod(total_ms, 1000)
    minutes, sec = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    prefix = f"{hours:02d}:" if hours else ""
    return f"{prefix}{minute:02d}:{sec:02d}.{millis:03d}"


def normalize_cell(cell: Optional[str]) -> Optional[str]:
    if cell is None or not str(cell).strip():
        return None
    text = str(cell).strip()
    raw = re.fullmatch(r"r(\d+)c(\d+)", text, re.IGNORECASE)
    if raw:
        row, col = map(int, raw.groups())
        letters = ""
        col += 1
        while col:
            col, remainder = divmod(col - 1, 26)
            letters = chr(65 + remainder) + letters
        return f"{letters}{row + 1}"
    text = text.upper()
    if not re.fullmatch(r"[A-Z]+[1-9]\d*", text):
        raise ValueError(f"Invalid grid cell: {cell!r}")
    return text


def a1_to_row_col(cell: str) -> tuple[int, int]:
    text = normalize_cell(cell)
    if text is None:
        raise ValueError("Cell cannot be empty")
    letters, row = re.fullmatch(r"([A-Z]+)(\d+)", text).groups()
    col = 0
    for letter in letters:
        col = col * 26 + ord(letter) - 64
    return int(row) - 1, col - 1


def expectation_window(expectation: dict, tolerance: float) -> tuple[float, float]:
    if "between" in expectation:
        window = expectation["between"]
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            raise ValueError("between must contain a start and end time")
        start, end = map(parse_time_seconds, window)
        if end < start:
            raise ValueError("Movement window ends before it starts")
        return start, end
    if "at" not in expectation and "time" not in expectation:
        raise ValueError("Each expectation needs at or between")
    at = parse_time_seconds(expectation.get("at", expectation.get("time")))
    tolerance = float(expectation.get("tolerance_seconds", tolerance))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_seconds must be finite and nonnegative")
    return max(0, at - tolerance), at + tolerance


def validate_case(case: dict) -> None:
    if not case.get("video"):
        raise ValueError("Case needs a video path")
    expectations = case.get("expectations") or []
    if not expectations and not case.get("stationary", False):
        raise ValueError("No expectations; use stationary: true for a no-movement test")
    tolerance = float(case.get("tolerance_seconds", 2.0))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_seconds must be finite and nonnegative")
    for flag in ("stationary", "allow_unexpected", "frames_undistorted", "restore_recorded_profiles"):
        if flag in case and not isinstance(case[flag], bool):
            raise ValueError(f"{flag} must be true or false, not text")
    start = parse_time_seconds(case.get("ignore_before", 0))
    end = parse_time_seconds(case["ignore_after"]) if case.get("ignore_after") is not None else math.inf
    if start > end:
        raise ValueError("ignore_after precedes ignore_before")
    for expectation in expectations:
        if not isinstance(expectation.get("mini"), str) or not expectation["mini"].strip() or not normalize_cell(expectation.get("to")):
            raise ValueError("Each expectation needs a mini identity and destination cell")
        normalize_cell(expectation.get("from"))
        low, high = expectation_window(expectation, tolerance)
        if high < start or low > end:
            raise ValueError("Expected movement lies outside the evaluated time range")
        for key in ("settled_at", "confirmed_at"):
            if key in expectation:
                parse_time_seconds(expectation[key])
    for cell in (case.get("initial_positions") or {}).values():
        if normalize_cell(cell) is None:
            raise ValueError("Initial positions need nonempty cells")


def event_matches_expectation(event: MovementEvent, expectation: dict, tolerance: float) -> bool:
    if str(event.mini) != str(expectation.get("mini")):
        return False
    if normalize_cell(event.to_cell) != normalize_cell(expectation.get("to")):
        return False
    source = normalize_cell(expectation.get("from"))
    if source and normalize_cell(event.from_cell) != source:
        return False
    if event.source != expectation.get("source", "detection"):
        return False
    start, end = expectation_window(expectation, tolerance)
    return start <= event.time_seconds <= end


def _target_time(expectation: dict, tolerance: float) -> float:
    if "between" in expectation:
        low, high = expectation_window(expectation, tolerance)
        return (low + high) / 2
    return parse_time_seconds(expectation.get("at", expectation.get("time")))


def _ordered_matches(expectations: list, events: list, tolerance: float) -> list:
    # Sequence alignment maximizes one-to-one matches before minimizing timing
    # error. A greedy nearest match can consume the only match for a later move.
    rows, cols = len(expectations), len(events)
    scores = [[(0, 0.0)] * (cols + 1) for _ in range(rows + 1)]
    choices = {}
    for i in range(1, rows + 1):
        _, expected = expectations[i - 1]
        for j in range(1, cols + 1):
            _, actual = events[j - 1]
            options = [(scores[i - 1][j], "expected"), (scores[i][j - 1], "event")]
            if event_matches_expectation(actual, expected, tolerance):
                count, cost = scores[i - 1][j - 1]
                options.append(((count + 1, cost - abs(actual.time_seconds - _target_time(expected, tolerance))), "match"))
            scores[i][j], choices[i, j] = max(options, key=lambda item: item[0])
    matched = []
    i, j = rows, cols
    while i and j:
        choice = choices[i, j]
        if choice == "match":
            matched.append((expectations[i - 1][0], events[j - 1][0]))
            i -= 1
            j -= 1
        elif choice == "expected":
            i -= 1
        else:
            j -= 1
    return list(reversed(matched))


def evaluate_events(case: dict, events: list[MovementEvent], *, video: str = "") -> CaseResult:
    validate_case(case)
    expectations = case.get("expectations") or []
    tolerance = float(case.get("tolerance_seconds", 2.0))
    start = parse_time_seconds(case.get("ignore_before", 0))
    end = parse_time_seconds(case["ignore_after"]) if case.get("ignore_after") is not None else math.inf
    scoped = [(i, event) for i, event in enumerate(events) if start <= event.time_seconds <= end]
    scoped.sort(key=lambda item: (item[1].time_seconds, item[0]))
    pairs = []
    for mini in sorted({str(item["mini"]) for item in expectations}):
        pairs.extend(_ordered_matches(
            [(i, item) for i, item in enumerate(expectations) if str(item["mini"]) == mini],
            [(i, item) for i, item in scoped if item.mini == mini], tolerance,
        ))
    used_expected = {i for i, _ in pairs}
    used_events = {j for _, j in pairs}
    result = CaseResult(
        name=str(case.get("name") or case["video"]), video=video or str(case["video"]),
        events=events, failures=[], matched_expectations=len(pairs),
        total_expectations=len(expectations), label_source=case.get("label_source", "unverified"),
    )
    for i, j in sorted(pairs):
        expectation, event = expectations[i], events[j]
        match = {"expectation_index": i, "event_index": j, "mini": event.mini, "to": event.to_cell,
                 "time_seconds": event.time_seconds, "expectation": expectation}
        if "between" not in expectation:
            match["timing_error_seconds"] = event.time_seconds - _target_time(expectation, tolerance)
        # Confirmation is an upper bound on placement time, not measured latency.
        for key, metric in (("settled_at", "response_seconds"), ("confirmed_at", "confirmation_offset_seconds")):
            if key in expectation:
                match[metric] = event.time_seconds - parse_time_seconds(expectation[key])
        result.matches.append(match)
    for i, expectation in enumerate(expectations):
        if i not in used_expected:
            result.missing.append({"expectation_index": i, "expectation": expectation})
            result.failures.append(f"Missing expected movement: {expectation['mini']} ->{normalize_cell(expectation['to'])} in {expectation_window(expectation, tolerance)}")
    for j, event in scoped:
        if j in used_events:
            continue
        result.unexpected.append({"event_index": j, "mini": event.mini, "to": event.to_cell, "time_seconds": event.time_seconds})
        if not case.get("allow_unexpected", False):
            result.failures.append(f"Unexpected movement: {event.mini} {event.from_cell or '?'}->{event.to_cell} at {format_time(event.time_seconds)}")
    for mini in sorted({item.mini for _, item in scoped} | {str(item["mini"]) for item in expectations}):
        count = sum(item["mini"] == mini for item in result.matches)
        missing = sum(item["expectation"]["mini"] == mini for item in result.missing)
        extra = sum(item["mini"] == mini for item in result.unexpected)
        errors = [abs(item["timing_error_seconds"]) for item in result.matches if item["mini"] == mini and "timing_error_seconds" in item]
        delays = [item["response_seconds"] for item in result.matches if item["mini"] == mini and "response_seconds" in item]
        result.metrics[mini] = {"matched": count, "missed": missing, "extra": extra,
                                "mean_absolute_timing_error_seconds": mean(errors) if errors else None,
                                "mean_response_seconds": mean(delays) if delays else None}
    return result
