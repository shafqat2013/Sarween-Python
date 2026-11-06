"""Score completed offline observations against separately loaded review labels.

This module never detects minis, initializes tracks, or feeds reviewed moves
back into an observer. The caller loads a review only after processing footage.
"""

from __future__ import annotations

import itertools
import math
import re
import statistics


_CELL = re.compile(r"^[A-Z]+[1-9][0-9]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _cell(value):
    if not isinstance(value, str):
        return None
    value = value.strip().upper()
    return value if _CELL.fullmatch(value) else None


def _cells(value):
    if not isinstance(value, (list, tuple)):
        return set()
    return {cell for item in value if (cell := _cell(item)) is not None}


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _source_sha(report):
    metadata = report.get("metadata") or {}
    sources = [report.get("source"), metadata.get("source"),
               report.get("video_sha256"), metadata.get("video_sha256")]
    for source in sources:
        if isinstance(source, dict):
            source = source.get("video_sha256") or source.get("sha256")
        if isinstance(source, str) and _SHA256.fullmatch(source.lower()):
            return source.lower()
    return None


def _reference_time(expected):
    for name in ("visible_stationary_capture_seconds",
                 "provisional_visible_stationary_capture_seconds"):
        value = _number(expected.get(name))
        if value is not None:
            return value
    return None


def _timing(event, expected):
    reference = _reference_time(expected)
    confirmed = _number(event.get("confirmed_time"))
    if confirmed is None:
        confirmed = _number(event.get("capture_seconds"))
    verified = expected.get("timing_reference_status") == "assistant-frame-verified"
    response = confirmed - reference if confirmed is not None and reference is not None else None
    result = {
        "reference_status": expected.get("timing_reference_status", "missing"),
        "reference_capture_seconds": reference,
        "confirmed_capture_seconds": confirmed,
        "response_seconds": response,
        "status": "unverified-reference" if not verified else "missing-observer-time",
    }
    interval = expected.get("visible_stationary_capture_seconds_interval")
    if isinstance(interval, list) and len(interval) == 2:
        low, high = (_number(value) for value in interval)
        if low is not None and high is not None and low <= high and confirmed is not None:
            result["response_seconds_interval"] = [confirmed - high, confirmed - low]
    if verified and response is not None:
        result["status"] = "confirmed-before-reference" if response < -1e-6 else "measured"
    available = _number(event.get("simulated_available_capture_seconds"))
    adjusted = available - reference if available is not None and reference is not None else None
    result["simulated_available_capture_seconds"] = available
    result["processing_adjusted_response_seconds"] = adjusted
    if result["status"] != "measured":
        result["performance_status"] = result["status"]
    elif available is None:
        result["performance_status"] = "unverified-inference-availability"
    elif available < confirmed - 1e-6:
        result["performance_status"] = "invalid-inference-availability"
    else:
        result["performance_status"] = "measured"
    if "response_seconds_interval" in result and available is not None and reference is not None:
        low, high = expected["visible_stationary_capture_seconds_interval"]
        result["processing_adjusted_response_seconds_interval"] = [available - high, available - low]
    first_visible = _number(event.get("first_visible_time"))
    if first_visible is not None and confirmed is not None:
        result["observer_settle_seconds"] = confirmed - first_visible
    return result


def evaluate_observer(report: dict, review: dict) -> dict:
    """Return strict ordered movement, uncertainty and timing acceptance checks.

    Boundary labels require explicit abstention over exactly the reviewed pair.
    A prediction of either neighboring cell is not an exact-cell success.
    Latency is measured against independently reviewed source frames rather than
    the observer's own first-visible estimate or the MP4 playback frame rate.
    """
    if not isinstance(report, dict) or not isinstance(review, dict):
        raise ValueError("Observer report and review must be objects")
    events = report.get("events", [])
    expected_moves = review.get("expected_moves", [])
    if not isinstance(events, list) or not isinstance(expected_moves, list):
        raise ValueError("Events and expected moves must be lists")
    if any(not isinstance(item, dict) for item in events + expected_moves):
        raise ValueError("Each event and expected move must be an object")
    policy = review.get("scoring_policy") or {}
    required_count = policy.get("ordered_movement_identity_count", len(expected_moves))
    reference_complete = bool(expected_moves) and len(expected_moves) == required_count
    expected_frames = review.get("expected_frames", report.get("expected_frames"))
    recording_complete = report.get("complete") is True
    if expected_frames is not None:
        expected_count = _number(expected_frames)
        consumed = _number(report.get("frames_consumed"))
        recording_complete = (recording_complete and expected_count is not None
                              and expected_count > 0 and expected_count.is_integer()
                              and consumed == expected_count)
    rows = []
    for index, (expected, event) in enumerate(itertools.zip_longest(expected_moves, events), 1):
        row = {"sequence": index, "expected": expected, "observed": event, "discrepancies": []}
        errors = row["discrepancies"]
        if expected is None:
            errors.append("Unexpected extra movement")
            rows.append(row)
            continue
        if event is None:
            errors.append("Expected movement was missed")
            rows.append(row)
            continue
        row["sequence_matches"] = event.get("sequence", index) == expected.get("sequence", index)
        row["identity_matches"] = event.get("mini") == expected.get("mini")
        row["origin_matches"] = (_cell(event.get("from_cell")) is not None
                                 and _cell(event.get("from_cell")) == _cell(expected.get("from_cell")))
        if not row["sequence_matches"]:
            errors.append("Movement sequence number differs")
        if not row["identity_matches"]:
            errors.append("Wrong moving mini at this point in the sequence")
        if not row["origin_matches"]:
            errors.append("Starting cell differs or is unavailable")
        expected_pair = _cells(expected.get("destination_boundary_cells"))
        if expected_pair:
            row["destination_kind"] = "boundary"
            row["destination_matches"] = (event.get("to_cell") is None
                                          and event.get("boundary_uncertain") is True
                                          and _cells(event.get("destination_cells")) == expected_pair)
            if len(expected_pair) != 2:
                reference_complete = False
                errors.append("Reference boundary does not describe exactly two cells")
            if not row["destination_matches"]:
                errors.append("Boundary needs abstention over exactly the reviewed adjacent cells")
        else:
            expected_cell = _cell(expected.get("to_cell"))
            row["destination_kind"] = "exact"
            observed_cells = _cells(event.get("destination_cells"))
            row["destination_matches"] = (expected_cell is not None
                                          and _cell(event.get("to_cell")) == expected_cell
                                          and event.get("boundary_uncertain") is not True
                                          and (not observed_cells or observed_cells == {expected_cell}))
            if expected_cell is None:
                reference_complete = False
                errors.append("Reference lacks a destination label")
            if not row["destination_matches"]:
                errors.append("Unambiguous destination differs or was not confirmed")
        row["timing"] = _timing(event, expected)
        window = expected.get("observed_movement_frame_window")
        event_frame = _number(event.get("frame"))
        row["after_movement_started"] = not (isinstance(window, list) and window
                                                and event_frame is not None and event_frame < window[0])
        if not row["after_movement_started"]:
            errors.append("Event predates the reviewed physical movement")
        rows.append(row)

    paired = [row for row in rows if row.get("expected") is not None and row.get("observed") is not None]
    exact = [row for row in paired if row.get("destination_kind") == "exact"]
    boundary = [row for row in paired if row.get("destination_kind") == "boundary"]
    exact_required = sum(not _cells(item.get("destination_boundary_cells")) for item in expected_moves)
    boundary_required = len(expected_moves) - exact_required
    final_positions = report.get("final_positions") or {}
    stationary_checks = {}
    for mini, expected_cell in (review.get("stationary_minis") or {}).items():
        position = final_positions.get(mini) or {}
        stationary_checks[mini] = (not any(event.get("mini") == mini for event in events)
                                   and _cell(position.get("cell")) == _cell(expected_cell)
                                   and position.get("boundary_uncertain") is not True)
    expected_sha, observed_sha = _source_sha(review), _source_sha(report)
    source_status = "unverified" if expected_sha is None or observed_sha is None else (
        "matched" if expected_sha == observed_sha else "mismatch")
    criteria = {
        "reference_complete": reference_complete,
        "recording_complete": recording_complete,
        "move_count_exact": len(events) == len(expected_moves),
        "no_extra_moves": len(events) <= len(expected_moves),
        "ordered_identities": len(paired) == len(expected_moves) and all(
            row.get("identity_matches") and row.get("sequence_matches") for row in paired),
        "starting_cells": len(paired) == len(expected_moves) and all(row.get("origin_matches") for row in paired),
        "unambiguous_destinations": len(exact) == exact_required and all(row["destination_matches"] for row in exact),
        "boundary_abstentions": len(boundary) == boundary_required and all(row["destination_matches"] for row in boundary),
        "stationary_minis_preserved": all(stationary_checks.values()),
        "events_after_movement_started": all(row.get("after_movement_started", True) for row in paired),
    }
    movement_acceptance = all(criteria.values()) and source_status != "mismatch"
    targets = review.get("timing_targets") or {}
    median_target = float(targets.get("median_seconds", 0.5))
    maximum_target = float(targets.get("maximum_seconds", 1.0))
    timing_rows = [row["timing"] for row in exact]
    capture_timing_verified = (len(timing_rows) == exact_required and bool(timing_rows)
                              and all(row["status"] == "measured" for row in timing_rows))
    timing_verified = (capture_timing_verified
                       and all(row["performance_status"] == "measured" for row in timing_rows))
    responses = [row["response_seconds"] for row in timing_rows if row["status"] == "measured"]
    median_response = statistics.median(responses) if responses else None
    maximum_response = max(responses) if responses else None
    adjusted_responses = [row["processing_adjusted_response_seconds"] for row in timing_rows
                          if row["performance_status"] == "measured"]
    adjusted_median = statistics.median(adjusted_responses) if adjusted_responses else None
    adjusted_maximum = max(adjusted_responses) if adjusted_responses else None
    targets_met = (adjusted_median <= median_target and adjusted_maximum <= maximum_target
                   if timing_verified else None)
    timing_status = "measured" if timing_verified else (
        "early-confirmation-needs-review" if any(row["status"] == "confirmed-before-reference" for row in timing_rows)
        else "unverified")
    overall = movement_acceptance and source_status == "matched" and targets_met is True
    return {
        "scope": "development-recording-only",
        "reference_status": review.get("reference_status", "unspecified"),
        "source_validation": {"status": source_status, "expected_sha256": expected_sha, "observed_sha256": observed_sha},
        "criteria": criteria,
        "counts": {"expected_moves": len(expected_moves), "observed_moves": len(events),
                   "expected_frames": expected_frames, "frames_consumed": report.get("frames_consumed"),
                   "exact_destination_passes": sum(row.get("destination_matches", False) for row in exact),
                   "exact_destination_required": exact_required,
                   "boundary_abstentions": sum(row.get("destination_matches", False) for row in boundary),
                   "boundary_required": boundary_required},
        "stationary_checks": stationary_checks,
        "per_move": rows,
        "timing": {"status": timing_status,
                   "basis": "reviewed capture timestamps plus simulated inference availability",
                   "eligible_sequences": [row["sequence"] for row in exact],
                   "median_response_seconds": median_response, "maximum_response_seconds": maximum_response,
                   "capture_clock_status": "measured" if capture_timing_verified else "unverified",
                   "processing_adjusted_median_response_seconds": adjusted_median,
                   "processing_adjusted_maximum_response_seconds": adjusted_maximum,
                   "targets": {"median_seconds": median_target, "maximum_seconds": maximum_target},
                   "targets_met": targets_met,
                   "note": "median_response_seconds/maximum_response_seconds describe source-clock delay only. "
                           "Target acceptance uses processing-adjusted values including simulated inference backlog. "
                           "This excludes decode, rendering, encoding and live device/display I/O; it is not live gameplay latency. "
                           "Boundary moves are reported individually but excluded from unambiguous-placement timing targets."},
        "movement_acceptance": movement_acceptance,
        "overall_development_acceptance": overall,
        "acceptance_status": "passed" if overall else "failed" if not movement_acceptance or targets_met is False else "incomplete",
    }
