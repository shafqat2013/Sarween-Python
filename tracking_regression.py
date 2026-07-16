"""Headless video regression runner for Sarween mini tracking.

This module reuses the combo tracking engine without opening the Tk control
panel. It can either print the movements found in a video or compare them with
expected movements from a JSON case file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

import cv_core as core
import v3_tracking as tracking
from control_panel import rc_to_a1
from mini_calibration import load_profiles_with_curves


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = ROOT / "tests" / "fixtures" / "tracking_cases.json"


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


@dataclass
class CaseResult:
    name: str
    video: str
    events: List[MovementEvent]
    failures: List[str]
    matched_expectations: int
    total_expectations: int

    @property
    def ok(self) -> bool:
        return not self.failures


def parse_time_seconds(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Invalid time value: {value!r}")


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    minutes, sec = divmod(whole, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute:02d}:{sec:02d}.{millis:03d}"
    return f"{minute:02d}:{sec:02d}.{millis:03d}"


def normalize_cell(cell: Optional[str]) -> Optional[str]:
    if cell is None:
        return None
    text = str(cell).strip()
    if not text:
        return None
    if text.startswith("r") and "c" in text:
        row_text, col_text = text[1:].split("c", 1)
        return rc_to_a1(int(row_text), int(col_text)).upper()
    return text.upper()


def a1_to_row_col(cell: str) -> Tuple[int, int]:
    text = normalize_cell(cell)
    if not text:
        raise ValueError("Cell cannot be empty")
    idx = 0
    col = 0
    while idx < len(text) and text[idx].isalpha():
        col = col * 26 + (ord(text[idx]) - ord("A") + 1)
        idx += 1
    row_text = text[idx:]
    if not row_text.isdigit() or col <= 0:
        raise ValueError(f"Invalid A1 cell: {cell!r}")
    return int(row_text) - 1, col - 1


def a1_to_raw(cell: str) -> str:
    row, col = a1_to_row_col(cell)
    return tracking._cell_label(row, col)


def raw_to_a1(cell: Optional[str]) -> Optional[str]:
    return normalize_cell(cell)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _video_metadata(video_path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, frame_count


def _seed_initial_positions(
    initial_positions: Dict[str, str],
    grid_w: int,
    grid_h: int,
    warp_w: int,
    warp_h: int,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    prev_state: Dict[str, Dict[str, Any]] = {}
    last_emitted: Dict[str, str] = {}
    cell_w = warp_w / float(grid_w)
    cell_h = warp_h / float(grid_h)
    for mini, cell in (initial_positions or {}).items():
        row, col = a1_to_row_col(cell)
        raw = tracking._cell_label(row, col)
        last_emitted[str(mini)] = raw
        prev_state[str(mini)] = {
            "last_xy": ((col + 0.5) * cell_w, (row + 0.5) * cell_h),
            "last_dist": None,
        }
    return prev_state, last_emitted


def run_video(
    video_path: Path,
    *,
    profiles_path: Path = ROOT / "combo_profiles.json",
    grid_w: Optional[int] = None,
    grid_h: Optional[int] = None,
    warp_w: Optional[int] = None,
    warp_h: Optional[int] = None,
    initial_positions: Optional[Dict[str, str]] = None,
    max_seconds: Optional[float] = None,
    max_frames: Optional[int] = None,
    motion_thresh: int = core.WARP_MOTION_THRESH,
    consensus_n: int = tracking.CONSENSUS_N,
    consensus_k: int = tracking.CONSENSUS_K,
    lost_timeout: float = 2.0,
    verbose: bool = False,
) -> List[MovementEvent]:
    video_path = video_path.resolve()
    profiles_path = profiles_path.resolve()
    profiles = load_profiles_with_curves(profiles_path)
    if not profiles:
        raise RuntimeError(f"No mini profiles found in {profiles_path}")

    fps, frame_count = _video_metadata(video_path)
    if max_seconds is not None:
        by_seconds = max(1, int(math.ceil(float(max_seconds) * fps)))
        max_frames = min(max_frames, by_seconds) if max_frames else by_seconds
    if max_frames is None and frame_count > 0:
        # CVCoreSession consumes one frame to seed camera-space background before
        # frames() starts yielding, so this prevents one looped playback frame
        # from being considered at EOF.
        max_frames = max(1, frame_count - 1)

    sess = core.CVCoreSession(
        source_path=str(video_path),
        warp_w=warp_w,
        warp_h=warp_h,
        grid_w=grid_w,
        grid_h=grid_h,
    )
    sess.warp_motion_thresh = int(motion_thresh)

    initial_positions = initial_positions or {}
    prev_state, last_emitted = _seed_initial_positions(
        initial_positions,
        sess.grid_w,
        sess.grid_h,
        sess.warp_w,
        sess.warp_h,
    )
    cell_hist: Dict[str, deque] = {}
    last_seen: Dict[str, float] = {str(mini): 0.0 for mini in initial_positions}
    events: List[MovementEvent] = []

    try:
        for bundle in sess.frames():
            if max_frames is not None and bundle.frame_idx > max_frames:
                break
            if not bundle.locked or bundle.warp_bgr is None:
                continue

            video_frame_idx = int(sess.cap.get(cv2.CAP_PROP_POS_FRAMES) or bundle.frame_idx)
            time_seconds = max(0, video_frame_idx - 1) / fps
            grid_px = min(
                bundle.warp_w / float(bundle.grid_w),
                bundle.warp_h / float(bundle.grid_h),
            )

            for mini, state in list(prev_state.items()):
                if state.get("last_xy") is None:
                    continue
                age = time_seconds - last_seen.get(mini, time_seconds)
                if age > lost_timeout:
                    state["last_xy"] = None
                    cell_hist.pop(mini, None)
                    if verbose:
                        print(f"{format_time(time_seconds)} {mini} lost; search mode")

            dets, prev_state = tracking.detect_minis(
                bundle=bundle,
                profiles=profiles,
                grid_px=grid_px,
                prev_state=prev_state,
                verbose=verbose,
            )

            for mini, det in dets.items():
                if det is None:
                    continue
                previous_xy = prev_state.get(mini, {}).get("last_xy")
                if previous_xy is not None:
                    distance_cells = math.hypot(
                        det.cx - previous_xy[0],
                        det.cy - previous_xy[1],
                    ) / grid_px
                    if distance_cells < 0.5:
                        continue
                last_seen[mini] = time_seconds

            for mini, det in dets.items():
                if det is None:
                    continue
                col, row = tracking._warp_to_cell(
                    det.cx,
                    det.cy,
                    bundle.grid_w,
                    bundle.grid_h,
                    bundle.warp_w,
                    bundle.warp_h,
                )
                raw_cell = tracking._cell_label(row, col)
                buf = cell_hist.setdefault(mini, deque(maxlen=consensus_n))
                buf.append(raw_cell)
                most, count = Counter(buf).most_common(1)[0]
                if count < consensus_k or most == last_emitted.get(mini):
                    continue

                raw_from = last_emitted.get(mini)
                last_emitted[mini] = most

                if prev_state.get(mini, {}).get("last_xy") is None:
                    cell_w = bundle.warp_w / float(bundle.grid_w)
                    cell_h = bundle.warp_h / float(bundle.grid_h)
                    parts = most[1:].split("c", 1)
                    anchor_row, anchor_col = int(parts[0]), int(parts[1])
                    prev_state.setdefault(mini, {})["last_xy"] = (
                        (anchor_col + 0.5) * cell_w,
                        (anchor_row + 0.5) * cell_h,
                    )

                event = MovementEvent(
                    mini=mini,
                    from_cell=raw_to_a1(raw_from),
                    to_cell=raw_to_a1(most) or most,
                    frame_idx=video_frame_idx,
                    time_seconds=time_seconds,
                    raw_from=raw_from,
                    raw_to=most,
                    lab_dist=float(det.lab_dist),
                    score=float(det.score),
                )
                events.append(event)
                if verbose:
                    from_text = event.from_cell or "?"
                    print(
                        f"{format_time(event.time_seconds)} "
                        f"{event.mini}: {from_text} -> {event.to_cell}"
                    )
    finally:
        sess.close()

    return events


def _expectation_label(expectation: Dict[str, Any]) -> str:
    mini = expectation.get("mini", "*")
    from_cell = normalize_cell(expectation.get("from"))
    to_cell = normalize_cell(expectation.get("to"))
    at = expectation.get("at", expectation.get("time", "?"))
    if from_cell:
        return f"{mini} {from_cell}->{to_cell} at {at}"
    return f"{mini} ->{to_cell} at {at}"


def _event_matches_expectation(
    event: MovementEvent,
    expectation: Dict[str, Any],
    default_tolerance: float,
) -> bool:
    if str(event.mini) != str(expectation.get("mini")):
        return False
    expected_to = normalize_cell(expectation.get("to"))
    if expected_to and event.to_cell != expected_to:
        return False
    expected_from = normalize_cell(expectation.get("from"))
    if expected_from and event.from_cell != expected_from:
        return False
    expected_time = parse_time_seconds(expectation.get("at", expectation.get("time")))
    tolerance = float(expectation.get("tolerance_seconds", default_tolerance))
    return abs(event.time_seconds - expected_time) <= tolerance


def check_case(case: Dict[str, Any], *, cases_base_dir: Path) -> CaseResult:
    name = str(case.get("name") or case.get("video") or "unnamed")
    video = _resolve_path(str(case["video"]), cases_base_dir)
    profiles = _resolve_path(str(case.get("profiles", ROOT / "combo_profiles.json")), cases_base_dir)
    grid = case.get("grid") or {}
    expectations = list(case.get("expectations") or [])
    default_tolerance = float(case.get("tolerance_seconds", 2.0))
    ignore_before = parse_time_seconds(case.get("ignore_before", 0))
    ignore_after_value = case.get("ignore_after")
    ignore_after = parse_time_seconds(ignore_after_value) if ignore_after_value is not None else None

    events = run_video(
        video,
        profiles_path=profiles,
        grid_w=grid.get("cols") or case.get("grid_cols"),
        grid_h=grid.get("rows") or case.get("grid_rows"),
        warp_w=grid.get("warp_width") or case.get("warp_w"),
        warp_h=grid.get("warp_height") or case.get("warp_h"),
        initial_positions=case.get("initial_positions") or {},
        max_seconds=case.get("max_seconds"),
        max_frames=case.get("max_frames"),
        motion_thresh=int(case.get("motion_thresh", core.WARP_MOTION_THRESH)),
        consensus_n=int(case.get("consensus_n", tracking.CONSENSUS_N)),
        consensus_k=int(case.get("consensus_k", tracking.CONSENSUS_K)),
        lost_timeout=float(case.get("lost_timeout", 2.0)),
        verbose=bool(case.get("verbose", False)),
    )

    scoped_events = [
        event for event in events
        if event.time_seconds >= ignore_before
        and (ignore_after is None or event.time_seconds <= ignore_after)
    ]

    failures: List[str] = []
    used_event_indexes: set[int] = set()
    for expectation in expectations:
        matches = [
            (idx, event) for idx, event in enumerate(scoped_events)
            if idx not in used_event_indexes
            and _event_matches_expectation(event, expectation, default_tolerance)
        ]
        if not matches:
            failures.append(f"Missing expected movement: {_expectation_label(expectation)}")
            continue
        expected_time = parse_time_seconds(expectation.get("at", expectation.get("time")))
        idx, _ = min(matches, key=lambda item: abs(item[1].time_seconds - expected_time))
        used_event_indexes.add(idx)

    if not bool(case.get("allow_unexpected", False)):
        for idx, event in enumerate(scoped_events):
            if idx in used_event_indexes:
                continue
            from_text = event.from_cell or "?"
            failures.append(
                "Unexpected movement: "
                f"{event.mini} {from_text}->{event.to_cell} "
                f"at {format_time(event.time_seconds)}"
            )

    return CaseResult(
        name=name,
        video=str(video),
        events=events,
        failures=failures,
        matched_expectations=len(used_event_indexes),
        total_expectations=len(expectations),
    )


def load_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("cases") or [])


def check_cases(path: Path) -> List[CaseResult]:
    cases = load_cases(path)
    return [check_case(case, cases_base_dir=path.parent) for case in cases]


def events_to_json(events: Iterable[MovementEvent]) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        row = asdict(event)
        row["time"] = format_time(event.time_seconds)
        out.append(row)
    return out


def _print_case_result(result: CaseResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"{status} {result.name}: {result.matched_expectations}/{result.total_expectations} expected movements matched")
    for event in result.events:
        from_text = event.from_cell or "?"
        print(f"  actual {format_time(event.time_seconds)} {event.mini}: {from_text}->{event.to_cell}")
    for failure in result.failures:
        print(f"  {failure}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run Sarween mini tracking against video regressions.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Print movement events detected in one video.")
    run_parser.add_argument("video", help="Video file to process.")
    run_parser.add_argument("--profiles", default=str(ROOT / "combo_profiles.json"))
    run_parser.add_argument("--grid-cols", type=int, default=None)
    run_parser.add_argument("--grid-rows", type=int, default=None)
    run_parser.add_argument("--warp-width", type=int, default=None)
    run_parser.add_argument("--warp-height", type=int, default=None)
    run_parser.add_argument("--max-seconds", type=float, default=None)
    run_parser.add_argument("--max-frames", type=int, default=None)
    run_parser.add_argument("--motion-thresh", type=int, default=core.WARP_MOTION_THRESH)
    run_parser.add_argument("--verbose", action="store_true")

    check_parser = sub.add_parser("check", help="Check a JSON regression case file.")
    check_parser.add_argument("cases", nargs="?", default=str(DEFAULT_CASES_PATH))

    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            events = run_video(
                Path(args.video).expanduser().resolve(),
                profiles_path=Path(args.profiles).expanduser().resolve(),
                grid_w=args.grid_cols,
                grid_h=args.grid_rows,
                warp_w=args.warp_width,
                warp_h=args.warp_height,
                max_seconds=args.max_seconds,
                max_frames=args.max_frames,
                motion_thresh=args.motion_thresh,
                verbose=args.verbose,
            )
            print(json.dumps(events_to_json(events), indent=2))
            return 0

        results = check_cases(Path(args.cases).expanduser().resolve())
        for result in results:
            _print_case_result(result)
        if any(not result.ok for result in results):
            return 1
        return 0
    except Exception as exc:
        print(f"tracking_regression: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
