"""Headless video regression runner for Sarween mini tracking.

This module reuses the combo tracking engine without opening the Tk control
panel. It can either print the movements found in a video or compare them with
expected movements from a JSON case file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2

import cv_core as core
import foundryoutput as fo
import v3_tracking as tracking
from mini_calibration import load_profiles_with_curves
from tracking_engine import TrackingEngine
from app_paths import data_path
from tracking_evaluation import (
    CaseResult, MovementEvent, a1_to_row_col, evaluate_events,
    event_matches_expectation as _event_matches_expectation,
    format_time, normalize_cell, parse_time_seconds, validate_case,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = ROOT / "tests" / "fixtures" / "tracking_cases.json"


class FoundryTimelineReplay:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if int(self.data.get("schemaVersion", 0)) not in (1, 2):
            raise ValueError(f"Unsupported Foundry timeline schema: {self.path}")
        self.events = sorted(
            list(self.data.get("events") or []), key=lambda event: int(event["frame"])
        )
        if not self.events:
            raise ValueError(f"Foundry timeline has no events: {self.path}")
        self._next_event = 0
        self._last_visual_revision: Optional[int] = None
        self.selected_mini = None
        self.paused = False
        self.has_controls = all("trackingControls" in event for event in self.events)
        self.clocks = {int(item["frame"]): item for item in self.data.get("frameClocks", [])}
        self.checkpoints = self.data.get("checkpoints", [])
        self.has_checkpoint = any(item["frame"] == 0 and item.get("kind") == "initial" for item in self.checkpoints)

    @property
    def prediction_paused(self):
        # Guided footage pauses live predictions to collect labels; evaluation
        # must run the tracker rather than reproduce that intentional pause.
        return self.paused and not bool(self.data.get("capture"))

    def apply_through(self, frame: int) -> None:
        while self._next_event < len(self.events):
            event = self.events[self._next_event]
            if int(event["frame"]) > int(frame):
                break
            self._apply_event(event)
            self._next_event += 1

    def _apply_event(self, event: Dict[str, Any]) -> None:
        controls = event.get("trackingControls") or {}
        self.selected_mini = controls.get("selectedMini")
        self.paused = bool(controls.get("paused", False))
        scene = event.get("sceneInfo") or {}
        if scene:
            fo.set_scene_params(
                scene_id=scene.get("sceneId"),
                scene_w=scene.get("width"),
                scene_h=scene.get("height"),
                grid_px=scene.get("gridSize"),
                shift_x=scene.get("shiftX", 0),
                shift_y=scene.get("shiftY", 0),
                grid_type=scene.get("gridType"),
                background=scene.get("background"),
            )

        view = event.get("viewTransform")
        if view is None:
            fo.clear_view_transform()
        else:
            fo.set_view_transform(view)

        visual_revision = int(event.get("sceneVisualRevision", 0))
        if (
            self._last_visual_revision is not None
            and visual_revision != self._last_visual_revision
        ):
            fo.mark_scene_visual_changed(
                str(event.get("sceneVisualReason") or "timelineReplay")
            )
        self._last_visual_revision = visual_revision
        if self.data.get("schemaVersion") == 2:
            # This only runs in the isolated replay process.
            with fo._view_transform_lock:
                fo._view_transform_revision = int(event.get("viewTransformRevision", fo._view_transform_revision))
                fo._scene_visual_revision = visual_revision


def find_timeline_path(video_path: Path) -> Optional[Path]:
    candidate = fo.timeline_path_for_video(video_path)
    return candidate if candidate.exists() else None


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


def _run_video(
    video_path: Path,
    *,
    profiles_path: Optional[Path] = None,
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
    timeline_path: Optional[Path] = None,
    marker_mode: Optional[str] = None,
    view_settle_seconds: float = core.VIEW_SETTLE_SECONDS,
    frames_undistorted: Optional[bool] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    restore_recorded_profiles: bool = True,
    progress_path: Optional[str] = None,
) -> List[MovementEvent]:
    started = time.monotonic()
    diagnostics = diagnostics if diagnostics is not None else {}
    video_path = video_path.resolve()
    timeline_path = timeline_path or find_timeline_path(video_path)
    timeline = FoundryTimelineReplay(timeline_path) if timeline_path else None
    saved_profiles = video_path.with_suffix(".profiles.json")
    profiles_path = (profiles_path or (saved_profiles if saved_profiles.exists() else data_path("combo_profiles.json"))).resolve()
    profiles = load_profiles_with_curves(profiles_path)
    if not profiles:
        raise RuntimeError(f"No mini profiles found in {profiles_path}")
    for value in (grid_w, grid_h, warp_w, warp_h):
        if value is not None and int(value) <= 0:
            raise ValueError("Grid and warp dimensions must be positive")

    if timeline_path is None:
        timeline_path = find_timeline_path(video_path)
    source_frames_undistorted = False
    if timeline is not None:
        timeline.apply_through(0)
        marker_mode = marker_mode or str(timeline.data.get("markerMode") or "viewport")
        grid_w = grid_w or timeline.data.get("gridCols")
        grid_h = grid_h or timeline.data.get("gridRows")
        warp_w = warp_w or timeline.data.get("warpWidth")
        warp_h = warp_h or timeline.data.get("warpHeight")
        # Timeline sidecars were introduced for videos recorded by CVCoreSession,
        # which stores frames after lens correction. Default true for the first
        # schema revision so sidecars written before this field remain replayable.
        source_frames_undistorted = bool(
            timeline.data.get("framesUndistorted", True)
        )
        print(f"TrackingRegression | Replaying Foundry timeline: {timeline.path}")

    if frames_undistorted is not None:
        source_frames_undistorted = bool(frames_undistorted)
    # Offline results must not depend on the last interactive hardware setup.
    marker_mode = marker_mode or "legacy"
    grid_w, grid_h = int(grid_w or 23), int(grid_h or 16)
    warp_w, warp_h = int(warp_w or 1280), int(warp_h or 720)
    if min(grid_w, grid_h, warp_w, warp_h) <= 0:
        raise ValueError("Grid and warp dimensions must be positive")
    if not 1 <= consensus_k <= consensus_n:
        raise ValueError("Consensus requires 1 <= k <= n")
    for name, value in (("max_seconds", max_seconds), ("max_frames", max_frames)):
        if value is not None and (not math.isfinite(float(value)) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")

    fps, frame_count = _video_metadata(video_path)
    if max_seconds is not None:
        by_seconds = max(1, int(math.ceil(float(max_seconds) * fps)))
        max_frames = min(max_frames, by_seconds) if max_frames else by_seconds
    if max_frames is None and frame_count > 0:
        # The initial frame seeds camera background; remaining frames are scored.
        max_frames = max(1, frame_count if timeline and timeline.has_checkpoint else frame_count - 1)

    diagnostics.update({
        "fps": fps, "source_frames": frame_count, "processed_frames": 0,
        "locked_frames": 0, "unlocked_frames": 0, "unmapped_frames": 0,
        "tracked_frames": 0, "unusable_warp_frames": 0,
        "marker_missing_frames": {}, "profile_ids": sorted(profiles),
        "marker_mode": marker_mode, "frames_undistorted": source_frames_undistorted,
        "grid": [grid_w, grid_h], "warp": [warp_w, warp_h],
        "timeline_ground_truth_count": len(timeline.data.get("groundTruth", [])) if timeline else 0,
        "rendered_reference_count": len(timeline.data.get("referenceFrames", [])) if timeline else 0,
        "frame_limit": max_frames,
        "paused_frames": 0, "taps": [],
        "clock_source": "recorded_monotonic" if timeline and timeline.has_checkpoint else "video_fps",
        "map_states": [],
        "controls_source": "timeline" if timeline and timeline.has_controls else "defaults",
        "pause_policy": "evaluate_guided_capture" if timeline and timeline.data.get("capture") else "recorded",
        "replay_limitations": [
            "Timeouts use nominal video FPS; original per-frame wall-clock timing was not recorded.",
            "Replay starts with fresh tracking/background state; mid-recording rescans and background recaptures are not restored.",
        ],
    })
    if timeline and timeline.has_checkpoint:
        diagnostics["replay_limitations"] = [
            "Video compression can change pixel values; checkpoint replay is not a lossless copy of camera frames."]
        if any(timeline.clocks.get(i, {}).get("captureTime") is None for i in range(frame_count)):
            diagnostics["replay_limitations"].append("Some capture clocks are missing; timing fidelity is partial.")
            diagnostics["clock_source"] = "partial_recorded_monotonic"
    if not timeline or not timeline.has_controls:
        diagnostics["replay_limitations"].append(
            "Selection/pause inputs were not fully recorded; missing inputs default to no selection and tracking enabled.")
    for limitation in diagnostics["replay_limitations"]:
        print(f"TrackingRegression | Replay limitation: {limitation}", file=sys.stderr, flush=True)

    engine = TrackingEngine(tracking.detect_minis, consensus_n=consensus_n,
                            consensus_k=consensus_k, lost_timeout=lost_timeout)
    sess = core.CVCoreSession(
        source_path=str(video_path),
        warp_w=warp_w,
        warp_h=warp_h,
        grid_w=grid_w,
        grid_h=grid_h,
        marker_mode=marker_mode,
        before_frame_callback=timeline.apply_through if timeline is not None else None,
        source_frames_undistorted=source_frames_undistorted,
        view_settle_seconds=view_settle_seconds,
    )
    sess.warp_motion_thresh = int(motion_thresh)
    if timeline and timeline.has_checkpoint:
        from recording_state import restore_checkpoint
        sess.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        sess.frame_idx = -1
        def before_exact_frame(frame):
            nonlocal profiles
            timeline.apply_through(frame)
            clock = timeline.clocks.get(frame, {})
            if clock.get("stateEvent") is not None:
                timeline._apply_event(timeline.data["events"][int(clock["stateEvent"])])
            for item in timeline.checkpoints:
                if item["frame"] == frame:
                    recorded = restore_checkpoint(timeline.path, item, sess, engine)
                    if recorded is not None and restore_recorded_profiles:
                        profiles = recorded
            sess.replay_frame_clock = clock.get("captureTime")
            if clock.get("motionThreshold") is not None:
                sess.warp_motion_thresh = int(clock["motionThreshold"])
        sess.before_frame_callback = before_exact_frame
    last_map_time = -1.0

    events: List[MovementEvent] = []
    next_progress = time.monotonic() + 5

    try:
        engine.seed_positions({str(mini): a1_to_raw(cell) for mini, cell in (initial_positions or {}).items()}, sess)
        for bundle in sess.frames():
            if max_frames is not None and diagnostics["processed_frames"] >= max_frames:
                break
            diagnostics["processed_frames"] += 1
            diagnostics["locked_frames" if bundle.locked else "unlocked_frames"] += 1
            for marker in bundle.last_missing_ids:
                key = str(marker)
                diagnostics["marker_missing_frames"][key] = diagnostics["marker_missing_frames"].get(key, 0) + 1
            if time.monotonic() >= next_progress:
                print(f"TrackingRegression | {video_path.name}: frame {bundle.frame_idx}/{frame_count}, "
                      f"{len(events)} moves, {time.monotonic() - started:.0f}s elapsed", file=sys.stderr, flush=True)
                next_progress = time.monotonic() + 5
            video_frame_idx = int(sess.cap.get(cv2.CAP_PROP_POS_FRAMES) or bundle.frame_idx)
            time_seconds = max(0, video_frame_idx - 1) / fps
            clock = timeline.clocks.get(video_frame_idx - 1, {}) if timeline and timeline.has_checkpoint else {}
            if clock.get("trackingState"):
                timeline._apply_event(clock["trackingState"])
            if timeline and timeline.has_checkpoint:
                for change in timeline.data.get("profileChanges", []):
                    if change["frame"] == video_frame_idx - 1:
                        if restore_recorded_profiles:
                            profiles = change["profiles"]
                        engine.reset_mini(change["mini"])
            tracking_now = clock.get("trackingTime") or clock.get("captureTime") or time_seconds
            context = tracking.tracking_context(
                bundle, now=tracking_now, selected_mini=timeline.selected_mini if timeline else None,
                paused=timeline.prediction_paused if timeline else False)
            if clock and not clock.get("tracked") and not timeline.data.get("capture"):
                engine.synchronize(context)
                continue
            step = engine.step(bundle, profiles, context, verbose=verbose)
            if time_seconds - last_map_time >= .2 or step.moves or step.lost_minis:
                scene = fo.get_scene_params()
                grid = [scene.get("gridCols") or bundle.grid_w, scene.get("gridRows") or bundle.grid_h]
                diagnostics["map_states"].append({"time_seconds": time_seconds, "grid": grid,
                    "scene": fo.get_scene_params().get("sceneId"), "locked": bool(bundle.locked),
                    "positions": {mini: {"cell": raw_to_a1(cell),
                        "status": "tracked" if not step.reason and tracking_now - engine.last_seen.get(mini, -1e9) <= 2 else "lost"}
                        for mini, cell in engine.last_emitted.items()}})
                last_map_time = time_seconds
            if progress_path and (diagnostics["processed_frames"] % 15 == 0):
                from app_paths import atomic_write_json
                atomic_write_json(progress_path, {"frame": video_frame_idx, "total": frame_count,
                                                  "moves": len(events)}, backup=False)
            if bundle.locked and bundle.warp_bgr is None:
                diagnostics["unusable_warp_frames"] += 1
            if step.reason == "unmapped":
                diagnostics["unmapped_frames"] += 1
            if step.reason == "paused":
                diagnostics["paused_frames"] += 1
            if step.reason:
                continue
            diagnostics["tracked_frames"] += 1
            if step.tapped_mini:
                diagnostics["taps"].append({"mini": step.tapped_mini, "frame": video_frame_idx,
                                            "time_seconds": time_seconds})
            for move in step.moves:
                event = MovementEvent(
                    mini=move.mini,
                    from_cell=raw_to_a1(move.raw_from),
                    to_cell=raw_to_a1(move.raw_to) or move.raw_to,
                    frame_idx=video_frame_idx,
                    time_seconds=time_seconds,
                    raw_from=move.raw_from,
                    raw_to=move.raw_to,
                    lab_dist=move.lab_dist,
                    score=move.score,
                    source=move.source,
                )
                events.append(event)
                if verbose:
                    print(f"{format_time(event.time_seconds)} {event.mini}: "
                          f"{event.from_cell or '?'} -> {event.to_cell} ({event.source})")
    finally:
        sess.close()
        diagnostics["elapsed_seconds"] = time.monotonic() - started
        diagnostics["processed_fps"] = diagnostics["processed_frames"] / max(0.001, diagnostics["elapsed_seconds"])

    if frame_count > 1:
        available = frame_count if timeline and timeline.has_checkpoint else frame_count - 1
        expected_frames = min(available, max_frames) if max_frames is not None else available
        if diagnostics["processed_frames"] < expected_frames:
            raise RuntimeError(f"Video decoding ended early: {diagnostics['processed_frames']}/{expected_frames} frames processed")
    return events


def file_fingerprint(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def run_video(video_path: Path, *, timeout_seconds: float = 180, threads: int = 1,
              diagnostics: Optional[dict] = None, **options) -> List[MovementEvent]:
    """Run in a disposable process, with a wall-clock deadline and no live state."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if threads < 1:
        raise ValueError("threads must be positive")
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise ValueError(f"Video file does not exist: {video_path}")
    options = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in options.items()}
    payload = {"video_path": str(video_path), "threads": threads, "options": options,
               "timeout_seconds": min(timeout_seconds, 3600), "parent_pid": os.getpid()}
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(threads)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="sarween-replay-") as directory:
        request = Path(directory) / "request.json"
        response = Path(directory) / "response.json"
        request.write_text(json.dumps(payload), encoding="utf-8")
        try:
            # subprocess.run kills AND waits for this worker on timeout or Ctrl-C.
            completed = subprocess.run(
                worker_command(request, response),
                cwd=ROOT, env=env, timeout=timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Replay exceeded {timeout_seconds:g}s wall-clock limit; worker stopped") from exc
        if not response.exists():
            raise RuntimeError(f"Replay worker exited with code {completed.returncode} without a result")
        data = json.loads(response.read_text(encoding="utf-8"))
        if completed.returncode or "error" in data:
            raise RuntimeError(data.get("error") or f"Replay worker exited with code {completed.returncode}")
        if diagnostics is not None:
            diagnostics.update(data["diagnostics"])
        return [MovementEvent(**item) for item in data["events"]]


def worker_command(request, response):
    if getattr(sys, "frozen", False):
        return [sys.executable, "--replay-worker", str(request), str(response)]
    return [sys.executable, str(Path(__file__).resolve()), "_worker", str(request), str(response)]


def start_worker_guard(timeout_seconds, parent_pid=None):
    """The worker exits even if its parent UI freezes or is force-quit."""
    import threading
    deadline = time.monotonic() + float(timeout_seconds)
    stopped = threading.Event()
    def guard():
        while not stopped.wait(.25):
            expired = time.monotonic() >= deadline
            if parent_pid:
                try:
                    os.kill(int(parent_pid), 0)
                except ProcessLookupError:
                    expired = True
                except PermissionError:
                    pass
            if expired:
                os._exit(124)
    thread = threading.Thread(target=guard, name="replay-deadline", daemon=True)
    thread.start()
    return stopped, thread


def _worker(request: Path, response: Path) -> int:
    guard = None
    try:
        payload = json.loads(request.read_text(encoding="utf-8"))
        timeout = float(payload.get("timeout_seconds", 180))
        if not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("Invalid worker time limit")
        guard = start_worker_guard(timeout, payload.get("parent_pid"))
        # Apple's GCD backend ignores positive limits; zero disables parallelism.
        cv2.setNumThreads(0 if int(payload["threads"]) == 1 else int(payload["threads"]))
        cv2.setRNGSeed(0)
        if hasattr(os, "nice"):
            try:
                os.nice(10)
            except OSError:
                pass  # Sandboxed macOS may disallow even lowering priority.
        options = payload["options"]
        video = Path(payload["video_path"])
        for key in ("profiles_path", "timeline_path"):
            if options.get(key) is not None:
                options[key] = Path(options[key])
        saved_profiles = video.with_suffix(".profiles.json")
        profiles = options.get("profiles_path") or (saved_profiles if saved_profiles.exists() else data_path("combo_profiles.json"))
        timeline = options.get("timeline_path") or find_timeline_path(video)
        diagnostics = {"opencv_version": cv2.__version__, "threads": cv2.getNumThreads(),
                       "inputs": {"video": file_fingerprint(video), "profiles": file_fingerprint(profiles)}}
        diagnostics["code_sha256"] = {
            name: file_fingerprint(ROOT / name)["sha256"] for name in
            ("tracking_regression.py", "tracking_evaluation.py", "cv_core.py", "v3_tracking.py",
             "tracking_engine.py", "tap_selection.py", "mini_tracking.py", "mini_calibration.py", "foundryoutput.py", "app_paths.py", "recording_state.py")
            if (ROOT / name).is_file()
        }
        if getattr(sys, "frozen", False):
            diagnostics["executable_sha256"] = file_fingerprint(Path(sys.executable))["sha256"]
        if timeline:
            diagnostics["inputs"]["timeline"] = file_fingerprint(timeline)
        for filename in ("camera_matrix.npy", "dist_coeffs.npy"):
            if data_path(filename).exists():
                diagnostics["inputs"][filename] = file_fingerprint(data_path(filename))
        events = _run_video(video, diagnostics=diagnostics, **options)
        response.write_text(json.dumps({"events": [asdict(event) for event in events], "diagnostics": diagnostics}), encoding="utf-8")
        return 0
    except Exception as exc:
        response.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), encoding="utf-8")
        return 2
    finally:
        if guard:
            guard[0].set()
            guard[1].join(timeout=1)


def check_case(case: Dict[str, Any], *, cases_base_dir: Path,
               timeout_seconds: float = 180, threads: int = 1,
               profiles_override: Optional[Path] = None) -> CaseResult:
    validate_case(case)
    name = str(case.get("name") or case.get("video") or "unnamed")
    video = _resolve_path(str(case["video"]), cases_base_dir)
    profiles = profiles_override or _resolve_path(str(case.get("profiles", data_path("combo_profiles.json"))), cases_base_dir)
    grid = case.get("grid") or {}
    expectations = list(case.get("expectations") or [])
    timeline_value = case.get("timeline")
    timeline_path = (
        _resolve_path(str(timeline_value), cases_base_dir)
        if timeline_value
        else None
    )

    missing = []
    if not video.exists():
        missing.append(f"video {video}")
    if timeline_path is not None and not timeline_path.exists():
        missing.append(f"timeline {timeline_path}")
    if missing:
        return CaseResult(
            name=name,
            video=str(video),
            events=[],
            failures=[],
            matched_expectations=0,
            total_expectations=len(expectations),
            skipped="Missing local " + " and ".join(missing),
            label_source=case.get("label_source", "unverified"),
        )

    loaded_profiles = load_profiles_with_curves(profiles)
    expected_minis = {str(item["mini"]) for item in expectations} | set(case.get("initial_positions") or {})
    absent = expected_minis - set(loaded_profiles)
    if absent:
        raise ValueError(f"Profiles missing for minis: {', '.join(sorted(absent))}")
    diagnostics = {}
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
        timeline_path=timeline_path,
        marker_mode=case.get("marker_mode"),
        view_settle_seconds=float(
            case.get("view_settle_seconds", core.VIEW_SETTLE_SECONDS)
        ),
        frames_undistorted=case.get("frames_undistorted"),
        restore_recorded_profiles=profiles_override is None and bool(case.get("restore_recorded_profiles", True)),
        timeout_seconds=timeout_seconds,
        threads=threads,
        diagnostics=diagnostics,
    )
    result = evaluate_events(case, events, video=str(video))
    result.diagnostics = diagnostics
    result.diagnostics["case"] = case
    if not diagnostics.get("locked_frames"):
        result.failures.append("No frames had marker lock; tracking was not evaluated")
    elif not diagnostics.get("tracked_frames"):
        result.failures.append("No locked frames had a usable Foundry grid transform")
    if diagnostics.get("unusable_warp_frames"):
        result.failures.append(f"Camera processing failed on {diagnostics['unusable_warp_frames']} locked frames")
    return result


def load_cases(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return list(data.get("cases") or [])


def check_cases(path: Path, *, names: Optional[List[str]] = None, **options) -> List[CaseResult]:
    cases = load_cases(path)
    case_names = [str(case.get("name") or case.get("video")) for case in cases]
    if len(case_names) != len(set(case_names)):
        raise ValueError("Case names must be unique")
    if names and set(names) - set(case_names):
        raise ValueError(f"Unknown case names: {sorted(set(names) - set(case_names))}")
    results = []
    for case, name in zip(cases, case_names):
        if names and name not in names:
            continue
        print(f"TrackingRegression | Starting {name}", flush=True)
        try:
            result = check_case(case, cases_base_dir=path.parent, **options)
        except Exception as exc:
            result = CaseResult(name=name, video=str(case.get("video", "")), events=[],
                                failures=[f"Replay error: {exc}"], matched_expectations=0,
                                total_expectations=len(case.get("expectations") or []),
                                label_source=case.get("label_source", "unverified"))
        results.append(result)
        _print_case_result(result)
    return results


def events_to_json(events: Iterable[MovementEvent]) -> List[Dict[str, Any]]:
    out = []
    for event in events:
        row = asdict(event)
        row["time"] = format_time(event.time_seconds)
        out.append(row)
    return out


def _print_case_result(result: CaseResult) -> None:
    if result.skipped:
        print(f"SKIP {result.name}: {result.skipped}")
        return
    status = "PASS" if result.ok else "FAIL"
    print(f"{status} {result.name}: {result.matched_expectations}/{result.total_expectations} expected movements matched; "
          f"{len(result.unexpected)} extra; labels={result.label_source}", flush=True)
    for event in result.events:
        from_text = event.from_cell or "?"
        source = f" [{event.source}]" if event.source != "detection" else ""
        print(
            f"  actual {format_time(event.time_seconds)} "
            f"{event.mini}: {from_text}->{event.to_cell}{source}"
        )
    for failure in result.failures:
        print(f"  {failure}")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        return _worker(Path(argv[1]), Path(argv[2]))
    parser = argparse.ArgumentParser(description="Run Sarween mini tracking against video regressions.")
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Print movement events detected in one video.")
    run_parser.add_argument("video", help="Video file to process.")
    run_parser.add_argument("--profiles", help="Override recorded profiles and profile changes.")
    run_parser.add_argument("--grid-cols", type=int, default=None)
    run_parser.add_argument("--grid-rows", type=int, default=None)
    run_parser.add_argument("--warp-width", type=int, default=None)
    run_parser.add_argument("--warp-height", type=int, default=None)
    run_parser.add_argument("--max-seconds", type=float, default=None)
    run_parser.add_argument("--max-frames", type=int, default=None)
    run_parser.add_argument("--motion-thresh", type=int, default=core.WARP_MOTION_THRESH)
    run_parser.add_argument(
        "--timeline",
        default=None,
        help="Foundry timeline JSON. Defaults to VIDEO.tracking.json when present.",
    )
    run_parser.add_argument(
        "--marker-mode", choices=("legacy", "viewport"), default=None
    )
    run_parser.add_argument(
        "--view-settle-seconds",
        type=float,
        default=core.VIEW_SETTLE_SECONDS,
    )
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.add_argument("--frames-undistorted", action="store_true", default=None)

    check_parser = sub.add_parser("check", help="Check a JSON regression case file.")
    check_parser.add_argument("cases", nargs="?", default=str(DEFAULT_CASES_PATH))
    check_parser.add_argument("--case", action="append", dest="names", help="Run only this case; repeat to select several.")
    check_parser.add_argument("--profiles", help="Override frozen profiles for a comparison run.")
    check_parser.add_argument("--report-dir", default=str(ROOT / "tracking_reports" / "latest"))
    check_parser.add_argument("--baseline", help="Previous report.json to compare against.")
    check_parser.add_argument("--require-videos", action="store_true", help="Fail when any selected local video is missing.")
    for command in (run_parser, check_parser):
        command.add_argument("--timeout-seconds", type=float, default=180, help="Wall-clock deadline per video (default: 180).")
        command.add_argument("--threads", type=int, default=1, help="OpenCV/BLAS threads per replay (default: 1).")

    inventory_parser = sub.add_parser("inventory", help="List recordings and independent labels without replaying them.")
    inventory_parser.add_argument("directory", nargs="?", default=str(ROOT))

    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            for video in sorted(Path(args.directory).expanduser().glob("*.mp4")):
                timeline_path = find_timeline_path(video)
                data = json.loads(timeline_path.read_text(encoding="utf-8")) if timeline_path else {}
                print(f"{video.name}: timeline={'yes' if timeline_path else 'no'}, "
                      f"confirmed placements={len(data.get('groundTruth', []))}, "
                      f"rendered references={len(data.get('referenceFrames', []))}, "
                      f"scan snapshot={'yes' if video.with_suffix('.profiles.json').exists() else 'no'}")
            return 0
        if args.command == "run":
            events = run_video(
                Path(args.video).expanduser().resolve(),
                profiles_path=Path(args.profiles).expanduser().resolve() if args.profiles else None,
                restore_recorded_profiles=not bool(args.profiles),
                grid_w=args.grid_cols,
                grid_h=args.grid_rows,
                warp_w=args.warp_width,
                warp_h=args.warp_height,
                max_seconds=args.max_seconds,
                max_frames=args.max_frames,
                motion_thresh=args.motion_thresh,
                verbose=args.verbose,
                timeline_path=(
                    Path(args.timeline).expanduser().resolve()
                    if args.timeline
                    else None
                ),
                marker_mode=args.marker_mode,
                view_settle_seconds=args.view_settle_seconds,
                frames_undistorted=args.frames_undistorted,
                timeout_seconds=args.timeout_seconds,
                threads=args.threads,
            )
            print(json.dumps(events_to_json(events), indent=2))
            return 0

        cases_path = Path(args.cases).expanduser().resolve()
        results = check_cases(cases_path, names=args.names, timeout_seconds=args.timeout_seconds,
                              threads=args.threads,
                              profiles_override=Path(args.profiles).expanduser().resolve() if args.profiles else None)
        from tracking_reports import write_report
        paths = write_report(results, Path(args.report_dir).expanduser(), cases_path=cases_path,
                             baseline_path=Path(args.baseline).expanduser() if args.baseline else None)
        print(f"Reports: {paths[0]} and {paths[1]}")
        if not results or all(result.skipped for result in results):
            print("No videos were evaluated", file=sys.stderr)
            return 2
        if any(not result.ok or (args.require_videos and result.skipped) for result in results):
            return 1
        return 0
    except KeyboardInterrupt:
        print("Replay cancelled; worker stopped.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"tracking_regression: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
