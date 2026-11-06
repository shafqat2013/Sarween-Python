"""Lossless observer inputs and deterministic replay, independent of the UI.

PNG inputs preserve exactly the rectified pixels and exclusion mask consumed
by ObserverEngine. A normal camera MP4 is useful for watching, but its lossy
encoding cannot establish identical live/replay decisions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import marshal
import math
from pathlib import Path
import time

import cv2
import numpy as np

from observer_engine import BoardGeometry, ObserverEngine, draw_observer_frame
from app_paths import resource_path


ENGINE_SOURCES = ("observer_engine.py", "poc_ring_detector.py", "poc_scene_reference.py",
                  "single_mover_state.py", "annular_ring_detector.py", "hybrid_ring_detector.py",
                  "watermark_detection.py")
ENGINE_ASSETS = ("assets/watermarks/obs-caption-center.png",
                 "assets/watermarks/obs-caption-left.png")


def fingerprints():
    result = {}
    for name in ENGINE_SOURCES:
        try:
            data = Path(__file__).with_name(name).read_bytes()
            key = name
        except FileNotFoundError:
            # PyInstaller can import a module from its bundled bytecode without
            # exposing a sibling .py file. Record that different provenance
            # explicitly; never label compiled code as a source-file hash.
            module = name.removesuffix(".py")
            spec = importlib.util.find_spec(module)
            loader = getattr(spec, "loader", None)
            getter = getattr(loader, "get_code", None)
            code = getter(module) if getter else None
            if code is None:
                raise RuntimeError(f"Cannot fingerprint observer module {module}")
            data, key = marshal.dumps(code), name + ":bytecode"
        result[key] = hashlib.sha256(data).hexdigest()
    for name in ENGINE_ASSETS:
        try:
            with open(resource_path(name), "rb") as stream:
                data = stream.read()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Cannot fingerprint observer watermark asset {name}") from exc
        result[name] = hashlib.sha256(data).hexdigest()
    return result


# The app may remain open while source files are edited. A new recording must
# not label old imported tracker objects with fingerprints read from newer disk
# files. The engine modules have been imported above before this baseline.
LOADED_ENGINE_FINGERPRINTS = fingerprints()


def _recording_fingerprints():
    current = fingerprints()
    if current != LOADED_ENGINE_FINGERPRINTS:
        raise RuntimeError("Restart Sarween after tracker code changes before recording observer inputs")
    return current


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ObserverJournal:
    """An explicit, bounded recording interval with no automatic overwrites."""

    def __init__(self, folder, initialization, *, max_seconds=300, max_bytes=2_000_000_000):
        self.max_seconds, self.max_bytes = float(max_seconds), int(max_bytes)
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0 or self.max_bytes <= 0:
            raise ValueError("Observer recording limits must be positive and finite")
        canonical(initialization)
        code_fingerprints = _recording_fingerprints()
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder / "inputs").mkdir()
        self.frames = self.bytes = 0
        self.first_timestamp = None
        self.last_timestamp = None
        self.closed = False
        self.manifest = {"schema_version": 1, "kind": "sarween-observer-journal",
                         "created_at": datetime.now(timezone.utc).isoformat(),
                         "initialization": deepcopy(initialization),
                         "code_sha256": code_fingerprints, "frames": 0, "complete": False,
                         "foundry_token_output": False,
                         "scope": "recorded observer interval only",
                         "byte_count_scope": "PNG and JSONL frame payloads; excludes manifest",
                         "limits": {"seconds": self.max_seconds, "bytes": self.max_bytes}}
        _save_json(self.folder / "manifest.json", self.manifest)
        self._inputs = (self.folder / "inputs.jsonl").open("w")
        self._decisions = (self.folder / "decisions.jsonl").open("w")

    def append(self, board, mask, timestamp, record, *, locked=True, geometry_key=None,
               frame_index=None, metadata=None):
        if self.closed:
            raise RuntimeError("Observer recording has ended")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.last_timestamp is not None and timestamp <= self.last_timestamp):
            raise ValueError("Observer recording timestamps must be finite and strictly increasing")
        if self.first_timestamp is not None and timestamp - self.first_timestamp > self.max_seconds:
            self.manifest["limit_reached"] = "seconds"
            self.close("recordingLimit")
            return False
        if (not isinstance(board, np.ndarray) or board.dtype != np.uint8
                or board.ndim != 3 or board.shape[2] != 3 or min(board.shape[:2]) < 1):
            raise ValueError("Recording needs exact uint8 board pixels")
        if mask is not None and (not isinstance(mask, np.ndarray) or mask.ndim != 2):
            raise ValueError("Recording needs a two-dimensional exclusion mask")
        mask = np.zeros(board.shape[:2], np.uint8) if mask is None else np.where(mask > 0, 255, 0).astype(np.uint8)
        if mask.shape != board.shape[:2]:
            raise ValueError("Recording mask dimensions differ from the board")
        entry = {"index": self.frames, "timestamp": timestamp, "locked": bool(locked),
                 "geometry_key": deepcopy(geometry_key), "frame_index": frame_index,
                 "metadata": deepcopy(metadata or {})}
        decision_line = canonical(record) + "\n"
        packets = []
        for name, pixels in (("board", board), ("mask", mask)):
            ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                raise OSError(f"Could not encode observer {name}")
            data = encoded.tobytes()
            relative = f"inputs/{self.frames:06d}-{name}.png"
            entry[name] = relative
            entry[name + "_sha256"] = hashlib.sha256(data).hexdigest()
            packets.append((relative, data))
        input_line = canonical(entry) + "\n"
        packet_bytes = sum(len(data) for _path, data in packets) + len(input_line.encode()) + len(decision_line.encode())
        if self.bytes + packet_bytes > self.max_bytes:
            self.manifest["limit_reached"] = "bytes"
            self.close("recordingLimit")
            return False
        for relative, data in packets:
            (self.folder / relative).write_bytes(data)
        self._inputs.write(input_line)
        self._decisions.write(decision_line)
        self._inputs.flush()
        self._decisions.flush()
        self.bytes += packet_bytes
        self.frames += 1
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        self.last_timestamp = timestamp
        return True

    def close(self, reason="userStopped", *, error=None):
        if self.closed:
            return
        self.closed = True
        self._inputs.close()
        self._decisions.close()
        self.manifest.update(frames=self.frames, bytes=self.bytes, end_reason=reason,
                             complete=error is None and self.frames > 0, error=str(error) if error else None,
                             sources_unchanged=fingerprints() == self.manifest["code_sha256"])
        _save_json(self.folder / "manifest.json", self.manifest)


def _read_pixels(folder, entry, name):
    relative = Path(entry[name])
    path = (folder / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(folder.resolve()):
        raise ValueError("Observer input must stay in its recording folder")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry[name + "_sha256"]:
        raise ValueError(f"Observer {name} input fingerprint differs")
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR if name == "board" else cv2.IMREAD_GRAYSCALE)
    if pixels is None:
        raise ValueError(f"Observer {name} image cannot be decoded")
    return pixels


def _resolve_detector(initialization):
    """Select a recorded implementation without importing arbitrary names.

    Early journals did not name their detector and always used RingDetector.
    Keeping that historical default is necessary for honest comparisons when
    ``--allow-changed-code`` is explicitly requested.
    """
    description = initialization.get("detector")
    if description is None:
        from poc_ring_detector import RingDetector
        return RingDetector
    if not isinstance(description, dict):
        raise ValueError("Observer detector metadata must name an allowed detector")
    name = (description.get("module"), description.get("class"))
    if name == ("poc_ring_detector", "RingDetector"):
        from poc_ring_detector import RingDetector
        return RingDetector
    if name == ("annular_ring_detector", "AnnularRingDetector"):
        from annular_ring_detector import AnnularRingDetector
        return AnnularRingDetector
    if name == ("hybrid_ring_detector", "HybridRingDetector"):
        from hybrid_ring_detector import HybridRingDetector
        return HybridRingDetector
    raise ValueError("Observer recording names an unsupported detector implementation")


class _PreviewWriter:
    """Causal output-only rendering, sampled from the source capture clock."""

    fps = 10.0

    def __init__(self, folder, geometry):
        self.folder, self.geometry = folder, geometry
        self.writer = None
        self.first_timestamp = None
        self.next_time = 0.0
        self.last_frame = None
        self.frames = 0

    def push(self, board, record, timestamp):
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        now = timestamp - self.first_timestamp
        # The preview's footer is relative to this recording. The measured
        # decision and its absolute source-clock values remain untouched.
        display_record = {**record, "capture_seconds": now}
        display_board = board
        if board.shape[:2] != (self.geometry.height, self.geometry.width):
            # A held geometry-change frame still belongs in the replay. Keep
            # the video dimensions fixed, but suppress positions in the old
            # coordinate system instead of drawing them on remapped pixels.
            scale = min(self.geometry.width / board.shape[1], self.geometry.height / board.shape[0])
            width, height = max(1, round(board.shape[1] * scale)), max(1, round(board.shape[0] * scale))
            display_board = np.zeros((self.geometry.height, self.geometry.width, 3), np.uint8)
            left, top = (self.geometry.width - width) // 2, (self.geometry.height - height) // 2
            display_board[top:top + height, left:left + width] = cv2.resize(board, (width, height))
            display_record = {**display_record, "positions": {}, "candidates": []}
        annotated = draw_observer_frame(display_board, display_record, self.geometry)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.folder / "annotated.pending.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                (annotated.shape[1], annotated.shape[0]))
            if not self.writer.isOpened():
                self.writer.release()
                self.writer = None
                raise OSError("Could not open observer preview encoder")
        while self.next_time < now - 1e-8:
            self.writer.write(self.last_frame if self.last_frame is not None else annotated)
            self.frames += 1
            self.next_time += 1 / self.fps
        self.last_frame = annotated

    def close(self, *, success):
        temporary = self.folder / "annotated.pending.mp4"
        if self.writer is not None:
            if success and self.last_frame is not None:
                self.writer.write(self.last_frame)
                self.frames += 1
            self.writer.release()
            self.writer = None
        if success and temporary.exists():
            temporary.replace(self.folder / "annotated.mp4")
            return {"path": "annotated.mp4", "fps": self.fps, "frames": self.frames,
                    "encoded_duration_seconds": self.frames / self.fps,
                    "clock_basis": "source capture timestamps, relative to first recorded frame",
                    "timing_scope": "encoding/rendering excluded from inference timing"}
        if temporary.exists():
            temporary.unlink()
        return None


def _write_transcripts(folder, events, first_timestamp, report):
    with (folder / "movements.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sequence", "mini", "from", "to", "capture_seconds",
                         "source_capture_timestamp", "boundary_uncertain"))
        for event in events:
            destination = event.get("to_cell") or "/".join(event.get("destination_cells") or []) or "uncertain"
            writer.writerow((event["sequence"], event["mini"], event.get("from_cell") or "?",
                             destination, round(event["capture_seconds"] - first_timestamp, 6),
                             event["capture_seconds"], event.get("boundary_uncertain", True)))
    lines = ["SARWEEN OBSERVER MOVEMENT TRANSCRIPT", "",
             "Times are relative to the first recorded capture frame.",
             "Automatic observations; this transcript is not independent accuracy validation.",
             "Foundry token movement was OFF.",
             f"Replay verification passed: {report['verification_passed']}", ""]
    for event in events:
        elapsed = event["capture_seconds"] - first_timestamp
        minutes, seconds = divmod(elapsed, 60)
        destination = event.get("to_cell") or "/".join(event.get("destination_cells") or []) or "unknown"
        suffix = " (uncertain destination)" if event.get("boundary_uncertain") else ""
        lines.append(f"{int(minutes):02d}:{seconds:06.3f}  {event['sequence']}. {event['mini']}: "
                     f"{event.get('from_cell') or '?'} -> {destination}{suffix}")
    if not events:
        lines.append("No movement events were emitted in this recorded interval.")
    (folder / "transcript.txt").write_text("\n".join(lines) + "\n")


def replay_journal(folder, *, require_same_code=True, write_preview=False):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("kind") != "sarween-observer-journal":
        raise ValueError("Choose an observer input recording")
    current_code = fingerprints()
    loaded_code_matches_disk = current_code == LOADED_ENGINE_FINGERPRINTS
    same_code = loaded_code_matches_disk and current_code == manifest["code_sha256"]
    if require_same_code and (not same_code or manifest.get("sources_unchanged") is not True):
        raise ValueError("Observer source fingerprints differ from the recording")
    initial = manifest["initialization"]
    watermark = initial.get("watermark")
    if watermark is not None and (not isinstance(watermark, dict)
                                  or watermark.get("mode") not in ("provided", "board-template-v1")):
        raise ValueError("Unsupported observer watermark mode")
    engine = ObserverEngine(BoardGeometry(initial["view"], initial["warp_size"]),
                            initial["initial_positions"], initial["initial_cells"],
                            geometry_key=initial.get("geometry_key"),
                            detector_factory=_resolve_detector(initial),
                            watermark_mode=watermark["mode"] if watermark is not None else "provided")
    inputs = [json.loads(line) for line in (folder / "inputs.jsonl").read_text().splitlines()]
    decisions = [json.loads(line) for line in (folder / "decisions.jsonl").read_text().splitlines()]
    timestamps = [float(entry["timestamp"]) for entry in inputs]
    if (not all(math.isfinite(value) for value in timestamps)
            or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError("Observer input capture timestamps must be finite and strictly increasing")
    duration = timestamps[-1] - timestamps[0] if timestamps else 0.0
    time_limit = float(manifest["limits"]["seconds"])
    if not math.isfinite(time_limit) or time_limit <= 0 or duration > time_limit:
        raise ValueError("Observer input timestamps exceed the declared recording interval")
    discrepancies, elapsed = [], []
    preview = _PreviewWriter(folder, engine.geometry) if write_preview else None
    preview_metadata = None
    try:
        for index, entry in enumerate(inputs):
            if entry["index"] != index:
                raise ValueError("Observer inputs must retain their original order")
            board, mask = _read_pixels(folder, entry, "board"), _read_pixels(folder, entry, "mask")
            started = time.perf_counter()
            record = engine.process(board, mask, entry["timestamp"], locked=entry["locked"],
                                    geometry_key=entry.get("geometry_key"), frame_index=entry.get("frame_index"))
            elapsed.append((time.perf_counter() - started) * 1000)
            if index >= len(decisions) or canonical(record) != canonical(decisions[index]):
                discrepancies.append(index)
            if preview:
                preview.push(board, record, entry["timestamp"])
    except BaseException:
        if preview:
            preview.close(success=False)
        raise
    if preview:
        preview_metadata = preview.close(success=True)
    count_matches = len(inputs) == len(decisions) == manifest.get("frames")
    identical = bool(inputs) and count_matches and not discrepancies
    complete = manifest.get("complete") is True
    sources_unchanged = manifest.get("sources_unchanged") is True
    report = {"kind": "observer-replay-verification", "frames": len(inputs),
              "recording_complete": complete,
              "code_matches": same_code, "frame_counts_match": count_matches,
              "loaded_code_matches_disk": loaded_code_matches_disk,
              "sources_unchanged": sources_unchanged,
              "identical_decisions": identical,
              "verification_passed": bool(identical and complete and same_code and sources_unchanged),
              "discrepant_frames": discrepancies, "events": engine.events,
              "capture_duration_seconds": duration,
              "source_clock_origin": timestamps[0] if timestamps else None,
              "preview": preview_metadata,
              "median_processing_ms": float(np.median(elapsed)) if elapsed else None,
              "scope": "recorded observer interval; not independent tracking accuracy",
              "foundry_token_output": False}
    _save_json(folder / "replay-verification.json", report)
    _write_transcripts(folder, engine.events, timestamps[0] if timestamps else 0, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--allow-changed-code", action="store_true",
                        help="Compare a changed engine, without claiming identical-version reproduction")
    parser.add_argument("--preview", action="store_true",
                        help="Export an annotated replay at source capture-clock speed")
    args = parser.parse_args()
    result = replay_journal(args.recording, require_same_code=not args.allow_changed_code,
                            write_preview=args.preview)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["verification_passed"] else 1)
