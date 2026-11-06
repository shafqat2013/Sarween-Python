"""File-only Phase 1 observer. No camera, GUI, relay, or live tracking imports.

The detector receives each current image and a caption exclusion mask. The
association step receives only current measurements and declared initial
bindings. A separate scoring module is called after the entire replay ends.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from observer_engine import (BoardGeometry, ObserverEngine, draw_observer_frame,
                             masked_anchors, motion_guard)
from watermark_detection import ObsWatermarkDetector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(video, initial_path):
    video, initial_path = Path(video).expanduser().resolve(), Path(initial_path).resolve()
    if not video.is_file():
        raise ValueError("Provide an existing video file; camera devices are unsupported")
    initial = json.loads(initial_path.read_text())
    if initial.get("kind") != "declared-initial-bindings" or "expected_moves" in initial:
        raise ValueError("Initialization must contain starting bindings only")
    source = initial["source"]
    if video.name != source["video_basename"] or sha256(video) != source["video_sha256"]:
        raise ValueError("Video does not match the declared initialization")
    timeline_path = video.parent / source["timeline_basename"]
    if sha256(timeline_path) != source["timeline_sha256"]:
        raise ValueError("Timeline fingerprint differs from reviewed geometry/clocks")
    data = json.loads(timeline_path.read_text())
    if data.get("framesUndistorted") is not True:
        raise ValueError("Phase 1 requires recordings already corrected for the camera lens")
    views = [event["viewTransform"] for event in data["events"] if event.get("viewTransform")]
    if not views or any(view != views[0] for view in views):
        raise ValueError("Phase 1 supports one fixed recorded viewport only")
    checkpoint = (video.parent / source["checkpoint"]["path"]).resolve()
    if not checkpoint.is_relative_to(video.parent):
        raise ValueError("Checkpoint must stay alongside the recording")
    if sha256(checkpoint) != source["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint fingerprint does not match")
    with np.load(checkpoint, allow_pickle=False) as arrays:
        H = arrays["H_saved"].copy()
    if H.shape != (3, 3) or not np.isfinite(H).all():
        raise ValueError("Missing finite camera-to-board calibration")
    clocks = {int(item["frame"]): float(item["captureTime"]) for item in data["frameClocks"]}
    times = [clocks[index] for index in range(int(data["framesRecorded"]))]
    if not times or not np.isfinite(times).all() or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Complete, increasing capture clocks are required")
    return video, initial, BoardGeometry(views[0], initial["warp_size"]), H, np.array(times) - times[0]


def observe(video, initial_path, output, *, frame_limit=None, write_preview=True):
    code_files = ("single_mover_observer.py", "observer_engine.py", "single_mover_state.py", "poc_ring_detector.py",
                  "poc_scene_reference.py", "watermark_detection.py")
    code_fingerprints = {name: sha256(Path(__file__).with_name(name)) for name in code_files}
    initial_fingerprint = sha256(initial_path)
    video, initial, geometry, H, times = load_inputs(video, initial_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Could not decode the saved video")
    caption = ObsWatermarkDetector()
    engine = ObserverEngine(geometry, initial["initial_positions"], initial["initial_cells"])
    events, processing_ms, reasons = [], [], Counter()
    available_time = 0.0
    waiting_ms = []
    writer, last_preview = None, None
    next_preview_time = 0.0
    preview_fps = 10.0
    encoded_fps = cap.get(cv2.CAP_PROP_FPS)
    consumed = 0
    try:
        with (output / "frames.jsonl").open("w") as frame_log:
            for index, now in enumerate(times):
                if frame_limit is not None and index >= frame_limit:
                    break
                ok, raw = cap.read()
                if not ok:
                    raise ValueError(f"Video ended before declared frame {index}")
                started = time.perf_counter()
                board = cv2.warpPerspective(raw, H, (geometry.width, geometry.height))
                watermark = caption.detect(raw)
                mask = cv2.warpPerspective(watermark.exclusion_mask, H,
                                          (geometry.width, geometry.height), flags=cv2.INTER_NEAREST)
                record = engine.process(board, mask, float(now), frame_index=index)
                elapsed = (time.perf_counter() - started) * 1000
                processing_ms.append(elapsed)
                waiting_ms.append(max(0.0, available_time - float(now)) * 1000)
                available_time = max(available_time, float(now)) + elapsed / 1000
                for item in record["events"]:
                    events.append({**item, "processing_ms": elapsed,
                                   "simulated_available_capture_seconds": available_time})
                reasons[record["state"]["reason"]] += 1
                frame_log.write(json.dumps({**record, "watermark_detected": watermark.detected,
                    "processing_ms": elapsed, "simulated_waiting_ms": waiting_ms[-1]}) + "\n")
                if write_preview:
                    annotated = draw_observer_frame(board, record, geometry)
                    if writer is None:
                        writer = cv2.VideoWriter(str(output / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                 preview_fps, (geometry.width, geometry.height + 64))
                        if not writer.isOpened():
                            raise ValueError("Annotated video encoder unavailable")
                    while next_preview_time < now - 1e-8:
                        writer.write(last_preview if last_preview is not None else annotated)
                        next_preview_time += 1 / preview_fps
                    last_preview = annotated
                consumed += 1
    finally:
        cap.release()
        if writer is not None:
            if last_preview is not None:
                writer.write(last_preview)
            writer.release()
    if any(sha256(Path(__file__).with_name(name)) != value for name, value in code_fingerprints.items()):
        raise RuntimeError("Prototype sources changed during replay; rerun with frozen code")
    if sha256(video) != initial["source"]["video_sha256"]:
        raise RuntimeError("Source video changed during replay")
    if sha256(initial_path) != initial_fingerprint:
        raise RuntimeError("Initial bindings changed during replay")
    final = engine.final_positions()
    duration = float(times[consumed - 1]) if consumed else 0
    stats = {"median_ms": float(np.median(processing_ms)),
             "p95_ms": float(np.percentile(processing_ms, 95)),
             "max_ms": float(max(processing_ms)),
             "capture_mean_interval_ms": float(np.mean(np.diff(times)) * 1000),
             "simulated_max_waiting_backlog_ms": max(waiting_ms),
             "simulated_final_waiting_backlog_ms": waiting_ms[-1],
             "queue_simulation_scope": "inference only; excludes decode/render/encode and live I/O"} if processing_ms else {}
    report = {"schema_version": 1, "kind": "offline-single-mover-observation", "engine": "shared-observer-engine",
              "source": initial["source"], "initial_bindings_file": str(Path(initial_path).resolve()),
              "initial_bindings_sha256": initial_fingerprint, "expected_frames": len(times),
              "initialization": initial["initialization_source"],
              "frames_consumed": consumed, "complete": consumed == len(times),
              "capture_duration_seconds": duration, "encoded_input_fps": encoded_fps,
              "clock_basis": "recorded frameClocks.captureTime; no encoded-FPS timing",
              "causal": True, "events": events, "final_positions": final,
              "processing": stats, "state_reason_counts": dict(reasons),
              "runtime": {"opencv": cv2.__version__, "numpy": np.__version__},
              "code_sha256": code_fingerprints,
              "scope": {"development_clip": True, "llm_used_for_measurement": False,
                        "live_token_commands": False, "fog_validated": False,
                        "unseen_recording_validated": False},
              "limitations": ["Identity/position bindings supplied once from the first frame.",
                  "The shared detector is being developed on this recording.",
                  "Sub-0.45-grid relocations are inside the stationary-anchor gate.",
                  "Geometry and camera pose are held fixed; no scene pan/zoom recovery.",
                  "Shared color/shape plus settled-image reference assumes a substantially static background.",
                  "Global motion gating does not distinguish hands from changing map content.",
                  "Inference time excludes decoding, rendering, encoding and actual live camera/display latency."]}
    (output / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output / "movements.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "mini", "from", "to", "capture_seconds", "boundary_uncertain"])
        for event in events:
            writer.writerow([event["sequence"], event["mini"], event["from_cell"],
                             "/".join(event["destination_cells"]), round(event["capture_seconds"], 3),
                             event["boundary_uncertain"]])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--review", type=Path, help="Scoring labels, loaded only after replay finishes")
    parser.add_argument("--frame-limit", type=int)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    cv2.setNumThreads(0)
    report = observe(args.video, args.initial, args.out,
                     frame_limit=args.frame_limit, write_preview=not args.no_preview)
    if args.review:
        from single_mover_evaluation import evaluate_observer
        review = json.loads(args.review.read_text())
        evaluation = evaluate_observer(report, review)
        evaluation["review_sha256"] = sha256(args.review)
        evaluation["evaluator_sha256"] = sha256(Path(__file__).with_name("single_mover_evaluation.py"))
        (args.out / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
    print(json.dumps({"frames": report["frames_consumed"], "moves": report["events"],
                      "processing": report["processing"], "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
