"""Compare an explicit detector on exact saved observer inputs, without labels.

Review labels may be read only after the entire observation pass completes.
This is an offline development tool, never a camera or Foundry controller.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from observer_engine import BoardGeometry, ObserverEngine
from observer_journal import _read_pixels, fingerprints
from annular_ring_detector import AnnularRingDetector
from poc_ring_detector import RingDetector
from hybrid_ring_detector import HybridRingDetector


def run(source, output, detector_name, watermark_mode="provided"):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    initial = manifest["initialization"]
    entries = [json.loads(line) for line in (source / "inputs.jsonl").read_text().splitlines()]
    code = fingerprints()
    factory = {"annular": AnnularRingDetector, "legacy": RingDetector,
               "hybrid": HybridRingDetector}[detector_name]
    engine = ObserverEngine(BoardGeometry(initial["view"], initial["warp_size"]),
        initial["initial_positions"], initial["initial_cells"],
        geometry_key=initial.get("geometry_key"), detector_factory=factory,
        watermark_mode=watermark_mode)
    first = None
    reasons, visible = Counter(), Counter()
    durations = []
    cv2.setNumThreads(0)
    with (output / "decisions.jsonl").open("w") as stream:
        for index, entry in enumerate(entries):
            board, mask = _read_pixels(source, entry, "board"), _read_pixels(source, entry, "mask")
            started = time.perf_counter()
            row = engine.process(board, mask, entry["timestamp"], frame_index=entry["frame_index"],
                locked=entry["locked"], geometry_key=entry["geometry_key"])
            durations.append((time.perf_counter() - started) * 1000)
            reasons[row["state"]["reason"]] += 1
            visible[row["acquisition"]["visible_count"]] += 1
            if row["acquired"] and first is None:
                first = {"index": index, "frame": entry["frame_index"],
                         "capture_seconds": entry["timestamp"],
                         "elapsed": entry["timestamp"] - entries[0]["timestamp"]}
            stream.write(json.dumps(row) + "\n")
            if index % 150 == 0:
                print(json.dumps({"processed": index, "acquired": row["acquired"],
                                  "visible": row["acquisition"]["visible_count"],
                                  "event_count": len(engine.events)}), flush=True)
    if code != fingerprints():
        raise RuntimeError("Tracker sources changed during candidate evaluation")
    report = {"source": str(source), "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
              "detector": detector_name, "frames": len(entries),
              "watermark_mode": watermark_mode,
              "recording_complete": manifest.get("complete"),
              "scope": "saved-input candidate comparison; accuracy requires separate review",
              "first_acquired": first, "state_counts": dict(reasons), "visible_counts": dict(visible),
              "events": engine.events, "final_positions": engine.final_positions(),
              "median_processing_ms": float(np.median(durations)),
              "p95_processing_ms": float(np.percentile(durations, 95)), "code_sha256": code}
    (output / "observation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--detector", choices=("annular", "legacy", "hybrid"), required=True)
    parser.add_argument("--watermark-mode", choices=("provided", "board-template-v1"), default="provided")
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output, args.detector, args.watermark_mode), indent=2))
