"""Persistent human annotations, kept separate from each detector run."""

import hashlib
import json
from pathlib import Path

from app_paths import data_dir, atomic_write_json
from movement_transcript import detected_rows, timeline_rows, validate_label, export_rows


class ReviewSession:
    def __init__(self, video, *, folder=None):
        self.video = Path(video).resolve()
        stat = self.video.stat()
        self.identity = {"path": str(self.video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:24]
        self.path = Path(folder or data_dir() / "Reviews") / key / "review.json"
        self.data = {"schemaVersion": 1, "video": self.identity, "labels": [], "decisions": {},
                     "complete": False, "analysis": None, "options": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        sidecar = self.video.with_suffix(".tracking.json")
        self.timeline = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.path, self.data)
        export_rows([row for track in ("Detected", "Reviewed", "Foundry") for row in self.rows(track)],
                    self.path.parent / "movements")

    def accept_analysis(self, result, options):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        import time
        atomic_write_json(self.path.parent / f"analysis-{time.time_ns()}.json", result, backup=False)
        self.data.update(analysis=result, options=options, decisions={}, complete=False, run_id=str(time.time_ns()))
        self.save()

    def rows(self, track):
        if track == "Reviewed":
            return sorted(self.data["labels"], key=lambda row: row["time_seconds"])
        if track == "Foundry":
            return timeline_rows(self.timeline, "foundry")
        result = self.data.get("analysis")
        rows = detected_rows(result["events"]) if result else timeline_rows(self.timeline)
        if result:
            rows = [{**row, "id": self.data.get("run_id", "analysis") + "-" + row["id"]} for row in rows]
        return [{**row, "decision": self.data["decisions"].get(row["id"], row["decision"])} for row in rows]

    def label(self, row, *, duration, grid, replace=None):
        valid = validate_label(row, duration=duration, grid=grid)
        if replace:
            valid["id"] = replace
        self.data["labels"] = [item for item in self.data["labels"] if item["id"] != valid["id"]] + [valid]
        self.data["complete"] = False
        self.save()
        return valid

    def decide(self, row, accepted, *, duration, grid):
        if accepted:
            # Stable linkage prevents duplicate labels on repeated confirmation.
            label = {**row, "id": "review-" + row["id"], "provenance": "confirmed-prediction"}
            self.label(label, duration=duration, grid=grid)
        else:
            self.data["labels"] = [item for item in self.data["labels"] if item["id"] != "review-" + row["id"]]
        self.data["decisions"][row["id"]] = "confirmed" if accepted else "rejected"
        self.data["complete"] = False
        self.save()

    def remove(self, row_id):
        self.data["labels"] = [row for row in self.data["labels"] if row["id"] != row_id]
        self.data["complete"] = False
        self.save()
