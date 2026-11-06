"""Saved-video audit data and provenance, without Tk, CV, cameras or relay IO."""

import copy
import hashlib
import json
import math
from pathlib import Path

from app_paths import atomic_write_bytes, atomic_write_json, data_dir
from movement_transcript import export_rows, validate_label
from replay_review import ReviewSession
from tracking_evaluation import normalize_cell, parse_time_seconds


ROOT = Path(__file__).resolve().parent
SOURCES_FILE = ROOT / "audit_sources.json"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
EXCLUDED_FOLDERS = {"demo", "build", "dist", "deprecated", ".git", ".venv", "node_modules"}
_hash_cache = {}


def discover_videos(folders=None):
    """List real source videos directly in the chosen folders, newest names first."""
    if folders is None:
        folders = (ROOT, data_dir() / "Recordings")
    elif isinstance(folders, (str, Path)):
        folders = (folders,)
    found = set()
    for folder in folders:
        folder = Path(folder).expanduser().resolve()
        if not folder.is_dir() or any(part.lower() in EXCLUDED_FOLDERS for part in folder.parts):
            continue
        for path in folder.iterdir():
            if path.suffix.lower() in VIDEO_SUFFIXES and path.is_file():
                resolved = path.resolve()
                if not any(part.lower() in EXCLUDED_FOLDERS for part in resolved.parts):
                    found.add(resolved)
    return sorted(found, key=lambda path: (path.name.lower(), str(path)), reverse=True)


def _json(path, description):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read {description} {Path(path).name}: {exc}") from exc


def _manifest():
    if not SOURCES_FILE.exists():
        return {"sources": [], "unavailable": []}
    value = _json(SOURCES_FILE, "audit source manifest")
    if (not isinstance(value, dict) or value.get("schemaVersion") != 1
            or not isinstance(value.get("sources"), list)
            or not isinstance(value.get("unavailable", []), list)):
        raise ValueError("Invalid audit source manifest")
    return value


def _fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _hash_cache:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        _hash_cache[key] = digest.hexdigest()
    return _hash_cache[key]


def _source_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else SOURCES_FILE.parent / path


def _unavailable_reason(video, manifest):
    for entry in manifest.get("unavailable", []):
        if not isinstance(entry, dict) or entry.get("video") != video.name:
            continue
        if entry.get("videoSha256") and entry["videoSha256"] != _fingerprint(video):
            continue  # A repaired/replaced recording must not inherit the old flag.
        return str(entry.get("reason") or "This recording cannot be decoded")
    return ""


def _review_path(video, folder=None):
    stat = video.stat()
    identity = {"path": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return Path(folder or data_dir() / "Reviews") / key / "review.json"


def review_status(video, folder=None):
    """Cheap queue status: never load transcripts, decode video or create reviews."""
    video = Path(video).expanduser().resolve()
    try:
        if _unavailable_reason(video, _manifest()):
            return "Unavailable"
        path = _review_path(video, folder)
        if path.exists():
            value = _json(path, "saved review")
            if not isinstance(value, dict) or not isinstance(value.get("complete", False), bool):
                return "Unavailable"
            if value.get("complete"):
                return "Reviewed"
    except (OSError, ValueError):
        return "Unavailable"
    return "To review"


def _events(value):
    if not isinstance(value, list):
        raise ValueError("Tracking transcript must contain an events list")
    result = []
    for index, event in enumerate(value):
        if (not isinstance(event, dict) or "time_seconds" not in event
                or event["time_seconds"] is None or isinstance(event["time_seconds"], bool)):
            raise ValueError(f"Tracking transcript event {index + 1} is invalid")
        row = copy.deepcopy(event)
        row["mini"] = str(row.get("mini") or "").strip()
        row["time_seconds"] = parse_time_seconds(row["time_seconds"])
        row["from_cell"] = normalize_cell(row.get("from_cell", row.get("raw_from")))
        row["to_cell"] = normalize_cell(row.get("to_cell", row.get("raw_to")))
        if not row["mini"] or not row["to_cell"]:
            raise ValueError(f"Tracking transcript event {index + 1} needs a mini and destination")
        row.setdefault("source", "detection")
        result.append(row)
    return result


def _curated(video, manifest):
    for entry in manifest["sources"]:
        if not isinstance(entry, dict) or entry.get("video") != video.name:
            continue
        if not isinstance(entry.get("path"), str) or not isinstance(entry.get("label"), str):
            raise ValueError(f"Invalid audit source entry for {video.name}")
        path = _source_path(entry["path"]).resolve()
        value = _json(path, "tracking transcript")
        if isinstance(value, dict) and "cases" in value:
            if not isinstance(value["cases"], list):
                raise ValueError(f"Invalid case list in {path.name}")
            cases = [case for case in value["cases"] if isinstance(case, dict)
                     and Path(str(case.get("video", ""))).name == entry.get("caseVideo", video.name)]
            if len(cases) != 1:
                raise ValueError(f"Expected one full-video transcript for {video.name} in {path.name}")
            value = cases[0]
        if isinstance(value, list):
            value = {"events": value, "diagnostics": {}}
        if not isinstance(value, dict):
            raise ValueError(f"Invalid tracking transcript in {path.name}")
        diagnostics = copy.deepcopy(value.get("diagnostics") or {})
        if not isinstance(diagnostics, dict):
            raise ValueError(f"Invalid transcript diagnostics in {path.name}")
        inputs = _json(_source_path(entry["inputs"]), "analysis provenance") if entry.get("inputs") else {}
        if not isinstance(inputs, dict):
            raise ValueError("Analysis provenance must be an object")
        diagnostic_inputs = diagnostics.get("inputs") or {}
        if not isinstance(diagnostic_inputs, dict) or not isinstance(diagnostic_inputs.get("video", {}), dict):
            raise ValueError(f"Invalid video provenance in {path.name}")
        expected = (inputs.get("videoSha256") or diagnostic_inputs.get("video", {}).get("sha256")
                    or entry.get("videoSha256"))
        if not expected or expected != _fingerprint(video):
            raise ValueError(f"Tracking transcript video fingerprint does not match {video.name}")
        events = _events(value.get("events"))
        note = str(entry.get("note") or "Saved replay predictions; confirm against the video.")
        if inputs.get("initialReseedCounterfactual"):
            note += " The background was refreshed for this replay; these were not live detections."
        source = {"label": entry["label"], "note": note, "origin": "Replay", "path": str(path),
                  "sha256": _fingerprint(path), "videoSha256": expected, "inputs": inputs}
        run_key = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:24]
        return {"events": events, "diagnostics": diagnostics}, source, "audit-" + run_key
    return None


class AuditRecord:
    def __init__(self, video, folder=None):
        self.video = Path(video).expanduser().resolve()
        if not self.video.is_file():
            raise ValueError(f"Video does not exist: {self.video}")
        self.title = self.video.name
        self.review = ReviewSession(self.video, folder=folder)
        data = self.review.data
        if (not isinstance(data, dict) or not isinstance(data.get("labels"), list)
                or not isinstance(data.get("decisions"), dict) or not isinstance(data.get("complete"), bool)):
            raise ValueError(f"Invalid saved review: {self.review.path}")
        self._number(data.get("position_seconds", 0))
        if not isinstance(data.get("notes", ""), str):
            raise ValueError("Saved review notes must be text")
        manifest = _manifest()
        self.unavailable_reason = _unavailable_reason(self.video, manifest)
        if self.unavailable_reason:
            self.source_label, self.source_note, self._origin = "Unavailable", self.unavailable_reason, "Unavailable"
            return
        if data.get("analysis") is not None:
            analysis = data["analysis"]
            if not isinstance(analysis, dict):
                raise ValueError("Invalid saved analysis")
            _events(analysis.get("events"))
            source = data.get("audit_source") or {}
            self.source_label = source.get("label") or "Replay · saved analysis"
            self.source_note = source.get("note") or "Previously saved replay predictions; confirm against the video."
            self._origin = "Replay"
        else:
            imported = _curated(self.video, manifest)
            if imported:
                analysis, source, run_id = imported
                # Freeze the chosen predictions without accepting a new run or
                # clearing prior labels, decisions, completion or progress.
                data.update(analysis=analysis, audit_source=source, run_id=run_id)
                self.source_label, self.source_note, self._origin = source["label"], source["note"], "Replay"
            elif self.review.timeline:
                self.source_label, self._origin = "Live · recorded tracking", "Live"
                self.source_note = "Recorded software predictions, not confirmed physical movements."
            else:
                self.source_label, self._origin = "No saved transcript", "Unavailable"
                self.source_note = "No trusted saved tracking transcript is available. You can add observed movements manually."

    @staticmethod
    def _number(value):
        if isinstance(value, bool):
            raise ValueError("Video position must be a finite nonnegative number")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("Video position must be a finite nonnegative number") from None
        if not math.isfinite(value) or value < 0:
            raise ValueError("Video position must be a finite nonnegative number")
        return value

    @property
    def position_seconds(self):
        return self._number(self.review.data.get("position_seconds", 0))

    @property
    def complete(self):
        return False if self.unavailable_reason else bool(self.review.data["complete"])

    @property
    def notes(self):
        return self.review.data.get("notes", "")

    def _predictions(self):
        if self.unavailable_reason:
            return []
        result = []
        for row in self.review.rows("Detected"):
            clean = _events([row])[0]
            result.append(clean)
        return result

    @property
    def rows(self):
        predictions = self._predictions()
        linked = {"review-" + row["id"] for row in predictions}
        rows = []
        for row in predictions:
            decision = row.get("decision")
            rows.append({**row, "kind": "prediction", "origin": self._origin,
                "status": "Confirmed" if decision == "confirmed" else "Wrong" if decision == "rejected" else "Unreviewed",
                "notes": row.get("notes", ""), "source_label": self.source_label})
        for label in self.review.rows("Reviewed"):
            if label["id"] not in linked:
                clean = validate_label(label)
                rows.append({**clean, "kind": "manual", "origin": "Manual", "status": "Confirmed",
                             "notes": clean.get("notes", ""), "source_label": "Manual video review"})
        return sorted(rows, key=lambda row: (row["time_seconds"], row["id"]))

    def _commit(self, data):
        atomic_write_json(self.review.path, data)
        self.review.data = data

    def restore(self, snapshot):
        """Atomically restore this video's review without regenerating exports.

        The controller keeps playhead and notes current before calling this for
        Undo. Never accept a snapshot belonging to another saved media file.
        """
        data = copy.deepcopy(snapshot)
        if (not isinstance(data, dict) or data.get("schemaVersion") != 1
                or data.get("video") != self.review.identity
                or not isinstance(data.get("labels"), list)
                or not isinstance(data.get("decisions"), dict)
                or not isinstance(data.get("complete"), bool)):
            raise ValueError("Invalid review snapshot for this video")
        if self.unavailable_reason and data["complete"]:
            raise ValueError("An unavailable recording cannot be marked reviewed")
        self._number(data.get("position_seconds", 0))
        if not isinstance(data.get("notes", ""), str):
            raise ValueError("Saved review notes must be text")
        for label in data["labels"]:
            if not isinstance(label, dict) or not isinstance(label.get("id"), str) or not label["id"]:
                raise ValueError("Invalid saved movement label")
            validate_label(label)
        if any(not isinstance(key, str) or not key or decision not in ("confirmed", "rejected")
               for key, decision in data["decisions"].items()):
            raise ValueError("Invalid saved movement decision")
        if data.get("analysis") is not None:
            if not isinstance(data["analysis"], dict):
                raise ValueError("Invalid saved analysis")
            _events(data["analysis"].get("events"))
        self._commit(data)

    def reset_decision(self, row_id):
        if not any(row["id"] == row_id for row in self._predictions()):
            raise ValueError("Choose a predicted movement to reset")
        data = copy.deepcopy(self.review.data)
        data["decisions"].pop(row_id, None)
        data["labels"] = [label for label in data["labels"] if label["id"] != "review-" + row_id]
        data["complete"] = False
        self._commit(data)

    def decide(self, row_id, accepted, duration, grid=None):
        if not isinstance(accepted, bool):
            raise ValueError("Choose Confirmed or Wrong")
        row = next((row for row in self._predictions() if row["id"] == row_id), None)
        if row is None:
            raise ValueError("Choose a predicted movement to confirm or mark wrong")
        data = copy.deepcopy(self.review.data)
        linked = "review-" + row_id
        data["labels"] = [label for label in data["labels"] if label["id"] != linked]
        if accepted:
            label = validate_label({**row, "id": linked, "provenance": "confirmed-prediction"}, duration=duration, grid=grid)
            data["labels"].append(label)
        data["decisions"][row_id] = "confirmed" if accepted else "rejected"
        data["complete"] = False
        self._commit(data)

    def add_missed(self, seconds, mini, to_cell, from_cell=None, *, duration, grid=None, notes=""):
        if not isinstance(notes, str):
            raise ValueError("Movement notes must be text")
        label = validate_label({"mini": mini, "time_seconds": seconds, "from_cell": from_cell,
            "to_cell": to_cell, "notes": notes, "provenance": "manual-video-review", "source": "detection"},
            duration=duration, grid=grid)
        data = copy.deepcopy(self.review.data)
        data["labels"].append(label)
        data["complete"] = False
        self._commit(data)
        return label

    def remove_label(self, row_id):
        data = copy.deepcopy(self.review.data)
        labels = [label for label in data["labels"] if label["id"] != row_id]
        if len(labels) == len(data["labels"]):
            raise ValueError("Choose a manually added movement to remove")
        data["labels"] = labels
        if row_id.startswith("review-"):
            data["decisions"].pop(row_id[len("review-"):], None)
        data["complete"] = False
        self._commit(data)

    def set_position(self, seconds):
        seconds = self._number(seconds)
        if seconds != self.position_seconds:
            data = copy.deepcopy(self.review.data)
            data["position_seconds"] = seconds
            self._commit(data)

    def mark_complete(self, complete):
        if not isinstance(complete, bool):
            raise ValueError("Review completion must be on or off")
        if complete and self.unavailable_reason:
            raise ValueError("An unavailable recording cannot be marked reviewed")
        data = copy.deepcopy(self.review.data)
        data["complete"] = complete
        self._commit(data)

    def save_notes(self, notes):
        if not isinstance(notes, str):
            raise ValueError("Review notes must be text")
        data = copy.deepcopy(self.review.data)
        data["notes"] = notes
        self._commit(data)

    def export(self, folder):
        base = Path(folder) / (self.video.stem + ".audit")
        rows = [{**row, "track": row["source_label"], "decision": row["status"].lower()} for row in self.rows]
        export_rows(rows, base, include_notes=True)
        output = {"schemaVersion": 1, "video": str(self.video), "movements": rows,
            "transcript_source": {"label": self.source_label, "note": self.source_note},
            "audit_source": copy.deepcopy(self.review.data.get("audit_source")),
            "video_identity": copy.deepcopy(self.review.identity),
            "notes": self.notes, "complete": self.complete, "position_seconds": self.position_seconds}
        atomic_write_json(str(base) + ".json", output, backup=False)
        text_path = Path(str(base) + ".txt")
        header = self.source_label + "\n" + self.source_note + "\n"
        if self.notes:
            header += "\nVideo notes:\n" + self.notes + "\n"
        atomic_write_bytes(text_path, (header + "\n" + text_path.read_text(encoding="utf-8")).encode(), backup=False)
        return base
