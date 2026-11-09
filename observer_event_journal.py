"""Durable accepted-move history, independent of bounded camera recordings.

One controller owns each run file. JSONL records are append-only, flushed and
fsynced. A failed write remains pending; retries rescan IDs before appending so
a completed write followed by a failed fsync cannot duplicate a movement.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import os


SCHEMA_VERSION = 1


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def read_event_journal(path):
    """Read valid unique records and report fragments; never hide damage.

    An interrupted fragment is retained on disk. A writer may append a newline
    and retry its pending record; the invalid fragment stays visible as a warning.
    """
    path = Path(path)
    data = path.read_bytes() if path.exists() else b""
    records, warnings, seen = [], [], {}
    for number, raw in enumerate(data.splitlines(keepends=True), 1):
        if not raw.endswith(b"\n"):
            warnings.append(f"Unterminated record at line {number}")
        try:
            value = json.loads(raw)
            if (not isinstance(value, dict) or value.get("schemaVersion") != SCHEMA_VERSION
                    or not isinstance(value.get("recordId"), str)
                    or type(value.get("recordSequence")) is not int):
                raise ValueError("Invalid event record")
            canonical = _canonical(value)
        except (ValueError, UnicodeError, TypeError):
            warnings.append(f"Invalid record retained at line {number}")
            continue
        identity = value["recordId"]
        if identity in seen:
            if seen[identity] != canonical:
                raise ValueError("Conflicting observer event journal record IDs")
            warnings.append(f"Duplicate record ID at line {number}")
            continue
        seen[identity] = canonical
        records.append(value)
    return {"records": records, "warnings": warnings,
            "needs_separator": bool(data and not data.endswith(b"\n"))}


class ObserverEventJournal:
    def __init__(self, path, run_id, *, recent_limit=100, foundry_token_output=False):
        if not isinstance(run_id, str) or not run_id or len(run_id) > 256:
            raise ValueError("Provide an observer run ID")
        if type(recent_limit) is not int or recent_limit < 1:
            raise ValueError("Recent event limit must be positive")
        self.path, self.run_id = Path(path), run_id
        self.foundry_token_output = bool(foundry_token_output)
        self._recent = deque(maxlen=recent_limit)
        self._pending = deque()
        self._accepted = {}
        self._durable = set()
        self._record_sequence = self._move_sequence = self._generation = 0
        self._current_generation = None
        self._needs_separator = self._rescan = False
        self._close_requested = self._closed = False
        self.error = ""
        self.export_error = ""
        export_base = self.path.name.removesuffix(".jsonl")
        self.text_path = self.path.with_name(export_base + ".txt")
        self.csv_path = self.path.with_name(export_base + ".csv")
        self._export_attempt = self._exported = None
        self._warnings = []
        existing = read_event_journal(self.path)
        self._recover(existing)
        if existing["records"]:
            self._closed = existing["records"][-1].get("type") == "run_closed"
            self._close_requested = self._closed
            if not self._closed:
                self._enqueue("run_resumed")
        else:
            self._enqueue("run_started", foundry_token_output=self.foundry_token_output)
        self.flush()

    def _recover(self, existing):
        pending = {record["recordId"]: _canonical(record) for record in self._pending}
        # Validate before changing saved counts or dropping a pending record.
        # This also protects lifecycle records after an uncertain fsync result.
        for record in existing["records"]:
            if record.get("runId") != self.run_id:
                raise ValueError("Observer event file belongs to another run")
            identity = record["recordId"]
            if identity in pending and pending[identity] != _canonical(record):
                raise ValueError("Conflicting on-disk observer event record")
            if record.get("type") == "move" and identity in self._accepted:
                digest = hashlib.sha256(_canonical(record["event"]).encode()).hexdigest()
                if digest != self._accepted[identity]:
                    raise ValueError("Conflicting on-disk accepted movement")
        self._needs_separator = existing["needs_separator"]
        self._warnings = list(dict.fromkeys([*self._warnings, *existing["warnings"]]))[-20:]
        self._durable = set()
        for record in existing["records"]:
            if record.get("runId") != self.run_id:
                raise ValueError("Observer event file belongs to another run")
            self._durable.add(record["recordId"])
            self._record_sequence = max(self._record_sequence, record["recordSequence"])
            self._generation = max(self._generation, record.get("generation", 0))
            if record.get("type") == "move":
                identity = record["recordId"]
                digest = hashlib.sha256(_canonical(record["event"]).encode()).hexdigest()
                if identity not in self._accepted:
                    self._accepted[identity] = digest
                    self._recent.append(record)
                self._move_sequence = max(self._move_sequence, record["movementSequence"])

    def _enqueue(self, kind, *, record_id=None, **fields):
        self._record_sequence += 1
        record = {"schemaVersion": SCHEMA_VERSION, "runId": self.run_id,
                  "recordId": record_id or f"{self.run_id}:record:{self._record_sequence}",
                  "recordSequence": self._record_sequence, "type": kind,
                  "createdAt": datetime.now(timezone.utc).isoformat(), **deepcopy(fields)}
        _canonical(record)
        self._pending.append(record)
        return record

    def begin_generation(self, initialization, *, reason):
        if self._close_requested:
            raise RuntimeError("Observer event journal has ended")
        self._generation += 1
        self._current_generation = self._generation
        self._enqueue("generation_started", generation=self._generation, reason=reason,
                      initialization=initialization)
        self.flush()
        return self._generation

    def end_generation(self, reason):
        if self._current_generation is not None and not self._close_requested:
            self._enqueue("generation_ended", generation=self._current_generation, reason=reason)
            self._current_generation = None
            self.flush()

    def append_events(self, generation, events, *, reason, recording_folder=None):
        if self._close_requested:
            raise RuntimeError("Observer event journal has ended")
        for event in events:
            sequence = event.get("sequence")
            if type(sequence) is not int or sequence < 1:
                raise ValueError("Accepted observer movement needs its engine sequence")
            identity = f"{self.run_id}:generation:{generation}:event:{sequence}"
            digest = hashlib.sha256(_canonical(event).encode()).hexdigest()
            if identity in self._accepted:
                if self._accepted[identity] != digest:
                    raise ValueError("Conflicting accepted movement for an observer event ID")
                continue
            if generation != self._current_generation:
                raise ValueError("Movement belongs to an inactive observer generation")
            self._move_sequence += 1
            record = self._enqueue("move", record_id=identity, generation=generation,
                movementSequence=self._move_sequence, event=event, reason=reason,
                recordingFolder=recording_folder,
                recordingCoverage="not asserted; consult the input recording manifest",
                foundry_token_output=self.foundry_token_output)
            self._accepted[identity] = digest
            self._recent.append(record)
        return self.flush()

    def _append_record(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = (("\n" if self._needs_separator else "") + _canonical(record) + "\n").encode()
        with self.path.open("ab") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        self._needs_separator = False

    def flush(self):
        try:
            if self._rescan or self._needs_separator:
                existing = read_event_journal(self.path)
                # A prior fsync may have failed after the full line was written.
                if self.path.exists():
                    with self.path.open("ab") as stream:
                        if existing["needs_separator"]:
                            stream.write(b"\n")
                            stream.flush()
                        os.fsync(stream.fileno())
                self._recover(read_event_journal(self.path))
                self._rescan = False
            while self._pending:
                record = self._pending[0]
                if record["recordId"] not in self._durable:
                    self._append_record(record)
                    self._durable.add(record["recordId"])
                self._pending.popleft()
            self.error = ""
            self._closed = self._close_requested
            signature = (sum(identity in self._durable for identity in self._accepted), self._closed)
            if signature != self._export_attempt and (signature[0] or self._closed):
                self.export_saved_moves()
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.error = f"Movement history is not fully saved: {exc}"
            self._rescan = True
            return False

    def close(self):
        if not self._close_requested:
            self.end_generation("observer_closed")
            self._enqueue("run_closed", acceptedMoveCount=len(self._accepted))
            self._close_requested = True
        return self.flush()

    def recent_events(self):
        return [{**deepcopy(record), "saved": record["recordId"] in self._durable}
                for record in self._recent]

    def _write_export(self, path, text):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def export_saved_moves(self):
        """Regenerate derived files from saved entries; JSONL remains authoritative.

        This is called only for newly saved moves or close, not for each camera
        frame. Explicit callers may retry a failed export without changing JSONL.
        """
        saved_count = sum(identity in self._durable for identity in self._accepted)
        self._export_attempt = (saved_count, self._closed)
        try:
            parsed = read_event_journal(self.path)
            moves = [record for record in parsed["records"]
                     if record["type"] == "move" and record["recordId"] in self._durable]
            for record in moves:
                digest = hashlib.sha256(_canonical(record["event"]).encode()).hexdigest()
                if digest != self._accepted[record["recordId"]]:
                    raise ValueError("Saved movement differs from accepted event")
            if len(moves) != saved_count:
                raise ValueError("Saved movement count differs from the journal")
            fields = ("run_id", "movement_sequence", "event_id", "generation", "engine_sequence",
                      "created_at_utc", "capture_seconds", "frame", "mini", "from_cell", "to_cell",
                      "destination_cells", "boundary_uncertain", "from_xy", "to_xy", "uncertainty_px", "reason")
            csv_text = io.StringIO(newline="")
            writer = csv.DictWriter(csv_text, fieldnames=fields)
            writer.writeheader()
            text = ["SARWEEN — SAVED ACCEPTED MOVEMENTS", f"Run: {self.run_id}",
                    f"Authoritative JSONL: {self.path.resolve()}",
                    f"Saved moves: {saved_count}; run closed: {self._closed}",
                    "These are software-accepted events, not an exhaustive list of physical moves.",
                    "Missed moves are not reconstructed. Camera-recording coverage is not asserted.",
                    "created_at_utc is event acceptance UTC. capture_seconds uses the original",
                    "source clock; it is NOT a video-relative timestamp."]
            if parsed["warnings"]:
                text.append("Journal warnings: " + "; ".join(parsed["warnings"]))
            text.append("")
            for record in moves:
                event = record["event"]
                row = {key: event.get(key) for key in fields if key in event}
                row.update(run_id=record["runId"], movement_sequence=record["movementSequence"],
                           event_id=record["recordId"], generation=record["generation"],
                           engine_sequence=event["sequence"], created_at_utc=record["createdAt"],
                           reason=record.get("reason"))
                for key in ("destination_cells", "from_xy", "to_xy"):
                    row[key] = _canonical(event.get(key))
                writer.writerow(row)
                destination = event.get("to_cell") or "/".join(event.get("destination_cells") or []) or "unknown"
                if event.get("boundary_uncertain"):
                    destination += " (uncertain)"
                text.extend([
                    f"{record['movementSequence']}. {event.get('mini', 'unknown')}: {event.get('from_cell') or 'unknown'} -> {destination}",
                    f"   event_id={record['recordId']}; generation={record['generation']}; engine_sequence={event['sequence']}",
                    f"   created_at_utc={record['createdAt']}; capture_seconds={event.get('capture_seconds')}; frame={event.get('frame')}",
                    f"   reason={record.get('reason')}; from_xy={row['from_xy']}; to_xy={row['to_xy']}; uncertainty_px={event.get('uncertainty_px')}",
                    "",
                ])
            self._write_export(self.text_path, "\n".join(text) + "\n")
            self._write_export(self.csv_path, csv_text.getvalue())
            self._exported = self._export_attempt
            self.export_error = ""
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.export_error = f"TXT/CSV export could not be updated; durable JSONL is unchanged: {exc}"
            return False

    def status(self):
        return {"runId": self.run_id, "path": str(self.path.resolve()),
                "acceptedMoveCount": len(self._accepted),
                "savedMoveCount": sum(identity in self._durable for identity in self._accepted),
                "pendingRecords": len(self._pending),
                "pendingMoves": sum(record["type"] == "move" for record in self._pending),
                "generation": self._current_generation, "closed": self._closed,
                "error": self.error, "integrityWarnings": list(self._warnings),
                "exportError": self.export_error,
                "exports": {"textPath": str(self.text_path.resolve()), "csvPath": str(self.csv_path.resolve()),
                            "savedMoveCount": self._exported[0] if self._exported is not None else None,
                            "runClosed": self._exported[1] if self._exported is not None else None,
                            "current": self._exported == (sum(identity in self._durable for identity in self._accepted), self._closed)},
                "recentLimit": self._recent.maxlen,
                "scope": "accepted observer moves; independent of camera recording limits"}
