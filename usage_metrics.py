"""Small, bounded usage counters. Never collects gameplay content or credentials."""

from datetime import datetime, timezone
import json
import threading
import time
import uuid

from app_paths import atomic_write_bytes


COUNTERS = ("app_seconds", "tracking_seconds", "foundry_seconds", "tracking_sessions",
            "replay_opens", "demo_opens", "foundry_connections", "live_errors")
VERSION = "0.2.0-metrics1"


class UsageMetrics:
    def __init__(self, path, *, wall=time.time, monotonic=time.monotonic):
        self.path, self.wall, self.monotonic = path, wall, monotonic
        self.lock = threading.RLock()
        self.user_id = None
        self.run_id = str(uuid.uuid4())
        self.rows = {}
        self.tracking = self.foundry = False
        self.last_tick = monotonic()
        self.previous_day = self._day()
        # Loaded only on the worker that constructs the authentication adapter.
        try:
            if path.exists() and path.stat().st_size <= 262144:
                value = json.loads(path.read_text())
                uuid.UUID(value["user_id"])
                self.user_id = value["user_id"]
                for row in value["reports"][:128]:
                    uuid.UUID(row["run_id"])
                    if set(row) != set(COUNTERS) | {"run_id", "activity_date", "app_version"}:
                        continue
                    if any(type(row[k]) is not int or not 0 <= row[k] <= (86400 if k.endswith("seconds") else 1000) for k in COUNTERS):
                        continue
                    if row["tracking_seconds"] > row["app_seconds"] or row["foundry_seconds"] > row["app_seconds"]:
                        continue
                    if row["app_version"] != VERSION:
                        continue
                    self.rows[(row["run_id"], row["activity_date"])] = row
                self._prune()
        except Exception:
            self.user_id, self.rows = None, {}

    def _day(self):
        return datetime.fromtimestamp(self.wall(), timezone.utc).date().isoformat()

    def _prune(self):
        today = datetime.fromtimestamp(self.wall(), timezone.utc).date()
        valid = {}
        for key, row in self.rows.items():
            try:
                age = (today - datetime.fromisoformat(row["activity_date"]).date()).days
                if 0 <= age <= 30:
                    valid[key] = row
            except (ValueError, TypeError):
                continue
        self.rows = dict(sorted(valid.items(), key=lambda item: item[0][1])[-128:])

    def bind(self, user_id):
        with self.lock:
            if user_id != self.user_id:
                # Never transfer pending counters to a different account.
                self.rows = {}
                self.user_id = user_id
                self.run_id = str(uuid.uuid4())
                self.tracking = self.foundry = False
                self.last_tick = self.monotonic()

    def _row(self):
        key = (self.run_id, self._day())
        if key not in self.rows:
            self.rows[key] = {"run_id": key[0], "activity_date": key[1],
                "app_version": VERSION, **dict.fromkeys(COUNTERS, 0)}
        return self.rows[key]

    def tick(self, allowed):
        with self.lock:
            now, day = self.monotonic(), self._day()
            elapsed = now - self.last_tick
            self.last_tick = now
            # Skip suspend/blocked UI gaps, rather than count a sleeping Mac as play.
            if allowed and self.user_id:
                row = self._row()
                if 0 <= elapsed <= 5 and day == self.previous_day:
                    row["app_seconds"] = min(86400, row["app_seconds"] + elapsed)
                    for mode in ("tracking", "foundry"):
                        if getattr(self, mode):
                            row[mode + "_seconds"] = min(row["app_seconds"], row[mode + "_seconds"] + elapsed)
            self.previous_day = day

    def state(self, mode, active):
        if mode not in ("tracking", "foundry"):
            return
        with self.lock:
            if not self.user_id:
                return
            if active and not getattr(self, mode):
                self.event("tracking_sessions" if mode == "tracking" else "foundry_connections")
            setattr(self, mode, bool(active))

    def event(self, name):
        if name not in COUNTERS or name.endswith("seconds"):
            return
        with self.lock:
            if self.user_id:
                row = self._row()
                row[name] = min(1000, row[name] + 1)

    def snapshot(self):
        with self.lock:
            self._prune()
            return [{**row, **{k: int(row[k]) for k in COUNTERS}}
                    for row in list(self.rows.values())[:32]]

    def acknowledge(self, sent):
        with self.lock:
            for row in sent:
                key = (row["run_id"], row["activity_date"])
                current = self.rows.get(key)
                # Retain the live run's cumulative totals and any counters that
                # changed while the network request was in flight.
                if current and key != (self.run_id, self._day()) and all(int(current[k]) == row[k] for k in COUNTERS):
                    del self.rows[key]

    def persist(self):
        with self.lock:
            self._prune()
            payload = {"user_id": self.user_id, "reports": [
                {**r, **{k: int(r[k]) for k in COUNTERS}} for r in self.rows.values()]}
            atomic_write_bytes(self.path, json.dumps(payload).encode(), backup=False)
