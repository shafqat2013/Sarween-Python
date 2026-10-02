"""Shared per-frame tracking decisions, independent of UI, video IO and Foundry."""

from collections import Counter, deque
from dataclasses import dataclass, field
from dataclasses import asdict
import copy
import math
from typing import Any, Callable

from tap_selection import TapGestureDetector, contact_minis


@dataclass(frozen=True)
class TrackingContext:
    now: float
    scene_key: tuple
    view_revision: int
    grid_px: float | None
    to_cell: Callable[[float, float], tuple[int, int] | None]
    selected_mini: str | None = None
    paused: bool = False


@dataclass(frozen=True)
class TrackingMove:
    mini: str
    raw_from: str | None
    raw_to: str
    source: str = "detection"
    lab_dist: float = 0.0
    score: float = 1.0


@dataclass
class TrackingStep:
    detections: dict = field(default_factory=dict)
    moves: list[TrackingMove] = field(default_factory=list)
    lost_minis: list[str] = field(default_factory=list)
    tapped_mini: str | None = None
    reason: str | None = None


class TrackingEngine:
    def __init__(self, detector, *, consensus_n=6, consensus_k=4,
                 lost_timeout=2.0, selected_lost_timeout=0.75):
        if not 1 <= consensus_k <= consensus_n:
            raise ValueError("Consensus requires 1 <= k <= n")
        if not all(math.isfinite(value) and value > 0 for value in (lost_timeout, selected_lost_timeout)):
            raise ValueError("Lost-mini timeouts must be finite and positive")
        self.detector = detector
        self.consensus_n, self.consensus_k = consensus_n, consensus_k
        self.lost_timeout, self.selected_lost_timeout = lost_timeout, selected_lost_timeout
        self.prev_state: dict[str, Any] = {}
        self.cell_hist: dict[str, deque] = {}
        self.last_emitted: dict[str, str] = {}
        self.last_physical_xy: dict[str, tuple] = {}
        self.last_seen: dict[str, float] = {}
        self.tap_detector = TapGestureDetector()
        self._context_key = None
        self._view_revision = None
        self._last_time = None

    def reset(self):
        self.prev_state.clear()
        self.cell_hist.clear()
        self.last_emitted.clear()
        self.last_physical_xy.clear()
        self.last_seen.clear()
        self.tap_detector = TapGestureDetector()

    def reset_mini(self, mini):
        for state in (self.prev_state, self.cell_hist, self.last_emitted,
                      self.last_physical_xy, self.last_seen):
            state.pop(mini, None)
        self.tap_detector = TapGestureDetector()

    def snapshot(self):
        from recording_state import json_value
        return json_value({
            "prev_state": self.prev_state, "cell_hist": {key: list(value) for key, value in self.cell_hist.items()},
            "last_emitted": self.last_emitted, "last_physical_xy": self.last_physical_xy,
            "last_seen": self.last_seen, "context_key": self._context_key,
            "view_revision": self._view_revision, "last_time": self._last_time,
            "tap_candidate": asdict(self.tap_detector.candidate) if self.tap_detector.candidate else None,
            "tap_cooldown": self.tap_detector.cooldown_until,
        })

    def restore(self, state):
        from tap_selection import _TapCandidate
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, list) else value
        self.reset()
        for name in ("prev_state", "last_emitted", "last_physical_xy", "last_seen"):
            getattr(self, name).update(copy.deepcopy(state.get(name, {})))
        self.cell_hist.update({key: deque(value, maxlen=self.consensus_n)
                               for key, value in state.get("cell_hist", {}).items()})
        self._context_key = tuples(state.get("context_key"))
        self._view_revision = state.get("view_revision")
        self._last_time = state.get("last_time")
        self.tap_detector.cooldown_until = state.get("tap_cooldown", 0)
        if state.get("tap_candidate"):
            self.tap_detector.candidate = _TapCandidate(**state["tap_candidate"])

    def synchronize(self, context):
        key = (context.scene_key, context.paused)
        if self._context_key is not None and key != self._context_key:
            self.reset()
        if key != self._context_key:
            self._view_revision = context.view_revision
        self._context_key = key

    def seed_positions(self, cells, bundle, *, now=0.0):
        for mini, raw in cells.items():
            row, col = (int(value) for value in raw[1:].split("c"))
            self.last_emitted[mini] = raw
            self.prev_state[mini] = {"last_xy": self._anchor(bundle, col, row), "last_dist": None}
            self.last_seen[mini] = now

    @staticmethod
    def _anchor(bundle, col, row, detection=None):
        if getattr(bundle, "marker_mode", "legacy") == "viewport":
            return (detection.cx, detection.cy) if detection is not None else None
        return ((col + .5) * bundle.warp_w / bundle.grid_w,
                (row + .5) * bundle.warp_h / bundle.grid_h)

    def step(self, bundle, profiles, context, *, verbose=False):
        if not math.isfinite(context.now) or (self._last_time is not None and context.now < self._last_time):
            raise ValueError("Tracking clock must be finite and monotonic")
        self._last_time = context.now
        self.synchronize(context)
        result = TrackingStep()
        if context.paused:
            result.reason = "paused"
            return result
        if not bundle.locked or bundle.warp_bgr is None:
            self.tap_detector = TapGestureDetector()
            result.reason = "unlocked"
            return result
        if context.grid_px is None:
            self.tap_detector = TapGestureDetector()
            result.reason = "unmapped"
            return result

        if context.view_revision != self._view_revision:
            self._view_revision = context.view_revision
            self.tap_detector = TapGestureDetector()
            for mini, point in self.last_physical_xy.items():
                mapped = context.to_cell(*point)
                if mapped is None:
                    continue
                col, row = mapped
                raw = f"r{row}c{col}"
                previous = self.last_emitted.get(mini)
                if raw != previous:
                    self.last_emitted[mini] = raw
                    self.cell_hist.pop(mini, None)
                    result.moves.append(TrackingMove(mini, previous, raw, "viewportTransform"))

        for mini, state in self.prev_state.items():
            timeout = self.selected_lost_timeout if mini == context.selected_mini else self.lost_timeout
            if state.get("last_xy") is not None and context.now - self.last_seen.get(mini, context.now) > timeout:
                state["last_xy"] = None
                self.cell_hist.pop(mini, None)
                result.lost_minis.append(mini)

        detections, state = self.detector(bundle=bundle, profiles=profiles, grid_px=context.grid_px,
                                          prev_state=self.prev_state, verbose=verbose)
        if state is not self.prev_state:
            self.prev_state.clear()
            self.prev_state.update(state)
        result.detections = detections
        for mini, detection in detections.items():
            if detection is not None:
                self.last_seen[mini] = context.now

        contacts = contact_minis(bundle, self.last_physical_xy, context.grid_px)
        detected_positions = {name: (det.cx, det.cy) for name, det in detections.items() if det is not None}
        result.tapped_mini = self.tap_detector.update(
            context.now, contacts, self.last_physical_xy if contacts else detected_positions, context.grid_px)

        for mini, detection in detections.items():
            if detection is None:
                continue
            mapped = context.to_cell(detection.cx, detection.cy)
            if mapped is None:
                continue
            col, row = mapped
            cell = f"r{row}c{col}"
            history = self.cell_hist.setdefault(mini, deque(maxlen=self.consensus_n))
            history.append(cell)
            most, count = Counter(history).most_common(1)[0]
            if verbose:
                print(f"  CONSENSUS | {mini}: {most} ({count}/{len(history)})")
            if count >= self.consensus_k and cell == most:
                self.last_physical_xy[mini] = (detection.cx, detection.cy)
            if count < self.consensus_k or most == self.last_emitted.get(mini):
                continue
            previous = self.last_emitted.get(mini)
            self.last_emitted[mini] = most
            row, col = (int(value) for value in most[1:].split("c"))
            self.prev_state.setdefault(mini, {})["last_xy"] = self._anchor(bundle, col, row, detection)
            result.moves.append(TrackingMove(mini, previous, most, lab_dist=float(detection.lab_dist),
                                             score=float(detection.score)))
        return result
