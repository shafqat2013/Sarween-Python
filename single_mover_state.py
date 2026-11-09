"""Causal association for the offline, one-moving-piece proof of concept.

This module knows neither ring colors nor the expected movement sequence. Its
inputs are accepted base-center measurements, a capture timestamp, and explicit
visibility/obstruction information. Confirmed positions are preserved until an
unambiguous departure and a stable replacement measurement establish one move.
Coordinates remain continuous; the caller decides how to represent grid borders.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections import deque
from collections.abc import Mapping
from statistics import median


TIMING_POLICIES = ("fixed-gap-v1", "bounded-cadence-v1")


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be finite") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class _Candidate:
    xy: tuple[float, float]
    confidence: float
    uncertainty: float


class SingleMover:
    """Associate indistinguishable bases using their last confirmed layout.

    ``unavailable`` names anchors whose measurement region is masked (for
    example, by the video caption). Those anchors are held and excluded from
    the departure count. It does not mean the piece has disappeared. A latched
    mover becoming unavailable interrupts confirmation until it is observable.

    Low confidence measurements are ignored. Nearby duplicate measurements are
    collapsed before association, and an observation that can match two anchors
    is considered ambiguous. Missing identities remain unresolved until their
    old anchor is positively observed or a move is confirmed; a caption cannot
    turn a previously ambiguous disappearance into a unique mover. A bounded
    anonymous destination track can survive temporary stationary-base dropouts,
    but never supplies identity by proximity to an arbitrary new coordinate.
    """

    def __init__(self, initial_positions: dict[str, tuple[float, float]],
                 grid_px: float, settle_seconds: float = 0.20, *,
                 timing_policy: str = "bounded-cadence-v1"):
        self.grid_px = _finite(grid_px, "Grid spacing")
        self.settle_seconds = _finite(settle_seconds, "Settling duration")
        if self.grid_px <= 0 or self.settle_seconds < 0:
            raise ValueError("Grid spacing must be positive and settling duration nonnegative")
        if timing_policy not in TIMING_POLICIES:
            raise ValueError("Unsupported observer timing policy")
        self.timing_policy = timing_policy
        if not isinstance(initial_positions, Mapping) or not initial_positions:
            raise ValueError("Provide at least one initial identity/position binding")
        self.positions = {}
        for mini, xy in initial_positions.items():
            if not isinstance(mini, str) or not mini or not isinstance(xy, (tuple, list)) or len(xy) != 2:
                raise ValueError("Initial bindings need nonempty names and two coordinates")
            self.positions[mini] = (_finite(xy[0], "Initial x"), _finite(xy[1], "Initial y"))
        if len(set(self.positions.values())) != len(self.positions):
            raise ValueError("Initial identities cannot occupy the same position")
        self.anchor_gate = self.grid_px * 0.45
        self.move_threshold = self.grid_px * 0.40
        self.duplicate_gate = self.grid_px * 0.18
        self.settle_radius = self.grid_px * 0.15
        # Settling is a minimum observation duration, not an assumed frame
        # rate. Learn regular slower capture cadence without allowing a single
        # outage to increase its own validity window. Legacy recordings retain
        # their original fixed-gap behavior when explicitly replayed that way.
        self.maximum_observation_gap = (max(0.5, self.settle_seconds * 2.5)
                                        if timing_policy == "fixed-gap-v1" else 0.5)
        self._cadence_intervals = deque(maxlen=9)
        self._cadence_gap_cap = 1.0
        self.minimum_confidence = 0.45
        self.last_seen = {mini: None for mini in self.positions}
        self._last_timestamp = None
        self._missing_since = {}
        self._active_mini = None
        self._pending = None
        self._unresolved = set()
        self.destination_cache_seconds = 1.0
        self.destination_cache_max_samples = 16
        self.status = {"phase": "initialized", "reason": "initial_bindings",
                       "pending": None, "missing": [], "unavailable": [],
                       "unassigned": [], "missing_for": {}, "unresolved_departures": []}

    def _measurements(self, candidates):
        parsed = []
        for item in candidates:
            if not isinstance(item, Mapping):
                raise ValueError("Each candidate must contain base-center coordinates")
            try:
                xy = (_finite(item["cx"], "Candidate x"), _finite(item["cy"], "Candidate y"))
            except KeyError:
                raise ValueError("Each candidate needs cx and cy") from None
            confidence = _finite(item.get("confidence", 1.0), "Candidate confidence")
            uncertainty = _finite(item.get("uncertainty_px", 0.0), "Candidate uncertainty")
            if not 0 <= confidence <= 1 or uncertainty < 0:
                raise ValueError("Confidence must be between zero and one; uncertainty nonnegative")
            if "radius" in item and _finite(item["radius"], "Candidate radius") <= 0:
                raise ValueError("Candidate radius must be positive")
            if confidence >= self.minimum_confidence:
                parsed.append(_Candidate(xy, confidence, uncertainty))
        # Prefer the better supported measurement without assigning the same
        # physical spot twice when inner/outer ring boundaries share a center.
        accepted = []
        for item in sorted(parsed, key=lambda c: (-c.confidence, c.uncertainty, c.xy)):
            if not any(_distance(item.xy, other.xy) <= self.duplicate_gate for other in accepted):
                accepted.append(item)
        return accepted

    def _set_status(self, phase, reason, missing, unavailable, extras, timestamp):
        pending = None
        if self._pending is not None:
            pending = {"mini": self._active_mini if self._unresolved == {self._active_mini} else None,
                       "xy": self._pending["reference"],
                       "first_visible_time": self._pending["started"],
                       "samples": len(self._pending["samples"]),
                       "last_visible_time": self._pending["last_visible"],
                       "identity_hypotheses": sorted(self._unresolved)}
        self.status = {
            "phase": phase, "reason": reason,
            "pending": pending, "active_mini": self._active_mini,
            "missing": list(missing), "unavailable": sorted(unavailable),
            "unassigned": [item.xy for item in extras],
            "missing_for": {mini: max(0.0, timestamp - self._missing_since[mini])
                            for mini in missing if mini in self._missing_since},
            "unresolved_departures": sorted(self._unresolved),
        }
        if self.timing_policy == "bounded-cadence-v1":
            self.status["timing"] = {
                "policy": self.timing_policy,
                "maximum_observation_gap": self.maximum_observation_gap,
                "cadence_seconds": (float(median(self._cadence_intervals))
                                    if len(self._cadence_intervals) >= 3 else None),
            }

    def _observe_capture_time(self, now):
        if self.timing_policy == "bounded-cadence-v1":
            # Only earlier intervals influence this frame's limit. In
            # particular, the first delayed frame cannot excuse its own gap.
            self.maximum_observation_gap = (max(0.5, min(self._cadence_gap_cap,
                1.5 * median(self._cadence_intervals))) if len(self._cadence_intervals) >= 3 else 0.5)
        gap = None if self._last_timestamp is None else now - self._last_timestamp
        if gap is not None and gap > self.maximum_observation_gap:
            self._pending = None
        if (self.timing_policy == "bounded-cadence-v1" and gap is not None
                and 0 < gap <= self._cadence_gap_cap):
            self._cadence_intervals.append(gap)
        self._last_timestamp = now

    def _observe_destination(self, extras, now):
        """Cache fresh geometry without assigning an identity while ambiguous."""
        if len(extras) != 1:
            self._pending = None
            return
        item = extras[0]
        if (self._pending is None
                or now - self._pending["last_visible"] > self.maximum_observation_gap
                or _distance(item.xy, self._pending["reference"]) > self.settle_radius):
            self._pending = {"reference": item.xy, "started": now,
                             "last_visible": now, "samples": [(now, item)]}
            return
        if now > self._pending["last_visible"]:
            samples = [(timestamp, sample) for timestamp, sample in self._pending["samples"]
                       if now - timestamp <= self.destination_cache_seconds]
            samples.append((now, item))
            self._pending["samples"] = samples[-self.destination_cache_max_samples:]
            self._pending["last_visible"] = now

    def update(self, candidates: list[dict], timestamp: float, *,
               obstructed: bool = False, unavailable: set[str] | None = None) -> list[dict]:
        """Consume only this frame's evidence and emit zero or one confirmed move.

        A backward or non-finite clock and malformed measurements are rejected
        before changing any state. Repeated timestamps cannot advance settling.
        A destination gap/miss, competing destination, obstruction, or large
        candidate jump starts a fresh settling interval. Multiple missing IDs
        preserve only freshly measured destination geometry, not an identity.
        """
        now = _finite(timestamp, "Capture timestamp")
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("Capture timestamps must be monotonic")
        hidden = set() if unavailable is None else set(unavailable)
        if not hidden.issubset(self.positions):
            raise ValueError("Unavailable identities must belong to the initial layout")
        measured = self._measurements(candidates)
        self._observe_capture_time(now)
        if obstructed:
            self._pending = None
            # During a broad unobserved interval, no identity may be removed
            # from the possible departures merely because its origin is later
            # caption-covered. Fresh old-anchor measurements resolve this set.
            self._unresolved.update(self.positions)
            self._set_status("uncertain", "obstructed", [], hidden, [], now)
            return []

        if hidden == set(self.positions):
            self._unresolved.update(hidden)

        # A partially masked anchor must not manufacture a new destination.
        measured = [item for item in measured if not any(
            _distance(item.xy, self.positions[mini]) <= self.anchor_gate for mini in hidden)]
        matches = {}
        used = set()
        ambiguous = False
        ambiguous_anchors = set()
        for index, item in enumerate(measured):
            nearby = [mini for mini, xy in self.positions.items()
                      if mini not in hidden and _distance(item.xy, xy) <= self.anchor_gate]
            if len(nearby) > 1:
                ambiguous = True
                ambiguous_anchors.update(nearby)
            elif nearby:
                mini = nearby[0]
                # More than one distinct plausible base at an old anchor is
                # also a detector ambiguity, not a reason to move that anchor.
                if mini in matches:
                    ambiguous = True
                    ambiguous_anchors.add(mini)
                matches[mini] = index
                used.add(index)
        missing = [mini for mini in self.positions if mini not in matches and mini not in hidden]
        extras = [item for index, item in enumerate(measured) if index not in used]
        for mini in missing:
            self._missing_since.setdefault(mini, now)
        self._unresolved.update(missing)
        for mini in matches:
            if mini not in ambiguous_anchors:
                self.last_seen[mini] = now
                self._missing_since.pop(mini, None)
                self._unresolved.discard(mini)

        if ambiguous:
            self._unresolved.update(ambiguous_anchors)
            for mini in ambiguous_anchors:
                self._missing_since.setdefault(mini, now)
            self._pending = None
            self._set_status("uncertain", "ambiguous_anchor_match", missing, hidden, extras, now)
            return []
        if self._active_mini in hidden:
            self._pending = None
            self._set_status("uncertain", "active_mini_unavailable", missing, hidden, extras, now)
            return []
        if self._active_mini in matches:
            self._active_mini = None

        # A stable new spot may be observed before identity becomes unique.
        # Retain only its recent real measurements through transient dropouts;
        # masks never delete another unresolved identity from this decision.
        self._observe_destination(extras, now)
        if len(self._unresolved) != 1:
            if not self._unresolved:
                self._active_mini = None
                self._pending = None
                reason = "layout_confirmed" if not extras else "extra_candidates_with_full_layout"
                self._set_status("settled", reason, missing, hidden, extras, now)
            else:
                reason = "multiple_missing_anchors" if len(missing) > 1 else "unresolved_departure_history"
                self._set_status("uncertain", reason, missing, hidden, extras, now)
            return []

        mover, = self._unresolved
        if mover in hidden:
            self._pending = None
            self._set_status("uncertain", "active_mini_unavailable", missing, hidden, extras, now)
            return []
        if self._active_mini is None:
            self._active_mini = mover
        elif self._active_mini != mover:
            self._pending = None
            self._set_status("uncertain", "conflicting_departure", missing, hidden, extras, now)
            return []
        if len(extras) != 1:
            self._pending = None
            reason = "destination_not_visible" if not extras else "competing_destinations"
            self._set_status("moving" if not extras else "uncertain", reason, missing, hidden, extras, now)
            return []

        item = extras[0]
        origin = self.positions[mover]
        if _distance(item.xy, origin) < self.move_threshold:
            self._pending = None
            self._set_status("uncertain", "insufficient_displacement", missing, hidden, extras, now)
            return []
        if (now - self._pending["started"] + 1e-9 < self.settle_seconds
                or (self.timing_policy == "bounded-cadence-v1" and len(self._pending["samples"]) < 2)):
            self._set_status("settling", "one_mover_settling", missing, hidden, extras, now)
            return []

        samples = [sample for _timestamp, sample in self._pending["samples"]]
        total = sum(sample.confidence for sample in samples)
        destination = tuple(sum(sample.xy[axis] * sample.confidence for sample in samples) / total
                            for axis in (0, 1))
        uncertainty = max(max(sample.uncertainty for sample in samples),
                          max(_distance(sample.xy, destination) for sample in samples))
        event = {"mini": mover, "from_xy": origin, "to_xy": destination,
                 "departure_time": self._missing_since.get(mover, now),
                 "first_visible_time": self._pending["started"], "confirmed_time": now,
                 "uncertainty_px": uncertainty, "samples": len(samples)}
        self.positions[mover] = destination
        self.last_seen[mover] = now
        self._missing_since.pop(mover, None)
        self._unresolved.discard(mover)
        self._pending = None
        self._active_mini = None
        self._set_status("confirmed", "single_move_confirmed", [], hidden, [], now)
        return [event]
