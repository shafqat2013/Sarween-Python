"""One causal ring observer shared by Sarween live input and file replay.

Inputs are rectified board pixels, the matching caption exclusion mask and the
capture clock. This module never opens a camera, imports a GUI or sends Foundry
commands. Its static-scene/color assumptions remain experimental; sharing the
engine does not establish that changing map artwork is a physical miniature.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math

import cv2
import numpy as np

from poc_ring_detector import RingDetector
from poc_scene_reference import SceneReference
from single_mover_state import SingleMover

DISPLAY_TRANSITION_POLICIES = ("disabled", "hold-known-display-changes-v1")
_UNKNOWN_REVISION = object()


class BoardGeometry:
    """Transform rectified board coordinates into continuous Foundry cells."""

    def __init__(self, view, warp_size):
        self.view = deepcopy(view)
        self.width, self.height = map(int, warp_size)
        self.registration = self.view["registration"]
        self.transform = self.view["clientToCanvasTransform"]
        self.grid = float(self.view["gridSize"])
        values = [self.grid, *self.registration.values(), *self.transform.values(),
                  self.view.get("gridOriginX", 0), self.view.get("gridOriginY", 0)]
        if (not all(math.isfinite(float(value)) for value in values)
                or self.grid <= 0 or min(self.width, self.height) <= 0):
            raise ValueError("Invalid recorded grid or warp size")
        r, t = self.registration, self.transform
        if r["right"] <= r["left"] or r["bottom"] <= r["top"] or not t["a"] or not t["d"]:
            raise ValueError("Invalid board registration or view transform")
        self.grid_px = min(
            self.grid / abs(t["a"]) * self.width / (r["right"] - r["left"]),
            self.grid / abs(t["d"]) * self.height / (r["bottom"] - r["top"]),
        )

    def grid_point(self, point):
        x, y = map(float, point)
        r, t = self.registration, self.transform
        client_x = r["left"] + x / self.width * (r["right"] - r["left"])
        client_y = r["top"] + y / self.height * (r["bottom"] - r["top"])
        world_x = t["a"] * client_x + t["c"] * client_y + t["tx"]
        world_y = t["b"] * client_x + t["d"] * client_y + t["ty"]
        return ((world_x - self.view.get("gridOriginX", 0)) / self.grid,
                (world_y - self.view.get("gridOriginY", 0)) / self.grid)

    @staticmethod
    def cell(column, row):
        if min(column, row) < 0:
            return None
        name, value = "", column + 1
        while value:
            value, digit = divmod(value - 1, 26)
            name = chr(65 + digit) + name
        return f"{name}{row + 1}"

    def describe(self, point, uncertainty_px=0):
        gx, gy = self.grid_point(point)
        error = max(0.12, float(uncertainty_px) / self.grid_px)
        cells = [self.cell(col, row)
                 for row in range(math.floor(gy - error), math.floor(gy + error) + 1)
                 for col in range(math.floor(gx - error), math.floor(gx + error) + 1)]
        cells = list(dict.fromkeys(cell for cell in cells if cell is not None))
        return {"xy": [round(float(v), 3) for v in point],
                "grid_xy": [round(gx, 4), round(gy, 4)], "cells": cells,
                "cell": cells[0] if len(cells) == 1 else None,
                "boundary_uncertain": len(cells) != 1,
                "uncertainty_grid": round(error, 4)}


def masked_anchors(mask, positions, radius):
    unavailable = set()
    height, width = mask.shape
    for mini, (x, y) in positions.items():
        x0, x1 = max(0, int(x - radius)), min(width, int(x + radius + 1))
        y0, y1 = max(0, int(y - radius)), min(height, int(y + radius + 1))
        patch = mask[y0:y1, x0:x1]
        if patch.size and np.count_nonzero(patch) / patch.size > 0.12:
            unavailable.add(mini)
    return unavailable


def motion_guard(current, previous, mask, previous_mask, grid_px):
    if previous is None:
        return False, 0
    residual = np.max(cv2.absdiff(current, previous), axis=2)
    valid = (mask == 0) & (previous_mask == 0)
    changed = int(np.count_nonzero((residual > 35) & valid))
    return changed > 8 * math.pi * (grid_px * 0.75) ** 2, changed


def _key(value):
    """Copy arbitrary JSON geometry tokens so caller mutation cannot alter them."""
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


class ObserverEngine:
    """Observe a declared layout without side effects outside owned state.

    A lost marker lock, changed geometry key, or changed board size latches a
    hold requiring explicit reset/rebinding. Reacquiring a lock alone does not
    pretend positions retained during an unobserved interval were measured.
    """

    def __init__(self, geometry, initial_positions, initial_cells=None, *, geometry_key=None,
                 detector_factory=None, watermark_mode="provided", display_transition_policy="disabled"):
        self.geometry = geometry
        if watermark_mode not in ("provided", "board-template-v1"):
            raise ValueError("Unsupported observer watermark mode")
        self.watermark_mode = watermark_mode
        if display_transition_policy not in DISPLAY_TRANSITION_POLICIES:
            raise ValueError("Unsupported observer display-transition policy")
        self.display_transition_policy = display_transition_policy
        self._detector_factory = RingDetector if detector_factory is None else detector_factory
        if not callable(self._detector_factory):
            raise ValueError("Observer detector factory must be callable")
        self.reset(initial_positions, initial_cells, geometry_key=geometry_key)

    def reset(self, initial_positions, initial_cells=None, *, geometry=None, geometry_key=None):
        if geometry is not None:
            self.geometry = geometry
        geometry = self.geometry
        mover = SingleMover(initial_positions, geometry.grid_px)
        for xy in mover.positions.values():
            if not (0 <= xy[0] < geometry.width and 0 <= xy[1] < geometry.height):
                raise ValueError("Initial base centers must be inside the board image")
        self.detector = self._detector_factory(geometry.grid_px)
        self._detector_initialized = False
        self.reference = SceneReference(geometry.grid_px)
        self.mover = mover
        self.initial_positions = deepcopy(mover.positions)
        self.confirmed_cells = {mini: geometry.describe(xy)["cell"] for mini, xy in mover.positions.items()}
        if initial_cells is not None:
            if set(initial_cells) != set(mover.positions):
                raise ValueError("Initial cells must describe each initial identity exactly once")
            self.confirmed_cells.update(deepcopy(initial_cells))
        self.initial_cells = deepcopy(self.confirmed_cells)
        self.position_uncertainty = {mini: 2.5 for mini in mover.positions}
        self.geometry_key = deepcopy(geometry_key)
        self._geometry_key = _key(geometry_key)
        self._previous = self._previous_mask = None
        self._caption_detector = None
        self._watermark_evidence = {"mode": self.watermark_mode, "recognized": False}
        self._last_timestamp = None
        self._frame_count = 0
        self._invalid_reason = None
        self._display_baseline_revision = _UNKNOWN_REVISION
        self._display_observed_revision = None
        self.events = []
        self.last_record = None
        # Clicks declare identity and approximate location. They are not visual
        # evidence that the detector can actually see those physical bases.
        self._acquired = False
        self._acquisition_samples = {mini: [] for mini in mover.positions}
        self._acquisition_matches = {}
        self._acquisition_ready = set()
        self.acquisition_settle_seconds = .20
        self.acquisition_window_seconds = 5.0

    @property
    def needs_reset(self):
        return self._invalid_reason is not None

    def initialization(self):
        return {"schema_version": 1, "kind": "observer-engine-initialization",
                "view": deepcopy(self.geometry.view),
                "warp_size": [self.geometry.width, self.geometry.height],
                "initial_positions": deepcopy(self.initial_positions),
                "initial_cells": deepcopy(self.initial_cells),
                "geometry_key": deepcopy(self.geometry_key),
                "detector": {"module": type(self.detector).__module__,
                             "class": type(self.detector).__name__},
                "watermark": {"mode": self.watermark_mode},
                "display_transitions": {"policy": self.display_transition_policy,
                    "notification_is_not_camera_render_synchronization": True,
                    "automatic_recovery": False},
                "acquisition": {"settle_seconds": self.acquisition_settle_seconds,
                                "window_seconds": self.acquisition_window_seconds,
                                "minimum_measurements": 2},
                "assumptions": {"one_mini_moves_at_a_time": True,
                                "static_camera_and_substantially_static_background": True},
                "foundry_token_output": False}

    def final_positions(self):
        return {mini: self.geometry.describe(xy, self.position_uncertainty[mini])
                for mini, xy in self.mover.positions.items()}

    def _anchor_matches(self, candidates, unavailable):
        """Find unique supported measurements, never count one spot twice."""
        measured = self.mover._measurements(candidates)
        nearby = {mini: [] for mini in self.mover.positions if mini not in unavailable}
        ambiguous = set()
        for item in measured:
            matches = [mini for mini in nearby
                       if math.dist(item.xy, self.mover.positions[mini]) <= self.mover.anchor_gate]
            if len(matches) == 1:
                nearby[matches[0]].append(item)
            elif len(matches) > 1:
                ambiguous.update(matches)
        return {mini: items[0] for mini, items in nearby.items()
                if len(items) == 1 and mini not in ambiguous}

    def _acquire(self, candidates, unavailable, obstructed, now):
        """Require recent repeated base evidence before any movement inference.

        A translating caption may prevent seeing all bases simultaneously.
        Each base therefore needs two consistent measurements spanning 0.2 s
        within the last five seconds. A visible miss clears that base's proof;
        a masked base retains only its still-recent proof and remains held in
        the display. Broad obstruction clears all acquisition evidence.
        """
        matches = {} if obstructed else self._anchor_matches(candidates, unavailable)
        self._acquisition_matches = matches
        ready = set()
        for mini in self.mover.positions:
            samples = [sample for sample in self._acquisition_samples[mini]
                       if now - sample[0] <= self.acquisition_window_seconds]
            if obstructed or (mini not in unavailable and mini not in matches):
                samples = []
            elif mini in matches:
                xy = matches[mini].xy
                if any(math.dist(xy, previous_xy) > self.mover.settle_radius
                       for _time, previous_xy in samples):
                    samples = []
                samples.append((now, xy))
            self._acquisition_samples[mini] = samples
            if (len(samples) >= 2
                    and samples[-1][0] - samples[0][0] >= self.acquisition_settle_seconds):
                ready.add(mini)
        self._acquisition_ready = ready
        if len(ready) == len(self.mover.positions):
            self._acquired = True
        return self._acquired

    def _acquisition_status(self):
        names = list(self.mover.positions)
        matches = {} if self.needs_reset else self._acquisition_matches
        ready = set() if self.needs_reset else self._acquisition_ready
        return {"required": len(names), "visible_count": len(matches),
                "ready_count": len(ready),
                "missing": [mini for mini in names if mini not in matches],
                "pending": [mini for mini in names if mini not in ready],
                "settle_seconds": self.acquisition_settle_seconds,
                "window_seconds": self.acquisition_window_seconds,
                "minimum_measurements": 2}

    def _record(self, timestamp, frame_index, candidates, raw_count, unavailable,
                obstructed, changed, emitted, state=None):
        state = deepcopy(state if state is not None else self.mover.status)
        held = set(unavailable) | set(state.get("missing", []))
        held.update(state.get("unresolved_departures", []))
        if state.get("active_mini"):
            held.add(state["active_mini"])
        if obstructed or self.needs_reset or not self._acquired:
            held.update(self.mover.positions)
        acquisition = self._acquisition_status()
        record = {"frame": frame_index, "capture_seconds": timestamp,
                  "candidates": deepcopy(candidates), "raw_candidate_count": raw_count,
                  "state": state, "positions": deepcopy(self.mover.positions),
                  "current_cells": self.final_positions(), "events": deepcopy(emitted),
                  "move_count": len(self.events), "obstructed": bool(obstructed),
                  "changed_pixels": changed, "unavailable": sorted(unavailable),
                  "watermark": deepcopy(self._watermark_evidence),
                  "held": sorted(held), "needs_reset": self.needs_reset,
                  "acquired": self._acquired and not self.needs_reset,
                  "acquisition": acquisition,
                  "acquisition_visible_count": acquisition["visible_count"],
                  "acquisition_required": acquisition["required"],
                  "acquisition_missing": acquisition["missing"],
                  "reference_updates": self.reference.reference_updates,
                  "reference_evidence": deepcopy(self.reference.last_evidence),
                  "foundry_token_output": False}
        if self.display_transition_policy != "disabled":
            record["display_transition"] = {
                "policy": self.display_transition_policy,
                "baseline_revision": (None if self._display_baseline_revision is _UNKNOWN_REVISION
                                      else self._display_baseline_revision),
                "observed_revision": self._display_observed_revision,
                "revision_available": self._display_observed_revision is not None,
                "requires_explicit_rebind": self._invalid_reason in (
                    "display_changed_rebind_required", "display_revision_unavailable"),
                "automatic_recovery": False,
                "scope": "Known Foundry changes only; no fog segmentation or TV timing verification"}
        self.last_record = deepcopy(record)
        return record

    def process(self, board_bgr, excluded_mask, timestamp, *, frame_index=None,
                locked=True, geometry_key=None, scene_visual_revision=None):
        now = float(timestamp)
        if not math.isfinite(now) or (self._last_timestamp is not None and now <= self._last_timestamp):
            raise ValueError("Observer capture timestamps must be finite and strictly increasing")
        if not isinstance(board_bgr, np.ndarray) or board_bgr.dtype != np.uint8 or board_bgr.ndim != 3 or board_bgr.shape[2] != 3:
            raise ValueError("Observer needs a uint8 BGR board image")
        if excluded_mask is None:
            excluded_mask = np.zeros(board_bgr.shape[:2], np.uint8)
        elif not isinstance(excluded_mask, np.ndarray) or excluded_mask.ndim != 2 or excluded_mask.shape != board_bgr.shape[:2]:
            raise ValueError("Caption exclusion mask must match the board image")
        if self.display_transition_policy != "disabled" and scene_visual_revision is not None:
            if type(scene_visual_revision) is not int or not 0 <= scene_visual_revision < 2**63:
                raise ValueError("Scene visual revision must be a nonnegative integer or unavailable")
        mask = np.where(excluded_mask > 0, 255, 0).astype(np.uint8)
        self._watermark_evidence = {"mode": self.watermark_mode, "recognized": False,
                                    "provided_pixels": int(np.count_nonzero(mask))}
        self._last_timestamp = now
        index = self._frame_count if frame_index is None else int(frame_index)
        self._frame_count += 1
        if not locked:
            self._invalid_reason = self._invalid_reason or "camera_lock_lost"
        if _key(geometry_key) != self._geometry_key:
            self._invalid_reason = self._invalid_reason or "geometry_changed"
        if board_bgr.shape[:2] != (self.geometry.height, self.geometry.width):
            self._invalid_reason = self._invalid_reason or "board_size_changed"
        if self.display_transition_policy == "hold-known-display-changes-v1":
            self._display_observed_revision = scene_visual_revision
            if self._display_baseline_revision is _UNKNOWN_REVISION:
                if scene_visual_revision is not None:
                    self._display_baseline_revision = scene_visual_revision
            elif scene_visual_revision is None:
                self._invalid_reason = self._invalid_reason or "display_revision_unavailable"
            elif scene_visual_revision != self._display_baseline_revision:
                # A relay notification is not proof that AirPlay/the camera has
                # seen the new pixels. Never resume just because old pixels are
                # quiet or resemble the old anchors. Explicit rebinding starts
                # a new causal experiment after the user sees the new display.
                self._invalid_reason = self._invalid_reason or "display_changed_rebind_required"
        if self.needs_reset:
            state = {"phase": "held", "reason": self._invalid_reason,
                     "pending": None, "active_mini": None, "missing": [],
                     "unavailable": list(self.mover.positions), "unassigned": [], "missing_for": {}}
            return self._record(now, index, [], 0, set(self.mover.positions), True, 0, [], state)

        if self.watermark_mode == "board-template-v1":
            if self._caption_detector is None:
                from watermark_detection import BoardCaptionDetector
                self._caption_detector = BoardCaptionDetector()
            caption = self._caption_detector.detect(board_bgr)
            supplemental = caption.exclusion_mask
            if (not isinstance(supplemental, np.ndarray)
                    or supplemental.shape != mask.shape or supplemental.dtype != np.uint8):
                raise ValueError("Board caption detector returned an invalid exclusion mask")
            mask = cv2.bitwise_or(mask, supplemental)
            self._watermark_evidence.update(
                recognized=bool(caption.detected),
                bbox=list(caption.bbox) if caption.bbox is not None else None,
                confidence=float(caption.confidence),
                supplemental_pixels=int(np.count_nonzero(supplemental)))
            self._watermark_evidence["recognition"] = deepcopy(getattr(caption, "provenance", None))
        self._watermark_evidence["effective_pixels"] = int(np.count_nonzero(mask))

        blur = cv2.GaussianBlur(board_bgr, (5, 5), 0)
        obstructed, changed = motion_guard(blur, self._previous, mask, self._previous_mask, self.geometry.grid_px)
        if not self._detector_initialized:
            if hasattr(self.detector, "initialize"):
                self.detector.initialize(board_bgr, list(self.mover.positions.values()),
                                         excluded_mask=mask)
            self._detector_initialized = True
        raw_candidates = self.detector.detect(board_bgr, mask)
        unavailable = masked_anchors(mask, self.mover.positions, self.geometry.grid_px * 0.75)
        if not self._acquired:
            acquired = self._acquire(raw_candidates, unavailable, obstructed, now)
            if not acquired:
                acquisition = self._acquisition_status()
                state = {"phase": "acquiring", "reason": "waiting_for_base_acquisition",
                         "pending": None, "active_mini": None,
                         "missing": acquisition["missing"], "unavailable": sorted(unavailable),
                         "unassigned": [], "missing_for": {},
                         "acquisition_pending": acquisition["pending"]}
                self._previous, self._previous_mask = blur, mask.copy()
                return self._record(now, index, raw_candidates, len(raw_candidates), unavailable,
                                    obstructed, changed, [], state)
            # Seed change evidence only after the declared layout is actually
            # acquired. Never use a partially acquired/moving layout as a map.
            self.reference.update(board_bgr, mask)
        else:
            self._acquisition_matches = ({} if obstructed
                                         else self._anchor_matches(raw_candidates, unavailable))
        candidates = self.reference.filter(board_bgr, raw_candidates, self.mover.positions, mask)
        active = self.mover.status.get("active_mini")
        if active is not None:
            unavailable.discard(active)
        emitted = self.mover.update(candidates, now, obstructed=obstructed, unavailable=unavailable)
        if (not obstructed and not unavailable and not self.mover.status["missing"]
                and self.mover.status["phase"] in ("settled", "confirmed")):
            self.reference.update(board_bgr, mask)
        events = []
        for item in emitted:
            destination = self.geometry.describe(item["to_xy"], item["uncertainty_px"])
            event = {**item, "sequence": len(self.events) + 1, "frame": index,
                     "capture_seconds": now, "from_cell": self.confirmed_cells[item["mini"]],
                     "to_cell": destination["cell"], "destination_cells": destination["cells"],
                     "boundary_uncertain": destination["boundary_uncertain"]}
            self.confirmed_cells[item["mini"]] = event["to_cell"]
            self.position_uncertainty[item["mini"]] = item["uncertainty_px"]
            self.events.append(deepcopy(event))
            events.append(event)
        self._previous, self._previous_mask = blur, mask.copy()
        return self._record(now, index, candidates, len(raw_candidates), unavailable, obstructed, changed, events)


def draw_observer_frame(image, record, geometry):
    """Render last confirmed positions, with held observations clearly marked."""
    result = cv2.copyMakeBorder(image, 0, 64, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    for item in record["candidates"]:
        cv2.circle(result, (round(item["cx"]), round(item["cy"])), round(item["radius"]), (130, 130, 130), 1)
    for mini, xy in record["positions"].items():
        point = tuple(round(v) for v in xy)
        held = mini in record["held"]
        color = (0, 180, 255) if held else (70, 255, 70)
        label = mini + (" (held)" if held else "")
        cv2.circle(result, point, round(geometry.grid_px * 0.8), color, 2)
        for ink, thickness in (((0, 0, 0), 3), (color, 1)):
            cv2.putText(result, label, (point[0] + 24, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, ink, thickness, cv2.LINE_AA)
    text = f'{record["capture_seconds"]:5.1f}s | moves {record["move_count"]} | {record["state"]["reason"]}'
    cv2.putText(result, text, (12, image.shape[0] + 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(result, "Green: last confirmed | Amber: held | Gray: candidates | Foundry moves OFF",
                (12, image.shape[0] + 49), cv2.FONT_HERSHEY_SIMPLEX,
                0.46, (230, 230, 230), 1, cv2.LINE_AA)
    return result
