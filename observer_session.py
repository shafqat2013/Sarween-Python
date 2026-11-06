"""Live observer adapter driven by Sarween's existing camera and Tk loop.

There is no second camera reader and no Foundry command sender here. Latest
board images/status are exported for read-only inspection without desktop or
OBS control. Lossless recordings start at explicit position initialization.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np

from observer_engine import BoardGeometry, ObserverEngine, draw_observer_frame
from observer_journal import ObserverJournal
from observer_event_journal import ObserverEventJournal
from hybrid_ring_detector import HybridRingDetector


def _atomic(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def geometry_key(view, shape, calibration=None, registration=None):
    """Use captured mapping, rather than a potentially newer relay revision."""
    key = {"view": deepcopy(view), "warp_size": [int(shape[1]), int(shape[0])],
           "calibration": deepcopy(calibration)}
    if registration is not None:
        key["registration"] = deepcopy(registration)
    return key


class LiveObserverController:
    def __init__(self, parent, output_root, *, window_factory=None, mini_count=6,
                 event_journal_factory=None):
        if window_factory is None:
            from observer_window import ObserverWindow
            window_factory = ObserverWindow
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = Path(output_root)
        self.folder = self.output_root / f"observer_{stamp}_{uuid.uuid4().hex[:6]}"
        self.folder.mkdir(parents=True)
        self.movement_journal = (event_journal_factory or ObserverEventJournal)(
            self.folder / "movement-events.jsonl", self.folder.name)
        self._event_generation = None
        self.engine = None
        self.journal = None
        self.current = None
        self.result = None
        self.closed = False
        self.error = ""
        self._journal_index = 0
        self._last_publish = float("-inf")
        self._last_preview = 0.0
        self._last_processed_timestamp = None
        self._last_reference_key = None
        self._last_reference = None
        self._reference_count = 0
        self.window = window_factory(parent, self.initialize, on_reset=self.reset,
                                     on_record=self.toggle_recording, on_close=self.close,
                                     mini_count=mini_count)
        self._write_index()

    @property
    def is_recording(self):
        return self.journal is not None and not self.journal.closed

    def _write_index(self):
        _atomic(self.output_root / "latest.json", _json_bytes({
            "kind": "sarween-live-observer-location", "folder": str(self.folder.resolve()),
            "status": str((self.folder / "latest.json").resolve()),
            "image": str((self.folder / "latest.jpg").resolve()),
            "movement_event_journal": self.movement_journal.status(),
            "active": not self.closed, "foundry_token_output": False}))

    def _stop_recording(self, reason="userStopped", error=None):
        saved = True
        if self.journal is not None and not self.journal.closed:
            try:
                self.journal.close(reason, error=error)
            except Exception as exc:
                self.error = f"Observer recording could not finish saving: {exc}"
                saved = False
        path = str(self.journal.folder) if self.journal is not None else ""
        if not self.window.closed:
            self.window.set_recording(False, path)
        return saved

    def reset(self):
        saved = self._stop_recording("reinitialized")
        self.movement_journal.end_generation("bindings_reset")
        self._event_generation = None
        self.engine = self.result = None
        self._last_processed_timestamp = None
        if saved:
            self.error = ""

    def initialize(self, positions, frozen_board, frozen_context):
        if self.closed or self.current is None or not isinstance(frozen_context, dict):
            raise ValueError("Wait for a current locked camera frame")
        if (not frozen_context.get("locked") or not self.current[1].get("locked")
                or frozen_context.get("geometry_key") != self.current[1].get("geometry_key")):
            raise ValueError("The camera lock or map geometry changed. Choose starting positions again.")
        self.reset()
        geometry = BoardGeometry(frozen_context["view"], (frozen_board.shape[1], frozen_board.shape[0]))
        self.engine = ObserverEngine(geometry, positions, geometry_key=frozen_context["geometry_key"],
                                     detector_factory=HybridRingDetector,
                                     watermark_mode="board-template-v1",
                                     display_transition_policy="hold-known-display-changes-v1")
        try:
            self._event_generation = self.movement_journal.begin_generation(
                self.engine.initialization(), reason="initial_bindings")
            self._start_recording()
            self._consume(frozen_board, frozen_context)
        except Exception as exc:
            self._stop_recording("initializationError", error=exc)
            self.engine = self.result = None
            self.movement_journal.end_generation("initialization_failed")
            self._event_generation = None
            self.error = f"Observer initialization could not finish: {exc}"
            raise
        self.window.set_status(self._movement_message(
            "Starting positions saved. Keep the minis still while their bases are acquired."))
        return True

    def _start_recording(self):
        if self.engine is None:
            raise ValueError("Choose starting positions first")
        self._journal_index += 1
        self.journal = ObserverJournal(self.folder / f"recording_{self._journal_index:03d}",
                                       self.engine.initialization())
        self.window.set_recording(True, str(self.journal.folder))

    def toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
            self.window.set_status("Observer inputs saved. Observation continues without recording.")
            return
        if self.engine is None or self.current is None or self.engine.needs_reset:
            raise ValueError("Choose starting positions before recording")
        if self.result is None or self.result["held"] or self.result["state"]["phase"] not in ("settled", "confirmed"):
            raise ValueError("Wait until all mini bases are visible and still before recording")
        # An independent recording starts from an explicitly settled layout,
        # not a partial state machine whose earlier inputs have been omitted.
        board, context = self.current
        positions = deepcopy(self.engine.mover.positions)
        cells = deepcopy(self.engine.confirmed_cells)
        self.movement_journal.end_generation("input_recording_restarted")
        self.engine = ObserverEngine(self.engine.geometry, positions, cells,
                                     geometry_key=context["geometry_key"],
                                     detector_factory=HybridRingDetector,
                                     watermark_mode="board-template-v1",
                                     display_transition_policy="hold-known-display-changes-v1")
        self._event_generation = self.movement_journal.begin_generation(
            self.engine.initialization(), reason="input_recording_restarted")
        self._last_processed_timestamp = None
        self._start_recording()
        self._consume(board, context)

    def _consume(self, board, context):
        timestamp = context["timestamp"]
        if self._last_processed_timestamp is not None and timestamp <= self._last_processed_timestamp:
            raise ValueError("Observer capture clock did not advance; choose starting positions again")
        result = self.engine.process(board, context["mask"], timestamp,
                                     locked=context["locked"], geometry_key=context["geometry_key"],
                                     frame_index=context.get("frame_index"),
                                     scene_visual_revision=context.get("metadata", {}).get("scene_visual_revision"))
        self._last_processed_timestamp = timestamp
        self.result = result
        # A confirmed movement must not depend on the video budget or the next
        # one-second preview tick. Keep the original engine result unchanged.
        self.movement_journal.append_events(self._event_generation, result.get("events", []),
            reason=result.get("state", {}).get("reason"),
            recording_folder=str(self.journal.folder) if self.is_recording else None)
        if result.get("needs_reset"):
            self.movement_journal.end_generation(result.get("state", {}).get("reason") or "observer_held")
        if self.is_recording:
            try:
                written = self.journal.append(board, context["mask"], timestamp, result,
                    locked=context["locked"], geometry_key=context["geometry_key"],
                    frame_index=context.get("frame_index"), metadata=context.get("metadata"))
                if not written:
                    self.window.set_recording(False, str(self.journal.folder))
                    self.error = "Recording limit reached; saved inputs are available for replay."
            except Exception as exc:
                self._stop_recording("recordingError", error=exc)
                self.error = f"Observer input recording stopped: {exc}"
        return result

    def ingest_reference(self, reference):
        """Retain distinct real Foundry images, separately from camera inputs."""
        if not reference or not isinstance(reference.get("image"), bytes):
            return
        entry = reference.get("entry") or {}
        capture = entry.get("capture") or {}
        key = (capture.get("sessionId"), capture.get("sequence"), entry.get("receivedAtMonotonic"))
        if key == self._last_reference_key:
            return
        folder = self.folder / "references"
        folder.mkdir(exist_ok=True)
        self._reference_count += 1
        name = f"reference_{self._reference_count:05d}"
        path = folder / (name + ".jpg")
        _atomic(path, reference["image"])
        info = {**deepcopy(entry), "image": str(path.resolve()),
                "sha256": hashlib.sha256(reference["image"]).hexdigest(),
                "camera_pairing": "asynchronous; not alignment verified"}
        _atomic(folder / (name + ".json"), _json_bytes(info))
        self._last_reference_key = key
        self._last_reference = info

    def update(self, board, mask, timestamp, *, view, locked, frame_index=None,
               calibration=None, registration=None, metadata=None):
        if self.closed:
            return self.result
        fresh = isinstance(board, np.ndarray) and board.ndim == 3
        if not fresh:
            if self.current is None:
                self.window.set_status("Waiting for camera pixels and marker lock.")
                return None
            board = self.current[0]
            mask = np.zeros(board.shape[:2], np.uint8)
            locked = False
        mask = np.zeros(board.shape[:2], np.uint8) if mask is None else np.where(mask > 0, 255, 0).astype(np.uint8)
        key = geometry_key(view, board.shape, calibration, registration)
        context = {"timestamp": float(timestamp), "frame_index": frame_index,
                   "view": deepcopy(view), "geometry_key": key, "locked": bool(locked and view),
                   "mask": mask.copy(), "metadata": {**deepcopy(metadata or {}), "fresh_board_pixels": fresh}}
        if registration is not None:
            context["metadata"]["marker_registration"] = deepcopy(registration)
        if self._last_reference:
            context["metadata"]["reference_image"] = deepcopy(self._last_reference)
        self.current = (board.copy(), context)
        if self.engine is not None:
            try:
                self._consume(board, context)
            except Exception as exc:
                self.error = f"Observer held: {exc}"
                self._stop_recording("observerError", error=exc)
                if self.result:
                    self.result = {**deepcopy(self.result), "events": [],
                                   "held": list(self.result["positions"]), "needs_reset": True,
                                   "state": {"phase": "held", "reason": "observer_error"}}
                self.engine = None
                self.movement_journal.end_generation("observer_error")
                self._event_generation = None
                self.window.invalidate_initialization(self.error)
        result = self.result
        reason = (result or {}).get("state", {}).get("reason", "Choose starting positions to begin.")
        if result and result.get("needs_reset") and self.window.initialized:
            self.window.invalidate_initialization("Lock or geometry changed. Choose starting positions again.")
        ref = context["metadata"].get("reference_status") or {}
        acquisition = (result or {}).get("acquisition") or {}
        acquiring = bool(result and result.get("acquired") is False and not result.get("needs_reset"))
        status = self.error or (
            "Connect Foundry and open a square-grid scene before choosing positions." if not view else
            "Wait for a verified marker lock before choosing positions." if not context["locked"] else
            f"Acquiring bases: {acquisition.get('visible_count', 0)}/{acquisition.get('required', 6)} visible, "
            f"{acquisition.get('ready_count', 0)}/{acquisition.get('required', 6)} confirmed. Keep minis still."
            if acquiring else
            f"Bases acquired: {acquisition.get('required', 6)}/{acquisition.get('required', 6)} · "
            f"{acquisition.get('visible_count', 0)}/{acquisition.get('required', 6)} visible · "
            + str(reason).replace("_", " ") if result and result.get("acquired") else
            str(reason).replace("_", " "))
        if ref.get("message"):
            status += " · Reference: " + str(ref["message"])
        status = self._movement_message(status)
        if self.movement_journal.error:
            # A save failure must be visible even between image refreshes.
            self.window.set_status(status)
        now = time.perf_counter()
        if now - self._last_preview >= 1 / 12:
            displayed = ({**result, "events": deepcopy(self.engine.events)}
                         if result and self.engine else result)
            self.window.update_frame(board, displayed, context=context, status=status)
            self._last_preview = now
        try:
            self._publish(board, context, result, status)
        except Exception as exc:
            self.error = f"Observer preview could not be saved: {exc}"
            self.window.set_status(self.error)
        return result

    def _movement_message(self, message):
        history = self.movement_journal.status()
        errors = list(dict.fromkeys(value for value in (history["error"], history.get("exportError"))
                                    if value and value not in message))
        return " · ".join([*errors, message])

    def _publish(self, board, context, result, status, *, force=False):
        now = time.perf_counter()
        if not force and now - self._last_publish < 1:
            return
        self._last_publish = now
        self.movement_journal.flush()
        preview = draw_observer_frame(board, result, self.engine.geometry) if result and self.engine else board
        ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise OSError("Could not export observer preview")
        preview_bytes = encoded.tobytes()
        _atomic(self.folder / "latest.jpg", preview_bytes)
        _atomic(self.folder / "latest.json", _json_bytes({
            "kind": "sarween-live-observer-status", "active": not self.closed,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "capture_timestamp": context["timestamp"], "frame_index": context["frame_index"],
            "preview": {"file": "latest.jpg", "sha256": hashlib.sha256(preview_bytes).hexdigest(),
                        "frame_index": context["frame_index"], "capture_timestamp": context["timestamp"],
                        "pairing": "Verify the image hash; image and status are individually atomic"},
            "fresh_board_pixels": context["metadata"]["fresh_board_pixels"],
            "locked": context["locked"], "message": self._movement_message(status),
            "recording": self.is_recording,
            "recording_folder": str(self.journal.folder) if self.journal else None,
            "reference_status": context["metadata"].get("reference_status"),
            "scene_visual_revision": context["metadata"].get("scene_visual_revision"),
            "movement_event_journal": self.movement_journal.status(),
            "movement_events": self.movement_journal.recent_events(),
            "movement_events_scope": "bounded recent run history; full accepted-move history is in movement_event_journal.path",
            "observation": result, "foundry_token_output": False, "fog_validated": False}))

    def snapshot(self):
        return {"kind": "live-observer", "foundry_token_output": False,
                "movement_event_journal": self.movement_journal.status(),
                "initialization": self.engine.initialization() if self.engine else None,
                "last_observation": deepcopy(self.result)}

    def close(self):
        if self.closed:
            return
        self._stop_recording("observerClosed")
        self.movement_journal.close()
        if self.movement_journal.error:
            self.error = self.movement_journal.error
            if not self.window.closed:
                self.window.set_status(self.error)
        self.closed = True
        try:
            if self.current:
                self._publish(*self.current, self.result, "Observer closed; Foundry movement remains off.", force=True)
        except Exception as exc:
            self.error = f"Observer status could not finish saving: {exc}"
        try:
            self._write_index()
        except Exception as exc:
            self.error = f"Observer location could not finish saving: {exc}"
        if not self.window.closed:
            self.window.close()
