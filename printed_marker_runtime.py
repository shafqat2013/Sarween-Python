"""Queued printed-registration requests, consumed by the existing camera loop.

This module never opens a camera, creates a window, or sends Foundry commands.
The GUI may queue/cancel a request; only co-observed camera detections can finish
it. The profile store performs the final active-profile check atomically.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import threading
import time

import cv2
import numpy as np

import printed_markers as geometry


class PrintedMarkerController:
    def __init__(self, store=None, *, context_provider=None, confirmation_frames=3,
                 clock=time.monotonic, heartbeat_seconds=3.):
        if store is None:
            from hardware_profiles import HardwareProfiles
            store = HardwareProfiles()
        self.store = store
        self._context_provider = context_provider
        self._lock = threading.RLock()
        self._pending = None
        self._samples = []
        self._revision = 0
        self._message = ""
        self.confirmation_frames = max(1, int(confirmation_frames))
        self._clock = clock
        self._heartbeat_seconds = float(heartbeat_seconds)
        self._source = None
        self._last_heartbeat = None

    def _expire_source(self):
        if (self._source is not None and self._last_heartbeat is not None
                and self._clock() - self._last_heartbeat > self._heartbeat_seconds):
            self.camera_stopped(self._source)

    def heartbeat(self, source):
        """A consumed camera frame proves that this existing source is alive."""
        with self._lock:
            self._expire_source()
            if source is not self._source:
                if self._pending is not None:
                    self._pending = None
                    self._samples = []
                    self._message = "Camera source changed; printed registration cancelled. Start again."
                self._source = source
                self._revision += 1
            self._last_heartbeat = self._clock()

    def camera_stopped(self, source):
        with self._lock:
            if source is not self._source:
                return
            self._source = None
            self._last_heartbeat = None
            self._pending = None
            self._samples = []
            self._revision += 1
            self._message = "Camera is off; start a live camera session before registering or changing printed markers."

    def _require_source(self):
        self._expire_source()
        if self._source is None:
            raise ValueError("Start a live camera session before registering or changing printed markers")

    def _profile(self):
        return self.store.active_profile("display")

    def _cancel_if_profile_changed(self, profile):
        if self._pending is not None and profile != self._pending["profile"]:
            self._pending = None
            self._samples = []
            self._revision += 1
            self._message = "Display profile changed; printed registration cancelled. Start again for this profile."

    def configuration(self):
        """Return a fresh detached configuration; malformed profiles block use."""
        with self._lock:
            self._expire_source()
            try:
                profile = self._profile()
                self._cancel_if_profile_changed(profile)
                layout = profile.get("printedMarkerLayout") if profile else None
                layout = geometry.validate_layout(layout) if layout is not None else None
                enabled = profile.get("printedMarkersEnabled", False) if profile else False
                if type(enabled) is not bool or (enabled and layout is None):
                    raise ValueError("Register printed markers before using them")
                mode = "printed" if enabled else "screen"
                fingerprint = geometry.layout_fingerprint(layout) if layout is not None else None
                token = {"mode": mode, "profileId": profile["id"] if profile else None,
                         "layoutFingerprint": fingerprint, "revision": self._revision}
                return {"mode": mode, "profileId": profile["id"] if profile else None,
                        "profileName": profile.get("name", "") if profile else "",
                        "layout": layout, "pending": self._pending is not None,
                        "token": token, "blocked": False, "cameraActive": self._source is not None}
            except (ValueError, OSError, KeyError) as exc:
                return {"mode": "screen", "profileId": None, "profileName": "", "layout": None,
                        "pending": self._pending is not None, "blocked": True,
                        "cameraActive": self._source is not None,
                        "token": {"error": str(exc), "revision": self._revision},
                        "message": str(exc)}

    def status(self):
        with self._lock:
            config = self.configuration()
            message = config.get("message") or self._message
            if not message:
                message = ("Choose a saved TV/display profile first." if not config["profileId"] else
                           "Printed markers active. Keep Foundry fullscreen on this TV." if config["mode"] == "printed" else
                           "Printed layout saved; screen markers currently active." if config["layout"] else
                           "Print and place the four cards, then register them in a live camera session.")
            if not config["cameraActive"]:
                message = "Camera is off; start a live camera session before registering or changing printed markers."
            return {"available": bool(config["profileId"] and not config["blocked"] and config["cameraActive"]),
                    "mode": config["mode"], "message": message, "pending": config["pending"],
                    "calibrated": config["layout"] is not None, "profileName": config["profileName"],
                    "profileId": config["profileId"]}

    def request_calibration(self, aligned_to_guides=False):
        if type(aligned_to_guides) is not bool:
            raise ValueError("Guide alignment confirmation must be true or false")
        with self._lock:
            self._require_source()
            profile = self._profile()
            if not profile:
                raise ValueError("Choose a saved TV/display profile before registering printed markers")
            self._pending = {"profile": deepcopy(profile), "aligned_to_guides": aligned_to_guides}
            self._samples = []
            self._revision += 1
            self._message = ("Waiting for four paper markers in the live camera. Guide placement is your declared alignment."
                             if aligned_to_guides else
                             "Waiting for all four screen and all four paper markers together in the live camera.")

    def cancel_calibration(self):
        with self._lock:
            if self._pending is not None:
                self._pending = None
                self._samples = []
                self._revision += 1
                self._message = "Printed registration cancelled; the previous marker mode is retained."

    def use_printed(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Printed marker selection must be true or false")
        with self._lock:
            self._require_source()
            profile = self._profile()
            if not profile:
                raise ValueError("Choose a saved TV/display profile first")
            if enabled:
                geometry.validate_layout(profile.get("printedMarkerLayout"))
            changed = profile.get("printedMarkersEnabled", False) != enabled or self._pending is not None
            self.store.set_printed_markers_enabled(profile["id"], enabled, expected_profile=profile)
            self._pending = None
            self._samples = []
            if changed:
                self._revision += 1
            self._message = ("Printed markers selected. Camera lock and mini bindings must be acquired again."
                             if enabled else "Screen markers selected. Camera lock and mini bindings must be acquired again.")

    def view_context(self):
        """Read current same-connection Foundry state, without requesting actions."""
        if self._context_provider is not None:
            return deepcopy(self._context_provider())
        import foundryoutput as fo
        status = fo.get_vtt_status()
        view = fo.get_view_transform_payload()
        ready = bool(status.get("connected") and view and view.get("sceneId") == status.get("sceneId"))
        return {"ready": ready, "view": view if ready else None,
                "settings": deepcopy(status.get("settings") or {})}

    def consume_detections(self, corners, ids, context, warp_size):
        """Try a pending calibration using this frame only; failures remain pending."""
        with self._lock:
            self._expire_source()
            if self._pending is None:
                return
            profile = self._profile()
            self._cancel_if_profile_changed(profile)
            if self._pending is None:
                return
            try:
                if not context.get("ready") or not context.get("view"):
                    raise ValueError("Connect Foundry fullscreen on the calibrated TV before registering markers")
                if self._pending["aligned_to_guides"]:
                    settings = context.get("settings") or {}
                    if settings.get("viewportMarkersEnabled") is not True:
                        raise ValueError("Turn on the on-screen placement guides before declaring paper alignment")
                    layout = geometry.guide_aligned_layout(context["view"],
                        settings.get("viewportMarkerSize"), settings.get("viewportMarkerInset"),
                        pattern_size=settings.get("viewportMarkerPatternSize"),
                        margin=settings.get("viewportMarkerMargin"))
                else:
                    layout = geometry.calibrate_layout(corners, ids, context["view"])
                if geometry.solve_homography(corners, ids, layout, context["view"], warp_size) is None:
                    raise ValueError("All four paper cards must be visible, correctly oriented, flat and aligned")
                if self._samples:
                    previous = np.array(list(self._samples[-1]["markerCorners"].values()))
                    current = np.array(list(layout["markerCorners"].values()))
                    if np.max(np.abs(previous - current)) > .004:
                        self._samples = []
                self._samples.append(layout)
                if len(self._samples) < self.confirmation_frames:
                    self._message = f"Printed registration: {len(self._samples)}/{self.confirmation_frames} consistent observations. Keep hands clear."
                    return
                layout = deepcopy(layout)
                stacked = np.array([list(sample["markerCorners"].values()) for sample in self._samples])
                medians = np.median(stacked, axis=0)
                layout["markerCorners"] = {key: value.tolist() for key, value in zip(layout["markerCorners"], medians)}
                layout["calibratedAt"] = datetime.now(timezone.utc).isoformat()
                layout = geometry.validate_layout(layout)
                # Do not save a result into a different, edited, or inactive TV
                # profile. The real store repeats this check within its write lock.
                expected = self._pending["profile"]
                if self._profile() != expected:
                    raise ValueError("Display profile changed during registration; start again")
                saver = getattr(self.store, "save_printed_marker_layout", None)
                if saver is None:
                    raise ValueError("This build cannot save printed layouts atomically")
                saver(profile["id"], layout, expected_profile=expected)
                self._pending = None
                self._samples = []
                self._revision += 1
                self._message = ("Printed markers registered and selected. Guide alignment was user-declared; acquire camera lock and bind minis again."
                                 if layout["calibrationMethod"] == "guide-aligned-user-declared" else
                                 "Printed markers registered and selected from co-visible screen markers. Acquire camera lock and bind minis again.")
            except (ValueError, OSError, cv2.error) as exc:
                self._samples = []
                self._message = str(exc)


_singleton = None
_singleton_lock = threading.Lock()


def get_printed_marker_controller():
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = PrintedMarkerController()
        return _singleton


class RecordedPrintedMarkerController:
    """Immutable checkpoint registration; never reads profiles or connections.

    The caller supplies a view provider populated by the recording's timeline
    replay. No live module handshake is consulted in this isolated file source.
    """
    def __init__(self, metadata, *, view_provider):
        if not isinstance(metadata, dict) or metadata.get("mode") not in ("screen", "printed"):
            raise ValueError("Unsupported recorded marker registration")
        self.metadata = deepcopy(metadata)
        self._view_provider = view_provider
        layout = metadata.get("printedMarkerLayout")
        if layout is not None:
            layout = geometry.validate_layout(layout)
        if metadata["mode"] == "printed" and layout is None:
            raise ValueError("Printed recording has no saved marker layout")
        self._layout = layout
        fingerprint = geometry.layout_fingerprint(layout) if layout is not None else None
        if metadata.get("layoutFingerprint") not in (None, fingerprint):
            raise ValueError("Recorded printed layout fingerprint does not match")
        self._fingerprint = fingerprint

    def heartbeat(self, _source):
        pass

    def camera_stopped(self, _source):
        pass

    def configuration(self):
        m = self.metadata
        pending = bool(m.get("pending", False))
        token = {"mode": m["mode"], "profileId": m.get("profileId"),
                 "layoutFingerprint": self._fingerprint,
                 "revision": int(m.get("controllerRevision", m.get("revision", 0)))}
        return {"mode": m["mode"], "profileId": m.get("profileId"),
                "profileName": m.get("profileName", ""), "layout": deepcopy(self._layout),
                "pending": pending, "token": token, "blocked": pending,
                "message": "Recorded marker calibration was incomplete" if pending else "",
                "cameraActive": False}

    def view_context(self):
        view = deepcopy(self._view_provider())
        return {"ready": view is not None, "view": view, "settings": {}}
