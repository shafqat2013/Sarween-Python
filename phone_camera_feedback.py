"""Fresh camera guidance and guarded relative LCD brightness, never tracker truth.

The CPL index is a rough 100*current/reference encoded LCD gray median. It is
neither a physical angle nor optical transmission. Current marker geometry
registers the same LCD region independently in every frame, so ordinary phone
or filter jitter does not discard the reference. Exposure, scene and optics
must remain comparable. No reference is set implicitly.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import time

import cv2
import numpy as np

from camera_sources import stable_stream_key
from phone_camera_protocol import validate_feedback


class PhoneCameraFeedback:
    def __init__(self, capture, *, kind="preview", marker_ids=(10, 11, 12, 13),
                 detect_interval=.5, detector=None, clock=time.monotonic):
        if kind not in ("preview", "tracking", "calibration"):
            raise ValueError("Unknown camera guidance kind")
        self.capture, self.kind, self.clock = capture, kind, clock
        self.marker_ids = tuple(int(value) for value in marker_ids)
        if len(self.marker_ids) != 4 or len(set(self.marker_ids)) != 4:
            raise ValueError("Camera guidance needs four distinct marker IDs")
        self.interval = max(.25, float(detect_interval))
        self.detector = detector
        self._last_detection = -float("inf")
        self._observation = self._reference = self._target = None
        self._reason = "Wait for a fresh camera image"
        self._request_error = None
        self._last_metadata = None
        self.last_feedback = None

    def _detect(self, frame):
        if self.detector is not None:
            return self.detector(frame)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        return detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def _fresh(self, frame, metadata):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or not isinstance(metadata, dict)
                or metadata.get("freshFrame") is False or metadata.get("fresh") is False):
            return False
        age = metadata.get("hostReceiveAgeSeconds", 0.)
        if type(age) not in (int, float) or not math.isfinite(age) or not 0 <= age <= 1.25:
            return False
        received = metadata.get("hostReceivedMonotonic")
        if received is not None and (type(received) not in (int, float)
                or not 0 <= self.clock() - received <= 1.25):
            return False
        status = getattr(self.capture, "status", None)
        if isinstance(status, dict) and status.get("sessionId") is not None:
            if (status["sessionId"] != metadata.get("sessionId")
                    or status.get("connectionGeneration") != metadata.get("connectionGeneration")):
                return False
        return True

    @staticmethod
    def _exposure(metadata):
        if metadata.get("exposureLocked") is not True:
            raise ValueError("Lock exposure on the iPhone before setting a CPL reference")
        iso, shutter = metadata.get("iso"), metadata.get("exposureDurationNs")
        if (type(iso) not in (int, float) or not math.isfinite(iso) or iso <= 0
                or type(shutter) is not int or shutter <= 0):
            raise ValueError("Exposure measurements are unavailable")
        gains = metadata.get("whiteBalanceGains")
        if (not isinstance(gains, list) or len(gains) != 3
                or not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in gains)):
            raise ValueError("White balance measurements are unavailable")
        return (float(iso), float(shutter), *map(float, gains))

    def _measure(self, frame, metadata, corners, ids, view_key, scene_revision, registration, excluded_regions):
        self._observation = None
        try:
            if isinstance(registration, dict) and (registration.get("blocked") or registration.get("pending")):
                raise ValueError(registration.get("message") or "Complete printed-marker registration in the live session first")
            exposure = self._exposure(metadata)
            if view_key is None or scene_revision is None:
                raise ValueError("Connect Foundry and keep the map fixed before setting a CPL reference")
            found = {}
            if ids is not None:
                for marker_id, quad in zip(np.asarray(ids).ravel(), corners):
                    marker_id = int(marker_id)
                    if marker_id in self.marker_ids:
                        if marker_id in found:
                            raise ValueError("Duplicate registration markers are ambiguous")
                        found[marker_id] = np.asarray(quad, np.float32).reshape(4, 2)
            if len(found) != 4:
                raise ValueError("All four current registration markers must be visible for the CPL index")
            size = (256, 144)
            if self.marker_ids == (10, 11, 12, 13):
                quad = np.array([found[marker][corner] for corner, marker in enumerate(self.marker_ids)], np.float32)
                if not cv2.isContourConvex(quad) or abs(cv2.contourArea(quad)) < frame.shape[0] * frame.shape[1] * .02:
                    raise ValueError("Screen-marker geometry is invalid")
                H = cv2.getPerspectiveTransform(quad, np.array([[0, 0], [255, 0], [255, 143], [0, 143]], np.float32))
            elif self.marker_ids == (20, 21, 22, 23):
                from printed_markers import solve_homography
                layout = (registration or {}).get("printedMarkerLayout")
                if layout is None:
                    raise ValueError("Registered printed-marker layout is required for the CPL index")
                H = solve_homography(corners, ids, layout, view_key, size)
                if H is None:
                    raise ValueError("Four printed markers do not verify the current LCD geometry")
            else:
                raise ValueError("A known LCD registration is required for the CPL index")
            gray = cv2.warpPerspective(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), H, size)
            valid = np.ones(frame.shape[:2], np.uint8) * 255
            for x, y, radius in excluded_regions:
                if all(math.isfinite(float(v)) for v in (x, y, radius)) and radius > 0:
                    cv2.circle(valid, (round(x), round(y)), math.ceil(radius), 0, -1)
            valid = cv2.warpPerspective(valid, H, size, flags=cv2.INTER_NEAREST) > 0
            border = np.zeros((size[1], size[0]), bool)
            border[22:-22, 38:-38] = True
            valid &= border
            if np.count_nonzero(valid) < .75 * np.count_nonzero(border):
                raise ValueError("Too little unobstructed LCD area is available")
            median = float(np.median(gray[valid]))
            key = json.dumps({"stream": stable_stream_key(metadata), "view": view_key,
                              "scene": scene_revision, "registration": registration,
                              "torchOn": metadata.get("torchOn"), "torchLevel": metadata.get("torchLevel")},
                             sort_keys=True, allow_nan=False)
            self._observation = {"median": median, "key": key,
                                 "exposure": exposure, "when": self.clock(), "metadata": deepcopy(metadata)}
            self._reason = ""
        except (ValueError, TypeError, KeyError, cv2.error) as exc:
            self._reason = str(exc)

    def _cpl(self):
        current = self._observation
        if self._request_error:
            return {"state": "unavailable", "message": self._request_error}
        if current is None or self.clock() - current["when"] > 1.25:
            return {"state": "unavailable", "message": self._reason or "Wait for a fresh comparable LCD image"}
        if self._reference is None:
            return {"state": "reference-needed", "message": "Set a bright LCD reference, then rotate only the filter. This is a relative brightness index."}
        reference = self._reference
        if current["key"] != reference["key"]:
            return {"state": "invalid", "message": "Camera, lens, scene or map view changed. Set a new bright reference."}
        if not np.allclose(current["exposure"], reference["exposure"], rtol=.005, atol=.001):
            return {"state": "invalid", "message": "ISO, shutter or white balance changed. Lock them and set a new bright reference."}
        index = 100 * current["median"] / reference["median"]
        if not math.isfinite(index) or not 0 <= index <= 10000:
            return {"state": "invalid", "message": "LCD brightness is outside the comparable range"}
        result = {"state": "comparable", "index": round(index, 1),
                  "message": "Relative LCD brightness; reference=100. Keep map and room lighting fixed. Not an angle or optical transmission."}
        if self._target is not None:
            result["target"] = round(self._target, 1)
            if abs(index - self._target) <= 3:
                result["state"] = "target-matched"
                result["message"] = "Within 3 index units of the saved target. Keep map/lighting fixed and recheck marker visibility."
        return result

    def take_requests(self):
        method = getattr(self.capture, "take_requests", None)
        requests = method() if callable(method) else []
        if not isinstance(requests, (list, tuple)):
            return []
        remaining = []
        for request in requests[:16]:
            if not isinstance(request, dict):
                continue
            action = request.get("action", "")
            if not isinstance(action, str) or not action.startswith("cpl."):
                remaining.append(request)
                continue
            status = getattr(self.capture, "status", None)
            context = (status if isinstance(status, dict) and status.get("sessionId") is not None
                       else self._last_metadata)
            received = request.get("hostReceivedMonotonic")
            if (not isinstance(context, dict) or request.get("sessionId") != context.get("sessionId")
                    or request.get("connectionGeneration") != context.get("connectionGeneration")
                    or (received is not None and (type(received) not in (int, float)
                        or not 0 <= self.clock() - received <= 3))):
                continue
            if action == "cpl.reset":
                self._reference = self._target = None
                self._request_error = None
                continue
            current = self._observation
            if (current is None or self.clock() - current["when"] > 1.25
                    or request.get("sessionId") != current["metadata"].get("sessionId")
                    or request.get("connectionGeneration") != current["metadata"].get("connectionGeneration")):
                self._request_error = "CPL request expired; retry with a fresh comparable frame"
                continue
            if action == "cpl.reference":
                if current["median"] >= 8:
                    self._reference, self._target = deepcopy(current), None
                    self._request_error = None
                else:
                    self._request_error = "Reference LCD is too dark. Brighten it before setting the reference."
            elif action == "cpl.target":
                self._request_error = None
                value = self._cpl()
                if "index" in value:
                    self._target = value["index"]
                else:
                    self._request_error = value["message"]
        return remaining

    def publish(self, frame=None, metadata=None, *, calibration=None, markers=None, locked=None,
                view_key=None, scene_revision=None, registration=None, excluded_regions=()):
        now = self.clock()
        fresh = self._fresh(frame, metadata)
        if fresh:
            self._last_metadata = deepcopy(metadata)
        feedback = {"kind": self.kind}
        if fresh and type(metadata.get("seq")) is int:
            feedback["sourceSeq"] = metadata["seq"]
        if self.kind == "calibration":
            self._observation = None
            feedback["markers"] = {"ids": [], "requiredIds": list(self.marker_ids), "count": 0, "required": 4}
            feedback["cpl"] = {"state": "unavailable", "message": "CPL index is unavailable while the calibration board is active"}
        elif not fresh:
            self._observation = None
            self._reason = "Camera frame is stale or unavailable"
            feedback["markers"] = {"ids": [], "requiredIds": list(self.marker_ids), "count": 0, "required": 4, "locked": False}
            feedback["cpl"] = self._cpl()
        else:
            if now - self._last_detection < self.interval:
                return None
            self._last_detection = now
            if (isinstance(markers, dict) and type(markers.get("sourceSeq")) is int
                    and markers["sourceSeq"] == metadata.get("seq") and "corners" in markers and "ids" in markers):
                corners, ids = markers["corners"], markers["ids"]
            else:
                corners, ids, _rejected = self._detect(frame)
            visible = sorted(set() if ids is None else set(map(int, np.asarray(ids).ravel())) & set(self.marker_ids))
            feedback["markers"] = {"ids": visible, "requiredIds": list(self.marker_ids), "count": len(visible), "required": 4}
            if locked is not None:
                feedback["markers"]["locked"] = bool(locked)
            self._measure(frame, metadata, corners, ids, view_key, scene_revision, registration, excluded_regions)
            feedback["cpl"] = self._cpl()
        if calibration is not None:
            feedback["calibration"] = deepcopy(calibration)
        value = validate_feedback(feedback)
        self.last_feedback = deepcopy(value)
        method = getattr(self.capture, "send_feedback", None)
        if callable(method):
            method(value)
        return value
