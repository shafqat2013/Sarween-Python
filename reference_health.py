"""Quality and provenance checks for camera/Foundry comparison evidence.

A dark or uniform image is not proof of a failed export: fully fogged scenes
can legitimately be black. Such an image is retained for audit but cannot by
itself establish pixel alignment or camera/display gain. These checks describe
evidence availability; they do not establish physical-mini tracking accuracy.
"""

from __future__ import annotations

import copy
import math

import cv2
import numpy as np


CAPTURE_METHOD = "pixi-stage-render-texture-v1"
MAX_REFERENCE_AGE_SECONDS = 0.75


def validated_view(view, scene_id):
    """Return the captured geometry unchanged after checking finite bounds."""
    if not isinstance(view, dict) or str(view.get("sceneId")) != str(scene_id):
        raise ValueError("Captured scene geometry is missing or belongs to another scene")
    try:
        width, height, grid = (float(view[key]) for key in ("viewportWidth", "viewportHeight", "gridSize"))
        r = view["registration"]
        left, top, right, bottom = (float(r[key]) for key in ("left", "top", "right", "bottom"))
        values = [width, height, grid, left, top, right, bottom,
                  float(view["gridOriginX"]), float(view["gridOriginY"])]
        for name in ("canvasTransform", "clientToCanvasTransform"):
            matrix = view[name]
            parts = [float(matrix[key]) for key in ("a", "b", "c", "d", "tx", "ty")]
            values.extend(parts)
            if abs(parts[0] * parts[3] - parts[1] * parts[2]) <= 1e-9:
                raise ValueError("Captured geometry has a singular transform")
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Captured geometry has non-finite coordinates")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height and grid > 0):
            raise ValueError("Captured geometry has invalid viewport bounds")
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Captured geometry is incomplete") from exc
    return copy.deepcopy(view)


def matching_view(left, right):
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    keys = ("sceneId", "viewportWidth", "viewportHeight", "registration", "canvasTransform",
            "clientToCanvasTransform", "gridSize", "gridOriginX", "gridOriginY")
    return all(left.get(key) == right.get(key) for key in keys)


def image_quality(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.shape != (360, 640, 3)):
        raise ValueError("Rendered reference must decode as a 640 × 360 BGR image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low, high = np.percentile(gray, (1, 99))
    standard_deviation = float(gray.std())
    informative = standard_deviation >= 2 and high - low >= 5
    return {
        "width": 640, "height": 360,
        "meanLuma": round(float(gray.mean()), 4),
        "stdLuma": round(standard_deviation, 4),
        "minimumLuma": int(gray.min()), "maximumLuma": int(gray.max()),
        "percentileRangeLuma": round(float(high - low), 4),
        "informative": bool(informative),
        "status": "informative" if informative else "low_information",
        "note": ("Image contains spatial contrast; actual display alignment still requires verification."
                 if informative else
                 "Uniform/dark content may be legitimate fog; export correctness and alignment are unverified."),
    }


def capture_provenance(capture, *, received_epoch_ms):
    """Validate the sender clock and explicit-render metadata without aligning clocks by guess."""
    if not isinstance(capture, dict):
        return {"verified": False, "reason": "legacy_capture_without_provenance", "ageAtReceiptSeconds": None}
    try:
        timestamp = float(capture["capturedAtEpochMs"])
        duration = float(capture["durationMs"])
        age = (float(received_epoch_ms) - timestamp) / 1000
        sequence = int(capture["sequence"])
        source_width, source_height = int(capture["sourceWidth"]), int(capture["sourceHeight"])
        if (not all(math.isfinite(v) for v in (timestamp, duration, age))
                or timestamp <= 0 or duration < 0 or duration > 10_000
                or age < -0.1 or age > 10 or sequence < 1
                or not 1 <= source_width <= 16384 or not 1 <= source_height <= 16384
                or not isinstance(capture.get("sessionId"), str) or not capture["sessionId"]
                or capture.get("method") != CAPTURE_METHOD):
            raise ValueError("Unusable capture provenance")
    except (KeyError, TypeError, ValueError, OverflowError):
        return {"verified": False, "reason": "invalid_capture_provenance", "ageAtReceiptSeconds": None}
    return {"verified": True, "reason": "explicit_render_with_source_clock",
            "ageAtReceiptSeconds": max(0.0, age)}


def reference_status(entry, *, current_view, now_monotonic,
                     max_age_seconds=MAX_REFERENCE_AGE_SECONDS):
    """Report readiness for an observational comparison, never movement authority."""
    if not entry:
        return {"state": "waiting", "message": "Waiting for a Foundry rendered reference.",
                "usableForComparison": False, "captureAgeSeconds": None,
                "geometryMatches": False, "provenanceVerified": False}
    if entry.get("error"):
        return {"state": "capture_error", "message": str(entry["error"]),
                "usableForComparison": False, "captureAgeSeconds": None,
                "geometryMatches": False, "provenanceVerified": False,
                "stopped": bool(entry.get("stopped"))}
    result = copy.deepcopy(entry)
    result.pop("viewTransform", None)
    provenance = entry.get("provenance") or {}
    geometry_matches = matching_view(entry.get("viewTransform"), current_view)
    initial_age = provenance.get("ageAtReceiptSeconds")
    age = None if initial_age is None else max(0.0, float(now_monotonic) - entry["receivedAtMonotonic"]) + initial_age
    verified = bool(provenance.get("verified"))
    informative = bool(entry.get("quality", {}).get("informative"))
    if not entry.get("geometryValid"):
        state, message = "invalid_geometry", "Reference geometry is missing or invalid."
    elif not verified:
        state, message = "unverified_capture", "Reference export has no verified render target and capture clock; reload the updated Foundry module."
    elif not geometry_matches:
        state, message = "geometry_changed", "Reference belongs to a different scene or viewport."
    elif age is None or age > max_age_seconds:
        state, message = "stale", "Foundry reference is stale; it cannot explain the current camera frame."
    elif not informative:
        state, message = "low_information", "Reference is uniform/dark. This may be fog; export and alignment remain unverified."
    else:
        state, message = "available", "Reference has contrast, matching geometry and a recent capture clock; camera/display alignment is unverified."
    result.update(state=state, message=message, usableForComparison=state == "available",
                  captureAgeSeconds=age, geometryMatches=geometry_matches,
                  provenanceVerified=verified, displayAlignmentVerified=False)
    return result
