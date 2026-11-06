"""Pure printed-marker registration geometry; no camera, files, UI or relay.

Stored corners use the complete LCD/browser viewport coordinate system, not the
inset rectangle currently occupied by the on-screen registration guides. Thus a
flat printed card can be on the bezel without being mistaken for an LCD corner.
Foundry must occupy the same full display used for registration.
"""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np


DICTIONARY = "DICT_6X6_250"
PRINTED_IDS = {"TL": 20, "TR": 21, "BR": 22, "BL": 23}
SCREEN_IDS = {"TL": 10, "TR": 11, "BR": 12, "BL": 13}
_ORDER = ("TL", "TR", "BR", "BL")
_METHODS = {"co-visible", "guide-aligned-user-declared"}


def _number(value, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be finite")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _quad(value, *, label, min_area):
    try:
        quad = np.asarray(value, dtype=np.float64).reshape(4, 2)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} needs four ordered corners") from exc
    if not np.isfinite(quad).all():
        raise ValueError(f"{label} has nonfinite corners")
    contour = quad.astype(np.float32)
    if (not cv2.isContourConvex(contour)
            or abs(cv2.contourArea(contour)) < min_area
            or min(np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)) <= math.sqrt(min_area) / 5):
        raise ValueError(f"{label} is degenerate or self-intersecting")
    return quad


def viewport(view):
    """Validate current viewport bounds without requiring visible LCD markers."""
    if not isinstance(view, dict):
        raise ValueError("Waiting for the current Foundry viewport")
    try:
        width = _number(view["viewportWidth"], "Viewport width")
        height = _number(view["viewportHeight"], "Viewport height")
        registration = view["registration"]
        left, top, right, bottom = (_number(registration[k], "Registration")
                                    for k in ("left", "top", "right", "bottom"))
    except KeyError as exc:
        raise ValueError("Foundry has not supplied its viewport registration") from exc
    if (width < 64 or height < 64 or not 0 <= left < right <= width
            or not 0 <= top < bottom <= height or right - left < 16 or bottom - top < 16):
        raise ValueError("Foundry supplied an invalid viewport registration")
    return width, height, left, top, right, bottom


def validate_layout(layout):
    """Return a detached, JSON-native canonical physical marker layout."""
    fields = {"schemaVersion", "dictionary", "markerIds", "markerCorners", "viewportAspectRatio",
              "calibratedAt", "calibrationMethod"}
    if not isinstance(layout, dict) or set(layout) - fields:
        raise ValueError("Invalid printed marker layout fields")
    if type(layout.get("schemaVersion")) is not int or layout["schemaVersion"] != 1:
        raise ValueError("Unsupported printed marker layout version")
    if layout.get("dictionary") != DICTIONARY or layout.get("markerIds") != PRINTED_IDS:
        raise ValueError("Printed markers require DICT_6X6_250 IDs20,21,22,23")
    raw = layout.get("markerCorners")
    if not isinstance(raw, dict) or set(raw) != {str(v) for v in PRINTED_IDS.values()}:
        raise ValueError("Printed layout must contain exactly four distinct marker IDs")
    corners = {}
    for marker_id in PRINTED_IDS.values():
        if np.shape(raw[str(marker_id)]) != (4, 2):
            raise ValueError(f"Marker {marker_id} needs four two-dimensional corners")
        q = _quad(raw[str(marker_id)], label=f"Marker {marker_id}", min_area=1e-7)
        if np.max(np.abs(q)) > 10:
            raise ValueError("Printed marker coordinates are outside a plausible LCD layout")
        corners[str(marker_id)] = q.tolist()
    centers = np.array([np.mean(corners[str(PRINTED_IDS[k])], axis=0) for k in _ORDER])
    _quad(centers, label="Printed corner placement", min_area=.20)
    if not (centers[0, 0] < centers[1, 0] and centers[3, 0] < centers[2, 0]
            and centers[0, 1] < centers[3, 1] and centers[1, 1] < centers[2, 1]):
        raise ValueError("Place printed IDs20/21/22/23 at TL/TR/BR/BL")
    aspect = _number(layout.get("viewportAspectRatio"), "Viewport aspect ratio")
    if not .1 <= aspect <= 10:
        raise ValueError("Invalid viewport aspect ratio")
    result = {"schemaVersion": 1, "dictionary": DICTIONARY, "markerIds": dict(PRINTED_IDS),
              "markerCorners": corners, "viewportAspectRatio": aspect}
    method = layout.get("calibrationMethod", "co-visible")
    if method not in _METHODS:
        raise ValueError("Unknown printed marker calibration method")
    result["calibrationMethod"] = method
    if "calibratedAt" in layout:
        if not isinstance(layout["calibratedAt"], str) or len(layout["calibratedAt"]) > 100:
            raise ValueError("Invalid printed marker calibration date")
        result["calibratedAt"] = layout["calibratedAt"]
    return result


def layout_fingerprint(layout):
    value = validate_layout(layout)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def detections(corners, ids, relevant=None):
    """Decode OpenCV arrays without silently overwriting repeated IDs."""
    if ids is None or corners is None:
        return {}
    flat = np.asarray(ids).reshape(-1)
    if len(flat) != len(corners):
        raise ValueError("Marker IDs and corner counts differ")
    result = {}
    for value, points in zip(flat, corners):
        marker_id = int(value)
        if relevant is not None and marker_id not in relevant:
            continue
        if marker_id in result:
            raise ValueError(f"Duplicate ArUco ID{marker_id}; hide or remove its duplicate")
        result[marker_id] = _quad(points, label=f"Detected marker {marker_id}", min_area=4.)
    return result


def _layout(corners, aspect, method):
    return validate_layout({"schemaVersion": 1, "dictionary": DICTIONARY,
                            "markerIds": dict(PRINTED_IDS), "markerCorners": corners,
                            "viewportAspectRatio": aspect, "calibrationMethod": method})


def calibrate_layout(corners, ids, view):
    """Learn paper corners from eight co-visible markers in this exact frame."""
    width, height, left, top, right, bottom = viewport(view)
    found = detections(corners, ids, set(PRINTED_IDS.values()) | set(SCREEN_IDS.values()))
    if not (set(PRINTED_IDS.values()) | set(SCREEN_IDS.values())).issubset(found):
        raise ValueError("Keep all four screen markers and all four paper markers visible together")
    source = np.array([found[SCREEN_IDS[key]][i] for i, key in enumerate(_ORDER)])
    _quad(source, label="Screen registration", min_area=64.)
    destination = np.array([[left / width, top / height], [right / width, top / height],
                            [right / width, bottom / height], [left / width, bottom / height]])
    H = cv2.getPerspectiveTransform(source.astype(np.float32), destination.astype(np.float32))
    result = {}
    for marker_id in PRINTED_IDS.values():
        result[str(marker_id)] = cv2.perspectiveTransform(found[marker_id].reshape(1, 4, 2), H)[0].tolist()
    return _layout(result, width / height, "co-visible")


def guide_aligned_layout(view, marker_size, inset):
    """Bind user-declared paper placement over the current digital guide tiles.

    This computes the known black-square corners, excluding the white quiet
    margin. It does not prove that the user put the paper in those positions.
    """
    width, height, left, top, right, bottom = viewport(view)
    size, inset = _number(marker_size, "Marker size"), _number(inset, "Marker inset")
    if not 72 <= size <= 240 or not 0 <= inset <= 80:
        raise ValueError("Waiting for valid current marker guide settings")
    quiet = size / 6
    expected = [inset + quiet, inset + quiet, width - inset - quiet, height - inset - quiet]
    if max(abs(a - b) for a, b in zip((left, top, right, bottom), expected)) > .75:
        raise ValueError("Current marker guide settings and viewport registration do not match")
    black = size - 2 * quiet
    positions = {20: (left, top), 21: (right - black, top),
                 22: (right - black, bottom - black), 23: (left, bottom - black)}
    corners = {str(marker_id): [[x / width, y / height], [(x + black) / width, y / height],
                                [(x + black) / width, (y + black) / height], [x / width, (y + black) / height]]
               for marker_id, (x, y) in positions.items()}
    return _layout(corners, width / height, "guide-aligned-user-declared")


def target_corners(layout, view, warp_size):
    layout = validate_layout(layout)
    width, height, left, top, right, bottom = viewport(view)
    if abs(width / height / layout["viewportAspectRatio"] - 1) > .01:
        raise ValueError("Foundry viewport aspect changed; use fullscreen on the calibrated TV or register again")
    ww, wh = (_number(v, "Warp size") for v in warp_size)
    if min(ww, wh) < 8:
        raise ValueError("Invalid registration warp size")
    result = {}
    for key, value in layout["markerCorners"].items():
        q = np.asarray(value, np.float64)
        result[int(key)] = np.column_stack(((q[:, 0] * width - left) / (right - left) * (ww - 1),
                                            (q[:, 1] * height - top) / (bottom - top) * (wh - 1)))
    return result


def solve_homography(corners, ids, layout, view, warp_size, *, max_error_px=4.):
    """Fit all sixteen corners and require every card to agree, including outliers."""
    expected = target_corners(layout, view, warp_size)
    found = detections(corners, ids, set(PRINTED_IDS.values()))
    if len(found) != 4:
        return None
    source = np.vstack([found[i] for i in PRINTED_IDS.values()])
    destination = np.vstack([expected[i] for i in PRINTED_IDS.values()])
    H, _inliers = cv2.findHomography(source, destination, cv2.RANSAC, 3.0)
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
        return None
    projected = cv2.perspectiveTransform(source.reshape(1, -1, 2), H)[0]
    errors = np.linalg.norm(projected - destination, axis=1).reshape(4, 4)
    if np.max(errors) > max_error_px or np.max(np.sqrt(np.mean(errors ** 2, axis=1))) > 2.5:
        return None
    # Four displaced card centers can deceptively define a new projectivity:
    # their small local squares alone give only weak residuals. Independently
    # predict each card using the other three cards before accepting a refresh.
    # The larger allowance accounts for extrapolation from three corner groups.
    for withheld in range(4):
        keep = np.repeat(np.arange(4) != withheld, 4)
        trial, _ = cv2.findHomography(source[keep], destination[keep], cv2.RANSAC, 3.0)
        if trial is None:
            return None
        predicted = cv2.perspectiveTransform(source[~keep].reshape(1, 4, 2), trial)[0]
        if np.max(np.linalg.norm(predicted - destination[~keep], axis=1)) > 3 * max_error_px:
            return None
    return H.astype(np.float32)


def agrees_with_homography(corners, ids, H, layout, view, warp_size, *, max_error_px=4.):
    if H is None:
        return False
    expected = target_corners(layout, view, warp_size)
    found = detections(corners, ids, set(PRINTED_IDS.values()))
    if len(found) < 3:
        return False
    for marker_id, points in found.items():
        projected = cv2.perspectiveTransform(points.reshape(1, 4, 2), np.asarray(H, np.float64))[0]
        if np.max(np.linalg.norm(projected - expected[marker_id], axis=1)) > max_error_px:
            return False
    return True


def roi_masks(image_shape, H, warp_size):
    """Mask the mapped LCD registration rectangle, never the paper/bezel hull."""
    height, width = image_shape[:2]
    ww, wh = map(int, warp_size)
    corners = np.array([[[0., 0.], [ww - 1., 0.], [ww - 1., wh - 1.], [0., wh - 1.]]], np.float32)
    try:
        source = cv2.perspectiveTransform(corners, np.linalg.inv(H))[0]
    except np.linalg.LinAlgError as exc:
        raise ValueError("Printed registration is singular") from exc
    _quad(source, label="Mapped LCD rectangle", min_area=64.)
    camera = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(camera, np.rint(source).astype(np.int32), 255)
    return camera, np.full((wh, ww), 255, np.uint8)
