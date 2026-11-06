"""Explicit camera sources: native OpenCV devices or the wired phone companion.

Discovery only lists USB attachments; it never connects to the phone app or
opens a camera. Reserved negative indices are stable UI handles, never passed
to OpenCV (where negative values can select an unintended default camera).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib

import cv2

PHONE_SOURCE = "sarween-phone-usb"
PHONE_PROFILE_HINT = "Sarween Camera USB"
_MIN_PHONE_INDEX = 100_000


def phone_camera_index(device_id):
    if not isinstance(device_id, str) or not device_id or len(device_id) > 256:
        raise ValueError("Choose a valid USB iPhone.")
    return -(_MIN_PHONE_INDEX + (int.from_bytes(hashlib.sha256(device_id.encode()).digest()[:4], "big") & 0x3fffffff))


def is_phone_camera_index(index):
    return type(index) is int and index <= -_MIN_PHONE_INDEX


def phone_camera_inventory(*, devices=None):
    if devices is None:
        from phone_usb import list_phone_cameras
        devices = list_phone_cameras()
    rows, seen = [], set()
    for device in devices:
        identity = device.get("id")
        if device.get("transport") != "usb" or not isinstance(identity, str) or not identity:
            continue
        index = phone_camera_index(identity)
        if index in seen:
            continue
        seen.add(index)
        rows.append({"index": index, "model": "Sarween Camera · " + str(device.get("label") or "iPhone"),
                     "resolution": "USB · open Sarween Camera on iPhone", "enumeration_source": PHONE_SOURCE,
                     "availability": "not_opened", "camera_source": {"type": PHONE_SOURCE, "deviceId": identity},
                     "inventory_note": "Open Sarween Camera on your iPhone. Use its own camera profile; OBS calibration is a different feed."})
    return rows


def source_for_index(index, *, selection=None, inventory=None):
    if type(index) is not int:
        raise ValueError("Choose a camera from the available inputs.")
    if not is_phone_camera_index(index):
        if index < 0:
            raise ValueError("Unknown camera input. Refresh camera choices.")
        return {"type": "opencv", "index": index}
    if selection and selection.get("webcam_index") == index:
        source = selection.get("camera_source")
        if (isinstance(source, dict) and source.get("type") == PHONE_SOURCE
                and phone_camera_index(source.get("deviceId")) == index):
            return deepcopy(source)
    rows = phone_camera_inventory() if inventory is None else inventory
    matches = [row["camera_source"] for row in rows if row.get("index") == index and row.get("camera_source")]
    if len(matches) != 1:
        raise ValueError("Reconnect the selected iPhone by USB and refresh camera choices.")
    return deepcopy(matches[0])


def open_camera_capture(index, *, selection=None, source=None):
    index = int(index)
    source = source or source_for_index(index, selection=selection)
    if source.get("type") == PHONE_SOURCE:
        identity = source.get("deviceId")
        if phone_camera_index(identity) != index:
            raise ValueError("The selected iPhone does not match this camera input.")
        from phone_camera import PhoneCameraCapture
        return PhoneCameraCapture(identity)
    if index < 0 or source.get("type") != "opencv":
        raise ValueError("Unknown camera input.")
    return cv2.VideoCapture(index)


def profile_matches_phone(profile, source):
    return bool(profile and source.get("type") == PHONE_SOURCE
                and profile.get("sourceHint") == PHONE_PROFILE_HINT
                and profile.get("deviceLabel") == source.get("deviceId"))


def profile_compatible_with_source(profile, source):
    """Reject known USB/native mismatches; native pairings still need confirmation."""
    if source.get("type") == PHONE_SOURCE:
        return profile_matches_phone(profile, source)
    return bool(source.get("type") == "opencv" and
                (not profile or profile.get("sourceHint") != PHONE_PROFILE_HINT))


def create_phone_profile(store, source, *, name="iPhone · Sarween Camera"):
    if source.get("type") != PHONE_SOURCE:
        raise ValueError("Choose Sarween Camera on a USB iPhone first.")
    phone_camera_index(source.get("deviceId"))
    names = {profile["name"].casefold() for profile in store.list_profiles("camera")}
    label, suffix = name, 2
    while label.casefold() in names:
        label = f"{name} {suffix}"
        suffix += 1
    return store.upsert_profile("camera", {"name": label,
        "deviceLabel": source["deviceId"], "sourceHint": PHONE_PROFILE_HINT,
        "lens": "Back wide", "zoom": "1x", "resolution": [1080, 1920],
        "transformNotes": "Sarween Camera USB; portrait; no crop; stabilization and geometric distortion correction off"})


def stable_stream_key(metadata):
    """Stream history identity excludes changing timestamps, ages and counters."""
    if not metadata:
        return None
    fmt = metadata.get("format") or {}
    return {"source": PHONE_SOURCE, "deviceId": metadata.get("deviceId"),
            "sourceId": metadata.get("sourceId"),
            "sessionId": metadata.get("sessionId"),
            "connectionGeneration": metadata.get("connectionGeneration"),
            "width": fmt.get("width"), "height": fmt.get("height"),
            "pixelFormat": fmt.get("pixelFormat"), "rotationDegrees": 0, "mirrored": False,
            "optics": {key: fmt.get(key) for key in (
                "orientation", "cameraPosition", "lens", "zoom", "stabilization",
                "geometricDistortionCorrection", "hdr", "colorSpace")}}
