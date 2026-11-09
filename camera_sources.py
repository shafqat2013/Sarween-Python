"""Explicit camera sources: native OpenCV devices or the wired phone companion.

Discovery only lists USB attachments; it never connects to the phone app or
opens a camera. Reserved negative indices are stable UI handles, never passed
to OpenCV (where negative values can select an unintended default camera).
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import cv2

PHONE_SOURCE = "sarween-phone-usb"
PHONE_PROFILE_HINT = "Sarween Camera USB"
PHONE_LENS_LABELS = {"wide": "Wide", "ultra-wide": "Ultra Wide", "telephoto": "Telephoto"}
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


def profile_matches_phone(profile, source, *, metadata=None):
    identity_matches = bool(profile and source.get("type") == PHONE_SOURCE
                and profile.get("sourceHint") == PHONE_PROFILE_HINT
                and profile.get("deviceLabel") == source.get("deviceId"))
    return identity_matches and (metadata is None or phone_profile_mismatch_reason(profile, source, metadata) is None)


def profile_compatible_with_source(profile, source, *, metadata=None):
    """Reject known USB/native mismatches; native pairings still need confirmation."""
    if source.get("type") == PHONE_SOURCE:
        return profile_matches_phone(profile, source, metadata=metadata)
    return bool(source.get("type") == "opencv" and
                (not profile or profile.get("sourceHint") != PHONE_PROFILE_HINT))


def phone_profile_lens(profile):
    """Recognize explicit old Wide labels without guessing from a USB serial."""
    label = "".join(character for character in str((profile or {}).get("lens", "")).lower() if character.isalpha())
    return {"wide": "wide", "backwide": "wide", "ultrawide": "ultra-wide",
            "backultrawide": "ultra-wide", "telephoto": "telephoto", "backtelephoto": "telephoto"}.get(label)


def phone_camera_lens(metadata):
    """Read actual negotiated physical-camera metadata, never inventory guesses."""
    fmt = metadata.get("format") if isinstance(metadata, dict) else None
    if (not isinstance(fmt, dict) or not isinstance(fmt.get("lens"), str)
            or fmt["lens"] not in PHONE_LENS_LABELS):
        raise ValueError("The iPhone's physical lens is unavailable. Open Sarween Camera and wait for a fresh feed before choosing a camera profile.")
    if fmt.get("cameraPosition", "back") != "back" or fmt.get("zoom", 1) != 1:
        raise ValueError("Use a physical back camera at 1x in Sarween Camera before choosing its calibration profile.")
    return fmt["lens"]


def phone_profile_mismatch_reason(profile, source, metadata):
    """Actionable physical-lens check for a current decoded phone stream."""
    if not profile_matches_phone(profile, source):
        return "Choose this iPhone's Sarween Camera USB profile. Native/OBS calibration belongs to a different feed."
    try:
        actual = phone_camera_lens(metadata)
    except ValueError as exc:
        return str(exc)
    if metadata.get("deviceId") != source.get("deviceId"):
        return "The current USB phone does not match the selected camera profile. Select this iPhone's own profile."
    expected = phone_profile_lens(profile)
    if expected != actual:
        actual_name = PHONE_LENS_LABELS[actual]
        expected_name = PHONE_LENS_LABELS.get(expected, "an unspecified lens")
        return (f"The iPhone is using {actual_name}, but profile '{profile.get('name', 'Camera')}' is for {expected_name}. "
                f"Select or create a {actual_name} profile and calibrate that lens, or switch the phone back to {expected_name}.")
    return None


def require_phone_profile_metadata(profile, source, metadata):
    reason = phone_profile_mismatch_reason(profile, source, metadata)
    if reason:
        raise ValueError(reason)
    return phone_camera_lens(metadata)


def create_phone_profile(store, source, *, metadata, name=None):
    if source.get("type") != PHONE_SOURCE:
        raise ValueError("Choose Sarween Camera on a USB iPhone first.")
    phone_camera_index(source.get("deviceId"))
    actual = phone_camera_lens(metadata)
    if metadata.get("deviceId") != source["deviceId"]:
        raise ValueError("The current iPhone does not match the selected USB camera.")
    fmt = metadata["format"]
    width, height = fmt.get("width"), fmt.get("height")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("Wait for the iPhone's negotiated frame size before creating its profile.")
    name = name or f"iPhone · {PHONE_LENS_LABELS[actual]} · Sarween Camera"
    names = {profile["name"].casefold() for profile in store.list_profiles("camera")}
    label, suffix = name, 2
    while label.casefold() in names:
        label = f"{name} {suffix}"
        suffix += 1
    return store.upsert_profile("camera", {"name": label,
        "deviceLabel": source["deviceId"], "sourceHint": PHONE_PROFILE_HINT,
        "lens": f"Back {actual}", "zoom": "1x", "resolution": [width, height],
        "transformNotes": (f"Sarween Camera USB; {fmt.get('orientation') or ('portrait' if height > width else 'landscape')}; "
                           "no crop; stabilization and geometric distortion correction off")})


def matching_phone_profiles(store, source, metadata):
    """Offer existing profiles for this actual physical lens and frame layout."""
    phone_camera_lens(metadata)
    fmt = metadata["format"]
    size = [fmt.get("width"), fmt.get("height")]
    return [profile for profile in store.list_profiles("camera")
            if profile_matches_phone(profile, source, metadata=metadata)
            and (profile.get("resolution") is None or profile["resolution"] == size)]


class PhoneMetadataProbe:
    """Cancellable scheduler-driven handshake for an explicit profile action.

    This polls the existing bounded receiver's status; it never consumes frames
    or runs a second camera reader, and it releases its connection on every exit.
    """
    def __init__(self, index, source, *, schedule, cancel_schedule, complete, failed,
                 guard=lambda: None, capture_factory=None, timeout=10., clock=time.monotonic):
        self.schedule, self.cancel_schedule = schedule, cancel_schedule
        self.complete, self.failed, self.guard = complete, failed, guard
        self.clock, self.deadline = clock, clock() + timeout
        self.job = None
        self.active = True
        self.capture = None
        try:
            guard()
            self.capture = (capture_factory or open_camera_capture)(index, source=source)
            self.job = schedule(0, self._poll)
        except BaseException:
            self.cancel()
            raise

    def cancel(self):
        self.active = False
        if self.job is not None:
            try:
                self.cancel_schedule(self.job)
            except Exception:
                pass
            self.job = None
        capture, self.capture = self.capture, None
        if capture is not None:
            capture.release()

    def _poll(self):
        self.job = None
        if not self.active:
            return
        try:
            self.guard()
            status = self.capture.status
            if status.get("fresh"):
                metadata = deepcopy(status)
                phone_camera_lens(metadata)
                self.cancel()
                self.complete(metadata)
            elif self.clock() >= self.deadline:
                raise ValueError("No fresh iPhone feed arrived. Open Sarween Camera, choose a lens, resume the camera, then retry Create iPhone profile.")
            else:
                self.job = self.schedule(100, self._poll)
        except Exception as exc:
            self.cancel()
            self.failed(str(exc))


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
                "geometricDistortionCorrection", "hdr", "colorSpace", "nativeFieldOfViewDegrees")}}
