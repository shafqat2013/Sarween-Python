"""Named camera calibration and physical-display profiles.

Importing this module, constructing a store, and reading profiles never write
files. Selection is explicit; legacy camera files are never selected or applied
automatically. The registry is one atomic JSON document shared by the app and
camera process.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import threading
from uuid import UUID, uuid4

from app_paths import atomic_write_json, data_path


SCHEMA_VERSION = 1
PROFILE_FILENAME = "hardware_profiles.json"
DISPLAY_SETTING_RULES = {
    "targetGridInches": (0.25, 3),
    "displayDiagonalInches": (10, 150),
    "physicalGridCalibration": (0.5, 1.5),
    "viewportMarkersEnabled": bool,
    "viewportMarkerSize": (72, 240),
    "viewportMarkerInset": (0, 80),
    "physicalGridAutoApply": bool,
}
_KINDS = {"camera": ("cameras", "activeCameraId"), "display": ("displays", "activeDisplayId")}
_CAMERA_FIELDS = {"name", "deviceLabel", "sourceHint", "lens", "zoom", "transformNotes", "resolution", "bindingVerified", "calibration"}
_DISPLAY_FIELDS = {"name", "settings", "targetDisplayId", "printedMarkerLayout", "printedMarkersEnabled"}
_BINDING_FIELDS = {"deviceLabel", "sourceHint", "lens", "zoom", "transformNotes", "resolution"}
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class ProfileStoreError(ValueError):
    """A profile or registry cannot safely be used or replaced."""


def _empty_registry() -> dict:
    return {"schemaVersion": SCHEMA_VERSION, "generation": 0,
            "activeCameraId": None, "activeDisplayId": None, "cameras": [], "displays": []}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _invalid_json_number(value):
    raise ProfileStoreError(f"Invalid JSON number: {value}")


def _namespace(kind: str) -> tuple[str, str]:
    try:
        return _KINDS[kind]
    except (KeyError, TypeError):
        raise ProfileStoreError("Profile kind must be 'camera' or 'display'.") from None


def _name(value) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 100:
        raise ProfileStoreError("Profile names must contain 1–100 characters.")
    if any(ord(character) < 32 for character in value):
        raise ProfileStoreError("Profile names must fit on one line.")
    return value.strip()


def _text(value, field: str, limit: int = 4096) -> str:
    if not isinstance(value, str) or len(value) > limit or "\0" in value:
        raise ProfileStoreError(f"Invalid {field}.")
    return value


def _image_size(value, *, optional: bool = False):
    if value is None and optional:
        return None
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ProfileStoreError("Image size must be [width, height].")
    if any(isinstance(part, bool) or not isinstance(part, Integral) or part <= 0 for part in value):
        raise ProfileStoreError("Image width and height must be positive integers.")
    return [int(part) for part in value]


def _number(value, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ProfileStoreError(f"Invalid {description}: expected a finite number.")
    return float(value)


def _array(value):
    return value.tolist() if hasattr(value, "tolist") else value


def _calibration(value, *, allow_unknown_size: bool = False) -> dict:
    if not isinstance(value, dict):
        raise ProfileStoreError("Camera calibration must be an object.")
    matrix = _array(value.get("cameraMatrix"))
    if not isinstance(matrix, (tuple, list)) or len(matrix) != 3:
        raise ProfileStoreError("Camera matrix must be 3×3.")
    rows = []
    for row in matrix:
        if not isinstance(row, (tuple, list)) or len(row) != 3:
            raise ProfileStoreError("Camera matrix must be 3×3.")
        rows.append([_number(part, "camera matrix") for part in row])
    if rows[0][0] <= 0 or rows[1][1] <= 0 or any(abs(rows[2][i] - expected) > 1e-9 for i, expected in enumerate((0, 0, 1))):
        raise ProfileStoreError("Camera matrix has invalid focal lengths or homogeneous coordinates.")
    distortion = _array(value.get("distCoeffs"))
    if isinstance(distortion, (tuple, list)) and distortion and isinstance(distortion[0], (tuple, list)):
        if len(distortion) == 1:
            distortion = distortion[0]
        elif all(isinstance(row, (tuple, list)) and len(row) == 1 for row in distortion):
            distortion = [row[0] for row in distortion]
    if not isinstance(distortion, (tuple, list)) or len(distortion) not in (4, 5, 8, 12, 14):
        raise ProfileStoreError("Distortion coefficients must contain 4, 5, 8, 12, or 14 values.")
    coefficients = [_number(part, "distortion coefficient") for part in distortion]
    rms = value.get("rms")
    if rms is not None:
        rms = _number(rms, "calibration RMS")
        if rms < 0:
            raise ProfileStoreError("Calibration RMS cannot be negative.")
    return {"cameraMatrix": rows, "distCoeffs": coefficients,
            "imageSize": _image_size(value.get("imageSize"), optional=allow_unknown_size),
            "createdAt": _text(value.get("createdAt", _now()), "calibration date", 100), "rms": rms}


def _display_settings(value: dict) -> dict:
    if not isinstance(value, dict) or not value or set(value) - set(DISPLAY_SETTING_RULES):
        raise ProfileStoreError("Display profiles require known grid or marker settings.")
    result = {}
    for key, part in value.items():
        rule = DISPLAY_SETTING_RULES[key]
        if rule is bool:
            if type(part) is not bool:
                raise ProfileStoreError(f"{key} must be true or false.")
            result[key] = part
        else:
            number = _number(part, key)
            if not rule[0] <= number <= rule[1]:
                raise ProfileStoreError(f"{key} is outside its supported range.")
            result[key] = number
    return result


def _printed_marker_values(profile):
    """Validate optional paper-marker calibration without migrating old files."""
    enabled = profile.get("printedMarkersEnabled", False)
    if type(enabled) is not bool:
        raise ProfileStoreError("Printed marker selection must be true or false.")
    layout = profile.get("printedMarkerLayout")
    if layout is not None:
        if not isinstance(layout, dict):
            raise ProfileStoreError("Printed marker layout must be an object or null.")
        from printed_markers import validate_layout
        try:
            layout = validate_layout(deepcopy(layout))
        except (TypeError, ValueError) as exc:
            raise ProfileStoreError(f"Invalid printed marker layout: {exc}") from exc
    if enabled and layout is None:
        raise ProfileStoreError("Register printed markers before using them.")
    return layout, enabled


class HardwareProfiles:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else data_path(PROFILE_FILENAME)
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.path.absolute()), threading.RLock())
        self._cached = None
        self._signature = None
        self._resolved = {}

    def _stat_signature(self):
        if self.path.is_symlink():
            raise ProfileStoreError("The hardware profile registry must be a regular file.")
        try:
            info = self.path.stat()
        except FileNotFoundError:
            return None
        if not self.path.is_file():
            raise ProfileStoreError("The hardware profile registry must be a regular file.")
        return info.st_mtime_ns, info.st_ctime_ns, info.st_size, info.st_ino

    def _read_file(self):
        self._stat_signature()
        try:
            with self.path.open(encoding="utf-8") as stream:
                value = json.load(stream, parse_constant=_invalid_json_number)
        except FileNotFoundError:
            return _empty_registry()
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise ProfileStoreError(f"Cannot read hardware profiles: {exc}") from exc
        if not isinstance(value, dict) or type(value.get("schemaVersion")) is not int or value["schemaVersion"] != SCHEMA_VERSION:
            raise ProfileStoreError("Unsupported hardware profile registry format.")
        if type(value.get("generation")) is not int or value["generation"] < 0:
            raise ProfileStoreError("Invalid hardware profile generation.")
        for collection, active_key in _KINDS.values():
            if not isinstance(value.get(collection), list):
                raise ProfileStoreError(f"Invalid {collection} profile collection.")
            ids, names = set(), set()
            for profile in value[collection]:
                if not isinstance(profile, dict):
                    raise ProfileStoreError("Invalid hardware profile.")
                try:
                    profile_id = str(UUID(profile["id"]))
                    name = _name(profile["name"])
                except (KeyError, ValueError, TypeError, AttributeError) as exc:
                    raise ProfileStoreError("Invalid hardware profile ID or name.") from exc
                if profile["id"] != profile_id or profile["name"] != name or profile_id in ids or name.casefold() in names:
                    raise ProfileStoreError("Hardware profile IDs and names must be unique.")
                ids.add(profile_id)
                names.add(name.casefold())
                if collection == "displays":
                    layout, _enabled = _printed_marker_values(profile)
                    if "printedMarkerLayout" in profile:
                        profile["printedMarkerLayout"] = layout
            if active_key not in value or (value[active_key] is not None and not isinstance(value[active_key], str)):
                raise ProfileStoreError("Invalid selected hardware profile ID.")
        return value

    def _read_cached(self):
        with self._lock:
            signature = self._stat_signature()
            if self._cached is None or signature != self._signature:
                self._cached = self._read_file()
                self._signature = signature
                self._resolved.clear()
            return self._cached

    @contextmanager
    def _write_lock(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        if lock_path.is_symlink():
            raise ProfileStoreError("The hardware profile lock must be a regular file.")
        with lock_path.open("a+b") as stream:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                if not stream.read(1):
                    stream.write(b"\0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(stream, fcntl.LOCK_UN)

    def _mutate(self, operation):
        with self._lock, self._write_lock():
            previous = self._read_file()
            candidate = deepcopy(previous)
            result = operation(candidate)
            if candidate != previous:
                candidate["generation"] = previous["generation"] + 1
                atomic_write_json(self.path, candidate, backup=False)
            self._cached = candidate
            self._signature = self._stat_signature()
            self._resolved.clear()
            return deepcopy(result)

    def snapshot(self) -> dict:
        return deepcopy(self._read_cached())

    def revision(self) -> tuple:
        """Read-only cache key, refreshed when another process replaces the file."""
        with self._lock:
            registry = self._read_cached()
            return self._signature, registry["generation"]

    def list_profiles(self, kind: str) -> list[dict]:
        collection, _active = _namespace(kind)
        return deepcopy(self._read_cached()[collection])

    def get_profile(self, kind: str, profile_id: str) -> dict | None:
        collection, _active = _namespace(kind)
        return deepcopy(next((profile for profile in self._read_cached()[collection] if profile["id"] == profile_id), None))

    def active_profile(self, kind: str) -> dict | None:
        collection, active = _namespace(kind)
        registry = self._read_cached()
        return deepcopy(next((profile for profile in registry[collection] if profile["id"] == registry[active]), None))

    def upsert_profile(self, kind: str, values: dict, *, profile_id: str | None = None, select: bool = False) -> dict:
        return self._upsert_profile(kind, values, profile_id=profile_id, select=select)

    def _upsert_profile(self, kind: str, values: dict, *, profile_id=None, select=False,
                        calibration_write=False, expected_binding=None) -> dict:
        collection, active = _namespace(kind)
        fields = _CAMERA_FIELDS if kind == "camera" else _DISPLAY_FIELDS
        if not isinstance(values, dict) or set(values) - fields:
            raise ProfileStoreError("Unknown hardware profile fields.")
        if kind == "camera" and not calibration_write and (values.get("bindingVerified") is True or values.get("calibration") is not None):
            raise ProfileStoreError("Only a new camera calibration can verify a camera binding. Run camera calibration first.")
        if type(select) is not bool:
            raise ProfileStoreError("Profile selection must be true or false.")
        if expected_binding is not None and not isinstance(expected_binding, dict):
            raise ProfileStoreError("Expected camera binding must be a profile snapshot.")
        expected_binding = deepcopy(expected_binding)

        def update(registry):
            existing = next((profile for profile in registry[collection] if profile["id"] == profile_id), None)
            if profile_id is not None and existing is None:
                raise ProfileStoreError("The hardware profile no longer exists.")
            if expected_binding is not None and (existing is None or any(
                    existing.get(key) != expected_binding.get(key) for key in _BINDING_FIELDS)):
                raise ProfileStoreError("The camera profile changed during calibration. Check its lens and feed settings, then calibrate again.")
            stamp = _now()
            if existing is not None:
                profile = deepcopy(existing)
            elif kind == "camera":
                profile = {"id": str(uuid4()), "createdAt": stamp, "deviceLabel": "", "sourceHint": "", "lens": "", "zoom": "",
                           "transformNotes": "", "resolution": None, "bindingVerified": False, "calibration": None}
            else:
                profile = {"id": str(uuid4()), "createdAt": stamp, "settings": {}, "targetDisplayId": None,
                           "printedMarkerLayout": None, "printedMarkersEnabled": False}
            profile.update(deepcopy(values))
            profile["name"] = _name(profile.get("name"))
            if any(other["id"] != profile["id"] and other["name"].casefold() == profile["name"].casefold() for other in registry[collection]):
                raise ProfileStoreError("A profile with that name already exists.")
            if kind == "camera":
                for key in ("deviceLabel", "sourceHint", "lens", "zoom", "transformNotes"):
                    if key == "zoom" and isinstance(profile[key], Real) and not isinstance(profile[key], bool):
                        profile[key] = str(_number(profile[key], "camera zoom"))
                    profile[key] = _text(profile[key], key)
                profile["resolution"] = _image_size(profile.get("resolution"), optional=True)
                if type(profile.get("bindingVerified")) is not bool:
                    raise ProfileStoreError("Camera binding verification must be true or false.")
                if "calibration" in values and profile["calibration"] is not None:
                    profile["calibration"] = _calibration(profile["calibration"])
                changed_binding = existing is not None and any(profile.get(key) != existing.get(key) for key in _BINDING_FIELDS)
                if changed_binding and "calibration" not in values:
                    profile["bindingVerified"] = False
            else:
                profile["settings"] = _display_settings({**(existing or {}).get("settings", {}), **profile["settings"]})
                if profile.get("targetDisplayId") is not None:
                    profile["targetDisplayId"] = _text(profile["targetDisplayId"], "target display ID", 512)
                profile["printedMarkerLayout"], profile["printedMarkersEnabled"] = _printed_marker_values(profile)
            profile["updatedAt"] = stamp
            if existing is None:
                registry[collection].append(profile)
            else:
                registry[collection][registry[collection].index(existing)] = profile
            if select:
                registry[active] = profile["id"]
            return profile

        return self._mutate(update)

    def select_profile(self, kind: str, profile_id: str | None) -> None:
        collection, active = _namespace(kind)

        def select(registry):
            if profile_id is not None and not any(profile["id"] == profile_id for profile in registry[collection]):
                raise ProfileStoreError("The selected hardware profile no longer exists.")
            registry[active] = profile_id

        self._mutate(select)

    def save_printed_marker_layout(self, profile_id, layout, *, expected_profile=None):
        """Save alignment only to the unchanged, still-active display profile."""
        if expected_profile is not None and not isinstance(expected_profile, dict):
            raise ProfileStoreError("Expected display profile must be a profile snapshot.")
        expected_profile = deepcopy(expected_profile)
        canonical, _enabled = _printed_marker_values({"printedMarkerLayout": layout, "printedMarkersEnabled": True})

        def save(registry):
            existing = next((profile for profile in registry["displays"] if profile["id"] == profile_id), None)
            if existing is None or registry["activeDisplayId"] != profile_id:
                raise ProfileStoreError("The active display profile changed during printed marker alignment.")
            if expected_profile is not None and existing != expected_profile:
                raise ProfileStoreError("The display profile changed during printed marker alignment. Align again.")
            profile = deepcopy(existing)
            profile.update(printedMarkerLayout=deepcopy(canonical), printedMarkersEnabled=True, updatedAt=_now())
            registry["displays"][registry["displays"].index(existing)] = profile
            return profile

        return self._mutate(save)

    def set_printed_markers_enabled(self, profile_id, enabled, *, expected_profile=None):
        """Change mode only for the unchanged display selected by this action."""
        if type(enabled) is not bool:
            raise ProfileStoreError("Printed marker selection must be true or false.")
        if expected_profile is not None and not isinstance(expected_profile, dict):
            raise ProfileStoreError("Expected display profile must be a profile snapshot.")
        expected_profile = deepcopy(expected_profile)

        def select(registry):
            existing = next((profile for profile in registry["displays"] if profile["id"] == profile_id), None)
            if existing is None or registry["activeDisplayId"] != profile_id:
                raise ProfileStoreError("The active display profile changed before switching printed markers.")
            if expected_profile is not None and existing != expected_profile:
                raise ProfileStoreError("The display profile changed. Check it before switching printed markers.")
            profile = deepcopy(existing)
            layout, _flag = _printed_marker_values({**profile, "printedMarkersEnabled": enabled})
            profile.update(printedMarkerLayout=layout, printedMarkersEnabled=enabled, updatedAt=_now())
            registry["displays"][registry["displays"].index(existing)] = profile
            return profile

        return self._mutate(select)

    def rename_profile(self, kind: str, profile_id: str, name: str) -> dict:
        return self.upsert_profile(kind, {"name": name}, profile_id=profile_id)

    def delete_profile(self, kind: str, profile_id: str) -> None:
        collection, active = _namespace(kind)

        def delete(registry):
            profile = next((profile for profile in registry[collection] if profile["id"] == profile_id), None)
            if profile is None:
                raise ProfileStoreError("The hardware profile no longer exists.")
            registry[collection].remove(profile)
            if registry[active] == profile_id:
                registry[active] = None

        self._mutate(delete)

    def save_camera_calibration(self, profile_id, matrix, dist_coeffs, image_size, *, rms=None,
                                binding_verified=True, expected_binding=None) -> dict:
        calibration = _calibration({"cameraMatrix": matrix, "distCoeffs": dist_coeffs, "imageSize": image_size, "rms": rms})
        return self._upsert_profile("camera", {"calibration": calibration, "resolution": calibration["imageSize"],
                                              "bindingVerified": binding_verified}, profile_id=profile_id,
                                    calibration_write=True, expected_binding=expected_binding)

    def resolve_camera_calibration(self, image_size, *, profile_id=None) -> tuple[dict | None, str]:
        """Resolve only the explicit profile and exact final-feed dimensions.

        ``None`` uses the selected camera ID. Callers freezing no selection for a
        running camera session should retain that choice without calling again.
        Source/lens/OBS-transform binding is a user-confirmed profile choice;
        this method never guesses a camera from its device index or label.
        """
        try:
            size = _image_size(image_size)
            registry = self._read_cached()
            selected = profile_id if profile_id is not None else registry["activeCameraId"]
            if selected is None:
                return None, "No camera calibration selected."
            key = (registry["generation"], selected, *size)
            if key in self._resolved:
                return deepcopy(self._resolved[key])
            profile = next((item for item in registry["cameras"] if item["id"] == selected), None)
            if profile is None:
                return None, "The selected camera profile is missing. Choose a camera profile."
            label = profile["name"]
            if profile.get("calibration") is None:
                result = None, f"{label}: no camera calibration saved."
            elif profile.get("bindingVerified") is not True:
                result = None, f"{label}: camera binding is unverified. Calibrate the selected final camera feed."
            else:
                calibration = _calibration(profile["calibration"])
                expected = _image_size(profile.get("resolution"), optional=True)
                if calibration["imageSize"] != size:
                    width, height = calibration["imageSize"]
                    result = None, f"{label}: calibration expects {width}×{height}; current feed is {size[0]}×{size[1]}. Recalibrate this feed."
                elif expected is not None and expected != size:
                    result = None, f"{label}: profile expects {expected[0]}×{expected[1]}; current feed is {size[0]}×{size[1]}. Recalibrate this feed."
                else:
                    result = calibration, f"Using camera calibration: {label} ({size[0]}×{size[1]})."
            if len(self._resolved) >= 16:
                self._resolved.clear()
            self._resolved[key] = result
            return deepcopy(result)
        except (OSError, ValueError, TypeError) as exc:
            return None, f"Camera calibration unavailable: {exc}"

    def import_legacy_camera(self, matrix_path=None, distortion_path=None, *, image_size=None) -> dict | None:
        """Explicitly preserve global NPYs as an inactive, unverified profile."""
        matrix_path = Path(matrix_path) if matrix_path is not None else data_path("camera_matrix.npy")
        distortion_path = Path(distortion_path) if distortion_path is not None else data_path("dist_coeffs.npy")
        if not matrix_path.exists() and not distortion_path.exists():
            return None
        if any(path.is_symlink() or not path.is_file() for path in (matrix_path, distortion_path)):
            raise ProfileStoreError("Both legacy calibration files must be regular NPY files.")
        import numpy as np
        try:
            calibration = _calibration({"cameraMatrix": np.load(matrix_path, allow_pickle=False),
                                        "distCoeffs": np.load(distortion_path, allow_pickle=False),
                                        "imageSize": image_size}, allow_unknown_size=True)
        except (OSError, ValueError, TypeError) as exc:
            raise ProfileStoreError(f"Cannot import legacy camera calibration: {exc}") from exc
        fingerprint = hashlib.sha256(matrix_path.read_bytes() + b"\0" + distortion_path.read_bytes()).hexdigest()

        def import_profile(registry):
            for existing in registry["cameras"]:
                if existing.get("legacySource", {}).get("sha256") == fingerprint:
                    return existing
            base = "Legacy camera (unverified)"
            name = base
            suffix = 2
            names = {profile["name"].casefold() for profile in registry["cameras"]}
            while name.casefold() in names:
                name = f"{base} {suffix}"
                suffix += 1
            stamp = _now()
            profile = {"id": str(uuid4()), "name": name, "createdAt": stamp, "updatedAt": stamp,
                       "deviceLabel": "", "sourceHint": "Legacy global calibration", "lens": "", "zoom": "",
                       "transformNotes": "Original calibration source and transforms were not recorded.",
                       "resolution": calibration["imageSize"], "bindingVerified": False, "calibration": calibration,
                       "legacySource": {"matrixPath": str(matrix_path), "distortionPath": str(distortion_path), "sha256": fingerprint}}
            registry["cameras"].append(profile)
            return profile

        return self._mutate(import_profile)
