"""Persistent player-mini metadata and verified appearance samples."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


LIBRARY_PATH = Path(__file__).with_name("mini_library.json")
SCHEMA_VERSION = 1

DEFAULT_MINIS = (
    {"id": "red10", "name": "Red", "ringColor": "red", "tokenName": "Red"},
    {"id": "blue", "name": "Blue", "ringColor": "blue", "tokenName": "Blue"},
    {"id": "yellow", "name": "Yellow", "ringColor": "yellow", "tokenName": "Yellow"},
    {"id": "green", "name": "Green", "ringColor": "green", "tokenName": "Green"},
    {"id": "white", "name": "White", "ringColor": "white", "tokenName": "White"},
)


def _default_entry(spec: Mapping[str, str]) -> Dict[str, Any]:
    return {
        "name": spec["name"],
        "ringColor": spec["ringColor"],
        "tokenName": spec["tokenName"],
        "enabled": True,
        "samples": [],
    }


def default_library() -> Dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "minis": {spec["id"]: _default_entry(spec) for spec in DEFAULT_MINIS},
    }


def _normalize_library(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        data = {}
    result = default_library()
    supplied_minis = data.get("minis")
    if isinstance(supplied_minis, dict):
        for mini_id, supplied in supplied_minis.items():
            if not isinstance(supplied, dict):
                continue
            mini_id = str(mini_id).strip()
            if not mini_id:
                continue
            current = result["minis"].setdefault(
                mini_id,
                {
                    "name": mini_id,
                    "ringColor": "unknown",
                    "tokenName": mini_id,
                    "enabled": True,
                    "samples": [],
                },
            )
            for field in ("name", "ringColor", "tokenName", "enabled"):
                if field in supplied:
                    current[field] = supplied[field]
            samples = supplied.get("samples")
            if isinstance(samples, list):
                current["samples"] = [
                    copy.deepcopy(sample) for sample in samples if isinstance(sample, dict)
                ]
    return result


def load_library(path: Path = LIBRARY_PATH) -> Dict[str, Any]:
    try:
        return _normalize_library(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return default_library()
    except Exception as exc:
        print(f"MINI_LIBRARY | Could not load {path}: {exc}")
        return default_library()


def save_library(library: Mapping[str, Any], path: Path = LIBRARY_PATH) -> None:
    normalized = _normalize_library(copy.deepcopy(dict(library)))
    path.write_text(json.dumps(normalized, indent=2) + "\n", encoding="utf-8")


def _valid_lab(lab: Iterable[Any]) -> list[float]:
    values = [float(value) for value in lab]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("A scan sample must contain three finite Lab values")
    return values


def _sample_id(mini_id: str, lab: Iterable[float], source: str, condition: str) -> str:
    values = ",".join(f"{value:.4f}" for value in lab)
    digest = hashlib.sha1(
        f"{mini_id}|{values}|{source}|{condition}".encode("utf-8")
    ).hexdigest()[:12]
    return f"sample-{digest}"


def _contains_lab(samples: Iterable[Mapping[str, Any]], lab: list[float]) -> bool:
    for sample in samples:
        existing = sample.get("lab")
        if not isinstance(existing, list) or len(existing) != 3:
            continue
        if math.dist([float(value) for value in existing], lab) < 0.5:
            return True
    return False


def sync_profile_samples(
    library: Dict[str, Any], profiles: Mapping[str, Mapping[str, Any]]
) -> bool:
    """Import trusted points from the current tracker profiles exactly once."""
    changed = False
    minis = library.setdefault("minis", {})
    defaults = {spec["id"]: spec for spec in DEFAULT_MINIS}
    for mini_id, profile in profiles.items():
        mini_id = str(mini_id)
        spec = defaults.get(mini_id)
        entry = minis.setdefault(
            mini_id,
            _default_entry(spec)
            if spec
            else {
                "name": mini_id,
                "ringColor": "unknown",
                "tokenName": mini_id,
                "enabled": True,
                "samples": [],
            },
        )
        samples = entry.setdefault("samples", [])
        curve = profile.get("lab_curve") or []
        points = [point for point in curve if point is not None]
        if not points and profile.get("lab") is not None:
            points = [profile["lab"]]
        brightness = profile.get("brightness_steps") or []
        for index, point in enumerate(points):
            lab = _valid_lab(point)
            if _contains_lab(samples, lab):
                continue
            condition = (
                f"display-brightness:{brightness[index]}"
                if index < len(brightness)
                else "legacy-profile"
            )
            samples.append(
                {
                    "id": _sample_id(mini_id, lab, "profile-import", condition),
                    "lab": lab,
                    "verified": True,
                    "source": "profile-import",
                    "capturedAt": None,
                    "conditions": {"display": condition},
                }
            )
            changed = True
    return changed


def load_synced_library(
    profiles: Mapping[str, Mapping[str, Any]], path: Path = LIBRARY_PATH
) -> Dict[str, Any]:
    library = load_library(path)
    if sync_profile_samples(library, profiles) or not path.exists():
        save_library(library, path)
    return library


def add_verified_sample(
    mini_id: str,
    lab: Iterable[Any],
    *,
    source: str = "known-position-scan",
    conditions: Optional[Mapping[str, Any]] = None,
    path: Path = LIBRARY_PATH,
) -> str:
    """Store a user-confirmed sample, ignoring near-identical duplicates."""
    mini_id = str(mini_id).strip()
    if not mini_id:
        raise ValueError("Mini identity cannot be empty")
    values = _valid_lab(lab)
    library = load_library(path)
    entry = library["minis"].setdefault(
        mini_id,
        {
            "name": mini_id,
            "ringColor": "unknown",
            "tokenName": mini_id,
            "enabled": True,
            "samples": [],
        },
    )
    samples = entry.setdefault("samples", [])
    if _contains_lab(samples, values):
        return "duplicate"
    captured_at = datetime.now(timezone.utc).isoformat()
    condition_data = dict(conditions or {})
    samples.append(
        {
            "id": _sample_id(mini_id, values, source, captured_at),
            "lab": values,
            "verified": True,
            "source": source,
            "capturedAt": captured_at,
            "conditions": condition_data,
        }
    )
    save_library(library, path)
    return "added"


def library_rows(
    library: Mapping[str, Any],
    *,
    profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    mappings: Optional[Mapping[str, str]] = None,
    token_names: Optional[Mapping[str, str]] = None,
    detections: Optional[Mapping[str, Any]] = None,
    positions: Optional[Mapping[str, Optional[str]]] = None,
) -> list[Dict[str, Any]]:
    profiles = profiles or {}
    mappings = mappings or {}
    token_names = token_names or {}
    detections = detections or {}
    positions = positions or {}
    rows = []
    for mini_id, entry in (library.get("minis") or {}).items():
        if not entry.get("enabled", True):
            continue
        token_id = mappings.get(mini_id)
        detection = detections.get(mini_id)
        score = getattr(detection, "score", None) if detection is not None else None
        sample_count = len(
            [sample for sample in entry.get("samples", []) if sample.get("verified")]
        )
        rows.append(
            {
                "id": mini_id,
                "name": str(entry.get("name") or mini_id),
                "ringColor": str(entry.get("ringColor") or "unknown"),
                "scanStatus": "Ready" if mini_id in profiles else "Needs scan",
                "sampleCount": sample_count,
                "token": token_names.get(str(token_id), str(token_id)) if token_id else None,
                "confidence": float(score) if score is not None else None,
                "position": positions.get(mini_id),
            }
        )
    return rows
