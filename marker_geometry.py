"""Marker pattern and surrounding white space, in Foundry display pixels.

Old profiles and recordings described the whole tile with a fixed 1/6 margin.
Keep that geometry exact when reading them; new controls describe the pattern
and a separate margin so editing one never scales the other.
"""
from __future__ import annotations

import math


MARKER_KEYS = {"viewportMarkerSize", "viewportMarkerPatternSize", "viewportMarkerMargin"}


def _number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if not low <= value <= high:
        raise ValueError(f"{label} must be between {low:g} and {high:g} pixels.")
    return float(value)


def marker_geometry(settings):
    """Read effective geometry, preserving legacy-only recorded tile sizes."""
    if not isinstance(settings, dict):
        raise ValueError("Marker settings must be an object.")
    if "viewportMarkerPatternSize" in settings:
        pattern = _number(settings["viewportMarkerPatternSize"], "ArUco pattern width", 1, 240)
        margin = _number(settings.get("viewportMarkerMargin", 12), "White margin per side", 0, 12)
    else:
        total = _number(settings.get("viewportMarkerSize", 120), "Legacy marker size", 1, 240)
        pattern = total * 2 / 3
        margin = (_number(settings["viewportMarkerMargin"], "White margin per side", 0, 12)
                  if "viewportMarkerMargin" in settings else total / 6)
    return {"patternSize": pattern, "margin": margin, "totalSize": pattern + 2 * margin}


def normalized_marker_settings(settings):
    """Convert a saved profile to the editable ranges, without mutating it."""
    result = dict(settings)
    if "viewportMarkerPatternSize" in result and "viewportMarkerMargin" in result:
        # New live state also reports a derived legacy total for old readers;
        # it is not an independent editable setting (and can reach 264).
        result.pop("viewportMarkerSize", None)
    if "viewportMarkerSize" in result:
        total = _number(result["viewportMarkerSize"], "Legacy marker size", 1, 240)
        result.pop("viewportMarkerSize", None)
        result.setdefault("viewportMarkerPatternSize", max(1, total * 2 / 3))
        result.setdefault("viewportMarkerMargin", min(12, total / 6))
    # A profile containing only a margin must not also resize the pattern.
    if "viewportMarkerPatternSize" in result:
        result["viewportMarkerPatternSize"] = _number(result["viewportMarkerPatternSize"], "ArUco pattern width", 1, 240)
    if "viewportMarkerMargin" in result:
        result["viewportMarkerMargin"] = _number(result["viewportMarkerMargin"], "White margin per side", 0, 12)
    return result
