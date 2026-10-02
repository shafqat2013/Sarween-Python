"""Explicit ring-pixel sampling and non-destructive profile persistence."""

import copy
import json
import os
from pathlib import Path
from app_paths import atomic_write_json

import cv2
import numpy as np


def sample_ring_patch(image, point):
    """Sample only the user-selected 3x3 patch; reject mixed boundary pixels."""
    if image is None or point is None:
        return None
    x, y = (int(round(value)) for value in point)
    height, width = image.shape[:2]
    if not (1 <= x < width - 1 and 1 <= y < height - 1):
        return None
    patch = cv2.cvtColor(image[y-1:y+2, x-1:x+2], cv2.COLOR_BGR2Lab).astype(float)
    patch[:, :, 0] *= 100.0 / 255.0
    patch[:, :, 1:] -= 128.0
    pixels = patch.reshape(-1, 3)
    median = np.median(pixels, axis=0)
    if np.max(np.linalg.norm(pixels - median, axis=1)) > 15.0:
        return None
    return tuple(float(value) for value in median)


def profile_with_sample(existing, lab, *, replace_colors=False):
    values = np.asarray(lab, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("A scan must contain three finite Lab values")
    profile = copy.deepcopy(existing or {})
    old_curve = profile.get("lab_curve") or ([profile["lab"]] if profile.get("lab") else [])
    if replace_colors:
        old_curve = []
    curve = copy.deepcopy(old_curve)
    steps = list(profile.get("brightness_steps") or [])[:len(curve)]
    steps.extend([None] * (len(curve) - len(steps)))
    if not any(point is not None and np.allclose(point, values) for point in curve):
        curve.append(values.tolist())
        steps.append(None)
    profile.update(lab=values.tolist(), lab_curve=curve, brightness_steps=steps,
                   scan_source="manual-ring-sample")
    profile.setdefault("max_lab_dist", 40.0)
    profile.setdefault("presence_lab_dist", 20.0)
    return profile


def save_profiles(profiles, path):
    """Atomic replacement, with the prior complete file available for rollback."""
    atomic_write_json(path, profiles)


def merge_profiles(updates, path):
    path = Path(path)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if not isinstance(previous, dict):
        raise ValueError("Existing mini profiles are invalid; refusing to overwrite them")
    previous.update(updates)
    save_profiles(previous, path)
    return previous
