"""Past-settled-image evidence for the restricted offline observer.

This is a static-camera, static-background check, not a model of Foundry's
changing rendered output. A previously visible warm map feature or miniature
head must not become a new base merely because another anchor disappears.

The caller explicitly updates the reference only after the whole layout is
confirmed and unobstructed. Filtering never learns from or stores its input.
Moving video captions are unavailable in both the current and reference images.
No identities, expected moves, recording names, or future frames are learned.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{label} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 2):
        raise ValueError("Scene reference needs a uint8 BGR image")
    return image


def _mask(mask, shape):
    if mask is None:
        return np.zeros(shape, dtype=bool)
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != shape:
        raise ValueError("Exclusion mask must match the image dimensions")
    return mask > 0


class SceneReference:
    """Require local pixel change for unanchored base candidates.

    Near confirmed anchors, this filter defers to the ring detector and identity
    association. For new positions, at least 60% of the candidate's inner disk
    must be usable, and at least 20% of those usable pixels must differ by more
    than 25 in any BGR channel. Pixel thresholds are experimental allowances.

    This can establish change from the last confirmed layout, not physical
    object identity. Changing map artwork, lighting, or camera pose can also
    change pixels and therefore remains outside this check's guarantees.
    """

    def __init__(self, grid_px):
        self.grid_px = _number(grid_px, "Grid spacing")
        if self.grid_px <= 0:
            raise ValueError("Grid spacing must be positive")
        self.anchor_gate = self.grid_px * .45
        self.disk_fraction = .8
        self.pixel_threshold = 25
        self.minimum_valid_fraction = .60
        self.minimum_changed_fraction = .20
        self._reference = None
        self._excluded = None
        self.reference_updates = 0
        self.last_evidence = []

    @property
    def has_reference(self):
        return self._reference is not None

    def update(self, image, mask=None):
        """Store a copy of an externally confirmed, settled past observation.

        The caller owns the all-anchors-visible/no-obstruction gate. This method
        has no movement labels or timing knowledge with which to infer that gate.
        """
        image = _image(image)
        excluded = _mask(mask, image.shape[:2])
        # Copy both; later changes to a reused frame buffer or mask must not
        # rewrite what the observer actually saw in the past.
        self._reference = image.copy()
        self._excluded = excluded.copy()
        self.reference_updates += 1

    def filter(self, image, candidates, positions, excluded_mask=None):
        """Return candidates supported by confirmed anchors or observed change.

        Calling ``filter`` is observational: it does not update the reference.
        ``positions`` is the identity state's confirmed layout, not expected
        endpoints. ``last_evidence`` exposes each filtering decision for review.
        """
        image = _image(image)
        excluded = _mask(excluded_mask, image.shape[:2])
        if self._reference is not None and self._reference.shape != image.shape:
            raise ValueError("Image geometry changed; establish a new scene reference")
        if not isinstance(positions, Mapping):
            raise ValueError("Confirmed positions must be a mapping")
        anchors = []
        for xy in positions.values():
            if not isinstance(xy, (tuple, list, np.ndarray)) or len(xy) != 2:
                raise ValueError("Confirmed positions need two coordinates")
            anchors.append((_number(xy[0], "Anchor x"), _number(xy[1], "Anchor y")))

        height, width = image.shape[:2]
        parsed = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise ValueError("Each candidate must be a mapping")
            try:
                x = _number(candidate["cx"], "Candidate x")
                y = _number(candidate["cy"], "Candidate y")
                radius = _number(candidate["radius"], "Candidate radius")
            except KeyError:
                raise ValueError("Each candidate needs cx, cy, and radius") from None
            if radius <= 0 or radius > max(width, height):
                raise ValueError("Candidate radius must be positive and bounded by the image size")
            parsed.append((index, candidate, x, y, radius))

        output = []
        evidence = []
        for index, candidate, x, y, radius in parsed:
            record = {"candidate": index, "cx": x, "cy": y,
                      "accepted": False, "reason": None,
                      "valid_fraction": None, "changed_fraction": None}
            evidence.append(record)
            if not (0 <= x < width and 0 <= y < height):
                record["reason"] = "outside_image"
                continue
            if any(math.hypot(x-ax, y-ay) <= self.anchor_gate for ax, ay in anchors):
                record.update(accepted=True, reason="confirmed_anchor")
                output.append(candidate)
                continue
            if self._reference is None:
                record["reason"] = "reference_unavailable"
                continue

            disk_radius = max(1.0, self.disk_fraction * radius)
            raw_x0, raw_x1 = math.floor(x-disk_radius), math.ceil(x+disk_radius)+1
            raw_y0, raw_y1 = math.floor(y-disk_radius), math.ceil(y+disk_radius)+1
            # Count the full disk, including portions outside the frame. They
            # are unavailable observations rather than a smaller denominator.
            # The one-dimensional sum avoids allocating a potentially large
            # unbounded square just to count pixels near an image edge.
            total = 0
            for yy in range(raw_y0, raw_y1):
                span_squared = disk_radius**2 - (yy-y)**2
                if span_squared >= 0:
                    span = math.sqrt(span_squared)
                    total += max(0, math.floor(x+span)-math.ceil(x-span)+1)
            x0, x1 = max(0, raw_x0), min(width, raw_x1)
            y0, y1 = max(0, raw_y0), min(height, raw_y1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            disk = (xx-x)**2 + (yy-y)**2 <= disk_radius**2
            valid = disk & ~excluded[y0:y1, x0:x1] & ~self._excluded[y0:y1, x0:x1]
            valid_count = int(np.count_nonzero(valid))
            valid_fraction = valid_count / max(total, 1)
            record["valid_fraction"] = float(valid_fraction)
            if valid_fraction < self.minimum_valid_fraction:
                record["reason"] = "insufficient_unmasked_support"
                continue
            current = image[y0:y1, x0:x1].astype(np.int16)
            previous = self._reference[y0:y1, x0:x1].astype(np.int16)
            delta = np.max(np.abs(current-previous), axis=2)
            changed_fraction = int(np.count_nonzero((delta > self.pixel_threshold) & valid)) / max(valid_count, 1)
            record["changed_fraction"] = float(changed_fraction)
            if changed_fraction < self.minimum_changed_fraction:
                record["reason"] = "unchanged_from_settled_reference"
                continue
            record.update(accepted=True, reason="changed_from_settled_reference")
            output.append(candidate)
        self.last_evidence = evidence
        return output
