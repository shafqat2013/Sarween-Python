"""Detect a short physical touch near a known mini without continuous ML."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

import cv2
import numpy as np


@dataclass
class _TapCandidate:
    mini_id: str
    started_at: float
    origin: Tuple[float, float]
    released_at: Optional[float] = None


class TapGestureDetector:
    def __init__(
        self,
        *,
        min_contact_seconds: float = 0.08,
        max_contact_seconds: float = 0.85,
        settle_timeout_seconds: float = 1.25,
        cooldown_seconds: float = 0.9,
        max_move_cells: float = 0.35,
    ):
        self.min_contact_seconds = float(min_contact_seconds)
        self.max_contact_seconds = float(max_contact_seconds)
        self.settle_timeout_seconds = float(settle_timeout_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.max_move_cells = float(max_move_cells)
        self.candidate: Optional[_TapCandidate] = None
        self.cooldown_until = 0.0

    def update(
        self,
        now: float,
        contacts: Set[str],
        positions: Mapping[str, Tuple[float, float]],
        grid_px: float,
    ) -> Optional[str]:
        now = float(now)
        if self.candidate is None:
            if now < self.cooldown_until or len(contacts) != 1:
                return None
            mini_id = next(iter(contacts))
            origin = positions.get(mini_id)
            if origin is None:
                return None
            self.candidate = _TapCandidate(mini_id, now, tuple(origin))
            return None

        candidate = self.candidate
        if candidate.released_at is None:
            if candidate.mini_id in contacts:
                if now - candidate.started_at > self.max_contact_seconds:
                    self._cancel(now)
                return None
            duration = now - candidate.started_at
            if duration < self.min_contact_seconds or duration > self.max_contact_seconds:
                self._cancel(now)
                return None
            candidate.released_at = now

        if contacts:
            self._cancel(now)
            return None
        current = positions.get(candidate.mini_id)
        if current is not None:
            moved = math.hypot(
                float(current[0]) - candidate.origin[0],
                float(current[1]) - candidate.origin[1],
            )
            if moved <= max(1.0, float(grid_px) * self.max_move_cells):
                mini_id = candidate.mini_id
                self.candidate = None
                self.cooldown_until = now + self.cooldown_seconds
                return mini_id
            self._cancel(now)
            return None
        if now - float(candidate.released_at) > self.settle_timeout_seconds:
            self._cancel(now)
        return None

    def _cancel(self, now: float) -> None:
        self.candidate = None
        self.cooldown_until = float(now) + self.cooldown_seconds


def contact_minis(
    bundle: Any,
    known_positions: Mapping[str, Tuple[float, float]],
    grid_px: float,
    *,
    radius_cells: float = 0.6,
    minimum_changed_fraction: float = 0.06,
) -> Set[str]:
    """Return known minis whose immediate area is touched by foreground motion."""
    if not known_positions or bundle.final_mask_cam is None or bundle.H_use is None:
        return set()
    try:
        motion = cv2.warpPerspective(
            bundle.final_mask_cam,
            bundle.H_use,
            (int(bundle.warp_w), int(bundle.warp_h)),
            flags=cv2.INTER_NEAREST,
        )
    except Exception:
        return set()
    if bundle.mask_warp is not None:
        motion = cv2.bitwise_and(motion, bundle.mask_warp)
    if cv2.countNonZero(motion) < 20:
        return set()

    radius = max(4, int(round(float(grid_px) * radius_cells)))
    height, width = motion.shape[:2]
    contacts = set()
    for mini_id, (cx, cy) in known_positions.items():
        center_x, center_y = int(round(cx)), int(round(cy))
        x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
        y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        region = motion[y0:y1, x0:x1]
        circle = np.zeros(region.shape, dtype=np.uint8)
        cv2.circle(circle, (center_x - x0, center_y - y0), radius, 255, -1)
        circle_pixels = max(1, cv2.countNonZero(circle))
        changed = cv2.countNonZero(cv2.bitwise_and(region, circle))
        if changed / float(circle_pixels) >= minimum_changed_fraction:
            contacts.add(str(mini_id))
    return contacts
