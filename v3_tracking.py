# v3_tracking.py — Sarween combo tracking engine
#
# Runtime detection: persistent CIE Lab ring masks (independent of motion)
# Motion: used only to suppress frames obstructed by a hand/arm
# Profiles: combo_profiles.json — supports single Lab point OR multi-brightness curve

from __future__ import annotations

import json
import math
import re
import time
from collections import deque, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

import setup as s
import cv_core as core
import foundryoutput as fo
import mini_library as ml
import tap_selection as taps
from control_panel import ControlPanel, rc_to_a1
from mini_calibration import load_profiles_with_curves, min_lab_dist_to_profile

ensure_window = core.ensure_window

_PROFILES_PATH = Path(__file__).with_name("combo_profiles.json")

# Recording state carried across session restarts (e.g. "Calibrate Minis" mid-session)
_pending_recorder: Optional[Any] = None   # cv2.VideoWriter
_pending_record_path: Optional[str] = None

CONSENSUS_N = 6
CONSENSUS_K = 4

DEFAULT_MAX_LAB_DIST = 40.0

# A player's solid-color base ring remains visible after the mini stops moving.
# These filters deliberately describe the ring, not the much larger motion blob
# made by a hand carrying the mini.
PRESENCE_MAX_LAB_DIST = 20.0
# Perspective and the miniature itself can split the visible ring into a low-
# circularity crescent on the far side of the TV. Color, size, aspect, and
# temporal settling remain the stronger filters.
PRESENCE_MIN_CIRCULARITY = 0.18
PRESENCE_MAX_ASPECT_RATIO = 2.5
# A physical base ring occupies a substantial part of its grid square. Small
# same-color Foundry token artwork can otherwise become a false physical anchor
# and create a feedback loop as Sarween moves that token on screen.
PRESENCE_MIN_AREA_FRAC = 0.25
PRESENCE_MAX_AREA_FRAC = 3.0
PRESENCE_OPEN_PX = 3
PRESENCE_CLOSE_PX = 5
PRESENCE_ANCHOR_RADIUS_CELLS = 0.80
PRESENCE_PENDING_RADIUS_CELLS = 0.75
PRESENCE_SETTLE_FRAMES = 4
PRESENCE_MAX_CHANGE_RANGE = 0.0025
PRESENCE_LAB_BIN_SHIFT = 2
PRESENCE_PROCESS_SCALE = 0.50

_presence_lut_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

MIN_CIRCULARITY  = 0.1
MAX_ASPECT_RATIO = 2.5
MIN_AREA_FRAC    = 0.10
MAX_AREA_FRAC    = 9.0
SAMPLE_RADIUS_FRAC = 0.35
MORPH_CLOSE_FRAC   = 0.30

# Reject a whole frame only when both the total board change and one connected
# obstruction are large. This catches an arm sweep without suppressing a valid
# mini merely because several smaller Foundry token/fog regions also changed.
MAX_BOARD_CHANGE_RATIO = 0.025
MAX_CONNECTED_CHANGE_RATIO = 0.015
# A hand can fragment into several sub-threshold contours under fog. Even then,
# the total changed area stays distinctly above the settled Foundry residue.
MAX_TOTAL_CHANGE_RATIO = 0.035

# A tall mini is displaced in the planar homography.  With the camera mounted at
# the lower-right edge, its base is near this point within the warped silhouette.
POSITION_X_FRAC = 0.20
POSITION_Y_FRAC = 0.20

DISPLAY_FPS = 15.0


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ComboDetection:
    mini_id: str
    cx: float
    cy: float
    lab_dist: float
    contour_area: float
    circularity: float
    score: float
    bbox: Tuple[int, int, int, int]
    sampled_lab: Tuple[float, float, float]


# ──────────────────────────────────────────────────────────────────────────────
# Color sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_lab_at_centroid(
    warp_bgr: np.ndarray,
    cx: float,
    cy: float,
    radius_px: float,
) -> Optional[Tuple[float, float, float]]:
    h, w = warp_bgr.shape[:2]
    r = max(2, int(round(radius_px)))
    x0 = max(0, int(cx) - r)
    y0 = max(0, int(cy) - r)
    x1 = min(w, int(cx) + r + 1)
    y1 = min(h, int(cy) + r + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = warp_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    crop_h, crop_w = crop.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.circle(mask, (int(cx) - x0, int(cy) - y0), r, 255, -1)
    lab_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    pixels = lab_crop[mask > 0]
    if pixels.shape[0] == 0:
        return None
    med = np.median(pixels, axis=0)
    return (
        float(med[0]) * 100.0 / 255.0,
        float(med[1]) - 128.0,
        float(med[2]) - 128.0,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Blob detection — uses cv_core motion mask
# ──────────────────────────────────────────────────────────────────────────────

def _has_large_obstruction(bundle: "core.FrameBundle") -> bool:
    raw_motion_ratio = float(getattr(bundle, "raw_motion_ratio", 0.0) or 0.0)
    largest_motion_area = float(getattr(bundle, "largest_motion_area", 0.0) or 0.0)
    warp_area = max(1.0, float(bundle.warp_w * bundle.warp_h))
    return (
        raw_motion_ratio > MAX_TOTAL_CHANGE_RATIO
        or (
            raw_motion_ratio > MAX_BOARD_CHANGE_RATIO
            and largest_motion_area / warp_area > MAX_CONNECTED_CHANGE_RATIO
        )
    )


def _find_mini_blobs(
    bundle: "core.FrameBundle",
    grid_px: float,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, float, float, float, float]]:
    raw_motion_ratio = float(getattr(bundle, "raw_motion_ratio", 0.0) or 0.0)
    largest_motion_area = float(getattr(bundle, "largest_motion_area", 0.0) or 0.0)
    warp_area = max(1.0, float(bundle.warp_w * bundle.warp_h))
    connected_change_ratio = largest_motion_area / warp_area
    if _has_large_obstruction(bundle):
        if verbose:
            print(
                f"  BLOB | obstruction: board={raw_motion_ratio:.1%}, "
                f"largest={connected_change_ratio:.1%}; waiting for it to clear"
            )
        return []

    final_mask_cam = bundle.final_mask_cam
    mask = None
    if final_mask_cam is not None and bundle.H_use is not None:
        try:
            warped = cv2.warpPerspective(
                final_mask_cam, bundle.H_use,
                (bundle.warp_w, bundle.warp_h),
                flags=cv2.INTER_NEAREST,
            )
            nonzero = cv2.countNonZero(warped)
            if nonzero >= 50:
                mask = warped
            else:
                # Shadow filter produced an empty/sparse result — fall back to
                # raw motion_warp.  This happens on coloured map backgrounds where
                # chroma/brightness thresholds mis-classify real mini pixels.
                if verbose:
                    print(f"  BLOB | final_mask_cam warped to {nonzero} px → "
                          f"fallback to motion_warp")
                mask = bundle.motion_warp.copy() if bundle.motion_warp is not None else None
        except Exception:
            mask = bundle.motion_warp.copy() if bundle.motion_warp is not None else None
    elif bundle.motion_warp is not None:
        mask = bundle.motion_warp.copy()

    if mask is None:
        if verbose:
            print("  BLOB | no motion mask available (final_mask_cam and motion_warp both None)")
        return []

    if bundle.mask_warp is not None:
        mask = cv2.bitwise_and(mask, bundle.mask_warp)

    k_close = max(7, int(round(grid_px * MORPH_CLOSE_FRAC)) | 1)
    k_open  = max(3, int(round(grid_px * 0.08)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open)))

    cell_area = grid_px * grid_px
    min_area = cell_area * MIN_AREA_FRAC
    max_area = cell_area * MAX_AREA_FRAC
    cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    if verbose:
        print(f"  BLOB | {len(cnts)} raw contours  "
              f"(area range {min_area:.0f}–{max_area:.0f}  "
              f"circ≥{MIN_CIRCULARITY}  aspect≤{MAX_ASPECT_RATIO})")

    blobs = []
    n_rej_size = n_rej_circ = n_rej_aspect = 0
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            n_rej_size += 1
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        circ = float(4.0 * math.pi * area / (peri * peri))
        if circ < MIN_CIRCULARITY:
            n_rej_circ += 1
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(1.0, min(w, h)) > MAX_ASPECT_RATIO:
            n_rej_aspect += 1
            continue
        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-6:
            continue
        blobs.append((cnt, M["m10"]/M["m00"], M["m01"]/M["m00"], area, circ))

    if verbose:
        if cnts:
            print(f"  BLOB | rejected: size={n_rej_size}  circ={n_rej_circ}  aspect={n_rej_aspect}  "
                  f"→ {len(blobs)} blob(s) passed")
        for i, (_, cx, cy, area, circ) in enumerate(blobs):
            print(f"    [{i}] cx={cx:.0f} cy={cy:.0f}  area={area:.0f}  circ={circ:.2f}")

    return blobs


def _tracking_point(cnt: np.ndarray) -> Tuple[float, float]:
    """Estimate the mini's base rather than the center of its tall silhouette."""
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w * POSITION_X_FRAC, y + h * POSITION_Y_FRAC


# ──────────────────────────────────────────────────────────────────────────────
# Persistent ring detection
# ──────────────────────────────────────────────────────────────────────────────

def _profile_lab_points(profile: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    curve = profile.get("lab_curve")
    if curve:
        points = [tuple(float(v) for v in point) for point in curve if point is not None]
        if points:
            return points
    single = profile.get("lab")
    if single:
        return [tuple(float(v) for v in single)]
    return []


def _presence_profile_lut(profile: Dict[str, Any]) -> np.ndarray:
    """Build a cached quantized Lab lookup table for one ring profile."""
    points = _profile_lab_points(profile)
    max_dist = float(profile.get("presence_lab_dist", PRESENCE_MAX_LAB_DIST))
    key = (
        PRESENCE_LAB_BIN_SHIFT,
        round(max_dist, 4),
        tuple(tuple(round(value, 4) for value in point) for point in points),
    )
    cached = _presence_lut_cache.get(key)
    if cached is not None:
        return cached

    bin_count = 256 >> PRESENCE_LAB_BIN_SHIFT
    step = 1 << PRESENCE_LAB_BIN_SHIFT
    cv_values = np.arange(bin_count, dtype=np.float32) * step + (step - 1) / 2.0
    l_values = cv_values * (100.0 / 255.0)
    ab_values = cv_values - 128.0
    l_grid, a_grid, b_grid = np.meshgrid(
        l_values, ab_values, ab_values, indexing="ij"
    )
    min_dist_sq = np.full(l_grid.shape, np.inf, dtype=np.float32)
    for l_ref, a_ref, b_ref in points:
        dist_sq = (
            (l_grid - l_ref) * (l_grid - l_ref)
            + (a_grid - a_ref) * (a_grid - a_ref)
            + (b_grid - b_ref) * (b_grid - b_ref)
        )
        np.minimum(min_dist_sq, dist_sq, out=min_dist_sq)
    lut = (min_dist_sq <= max_dist * max_dist).reshape(-1)
    _presence_lut_cache[key] = lut
    return lut


def _find_presence_candidates(
    lab_cv: np.ndarray,
    color_index: np.ndarray,
    profile_name: str,
    profile: Dict[str, Any],
    grid_px: float,
    mask_warp: Optional[np.ndarray] = None,
    use_enclosing_center: bool = False,
    verbose: bool = False,
) -> List[ComboDetection]:
    """Find ring-shaped regions matching one mini's calibrated Lab color."""
    points = _profile_lab_points(profile)
    if not points:
        return []

    max_dist = float(profile.get("presence_lab_dist", PRESENCE_MAX_LAB_DIST))
    color_mask = np.where(
        _presence_profile_lut(profile)[color_index], 255, 0
    ).astype(np.uint8)
    if mask_warp is not None:
        color_mask = cv2.bitwise_and(color_mask, mask_warp)

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PRESENCE_OPEN_PX, PRESENCE_OPEN_PX)
    )
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PRESENCE_CLOSE_PX, PRESENCE_CLOSE_PX)
    )
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, open_kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, close_kernel)

    cell_area = grid_px * grid_px
    min_area = cell_area * float(
        profile.get("presence_min_area_frac", PRESENCE_MIN_AREA_FRAC)
    )
    max_area = cell_area * float(
        profile.get("presence_max_area_frac", PRESENCE_MAX_AREA_FRAC)
    )
    contours_info = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    candidates: List[ComboDetection] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        if circularity < PRESENCE_MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > PRESENCE_MAX_ASPECT_RATIO:
            continue
        moments = cv2.moments(cnt)
        if abs(moments["m00"]) < 1e-6:
            continue
        if use_enclosing_center or circularity < 0.35:
            # The miniature can occlude the middle of its ring, leaving only a
            # crescent. Its colored-pixel centroid is pulled toward the visible
            # arc, while the enclosing circle better estimates the base center.
            (cx, cy), _ = cv2.minEnclosingCircle(cnt)
            cx = float(cx)
            cy = float(cy)
        else:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])

        contour_mask = np.zeros(color_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [cnt], -1, 255, thickness=-1)
        ring_pixels = (contour_mask > 0) & (color_mask > 0)
        if not np.any(ring_pixels):
            continue
        pixel_values = lab_cv[ring_pixels]
        median_cv = np.median(pixel_values, axis=0)
        sampled_lab = (
            float(median_cv[0]) * 100.0 / 255.0,
            float(median_cv[1]) - 128.0,
            float(median_cv[2]) - 128.0,
        )
        lab_dist = min_lab_dist_to_profile(sampled_lab, profile)

        color_score = max(0.0, 1.0 - lab_dist / max(1.0, max_dist))
        shape_score = min(1.0, circularity)
        ideal_area = max(1.0, cell_area * 0.60)
        size_score = math.exp(-abs(math.log(max(1.0, area) / ideal_area)))
        score = 0.60 * color_score + 0.25 * shape_score + 0.15 * size_score
        candidates.append(
            ComboDetection(
                mini_id=profile_name,
                cx=cx,
                cy=cy,
                lab_dist=lab_dist,
                contour_area=area,
                circularity=circularity,
                score=score,
                bbox=(int(x), int(y), int(w), int(h)),
                sampled_lab=sampled_lab,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if verbose:
        print(
            f"  RING | {profile_name}: {len(candidates)} candidate(s), "
            f"Lab≤{max_dist:.0f}, area={min_area:.0f}–{max_area:.0f}"
        )
        for i, candidate in enumerate(candidates):
            print(
                f"    [{i}] ({candidate.cx:.0f},{candidate.cy:.0f}) "
                f"area={candidate.contour_area:.0f} circ={candidate.circularity:.2f} "
                f"dist={candidate.lab_dist:.1f} score={candidate.score:.3f}"
            )
    return candidates


def _rescale_detection(
    detection: ComboDetection,
    inverse_scale: float,
) -> ComboDetection:
    if inverse_scale == 1.0:
        return detection
    x, y, w, h = detection.bbox
    return ComboDetection(
        mini_id=detection.mini_id,
        cx=detection.cx * inverse_scale,
        cy=detection.cy * inverse_scale,
        lab_dist=detection.lab_dist,
        contour_area=detection.contour_area * inverse_scale * inverse_scale,
        circularity=detection.circularity,
        score=detection.score,
        bbox=(
            int(round(x * inverse_scale)),
            int(round(y * inverse_scale)),
            int(round(w * inverse_scale)),
            int(round(h * inverse_scale)),
        ),
        sampled_lab=detection.sampled_lab,
    )


def _select_presence_candidate(
    candidates: List[ComboDetection],
    previous: Dict[str, Any],
    grid_px: float,
) -> Optional[ComboDetection]:
    if not candidates:
        return None

    anchor_xy = previous.get("last_xy")
    pending_xy = previous.get("pending_xy")

    def _near(point: Tuple[float, float], radius_cells: float) -> List[ComboDetection]:
        radius_px = grid_px * radius_cells
        return [
            candidate
            for candidate in candidates
            if math.hypot(candidate.cx - point[0], candidate.cy - point[1]) <= radius_px
        ]

    if anchor_xy is not None:
        anchored = _near(anchor_xy, PRESENCE_ANCHOR_RADIUS_CELLS)
        if anchored:
            return max(anchored, key=lambda candidate: candidate.score)
    if pending_xy is not None:
        pending = _near(pending_xy, PRESENCE_PENDING_RADIUS_CELLS)
        if pending:
            return max(pending, key=lambda candidate: candidate.score)
    return candidates[0]


def _updated_change_history(previous: Dict[str, Any], ratio: float) -> List[float]:
    history = list(previous.get("change_hist") or [])
    history.append(float(ratio))
    return history[-PRESENCE_SETTLE_FRAMES:]


def _presence_board_is_stable(history: List[float]) -> bool:
    return (
        len(history) >= PRESENCE_SETTLE_FRAMES
        and max(history) - min(history) <= PRESENCE_MAX_CHANGE_RANGE
    )


def _empty_presence_state(
    previous: Dict[str, Any],
    change_hist: Optional[List[float]] = None,
) -> Dict[str, Any]:
    return {
        "last_xy": previous.get("last_xy"),
        "last_dist": None,
        "pending_xy": None,
        "pending_hits": 0,
        "change_hist": (
            list(change_hist)
            if change_hist is not None
            else list(previous.get("change_hist") or [])
        ),
    }


def detect_minis(
    bundle: "core.FrameBundle",
    profiles: Dict[str, dict],
    grid_px: float,
    prev_state: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Optional[ComboDetection]], Dict[str, Any]]:
    if prev_state is None:
        prev_state = {}

    detections: Dict[str, Optional[ComboDetection]] = {}
    new_state: Dict[str, Any] = {}
    raw_change = float(getattr(bundle, "raw_motion_ratio", 0.0) or 0.0)
    change_histories = {
        name: _updated_change_history(prev_state.get(name, {}), raw_change)
        for name in profiles
    }
    if bundle.warp_bgr is None or _has_large_obstruction(bundle):
        if verbose and bundle.warp_bgr is not None:
            print("  RING | hand/arm obstruction; ignoring this frame")
        for name in profiles:
            previous = prev_state.get(name, {})
            detections[name] = None
            new_state[name] = _empty_presence_state(
                previous, change_histories[name]
            )
        return detections, new_state

    stable_names = {
        name
        for name, history in change_histories.items()
        if _presence_board_is_stable(history)
    }
    if not stable_names:
        if verbose:
            print("  RING | board still changing; waiting for a settled frame")
        for name in profiles:
            previous = prev_state.get(name, {})
            detections[name] = None
            new_state[name] = _empty_presence_state(
                previous, change_histories[name]
            )
        return detections, new_state

    process_scale = (
        PRESENCE_PROCESS_SCALE
        if max(bundle.warp_bgr.shape[:2]) >= 800
        else 1.0
    )
    if process_scale < 1.0:
        process_bgr = cv2.resize(
            bundle.warp_bgr,
            None,
            fx=process_scale,
            fy=process_scale,
            interpolation=cv2.INTER_AREA,
        )
        process_mask = (
            cv2.resize(
                bundle.mask_warp,
                (process_bgr.shape[1], process_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            if bundle.mask_warp is not None
            else None
        )
    else:
        process_bgr = bundle.warp_bgr
        process_mask = bundle.mask_warp

    lab_cv = cv2.cvtColor(process_bgr, cv2.COLOR_BGR2Lab)
    lab_bins = lab_cv.astype(np.uint32) >> PRESENCE_LAB_BIN_SHIFT
    bits_per_channel = 8 - PRESENCE_LAB_BIN_SHIFT
    color_index = (
        (lab_bins[:, :, 0] << (bits_per_channel * 2))
        | (lab_bins[:, :, 1] << bits_per_channel)
        | lab_bins[:, :, 2]
    )

    for name, profile in profiles.items():
        previous = prev_state.get(name, {})
        change_history = change_histories[name]
        if name not in stable_names:
            detections[name] = None
            new_state[name] = _empty_presence_state(previous, change_history)
            continue
        candidates = _find_presence_candidates(
            lab_cv,
            color_index,
            name,
            profile,
            grid_px * process_scale,
            mask_warp=process_mask,
            use_enclosing_center=(
                getattr(bundle, "marker_mode", "legacy") == "viewport"
            ),
            verbose=verbose,
        )
        if process_scale < 1.0:
            candidates = [
                _rescale_detection(candidate, 1.0 / process_scale)
                for candidate in candidates
            ]
        selected = _select_presence_candidate(candidates, previous, grid_px)
        detections[name] = selected
        if selected is None:
            new_state[name] = _empty_presence_state(previous, change_history)
            continue

        pending_xy = previous.get("pending_xy")
        if pending_xy is not None and math.hypot(
            selected.cx - pending_xy[0], selected.cy - pending_xy[1]
        ) <= grid_px * PRESENCE_PENDING_RADIUS_CELLS:
            pending_hits = int(previous.get("pending_hits", 0)) + 1
        else:
            pending_hits = 1
        new_state[name] = {
            "last_xy": previous.get("last_xy"),
            "last_dist": selected.lab_dist,
            "pending_xy": (selected.cx, selected.cy),
            "pending_hits": pending_hits,
            "change_hist": change_history,
        }

    # Distinct ring colors are expected, but if two profiles select the same
    # physical region, retain only the stronger color/shape match.
    names = [name for name, detection in detections.items() if detection is not None]
    for i, name in enumerate(names):
        detection = detections.get(name)
        if detection is None:
            continue
        for other in names[i + 1:]:
            other_detection = detections.get(other)
            if other_detection is None:
                continue
            if math.hypot(
                detection.cx - other_detection.cx,
                detection.cy - other_detection.cy,
            ) > grid_px * 0.50:
                continue
            loser = other if detection.score >= other_detection.score else name
            detections[loser] = None
            new_state[loser] = _empty_presence_state(
                prev_state.get(loser, {}), change_histories[loser]
            )

    return detections, new_state


def _dump_tracking_state(
    profiles: Dict[str, dict],
    prev_state: Dict[str, Any],
    cell_hist: Dict[str, Any],
    last_emitted: Dict[str, str],
    last_seen: Optional[Dict[str, float]] = None,
    session: Optional[Any] = None,
    last_physical_xy: Optional[Dict[str, Tuple[float, float]]] = None,
    bundle: Optional["core.FrameBundle"] = None,
) -> None:
    """Print a full snapshot of the current tracking state to terminal."""
    now_t = time.perf_counter()
    print("=" * 60)
    print("DUMP STATE")
    print(f"  profiles loaded: {list(profiles.keys())}")
    if session is not None:
        print(
            f"  marker mode: {session.marker_mode}  "
            f"required IDs={sorted(session.required_ids)}  "
            f"warp={session.warp_w}x{session.warp_h}"
        )
    print(f"  Foundry scene: {fo.get_scene_params()}")
    print(f"  Foundry viewport: {fo.get_view_transform_info()}")
    if session is not None:
        print(
            "  grid size in camera warp: "
            f"{fo.warp_grid_dimensions(session.warp_w, session.warp_h)}"
        )
        print(f"  background settle frames: {session._view_settle_frames}")
    if bundle is not None:
        print(
            f"  raw board change: {bundle.raw_motion_ratio:.2%}  "
            f"largest contour area={bundle.largest_motion_area:.0f}"
        )
    print()
    print("  mini positions (prev_state):")
    for name in profiles:
        ps = prev_state.get(name, {})
        xy = ps.get("last_xy")
        d  = ps.get("last_dist")
        xy_str = f"({xy[0]:.0f},{xy[1]:.0f})" if xy else "None (unanchored)"
        d_str  = f"dist={d:.1f}" if d is not None else "dist=None"
        age_str = ""
        if last_seen and name in last_seen:
            age = now_t - last_seen[name]
            age_str = f"  last_seen={age:.1f}s ago"
        elif xy is not None:
            age_str = "  last_seen=unknown"
        print(f"    {name:20s}  last_xy={xy_str}  {d_str}{age_str}")
        if last_physical_xy and name in last_physical_xy:
            physical = last_physical_xy[name]
            mapped = _bundle_to_cell(session, *physical) if session is not None else None
            print(f"    {'':20s}  physical_xy={physical}  mapped_cell={mapped}")
    print()
    print("  consensus buffers (cell_hist):")
    for name in profiles:
        buf = cell_hist.get(name)
        emitted = last_emitted.get(name, "None")
        if buf is None or len(buf) == 0:
            print(f"    {name:20s}  buffer=[]  last_emitted={emitted}")
        else:
            counts = Counter(buf)
            top, top_n = counts.most_common(1)[0]
            status = f"→ EMIT {top}" if top_n >= CONSENSUS_K else f"→ hold (need {CONSENSUS_K}/{CONSENSUS_N})"
            print(f"    {name:20s}  buffer={list(buf)}  top={top}({top_n}/{len(buf)})  {status}  last_emitted={emitted}")
    print("=" * 60, flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ──────────────────────────────────────────────────────────────────────────────

def _warp_to_cell(cx, cy, grid_w, grid_h, warp_w, warp_h):
    col = int(np.clip(cx / (warp_w / float(grid_w)), 0, grid_w - 1))
    row = int(np.clip(cy / (warp_h / float(grid_h)), 0, grid_h - 1))
    return col, row


def _grid_px_for_bundle(bundle: "core.FrameBundle") -> Optional[float]:
    if getattr(bundle, "marker_mode", "legacy") == "viewport":
        return fo.warp_grid_size(bundle.warp_w, bundle.warp_h)
    return min(
        bundle.warp_w / float(bundle.grid_w),
        bundle.warp_h / float(bundle.grid_h),
    )


def _bundle_to_cell(
    bundle: "core.FrameBundle", cx: float, cy: float
) -> Optional[Tuple[int, int]]:
    if getattr(bundle, "marker_mode", "legacy") == "viewport":
        return fo.warp_to_grid_cell(cx, cy, bundle.warp_w, bundle.warp_h)
    return _warp_to_cell(
        cx, cy, bundle.grid_w, bundle.grid_h, bundle.warp_w, bundle.warp_h
    )


def _cell_label(row, col):
    return f"r{int(row)}c{int(col)}"


# ──────────────────────────────────────────────────────────────────────────────
# Single-point calibration (fallback — still works without full curve capture)
# ──────────────────────────────────────────────────────────────────────────────

def _save_profiles(profiles: Dict) -> None:
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def _calibration_rejection_reason(
    name: str,
    lab: Tuple[float, float, float],
) -> Optional[str]:
    """Reject obviously wrong samples for the named solid-color player rings."""
    normalized = re.sub(r"[\s_-]+", "", str(name).strip().lower())
    match = re.fullmatch(r"(red|blue|green|yellow|white)\d*", normalized)
    if match is None or match.group(1) == "white":
        return None

    color_name = match.group(1)
    _, a_value, b_value = lab
    chroma = math.hypot(a_value, b_value)
    if chroma < 12.0:
        return f"sample is nearly gray (Lab chroma {chroma:.1f})"

    wrong_hue = {
        "red": a_value < 6.0,
        "blue": b_value > -6.0,
        "green": a_value > -6.0,
        "yellow": b_value < 6.0,
    }[color_name]
    if wrong_hue:
        return f"sample does not look {color_name} (Lab a={a_value:.1f}, b={b_value:.1f})"
    return None


def calibrate_from_bundle(
    bundle: "core.FrameBundle",
    name: str,
    existing_profiles: Dict,
) -> Optional[Dict]:
    if not bundle.locked or bundle.warp_bgr is None:
        return None

    grid_px = _grid_px_for_bundle(bundle)
    if grid_px is None:
        return None
    blobs = _find_mini_blobs(bundle, grid_px)

    if not blobs:
        return None

    cnt, cx, cy, area, circ = max(blobs, key=lambda b: b[3])
    equiv_r = math.sqrt(area / math.pi)
    sample_r = max(2.0, equiv_r * SAMPLE_RADIUS_FRAC)
    lab = _sample_lab_at_centroid(bundle.warp_bgr, cx, cy, sample_r)

    if lab is None:
        return None

    diameter_squares = (equiv_r * 2.0) / grid_px
    print(f"COMBO CAL | name={name}  Lab=({lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f})  "
          f"diam_sq={diameter_squares:.2f}  area={area:.0f}")

    return {
        "lab": list(lab),
        "expected_diameter_squares": float(diameter_squares),
        "max_lab_dist": DEFAULT_MAX_LAB_DIST,
        "presence_lab_dist": PRESENCE_MAX_LAB_DIST,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Calibration preview
# ──────────────────────────────────────────────────────────────────────────────

def render_calibration_preview(bundle: "core.FrameBundle") -> np.ndarray:
    vis = bundle.warp_bgr.copy() if bundle.warp_bgr is not None else \
          np.zeros((720, 1280, 3), dtype=np.uint8)

    if not bundle.locked or bundle.warp_bgr is None:
        cv2.putText(vis, "Waiting for ArUco lock", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    grid_px = _grid_px_for_bundle(bundle)
    if grid_px is None:
        cv2.putText(vis, "Waiting for Foundry viewport transform", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
        return vis
    blobs = _find_mini_blobs(bundle, grid_px)

    if not blobs:
        cv2.putText(vis, "No motion blobs — move the mini", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        for i, (cnt, cx, cy, area, circ) in enumerate(blobs):
            equiv_r = math.sqrt(area / math.pi)
            sample_r = max(2.0, equiv_r * SAMPLE_RADIUS_FRAC)
            lab = _sample_lab_at_centroid(bundle.warp_bgr, cx, cy, sample_r)

            color = (0, 255, 0) if i == 0 else (180, 180, 0)
            cv2.drawContours(vis, [cnt.astype(np.int32)], -1, color, 2)
            cv2.circle(vis, (int(cx), int(cy)), int(sample_r), (255, 100, 0), 1)

            lines = [f"area={area:.0f}  circ={circ:.2f}"]
            if lab is not None:
                lines.append(f"L={lab[0]:.0f} a={lab[1]:.0f} b={lab[2]:.0f}")

            ty = max(28, int(cy) - int(sample_r) - 4)
            for j, line in enumerate(lines):
                y = ty - (len(lines) - 1 - j) * 14
                y = max(14, min(vis.shape[0] - 4, y))
                cv2.putText(vis, line, (int(cx) + 6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(vis, line, (int(cx) + 6, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, "COMBO PREVIEW — move mini then press Calibrate",
                (12, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


# ──────────────────────────────────────────────────────────────────────────────
# Session runner
# ──────────────────────────────────────────────────────────────────────────────

def begin_session(on_mini_moved, camera_index=None, show_windows=True):
    sel = s.load_last_selection() or {}
    mode = (sel.get("mode") or "self_hosted").strip().lower()

    # Load profiles — supports both single-point and curve-based
    profiles = load_profiles_with_curves(_PROFILES_PATH)
    mini_library = ml.load_synced_library(profiles)

    sess = core.CVCoreSession(camera_index=camera_index)

    # Re-attach recorder if it was carried over from a previous session
    global _pending_recorder, _pending_record_path
    if _pending_recorder is not None:
        sess.attach_recorder(_pending_recorder, _pending_record_path)
        _pending_recorder = None
        _pending_record_path = None

    try:
        from tk_camera_preview import _get_root as _tk_get_root
        _shared_root = _tk_get_root()
    except Exception:
        _shared_root = None

    panel = ControlPanel(mode=mode, tk_root=_shared_root)
    panel.show()

    def update_library_panel(detections=None, positions=None):
        mappings, token_names = fo.get_mini_assignments()
        panel.update_mini_library(
            ml.library_rows(
                mini_library,
                profiles=profiles,
                mappings=mappings,
                token_names=token_names,
                detections=detections,
                positions=positions,
            )
        )

    update_library_panel()

    # Sync the Record button label if we resumed recording
    if sess.is_recording:
        panel.set_recording_status(True, sess._record_path or "")

    try:
        fo.set_grid_params(sess.warp_w, sess.warp_h, sess.grid_w, sess.grid_h)
    except Exception:
        pass

    prev_state: Dict[str, Any] = {}
    cell_hist: Dict[str, deque] = {}
    last_emitted: Dict[str, str] = {}
    last_physical_xy: Dict[str, Tuple[float, float]] = {}
    last_view_transform_revision = fo.get_view_transform_revision()
    last_scene_geometry = None
    last_output_paused = fo.tracking_output_paused()
    capture_started_at = None
    pending_library_scan = None

    # Lost-mini tracking: if a mini goes undetected for this many seconds while
    # anchored, drop the spatial anchor so it can re-lock anywhere on the board.
    # This handles the pick-up-and-place pattern (teleport moves) where the mini
    # disappears and reappears more than 3 cells away.
    LOST_TIMEOUT: float = 2.0
    _last_seen: Dict[str, float] = {}  # timestamp of last successful detection per mini

    fps_count = 0
    last_print = time.perf_counter()
    _current_fps: float = 30.0          # updated every second; used for recording FPS
    _last_no_blob_msg: float = 0.0      # throttle the "no blobs" verbose heartbeat
    _last_library_update: float = 0.0
    tap_detector = taps.TapGestureDetector()

    cam_window_open = False
    warp_window_open = False
    homography_window_open = False
    calib_preview_window_open = False
    motion_warp_window_open = False
    motion_cam_window_open = False
    shadowfree_window_open = False
    final_mask_window_open = False

    _display_interval = 1.0 / DISPLAY_FPS
    _last_display = 0.0

    try:
        for bundle in sess.frames():
            if not panel.pump():
                break

            _now = time.perf_counter()
            _do_display = show_windows and (_now - _last_display >= _display_interval)
            if _do_display:
                _last_display = _now

            actions = panel.pop_actions()
            if actions.get("scan_mini"):
                pending_library_scan = (str(actions["scan_mini"]), time.perf_counter())
            if pending_library_scan and not actions.get("calibrate_band"):
                scan_name, scan_started = pending_library_scan
                if time.perf_counter() - scan_started <= 15.0:
                    actions["calibrate_band"] = {
                        "name": scan_name,
                        "portfolio_scan": True,
                    }
                else:
                    pending_library_scan = None
                    panel.set_hint(f"Scan timed out for {scan_name}; try Scan selected again")
            fo.update_camera_lock(bundle.locked, bundle.last_missing_ids)
            scene = fo.get_scene_params()
            scene_geometry = (scene["sceneId"], scene["sceneW"], scene["sceneH"], scene["gridPx"], scene["shiftX"], scene["shiftY"])
            output_paused = fo.tracking_output_paused()
            if scene_geometry != last_scene_geometry or output_paused != last_output_paused:
                prev_state.clear()
                cell_hist.clear()
                last_emitted.clear()
                last_physical_xy.clear()
                _last_seen.clear()
                last_view_transform_revision = fo.get_view_transform_revision()
                last_scene_geometry = scene_geometry
                last_output_paused = output_paused

            command = fo.pop_capture_command()
            if command:
                if command["action"] == "start":
                    capture_started_here = False
                    try:
                        view = fo.get_view_transform_payload()
                        if not bundle.locked or view is None or view["sceneId"] != scene["sceneId"]:
                            raise ValueError("Wait for marker lock and Foundry geometry before starting capture")
                        if sess.is_recording:
                            raise ValueError("Stop the existing recording before starting guided capture")
                        rec_path = str(Path(__file__).parent / f"sarween_rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
                        if not sess.start_recording(rec_path, fps=max(1.0, _current_fps)):
                            raise ValueError("Camera recording failed to start")
                        capture_started_here = True
                        fo.ready_guided_capture(rec_path, profiles)
                        panel.set_recording_status(True, rec_path)
                        capture_started_at = time.perf_counter()
                    except Exception as exc:
                        fo.finish_guided_capture("startFailed")
                        if capture_started_here:
                            sess.stop_recording()
                            panel.set_recording_status(False)
                        fo.send_capture_status("error", message=str(exc))
                        panel.set_hint(str(exc))
                elif command["action"] == "stop":
                    fo.finish_guided_capture(command.get("reason", "userStopped"))
                    sess.stop_recording()
                    panel.set_recording_status(False)
                    capture_started_at = None
            if capture_started_at is not None and time.perf_counter() - capture_started_at > 900:
                fo.finish_guided_capture("timeLimit")
                sess.stop_recording()
                panel.set_recording_status(False)
                capture_started_at = None
            if fo.guided_capture_active() and (actions.get("calibrate_band") or actions.get("calibrate_minis")):
                panel.set_hint("Finish guided capture before changing mini scans")
                continue

            if actions.get("calibrate_band"):
                req = actions.get("calibrate_band")
                name = None
                portfolio_scan = False
                if isinstance(req, dict):
                    name = (req.get("name") or "").strip() or None
                    portfolio_scan = bool(req.get("portfolio_scan", False))
                if not name:
                    name = time.strftime("mini_%Y%m%d_%H%M%S")

                if not bundle.locked:
                    panel.set_hint("Calibrate: wait for ArUco lock")
                else:
                    prof = calibrate_from_bundle(bundle, name, profiles)
                    if prof is None:
                        if portfolio_scan:
                            panel.set_hint(f"Waiting for {name}; move that mini to a new square")
                        else:
                            panel.set_hint("Calibrate: no motion blob — move the mini first")
                    else:
                        lab = tuple(prof["lab"])
                        rejection = _calibration_rejection_reason(name, lab)
                        if rejection:
                            message = f"Calibration rejected for {name}: {rejection}"
                            print(f"COMBO CAL | {message}", flush=True)
                            panel.set_hint(message)
                            if portfolio_scan:
                                pending_library_scan = None
                            continue
                        profiles[name] = prof
                        try:
                            _save_profiles(profiles)
                            sample_result = ml.add_verified_sample(
                                name,
                                lab,
                                source="known-position-scan",
                                conditions={"fog": "unknown", "roomLighting": "current"},
                            )
                            mini_library = ml.load_synced_library(profiles)
                            update_library_panel()
                            sample_note = (
                                "portfolio sample added"
                                if sample_result == "added"
                                else "matching sample already saved"
                            )
                            panel.set_hint(
                                f"Calibrated: {name}  Lab=({lab[0]:.0f},{lab[1]:.0f},{lab[2]:.0f}); "
                                f"{sample_note}"
                            )
                            if portfolio_scan:
                                pending_library_scan = None
                        except Exception as e:
                            panel.set_hint(f"Save failed: {e}")
                continue

            if actions.get("recapture_bg"):
                try:
                    cam = bundle.cam_bgr
                    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (21, 21), 0)
                    sess.BG_cam = {"bgr": cam.copy(), "blur": blur}
                    bg_warp = core.warp_gray_blur(cam, sess.H_saved, sess.warp_w, sess.warp_h)
                    sess.BG_warp_f32 = bg_warp.astype(np.float32)
                    sess._bg_seeded = True
                    panel.set_hint("Background recaptured ✅")
                except Exception as e:
                    panel.set_hint(f"Recapture failed: {e}")

            if actions.get("toggle_recording"):
                if sess.is_recording:
                    if fo.guided_capture_active():
                        fo.finish_guided_capture("userStopped")
                        capture_started_at = None
                    sess.stop_recording()
                    panel.set_recording_status(False)
                    panel.set_hint("Recording and tracking timeline saved ✅")
                else:
                    rec_path = str(
                        Path(__file__).parent
                        / f"sarween_rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    )
                    ok = sess.start_recording(rec_path, fps=max(1.0, _current_fps))
                    if ok:
                        panel.set_recording_status(True, rec_path)
                        panel.set_hint(f"Recording video + Foundry timeline → {rec_path}")
                    else:
                        panel.set_hint("Recording failed to start ❌")

            if actions.get("calibrate_minis"):
                # Carry the recorder across the calibration session so the user
                # gets a single continuous video file for the whole run.
                if sess.is_recording:
                    _pending_recorder, _pending_record_path = sess.detach_recorder()
                    print(f"V3 | Recording detached — will resume after calibration "
                          f"({_pending_record_path})", flush=True)
                panel.set_hint("Stopping for mini calibration...")
                sess.close(preserve_recording=True)
                cv2.destroyAllWindows()
                panel.hide()
                return "recalibrate"

            if actions.get("exit"):
                break

            if actions.get("dump_state"):
                _dump_tracking_state(profiles, prev_state, cell_hist, last_emitted,
                                     last_seen=_last_seen, session=sess,
                                     last_physical_xy=last_physical_xy,
                                     bundle=bundle)

            tog = panel.get_toggles()
            show_cam_view         = bool(tog.get("show_live_camera", False))
            show_h_view           = bool(tog.get("show_homography", False))
            show_calib_preview    = bool(tog.get("show_calib_preview", True))
            show_warp_view        = bool(tog.get("show_identify", False))
            show_motion_warp_view = bool(tog.get("show_motion_warp", False))
            show_motion_cam_view  = bool(tog.get("show_motion_cam", False))
            show_shadowfree_view  = bool(tog.get("show_shadowfree", False))
            show_final_mask_view  = bool(tog.get("show_final_mask", False))
            verbose_tracking      = bool(tog.get("verbose_tracking", False))

            # Push timing flag to cv_core module-level variable
            core.PRINT_TIMING = bool(tog.get("show_timing", False))

            sess.warp_motion_thresh = panel.get_motion_thresh()

            panel.set_status(
                locked=bool(bundle.locked),
                marker_count=int(bundle.last_marker_count),
                missing_ids=list(bundle.last_missing_ids),
            )

            fps_count += 1
            now = time.perf_counter()
            if now - last_print >= 1.0:
                _current_fps = fps_count / (now - last_print)
                panel.set_status(fps=_current_fps)
                fps_count = 0
                last_print = now

            # ── Camera view ──────────────────────────────────────────────────
            if show_windows and show_cam_view:
                if _do_display:
                    vis_cam = bundle.cam_bgr.copy()
                    if not bundle.locked:
                        cv2.putText(vis_cam,
                                    f"Waiting for ArUco lock ({bundle.last_marker_count}/4)",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    ensure_window("Camera (combo)", 1280, 720)
                    cv2.imshow("Camera (combo)", vis_cam)
                cam_window_open = True
                cv2.waitKey(1)
            else:
                if cam_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Camera (combo)")
                    except Exception:
                        pass
                    cam_window_open = False

            if not bundle.locked or bundle.warp_bgr is None:
                continue

            if fo.tracking_output_paused():
                panel.set_hint("Capture mode: tracking predictions paused")
                if show_windows and show_h_view and _do_display:
                    core.show_homography_view(bundle.cam_bgr, bundle.H_use,
                                              bundle.warp_w, bundle.warp_h,
                                              bundle.grid_w, bundle.grid_h,
                                              marker_mode=bundle.marker_mode,
                                              corner_ids=sess.corner_ids)
                continue

            grid_px = _grid_px_for_bundle(bundle)
            if bundle.marker_mode == "viewport" and scene.get("gridType") not in (None, 1):
                panel.set_hint("Foundry tracking requires a square Foundry grid")
                continue
            if grid_px is None:
                panel.set_hint("Waiting for Foundry viewport transform...")
                continue

            current_view_revision = fo.get_view_transform_revision()
            if current_view_revision != last_view_transform_revision:
                last_view_transform_revision = current_view_revision
                for _name, (_px, _py) in list(last_physical_xy.items()):
                    mapped = _bundle_to_cell(bundle, _px, _py)
                    if mapped is None:
                        continue
                    _col, _row = mapped
                    _cell = _cell_label(_row, _col)
                    if _cell == last_emitted.get(_name):
                        continue
                    last_emitted[_name] = _cell
                    cell_hist.pop(_name, None)
                    print(
                        f"V3 | Foundry map moved under {_name}; "
                        f"now at {rc_to_a1(_row, _col)}",
                        flush=True,
                    )
                    fo.record_tracking_event(
                        _name, _cell, source="viewportTransform"
                    )
                    if on_mini_moved is not None:
                        on_mini_moved(_name, _cell)

            _do_verbose = verbose_tracking and bundle.locked

            # Lost-mini timeout: if anchored but undetected for >LOST_TIMEOUT s,
            # clear the spatial anchor so the mini can re-lock anywhere.
            _now_t = time.perf_counter()
            selected_mini = fo.get_selected_mini()
            for _name in list(prev_state.keys()):
                _ps = prev_state[_name]
                if _ps.get("last_xy") is not None:
                    _age = _now_t - _last_seen.get(_name, _now_t)
                    timeout = 0.75 if _name == selected_mini else LOST_TIMEOUT
                    if _age > timeout:
                        _ps["last_xy"] = None
                        cell_hist.pop(_name, None)   # clear stale consensus buffer
                        print(f"V3 | {_name} lost ({_age:.1f}s since last detection) "
                              f"→ search mode", flush=True)

            if _do_verbose:
                print(f"── TRACK f{bundle.frame_idx} ─────────────────────────────")

            dets, prev_state = detect_minis(
                bundle=bundle, profiles=profiles,
                grid_px=grid_px, prev_state=prev_state,
                verbose=_do_verbose,
            )

            # Presence detection sees a settled ring continuously, so every
            # successful detection refreshes last-seen.
            for _name, _det in dets.items():
                if _det is None:
                    continue
                _last_seen[_name] = _now_t

            contact_ids = taps.contact_minis(
                bundle,
                last_physical_xy,
                grid_px,
            )
            detected_positions = {
                name: (detection.cx, detection.cy)
                for name, detection in dets.items()
                if detection is not None
            }
            tapped_mini = tap_detector.update(
                _now_t,
                contact_ids,
                last_physical_xy if contact_ids else detected_positions,
                grid_px,
            )
            if tapped_mini:
                print(f"V3 | Physical tap recognized on {tapped_mini}", flush=True)
                fo.queue_control({"type": "miniTap", "miniId": tapped_mini})

            # Heartbeat when verbose but nothing detected (throttled to once/3s)
            if verbose_tracking and bundle.locked and not any(d is not None for d in dets.values()):
                if _now_t - _last_no_blob_msg >= 3.0:
                    _last_no_blob_msg = _now_t
                    print(f"TRACK | f{bundle.frame_idx} locked, no ring candidates detected")

            any_cv_window_open = cam_window_open

            # ── Warp debug view ──────────────────────────────────────────────
            if show_windows and show_warp_view:
                if _do_display:
                    vis_warp = bundle.warp_bgr.copy()
                    for mname, det in dets.items():
                        if det is None:
                            continue
                        x, y, w, h = det.bbox
                        cv2.rectangle(vis_warp, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        label = f"{mname}  d={det.lab_dist:.1f}  s={det.score:.2f}"
                        cv2.putText(vis_warp, label, (x, max(18, y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(vis_warp, label, (x, max(18, y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
                    ensure_window("Warp (combo)", 1280, 720)
                    cv2.imshow("Warp (combo)", vis_warp)
                any_cv_window_open = True
                warp_window_open = True
            else:
                if warp_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Warp (combo)")
                    except Exception:
                        pass
                    warp_window_open = False

            # ── Calibration preview ──────────────────────────────────────────
            if show_windows and show_calib_preview and bundle.locked:
                if _do_display:
                    cv2.imshow("Combo Calibration Preview",
                               render_calibration_preview(bundle))
                    ensure_window("Combo Calibration Preview", 1280, 720)
                any_cv_window_open = True
                calib_preview_window_open = True
            else:
                if calib_preview_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Combo Calibration Preview")
                    except Exception:
                        pass
                    calib_preview_window_open = False

            # ── Motion warp ──────────────────────────────────────────────────
            if show_windows and show_motion_warp_view and bundle.motion_warp is not None:
                if _do_display:
                    ensure_window("Motion (warp)", 1280, 720)
                    cv2.imshow("Motion (warp)", bundle.motion_warp)
                any_cv_window_open = True
                motion_warp_window_open = True
            else:
                if motion_warp_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Motion (warp)")
                    except Exception:
                        pass
                    motion_warp_window_open = False

            # ── Motion camera ────────────────────────────────────────────────
            if show_windows and show_motion_cam_view and bundle.motion_cam is not None:
                if _do_display:
                    ensure_window("Motion (camera)", 1280, 720)
                    cv2.imshow("Motion (camera)", bundle.motion_cam)
                any_cv_window_open = True
                motion_cam_window_open = True
            else:
                if motion_cam_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Motion (camera)")
                    except Exception:
                        pass
                    motion_cam_window_open = False

            # ── Shadow-free mask ─────────────────────────────────────────────
            if show_windows and show_shadowfree_view and bundle.shadowfree_cam is not None:
                if _do_display:
                    ensure_window("Shadow-free mask", 1280, 720)
                    cv2.imshow("Shadow-free mask", bundle.shadowfree_cam)
                any_cv_window_open = True
                shadowfree_window_open = True
            else:
                if shadowfree_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Shadow-free mask")
                    except Exception:
                        pass
                    shadowfree_window_open = False

            # ── Final mask ───────────────────────────────────────────────────
            if show_windows and show_final_mask_view and bundle.final_mask_cam is not None:
                if _do_display:
                    ensure_window("Final mask", 1280, 720)
                    cv2.imshow("Final mask", bundle.final_mask_cam)
                any_cv_window_open = True
                final_mask_window_open = True
            else:
                if final_mask_window_open:
                    try:
                        cv2.waitKey(1); cv2.destroyWindow("Final mask")
                    except Exception:
                        pass
                    final_mask_window_open = False

            # ── Homography view ──────────────────────────────────────────────
            if show_windows and show_h_view:
                core.show_homography_view(bundle.cam_bgr, bundle.H_use,
                                          bundle.warp_w, bundle.warp_h,
                                          bundle.grid_w, bundle.grid_h,
                                          marker_mode=bundle.marker_mode,
                                          corner_ids=sess.corner_ids)
                homography_window_open = True
                any_cv_window_open = True
            else:
                if homography_window_open:
                    try:
                        cv2.destroyWindow("Homography view (debug)")
                    except Exception:
                        pass
                    homography_window_open = False

            # ── Emit movements ───────────────────────────────────────────────
            if _do_verbose and any(d is not None for d in dets.values()):
                print("  CONSENSUS:")
            for mname, det in dets.items():
                if det is None:
                    continue
                mapped = _bundle_to_cell(bundle, det.cx, det.cy)
                if mapped is None:
                    continue
                col, row = mapped
                cell = _cell_label(row, col)
                buf = cell_hist.setdefault(mname, deque(maxlen=CONSENSUS_N))
                buf.append(cell)
                most, count = Counter(buf).most_common(1)[0]
                if _do_verbose:
                    a1 = rc_to_a1(row, col)
                    if count >= CONSENSUS_K:
                        if most != last_emitted.get(mname):
                            emit_note = f"→ EMIT {a1} ✅"
                        else:
                            emit_note = f"→ already at {a1}"
                    else:
                        emit_note = f"→ hold ({count}/{CONSENSUS_N} need {CONSENSUS_K})"
                    print(f"    {mname:20s}  buf={list(buf)[-CONSENSUS_N:]}  "
                          f"top={most}({count}/{len(buf)})  {emit_note}")
                if count >= CONSENSUS_K and cell == most:
                    last_physical_xy[mname] = (det.cx, det.cy)
                if count >= CONSENSUS_K and most != last_emitted.get(mname):
                    last_emitted[mname] = most
                    # Every confirmed placement becomes the new spatial anchor.
                    parts = most[1:].split('c')
                    r_idx, c_idx = int(parts[0]), int(parts[1])
                    if getattr(bundle, "marker_mode", "legacy") == "viewport":
                        anchor = (det.cx, det.cy)
                    else:
                        cell_w = bundle.warp_w / float(bundle.grid_w)
                        cell_h = bundle.warp_h / float(bundle.grid_h)
                        anchor = ((c_idx + 0.5) * cell_w, (r_idx + 0.5) * cell_h)
                    if mname not in prev_state:
                        prev_state[mname] = {}
                    prev_state[mname]["last_xy"] = anchor
                    print(f"V3 | {mname} anchored at {rc_to_a1(r_idx, c_idx)} "
                          f"({anchor[0]:.0f},{anchor[1]:.0f})", flush=True)
                    fo.record_tracking_event(mname, most)
                    if on_mini_moved is not None:
                        on_mini_moved(mname, most)

            # ── Update positions panel ───────────────────────────────────────
            positions = {}
            for mname in profiles:
                raw = last_emitted.get(mname)
                if raw is not None:
                    try:
                        parts = raw[1:].split('c')
                        positions[mname] = rc_to_a1(int(parts[0]), int(parts[1]))
                    except Exception:
                        positions[mname] = raw
                else:
                    positions[mname] = None
            panel.update_positions(positions)
            if _now_t - _last_library_update >= 0.5:
                update_library_panel(dets, positions)
                _last_library_update = _now_t

            if show_windows and any_cv_window_open:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

    finally:
        if fo.guided_capture_active():
            fo.finish_guided_capture("appClosed")
        fo.update_camera_lock(False, [])
        try:
            sess.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return None
