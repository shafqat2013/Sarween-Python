# band_tracking.py
#
# Band-tracking engine (color bands / circular markers) for Sarween.
#
# This file contains:
# 1) The band detection primitives (detect_bands, helpers)
# 2) A begin_session(...) runner that uses cv_core.CVCoreSession to share
#    camera/undistortion, ArUco lock, warp, and shared masking pipeline.
#
# Configuration:
# - Requires a JSON file "band_profiles.json" next to this file (repo root).
#   Format example:
#   {
#     "red": {
#       "ranges": [{"lower":[0,120,80],"upper":[10,255,255]}, {"lower":[170,120,80],"upper":[179,255,255]}],
#       "expected_diameter_squares": 1.0
#     },
#     "blue": {
#       "ranges": [{"lower":[95,120,80],"upper":[130,255,255]}],
#       "expected_diameter_squares": 1.0
#     }
#   }
#
# Identity:
# - mini_id emitted to on_mini_moved is the color_name (e.g. "red").

from __future__ import annotations

import json
import math
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

import setup as s
import cv_core as core
import foundryoutput as fo
from control_panel import ControlPanel, rc_to_a1



# ──────────────────────────────────────────────────────────────────────────────
# Window helper — provided by cv_core
# ──────────────────────────────────────────────────────────────────────────────
ensure_window = core.ensure_window

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HSVRange:
    lower: Tuple[int, int, int]  # (H,S,V)
    upper: Tuple[int, int, int]

@dataclass
class ColorProfile:
    # Some colors (notably red) may need two ranges
    ranges: List[HSVRange]
    expected_diameter_squares: float = 1.0

@dataclass
class BandDetection:
    color_name: str
    cx: float
    cy: float
    contour_area: float
    circularity: float
    ellipse_eccentricity: Optional[float]
    score: float
    bbox: Tuple[int, int, int, int]  # x,y,w,h in WARP space


# ──────────────────────────────────────────────────────────────────────────────
# Core helpers (detection)
# ──────────────────────────────────────────────────────────────────────────────

def _mask_for_profile(hsv_img: np.ndarray, profile: ColorProfile) -> np.ndarray:
    mask = None
    for r in profile.ranges:
        m = cv2.inRange(
            hsv_img,
            np.array(r.lower, dtype=np.uint8),
            np.array(r.upper, dtype=np.uint8),
        )
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    return mask

def _cleanup_mask(mask: np.ndarray, grid_px: float) -> np.ndarray:
    # Close small gaps inside the band blob, then open to remove thin noise
    # and HSV bleed from adjacent colors (e.g. yellow bleeding into orange).
    # Using slightly larger open kernel than close so stray pixels are stripped
    # more aggressively than genuine gaps are filled.
    k_close = max(3, int(round(grid_px * 0.08)) | 1)
    k_open  = max(3, int(round(grid_px * 0.12)) | 1)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    return mask

def _contour_centroid(cnt: np.ndarray) -> Optional[Tuple[float, float]]:
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)

def _contour_circularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))

def _ellipse_eccentricity(cnt: np.ndarray) -> Optional[float]:
    if len(cnt) < 5:
        return None
    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (MA, ma), angle = ellipse
    a = max(MA, ma) / 2.0
    b = min(MA, ma) / 2.0
    if a <= 1e-6:
        return None
    e = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
    return float(e)

def _score_candidate(
    area: float,
    circularity: float,
    ecc: Optional[float],
    expected_area: float,
    prev_xy: Optional[Tuple[float, float]],
    xy: Tuple[float, float],
    grid_px: float,
) -> float:
    # area closeness (log scale-ish)
    area_ratio = area / max(1.0, expected_area)
    area_term = math.exp(-abs(math.log(max(1e-6, area_ratio))) * 1.2)

    # circularity preference
    circ_term = max(0.0, min(1.0, (circularity - 0.25) / 0.55))

    # eccentricity: tolerate some tilt
    ecc_term = 1.0
    if ecc is not None:
        ecc_term = math.exp(-max(0.0, ecc - 0.55) * 3.0)

    # distance to previous — strong anchor so each color sticks to its last
    # known position and doesn't jump across the board to another mini.
    # Decays over 2 grid squares; if no prior position, neutral (1.0).
    dist_term = 1.0
    if prev_xy is not None:
        dx = xy[0] - prev_xy[0]
        dy = xy[1] - prev_xy[1]
        d = math.hypot(dx, dy)
        dist_term = math.exp(-d / max(1.0, grid_px * 2.0))

    # motion_boost: slightly favour blobs that are currently moving,
    # but do NOT gate on it — stationary minis must still be detected.
    base = (0.35 * area_term + 0.25 * circ_term + 0.10 * ecc_term + 0.30 * dist_term)
    return float(base)

def warp_centroid_to_cell(cx: float, cy: float, grid_w: int, grid_h: int, warp_w: int, warp_h: int) -> Tuple[int, int]:
    cell_w = warp_w / float(grid_w)
    cell_h = warp_h / float(grid_h)
    col = int(np.clip(cx / cell_w, 0, grid_w - 1))
    row = int(np.clip(cy / cell_h, 0, grid_h - 1))
    return col, row

def detect_bands(
    warp_bgr: np.ndarray,
    color_profiles: Dict[str, ColorProfile],
    grid_px: float,
    prev_state: Optional[Dict[str, Any]] = None,
    motion_mask: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Optional[BandDetection]], Dict[str, Any]]:
    """
    Two-mode detection per color:

    SEARCHING — motion pixels exist on the board.
                Hard-gate HSV detection to the motion region only.
                Find the best color-matching blob inside that window.

    HOLDING   — no motion anywhere (minis are stationary).
                Do NOT scan the whole board — that causes false positives.
                Hold the last known position from prev_state and emit a
                synthetic detection so consensus keeps ticking correctly.
    """
    if prev_state is None:
        prev_state = {}

    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)
    detections: Dict[str, Optional[BandDetection]] = {}
    new_state: Dict[str, Any] = {}

    expected_radius = 0.5 * grid_px
    expected_area = math.pi * expected_radius * expected_radius * 0.55

    min_area = (grid_px * grid_px) * 0.05
    max_area = (grid_px * grid_px) * 2.00

    # Build ring-annulus search windows from motion contours.
    # We don't just dilate the whole motion mask — that covers the entire mini
    # body and lets non-ring color blobs through.  Instead, for each motion
    # contour we erode its filled mask to get the interior, subtract to get the
    # outer ring border, and use that as the search region.  This means the HSV
    # match only fires on the actual band pixels, not the mini body.
    any_motion = False
    motion_search: Optional[np.ndarray] = None
    if motion_mask is not None:
        any_motion = cv2.countNonZero(motion_mask) > 0
        if any_motion:
            # Find contours of the motion blobs
            cnts_info = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

            ring_map = np.zeros(motion_mask.shape[:2], dtype=np.uint8)
            for mc in motion_cnts:
                mc_area = float(cv2.contourArea(mc))
                if mc_area < 4:
                    continue

                # Work on a tight crop around the contour — avoids eroding a
                # full 1280×720 image with a potentially huge kernel each frame.
                bx, by, bw, bh = cv2.boundingRect(mc)
                pad = 4
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(motion_mask.shape[1], bx + bw + pad)
                y1 = min(motion_mask.shape[0], by + bh + pad)

                crop_h, crop_w = y1 - y0, x1 - x0
                filled_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
                shifted = mc.astype(np.int32) - np.array([[[x0, y0]]])
                cv2.drawContours(filled_crop, [shifted], -1, 255, thickness=-1)

                equiv_r = math.sqrt(mc_area / math.pi)
                ring_thickness = max(2, int(round(equiv_r * 0.90)))
                # Cap kernel to crop size so erode doesn't exceed the image
                ring_thickness = min(ring_thickness, min(crop_h, crop_w) // 2 - 1)
                k = max(3, ring_thickness * 2 + 1)
                ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                interior_crop = cv2.erode(filled_crop, ek, iterations=1)
                ring_crop = cv2.subtract(filled_crop, interior_crop)
                ring_crop = cv2.dilate(ring_crop, None, iterations=2)

                ring_map[y0:y1, x0:x1] = cv2.bitwise_or(
                    ring_map[y0:y1, x0:x1], ring_crop
                )

            motion_search = ring_map if cv2.countNonZero(ring_map) > 0 else \
                cv2.dilate(motion_mask, None, iterations=4)  # fallback for tiny blobs

    for color_name, profile in color_profiles.items():
        prev = prev_state.get(color_name, {})
        prev_xy: Optional[Tuple[float, float]] = prev.get("last_xy")

        # ── HOLDING: no motion → hold last known position ──────────────────
        if not any_motion:
            if prev_xy is not None:
                px, py = prev_xy
                half = grid_px * 0.5
                synth = BandDetection(
                    color_name=color_name,
                    cx=px, cy=py,
                    contour_area=expected_area,
                    circularity=1.0,
                    ellipse_eccentricity=0.0,
                    score=1.0,
                    bbox=(int(px - half), int(py - half), int(half * 2), int(half * 2)),
                )
                detections[color_name] = synth
                new_state[color_name] = {"last_xy": prev_xy, "last_score": 1.0}
            else:
                detections[color_name] = None
                new_state[color_name] = {"last_xy": None, "last_score": 0.0}
            continue

        # ── SEARCHING: restrict HSV detection to motion region ─────────────
        color_mask = _mask_for_profile(hsv, profile)
        color_mask = _cleanup_mask(color_mask, grid_px)
        search_mask = cv2.bitwise_and(color_mask, motion_search)

        cnts_info = cv2.findContours(search_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

        best: Optional[BandDetection] = None

        for cnt in cnts:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue

            cxy = _contour_centroid(cnt)
            if cxy is None:
                continue
            cx, cy = cxy

            circ = _contour_circularity(cnt)
            if circ < 0.15:
                continue

            ecc = _ellipse_eccentricity(cnt)
            if ecc is not None and ecc > 0.92:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if max(w, h) / max(1.0, min(w, h)) > 3.5:
                continue

            score = _score_candidate(
                area=area,
                circularity=circ,
                ecc=ecc,
                expected_area=expected_area * (profile.expected_diameter_squares ** 2),
                prev_xy=prev_xy,
                xy=(cx, cy),
                grid_px=grid_px,
            )

            cand = BandDetection(
                color_name=color_name,
                cx=cx, cy=cy,
                contour_area=area,
                circularity=circ,
                ellipse_eccentricity=ecc,
                score=score,
                bbox=(int(x), int(y), int(w), int(h)),
            )

            if best is None or cand.score > best.score:
                best = cand

        detections[color_name] = best
        if best is not None:
            new_state[color_name] = {"last_xy": (best.cx, best.cy), "last_score": float(best.score)}
        else:
            # Motion exists but this color wasn't in it — hold last position
            new_state[color_name] = {"last_xy": prev_xy, "last_score": 0.0}

    # ── Spatial exclusivity (only during SEARCHING) ─────────────────────────
    if any_motion:
        color_names = list(detections.keys())
        conflict_threshold = grid_px * 1.0
        nulled: set = set()
        for i in range(len(color_names)):
            for j in range(i + 1, len(color_names)):
                a_name, b_name = color_names[i], color_names[j]
                if a_name in nulled or b_name in nulled:
                    continue
                a_det = detections[a_name]
                b_det = detections[b_name]
                if a_det is None or b_det is None:
                    continue
                if math.hypot(a_det.cx - b_det.cx, a_det.cy - b_det.cy) < conflict_threshold:
                    if a_det.score >= b_det.score:
                        detections[b_name] = None
                        new_state[b_name]["last_xy"] = prev_state.get(b_name, {}).get("last_xy")
                        nulled.add(b_name)
                    else:
                        detections[a_name] = None
                        new_state[a_name]["last_xy"] = prev_state.get(a_name, {}).get("last_xy")
                        nulled.add(a_name)

    return detections, new_state



# ──────────────────────────────────────────────────────────────────────────────
# Calibration helpers
# ──────────────────────────────────────────────────────────────────────────────

def _auto_band_name() -> str:
    return time.strftime("band_%Y%m%d_%H%M%S")

def _hue_ranges_from_samples(h_samples: np.ndarray, s_samples: np.ndarray, v_samples: np.ndarray) -> List[HSVRange]:
    """
    Build 1 or 2 HSV ranges from samples, handling hue wrap-around (red/magenta).
    Uses robust percentiles.
    """
    if h_samples.size == 0:
        return []

    # Percentiles
    h_lo = float(np.percentile(h_samples, 5))
    h_hi = float(np.percentile(h_samples, 95))
    s_lo = int(max(30, np.percentile(s_samples, 10)))
    v_lo = int(max(30, np.percentile(v_samples, 10)))
    s_hi = int(min(255, np.percentile(s_samples, 99)))
    v_hi = int(min(255, np.percentile(v_samples, 99)))

    # If the hue spread is huge, assume wrap-around and split into two clusters by circular distance to median
    # Simple approach: try splitting at 90 degrees in hue space (0..179).
    if (h_hi - h_lo) > 90:
        # Consider near-0 and near-179 as a wrap case:
        # Build two ranges: [0..h_lo2] and [h_hi2..179]
        # Use tighter percentiles around the extremes.
        low_cluster = h_samples[h_samples < 45]
        high_cluster = h_samples[h_samples > 135]
        ranges: List[HSVRange] = []
        if low_cluster.size > 10:
            lo2 = int(max(0, np.percentile(low_cluster, 5)))
            hi2 = int(min(179, np.percentile(low_cluster, 95)))
            ranges.append(HSVRange(lower=(lo2, s_lo, v_lo), upper=(hi2, s_hi, v_hi)))
        if high_cluster.size > 10:
            lo3 = int(max(0, np.percentile(high_cluster, 5)))
            hi3 = int(min(179, np.percentile(high_cluster, 95)))
            ranges.append(HSVRange(lower=(lo3, s_lo, v_lo), upper=(hi3, s_hi, v_hi)))
        return ranges if ranges else [HSVRange(lower=(int(h_lo), s_lo, v_lo), upper=(int(h_hi), s_hi, v_hi))]

    return [HSVRange(lower=(int(max(0, h_lo)), s_lo, v_lo), upper=(int(min(179, h_hi)), s_hi, v_hi))]


def calibrate_profile_from_bundle(
    bundle: core.FrameBundle,
    *,
    min_pixels: int = 150,
) -> Optional[ColorProfile]:
    """
    Calibrate a band profile by sampling ONLY the ring border pixels of the
    largest moving contour — not its interior.

    The ring is a thin colored strip around the base of the mini.  Sampling the
    full contour interior floods the HSV data with the mini body color (grey
    plastic, etc.) and produces a range that matches everything.

    Strategy:
      1. Find the largest moving contour (the whole mini footprint).
      2. Build a filled mask of that contour.
      3. Erode it by ~30% of the contour's inscribed-circle radius to get the
         interior.
      4. Subtract interior from filled -> ring-border pixels only.
      5. Sample HSV from those pixels with tighter percentiles (vivid pixels only).
    """
    if (not bundle.locked) or (bundle.warp_bgr is None) or (bundle.motion_warp is None) or (bundle.mask_warp is None):
        return None

    # Motion mask within ROI
    fg = cv2.bitwise_and(bundle.motion_warp, bundle.mask_warp)
    fg = cv2.erode(fg, None, iterations=1)
    fg = cv2.dilate(fg, None, iterations=1)

    cnts_info = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        return None

    # Choose largest moving contour as the mini footprint
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < float(min_pixels):
        return None

    # Filled mask of the whole mini footprint
    filled = np.zeros(fg.shape[:2], dtype=np.uint8)
    cv2.drawContours(filled, [cnt.astype(np.int32)], -1, 255, thickness=-1)

    # Estimate ring thickness: ~30% of the equivalent circle radius.
    # This peels off the outer ring while leaving the interior separate.
    equiv_radius = math.sqrt(area / math.pi)
    ring_thickness = max(2, int(round(equiv_radius * 0.90)))

    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (ring_thickness * 2 + 1, ring_thickness * 2 + 1)
    )
    interior = cv2.erode(filled, erode_kernel, iterations=1)

    # Ring = filled minus interior
    ring_mask = cv2.subtract(filled, interior)

    # Sample HSV from ring pixels only
    hsv = cv2.cvtColor(bundle.warp_bgr, cv2.COLOR_BGR2HSV)
    h  = hsv[:, :, 0][ring_mask > 0].astype(np.float32)
    s_ = hsv[:, :, 1][ring_mask > 0].astype(np.float32)
    v  = hsv[:, :, 2][ring_mask > 0].astype(np.float32)

    if h.size < min_pixels:
        # Ring too thin — fall back to full contour
        print(f"BAND CAL | Ring pixels too few ({h.size}), falling back to full contour")
        h  = hsv[:, :, 0][filled > 0].astype(np.float32)
        s_ = hsv[:, :, 1][filled > 0].astype(np.float32)
        v  = hsv[:, :, 2][filled > 0].astype(np.float32)
        if h.size < min_pixels:
            return None

    # Tighter percentiles — vivid ring pixels only, not washed-out antialiasing
    h_lo = float(np.percentile(h,   8))
    h_hi = float(np.percentile(h,  92))
    s_lo = int(max(60,  np.percentile(s_,  15)))   # enforce minimum saturation
    v_lo = int(max(40,  np.percentile(v,   15)))
    s_hi = int(min(255, np.percentile(s_,  98)))
    v_hi = int(min(255, np.percentile(v,   98)))

    # Detect hue wrap-around (red/magenta spans 0 and 179)
    ranges: List[HSVRange] = []
    if (h_hi - h_lo) > 90:
        low_cluster  = h[h <  45]
        high_cluster = h[h > 135]
        if low_cluster.size > 10:
            lo2 = int(max(0,   np.percentile(low_cluster,   5)))
            hi2 = int(min(179, np.percentile(low_cluster,  95)))
            ranges.append(HSVRange(lower=(lo2, s_lo, v_lo), upper=(hi2, s_hi, v_hi)))
        if high_cluster.size > 10:
            lo3 = int(max(0,   np.percentile(high_cluster,  5)))
            hi3 = int(min(179, np.percentile(high_cluster, 95)))
            ranges.append(HSVRange(lower=(lo3, s_lo, v_lo), upper=(hi3, s_hi, v_hi)))
        if not ranges:
            ranges = [HSVRange(lower=(int(h_lo), s_lo, v_lo), upper=(int(h_hi), s_hi, v_hi))]
    else:
        ranges = [HSVRange(lower=(int(max(0, h_lo)), s_lo, v_lo), upper=(int(min(179, h_hi)), s_hi, v_hi))]

    # Store measured ring diameter as a fraction of one grid square so the
    # area scorer knows what size to expect during detection.
    ring_diameter_squares = (equiv_radius * 2.0) / bundle.warp_w * bundle.grid_w \
        if bundle.warp_w > 0 else 1.0

    print(f"BAND CAL | ring_px={h.size}  H={h_lo:.0f}-{h_hi:.0f}  "
          f"S={s_lo}-{s_hi}  V={v_lo}-{v_hi}  "
          f"ring_diam_sq={ring_diameter_squares:.2f}")

    return ColorProfile(ranges=ranges, expected_diameter_squares=ring_diameter_squares)


def _profiles_to_jsonable(profiles: Dict[str, ColorProfile]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name, prof in profiles.items():
        out[name] = {
            "ranges": [{"lower": list(r.lower), "upper": list(r.upper)} for r in prof.ranges],
            "expected_diameter_squares": float(prof.expected_diameter_squares),
        }
    return out


def save_profiles(path: Path, profiles: Dict[str, ColorProfile]) -> None:
    path.write_text(json.dumps(_profiles_to_jsonable(profiles), indent=2), encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# Profiles loading
# ──────────────────────────────────────────────────────────────────────────────

_PROFILES_PATH = Path(__file__).with_name("band_profiles.json")

def _load_profiles(path: Path = _PROFILES_PATH) -> Dict[str, ColorProfile]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Create it to define HSV ranges for band tracking."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path.name} must be a non-empty JSON object.")

    out: Dict[str, ColorProfile] = {}
    for color_name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        ranges_cfg = cfg.get("ranges")
        if not isinstance(ranges_cfg, list) or not ranges_cfg:
            continue
        ranges: List[HSVRange] = []
        for r in ranges_cfg:
            if not isinstance(r, dict):
                continue
            lo = r.get("lower")
            hi = r.get("upper")
            if (
                isinstance(lo, list) and len(lo) == 3 and
                isinstance(hi, list) and len(hi) == 3
            ):
                ranges.append(HSVRange(tuple(int(x) for x in lo), tuple(int(x) for x in hi)))
        if not ranges:
            continue
        exp = cfg.get("expected_diameter_squares", 1.0)
        try:
            exp = float(exp)
        except Exception:
            exp = 1.0
        out[str(color_name)] = ColorProfile(ranges=ranges, expected_diameter_squares=exp)

    if not out:
        raise ValueError(f"{path.name} did not contain any valid profiles.")
    return out


def render_calibration_preview(
    bundle: "core.FrameBundle",
    profiles=None,
    min_pixels: int = 150,
) -> "np.ndarray":
    """
    Fast calibration preview. Uses bundle.motion_warp (already 200ms-filtered).
    Cheap: cropped erode, contour drawing instead of addWeighted, no detect_bands.
    """
    vis = bundle.warp_bgr.copy() if bundle.warp_bgr is not None else np.zeros((100, 100, 3), dtype=np.uint8)

    if bundle.motion_warp is None or bundle.mask_warp is None:
        cv2.putText(vis, "No motion data", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        return vis

    fg = cv2.bitwise_and(bundle.motion_warp, bundle.mask_warp)
    cnts_info = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    for cnt in cnts:
        cv2.drawContours(vis, [cnt.astype(np.int32)], -1, (80, 80, 80), 1)

    if not cnts:
        cv2.putText(vis, "No motion detected — move the mini", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)
    else:
        best = max(cnts, key=cv2.contourArea)
        area = float(cv2.contourArea(best))
        x, y, w, h = cv2.boundingRect(best)

        if area < min_pixels:
            cv2.drawContours(vis, [best.astype(np.int32)], -1, (0, 0, 255), 2)
            cv2.putText(vis, f"Too small (area={int(area)}, need {min_pixels})", (12, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        else:
            # Cropped erode — only the bounding box, not the full frame
            pad = 4
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(fg.shape[1], x + w + pad)
            y1 = min(fg.shape[0], y + h + pad)
            crop_h, crop_w = y1 - y0, x1 - x0

            filled_crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
            shifted = best.astype(np.int32) - np.array([[[x0, y0]]])
            cv2.drawContours(filled_crop, [shifted], -1, 255, thickness=-1)

            equiv_radius = math.sqrt(area / math.pi)
            ring_thickness = max(2, min(int(round(equiv_radius * 0.90)), min(crop_h, crop_w) // 2 - 1))
            k = max(3, ring_thickness * 2 + 1)
            ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            interior_crop = cv2.erode(filled_crop, ek, iterations=1)
            ring_crop = cv2.subtract(filled_crop, interior_crop)

            # Draw ring as contour outline (cheap, no addWeighted)
            ring_cnts_info = cv2.findContours(ring_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ring_cnts = ring_cnts_info[0] if len(ring_cnts_info) == 2 else ring_cnts_info[1]
            for rc in ring_cnts:
                cv2.drawContours(vis, [rc.astype(np.int32) + np.array([[[x0, y0]]])], -1, (255, 200, 0), 2)

            cv2.drawContours(vis, [best.astype(np.int32)], -1, (0, 255, 0), 2)

            # HSV swatch from crop only
            hsv_crop = cv2.cvtColor(bundle.warp_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            rh = hsv_crop[:, :, 0][ring_crop > 0].astype(np.float32)
            rs = hsv_crop[:, :, 1][ring_crop > 0].astype(np.float32)
            rv = hsv_crop[:, :, 2][ring_crop > 0].astype(np.float32)
            if rh.size > 0:
                mh, ms, mv = float(np.median(rh)), float(np.median(rs)), float(np.median(rv))
                swatch = cv2.cvtColor(
                    np.array([[[int(mh), int(ms), int(mv)]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                )[0][0].tolist()
                sw = 48
                vis[8:8 + sw, 8:8 + sw] = swatch
                cv2.rectangle(vis, (8, 8), (8 + sw, 8 + sw), (255, 255, 255), 1)
                cv2.putText(vis, f"ring_px={rh.size}  H={mh:.0f}  S={ms:.0f}  V={mv:.0f}",
                            (8 + sw + 8, 8 + sw // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(vis, "CALIBRATION PREVIEW — move band then press Calibrate band", (12, vis.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1, cv2.LINE_AA)



    return vis

MIN_SCORE = 0.55
CONSENSUS_N = 6
CONSENSUS_K = 4

# ──────────────────────────────────────────────────────────────────────────────
# Engine session runner
# ──────────────────────────────────────────────────────────────────────────────

def _cell_label_rc(row: int, col: int) -> str:
    return f"r{int(row)}c{int(col)}"


def begin_session(
    on_mini_moved,
    camera_index=None,
    show_windows=True,
):
    """
    Band-tracking session runner. Uses cv_core for lock/warp/masks.
    Emits mini_id=color_name.
    """
    sel = s.load_last_selection() or {}
    mode = (sel.get("mode") or "self_hosted").strip().lower()

    # Control panel is still used for toggles/status and consistent UX.
    panel = ControlPanel(mode=mode)
    panel.show()

    # Load color profiles from band_profiles.json
    # Load color profiles (optional at startup; user can calibrate new bands)
    try:
        profiles = _load_profiles(_PROFILES_PATH)
    except Exception:
        profiles = {}


    # Start core session
    sess = core.CVCoreSession(camera_index=camera_index)

    # Inform foundry output of grid params
    try:
        fo.set_grid_params(sess.warp_w, sess.warp_h, sess.grid_w, sess.grid_h)
    except Exception:
        pass

    control_panel_shown = False
    prev_state: Dict[str, Any] = {}
    switch_to: Optional[str] = None

    # Movement consensus buffers per color
    from collections import deque, Counter
    cell_hist: Dict[str, deque] = {}
    last_emitted: Dict[str, str] = {}

    fps_count = 0
    last_print = time.perf_counter()

    # Window open flags
    cam_window_open = False
    warp_window_open = False
    homography_window_open = False
    calib_preview_window_open = False
    motion_warp_window_open = False
    motion_cam_window_open = False
    shadowfree_window_open = False
    final_mask_window_open = False

    DRAW_ARUCO_OVERLAY = False

    # Display throttle — imshow is expensive on macOS (synchronous GPU upload).
    # CV detection runs every frame; windows only refresh at DISPLAY_FPS.
    DISPLAY_FPS = 15.0
    _display_interval = 1.0 / DISPLAY_FPS
    _last_display = 0.0

    try:
        for bundle in sess.frames():
            # keep UI responsive
            if not panel.pump():
                break

            # Compute display gate once per loop iteration — used by all imshow calls below
            _now = time.perf_counter()
            _do_display = show_windows and (_now - _last_display >= _display_interval)
            if _do_display:
                _last_display = _now

            actions = panel.pop_actions()
            if actions.get("calibrate_band"):
                req = actions.get("calibrate_band")
                name = None
                auto = False
                if isinstance(req, dict):
                    name = (req.get("name") or None)
                    auto = bool(req.get("auto", False))
                if not bundle.locked:
                    panel.set_hint("Calibrate: wait for ArUco lock")
                else:
                    prof = calibrate_profile_from_bundle(bundle)
                    if prof is None:
                        panel.set_hint("Calibrate: move ONE band in view")
                    else:
                        if not name:
                            name = _auto_band_name() if auto else _auto_band_name()
                        profiles[name] = prof
                        try:
                            save_profiles(_PROFILES_PATH, profiles)
                        except Exception:
                            pass
                        panel.set_hint(f"Calibrated band: {name}")
                # continue frame
                continue
            if actions.get("switch_engine") in ("blob", "band"):
                switch_to = actions.get("switch_engine")
                break
            if actions.get("recapture_bg"):
                try:
                    # Use the current live frame directly — avoids opening a second
                    # camera connection which can conflict with CVCoreSession on macOS.
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
            if actions.get("exit"):
                break

            tog = panel.get_toggles()
            show_cam_view = bool(tog.get("show_live_camera", False))
            show_h_view = bool(tog.get("show_homography", False))
            show_calib_preview = bool(tog.get("show_calib_preview", True))
            show_motion_warp_view = bool(tog.get("show_motion_warp", False))
            show_motion_cam_view  = bool(tog.get("show_motion_cam", False))
            show_shadowfree_view  = bool(tog.get("show_shadowfree", False))
            show_final_mask_view  = bool(tog.get("show_final_mask", False))
            show_warp_view = bool(tog.get("show_identify", False))  # reuse Identify toggle for Warp view

            # Apply motion threshold from slider — updates cv_core session live
            sess.warp_motion_thresh = panel.get_motion_thresh()

            # Update status (on every frame; lightweight)
            panel.set_status(
                locked=bool(bundle.locked),
                marker_count=int(bundle.last_marker_count),
                missing_ids=list(bundle.last_missing_ids),
            )

            # show panel after first lock
            if bundle.locked and (not control_panel_shown):
                panel.show()
                control_panel_shown = True

            # FPS update (once per second)
            fps_count += 1
            now = time.perf_counter()
            if now - last_print >= 1.0:
                fps = fps_count / (now - last_print)
                panel.set_status(fps=fps)
                fps_count = 0
                last_print = now

            # ── Camera view (shown even before lock — needed to aim the camera) ──
            if show_windows and show_cam_view:
                if _do_display:
                    vis_cam = bundle.cam_bgr.copy()
                    if DRAW_ARUCO_OVERLAY and core.ARUCO_DET is not None:
                        if bundle.last_aruco_ids is not None and bundle.last_aruco_corners is not None:
                            try:
                                import cv2.aruco as aruco
                                aruco.drawDetectedMarkers(vis_cam, bundle.last_aruco_corners, bundle.last_aruco_ids)
                            except Exception:
                                pass
                    if not bundle.locked:
                        msg = f"Waiting for ArUco lock ({bundle.last_marker_count}/4), missing: {bundle.last_missing_ids}"
                        cv2.putText(vis_cam, msg, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    ensure_window("Camera (band)", 1280, 720)
                    cv2.imshow("Camera (band)", vis_cam)
                cam_window_open = True
                cv2.waitKey(1)
            else:
                if cam_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Camera (band)")
                    except Exception:
                        pass
                    cam_window_open = False

            if not bundle.locked:
                # no lock -> no warp_bgr -> can't run band detection
                continue

            if bundle.warp_bgr is None:
                continue

            # Compute grid_px in warp space (min cell dimension)
            grid_px = min(bundle.warp_w / float(bundle.grid_w), bundle.warp_h / float(bundle.grid_h))

            dets, prev_state = detect_bands(
                warp_bgr=bundle.warp_bgr,
                color_profiles=profiles,
                grid_px=grid_px,
                prev_state=prev_state,
                motion_mask=bundle.motion_warp,
            )


            # ── Window rendering (optional) ─────────────────────────────────────
            # Throttled to DISPLAY_FPS — imshow is a synchronous GPU upload on macOS.
            # CV detection still runs every frame; only the screen refresh is limited.
            any_cv_window_open = False
            if cam_window_open:
                any_cv_window_open = True  # camera window is managed above the lock check

            if show_windows and show_warp_view and (bundle.warp_bgr is not None):
                if _do_display:
                    vis_warp = bundle.warp_bgr.copy()
                    for cname, det in dets.items():
                        if det is None:
                            continue
                        if det.score < MIN_SCORE:
                            continue
                        x, y, w, h = det.bbox
                        cv2.rectangle(vis_warp, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(vis_warp, f"{cname} {det.score:.2f}", (x, max(18, y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(vis_warp, f"{cname} {det.score:.2f}", (x, max(18, y-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
                    ensure_window("Warp (band)", 1280, 720)
                    cv2.imshow("Warp (band)", vis_warp)
                any_cv_window_open = True
                warp_window_open = True
            else:
                if warp_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Warp (band)")
                    except Exception:
                        pass
                    warp_window_open = False

            if show_windows and show_calib_preview and bundle.locked:
                if _do_display:
                    preview = render_calibration_preview(bundle, profiles=profiles)
                    ensure_window("Calibration Preview", 1280, 720)
                    cv2.imshow("Calibration Preview", preview)
                any_cv_window_open = True
                calib_preview_window_open = True
            else:
                if calib_preview_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Calibration Preview")
                    except Exception:
                        pass
                    calib_preview_window_open = False

            if show_windows and show_motion_warp_view and (bundle.motion_warp is not None):
                if _do_display:
                    ensure_window("Motion (warp)", 1280, 720)
                    cv2.imshow("Motion (warp)", bundle.motion_warp)
                any_cv_window_open = True
                motion_warp_window_open = True
            else:
                if motion_warp_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Motion (warp)")
                    except Exception:
                        pass
                    motion_warp_window_open = False

            if show_windows and show_motion_cam_view and (bundle.motion_cam is not None):
                if _do_display:
                    ensure_window("Motion (camera)", 1280, 720)
                    cv2.imshow("Motion (camera)", bundle.motion_cam)
                any_cv_window_open = True
                motion_cam_window_open = True
            else:
                if motion_cam_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Motion (camera)")
                    except Exception:
                        pass
                    motion_cam_window_open = False

            if show_windows and show_shadowfree_view and (bundle.shadowfree_cam is not None):
                if _do_display:
                    ensure_window("Shadow-free mask", 1280, 720)
                    cv2.imshow("Shadow-free mask", bundle.shadowfree_cam)
                any_cv_window_open = True
                shadowfree_window_open = True
            else:
                if shadowfree_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Shadow-free mask")
                    except Exception:
                        pass
                    shadowfree_window_open = False

            if show_windows and show_final_mask_view and (bundle.final_mask_cam is not None):
                if _do_display:
                    ensure_window("Final mask", 1280, 720)
                    cv2.imshow("Final mask", bundle.final_mask_cam)
                any_cv_window_open = True
                final_mask_window_open = True
            else:
                if final_mask_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Final mask")
                    except Exception:
                        pass
                    final_mask_window_open = False

            # Emit movements (with score gate + consensus)
            for color_name, det in dets.items():
                if det is None:
                    continue
                if det.score < MIN_SCORE:
                    continue

                col, row = warp_centroid_to_cell(det.cx, det.cy, bundle.grid_w, bundle.grid_h, bundle.warp_w, bundle.warp_h)
                cell = _cell_label_rc(row, col)

                buf = cell_hist.setdefault(color_name, deque(maxlen=CONSENSUS_N))
                buf.append(cell)

                most, count = Counter(buf).most_common(1)[0]
                prev = last_emitted.get(color_name)

                if count >= CONSENSUS_K and most != prev:
                    last_emitted[color_name] = most
                    if on_mini_moved is not None:
                        on_mini_moved(color_name, most)

            # Update positions panel every frame with last known positions
            positions = {}
            for color_name in dets:
                raw = last_emitted.get(color_name)
                if raw is not None:
                    parts = raw[1:].split('c')
                    try:
                        r_int, c_int = int(parts[0]), int(parts[1])
                        positions[color_name] = rc_to_a1(r_int, c_int)
                    except Exception:
                        positions[color_name] = raw
                else:
                    positions[color_name] = None
            panel.update_positions(positions)

            # Homography debug view
            if show_windows and show_h_view:
                core.show_homography_view(bundle.cam_bgr, bundle.H_use, bundle.warp_w, bundle.warp_h, bundle.grid_w, bundle.grid_h)
                homography_window_open = True
                any_cv_window_open = True
            else:
                if homography_window_open:
                    try:
                        cv2.destroyWindow("Homography view (debug)")
                    except Exception:
                        pass
                    homography_window_open = False

            # Allow quit via OpenCV key (non-blocking)
            if show_windows and any_cv_window_open:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('a'):
                    DRAW_ARUCO_OVERLAY = not DRAW_ARUCO_OVERLAY
                elif k == ord('h'):
                    show_h_view = not show_h_view
                elif k == ord('r'):
                    panel._act_recapture_bg()

    finally:
        try:
            sess.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    if switch_to:
        return {"switch_to": switch_to}
    return None