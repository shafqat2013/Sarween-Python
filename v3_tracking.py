# v3_tracking.py — Sarween combo tracking engine
#
# Blob detection: cv_core motion mask (same as blob_tracking — motion gated)
# Identity: CIE Lab color matching against stored profiles
# Profiles: combo_profiles.json — supports single Lab point OR multi-brightness curve

from __future__ import annotations

import json
import math
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

MIN_CIRCULARITY  = 0.1
MAX_ASPECT_RATIO = 2.5
MIN_AREA_FRAC    = 0.10
MAX_AREA_FRAC    = 9.0
SAMPLE_RADIUS_FRAC = 0.35
MORPH_CLOSE_FRAC   = 0.30

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

def _find_mini_blobs(
    bundle: "core.FrameBundle",
    grid_px: float,
    verbose: bool = False,
) -> List[Tuple[np.ndarray, float, float, float, float]]:
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


# ──────────────────────────────────────────────────────────────────────────────
# Matching — uses curve-aware Lab distance from mini_calibration
# ──────────────────────────────────────────────────────────────────────────────

def detect_minis(
    bundle: "core.FrameBundle",
    profiles: Dict[str, dict],
    grid_px: float,
    prev_state: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Tuple[Dict[str, Optional[ComboDetection]], Dict[str, Any]]:
    if prev_state is None:
        prev_state = {}

    warp_bgr = bundle.warp_bgr
    blobs = _find_mini_blobs(bundle, grid_px, verbose=verbose)

    if not blobs:
        if verbose:
            print("  MATCH | no blobs → all minis undetected this frame")
        detections: Dict[str, Optional[ComboDetection]] = {}
        new_state: Dict[str, Any] = {}
        for name in profiles:
            prev = prev_state.get(name, {})
            detections[name] = None
            new_state[name] = {"last_xy": prev.get("last_xy"), "last_dist": None}
        return detections, new_state

    # Sample Lab for each blob
    sampled: List[Optional[Tuple[float, float, float]]] = []
    for cnt, cx, cy, area, circ in blobs:
        equiv_r = math.sqrt(area / math.pi)
        sample_r = max(2.0, equiv_r * SAMPLE_RADIUS_FRAC)
        sampled.append(_sample_lab_at_centroid(warp_bgr, cx, cy, sample_r))

    if verbose:
        for i, lab in enumerate(sampled):
            if lab is not None:
                print(f"    [{i}] Lab=({lab[0]:.1f},{lab[1]:.1f},{lab[2]:.1f})")
            else:
                print(f"    [{i}] Lab=None (sampling failed)")

    profile_best: Dict[str, Optional[Tuple[int, float, float]]] = {}

    for name, prof in profiles.items():
        prev = prev_state.get(name, {})
        prev_xy = prev.get("last_xy")
        max_dist = float(prof.get("max_lab_dist", DEFAULT_MAX_LAB_DIST))
        expected_diam_sq = prof.get("expected_diameter_squares")
        if expected_diam_sq is not None:
            expected_r_px = expected_diam_sq * grid_px / 2.0
            expected_area_px = math.pi * expected_r_px ** 2
            # Tighter size window when unanchored (no prior position confirmed) to
            # reduce false positives from scene artifacts.  Once anchored, allow
            # wider range so a quick nudge or big sweep still registers.
            if prev_xy is None:
                area_lo = expected_area_px * 0.5
                area_hi = expected_area_px * 2.0
            else:
                area_lo = expected_area_px * 0.2
                area_hi = expected_area_px * 5.0
        else:
            expected_area_px = None
            area_lo = area_hi = None

        if verbose:
            prev_str = f"({prev_xy[0]:.0f},{prev_xy[1]:.0f})" if prev_xy else "None (unanchored)"
            area_str = f"area [{area_lo:.0f}–{area_hi:.0f}]" if area_lo is not None else "area [any]"
            anchor_note = " ← TIGHT filter, no spatial" if prev_xy is None else ""
            print(f"  MATCH | {name}  max_dist={max_dist:.0f}  {area_str}  prev={prev_str}{anchor_note}")

        best_idx, best_dist, best_score = None, float("inf"), -1.0

        for i, (cnt, cx, cy, area, circ) in enumerate(blobs):
            lab = sampled[i]
            if lab is None:
                if verbose:
                    print(f"    [{i}] SKIP  Lab sample failed")
                continue

            # Per-mini size filter: 20%–500% of calibrated area.
            if expected_area_px is not None:
                if area < area_lo or area > area_hi:
                    if verbose:
                        reason = "too small" if area < area_lo else "too large"
                        print(f"    [{i}] REJECT size  area={area:.0f} {reason} (range {area_lo:.0f}–{area_hi:.0f})")
                    continue

            # Lab distance filter (curve-aware: min across all brightness steps)
            dist = min_lab_dist_to_profile(lab, prof)
            if dist > max_dist:
                if verbose:
                    print(f"    [{i}] REJECT Lab  dist={dist:.1f} > max={max_dist:.0f}")
                continue

            # Spatial proximity: heavy penalty beyond 1 cell, hard reject beyond 3 cells
            if prev_xy is not None:
                d_px = math.hypot(cx - prev_xy[0], cy - prev_xy[1])
                if d_px > grid_px * 3.0:
                    if verbose:
                        print(f"    [{i}] REJECT spatial  d={d_px:.0f}px > {grid_px*3:.0f}px (3 cells)")
                    continue
                spatial_mult = 1.0 if d_px <= grid_px else 0.2
                spatial_note = f"d={d_px:.0f}px {'OK' if spatial_mult==1.0 else 'penalised×0.2'}"
            else:
                spatial_mult = 1.0
                spatial_note = "no-prev"

            color_term = 1.0 - (dist / max_dist)
            score = color_term * spatial_mult

            if verbose:
                best_mark = " ← best so far" if score > best_score else ""
                print(f"    [{i}] PASS  dist={dist:.1f}  {spatial_note}  score={score:.3f}{best_mark}")

            if score > best_score:
                best_score = score
                best_dist = dist
                best_idx = i

        profile_best[name] = (best_idx, best_dist, best_score) if best_idx is not None else None
        if verbose and best_idx is None:
            print(f"    → no candidate for {name}")

    # Conflict resolution: each blob goes to the highest-scoring profile
    claimed: Dict[int, str] = {}
    for name, result in profile_best.items():
        if result is None:
            continue
        idx, dist, score = result
        if idx not in claimed:
            claimed[idx] = name
        else:
            existing = profile_best[claimed[idx]]
            if existing is not None and score > existing[2]:
                if verbose:
                    print(f"  CONFLICT | blob[{idx}]: {claimed[idx]} (score={existing[2]:.3f}) "
                          f"→ overridden by {name} (score={score:.3f})")
                claimed[idx] = name
            elif verbose:
                print(f"  CONFLICT | blob[{idx}]: {name} (score={score:.3f}) "
                      f"lost to {claimed[idx]} (score={existing[2]:.3f})")

    # Merged blob detection: if a claimed blob's contour contains multiple minis'
    # last known positions, the blob is likely a merge — hold all those minis.
    merged_minis: set = set()
    for name, result in profile_best.items():
        if result is None or claimed.get(result[0]) != name:
            continue
        cnt = blobs[result[0]][0]
        minis_inside = [
            other for other in profiles
            if prev_state.get(other, {}).get("last_xy") is not None
            and cv2.pointPolygonTest(
                cnt,
                (float(prev_state[other]["last_xy"][0]), float(prev_state[other]["last_xy"][1])),
                False,
            ) >= 0
        ]
        if len(minis_inside) > 1:
            if verbose:
                print(f"  MERGE | blob[{result[0]}] contains prev positions of {minis_inside} → all held")
            merged_minis.update(minis_inside)

    detections, new_state = {}, {}
    for name, prof in profiles.items():
        result = profile_best.get(name)
        prev_xy = prev_state.get(name, {}).get("last_xy")

        if name in merged_minis or result is None or claimed.get(result[0]) != name:
            detections[name] = None
            new_state[name] = {"last_xy": prev_xy, "last_dist": None}
            continue

        idx, lab_dist, score = result
        cnt, cx, cy, area, circ = blobs[idx]
        lab = sampled[idx]
        x, y, w, h = cv2.boundingRect(cnt)

        detections[name] = ComboDetection(
            mini_id=name, cx=cx, cy=cy, lab_dist=lab_dist,
            contour_area=area, circularity=circ, score=score,
            bbox=(int(x), int(y), int(w), int(h)), sampled_lab=lab,
        )
        # Don't anchor last_xy on first detection — wait for consensus to fire
        # (see begin_session emit loop).  If we set it here, a one-frame artifact
        # match would anchor the spatial filter to the wrong location and then
        # reject the real mini as "too far away".
        if prev_xy is None:
            new_state[name] = {"last_xy": None, "last_dist": lab_dist}
        else:
            new_state[name] = {"last_xy": (cx, cy), "last_dist": lab_dist}

    return detections, new_state


def _dump_tracking_state(
    profiles: Dict[str, dict],
    prev_state: Dict[str, Any],
    cell_hist: Dict[str, Any],
    last_emitted: Dict[str, str],
    last_seen: Optional[Dict[str, float]] = None,
) -> None:
    """Print a full snapshot of the current tracking state to terminal."""
    now_t = time.perf_counter()
    print("=" * 60)
    print("DUMP STATE")
    print(f"  profiles loaded: {list(profiles.keys())}")
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


def _cell_label(row, col):
    return f"r{int(row)}c{int(col)}"


# ──────────────────────────────────────────────────────────────────────────────
# Single-point calibration (fallback — still works without full curve capture)
# ──────────────────────────────────────────────────────────────────────────────

def _save_profiles(profiles: Dict) -> None:
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")


def calibrate_from_bundle(
    bundle: "core.FrameBundle",
    name: str,
    existing_profiles: Dict,
) -> Optional[Dict]:
    if not bundle.locked or bundle.warp_bgr is None:
        return None

    grid_px = min(bundle.warp_w / float(bundle.grid_w),
                  bundle.warp_h / float(bundle.grid_h))
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

    grid_px = min(bundle.warp_w / float(bundle.grid_w),
                  bundle.warp_h / float(bundle.grid_h))
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

    # Sync the Record button label if we resumed recording
    if sess.is_recording:
        panel.set_recording_status(True, sess._record_path or "")

    try:
        fo.set_grid_params(sess.warp_w, sess.warp_h, sess.grid_w, sess.grid_h)
    except Exception:
        pass

    control_panel_shown = False
    prev_state: Dict[str, Any] = {}
    cell_hist: Dict[str, deque] = {}
    last_emitted: Dict[str, str] = {}

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

            if actions.get("calibrate_band"):
                req = actions.get("calibrate_band")
                name = None
                if isinstance(req, dict):
                    name = (req.get("name") or "").strip() or None
                if not name:
                    name = time.strftime("mini_%Y%m%d_%H%M%S")

                if not bundle.locked:
                    panel.set_hint("Calibrate: wait for ArUco lock")
                else:
                    prof = calibrate_from_bundle(bundle, name, profiles)
                    if prof is None:
                        panel.set_hint("Calibrate: no motion blob — move the mini first")
                    else:
                        profiles[name] = prof
                        try:
                            _save_profiles(profiles)
                            lab = prof["lab"]
                            panel.set_hint(f"Calibrated: {name}  Lab=({lab[0]:.0f},{lab[1]:.0f},{lab[2]:.0f})")
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
                    sess.stop_recording()
                    panel.set_recording_status(False)
                    panel.set_hint("Recording saved ✅")
                else:
                    rec_path = str(
                        Path(__file__).parent
                        / f"sarween_rec_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    )
                    ok = sess.start_recording(rec_path, fps=max(1.0, _current_fps))
                    if ok:
                        panel.set_recording_status(True, rec_path)
                        panel.set_hint(f"Recording → {rec_path}")
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
                                     last_seen=_last_seen)

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

            if bundle.locked and not control_panel_shown:
                panel.show()
                control_panel_shown = True

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

            grid_px = min(bundle.warp_w / float(bundle.grid_w),
                          bundle.warp_h / float(bundle.grid_h))

            _do_verbose = verbose_tracking and bundle.locked

            # Lost-mini timeout: if anchored but undetected for >LOST_TIMEOUT s,
            # clear the spatial anchor so the mini can re-lock anywhere.
            _now_t = time.perf_counter()
            for _name in list(prev_state.keys()):
                _ps = prev_state[_name]
                if _ps.get("last_xy") is not None:
                    _age = _now_t - _last_seen.get(_name, _now_t)
                    if _age > LOST_TIMEOUT:
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

            # Update last-seen timestamps for detected minis.
            # Exception: if the blob centroid is within 0.5 cells of the current
            # spatial anchor it's either a static background match or the "wake"
            # that lingers at the old position after a pick-up.  Don't reset the
            # lost-timer in that case — let it expire so search mode kicks in.
            for _name, _det in dets.items():
                if _det is None:
                    continue
                _prev_xy = prev_state.get(_name, {}).get("last_xy")
                if _prev_xy is not None:
                    _d_cells = math.hypot(
                        _det.cx - _prev_xy[0], _det.cy - _prev_xy[1]
                    ) / grid_px
                    if _d_cells < 0.5:
                        if verbose_tracking:
                            print(f"V3 | {_name} wake/static hit "
                                  f"({_d_cells:.2f} cells from anchor) — "
                                  f"not resetting lost timer")
                        continue  # static / wake — don't refresh timer
                _last_seen[_name] = _now_t

            # Heartbeat when verbose but nothing detected (throttled to once/3s)
            if verbose_tracking and bundle.locked and not any(d is not None for d in dets.values()):
                if _now_t - _last_no_blob_msg >= 3.0:
                    _last_no_blob_msg = _now_t
                    print(f"TRACK | f{bundle.frame_idx} locked, no motion blobs detected (move a mini)")

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
                                          bundle.grid_w, bundle.grid_h)
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
                col, row = _warp_to_cell(det.cx, det.cy,
                                         bundle.grid_w, bundle.grid_h,
                                         bundle.warp_w, bundle.warp_h)
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
                if count >= CONSENSUS_K and most != last_emitted.get(mname):
                    last_emitted[mname] = most
                    # First-ever consensus for this mini: anchor the spatial filter
                    # to the confirmed cell center so future frames can use it.
                    if prev_state.get(mname, {}).get("last_xy") is None:
                        parts = most[1:].split('c')
                        r_idx, c_idx = int(parts[0]), int(parts[1])
                        cell_w = bundle.warp_w / float(bundle.grid_w)
                        cell_h = bundle.warp_h / float(bundle.grid_h)
                        anchor = ((c_idx + 0.5) * cell_w, (r_idx + 0.5) * cell_h)
                        if mname not in prev_state:
                            prev_state[mname] = {}
                        prev_state[mname]["last_xy"] = anchor
                        print(f"V3 | {mname} anchored at {rc_to_a1(r_idx, c_idx)} "
                              f"({anchor[0]:.0f},{anchor[1]:.0f})", flush=True)
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

            if show_windows and any_cv_window_open:
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break

    finally:
        try:
            sess.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return None
