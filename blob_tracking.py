# blob_tracking.py
#
# Blob/DB-based tracking engine for Sarween.
# Uses shared CV pipeline from cv_core.py, then runs mini_tracking.identify_minis()
# and converts contours -> grid cells -> on_mini_moved callbacks.

import os
import cv2
import math
import numpy as np
import time
from collections import deque, Counter

import setup as s
import mini_tracking as mt
import calibration as c
import foundryoutput as fo
from control_panel import ControlPanel

import cv_core as core
from control_panel import rc_to_a1

# ──────────────────────────────────────────────────────────────────────────────
# Defaults / tuning (kept from your original tracking.py)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_W = 23
DEFAULT_GRID_H = 16
DEFAULT_WARP_W = 1280
DEFAULT_WARP_H = 720

MAX_MATCH_DIST = 50
STALE_FOR = 30
CONSENSUS_N = 5
CONSENSUS_K = 3

DRAW_ARUCO_OVERLAY = False  # toggle with 'a'

# recent known mini_ids (used to add views under an existing mini_id)
recent_known_ids = deque(maxlen=9)

known_hist_buf = {}
known_last_cell = {}

# ──────────────────────────────────────────────────────────────────────────────
# Window helper — provided by cv_core
# ──────────────────────────────────────────────────────────────────────────────
ensure_window = core.ensure_window

# ──────────────────────────────────────────────────────────────────────────────
# Unique name (best-fit) logic (copied)
# ──────────────────────────────────────────────────────────────────────────────
NAME_OWNER_STALE = 120
_name_owner = {}
_name_owner_last_seen = {}

def _norm_name(n: str) -> str:
    return (n or "").strip().lower()

def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _enforce_unique_names_best_fit(detections, cur_frame_idx):
    stale = [k for k, last in _name_owner_last_seen.items()
             if (cur_frame_idx - last) > NAME_OWNER_STALE]
    for k in stale:
        _name_owner.pop(k, None)
        _name_owner_last_seen.pop(k, None)

    by_name = {}
    for d in detections:
        if d.get("label") != "known" or not d.get("match_id"):
            continue
        nm = (d.get("name") or "").strip()
        if not nm:
            continue
        key = _norm_name(nm)
        if not key:
            continue
        by_name.setdefault(key, []).append(d)

    for name_key, group in by_name.items():
        if len(group) <= 1:
            mid = group[0].get("match_id")
            if mid:
                _name_owner[name_key] = mid
                _name_owner_last_seen[name_key] = cur_frame_idx
            continue

        preferred_mid = _name_owner.get(name_key)
        winner = None
        if preferred_mid is not None:
            for d in group:
                if d.get("match_id") == preferred_mid:
                    winner = d
                    break

        if winner is None:
            def rank(d):
                sc = _safe_float(d.get("score"))
                ar = _safe_float(d.get("area"))
                return (sc if sc is not None else -1e9, ar if ar is not None else 0.0)
            winner = max(group, key=rank)

        keep_name = (winner.get("name") or "").strip()
        for d in group:
            if d is winner:
                d["name"] = keep_name
            else:
                d["name"] = ""

        win_mid = winner.get("match_id")
        if win_mid:
            _name_owner[name_key] = win_mid
            _name_owner_last_seen[name_key] = cur_frame_idx

# ──────────────────────────────────────────────────────────────────────────────
# Text input modal (OpenCV) (copied)
# ──────────────────────────────────────────────────────────────────────────────
def _prompt_text(title, prompt, initial=""):
    text = (initial or "")
    W, H = 900, 220
    ensure_window(title, W, H)

    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        cv2.putText(canvas, prompt, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Enter = confirm   Esc = cancel", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

        cv2.rectangle(canvas, (20, 140), (W-20, 190), (80,80,80), 2)
        cv2.putText(canvas, text, (30, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(title, canvas)
        k = cv2.waitKey(0) & 0xFF

        if k in (27,):
            cv2.destroyWindow(title)
            return None
        if k in (10, 13):
            cv2.destroyWindow(title)
            return text.strip()
        if k in (8, 127):
            text = text[:-1]
            continue
        if 32 <= k <= 126:
            text += chr(k)

# ──────────────────────────────────────────────────────────────────────────────
# Capture Picker (copied)
# ──────────────────────────────────────────────────────────────────────────────
def _capture_picker_select_contour(frame_bgr, mask_u8, min_area, min_fill, min_solidity, max_candidates=6):
    if frame_bgr is None or mask_u8 is None:
        return None

    fg = cv2.erode(mask_u8, None, iterations=1)
    fg = cv2.dilate(fg, None, iterations=1)

    cnts_info = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        print("⚠️ Capture Picker: no contours found.")
        return None

    candidates = []
    for c_ in cnts:
        area = float(cv2.contourArea(c_))
        if area < float(min_area):
            continue
        x, y, w, h = cv2.boundingRect(c_)
        fill = area / float(w * h) if w > 0 and h > 0 else 0.0
        hull = cv2.convexHull(c_)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and hull.size > 0 else 0.0
        solidity = (area / hull_area) if hull_area > 0 else 0.0
        if fill < float(min_fill) or solidity < float(min_solidity):
            continue
        candidates.append((area, c_))

    if not candidates:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        candidates = [(float(cv2.contourArea(c_)), c_) for c_ in cnts_sorted[:max_candidates]]
    else:
        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:max_candidates]

    show = frame_bgr.copy()
    overlay = show.copy()
    alpha_ov = 0.35

    for _, c_ in candidates:
        cv2.drawContours(overlay, [c_.astype(np.int32)], -1, (255, 255, 0), thickness=-1)
    show = cv2.addWeighted(overlay, alpha_ov, show, 1.0 - alpha_ov, 0)

    for idx, (_, c_) in enumerate(candidates, start=1):
        x, y, w, h = cv2.boundingRect(c_)
        cv2.drawContours(show, [c_.astype(np.int32)], -1, (255, 255, 0), 2)
        cv2.rectangle(show, (x, y), (x+w, y+h), (255, 255, 0), 2)

        label = f"{idx}"
        cv2.putText(show, label, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(show, label, (x, max(20, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    header = "CAPTURE PICKER: press 1..9 to choose, Esc/q to cancel"
    cv2.putText(show, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(show, header, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    win = "Capture Picker"
    ensure_window(win, 1280, 720)
    cv2.imshow(win, show)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord('q')):
            cv2.destroyWindow(win)
            return None
        if ord('1') <= key <= ord('9'):
            pick = (key - ord('1'))
            if 0 <= pick < len(candidates):
                cv2.destroyWindow(win)
                return candidates[pick][1]

def _get_all_mini_ids():
    """Return list of (mini_id, name) tuples from the DB, for the capture picker."""
    import csv as _csv
    if not os.path.exists(mt.DB_CSV):
        return []
    seen = {}
    try:
        with open(mt.DB_CSV, "r", newline="") as f:
            for row in _csv.DictReader(f):
                mid = (row.get("mini_id") or "").strip()
                name = (row.get("name") or "").strip()
                if mid and mid not in seen:
                    seen[mid] = name
    except Exception:
        return []
    return [(mid, name) for mid, name in seen.items()]


def _choose_mini_id_for_capture():
    # Load all known minis from DB; fall back to recently-matched if DB empty
    db_ids = _get_all_mini_ids()  # list of (mini_id, name)
    ids = db_ids if db_ids else [(mid, "") for mid in recent_known_ids]

    canvas = np.zeros((320, 900, 3), dtype=np.uint8)

    def draw(y, txt, scale=0.6):
        cv2.putText(canvas, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

    draw(35,  "ASSIGN CAPTURE:", 0.75)
    draw(75,  "n = new mini (new mini_id)")
    draw(105, "1..9 = add as a new view to an existing mini")
    draw(135, "Esc/q = cancel")

    if ids:
        draw(175, "Known minis (from DB):" if db_ids else "Recent known minis:", 0.6)
        y = 205
        for i, (mid, name) in enumerate(ids[:9], start=1):
            label = f"{i}. {name}  [{mid}]" if name else f"{i}. {mid}"
            draw(y, label, 0.55)
            y += 22
    else:
        draw(175, "(No minis in DB yet — press 'n')", 0.6)

    win = "Assign Capture"
    ensure_window(win, 900, 320)
    cv2.imshow(win, canvas)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, ord('q')):
            cv2.destroyWindow(win)
            return None
        if k == ord('n'):
            cv2.destroyWindow(win)
            return "NEW"
        if ord('1') <= k <= ord('9') and ids:
            idx = (k - ord('1'))
            if 0 <= idx < len(ids):
                cv2.destroyWindow(win)
                return ids[idx][0]  # return just the mini_id

# ──────────────────────────────────────────────────────────────────────────────
# Grid cell selection from contour (warp space) (copied)
# ──────────────────────────────────────────────────────────────────────────────
def pick_base_cell_from_contour(contour_warped, grid_w, grid_h, warp_w, warp_h, min_coverage=0.6):
    if contour_warped is None or len(contour_warped) == 0:
        return None

    cell_w = warp_w / float(grid_w)
    cell_h = warp_h / float(grid_h)
    mask = np.zeros((int(warp_h), int(warp_w)), dtype=np.uint8)
    cv2.drawContours(mask, [contour_warped.astype(np.int32)], -1, 255, thickness=-1)

    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    c0 = max(0, min(int(x0 // cell_w), grid_w - 1))
    c1 = max(0, min(int(x1 // cell_w), grid_w - 1))
    r0 = max(0, min(int(y0 // cell_h), grid_h - 1))
    r1 = max(0, min(int(y1 // cell_h), grid_h - 1))

    covered_cells = []
    best_any = (None, None, 0.0)

    for r in range(r0, r1 + 1):
        for c_ in range(c0, c1 + 1):
            xA, xB = int(c_ * cell_w), int((c_ + 1) * cell_w)
            yA, yB = int(r * cell_h), int((r + 1) * cell_h)
            roi = mask[yA:yB, xA:xB]
            if roi.size == 0:
                continue
            cov = cv2.countNonZero(roi) / float(roi.size)

            if cov > best_any[2]:
                best_any = (r, c_, cov)

            if cov >= min_coverage:
                covered_cells.append((r, c_, cov))

    if not covered_cells:
        r_best, c_best, cov_best = best_any
        if r_best is not None and cov_best >= min_coverage:
            return f"r{r_best}c{c_best}"
        return None

    max_row = max(r for (r, c_, cov) in covered_cells)
    candidates = [entry for entry in covered_cells if entry[0] == max_row]
    candidates.sort(key=lambda t: (-t[2], t[1]))
    r_final, c_final, _ = candidates[0]
    return f"r{r_final}c{c_final}"

# ──────────────────────────────────────────────────────────────────────────────
# Detection consolidation + movement reporting (copied)
# ──────────────────────────────────────────────────────────────────────────────
def _merge_frame_detections(dets, max_dist=MAX_MATCH_DIST):
    by_known, unknowns = {}, []
    for d in dets:
        if d.get("label") == "known" and d.get("match_id"):
            mid = d["match_id"]
            if (mid not in by_known) or (d["area"] > by_known[mid]["area"]):
                by_known[mid] = d
        else:
            unknowns.append(d)

    taken, merged = set(), []
    for i, di in enumerate(unknowns):
        if i in taken:
            continue
        xi, yi, wi, hi = di["bbox"]
        cxi, cyi = xi + wi / 2.0, yi + hi / 2.0
        group = [di]
        taken.add(i)
        for j in range(i + 1, len(unknowns)):
            if j in taken:
                continue
            dj = unknowns[j]
            xj, yj, wj, hj = dj["bbox"]
            cxj, cyj = xj + wj / 2.0, yj + hj / 2.0
            if math.hypot(cxi - cxj, cyi - cyj) <= max_dist:
                group.append(dj)
                taken.add(j)
        merged.append(max(group, key=lambda q: q["area"]))

    return list(by_known.values()) + merged

def _maybe_report_movement(mini_id, base_cell, on_mini_moved=None, panel=None):
    if base_cell is None:
        return
    buf = known_hist_buf.setdefault(mini_id, deque(maxlen=CONSENSUS_N))
    buf.append(base_cell)
    most, count = Counter(buf).most_common(1)[0]
    last = known_last_cell.get(mini_id)
    if count >= CONSENSUS_K:
        if last is None:
            known_last_cell[mini_id] = most
        elif most != last:
            # Parse r,c for display
            parts = most[1:].split('c')
            r_int, c_int = int(parts[0]), int(parts[1])
            display_coord = rc_to_a1(r_int, c_int)
            print(f"{mini_id}, {last}, {most}")
            known_last_cell[mini_id] = most
            if panel is not None:
                panel.log_movement(mini_id, display_coord)
            if on_mini_moved is not None:
                on_mini_moved(mini_id, most)

# ──────────────────────────────────────────────────────────────────────────────
# Mask annotation (copied)
# ──────────────────────────────────────────────────────────────────────────────
def _annotate_mask_blobs(mask_u8, min_area, min_fill, min_solidity, max_k=12):
    if mask_u8 is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    m = (mask_u8 > 0).astype(np.uint8) * 255
    bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

    cnts_info = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        return bgr

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_k]

    for i, c_ in enumerate(cnts, start=1):
        area = float(cv2.contourArea(c_))
        x, y, w, h = cv2.boundingRect(c_)
        fill = area / float(w * h) if w > 0 and h > 0 else 0.0

        hull = cv2.convexHull(c_)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and hull.size > 0 else 0.0
        solidity = (area / hull_area) if hull_area > 0 else 0.0

        pass_area = area >= float(min_area)
        pass_fill = fill >= float(min_fill)
        pass_sol = solidity >= float(min_solidity)

        if pass_area and pass_fill and pass_sol:
            col = (0, 200, 0)
        elif not pass_area:
            col = (0, 0, 255)
        else:
            col = (0, 165, 255)

        cv2.rectangle(bgr, (x, y), (x + w, y + h), col, 2)

        y1 = max(12, y - 6)
        y2 = max(24, y1 - 14)

        stats = f"{i}: A={int(area)} f={fill:.2f} s={solidity:.2f}"
        flags = f"pass: A={'Y' if pass_area else 'N'} f={'Y' if pass_fill else 'N'} s={'Y' if pass_sol else 'N'}"

        cv2.putText(bgr, stats, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, stats, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(bgr, flags, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr, flags, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    return bgr

# ──────────────────────────────────────────────────────────────────────────────
# Homography debug view — provided by cv_core
# ──────────────────────────────────────────────────────────────────────────────
show_homography_view = core.show_homography_view

# ──────────────────────────────────────────────────────────────────────────────
# Main session (blob engine) — uses cv_core
# ──────────────────────────────────────────────────────────────────────────────
def begin_session(
    on_mini_moved,
    camera_index=None,
    known_threshold=0.50,
    min_fill=0.35,
    min_solidity=0.75,
    show_windows=True,
):
    global DRAW_ARUCO_OVERLAY

    # Load defaults from setup
    if camera_index is None:
        sel = s.load_last_selection() or {}
        camera_index = sel.get("webcam_index", 0)

    # Use values computed in calibration if available
    warp_w = getattr(s, "warp_w", None) or DEFAULT_WARP_W
    warp_h = getattr(s, "warp_h", None) or DEFAULT_WARP_H
    grid_w = getattr(s, "grid_cols", None) or DEFAULT_GRID_W
    grid_h = getattr(s, "grid_rows", None) or DEFAULT_GRID_H

    print(f"[BLOB] warp={warp_w}x{warp_h} camera_index={camera_index}", flush=True)

    mode = (s.load_last_selection() or {}).get("mode", "self_hosted")
    panel = ControlPanel(mode=mode)
    switch_to = None

    # Start hidden; show after lock
    panel.hide()

    # Blended only meaningful self-hosted
    show_blended_view = (mode == "self_hosted")

    if show_windows:
        if show_blended_view:
            c.show_blended_display_window()
        else:
            c.hide_blended_display_window()

    # Core session
    try:
        sess = core.CVCoreSession(
            camera_index=camera_index,
            warp_w=int(warp_w),
            warp_h=int(warp_h),
            grid_w=int(grid_w),
            grid_h=int(grid_h),
            lock_drop_after=core.LOCK_DROP_AFTER,
        )
    except Exception as e:
        print(f"❌ CV core init failed: {e}")
        return

    # Window state
    cam_window_open = False
    ident_window_open = False
    homography_window_open = False

    motion_warp_window_open = False
    motion_cam_window_open = False
    shadowfree_window_open = False
    final_mask_window_open = False

    control_panel_shown = False
    show_cam_view = True
    show_h_view = False
    show_ident_view = False

    fps_count = 0
    last_print = time.perf_counter()

    def _fmt2(v):
        return "--" if v is None else f"{float(v):.2f}"

    try:
        for bundle in sess.frames():
            fps_count += 1
            now = time.perf_counter()
            if now - last_print >= 1.0:
                fps = fps_count / (now - last_print)
                panel.set_status(fps=fps)
                fps_count = 0
                last_print = now

            # UI pump
            if not panel.pump():
                break

            actions = panel.pop_actions()

            # Switch engine (restart)
            if actions.get("switch_engine") in ("blob", "band"):
                switch_to = actions.get("switch_engine")
                break

            # Recapture BG: update core session state
            if actions.get("recapture_bg"):
                try:
                    cam = bundle.cam_bgr
                    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (21, 21), 0)
                    sess.BG_cam = {"bgr": cam.copy(), "blur": blur}
                    bg_init = core.warp_gray_blur(cam, sess.H_saved, sess.warp_w, sess.warp_h)
                    sess.BG_warp_f32 = bg_init.astype(np.float32)
                except Exception:
                    pass

            # Status update (markers/lock)
            try:
                foundry_connected = None
                if hasattr(fo, "is_connected"):
                    try:
                        foundry_connected = bool(fo.is_connected())
                    except Exception:
                        foundry_connected = None

                panel.set_status(
                    foundry_connected=foundry_connected,
                    locked=bool(bundle.locked),
                    marker_count=int(bundle.last_marker_count),
                    missing_ids=list(bundle.last_missing_ids),
                )
            except Exception:
                pass

            # Pre-lock camera view and exit early
            if not bundle.locked:
                if show_windows and show_cam_view:
                    vis_cam = bundle.cam_bgr.copy()

                    if DRAW_ARUCO_OVERLAY and core.ARUCO_DET is not None:
                        if bundle.last_aruco_ids is not None and bundle.last_aruco_corners is not None:
                            try:
                                import cv2.aruco as aruco
                                aruco.drawDetectedMarkers(vis_cam, bundle.last_aruco_corners, bundle.last_aruco_ids)
                            except Exception:
                                pass

                    msg = f"Waiting for ArUco lock ({bundle.last_marker_count}/4), missing: {bundle.last_missing_ids}"
                    cv2.putText(vis_cam, msg, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                    ensure_window("Camera + Detections", 1280, 720)
                    cv2.imshow("Camera + Detections", vis_cam)
                    cam_window_open = True

                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    DRAW_ARUCO_OVERLAY = not DRAW_ARUCO_OVERLAY
                elif key == ord('q'):
                    break
                continue

            # First-time lock: show panel, optionally close cam window
            if bundle.locked and not control_panel_shown:
                panel.show()
                control_panel_shown = True
                # optional: stop showing cam by default after lock
                show_cam_view = False
                try:
                    cv2.destroyWindow("Camera + Detections")
                except Exception:
                    pass
                cam_window_open = False

            # Control panel toggles
            tog = panel.get_toggles()
            show_cam_view = bool(tog.get("show_live_camera", False))
            show_h_view = bool(tog.get("show_homography", False))
            show_ident_view = bool(tog.get("show_identify", False))
            show_blended_view = bool(tog.get("show_blended", False))

            show_motion_warp_view = bool(tog.get("show_motion_warp", False))
            show_motion_cam_view = bool(tog.get("show_motion_cam", False))
            show_shadowfree_view = bool(tog.get("show_shadowfree", False))
            show_final_mask_view = bool(tog.get("show_final_mask", False))

            # Keep blended window in sync
            if show_windows:
                if show_blended_view:
                    if not c.is_blended_display_window_open():
                        c.show_blended_display_window()
                else:
                    if c.is_blended_display_window_open():
                        c.hide_blended_display_window()

            cam = bundle.cam_bgr
            final_mask_cam = bundle.final_mask_cam

            # Capture action is handled below, after min_area_for_ident is computed.
            # See "Handle capture action" block further down in this loop.

            # Dynamic thresholds (same math as original tracking.py)
            H_use = bundle.H_use
            warp_w = int(bundle.warp_w)
            warp_h = int(bundle.warp_h)
            grid_w = int(bundle.grid_w)
            grid_h = int(bundle.grid_h)

            try:
                H_inv_for_area = np.linalg.inv(H_use) if H_use is not None else np.eye(3, dtype=np.float32)
            except Exception:
                H_inv_for_area = np.eye(3, dtype=np.float32)

            canvas_pts = np.array([[[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]]], dtype=np.float32)
            canvas_cam_quad = cv2.perspectiveTransform(canvas_pts, H_inv_for_area).reshape(-1,2)
            x = canvas_cam_quad[:,0]; y = canvas_cam_quad[:,1]
            area_cam_canvas = 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
            area_warp_canvas = float(warp_w * warp_h) if warp_w*warp_h > 0 else 1.0
            area_scale = max(1e-6, area_cam_canvas / area_warp_canvas)

            cell_area_warp = (warp_w/float(grid_w)) * (warp_h/float(grid_h))
            cell_area_cam  = area_scale * cell_area_warp

            min_area_dyn = 0.64 * cell_area_cam
            max_area_dyn = 4.00 * cell_area_cam
            min_area_for_ident = max(100, int(0.10 * cell_area_cam))

            # Identify minis (blob engine)
            ident = mt.identify_minis(
                cam,
                diff_mask=final_mask_cam,
                db_entries=None,
                db_csv_path=mt.DB_CSV,
                min_area=int(min_area_for_ident),
                min_fill=min_fill,
                min_solidity=min_solidity,
                known_threshold=known_threshold,
                keep_top_k=10,
                skip_border=True,
                draw=True,
                show_components=True
            )

            raw_dets = ident.get("detections", [])
            detections = _merge_frame_detections(raw_dets, max_dist=MAX_MATCH_DIST)
            _enforce_unique_names_best_fit(detections, bundle.frame_idx)

            # Track recent known IDs
            for d in detections:
                if d.get("label") == "known" and d.get("match_id"):
                    mid = d["match_id"]
                    if mid in recent_known_ids:
                        recent_known_ids.remove(mid)
                    recent_known_ids.appendleft(mid)

            # Build camera visualization
            vis_cam = cam.copy()

            if DRAW_ARUCO_OVERLAY and core.ARUCO_DET is not None:
                if bundle.last_aruco_ids is not None and bundle.last_aruco_corners is not None:
                    try:
                        import cv2.aruco as aruco
                        aruco.drawDetectedMarkers(vis_cam, bundle.last_aruco_corners, bundle.last_aruco_ids)
                    except Exception:
                        pass

            # Draw detections + report movement
            for d in detections:
                x, y, w, h = d["bbox"]
                area = float(d.get("area", 0.0))
                label = d.get("label", "unknown")
                match_id = d.get("match_id")
                name = (d.get("name") or "").strip()

                eligible = (area >= min_area_dyn) and (area <= max_area_dyn)

                score_val = d.get("score", None)
                try:
                    score_val = float(score_val) if score_val is not None else None
                except Exception:
                    score_val = None

                h_sim = d.get("h_sim")
                o_sim = d.get("o_sim")
                s_sim = d.get("s_sim")

                if label == "known" and match_id:
                    color = (0, 200, 0)
                    display = (name + " " if name else "") + f"#{str(match_id)}"
                else:
                    color = (0, 165, 255) if eligible else (0, 0, 255)
                    display = "unknown"

                headline = f"{display}  A={int(area)}" if score_val is None else f"{display}  A={int(area)}  S={score_val:.3f}"
                comp = f"H={_fmt2(h_sim)}  O={_fmt2(o_sim)}  Sh={_fmt2(s_sim)}"

                cv2.rectangle(vis_cam, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

                y1 = max(0, int(y) - 6)
                y2 = max(0, y1 - 14)

                cv2.putText(vis_cam, headline, (int(x), y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis_cam, headline, (int(x), y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

                cv2.putText(vis_cam, comp, (int(x), y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis_cam, comp, (int(x), y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1, cv2.LINE_AA)

                cnt_cam = d.get("contour")
                if cnt_cam is not None and H_use is not None:
                    pts = cnt_cam.reshape(-1,1,2).astype(np.float32)
                    cnt_warp = cv2.perspectiveTransform(pts, H_use).astype(np.float32)
                    base_cell = pick_base_cell_from_contour(
                        cnt_warp, grid_w, grid_h, int(warp_w), int(warp_h), 0.60
                    )
                    if label == "known" and match_id:
                        _maybe_report_movement(match_id, base_cell, on_mini_moved, panel=panel)

            # Render windows
            if show_windows and show_cam_view:
                ensure_window("Camera + Detections", 1280, 720)
                cv2.imshow("Camera + Detections", vis_cam)
                cam_window_open = True
            else:
                if cam_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Camera + Detections")
                    except Exception:
                        pass
                    cam_window_open = False

            # Identify Debug (from mini_tracking)
            if show_windows and show_ident_view and ("annotated" in ident) and (ident["annotated"] is not None):
                ensure_window("Identify Debug", 1280, 720)
                cv2.imshow("Identify Debug", ident["annotated"])
                ident_window_open = True
            else:
                if ident_window_open:
                    try: cv2.destroyWindow("Identify Debug")
                    except Exception: pass
                    ident_window_open = False

            # Mask windows (use bundle masks)
            if show_windows and show_motion_warp_view and bundle.motion_warp is not None:
                ensure_window("Motion (warp)", 1280, 720)
                cv2.imshow("Motion (warp)", bundle.motion_warp)
                motion_warp_window_open = True
            else:
                if motion_warp_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Motion (warp)")
                    except Exception:
                        pass
                    motion_warp_window_open = False

            if show_windows and show_motion_cam_view and bundle.motion_cam is not None:
                ensure_window("Motion (camera)", 1280, 720)
                cv2.imshow("Motion (camera)", bundle.motion_cam)
                motion_cam_window_open = True
            else:
                if motion_cam_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Motion (camera)")
                    except Exception:
                        pass
                    motion_cam_window_open = False

            if show_windows and show_shadowfree_view and bundle.shadowfree_cam is not None:
                ensure_window("Shadow-free mask", 1280, 720)
                cv2.imshow("Shadow-free mask", bundle.shadowfree_cam)
                shadowfree_window_open = True
            else:
                if shadowfree_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Shadow-free mask")
                    except Exception:
                        pass
                    shadowfree_window_open = False

            if show_windows and show_final_mask_view and final_mask_cam is not None:
                final_mask_annot = _annotate_mask_blobs(
                    final_mask_cam,
                    min_area=int(min_area_for_ident),
                    min_fill=min_fill,
                    min_solidity=min_solidity,
                    max_k=12
                )
                ensure_window("Final mask (annotated)", 1280, 720)
                cv2.imshow("Final mask (annotated)", final_mask_annot)
                final_mask_window_open = True
            else:
                if final_mask_window_open:
                    try:
                        cv2.waitKey(1)
                        cv2.destroyWindow("Final mask (annotated)")
                    except Exception:
                        pass
                    final_mask_window_open = False

            # Homography debug view
            if show_windows and show_h_view:
                show_homography_view(cam, H_use, warp_w, warp_h, grid_w, grid_h)
                homography_window_open = True
            else:
                if homography_window_open:
                    try: cv2.destroyWindow("Homography view (debug)")
                    except Exception: pass
                    homography_window_open = False

            # Handle capture action now that min_area_for_ident is known
            if actions.get("capture") and final_mask_cam is not None:
                selected = _capture_picker_select_contour(
                    frame_bgr=cam,
                    mask_u8=final_mask_cam,
                    min_area=int(min_area_for_ident),
                    min_fill=min_fill,
                    min_solidity=min_solidity,
                    max_candidates=6
                )
                if selected is None:
                    print("ℹ️ Capture cancelled.")
                else:
                    choice = _choose_mini_id_for_capture()
                    if choice is None:
                        print("ℹ️ Capture cancelled.")
                    elif choice == "NEW":
                        name = _prompt_text("Mini Name", "Enter mini name (optional):", initial="")
                        if name is None:
                            name = ""
                        mt.save_mini_from_frame_and_contour(
                            frame_bgr=cam,
                            contour=selected,
                            min_area=int(min_area_for_ident),
                            save_dir="mini_captures",
                            return_info=False,
                            mini_id=None,
                            name=name
                        )
                    else:
                        mt.save_mini_from_frame_and_contour(
                            frame_bgr=cam,
                            contour=selected,
                            min_area=int(min_area_for_ident),
                            save_dir="mini_captures",
                            return_info=False,
                            mini_id=choice,
                            name=None
                        )

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                DRAW_ARUCO_OVERLAY = not DRAW_ARUCO_OVERLAY
            elif key == ord('h'):
                show_h_view = not show_h_view
            elif key == ord('q'):
                break

    finally:
        try:
            sess.close()
        except Exception:
            pass

        if show_windows:
            try:
                if cam_window_open:
                    cv2.destroyWindow("Camera + Detections")
                if ident_window_open:
                    cv2.destroyWindow("Identify Debug")
                if homography_window_open:
                    cv2.destroyWindow("Homography view (debug)")
                if motion_warp_window_open:
                    cv2.destroyWindow("Motion (warp)")
                if motion_cam_window_open:
                    cv2.destroyWindow("Motion (camera)")
                if shadowfree_window_open:
                    cv2.destroyWindow("Shadow-free mask")
                if final_mask_window_open:
                    cv2.destroyWindow("Final mask (annotated)")
                c.hide_blended_display_window()
                cv2.destroyAllWindows()
            except Exception:
                pass

    if switch_to:
        return {"switch_to": switch_to}
    return None