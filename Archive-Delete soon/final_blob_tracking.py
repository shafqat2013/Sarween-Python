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

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_W = 23
DEFAULT_GRID_H = 16
DEFAULT_WARP_W = 1280
DEFAULT_WARP_H = 720
DEFAULT_H = np.eye(3, dtype=np.float32)

DEST_PAD_PX = 8

# ──────────────────────────────────────────────────────────────────────────────
# Fog-of-war safe dynamic background (warped space)
# ──────────────────────────────────────────────────────────────────────────────
BG_ALPHA_SLOW = 0.02
BG_ALPHA_FAST = 0.15
FOG_CHANGE_RATIO = 0.08
WARP_MOTION_THRESH = 30

# ──────────────────────────────────────────────────────────────────────────────
# Performance tuning
# ──────────────────────────────────────────────────────────────────────────────
ARUCO_EVERY_N = 30
ARUCO_EVERY_N_FAST = 5
DRAW_ARUCO_OVERLAY = False  # toggle with 'a'

# ──────────────────────────────────────────────────────────────────────────────
# ArUco detector
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2.aruco as aruco
    ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    ARUCO_PARAMS = aruco.DetectorParameters()
    ARUCO_DET    = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
except Exception:
    ARUCO_DET = None

CORNER_IDS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
REQUIRED_IDS = set(CORNER_IDS.values())

# ──────────────────────────────────────────────────────────────────────────────
# State/tuning
# ──────────────────────────────────────────────────────────────────────────────
MAX_MATCH_DIST = 50
STALE_FOR      = 30
CONSENSUS_N    = 5
CONSENSUS_K    = 3

anon_tracks   = {}
anon_next_id  = 1
frame_idx     = 0

known_hist_buf  = {}
known_last_cell = {}

# recent known mini_ids (used to add views under an existing mini_id)
recent_known_ids = deque(maxlen=9)

# ──────────────────────────────────────────────────────────────────────────────
# ── UNIQUE NAME (BEST-FIT) LOGIC
#   Ensure only ONE detection per "name" (e.g. "red") displays that name.
#   Others keep match_id but have name cleared => renders as "#<mini_id>".
# ──────────────────────────────────────────────────────────────────────────────
NAME_OWNER_STALE = 120  # frames to keep a "name -> mini_id" preference alive
_name_owner = {}        # name_key -> match_id
_name_owner_last_seen = {}  # name_key -> frame_idx

def _norm_name(n: str) -> str:
    return (n or "").strip().lower()

def _safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _enforce_unique_names_best_fit(detections, cur_frame_idx):
    """
    Mutates detections in-place:
      - For each non-empty name among known detections:
          keep that name on exactly one "winner" detection,
          clear it on the rest.
      - Winner is chosen by:
          1) existing owner (if still present in this frame)
          2) otherwise max(score, area)
    """
    # purge stale owners
    stale = [k for k, last in _name_owner_last_seen.items()
             if (cur_frame_idx - last) > NAME_OWNER_STALE]
    for k in stale:
        _name_owner.pop(k, None)
        _name_owner_last_seen.pop(k, None)

    # group candidates by name_key
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
            # single usage => remember owner for stability
            mid = group[0].get("match_id")
            if mid:
                _name_owner[name_key] = mid
                _name_owner_last_seen[name_key] = cur_frame_idx
            continue

        # prefer existing owner if present in this frame
        preferred_mid = _name_owner.get(name_key)
        winner = None
        if preferred_mid is not None:
            for d in group:
                if d.get("match_id") == preferred_mid:
                    winner = d
                    break

        # otherwise choose best by (score, area)
        if winner is None:
            def rank(d):
                sc = _safe_float(d.get("score"))
                ar = _safe_float(d.get("area"))
                # None-safe: treat missing score as -inf, missing area as 0
                return (sc if sc is not None else -1e9, ar if ar is not None else 0.0)
            winner = max(group, key=rank)

        # keep name only on winner; clear on others
        keep_name = (winner.get("name") or "").strip()
        for d in group:
            if d is winner:
                d["name"] = keep_name
            else:
                d["name"] = ""  # clears label while preserving match_id

        # update owner
        win_mid = winner.get("match_id")
        if win_mid:
            _name_owner[name_key] = win_mid
            _name_owner_last_seen[name_key] = cur_frame_idx

# ──────────────────────────────────────────────────────────────────────────────
# Camera undistortion (lazy)
# ──────────────────────────────────────────────────────────────────────────────
_camera_params_cache = None
def get_camera_params():
    global _camera_params_cache
    if _camera_params_cache is None:
        if os.path.exists("camera_matrix.npy") and os.path.exists("dist_coeffs.npy"):
            _camera_params_cache = (
                np.load("camera_matrix.npy"),
                np.load("dist_coeffs.npy"),
            )
        else:
            _camera_params_cache = (None, None)
    return _camera_params_cache

# ──────────────────────────────────────────────────────────────────────────────
# Warped helpers
# ──────────────────────────────────────────────────────────────────────────────
def _warp_gray_blur(frame_bgr, H_use, warp_w, warp_h):
    warped = cv2.warpPerspective(frame_bgr, H_use, (int(warp_w), int(warp_h)))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur

def _update_bg_ema(bg_f32, cur_u8, update_mask_u8, alpha):
    if bg_f32 is None:
        return cur_u8.astype(np.float32)
    m = (update_mask_u8 > 0)
    bg_f32[m] = (1.0 - alpha) * bg_f32[m] + alpha * cur_u8[m].astype(np.float32)
    return bg_f32

def _make_mask_debug_view(motion_warp, motion_cam, shadowfree_cam, final_mask_cam):
    """
    Build a 2x2 debug visualization of all major masks.
    """
    def to_bgr(m):
        if m is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        if len(m.shape) == 2:
            return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        return m

    mw = to_bgr(motion_warp)
    mc = to_bgr(motion_cam)
    sf = to_bgr(shadowfree_cam)
    fm = to_bgr(final_mask_cam)

    h = min(mw.shape[0], mc.shape[0], sf.shape[0], fm.shape[0])
    w = min(mw.shape[1], mc.shape[1], sf.shape[1], fm.shape[1])

    mw = cv2.resize(mw, (w, h))
    mc = cv2.resize(mc, (w, h))
    sf = cv2.resize(sf, (w, h))
    fm = cv2.resize(fm, (w, h))

    top = np.hstack([mw, mc])
    bot = np.hstack([sf, fm])
    grid = np.vstack([top, bot])

    labels = [
        ("motion_warp", 10, 20),
        ("motion_cam", w + 10, 20),
        ("shadowfree", 10, h + 20),
        ("final_mask", w + 10, h + 20),
    ]
    for text, x, y in labels:
        cv2.putText(grid, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(grid, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    return grid

def _annotate_mask_blobs(mask_u8, min_area, min_fill, min_solidity, max_k=12):
    """
    Returns a BGR image of the mask with blob boxes + per-blob stats text.
    Stats:
      - area, fill (area/(w*h)), solidity (area/hull_area)
      - pass/fail vs thresholds
    """
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
        pass_sol  = solidity >= float(min_solidity)

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

        cv2.putText(bgr, stats, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(bgr, stats, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

        cv2.putText(bgr, flags, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(bgr, flags, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220,220,220), 1, cv2.LINE_AA)

    return bgr

# ──────────────────────────────────────────────────────────────────────────────
# ArUco + homography helpers
# ──────────────────────────────────────────────────────────────────────────────
def _detect_markers(cam_bgr):
    if ARUCO_DET is None:
        return None, None, None
    gray = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY)
    return ARUCO_DET.detectMarkers(gray)

def _corner_pt(id_to_c, marker_id, corner_idx):
    return id_to_c[marker_id][0][corner_idx].astype(np.float32)

def _solve_H_from_markers(cam_bgr, warp_w, warp_h, corner_ids):
    corners, ids, _ = _detect_markers(cam_bgr)
    if ids is None:
        return None, 0, [], None, None

    seen_ids = sorted([int(x[0]) for x in ids])
    detected_count = len(seen_ids)

    if detected_count < 4:
        return None, detected_count, seen_ids, corners, ids

    id_to_c = {int(ids[i][0]): corners[i] for i in range(len(ids))}
    try:
        tl = _corner_pt(id_to_c, corner_ids["TL"], 0)
        tr = _corner_pt(id_to_c, corner_ids["TR"], 1)
        br = _corner_pt(id_to_c, corner_ids["BR"], 2)
        bl = _corner_pt(id_to_c, corner_ids["BL"], 3)
    except KeyError:
        return None, detected_count, seen_ids, corners, ids

    pts_src = np.array([tl, tr, br, bl], dtype=np.float32)
    pts_dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    H_view, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    return (H_view.astype(np.float32) if H_view is not None else None), detected_count, seen_ids, corners, ids

def _roi_masks(cam_bgr, H_view, warp_w, warp_h, corner_ids):
    corners, ids, _ = _detect_markers(cam_bgr)
    if ids is None or H_view is None:
        return None, None
    id_to_c = {int(ids[i][0]): corners[i] for i in range(len(ids))}
    try:
        tl = _corner_pt(id_to_c, corner_ids["TL"], 0)
        tr = _corner_pt(id_to_c, corner_ids["TR"], 1)
        br = _corner_pt(id_to_c, corner_ids["BR"], 2)
        bl = _corner_pt(id_to_c, corner_ids["BL"], 3)
    except KeyError:
        return None, None

    poly_cam = np.array([tl, tr, br, bl], dtype=np.float32)
    h, w = cam_bgr.shape[:2]
    mask_cam = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_cam, poly_cam.astype(np.int32), 255)
    mask_warp = cv2.warpPerspective(mask_cam, H_view, (warp_w, warp_h), flags=cv2.INTER_NEAREST)
    return mask_cam, mask_warp

# ──────────────────────────────────────────────────────────────────────────────
# Homography debug view (green grid) - toggle with 'h'
# ──────────────────────────────────────────────────────────────────────────────
def _show_homography_view(cam_frame, H_view, warp_w, warp_h, grid_w, grid_h):
    count = 0
    if ARUCO_DET is not None:
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DET.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            count = len(ids)

    if H_view is not None:
        pad = int(DEST_PAD_PX)
        canvas_w = int(warp_w + 2 * pad)
        canvas_h = int(warp_h + 2 * pad)

        T = np.array([[1, 0, pad],
                      [0, 1, pad],
                      [0, 0, 1]], dtype=np.float32)
        H_pad = T @ H_view

        warped_dbg = cv2.warpPerspective(cam_frame, H_pad, (canvas_w, canvas_h))

        xs = np.linspace(0, warp_w, grid_w + 1)
        ys = np.linspace(0, warp_h, grid_h + 1)

        xs_i = np.unique(np.round(xs + pad).astype(int))
        ys_i = np.unique(np.round(ys + pad).astype(int))

        thick = 2
        lt = cv2.LINE_AA

        for xg in xs_i[1:-1]:
            cv2.line(warped_dbg, (int(xg), pad), (int(xg), pad + warp_h - 1), (0, 255, 0), thick, lt)
        for yg in ys_i[1:-1]:
            cv2.line(warped_dbg, (pad, int(yg)), (pad + warp_w - 1, int(yg)), (0, 255, 0), thick, lt)

        cv2.rectangle(warped_dbg, (pad, pad), (pad + warp_w - 1, pad + warp_h - 1), (0, 255, 0), thick, lt)
    else:
        warped_dbg = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)

    cv2.putText(warped_dbg, f"ArUco markers: {count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(warped_dbg, f"ArUco markers: {count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("Homography view (debug)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography view (debug)", 1280, 720)
    cv2.imshow("Homography view (debug)", warped_dbg)

# ──────────────────────────────────────────────────────────────────────────────
# Text input modal (OpenCV)
# ──────────────────────────────────────────────────────────────────────────────
def _prompt_text(title, prompt, initial=""):
    """
    Returns a string (possibly empty) on Enter, or None on Esc.
    """
    text = (initial or "")
    W, H = 900, 220

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, W, H)

    while True:
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        cv2.putText(canvas, prompt, (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.putText(canvas, "Enter = confirm   Esc = cancel", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

        # input box
        cv2.rectangle(canvas, (20, 140), (W-20, 190), (80,80,80), 2)
        shown = text
        cv2.putText(canvas, shown, (30, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(title, canvas)
        k = cv2.waitKey(0) & 0xFF

        if k in (27,):  # Esc
            cv2.destroyWindow(title)
            return None
        if k in (10, 13):  # Enter
            cv2.destroyWindow(title)
            return text.strip()

        if k in (8, 127):  # Backspace (varies by platform)
            text = text[:-1]
            continue

        # Accept printable ASCII
        if 32 <= k <= 126:
            text += chr(k)

# ──────────────────────────────────────────────────────────────────────────────
# Capture Picker (freeze + choose among candidates)
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
    alpha = 0.35

    for _, c_ in candidates:
        cv2.drawContours(overlay, [c_.astype(np.int32)], -1, (255, 255, 0), thickness=-1)
    show = cv2.addWeighted(overlay, alpha, show, 1.0 - alpha, 0)

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
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
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

def _choose_mini_id_for_capture():
    """
    Modal chooser:
      - 'n' => create new mini_id
      - '1'..'9' => add view to recent known mini_id
      - Esc/q => cancel
    Returns: "NEW", mini_id string, or None
    """
    ids = list(recent_known_ids)

    canvas = np.zeros((260, 900, 3), dtype=np.uint8)

    def draw(y, txt, scale=0.6):
        cv2.putText(canvas, txt, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2, cv2.LINE_AA)

    draw(35,  "ASSIGN CAPTURE:", 0.75)
    draw(75,  "n = new mini (new mini_id)")
    draw(105, "1..9 = add as a new view to a recent known mini_id")
    draw(135, "Esc/q = cancel")

    if ids:
        draw(175, "Recent known mini_ids:", 0.6)
        y = 205
        for i, mid in enumerate(ids[:9], start=1):
            draw(y, f"{i}. {mid}", 0.55)
            y += 18
    else:
        draw(175, "(No known minis seen yet — press 'n')", 0.6)

    win = "Assign Capture"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 900, 260)
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
                return ids[idx]

# ──────────────────────────────────────────────────────────────────────────────
# Grid cell selection from contour (warp space)
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
# Detection consolidation + movement reporting
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

def _maybe_report_movement(mini_id, base_cell, on_mini_moved=None):
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
            print(f"{mini_id}, {last}, {most}")
            known_last_cell[mini_id] = most
            if on_mini_moved is not None:
                on_mini_moved(mini_id, most)

# ──────────────────────────────────────────────────────────────────────────────
# Main session
# ──────────────────────────────────────────────────────────────────────────────
def begin_session(on_mini_moved, camera_index=None,
                  known_threshold=0.50,
                  min_fill=0.35,
                  min_solidity=0.75,
                  show_windows=True,
                  use_motion_gating=True,
                  motion_thresh=30):
    global frame_idx
    global DRAW_ARUCO_OVERLAY

    if camera_index is None:
        sel = s.load_last_selection() or {}
        camera_index = sel.get("webcam_index", 0)

    warp_w = DEFAULT_WARP_W
    warp_h = DEFAULT_WARP_H
    grid_w = DEFAULT_GRID_W
    grid_h = DEFAULT_GRID_H
    try:
        if hasattr(s, "warp_w") and s.warp_w is not None:
            warp_w = int(s.warp_w)
        if hasattr(s, "warp_h") and s.warp_h is not None:
            warp_h = int(s.warp_h)
        if hasattr(s, "grid_cols") and s.grid_cols is not None:
            grid_w = int(s.grid_cols)
        if hasattr(s, "grid_rows") and s.grid_rows is not None:
            grid_h = int(s.grid_rows)
    except Exception:
        pass

    print(f"[INFO] warp={warp_w}x{warp_h}  camera_index={camera_index}", flush=True)
    fo.set_grid_params(warp_w, warp_h, grid_w, grid_h)

    cam_mtx, cam_dist = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return
    for _ in range(8):
        cap.read()

    BG_cam = mt.capture_background_full(camera_index)
    if BG_cam is None:
        print("❌ Failed to capture camera-space background.")
        return

    H_saved = DEFAULT_H.copy()
    last_mask_cam = None
    last_mask_warp = None

    last_marker_count = 0
    last_seen_ids = []
    last_missing_ids = sorted(list(REQUIRED_IDS))
    lock_lost_reason = ""

    last_aruco_corners = None
    last_aruco_ids = None

    ok_init, cam_init = cap.read()
    if ok_init and cam_init is not None:
        if cam_mtx is not None and cam_dist is not None:
            cam_init = cv2.undistort(cam_init, cam_mtx, cam_dist)
        bg_init = _warp_gray_blur(cam_init, H_saved, warp_w, warp_h)
        BG_warp_f32 = bg_init.astype(np.float32)
    else:
        BG_warp_f32 = np.zeros((int(warp_h), int(warp_w)), dtype=np.float32)

    show_h_view = False
    homography_window_open = False

    show_blended_view = True
    show_cam_view     = True
    show_diff_view    = False

    cam_window_open   = False
    diff_window_open  = False
    ident_window_open = False  # Identify Debug window state

    pending_close_cam = False

    if show_windows:
        if show_blended_view:
            c.show_blended_display_window()
        else:
            c.hide_blended_display_window()

    fps_count = 0
    last_print = time.perf_counter()

    def _fmt2(v):
        return "--" if v is None else f"{float(v):.2f}"

    while True:
        frame_idx += 1

        for tid in list(anon_tracks.keys()):
            if frame_idx - anon_tracks[tid]["last_seen"] > STALE_FOR:
                del anon_tracks[tid]

        ok, cam = cap.read()
        if not ok or cam is None:
            print("❌ Webcam read failed.")
            break

        if cam_mtx is not None and cam_dist is not None:
            cam = cv2.undistort(cam, cam_mtx, cam_dist)

        have_lock = (last_mask_cam is not None) and (last_mask_warp is not None)
        need_fast = (not have_lock) or (last_marker_count < 4)
        aruco_interval = ARUCO_EVERY_N_FAST if need_fast else ARUCO_EVERY_N

        H_view = None
        detected_count = 0
        seen_ids = []
        det_corners = None
        det_ids = None

        if (frame_idx % aruco_interval) == 0:
            H_view, detected_count, seen_ids, det_corners, det_ids = _solve_H_from_markers(
                cam, warp_w, warp_h, CORNER_IDS
            )

            if det_ids is not None and det_corners is not None and len(det_ids) > 0:
                last_aruco_ids = det_ids
                last_aruco_corners = det_corners
            else:
                last_aruco_ids = None
                last_aruco_corners = None

            last_marker_count = int(detected_count)
            last_seen_ids = list(seen_ids)
            last_missing_ids = sorted(list(REQUIRED_IDS - set(last_seen_ids)))

            if H_view is not None and len(last_missing_ids) == 0:
                H_saved = H_view
                last_mask_cam, last_mask_warp = _roi_masks(cam, H_view, warp_w, warp_h, CORNER_IDS)
                lock_lost_reason = ""
            else:
                if have_lock and last_marker_count < 4:
                    lock_lost_reason = f"Lost markers ({last_marker_count}/4 visible), missing: {last_missing_ids}"
                    last_mask_cam = None
                    last_mask_warp = None

        H_use = H_saved
        mask_cam = last_mask_cam
        mask_warp = last_mask_warp

        # ── WAIT STATE (no lock) ───────────────────────────────────────────────
        if mask_cam is None or mask_warp is None:
            if show_windows and show_cam_view:
                vis_cam = cam.copy()

                if DRAW_ARUCO_OVERLAY and ARUCO_DET is not None:
                    if last_aruco_ids is not None and len(last_aruco_ids) > 0:
                        aruco.drawDetectedMarkers(vis_cam, last_aruco_corners, last_aruco_ids)

                msg = lock_lost_reason if lock_lost_reason else (
                    f"Waiting for ArUco lock ({last_marker_count}/4 visible), missing: {last_missing_ids}"
                )
                cv2.putText(vis_cam, msg, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

                cv2.namedWindow("Camera + Detections", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Camera + Detections", 1280, 720)
                cv2.imshow("Camera + Detections", vis_cam)
                cam_window_open = True
            else:
                if cam_window_open:
                    pending_close_cam = True

            if (not show_diff_view) and diff_window_open:
                try: cv2.destroyWindow("Mask Debug (2x2)")
                except Exception: pass
                diff_window_open = False

            # close Identify Debug window when waiting
            if ident_window_open:
                try: cv2.destroyWindow("Identify Debug")
                except Exception: pass
                ident_window_open = False

            key = cv2.waitKey(1) & 0xFF

            if key == ord('h'):
                show_h_view = not show_h_view
            elif key == ord('a'):
                DRAW_ARUCO_OVERLAY = not DRAW_ARUCO_OVERLAY
            elif key == ord('m'):
                show_blended_view = not show_blended_view
                if show_windows:
                    if show_blended_view: c.show_blended_display_window()
                    else: c.hide_blended_display_window()
            elif key == ord('v'):
                show_cam_view = not show_cam_view
                if not show_cam_view:
                    pending_close_cam = True
            elif key == ord('d'):
                show_diff_view = not show_diff_view
                if (not show_diff_view) and diff_window_open:
                    try: cv2.destroyWindow("Mask Debug (2x2)")
                    except Exception: pass
                    diff_window_open = False
            elif key == ord('q'):
                break

            if pending_close_cam and cam_window_open:
                try:
                    cv2.waitKey(1)
                    cv2.destroyWindow("Camera + Detections")
                except Exception:
                    pass
                cam_window_open = False
                pending_close_cam = False

            continue

        # ──────────────────────────────────────────────────────────────────────
        # Normal pipeline
        warp_blur = _warp_gray_blur(cam, H_use, warp_w, warp_h)

        bg_u8 = np.clip(BG_warp_f32, 0, 255).astype(np.uint8)
        diff_warp = cv2.absdiff(bg_u8, warp_blur)
        _, motion_warp = cv2.threshold(diff_warp, WARP_MOTION_THRESH, 255, cv2.THRESH_BINARY)
        motion_warp = cv2.erode(motion_warp, None, iterations=1)
        motion_warp = cv2.dilate(motion_warp, None, iterations=1)
        motion_warp = cv2.bitwise_and(motion_warp, mask_warp)

        roi_area = float(cv2.countNonZero(mask_warp))
        changed = float(cv2.countNonZero(motion_warp))
        change_ratio = (changed / max(1.0, roi_area))

        alpha_bg = BG_ALPHA_FAST if change_ratio >= FOG_CHANGE_RATIO else BG_ALPHA_SLOW
        update_mask = cv2.bitwise_not(motion_warp)
        update_mask = cv2.bitwise_and(update_mask, mask_warp)
        BG_warp_f32 = _update_bg_ema(BG_warp_f32, warp_blur, update_mask, alpha_bg)

        shadowfree_cam = mt.shadow_free_mask(BG_cam["bgr"], cam)

        try:
            H_inv = np.linalg.inv(H_use) if H_use is not None else np.eye(3, dtype=np.float32)
        except Exception:
            H_inv = np.eye(3, dtype=np.float32)

        motion_cam = cv2.warpPerspective(motion_warp, H_inv, (cam.shape[1], cam.shape[0]), flags=cv2.INTER_NEAREST)

        final_mask_cam = mt.combine_masks_componentwise(
            motion_cam, shadowfree_cam,
            keep_ratio=0.25,
            min_comp_area=150
        )
        final_mask_cam = cv2.bitwise_and(final_mask_cam, mask_cam)

        # Dynamic thresholds
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

        # ── UNIQUE NAME (BEST-FIT) APPLICATION ────────────────────────────────
        _enforce_unique_names_best_fit(detections, frame_idx)

        # Update recent known minis (for add-view capture)
        for d in detections:
            if d.get("label") == "known" and d.get("match_id"):
                mid = d["match_id"]
                if mid in recent_known_ids:
                    recent_known_ids.remove(mid)
                recent_known_ids.appendleft(mid)

        vis_cam = cam.copy()

        if DRAW_ARUCO_OVERLAY and ARUCO_DET is not None:
            if last_aruco_ids is not None and len(last_aruco_ids) > 0:
                aruco.drawDetectedMarkers(vis_cam, last_aruco_corners, last_aruco_ids)

        def _fmt2(v):
            return "--" if v is None else f"{float(v):.2f}"

        # Draw boxes + score breakdown
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

            # color + headline
            if label == "known" and match_id:
                color = (0, 200, 0)
                display = (name + " " if name else "") + f"#{str(match_id)}"
            else:
                color = (0, 165, 255) if eligible else (0, 0, 255)
                display = "unknown"

            if score_val is None:
                headline = f"{display}  A={int(area)}"
            else:
                headline = f"{display}  A={int(area)}  S={score_val:.3f}"

            # component breakdown line
            comp = f"H={_fmt2(h_sim)}  O={_fmt2(o_sim)}  Sh={_fmt2(s_sim)}"

            cv2.rectangle(vis_cam, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)

            # two lines above bbox
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
                    _maybe_report_movement(match_id, base_cell, on_mini_moved)

        # ── Window rendering (toggled) ─────────────────────────────────────────
        if show_windows and show_cam_view:
            cv2.namedWindow("Camera + Detections", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera + Detections", 1280, 720)
            cv2.imshow("Camera + Detections", vis_cam)
            cam_window_open = True
        else:
            if cam_window_open:
                pending_close_cam = True

        # Identify Debug window
        if show_windows and ("annotated" in ident) and (ident["annotated"] is not None):
            cv2.namedWindow("Identify Debug", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Identify Debug", 1280, 720)
            cv2.imshow("Identify Debug", ident["annotated"])
            ident_window_open = True
        else:
            if ident_window_open:
                try: cv2.destroyWindow("Identify Debug")
                except Exception: pass
                ident_window_open = False

        if show_windows and show_h_view:
            _show_homography_view(cam, H_use, warp_w, warp_h, grid_w, grid_h)
            homography_window_open = True
        else:
            if homography_window_open:
                try: cv2.destroyWindow("Homography view (debug)")
                except Exception: pass
                homography_window_open = False

        if show_windows and show_diff_view:
            final_mask_annot = _annotate_mask_blobs(
                final_mask_cam,
                min_area=int(min_area_for_ident),
                min_fill=min_fill,
                min_solidity=min_solidity,
                max_k=12
            )
            debug_grid = _make_mask_debug_view(motion_warp, motion_cam, shadowfree_cam, final_mask_annot)

            cv2.namedWindow("Mask Debug (2x2)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Mask Debug (2x2)", 1280, 720)
            cv2.imshow("Mask Debug (2x2)", debug_grid)
            diff_window_open = True
        else:
            if diff_window_open:
                try: cv2.destroyWindow("Mask Debug (2x2)")
                except Exception: pass
                diff_window_open = False

        # Keep blended window in sync
        if show_windows:
            if show_blended_view:
                if not c.is_blended_display_window_open():
                    c.show_blended_display_window()
            else:
                if c.is_blended_display_window_open():
                    c.hide_blended_display_window()

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            BG_cam = mt.capture_background_full(camera_index)
            if BG_cam is None:
                print("❌ Failed to recapture camera-space background.")
            bg_now = _warp_gray_blur(cam, H_use, warp_w, warp_h)
            BG_warp_f32 = bg_now.astype(np.float32)

        elif key == ord('h'):
            show_h_view = not show_h_view

        elif key == ord('a'):
            DRAW_ARUCO_OVERLAY = not DRAW_ARUCO_OVERLAY

        elif key == ord('m'):
            show_blended_view = not show_blended_view
            if show_windows:
                if show_blended_view:
                    c.show_blended_display_window()
                else:
                    c.hide_blended_display_window()

        elif key == ord('v'):
            show_cam_view = not show_cam_view
            if not show_cam_view:
                pending_close_cam = True

        elif key == ord('d'):
            show_diff_view = not show_diff_view
            if (not show_diff_view) and diff_window_open:
                try: cv2.destroyWindow("Mask Debug (2x2)")
                except Exception: pass
                diff_window_open = False

        elif key == ord('c'):
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
                        name = ""  # cancelled -> just save with no name
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

        elif key == ord('q'):
            break

        # Safe close camera window AFTER a UI tick (prevents freeze)
        if pending_close_cam and cam_window_open:
            try:
                cv2.waitKey(1)
                cv2.destroyWindow("Camera + Detections")
            except Exception:
                pass
            cam_window_open = False
            pending_close_cam = False

        # FPS print (optional)
        fps_count += 1
        now = time.perf_counter()
        if now - last_print >= 1.0:
            fps = fps_count / (now - last_print)
            print(f"[CV] FPS: {fps:.1f}", flush=True)
            fps_count = 0
            last_print = now

    cap.release()
    if show_windows:
        try:
            if cam_window_open:
                try:
                    cv2.waitKey(1)
                    cv2.destroyWindow("Camera + Detections")
                except Exception:
                    pass
            if diff_window_open:
                cv2.destroyWindow("Mask Debug (2x2)")
            if homography_window_open:
                cv2.destroyWindow("Homography view (debug)")
            if ident_window_open:
                cv2.destroyWindow("Identify Debug")
            c.hide_blended_display_window()
            cv2.destroyAllWindows()
        except Exception:
            pass
