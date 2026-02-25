import os
import cv2
import math
import numpy as np
from collections import deque, Counter

import setup as s
import mini_tracking as mt  # identify_minis, capture_mini, DB path
import foundryoutput as fo

# ──────────────────────────────────────────────────────────────────────────────
# Defaults (used only if setup/calibration didn't populate session values)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_W = 23
DEFAULT_GRID_H = 16
DEFAULT_WARP_W = 1920
DEFAULT_WARP_H = 1080
DEFAULT_H = np.eye(3, dtype=np.float32)  # identity fallback

# Padding (pixels) shown ONLY in the Homography debug view canvas.
# This increases the debug canvas size to reveal the ArUco white "quiet zone".
DEST_PAD_PX = 8

# ──────────────────────────────────────────────────────────────────────────────
# ArUco (match calibration: DICT_6X6_250) + detector
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2.aruco as aruco
    ARUCO_DICT  = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    ARUCO_PARAMS = aruco.DetectorParameters()
    ARUCO_DET    = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
except Exception:
    ARUCO_DET = None  # if OpenCV lacks aruco module

# Corner ID mapping (top-left, top-right, bottom-right, bottom-left)
CORNER_IDS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}

# ──────────────────────────────────────────────────────────────────────────────
# Tuning / state
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
# Warped background (for optional motion gating)
# ──────────────────────────────────────────────────────────────────────────────
def _capture_warped_background(cap, H_use, warp_w, warp_h, cam_mtx, cam_dist):
    """
    Grab one frame from the already-open camera, optionally undistort,
    warp it with H_use to canvas size, and return a blurred grayscale
    background in warped space for motion gating.

    Returns: {"warp_blur": <uint8 HxW>} or None on failure.
    """
    # warm-up reads (helps AVFoundation on macOS)
    for _ in range(8):
        cap.read()

    ok, frame = cap.read()
    if not ok or frame is None:
        return None

    if cam_mtx is not None and cam_dist is not None:
        frame = cv2.undistort(frame, cam_mtx, cam_dist)

    warped = cv2.warpPerspective(frame, H_use, (int(warp_w), int(warp_h)))
    blur   = cv2.GaussianBlur(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    return {"warp_blur": blur}

# ──────────────────────────────────────────────────────────────────────────────
# ArUco helpers: per-frame homography solve + ROI mask
# ──────────────────────────────────────────────────────────────────────────────
def _detect_markers(cam_bgr):
    if ARUCO_DET is None:
        return None, None, None
    gray = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY)
    return ARUCO_DET.detectMarkers(gray)

# Use true outer marker corners (not centroids)
def _corner_pt(id_to_c, marker_id, corner_idx):
    """
    corners array is shape (1, 4, 2) with order: TL=0, TR=1, BR=2, BL=3.
    Return the requested corner as float32.
    """
    return id_to_c[marker_id][0][corner_idx].astype(np.float32)

def _solve_H_from_markers(cam_bgr, warp_w, warp_h, corner_ids):
    corners, ids, _ = _detect_markers(cam_bgr)
    if ids is None or len(ids) < 4:
        return None, 0, None
    id_to_c = {int(ids[i][0]): corners[i] for i in range(len(ids))}
    try:
        tl = _corner_pt(id_to_c, corner_ids["TL"], 0)  # TL marker's TL corner
        tr = _corner_pt(id_to_c, corner_ids["TR"], 1)  # TR marker's TR corner
        br = _corner_pt(id_to_c, corner_ids["BR"], 2)  # BR marker's BR corner
        bl = _corner_pt(id_to_c, corner_ids["BL"], 3)  # BL marker's BL corner
    except KeyError:
        return None, len(ids), None

    pts_src = np.array([tl, tr, br, bl], dtype=np.float32)
    # No padding here: operational homography keeps your exact canvas mapping
    pts_dst = np.array([[0,0],[warp_w-1,0],[warp_w-1,warp_h-1],[0,warp_h-1]], dtype=np.float32)
    H_view, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    return (H_view.astype(np.float32) if H_view is not None else None), 4, (pts_src, pts_dst)

def _roi_masks(cam_bgr, H_view, warp_w, warp_h, corner_ids):
    """
    Returns (mask_cam, mask_warp). Both are binary uint8 masks.
    """
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
# Homography debug view (warped + green grid ONLY)
# ──────────────────────────────────────────────────────────────────────────────
def _show_homography_view(cam_frame, H_view, warp_w, warp_h, grid_w, grid_h):
    """
    Debug view: ONLY the warped homography canvas with the green grid overlaid.
    No side-by-side composite, no ArUco overlay.
    """
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

        # Draw grid lines aligned to the ORIGINAL warp rect, offset by pad
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

        # outline original warp rectangle
        cv2.rectangle(warped_dbg, (pad, pad), (pad + warp_w - 1, pad + warp_h - 1), (0, 255, 0), thick, lt)
    else:
        warped_dbg = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)

    # Overlay marker count on the warped debug canvas
    cv2.putText(warped_dbg, f"ArUco markers: {count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(warped_dbg, f"ArUco markers: {count}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.namedWindow("Homography view (debug)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Homography view (debug)", 1280, 720)
    cv2.imshow("Homography view (debug)", warped_dbg)

# ──────────────────────────────────────────────────────────────────────────────
# Geometry / detection consolidation
# ──────────────────────────────────────────────────────────────────────────────
def pick_base_cell_from_contour(contour_warped, grid_w, grid_h, warp_w, warp_h,
                                min_coverage=0.6):
    """
    Return 'rXcY' for the cell with max coverage by the warped contour.

    Biases toward the BOTTOM of the contour:
      1. Find all cells with coverage >= min_coverage.
      2. Among those, pick the cell with the largest row index (visually lowest).
      3. If multiple in that row, pick the one with the highest coverage.
      4. If no cell meets min_coverage, fall back to the single best coverage
         anywhere, but still require min_coverage overall.
    """
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

    covered_cells = []              # (row, col, coverage)
    best_any = (None, None, 0.0)    # (row, col, cov)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            xA, xB = int(c * cell_w), int((c + 1) * cell_w)
            yA, yB = int(r * cell_h), int((r + 1) * cell_h)
            roi = mask[yA:yB, xA:xB]
            if roi.size == 0:
                continue
            cov = cv2.countNonZero(roi) / float(roi.size)

            # Track global best (for fallback)
            if cov > best_any[2]:
                best_any = (r, c, cov)

            # Track only "good enough" cells for bottom-bias logic
            if cov >= min_coverage:
                covered_cells.append((r, c, cov))

    # No cell meets min_coverage at all → fallback to best_any if it meets threshold
    if not covered_cells:
        r_best, c_best, cov_best = best_any
        if r_best is not None and cov_best >= min_coverage:
            return f"r{r_best}c{c_best}"
        return None

    # Among covered cells, pick the largest row index (bottom-most)
    max_row = max(r for (r, c, cov) in covered_cells)
    candidates = [entry for entry in covered_cells if entry[0] == max_row]

    # Among those, pick the highest coverage; if tie, pick smallest col (left-most)
    candidates.sort(key=lambda t: (-t[2], t[1]))  # coverage desc, col asc
    r_final, c_final, _ = candidates[0]
    return f"r{r_final}c{c_final}"

def _merge_frame_detections(dets, max_dist=MAX_MATCH_DIST):
    """Largest area per known ID; merge unknowns by centroid proximity."""
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

def _assign_stable_id_for_unknown(centroid_cam):
    global anon_next_id
    cx, cy = centroid_cam
    best_d, best_id = 1e9, None
    for tid, tr in list(anon_tracks.items()):
        px, py = tr["centroid"]
        d = math.hypot(cx - px, cy - py)
        if d < best_d and d <= MAX_MATCH_DIST:
            best_d, best_id = d, tid
    if best_id is None:
        best_id = f"u{anon_next_id}"
        anon_next_id += 1
    anon_tracks[best_id] = {"centroid": (cx, cy), "last_seen": frame_idx}
    return best_id

def _maybe_report_movement(mini_id, base_cell, on_mini_moved=None):
    """Debounce known mini cell; print only when it changes. Optionally callback."""
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
# Main session: detect in CAMERA space; warp contours for grid math
# ──────────────────────────────────────────────────────────────────────────────
def begin_session(on_mini_moved, camera_index=None,
                  known_threshold=0.55,
                  min_fill=0.35,
                  min_solidity=0.75,
                  show_windows=True,
                  use_motion_gating=True,
                  motion_thresh=30):
    """
    - Per-frame ArUco homography (H_view) from 4 corners (if visible)
    - Build ROI masks; detect minis in CAMERA space (matches DB immediately after 'c')
    - Warp contours with H_view/H_saved to canvas for grid cell computation
    - Keys: c (capture), r (reset BG), h (homography view), q (quit)
    - Prints only moves for KNOWN minis: 'mini_id, old location, new location'
    """
    global frame_idx

    if camera_index is None:
        sel = s.load_last_selection() or {}
        camera_index = sel.get("webcam_index", 0)

    # Initialize canvas/grid from DEFAULTS, then override from setup/calibration session values
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

    # Let the Foundry output module know our warp + grid
    fo.set_grid_params(warp_w, warp_h, grid_w, grid_h)

    cam_mtx, cam_dist = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return
    for _ in range(8):
        cap.read()

    # Camera-space background for component-wise prefiltering (BGR + BLUR)
    BG_cam = mt.capture_background_full(camera_index)
    if BG_cam is None:
        print("❌ Failed to capture camera-space background.")
        return

    # Background for motion gating in WARPED space (using saved H initially)
    H_saved = DEFAULT_H.copy()
    BG_warp = None
    if use_motion_gating:
        BG_warp = _capture_warped_background(cap, H_saved, warp_w, warp_h, cam_mtx, cam_dist)

    show_h_view = False
    post_capture_grace = 0  # frames to skip motion-gating after 'c'

    while True:
        frame_idx += 1

        # prune stale unknown tracks
        for tid in list(anon_tracks.keys()):
            if frame_idx - anon_tracks[tid]["last_seen"] > STALE_FOR:
                del anon_tracks[tid]

        ok, cam = cap.read()
        if not ok or cam is None:
            print("❌ Webcam read failed.")
            break

        if cam_mtx is not None and cam_dist is not None:
            cam = cv2.undistort(cam, cam_mtx, cam_dist)

        # Solve live homography from markers (if 4 visible)
        H_view, n_used, pts_pair = _solve_H_from_markers(cam, warp_w, warp_h, CORNER_IDS)

        # If we got a good 4-corner homography, update the cached H_saved
        if H_view is not None and n_used == 4:
            H_saved = H_view

        # Always use the last good H (or identity if we've never seen 4 corners yet)
        H_use = H_saved


        # Build ROI masks (camera + warped)
        mask_cam, mask_warp = (None, None)
        if H_view is not None and n_used == 4:
            mask_cam, mask_warp = _roi_masks(cam, H_view, warp_w, warp_h, CORNER_IDS)

        # ── Build final camera-space mask via helper in mini_tracking ─────────
        final_mask_cam = mt.build_final_mask_for_frame(
            live_frame_bgr=cam,
            BG=BG_cam,
            keep_ratio=0.25,
            min_comp_area=150,
            motion_thresh=30
        )

        # Optionally gate by inside-quad ROI if available
        if mask_cam is not None:
            final_mask_cam = cv2.bitwise_and(final_mask_cam, mask_cam)

        # ── Grid-scale dynamic size thresholds (in CAMERA pixels)
        # Approximate area of the warped canvas in camera space:
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
        cell_area_cam  = area_scale * cell_area_warp  # ≈ area of 1 grid square in CAMERA px

        # Dynamic thresholds
        min_area_dyn = 0.64 * cell_area_cam    # ~0.8 x 0.8 of a cell
        max_area_dyn = 4.00 * cell_area_cam    # ~2 x 2 cells
        # Lower floor for ID to allow red visualization of smaller blobs
        min_area_for_ident = max(100, int(0.10 * cell_area_cam))

        # Detect MINIS in CAMERA space
        ident = mt.identify_minis(
            cam,
            diff_mask=final_mask_cam,           # component-wise filtered camera-space mask
            db_entries=None,
            db_csv_path=mt.DB_CSV,
            min_area=int(min_area_for_ident),
            min_fill=min_fill,
            min_solidity=min_solidity,
            known_threshold=known_threshold,
            draw=False  # we'll draw our own consistent 3-color overlay
        )

        raw_dets = ident.get("detections", [])
        detections = _merge_frame_detections(raw_dets, max_dist=MAX_MATCH_DIST)

        # Prepare main view and draw 3-color system
        vis_cam = cam.copy()

        # ArUco overlays in MAIN view
        if ARUCO_DET is not None:
            gray_aru = cv2.cvtColor(vis_cam, cv2.COLOR_BGR2GRAY)
            corners_aru, ids_aru, _ = ARUCO_DET.detectMarkers(gray_aru)
            if ids_aru is not None and len(ids_aru) > 0:
                aruco.drawDetectedMarkers(vis_cam, corners_aru, ids_aru)

        for d in detections:
            x, y, w, h = d["bbox"]
            area = float(d.get("area", 0.0))
            label = d.get("label", "unknown")
            match_id = d.get("match_id")
            color = (0, 165, 255)  # default orange for eligible-unknown; may override below
            tag = None

            # Eligibility by dynamic thresholds
            eligible = (area >= min_area_dyn) and (area <= max_area_dyn)

            if label == "known" and match_id:
                color = (0, 200, 0)  # green
                tag = f"known:{str(match_id)[-6:]}  A={int(area)}"
            else:
                if eligible:
                    color = (0, 165, 255)  # orange (unknown but eligible)
                    tag = f"unknown  A={int(area)}"
                else:
                    color = (0, 0, 255)    # red (ineligible)
                    if area < min_area_dyn:
                        tag = f"too small  A={int(area)}  min~{int(min_area_dyn)}"
                    else:
                        tag = f"too big    A={int(area)}  max~{int(max_area_dyn)}"

            cv2.rectangle(vis_cam, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            if tag:
                y1 = max(0, y - 6)
                cv2.putText(vis_cam, tag, (int(x), y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
                cv2.putText(vis_cam, tag, (int(x), y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

            # For grid reporting: warp contour and compute base cell
            cnt_cam = d.get("contour")
            if cnt_cam is not None and H_use is not None:
                pts = cnt_cam.reshape(-1,1,2).astype(np.float32)
                cnt_warp = cv2.perspectiveTransform(pts, H_use).astype(np.float32)
                base_cell = pick_base_cell_from_contour(
                    cnt_warp, grid_w, grid_h, int(warp_w), int(warp_h), 0.60
                )
                if label == "known" and match_id:
                    _maybe_report_movement(match_id, base_cell, on_mini_moved)

        # UI + keys
        if show_windows:
            cv2.namedWindow("Camera + Detections", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Camera + Detections", 1280, 720)
            cv2.imshow("Camera + Detections", vis_cam)

            if show_h_view:
                _show_homography_view(cam, H_view, warp_w, warp_h, grid_w, grid_h)
            else:
                cv2.destroyWindow("Homography view (debug)")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                # reset camera-space background for prefiltering
                BG_cam = mt.capture_background_full(camera_index)
                if BG_cam is None:
                    print("❌ Failed to recapture camera-space background.")

                # reset warped background using current H_use (kept for completeness)
                if use_motion_gating:
                    BG_warp = _capture_warped_background(
                        cap, H_use if H_use is not None else H_saved,
                        warp_w, warp_h, cam_mtx, cam_dist
                    )

            elif key == ord('c'):
                # Safe capture to DB in CAMERA space (matches future frames)
                cap.release()
                try:
                    mt.capture_mini(
                        camera_index=camera_index,
                        background_blur=BG_cam["blur"],
                        background_bgr=BG_cam["bgr"]
                    )
                finally:
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        print("❌ Failed to reopen webcam after capture; exiting.")
                        break
                    for _ in range(8):
                        cap.read()
                    if use_motion_gating:
                        BG_warp = _capture_warped_background(
                            cap, H_use if H_use is not None else H_saved,
                            warp_w, warp_h, cam_mtx, cam_dist
                        )
                    post_capture_grace = 10  # unchanged

            elif key == ord('h'):
                show_h_view = not show_h_view

            elif key == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_windows:
        cv2.destroyAllWindows()
