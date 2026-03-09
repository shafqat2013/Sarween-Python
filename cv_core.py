# cv_core.py
#
# Shared CV pipeline core for Sarween.
# This module owns the parts that BOTH engines should share:
# - camera open + optional undistortion
# - ArUco detection + homography solve + ROI masks (mask_cam/mask_warp)
# - warp-space background EMA + motion mask
# - shadow-free mask + final combined mask (camera-space)
#
# It does NOT do:
# - mini DB matching / capture picker / name logic
# - band color detection (that belongs in band_tracking.py)
# - any control panel UI (engines can consume status/bundles and render)
#
# Typical usage (engine side):
#
#   import cv_core as core
#
#   sess = core.CVCoreSession(camera_index=..., warp_w=..., warp_h=..., grid_w=..., grid_h=...)
#   for bundle in sess.frames():
#       if not bundle["locked"]:
#           continue
#       # blob engine: use bundle["final_mask_cam"] + bundle["cam_bgr"]
#       # band engine: use bundle["warp_bgr"] (or warp cam using bundle["H_use"])
#
#   sess.close()

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, Tuple, List

import cv2
import numpy as np

import setup as s
import mini_tracking as mt

try:
    import foundryoutput as fo
except Exception:
    fo = None


# ──────────────────────────────────────────────────────────────────────────────
# Defaults / constants (copied from your tracking.py)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_GRID_W = 23
DEFAULT_GRID_H = 16
DEFAULT_WARP_W = 1280
DEFAULT_WARP_H = 720
DEFAULT_H = np.eye(3, dtype=np.float32)

DEST_PAD_PX = 8

BG_ALPHA_SLOW = 0.02
BG_ALPHA_FAST = 0.15
FOG_CHANGE_RATIO = 0.08
WARP_MOTION_THRESH = 30

ARUCO_EVERY_N = 30
ARUCO_EVERY_N_FAST = 5

LOCK_DROP_AFTER = 10  # consecutive ArUco misses before dropping lock


# ──────────────────────────────────────────────────────────────────────────────
# Shared window / display helpers
# ──────────────────────────────────────────────────────────────────────────────
_WINDOW_SIZED: set = set()

def ensure_window(name: str, w: int = 1280, h: int = 720) -> None:
    """Create a resizable window and size it once."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if name not in _WINDOW_SIZED:
        try:
            cv2.resizeWindow(name, int(w), int(h))
        except Exception:
            pass
        _WINDOW_SIZED.add(name)


def show_homography_view(
    cam_frame: np.ndarray,
    H_view: Optional[np.ndarray],
    warp_w: int,
    warp_h: int,
    grid_w: int,
    grid_h: int,
) -> None:
    """Render a warped+gridded debug view of the current homography."""
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

        T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float32)
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

    ensure_window("Homography view (debug)", 1280, 720)
    cv2.imshow("Homography view (debug)", warped_dbg)


# ──────────────────────────────────────────────────────────────────────────────
# ArUco detector
# ──────────────────────────────────────────────────────────────────────────────
try:
    import cv2.aruco as aruco  # type: ignore
    ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    ARUCO_PARAMS = aruco.DetectorParameters()
    ARUCO_DET = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
except Exception:
    ARUCO_DET = None

CORNER_IDS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
REQUIRED_IDS = set(CORNER_IDS.values())


# ──────────────────────────────────────────────────────────────────────────────
# Camera undistortion (lazy, copied)
# ──────────────────────────────────────────────────────────────────────────────
_camera_params_cache: Optional[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = None


def get_camera_params() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
# Warped helpers (copied)
# ──────────────────────────────────────────────────────────────────────────────
def warp_gray_blur(frame_bgr: np.ndarray, H_use: np.ndarray, warp_w: int, warp_h: int) -> np.ndarray:
    warped = cv2.warpPerspective(frame_bgr, H_use, (int(warp_w), int(warp_h)))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur


def update_bg_ema(bg_f32: Optional[np.ndarray], cur_u8: np.ndarray, update_mask_u8: np.ndarray, alpha: float) -> np.ndarray:
    if bg_f32 is None:
        return cur_u8.astype(np.float32)
    m = (update_mask_u8 > 0)
    bg_f32[m] = (1.0 - alpha) * bg_f32[m] + alpha * cur_u8[m].astype(np.float32)
    return bg_f32


# ──────────────────────────────────────────────────────────────────────────────
# ArUco + homography helpers (copied)
# ──────────────────────────────────────────────────────────────────────────────
def detect_markers(cam_bgr: np.ndarray):
    if ARUCO_DET is None:
        return None, None, None
    gray = cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY)
    return ARUCO_DET.detectMarkers(gray)


def _corner_pt(id_to_c: Dict[int, np.ndarray], marker_id: int, corner_idx: int) -> np.ndarray:
    return id_to_c[marker_id][0][corner_idx].astype(np.float32)


def solve_H_from_markers(
    cam_bgr: np.ndarray,
    warp_w: int,
    warp_h: int,
    corner_ids: Dict[str, int],
):
    corners, ids, _ = detect_markers(cam_bgr)
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
    pts_dst = np.array([[0, 0], [warp_w - 1, 0], [warp_w - 1, warp_h - 1], [0, warp_h - 1]], dtype=np.float32)
    H_view, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 3.0)
    return (H_view.astype(np.float32) if H_view is not None else None), detected_count, seen_ids, corners, ids


def roi_masks(
    cam_bgr: np.ndarray,
    H_view: Optional[np.ndarray],
    warp_w: int,
    warp_h: int,
    corner_ids: Dict[str, int],
):
    corners, ids, _ = detect_markers(cam_bgr)
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
# Session / bundle
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CoreStatus:
    marker_count: int
    seen_ids: List[int]
    missing_ids: List[int]
    locked: bool
    lock_miss_streak: int
    lock_lost_reason: str


@dataclass
class FrameBundle:
    # Always present
    frame_idx: int
    cam_bgr: np.ndarray  # undistorted (if calibration present), camera space
    warp_w: int
    warp_h: int
    grid_w: int
    grid_h: int

    # Lock/homography
    locked: bool
    H_use: np.ndarray
    mask_cam: Optional[np.ndarray]
    mask_warp: Optional[np.ndarray]

    # Debug/ArUco
    last_marker_count: int
    last_seen_ids: List[int]
    last_missing_ids: List[int]
    last_aruco_ids: Optional[np.ndarray]
    last_aruco_corners: Optional[Any]  # OpenCV corners structure

    # Pipeline outputs (only valid when locked=True)
    warp_blur: Optional[np.ndarray]
    warp_bgr: Optional[np.ndarray]
    bg_warp_u8: Optional[np.ndarray]
    motion_warp: Optional[np.ndarray]
    motion_cam: Optional[np.ndarray]
    shadowfree_cam: Optional[np.ndarray]
    final_mask_cam: Optional[np.ndarray]


def _resolve_grid_and_warp_from_setup() -> Tuple[int, int, int, int]:
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
    return warp_w, warp_h, grid_w, grid_h


class CVCoreSession:
    """
    Owns camera + lock state + shared mask pipeline.
    Engines iterate frames() and consume the returned FrameBundle.
    """

    def __init__(
        self,
        camera_index: Optional[int] = None,
        warp_w: Optional[int] = None,
        warp_h: Optional[int] = None,
        grid_w: Optional[int] = None,
        grid_h: Optional[int] = None,
        *,
        aruco_every_n: int = ARUCO_EVERY_N,
        aruco_every_n_fast: int = ARUCO_EVERY_N_FAST,
        lock_drop_after: int = LOCK_DROP_AFTER,
        warp_motion_thresh: int = WARP_MOTION_THRESH,
        fog_change_ratio: float = FOG_CHANGE_RATIO,
        bg_alpha_slow: float = BG_ALPHA_SLOW,
        bg_alpha_fast: float = BG_ALPHA_FAST,
    ):
        sel = s.load_last_selection() or {}
        if camera_index is None:
            camera_index = sel.get("webcam_index", 0)

        if warp_w is None or warp_h is None or grid_w is None or grid_h is None:
            _ww, _wh, _gw, _gh = _resolve_grid_and_warp_from_setup()
            warp_w = _ww if warp_w is None else int(warp_w)
            warp_h = _wh if warp_h is None else int(warp_h)
            grid_w = _gw if grid_w is None else int(grid_w)
            grid_h = _gh if grid_h is None else int(grid_h)

        self.camera_index = int(camera_index)
        self.warp_w = int(warp_w)
        self.warp_h = int(warp_h)
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)

        self.aruco_every_n = int(aruco_every_n)
        self.aruco_every_n_fast = int(aruco_every_n_fast)
        self.lock_drop_after = int(lock_drop_after)

        self.warp_motion_thresh = int(warp_motion_thresh)
        self.fog_change_ratio = float(fog_change_ratio)
        self.bg_alpha_slow = float(bg_alpha_slow)
        self.bg_alpha_fast = float(bg_alpha_fast)

        # Foundry grid params (optional)
        if fo is not None and hasattr(fo, "set_grid_params"):
            try:
                fo.set_grid_params(self.warp_w, self.warp_h, self.grid_w, self.grid_h)
            except Exception:
                pass

        self.frame_idx = 0

        # Camera calibration
        self.cam_mtx, self.cam_dist = get_camera_params()

        # Open camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam.")

        # Warm-up reads — macOS cameras need time to stabilise after open
        for _ in range(8):
            self.cap.read()
        time.sleep(0.3)  # let the camera buffer refill before capturing background

        # Background in camera space — retry a few times in case the buffer
        # isn't ready yet (common on macOS with USB cameras).
        frame_bg = None
        for attempt in range(6):
            ok_bg, frame_bg = self.cap.read()
            if ok_bg and frame_bg is not None:
                break
            time.sleep(0.1)
        if frame_bg is None:
            raise RuntimeError("Failed to capture camera-space background after retries.")
        frame_bg = self._maybe_undistort(frame_bg)
        gray_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2GRAY)
        blur_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)
        self.BG_cam = {"bgr": frame_bg, "blur": blur_bg}

        # Lock state
        self.H_saved = DEFAULT_H.copy()
        try:
            self._H_inv: np.ndarray = np.linalg.inv(DEFAULT_H).astype(np.float32)
        except Exception:
            self._H_inv = np.eye(3, dtype=np.float32)
        self.last_mask_cam: Optional[np.ndarray] = None
        self.last_mask_warp: Optional[np.ndarray] = None

        # Shadow pipeline throttle — run every N frames to keep FPS stable.
        # Cached results are reused on skipped frames.
        self._SHADOW_EVERY_N: int = 6
        self._shadow_frame_count: int = 0
        self._last_shadowfree_cam: Optional[np.ndarray] = None
        self._last_final_mask_cam: Optional[np.ndarray] = None

        self.last_marker_count = 0
        self.last_seen_ids: List[int] = []
        self.last_missing_ids: List[int] = sorted(list(REQUIRED_IDS))
        self.lock_lost_reason = ""
        self.lock_miss_streak = 0

        self.last_aruco_corners = None
        self.last_aruco_ids = None

        # Warp BG init
        ok_init, cam_init = self.cap.read()
        if ok_init and cam_init is not None:
            cam_init = self._maybe_undistort(cam_init)
            bg_init = warp_gray_blur(cam_init, self.H_saved, self.warp_w, self.warp_h)
            self.BG_warp_f32 = bg_init.astype(np.float32)
        else:
            self.BG_warp_f32 = np.zeros((int(self.warp_h), int(self.warp_w)), dtype=np.float32)

        # Temporal motion filter: per-pixel frame counter.
        # A pixel must be active for this many consecutive frames before it is
        # passed through as real motion.  Computed dynamically from FPS.
        # Default assumes ~30 fps; updated from measured frame times.
        self._motion_persist_acc = np.zeros((int(self.warp_h), int(self.warp_w)), dtype=np.uint8)
        self._motion_min_ms: int = 200       # configurable from outside
        self._last_frame_time: float = time.perf_counter()
        self._fps_estimate: float = 30.0

        # Post-motion healing: after a pixel leaves motion_warp, keep updating
        # the background at bg_alpha_fast for this many ms so fog-of-war reveals
        # (and other permanent scene changes) get absorbed quickly instead of
        # leaving permanent false-positive green trails.
        self._heal_ms: int = 1500   # ~1.5 s of fast BG update after motion clears
        self._heal_acc = np.zeros((int(self.warp_h), int(self.warp_w)), dtype=np.uint16)

    def _maybe_undistort(self, cam_bgr: np.ndarray) -> np.ndarray:
        if self.cam_mtx is not None and self.cam_dist is not None:
            try:
                return cv2.undistort(cam_bgr, self.cam_mtx, self.cam_dist)
            except Exception:
                return cam_bgr
        return cam_bgr

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass

    def _update_lock_if_due(self, cam_bgr: np.ndarray) -> None:
        have_lock = (self.last_mask_cam is not None) and (self.last_mask_warp is not None)
        need_fast = (not have_lock) or (self.last_marker_count < 4)
        aruco_interval = self.aruco_every_n_fast if need_fast else self.aruco_every_n

        if (self.frame_idx % aruco_interval) != 0:
            return

        H_view, detected_count, seen_ids, det_corners, det_ids = solve_H_from_markers(
            cam_bgr, self.warp_w, self.warp_h, CORNER_IDS
        )

        if det_ids is not None and det_corners is not None and len(det_ids) > 0:
            self.last_aruco_ids = det_ids
            self.last_aruco_corners = det_corners
        else:
            self.last_aruco_ids = None
            self.last_aruco_corners = None

        self.last_marker_count = int(detected_count)
        self.last_seen_ids = list(seen_ids)
        self.last_missing_ids = sorted(list(REQUIRED_IDS - set(self.last_seen_ids)))

        # Lock acquisition / loss with hysteresis
        if H_view is not None and len(self.last_missing_ids) == 0:
            self.H_saved = H_view
            try:
                self._H_inv = np.linalg.inv(H_view).astype(np.float32)
            except Exception:
                self._H_inv = np.eye(3, dtype=np.float32)
            self.last_mask_cam, self.last_mask_warp = roi_masks(cam_bgr, H_view, self.warp_w, self.warp_h, CORNER_IDS)
            self.lock_lost_reason = ""
            self.lock_miss_streak = 0
        else:
            if have_lock:
                self.lock_miss_streak += 1
                self.lock_lost_reason = f"Lost markers ({self.last_marker_count}/4 visible), missing: {self.last_missing_ids}"
                if self.lock_miss_streak >= self.lock_drop_after:
                    self.last_mask_cam = None
                    self.last_mask_warp = None
            else:
                self.lock_miss_streak = 0

    def _compute_shared_masks(self, cam_bgr: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        """
        Compute the shared mask pipeline (only when locked).
        Returns dict of warp_blur, warp_bgr, bg_warp_u8, motion_warp, motion_cam, shadowfree_cam, final_mask_cam.
        """
        _t0 = time.perf_counter()

        H_use = self.H_saved
        mask_cam = self.last_mask_cam
        mask_warp = self.last_mask_warp
        if mask_cam is None or mask_warp is None:
            return {
                "warp_blur": None,
                "warp_bgr": None,
                "bg_warp_u8": None,
                "motion_warp": None,
                "motion_cam": None,
                "shadowfree_cam": None,
                "final_mask_cam": None,
            }

        # Warp once, then extract gray — avoids warping the same frame twice
        warp_bgr = cv2.warpPerspective(cam_bgr, H_use, (int(self.warp_w), int(self.warp_h)))
        gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
        warp_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _t1 = time.perf_counter()

        bg_u8 = np.clip(self.BG_warp_f32, 0, 255).astype(np.uint8)
        diff_warp = cv2.absdiff(bg_u8, warp_blur)
        _, motion_warp = cv2.threshold(diff_warp, self.warp_motion_thresh, 255, cv2.THRESH_BINARY)
        motion_warp = cv2.erode(motion_warp, None, iterations=1)
        motion_warp = cv2.dilate(motion_warp, None, iterations=1)
        motion_warp = cv2.bitwise_and(motion_warp, mask_warp)

        roi_area = float(cv2.countNonZero(mask_warp))
        changed = float(cv2.countNonZero(motion_warp))
        change_ratio = (changed / max(1.0, roi_area))
        _t2 = time.perf_counter()

        # ── Temporal persistence filter (disabled — was shrinking blobs) ──
        now = time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * (1.0 / dt)
        _t3 = time.perf_counter()

        alpha_bg = self.bg_alpha_fast if change_ratio >= self.fog_change_ratio else self.bg_alpha_slow

        # ── Post-motion healing ────────────────────────────────────────────
        heal_frames = max(1, int(round(self._fps_estimate * self._heal_ms / 1000.0)))
        currently_motion = (motion_warp > 0)
        self._heal_acc = np.where(
            currently_motion,
            np.uint16(heal_frames),
            np.clip(self._heal_acc.astype(np.int32) - 1, 0, heal_frames).astype(np.uint16),
        )
        healing = (self._heal_acc > 0) & (~currently_motion)

        update_mask = cv2.bitwise_not(motion_warp)
        update_mask = cv2.bitwise_and(update_mask, mask_warp)

        healing_u8 = (healing.astype(np.uint8) * 255)
        non_healing_mask = cv2.bitwise_and(update_mask, cv2.bitwise_not(healing_u8))
        healing_mask     = cv2.bitwise_and(update_mask, healing_u8)

        self.BG_warp_f32 = update_bg_ema(self.BG_warp_f32, warp_blur, non_healing_mask, alpha_bg)
        self.BG_warp_f32 = update_bg_ema(self.BG_warp_f32, warp_blur, healing_mask,     self.bg_alpha_fast)
        _t4 = time.perf_counter()

        # Use cached H_inv — recomputed only when H_saved changes (in _update_lock_if_due)
        motion_cam = cv2.warpPerspective(motion_warp, self._H_inv, (cam_bgr.shape[1], cam_bgr.shape[0]), flags=cv2.INTER_NEAREST)
        _t5 = time.perf_counter()

        # Throttle the expensive shadow pipeline to every _SHADOW_EVERY_N frames.
        self._shadow_frame_count += 1
        if self._shadow_frame_count % self._SHADOW_EVERY_N == 0:
            shadowfree_cam = mt.shadow_free_mask(self.BG_cam["bgr"], cam_bgr)
            final_mask_cam = mt.combine_masks_componentwise(
                motion_cam, shadowfree_cam,
                keep_ratio=0.25,
                min_comp_area=150
            )
            final_mask_cam = cv2.bitwise_and(final_mask_cam, mask_cam)
            self._last_shadowfree_cam = shadowfree_cam
            self._last_final_mask_cam = final_mask_cam
        else:
            shadowfree_cam = self._last_shadowfree_cam
            final_mask_cam = self._last_final_mask_cam
        _t6 = time.perf_counter()

        # ── Timing report (once per second) ───────────────────────────────
        self._timing_count = getattr(self, '_timing_count', 0) + 1
        self._timing_acc   = getattr(self, '_timing_acc',   [0.0]*6)
        self._timing_acc[0] += _t1 - _t0
        self._timing_acc[1] += _t2 - _t1
        self._timing_acc[2] += _t3 - _t2
        self._timing_acc[3] += _t4 - _t3
        self._timing_acc[4] += _t5 - _t4
        self._timing_acc[5] += _t6 - _t5
        self._timing_last  = getattr(self, '_timing_last', _t0)
        if _t6 - self._timing_last >= 1.0:
            n = max(1, self._timing_count)
            labels = ["warp+blur", "diff+motion", "persist_filter", "healing+bg_ema", "inv_warp", "shadow(throttled)"]
            parts = "  ".join(f"{l}:{self._timing_acc[i]/n*1000:.1f}ms" for i, l in enumerate(labels))
            total = sum(self._timing_acc) / n * 1000
            print(f"CV_CORE timing/frame ({n}f):  {parts}  TOTAL:{total:.1f}ms", flush=True)
            self._timing_count = 0
            self._timing_acc   = [0.0]*6
            self._timing_last  = _t6
        # ──────────────────────────────────────────────────────────────────

        return {
            "warp_blur": warp_blur,
            "warp_bgr": warp_bgr,
            "bg_warp_u8": bg_u8,
            "motion_warp": motion_warp,
            "motion_cam": motion_cam,
            "shadowfree_cam": shadowfree_cam,
            "final_mask_cam": final_mask_cam,
        }

    def frames(self) -> Iterator[FrameBundle]:
        """
        Generator yielding FrameBundle for each camera frame.
        Engines can consume these bundles.
        """
        while True:
            self.frame_idx += 1

            try:
                ok, cam = self.cap.read()
                if not ok or cam is None:
                    # macOS cameras occasionally drop a frame — retry before giving up
                    for _ in range(5):
                        time.sleep(0.05)
                        ok, cam = self.cap.read()
                        if ok and cam is not None:
                            break
                    if not ok or cam is None:
                        print("CV_CORE | Camera read failed after retries — stopping.", flush=True)
                        break

                cam = self._maybe_undistort(cam)

                # Update lock state when due
                self._update_lock_if_due(cam)

                locked = (self.last_mask_cam is not None) and (self.last_mask_warp is not None)
                H_use = self.H_saved

                if locked:
                    try:
                        shared = self._compute_shared_masks(cam)
                    except Exception:
                        print("CV_CORE | _compute_shared_masks error (frame "
                              f"{self.frame_idx}):", flush=True)
                        traceback.print_exc()
                        shared = {
                            "warp_blur": None, "warp_bgr": None, "bg_warp_u8": None,
                            "motion_warp": None, "motion_cam": None,
                            "shadowfree_cam": None, "final_mask_cam": None,
                        }
                else:
                    shared = {
                        "warp_blur": None, "warp_bgr": None, "bg_warp_u8": None,
                        "motion_warp": None, "motion_cam": None,
                        "shadowfree_cam": None, "final_mask_cam": None,
                    }

                yield FrameBundle(
                    frame_idx=self.frame_idx,
                    cam_bgr=cam,
                    warp_w=self.warp_w,
                    warp_h=self.warp_h,
                    grid_w=self.grid_w,
                    grid_h=self.grid_h,
                    locked=locked,
                    H_use=H_use,
                    mask_cam=self.last_mask_cam,
                    mask_warp=self.last_mask_warp,
                    last_marker_count=int(self.last_marker_count),
                    last_seen_ids=list(self.last_seen_ids),
                    last_missing_ids=list(self.last_missing_ids),
                    last_aruco_ids=self.last_aruco_ids,
                    last_aruco_corners=self.last_aruco_corners,
                    warp_blur=shared["warp_blur"],
                    warp_bgr=shared["warp_bgr"],
                    bg_warp_u8=shared["bg_warp_u8"],
                    motion_warp=shared["motion_warp"],
                    motion_cam=shared["motion_cam"],
                    shadowfree_cam=shared["shadowfree_cam"],
                    final_mask_cam=shared["final_mask_cam"],
                )

            except GeneratorExit:
                break
            except Exception:
                print(f"CV_CORE | Unhandled error in frames() at frame {self.frame_idx}:", flush=True)
                traceback.print_exc()
                break