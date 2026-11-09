# cv_core.py
#
# Shared CV pipeline core for Sarween.
# This module owns the CV inputs shared by live tracking and replay:
# - camera open + optional undistortion
# - ArUco detection + homography solve + ROI masks (mask_cam/mask_warp)
# - warp-space background EMA + motion mask
# - shadow-free mask + final combined mask (camera-space)
#
# It does NOT do:
# - mini DB matching / capture picker / name logic
# - ring detection or identity tracking (that belongs in tracking_engine.py)
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
#       # Pass bundle["warp_bgr"] and motion/registration state to the tracker.
#
#   sess.close()

from __future__ import annotations

import os
import json
import time
import traceback
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Any, Iterator, Optional, Tuple, List

# ── Timing print toggle ───────────────────────────────────────────────────────
# Set to True via control panel to see per-frame pipeline timing in terminal.
PRINT_TIMING: bool = False

import cv2
import numpy as np
from app_paths import data_path
from camera_profile_runtime import CameraCalibrationChanged, CameraCorrection, camera_params, legacy_offline_camera_params
from camera_sources import (is_phone_camera_index, source_for_index, open_camera_capture,
                            profile_compatible_with_source, phone_profile_mismatch_reason, stable_stream_key)

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
VIEW_SETTLE_SECONDS = 1.25

ARUCO_EVERY_N = 30
ARUCO_EVERY_N_FAST = 5

LOCK_DROP_AFTER = 10  # consecutive ArUco misses before dropping lock
MIN_MARKERS_TO_HOLD_LOCK = 3
PARTIAL_LOCK_MAX_ERROR_PX = 8.0


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
    marker_mode: str = "legacy",
    corner_ids: Optional[Dict[str, int]] = None,
    display_backend=None,
) -> None:
    """Render a warped debug view with the grid in its true visible position."""
    count = 0
    seen_ids: List[int] = []
    if ARUCO_DET is not None:
        gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = ARUCO_DET.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            seen_ids = sorted(int(value) for value in ids.flatten())
            required = set((corner_ids or CORNER_IDS).values())
            count = len(required.intersection(seen_ids))

    if H_view is not None:
        pad = int(DEST_PAD_PX)
        canvas_w = int(warp_w + 2 * pad)
        canvas_h = int(warp_h + 2 * pad)

        T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], dtype=np.float32)
        H_pad = T @ H_view

        warped_dbg = cv2.warpPerspective(cam_frame, H_pad, (canvas_w, canvas_h))

        thick = 2
        lt = cv2.LINE_AA
        if marker_mode == "viewport" and fo is not None:
            for start, end in fo.warp_grid_segments(warp_w, warp_h):
                p1 = (int(round(start[0] + pad)), int(round(start[1] + pad)))
                p2 = (int(round(end[0] + pad)), int(round(end[1] + pad)))
                cv2.line(warped_dbg, p1, p2, (0, 255, 0), thick, lt)
        else:
            xs = np.linspace(0, warp_w, grid_w + 1)
            ys = np.linspace(0, warp_h, grid_h + 1)
            xs_i = np.unique(np.round(xs + pad).astype(int))
            ys_i = np.unique(np.round(ys + pad).astype(int))
            for xg in xs_i[1:-1]:
                cv2.line(warped_dbg, (int(xg), pad), (int(xg), pad + warp_h - 1), (0, 255, 0), thick, lt)
            for yg in ys_i[1:-1]:
                cv2.line(warped_dbg, (pad, int(yg)), (pad + warp_w - 1, int(yg)), (0, 255, 0), thick, lt)
            cv2.rectangle(warped_dbg, (pad, pad), (pad + warp_w - 1, pad + warp_h - 1), (0, 255, 0), thick, lt)
    else:
        warped_dbg = np.zeros((warp_h, warp_w, 3), dtype=np.uint8)

    marker_text = f"Required markers: {count}/4  seen={seen_ids}"
    cv2.putText(warped_dbg, marker_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(warped_dbg, marker_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    if display_backend is None:
        ensure_window("Homography view (debug)", 1280, 720)
        cv2.imshow("Homography view (debug)", warped_dbg)
    else:
        display_backend.ensure_window("Homography view (debug)", 1280, 720)
        display_backend.imshow("Homography view (debug)", warped_dbg)


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

LEGACY_CORNER_IDS = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
VIEWPORT_CORNER_IDS = {"TL": 10, "TR": 11, "BR": 12, "BL": 13}
CORNER_IDS = LEGACY_CORNER_IDS
REQUIRED_IDS = set(CORNER_IDS.values())


# ──────────────────────────────────────────────────────────────────────────────
# Camera calibration profiles
# ──────────────────────────────────────────────────────────────────────────────
def get_camera_params(image_size=None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    return camera_params(image_size)


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
    required_ids = set(corner_ids.values())
    detected_count = len(required_ids.intersection(seen_ids))

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


def markers_agree_with_homography(
    corners: Any,
    ids: Optional[np.ndarray],
    H_view: Optional[np.ndarray],
    warp_w: int,
    warp_h: int,
    corner_ids: Dict[str, int],
    *,
    min_markers: int = MIN_MARKERS_TO_HOLD_LOCK,
    max_error_px: float = PARTIAL_LOCK_MAX_ERROR_PX,
) -> bool:
    """Return whether visible registration corners still match a saved lock."""
    if ids is None or corners is None or H_view is None:
        return False

    id_to_c = {int(ids[i][0]): corners[i] for i in range(len(ids))}
    marker_specs = (
        (corner_ids["TL"], 0, (0.0, 0.0)),
        (corner_ids["TR"], 1, (float(warp_w - 1), 0.0)),
        (corner_ids["BR"], 2, (float(warp_w - 1), float(warp_h - 1))),
        (corner_ids["BL"], 3, (0.0, float(warp_h - 1))),
    )
    source_points = []
    destination_points = []
    for marker_id, corner_index, destination in marker_specs:
        if marker_id not in id_to_c:
            continue
        source_points.append(_corner_pt(id_to_c, marker_id, corner_index))
        destination_points.append(destination)

    if len(source_points) < int(min_markers):
        return False

    projected = cv2.perspectiveTransform(
        np.asarray(source_points, dtype=np.float32).reshape(1, -1, 2),
        H_view,
    ).reshape(-1, 2)
    expected = np.asarray(destination_points, dtype=np.float32)
    errors = np.linalg.norm(projected - expected, axis=1)
    return bool(np.max(errors) <= float(max_error_px))


def roi_masks(
    cam_bgr: np.ndarray,
    H_view: Optional[np.ndarray],
    warp_w: int,
    warp_h: int,
    corner_ids: Dict[str, int],
    grid_w: int = 23,
    grid_h: int = 16,
    edge_cells: float = 1.0,
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

    # Expand mask_warp outward by edge_cells grid squares so minis sitting on
    # the border edges are not clipped. Homography is unchanged — only the
    # blob-gating mask is wider.
    cell_w = int(round(warp_w / float(grid_w) * edge_cells))
    cell_h = int(round(warp_h / float(grid_h) * edge_cells))
    kw = max(3, cell_w * 2 + 1)
    kh = max(3, cell_h * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
    mask_warp = cv2.dilate(mask_warp, kernel, iterations=1)

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
    raw_motion_ratio: float = 0.0
    largest_motion_area: float = 0.0
    marker_mode: str = "legacy"
    watermark_mask_cam: Optional[np.ndarray] = None
    watermark_mask_warp: Optional[np.ndarray] = None
    watermark_exclusion_warp: Optional[np.ndarray] = None
    watermark_coverage: float = 0.0
    watermark_excluded_fraction: float = 0.0
    registration_metadata: Optional[dict] = None
    camera_source_metadata: Optional[dict] = None


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
        source_path: Optional[str] = None,
        marker_mode: Optional[str] = None,
        before_frame_callback: Optional[Any] = None,
        source_frames_undistorted: bool = False,
        view_settle_seconds: float = VIEW_SETTLE_SECONDS,
        loop_video: bool = False,
        watermark_detector=None,
        printed_marker_controller=None,
        camera_capture=None,
        tracking_pixels_only: bool = False,
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
        self.before_frame_callback = before_frame_callback
        self.source_frames_undistorted = bool(source_frames_undistorted)
        self.view_settle_seconds = max(0.0, float(view_settle_seconds))
        self.loop_video = bool(loop_video)
        # The shared base tracker consumes registered color pixels, not the
        # legacy camera-space motion/shadow masks. Keep that expensive pipeline
        # as the default for legacy callers and historical replay.
        self.tracking_pixels_only = bool(tracking_pixels_only)
        self._frame_phase_offset = 0
        self.recording_state_provider = None
        self._checkpoint_kind = None
        self.replay_frame_clock = None
        self.frame_time = None
        self._watermark_detector = watermark_detector
        self.reset_watermark_background()

        _src = (source_path or os.environ.get("TRUESIGHT_SOURCE", "")).strip()
        self._source_is_phone = bool(not _src and is_phone_camera_index(self.camera_index))
        self._camera_source = (source_for_index(self.camera_index, selection=sel)
                               if self._source_is_phone else None)
        self._camera_stream_key = None
        self._camera_source_metadata = None
        self._phone_gap_invalidated = False
        self._camera_profile_rejection = None
        self._phone_feedback = None
        if marker_mode is None:
            is_live_foundry = not _src and str(sel.get("mode", "")).lower() == "foundry"
            marker_mode = "viewport" if is_live_foundry else "legacy"
        marker_mode = str(marker_mode).strip().lower()
        if marker_mode not in {"legacy", "viewport"}:
            raise ValueError(f"Unsupported marker mode: {marker_mode}")
        self.marker_mode = marker_mode
        self.corner_ids = (
            VIEWPORT_CORNER_IDS if marker_mode == "viewport" else LEGACY_CORNER_IDS
        )
        self.required_ids = set(self.corner_ids.values())
        self._screen_corner_ids = dict(self.corner_ids)
        self._printed_marker_controller = printed_marker_controller
        if printed_marker_controller is None and not _src and marker_mode == "viewport":
            from printed_marker_runtime import get_printed_marker_controller
            self._printed_marker_controller = get_printed_marker_controller()
        self._registration_token = None
        self._registration_revision = 0
        self.registration_metadata = {"mode": "screen" if marker_mode == "viewport" else "legacy",
                                      "revision": 0}
        self._view_transform_revision = 0
        self._scene_visual_revision = 0
        self._view_settle_frames = 0
        if self.marker_mode == "viewport" and fo is not None:
            try:
                self._view_transform_revision = fo.get_view_transform_revision()
                self._scene_visual_revision = fo.get_scene_visual_revision()
            except Exception:
                pass
        print(
            f"CV_CORE | Marker mode: {self.marker_mode} "
            f"IDs={sorted(self.required_ids)}",
            flush=True,
        )

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

        # Resolve only after reading the actual frame. The selected profile is
        # frozen for this session; unrelated legacy files are never a fallback.
        confirmed_profile_id = sel.get("camera_profile_id")
        if self.camera_index != sel.get("webcam_index"):
            confirmed_profile_id = None
        self._camera_correction = (None if self.source_frames_undistorted or _src
                                   else CameraCorrection(profile_id=confirmed_profile_id))
        if self._camera_correction is not None:
            correction = self._camera_correction
            profile = correction.profiles.get_profile("camera", correction.profile_id) if correction.profile_id else None
            source = self._camera_source or {"type": "opencv", "index": self.camera_index}
            if not profile_compatible_with_source(profile, source):
                self._camera_profile_rejection = (("Choose this iPhone's Sarween Camera USB profile; "
                    "OBS and other camera calibrations do not match this feed. Lens correction is off.")
                    if self._source_is_phone else ("The selected Sarween Camera USB profile belongs to the direct iPhone feed; "
                    "choose a profile for this native/OBS camera. Lens correction is off."))
                self._camera_correction = CameraCorrection(profiles=correction.profiles, profile_id=None)
        self._legacy_offline_calibration = bool(_src and not self.source_frames_undistorted)
        self.cam_mtx, self.cam_dist = (legacy_offline_camera_params()
                                      if self._legacy_offline_calibration else (None, None))
        self.camera_calibration_status = ("Historical lens correction for raw legacy replay."
                                          if self._legacy_offline_calibration else
                                          "Recorded frames; no additional lens correction." if _src else
                                          "Waiting for camera frame.")
        self._legacy_correction_applied = False

        # ── TRUESIGHT_SOURCE: optional video file playback (offline debugging) ──
        self._source_is_file = bool(_src)

        if self._source_is_file:
            print(f"CV_CORE | TRUESIGHT_SOURCE mode — reading from: {_src}", flush=True)
            self.cap = cv2.VideoCapture(_src)
            if not self.cap.isOpened():
                self.cap.release()
                raise RuntimeError(f"TRUESIGHT_SOURCE: failed to open '{_src}'")
            # No warm-up for file sources
        else:
            # Open camera
            self.cap = (camera_capture if camera_capture is not None else
                        open_camera_capture(self.camera_index, selection=sel, source=self._camera_source))
            if not self.cap.isOpened():
                self.cap.release()
                raise RuntimeError("Failed to open webcam.")

            # Warm-up reads — macOS cameras need time to stabilise after open
            if camera_capture is None and not self._source_is_phone:
                for _ in range(8):
                    self.cap.read()
                time.sleep(0.3)  # let the camera buffer refill before capturing background
            if self._source_is_phone:
                from phone_camera_feedback import PhoneCameraFeedback
                self._phone_feedback = PhoneCameraFeedback(self.cap, kind="tracking",
                    marker_ids=tuple(self.corner_ids[key] for key in ("TL", "TR", "BR", "BL")), detect_interval=1.)

        # ── Video recorder (optional) ─────────────────────────────────────────
        self._recorder: Optional[cv2.VideoWriter] = None
        self._record_path: Optional[str] = None

        # Background in camera space — retry a few times in case the buffer
        # isn't ready yet (common on macOS with USB cameras).
        frame_bg = None
        for attempt in range(1 if self._source_is_phone else 6):
            ok_bg, frame_bg = self.cap.read()
            if ok_bg and frame_bg is not None:
                break
            if self._source_is_phone:
                frame_bg = None
            if not self._source_is_phone:
                time.sleep(0.1)
        if frame_bg is None:
            if self._source_is_phone:
                # Startup has normally waited for a fresh frame. A cable loss
                # racing that handoff must still leave a cancellable live UI.
                width = max(1, min(8192, int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1080)))
                height = max(1, min(8192, int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1920)))
                frame_bg = np.zeros((height, width, 3), np.uint8)
                self._camera_source_metadata = self._phone_frame_metadata(False)
            else:
                self.cap.release()
                raise RuntimeError("Failed to capture camera-space background after retries.")
        elif self._source_is_phone:
            self._camera_source_metadata = self._phone_frame_metadata(True)
            self._camera_stream_key = stable_stream_key(self._camera_source_metadata)
        self._phone_has_camera_frame = bool(not self._source_is_phone or ok_bg and frame_bg is not None)
        if self._source_is_phone and self._camera_source_metadata.get("freshFrame") is False:
            self._phone_has_camera_frame = False
        try:
            if not self._source_is_phone or self._phone_has_camera_frame:
                if self._source_is_phone:
                    self._validate_phone_profile_metadata(self._camera_source_metadata)
                frame_bg = self._maybe_undistort(frame_bg)
        except Exception:
            self.cap.release()
            raise
        gray_bg = cv2.cvtColor(frame_bg, cv2.COLOR_BGR2GRAY)
        blur_bg = cv2.GaussianBlur(gray_bg, (21, 21), 0)
        self.BG_cam = {"bgr": frame_bg, "blur": blur_bg}
        self._last_camera_frame = frame_bg

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
        self.last_missing_ids: List[int] = sorted(self.required_ids)
        self.lock_lost_reason = ""
        self.lock_miss_streak = 0

        self.last_aruco_corners = None
        self.last_aruco_ids = None

        # Warp BG — seeded on first locked frame (after ArUco lock gives us the
        # real homography). Initialising with identity-warped content causes a
        # massive false-motion burst on frame 1 because H_saved is still eye(3).
        self.BG_warp_f32: Optional[np.ndarray] = None  # None = not yet seeded
        self._bg_seeded: bool = False

        # Post-motion healing: after a pixel that was genuinely moving goes
        # still, keep updating its background at bg_alpha_fast for _heal_ms so
        # fog-of-war reveals get absorbed quickly. Only fires on pixels that
        # were actually in motion — never on cold/noisy pixels.
        self._heal_ms: int = 1500
        self._heal_acc = np.zeros((int(self.warp_h), int(self.warp_w)), dtype=np.uint16)

        # File replay must use the recording's timeline rate for every
        # frame-count-based timeout. Otherwise results vary with CPU speed.
        source_fps = (
            float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if self._source_is_file
            else 0.0
        )
        self._last_frame_time: float = time.perf_counter()
        self._fps_estimate: float = source_fps if source_fps > 0 else 30.0

    def _maybe_undistort(self, cam_bgr: np.ndarray) -> np.ndarray:
        # Sarween writes recordings after this correction has already run.
        # Timeline-backed replay marks those frames so they are not corrected twice.
        if self.source_frames_undistorted:
            return cam_bgr
        if (hasattr(self, "BG_cam") and self.BG_cam["bgr"].shape[:2] != cam_bgr.shape[:2]
                and (not getattr(self, "_source_is_phone", False) or self._phone_has_camera_frame)):
            self.camera_calibration_status = "Camera frame size changed. Restart the live session to register the feed again."
            raise CameraCalibrationChanged(self.camera_calibration_status)
        correction = self._camera_correction
        if correction is None:
            if self._legacy_offline_calibration and self.cam_mtx is not None and self.cam_dist is not None:
                try:
                    corrected = cv2.undistort(cam_bgr, self.cam_mtx, self.cam_dist)
                    self._legacy_correction_applied = True
                    return corrected
                except Exception:
                    pass
            return cam_bgr
        previous_matrix, previous_distortion = self.cam_mtx, self.cam_dist
        if getattr(self, "_source_is_phone", False):
            self.cam_mtx, self.cam_dist = correction.resolve((cam_bgr.shape[1], cam_bgr.shape[0]),
                source_metadata=self._camera_source_metadata)
        else:
            self.cam_mtx, self.cam_dist = correction.resolve((cam_bgr.shape[1], cam_bgr.shape[0]))
        if getattr(self, "_camera_profile_rejection", None):
            correction.reason = self._camera_profile_rejection
        if (hasattr(self, "BG_cam")
                and (not getattr(self, "_source_is_phone", False) or self._phone_has_camera_frame) and
                (not np.array_equal(previous_matrix, self.cam_mtx)
                 or not np.array_equal(previous_distortion, self.cam_dist))):
            self.camera_calibration_status = "Camera calibration changed. Restart the live session to register the feed again."
            raise CameraCalibrationChanged(self.camera_calibration_status)
        if self.camera_calibration_status != correction.reason:
            self.camera_calibration_status = correction.reason
            print(f"CV_CORE | Camera calibration: {correction.reason}", flush=True)
        correction.applied = False
        if self.cam_mtx is not None and self.cam_dist is not None:
            try:
                corrected = cv2.undistort(cam_bgr, self.cam_mtx, self.cam_dist)
                correction.applied = True
                return corrected
            except Exception as exc:
                correction.reason = self.camera_calibration_status = f"Lens correction failed: {exc}"
                raise CameraCalibrationChanged(self.camera_calibration_status) from exc
        return cam_bgr

    def camera_calibration_metadata(self):
        if self._camera_correction is None:
            return {"profileId": None, "applied": self._legacy_correction_applied,
                    "legacyOffline": self._legacy_offline_calibration, "reason": self.camera_calibration_status}
        return self._camera_correction.snapshot()

    def registration_metadata_snapshot(self):
        metadata = deepcopy(getattr(self, "registration_metadata", {
            "mode": "screen" if getattr(self, "marker_mode", "legacy") == "viewport" else "legacy",
            "revision": 0}))
        if getattr(self, "_source_is_phone", False) and self._camera_stream_key is not None:
            metadata["cameraStream"] = deepcopy(self._camera_stream_key)
        return metadata

    def _phone_frame_metadata(self, fresh):
        """Volatile receive clocks stay out of calibration/geometry keys."""
        status = getattr(self.cap, "status", {})
        status = status() if callable(status) else status
        metadata = deepcopy(status) if isinstance(status, dict) else {}
        if fresh:
            last = getattr(self.cap, "last_frame_metadata", None)
            last = last() if callable(last) else last
            if isinstance(last, dict):
                metadata.update(deepcopy(last))
        metadata["freshFrame"] = bool(fresh)
        return metadata

    def _validate_phone_profile_metadata(self, metadata):
        """Reject a physical-lens switch before even one old-matrix correction."""
        correction = self._camera_correction
        if correction is None or not correction.profile_id:
            return
        profile = correction.profiles.get_profile("camera", correction.profile_id)
        reason = phone_profile_mismatch_reason(profile, self._camera_source, metadata)
        if reason:
            correction.matrix = correction.dist_coeffs = None
            correction.applied = False
            correction.reason = self.camera_calibration_status = reason
            self.cam_mtx = self.cam_dist = None
            raise CameraCalibrationChanged(reason)

    def _sync_phone_stream(self, cam, metadata):
        key = stable_stream_key(metadata)
        if key != self._camera_stream_key:
            self._camera_stream_key = key
            self._invalidate_registration(cam, "iPhone camera stream changed; acquire marker lock and bind minis again",
                                          configuration_changed=True)
            self.last_marker_count = 0
            self.last_seen_ids = []
            self.last_missing_ids = sorted(self.required_ids)
            self.last_aruco_ids = self.last_aruco_corners = None
        self._phone_gap_invalidated = False

    def _phone_wait_bundle(self):
        """Cached pixels are only a diagnostic preview, never a measurement."""
        cam = self._last_camera_frame
        status = self._phone_frame_metadata(False)
        reason = status.get("message") or "Waiting for a fresh iPhone camera frame"
        if not self._phone_gap_invalidated:
            self._invalidate_registration(cam, reason)
            self._phone_gap_invalidated = True
        self.last_marker_count = 0
        self.last_seen_ids = []
        self.last_missing_ids = sorted(self.required_ids)
        self.last_aruco_ids = self.last_aruco_corners = None
        self.frame_time = time.perf_counter()
        self._camera_source_metadata = status
        self._publish_phone_feedback(None, status, locked=False)
        return FrameBundle(frame_idx=self.frame_idx, cam_bgr=cam,
            warp_w=self.warp_w, warp_h=self.warp_h, grid_w=self.grid_w, grid_h=self.grid_h,
            locked=False, H_use=DEFAULT_H.copy(), mask_cam=None, mask_warp=None,
            last_marker_count=0, last_seen_ids=[], last_missing_ids=list(self.last_missing_ids),
            last_aruco_ids=None, last_aruco_corners=None,
            warp_blur=None, warp_bgr=None, bg_warp_u8=None, motion_warp=None, motion_cam=None,
            shadowfree_cam=None, final_mask_cam=None, marker_mode=self.marker_mode,
            registration_metadata=self.registration_metadata_snapshot(), camera_source_metadata=status)

    def _publish_phone_feedback(self, frame, metadata, *, locked):
        helper = getattr(self, "_phone_feedback", None)
        if helper is None:
            return
        try:
            helper.marker_ids = tuple(self.corner_ids[key] for key in ("TL", "TR", "BR", "BL"))
            view = (fo.get_view_transform_payload() if fo is not None else None)
            revision = (fo.get_scene_visual_revision() if fo is not None else None)
            markers = None
            if frame is not None and getattr(self, "_marker_detection_frame", None) == self.frame_idx:
                markers = {"sourceSeq": metadata.get("seq"), "corners": self.last_aruco_corners,
                           "ids": self.last_aruco_ids}
            helper.publish(frame, metadata, markers=markers, locked=bool(locked), view_key=view, scene_revision=revision,
                           registration=self.registration_metadata_snapshot())
            # Tracking cannot execute calibration requests. CPL requests are
            # explicit phone actions and only alter this diagnostic reference.
            helper.take_requests()
        except Exception as exc:
            # Optional phone guidance must never change tracker measurements.
            # The phone expires old feedback independently when updates stop.
            self.phone_feedback_error = str(exc)

    @staticmethod
    def _printed_registration_token(config, context):
        view_key = None
        if config["mode"] == "printed" and context.get("view"):
            view = context["view"]
            view_key = {"viewportWidth": view.get("viewportWidth"), "viewportHeight": view.get("viewportHeight"),
                        "registration": deepcopy(view.get("registration"))}
        token = json.dumps({"configuration": config["token"], "viewport": view_key,
                            "pending": config["pending"], "blocked": config["blocked"]}, sort_keys=True)
        return token, view_key

    def configure_recorded_registration(self, metadata, *, preserve_geometry=False):
        """Install only recorded layout/view data for a file replay.

        Checkpoint restore passes preserve_geometry after restoring its numeric
        H/masks/background. A timeline-only configuration starts unacquired.
        """
        if not getattr(self, "_source_is_file", False):
            raise ValueError("Recorded registration is only available for file sources")
        if not isinstance(metadata, dict):
            raise ValueError("Invalid recorded marker registration")
        if metadata.get("mode") == "legacy":
            self._printed_marker_controller = None
            self.corner_ids = dict(LEGACY_CORNER_IDS)
            self.required_ids = set(self.corner_ids.values())
            self.registration_metadata = deepcopy(metadata)
            return
        from printed_marker_runtime import RecordedPrintedMarkerController
        provider = lambda: fo.get_view_transform_payload() if fo is not None else None
        controller = RecordedPrintedMarkerController(metadata, view_provider=provider)
        self._printed_marker_controller = controller
        self._screen_corner_ids = dict(VIEWPORT_CORNER_IDS)
        config, context = controller.configuration(), controller.view_context()
        self._registration_token, _view_key = self._printed_registration_token(config, context)
        self._registration_revision = int(metadata.get("revision", 0))
        self.registration_metadata = deepcopy(metadata)
        from printed_markers import PRINTED_IDS
        self.corner_ids = dict(PRINTED_IDS if config["mode"] == "printed" else VIEWPORT_CORNER_IDS)
        self.required_ids = set(self.corner_ids.values())
        if not preserve_geometry:
            self.last_marker_count = 0
            self.last_seen_ids = []
            self.last_missing_ids = sorted(self.required_ids)
            self.last_aruco_ids = self.last_aruco_corners = None
            self.last_mask_cam = self.last_mask_warp = None
            self.H_saved = DEFAULT_H.copy()
            self._H_inv = DEFAULT_H.copy()
            self.BG_warp_f32 = None
            self._bg_seeded = False
            self._last_shadowfree_cam = self._last_final_mask_cam = None
            self.reset_watermark_background()

    def _invalidate_registration(self, cam_bgr, reason, *, configuration_changed=False):
        """Discard unsafe geometry once per loss or explicit configuration change."""
        already_invalid = (self.last_mask_cam is None and self.last_mask_warp is None
            and self.BG_warp_f32 is None and not self._bg_seeded
            and np.array_equal(self.H_saved, DEFAULT_H)
            and self._last_shadowfree_cam is None and self._last_final_mask_cam is None)
        self.lock_miss_streak = 0
        self.lock_lost_reason = reason
        if already_invalid and not configuration_changed:
            return
        self.last_mask_cam = self.last_mask_warp = None
        self.H_saved = DEFAULT_H.copy()
        self._H_inv = DEFAULT_H.copy()
        self.BG_warp_f32 = None
        self._bg_seeded = False
        self._last_shadowfree_cam = self._last_final_mask_cam = None
        self._heal_acc = np.zeros((self.warp_h, self.warp_w), np.uint16)
        self.BG_cam = {"bgr": cam_bgr.copy(), "blur": cv2.GaussianBlur(
            cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY), (21, 21), 0)}
        self.reset_watermark_background()
        if getattr(self, "_recorder", None) is not None:
            self._checkpoint_kind = "registration"

    def _sync_printed_registration(self, cam_bgr):
        controller = getattr(self, "_printed_marker_controller", None)
        if not controller:
            return None
        controller.heartbeat(self)
        config = controller.configuration()
        context = controller.view_context()
        self._printed_view_context = context
        token, view_key = self._printed_registration_token(config, context)
        changed = token != getattr(self, "_registration_token", None)
        if changed:
            self._registration_token = token
            self._registration_revision = getattr(self, "_registration_revision", 0) + 1
            self._invalidate_registration(cam_bgr, "Marker configuration changed; acquire lock and bind minis again",
                                          configuration_changed=True)
        from printed_markers import PRINTED_IDS
        self.corner_ids = dict(PRINTED_IDS if config["mode"] == "printed" else self._screen_corner_ids)
        self.required_ids = set(self.corner_ids.values())
        if changed:
            self.last_marker_count = 0
            self.last_seen_ids = []
            self.last_missing_ids = sorted(self.required_ids)
            self.last_aruco_ids = self.last_aruco_corners = None
        self.registration_metadata = {"mode": config["mode"], "profileId": config["profileId"],
            "profileName": config["profileName"], "layoutFingerprint": config["token"].get("layoutFingerprint"),
            "revision": self._registration_revision, "viewport": view_key, "pending": config["pending"],
            "calibrationMethod": (config["layout"] or {}).get("calibrationMethod"),
            "printedMarkerLayout": deepcopy(config["layout"]),
            "controllerRevision": config["token"].get("revision", 0)}
        return config

    def _update_printed_lock(self, cam_bgr, corners, ids, config):
        from printed_markers import (PRINTED_IDS, agrees_with_homography, detections,
                                     roi_masks as printed_roi_masks, solve_homography)
        self.last_marker_count = 0
        self.last_seen_ids = []
        self.last_missing_ids = sorted(self.required_ids)
        self.last_aruco_corners, self.last_aruco_ids = corners, ids
        try:
            found = detections(corners, ids, set(PRINTED_IDS.values()))
            self.last_seen_ids = sorted(found)
            self.last_marker_count = len(found)
            self.last_missing_ids = sorted(self.required_ids - set(found))
            self.last_aruco_corners, self.last_aruco_ids = corners, ids
            context = self._printed_view_context
            if not context.get("ready") or not context.get("view"):
                raise ValueError("Waiting for the current fullscreen Foundry viewport")
            layout = config["layout"]
            view = context["view"]
            warp_size = (self.warp_w, self.warp_h)
            H = solve_homography(corners, ids, layout, view, warp_size)
            if H is not None:
                self.H_saved = H
                self._H_inv = np.linalg.inv(H).astype(np.float32)
                self.last_mask_cam, self.last_mask_warp = printed_roi_masks(cam_bgr.shape, H, warp_size)
                self.lock_miss_streak = 0
                self.lock_lost_reason = ""
                return
            have_lock = self.last_mask_cam is not None and self.last_mask_warp is not None
            if have_lock and agrees_with_homography(corners, ids, self.H_saved, layout, view, warp_size):
                self.lock_miss_streak = 0
                self.lock_lost_reason = f"Holding verified printed lock ({len(found)}/4 visible)"
                return
            raise ValueError(f"Printed registration unverified ({len(found)}/4 visible); need four cards to acquire and three matching cards to hold")
        except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
            # A cached paper transform is not current evidence. Screen/legacy
            # modes retain their existing grace behavior below.
            self._invalidate_registration(cam_bgr, str(exc))

    def reset_watermark_background(self):
        """A new/legacy background needs caption detection before comparisons."""
        self._watermark_background_known = False
        self._watermark_background_ref = None
        for name in ("_watermark_bg_invalid_cam", "_watermark_bg_invalid_warp",
                     "_watermark_clear_count_cam", "_watermark_clear_count_warp",
                     "_watermark_prev_blur_cam", "_watermark_prev_blur_warp"):
            setattr(self, name, None)

    @staticmethod
    def _watermark_binary(mask, shape):
        if mask is None or not isinstance(mask, np.ndarray) or mask.shape != shape:
            return np.zeros(shape, np.uint8)
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _watermark_observation(self, image):
        if self._watermark_free_phone_source():
            zero = np.zeros(image.shape[:2], np.uint8)
            return zero, zero.copy()
        if getattr(self, "_watermark_detector", None) is None:
            from watermark_detection import ObsWatermarkDetector
            self._watermark_detector = ObsWatermarkDetector()
        observation = self._watermark_detector.detect(image)
        glyph = self._watermark_binary(observation.mask, image.shape[:2])
        exclusion = self._watermark_binary(getattr(observation, "exclusion_mask", None), image.shape[:2])
        if cv2.countNonZero(glyph) and not cv2.countNonZero(exclusion):
            exclusion = cv2.dilate(glyph, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23)))
        return glyph, cv2.bitwise_or(exclusion, glyph)

    def _watermark_free_phone_source(self):
        """Only the owned, validated USB source guarantees watermark-free pixels.

        A profile name, image size, native webcam or recorded file is not proof
        of that source. Unknown/OBS capture paths retain caption detection.
        """
        metadata = getattr(self, "_camera_source_metadata", None) or {}
        return bool(getattr(self, "_source_is_phone", False)
                    and not getattr(self, "_source_is_file", False)
                    and metadata.get("source") == "sarween-phone-usb"
                    and metadata.get("transport") == "usb")

    def _compute_tracking_pixels(self, cam_bgr):
        """Registered board pixels without unused legacy background/motion work."""
        warp_bgr = cv2.warpPerspective(cam_bgr, self.H_saved,
                                      (int(self.warp_w), int(self.warp_h)))
        if self._watermark_free_phone_source():
            # Replace the current state explicitly; old exclusions must not
            # survive a stream transition or be mistaken for current coverage.
            glyph_cam = exclusion_cam = None
            glyph_warp = np.zeros((self.warp_h, self.warp_w), np.uint8)
            exclusion_warp = np.zeros_like(glyph_warp)
        else:
            glyph_cam, exclusion_cam = self._watermark_observation(cam_bgr)
            glyph_warp = cv2.warpPerspective(glyph_cam, self.H_saved,
                (self.warp_w, self.warp_h), flags=cv2.INTER_NEAREST)
            exclusion_warp = self._watermark_project_exclusion(exclusion_cam)
        self._watermark_mask_cam = glyph_cam
        self._watermark_mask_warp = glyph_warp
        self._watermark_current_exclusion_cam = exclusion_cam
        self._watermark_current_exclusion_warp = exclusion_warp
        roi = self.last_mask_warp
        area = max(1, cv2.countNonZero(roi))
        return {"warp_bgr": warp_bgr, "warp_blur": None, "bg_warp_u8": None,
                "motion_warp": None, "motion_cam": None,
                "shadowfree_cam": None, "final_mask_cam": None,
                "raw_motion_ratio": 0.0, "largest_motion_area": 0.0,
                "watermark_mask_cam": glyph_cam, "watermark_mask_warp": glyph_warp,
                "watermark_exclusion_warp": exclusion_warp,
                "watermark_coverage": cv2.countNonZero(cv2.bitwise_and(glyph_warp, roi)) / area,
                "watermark_excluded_fraction": cv2.countNonZero(cv2.bitwise_and(exclusion_warp, roi)) / area}

    def _watermark_project_exclusion(self, mask):
        projected = cv2.warpPerspective(mask, self.H_saved, (self.warp_w, self.warp_h), flags=cv2.INTER_NEAREST)
        # The warp-space motion comparison is blurred with a 21-pixel kernel.
        return cv2.dilate(projected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))

    def _prepare_watermark_masks(self, image):
        glyph, exclusion = self._watermark_observation(image)
        self._watermark_mask_cam = glyph
        self._watermark_mask_warp = cv2.warpPerspective(glyph, self.H_saved,
            (self.warp_w, self.warp_h), flags=cv2.INTER_NEAREST)
        self._watermark_current_exclusion_cam = exclusion
        self._watermark_current_exclusion_warp = self._watermark_project_exclusion(exclusion)
        background = self.BG_cam["bgr"]
        if (not getattr(self, "_watermark_background_known", False)
                or getattr(self, "_watermark_background_ref", None) != id(background)):
            _glyph_bg, exclusion_bg = self._watermark_observation(background)
            self._watermark_bg_invalid_cam = exclusion_bg.copy()
            self._watermark_bg_invalid_warp = self._watermark_project_exclusion(exclusion_bg)
            self._watermark_clear_count_cam = np.zeros(image.shape[:2], np.uint8)
            self._watermark_clear_count_warp = np.zeros((self.warp_h, self.warp_w), np.uint8)
            self._watermark_prev_blur_cam = None
            self._watermark_prev_blur_warp = None
            self._watermark_background_known = True
            self._watermark_background_ref = id(background)

    def _seed_watermark_masks(self, warp_blur, *, camera=False):
        self._watermark_bg_invalid_warp = self._watermark_current_exclusion_warp.copy()
        self._watermark_clear_count_warp = np.zeros_like(warp_blur)
        self._watermark_prev_blur_warp = warp_blur.copy()
        if camera:
            self._watermark_bg_invalid_cam = self._watermark_current_exclusion_cam.copy()
            self._watermark_clear_count_cam = np.zeros(self.BG_cam["bgr"].shape[:2], np.uint8)
            self._watermark_prev_blur_cam = self.BG_cam["blur"].copy()
            self._watermark_background_ref = id(self.BG_cam["bgr"])
            self._watermark_background_known = True

    def _relearn_watermark_background(self, image, warp_blur):
        """Learn only observed pixels after two consecutive clear, stable pairs.

        Hidden pixels stay invalid; no image content is synthesized. A moving
        hand cannot immediately become the replacement background when text
        uncovers it. Ordinary motion handling remains unchanged elsewhere.
        """
        for space, current_exclusion, current in (
                ("warp", self._watermark_current_exclusion_warp, warp_blur),
                ("cam", self._watermark_current_exclusion_cam, None)):
            invalid = getattr(self, "_watermark_bg_invalid_" + space)
            if invalid is None or not cv2.countNonZero(invalid):
                continue
            if current is None:
                current = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (21, 21), 0)
            count = getattr(self, "_watermark_clear_count_" + space)
            if count is None or count.shape != invalid.shape:
                count = np.zeros_like(invalid)
            previous = getattr(self, "_watermark_prev_blur_" + space)
            clear = (invalid > 0) & (current_exclusion == 0)
            stable = clear & (cv2.absdiff(previous, current) <= 6) if previous is not None else np.zeros_like(clear)
            count[~stable] = 0
            count[stable] = np.minimum(count[stable].astype(np.uint16) + 1, 3).astype(np.uint8)
            learned = stable & (count >= 2)
            if np.any(learned):
                if space == "warp":
                    self.BG_warp_f32[learned] = current[learned]
                else:
                    self.BG_cam["bgr"][learned] = image[learned]
                    self.BG_cam["blur"][learned] = current[learned]
                invalid[learned] = 0
                count[learned] = 0
            setattr(self, "_watermark_clear_count_" + space, count)
            setattr(self, "_watermark_prev_blur_" + space, current.copy())

    def close(self, preserve_recording: bool = False) -> None:
        controller = getattr(self, "_printed_marker_controller", None)
        if controller:
            controller.camera_stopped(self)
        if not preserve_recording:
            self.stop_recording()
        try:
            self.cap.release()
        except Exception:
            pass

    def detach_recorder(self) -> Tuple[Optional["cv2.VideoWriter"], Optional[str]]:
        """Remove the recorder from this session without releasing it.
        The caller is responsible for attaching it to the next session or releasing it."""
        rec, path = self._recorder, self._record_path
        self._recorder = None
        self._record_path = None
        return rec, path

    def attach_recorder(self, recorder: "cv2.VideoWriter", path: str) -> None:
        """Adopt an already-open VideoWriter (e.g. carried over from a prior session)."""
        if self._recorder is not None:
            self.stop_recording()
        self._recorder = recorder
        self._record_path = path
        self._checkpoint_kind = "restart"
        print(f"CV_CORE | Recording resumed → {path}", flush=True)

    def start_recording(self, path: str, fps: Optional[float] = None) -> bool:
        """Open a VideoWriter and begin saving raw camera frames.  Returns True on success.

        fps — actual processing rate measured by the caller.  When None, the camera's
              reported FPS is used, which often causes the saved video to play back
              faster than real-time because the CV pipeline runs slower than the nominal
              camera rate.
        """
        if self._recorder is not None:
            self.stop_recording()
        try:
            # Use the camera's native size if available; fall back to a sensible default.
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
            if fps is None or fps <= 0:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
            if fps <= 0:
                fps = 30.0
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not writer.isOpened():
                print(f"CV_CORE | Recording: VideoWriter failed to open '{path}'", flush=True)
                return False
            self._recorder = writer
            self._record_path = path
            self._checkpoint_kind = "initial"
            if fo is not None and hasattr(fo, "start_timeline_recording"):
                try:
                    fo.start_timeline_recording(
                        path,
                        fps=float(fps),
                        frame_width=w,
                        frame_height=h,
                        marker_mode=self.marker_mode,
                        warp_width=self.warp_w,
                        warp_height=self.warp_h,
                        grid_cols=self.grid_w,
                        grid_rows=self.grid_h,
                        frames_undistorted=True,
                        camera_calibration=self.camera_calibration_metadata(),
                        marker_registration=self.registration_metadata_snapshot(),
                    )
                except Exception as exc:
                    print(f"CV_CORE | Foundry timeline start error: {exc}", flush=True)
            print(f"CV_CORE | Recording started → {path}  ({w}×{h} @ {fps:.0f}fps)", flush=True)
            return True
        except Exception as exc:
            print(f"CV_CORE | Recording start error: {exc}", flush=True)
            return False

    def stop_recording(self) -> None:
        """Flush and close the VideoWriter if one is open."""
        if self._recorder is not None:
            try:
                self._recorder.release()
                print(f"CV_CORE | Recording saved → {self._record_path}", flush=True)
            except Exception as exc:
                print(f"CV_CORE | Recording stop error: {exc}", flush=True)
            if fo is not None and hasattr(fo, "stop_timeline_recording"):
                try:
                    fo.stop_timeline_recording()
                except Exception as exc:
                    print(f"CV_CORE | Foundry timeline stop error: {exc}", flush=True)
            self._recorder = None
            self._record_path = None

    @property
    def is_recording(self) -> bool:
        return self._recorder is not None

    def _update_lock_if_due(self, cam_bgr: np.ndarray) -> None:
        config = self._sync_printed_registration(cam_bgr)
        if config is not None and config["blocked"]:
            self.last_mask_cam = self.last_mask_warp = None
            self.lock_lost_reason = config.get("message", "Printed marker configuration unavailable")
            return
        have_lock = (self.last_mask_cam is not None) and (self.last_mask_warp is not None)
        need_fast = (not have_lock) or (self.last_marker_count < 4) or bool(config and config["pending"])
        aruco_interval = self.aruco_every_n_fast if need_fast else self.aruco_every_n

        if ((self.frame_idx + self._frame_phase_offset) % aruco_interval) != 0:
            return

        if config is not None and (config["pending"] or config["mode"] == "printed"):
            det_corners, det_ids, _rejected = detect_markers(cam_bgr)
            self._marker_detection_frame = self.frame_idx
            if config["pending"]:
                self._printed_marker_controller.consume_detections(det_corners, det_ids,
                    self._printed_view_context, (self.warp_w, self.warp_h))
                config = self._sync_printed_registration(cam_bgr)
            if config["mode"] == "printed":
                self._update_printed_lock(cam_bgr, det_corners, det_ids, config)
                return

        H_view, detected_count, seen_ids, det_corners, det_ids = solve_H_from_markers(
            cam_bgr, self.warp_w, self.warp_h, self.corner_ids
        )
        self._marker_detection_frame = self.frame_idx

        if det_ids is not None and det_corners is not None and len(det_ids) > 0:
            self.last_aruco_ids = det_ids
            self.last_aruco_corners = det_corners
        else:
            self.last_aruco_ids = None
            self.last_aruco_corners = None

        self.last_marker_count = int(detected_count)
        self.last_seen_ids = list(seen_ids)
        self.last_missing_ids = sorted(self.required_ids - set(self.last_seen_ids))

        # Full visibility is required to acquire or refresh the homography. Once
        # acquired, three matching corners are enough to prove the camera and TV
        # have not moved, so a single dim/glared marker does not stop tracking.
        if H_view is not None and len(self.last_missing_ids) == 0:
            self.H_saved = H_view
            try:
                self._H_inv = np.linalg.inv(H_view).astype(np.float32)
            except Exception:
                self._H_inv = np.eye(3, dtype=np.float32)
            self.last_mask_cam, self.last_mask_warp = roi_masks(
                cam_bgr, H_view, self.warp_w, self.warp_h, self.corner_ids,
                grid_w=self.grid_w, grid_h=self.grid_h,
            )
            self.lock_lost_reason = ""
            self.lock_miss_streak = 0
        elif have_lock and markers_agree_with_homography(
            det_corners,
            det_ids,
            self.H_saved,
            self.warp_w,
            self.warp_h,
            self.corner_ids,
        ):
            self.lock_lost_reason = (
                f"Holding verified lock ({self.last_marker_count}/4 visible), "
                f"missing: {self.last_missing_ids}"
            )
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
                "raw_motion_ratio": 0.0,
                "largest_motion_area": 0.0,
            }

        # Warp once, then extract gray — avoids warping the same frame twice
        warp_bgr = cv2.warpPerspective(cam_bgr, H_use, (int(self.warp_w), int(self.warp_h)))
        gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
        warp_blur = cv2.GaussianBlur(gray, (21, 21), 0)
        self._prepare_watermark_masks(cam_bgr)

        # A Foundry pan changes the entire displayed map while the physical
        # minis remain still. Once the browser publishes the completed canvas
        # transform, reseed both backgrounds so the new map is not treated as
        # permanent motion.
        if self.marker_mode == "viewport" and fo is not None:
            try:
                revision = fo.get_view_transform_revision()
            except Exception:
                revision = self._view_transform_revision
            if revision != self._view_transform_revision:
                had_background = self._bg_seeded
                self._view_transform_revision = revision
                if had_background and revision > 0:
                    self._view_settle_frames = max(
                        4, int(round(self._fps_estimate * self.view_settle_seconds))
                    )
                    print(
                        f"CV_CORE | Foundry view changed (revision {revision}); "
                        "settling and reseeding background.",
                        flush=True,
                    )

            try:
                visual_revision = fo.get_scene_visual_revision()
            except Exception:
                visual_revision = self._scene_visual_revision
            if visual_revision != self._scene_visual_revision:
                self._scene_visual_revision = visual_revision
                if self._bg_seeded and visual_revision > 0:
                    self._view_settle_frames = max(
                        self._view_settle_frames,
                        max(4, int(round(self._fps_estimate * self.view_settle_seconds))),
                    )
                    print(
                        f"CV_CORE | Foundry visual changed (revision {visual_revision}); "
                        "settling and reseeding background.",
                        flush=True,
                    )

        if self._view_settle_frames > 0:
            self.BG_warp_f32 = warp_blur.astype(np.float32)
            self.BG_cam = {
                "bgr": cam_bgr.copy(),
                "blur": cv2.GaussianBlur(
                    cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2GRAY), (21, 21), 0
                ),
            }
            self._heal_acc.fill(0)
            self._last_shadowfree_cam = None
            self._last_final_mask_cam = None
            self._seed_watermark_masks(warp_blur, camera=True)
            self._view_settle_frames -= 1

        # Seed background on first locked frame so we always use the real H
        if not self._bg_seeded:
            self.BG_warp_f32 = warp_blur.astype(np.float32)
            self._bg_seeded = True
            self._seed_watermark_masks(warp_blur)
            print("CV_CORE | Warp background seeded on first locked frame.")
        _t1 = time.perf_counter()

        watermark_exclusion_warp = cv2.bitwise_or(self._watermark_current_exclusion_warp,
                                                  self._watermark_bg_invalid_warp)
        watermark_exclusion_cam = cv2.bitwise_or(self._watermark_current_exclusion_cam,
                                                 self._watermark_bg_invalid_cam)
        comparison_mask = cv2.bitwise_and(mask_warp, cv2.bitwise_not(watermark_exclusion_warp))
        camera_comparison_mask = cv2.bitwise_and(mask_cam, cv2.bitwise_not(watermark_exclusion_cam))

        bg_u8 = np.clip(self.BG_warp_f32, 0, 255).astype(np.uint8) if self.BG_warp_f32 is not None else warp_blur
        diff_warp = cv2.absdiff(bg_u8, warp_blur)
        _, motion_warp = cv2.threshold(diff_warp, self.warp_motion_thresh, 255, cv2.THRESH_BINARY)
        motion_warp = cv2.bitwise_and(motion_warp, comparison_mask)
        motion_warp = cv2.erode(motion_warp, None, iterations=1)
        motion_warp = cv2.dilate(motion_warp, None, iterations=1)
        motion_warp = cv2.bitwise_and(motion_warp, comparison_mask)

        roi_area = float(cv2.countNonZero(comparison_mask))
        changed = float(cv2.countNonZero(motion_warp))
        change_ratio = (changed / max(1.0, roi_area))
        _t2 = time.perf_counter()

        # Filter out blobs larger than MAX_BLOB_CELLS x MAX_BLOB_CELLS grid squares.
        # This removes large obstructions (hands, cats, fog-of-war reveals covering
        # many cells at once) while keeping genuine mini-sized blobs.
        # We filter per-contour so small blobs survive even if large ones are present.
        MAX_BLOB_CELLS = 5
        grid_dimensions = None
        if self.marker_mode == "viewport" and fo is not None:
            try:
                grid_dimensions = fo.warp_grid_dimensions(self.warp_w, self.warp_h)
            except Exception:
                grid_dimensions = None
        if grid_dimensions is None:
            grid_dimensions = (
                self.warp_w / float(self.grid_w),
                self.warp_h / float(self.grid_h),
            )
        cell_area_warp = grid_dimensions[0] * grid_dimensions[1]
        max_blob_area  = cell_area_warp * (MAX_BLOB_CELLS * MAX_BLOB_CELLS)

        cnts_info = cv2.findContours(motion_warp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_motion = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
        largest_motion_area = max((cv2.contourArea(mc) for mc in cnts_motion), default=0.0)
        motion_filtered = np.zeros_like(motion_warp)
        for mc in cnts_motion:
            if cv2.contourArea(mc) <= max_blob_area:
                cv2.drawContours(motion_filtered, [mc], -1, 255, thickness=-1)
        motion_warp = cv2.bitwise_and(motion_filtered, comparison_mask)

        _t3 = time.perf_counter()

        # Background EMA update with post-motion healing.
        # IMPORTANT: use raw motion_warp (pre-filter) to charge _heal_acc so
        # large fog-of-war changes still trigger fast healing after they settle.
        # Only the engine-visible motion_warp is filtered; the BG update sees all
        # genuine motion pixels.
        alpha_bg = self.bg_alpha_fast if change_ratio >= self.fog_change_ratio else self.bg_alpha_slow

        now = self.frame_time if self.frame_time is not None else time.perf_counter()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0 and (not self._source_is_file or self.replay_frame_clock is not None):
            self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * (1.0 / dt)
        heal_frames = max(1, int(round(self._fps_estimate * self._heal_ms / 1000.0)))

        # Charge heal_acc from ALL motion (pre-filter) — fog reveals charge this.
        # Use original contours (before size filter) to charge heal_acc
        motion_for_heal = np.zeros_like(motion_warp)
        for mc in cnts_motion:
            cv2.drawContours(motion_for_heal, [mc], -1, 255, thickness=-1)
        motion_for_heal = cv2.bitwise_and(motion_for_heal, comparison_mask)
        heal_active = (motion_for_heal > 0)

        self._heal_acc = np.where(
            heal_active,
            np.uint16(heal_frames),
            np.clip(self._heal_acc.astype(np.int32) - 1, 0, heal_frames).astype(np.uint16),
        )
        healing = (~heal_active) & (self._heal_acc > 0)

        # Pixels in motion (any size): excluded from BG update entirely
        # Pixels healing (were motion, now still): fast BG update
        # All other pixels: normal alpha update
        not_motion_u8 = cv2.bitwise_not(motion_for_heal)
        in_roi        = cv2.bitwise_and(not_motion_u8, comparison_mask)
        healing_u8    = (healing.astype(np.uint8) * 255)
        healing_mask  = cv2.bitwise_and(in_roi, healing_u8)
        normal_mask   = cv2.bitwise_and(in_roi, cv2.bitwise_not(healing_u8))

        self._relearn_watermark_background(cam_bgr, warp_blur)
        self.BG_warp_f32 = update_bg_ema(self.BG_warp_f32, warp_blur, normal_mask,  alpha_bg)
        self.BG_warp_f32 = update_bg_ema(self.BG_warp_f32, warp_blur, healing_mask, self.bg_alpha_fast)
        _t4 = time.perf_counter()

        # Use cached H_inv — recomputed only when H_saved changes (in _update_lock_if_due)
        motion_cam = cv2.warpPerspective(motion_warp, self._H_inv, (cam_bgr.shape[1], cam_bgr.shape[0]), flags=cv2.INTER_NEAREST)
        motion_cam = cv2.bitwise_and(motion_cam, camera_comparison_mask)
        _t5 = time.perf_counter()

        # Throttle the expensive shadow pipeline to every _SHADOW_EVERY_N frames.
        self._shadow_frame_count += 1
        if self._shadow_frame_count % self._SHADOW_EVERY_N == 0:
            shadowfree_cam = mt.shadow_free_mask(self.BG_cam["bgr"], cam_bgr)
            shadowfree_cam = cv2.bitwise_and(shadowfree_cam, camera_comparison_mask)
            final_mask_cam = mt.combine_masks_componentwise(
                motion_cam, shadowfree_cam,
                keep_ratio=0.25,
                min_comp_area=150
            )
            final_mask_cam = cv2.bitwise_and(final_mask_cam, camera_comparison_mask)
            self._last_shadowfree_cam = shadowfree_cam
            self._last_final_mask_cam = final_mask_cam
        else:
            shadowfree_cam = self._last_shadowfree_cam
            final_mask_cam = self._last_final_mask_cam
            if shadowfree_cam is not None:
                shadowfree_cam = cv2.bitwise_and(shadowfree_cam, camera_comparison_mask)
            if final_mask_cam is not None:
                final_mask_cam = cv2.bitwise_and(final_mask_cam, camera_comparison_mask)
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
            labels = ["warp+blur", "diff+motion", "bg_ema", "inv_warp", "shadow(throttled)", "unused"]
            parts = "  ".join(f"{l}:{self._timing_acc[i]/n*1000:.1f}ms" for i, l in enumerate(labels))
            total = sum(self._timing_acc) / n * 1000
            if PRINT_TIMING:
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
            "raw_motion_ratio": change_ratio,
            "largest_motion_area": float(largest_motion_area),
            "watermark_mask_cam": self._watermark_mask_cam,
            "watermark_mask_warp": self._watermark_mask_warp,
            "watermark_exclusion_warp": watermark_exclusion_warp,
            "watermark_coverage": cv2.countNonZero(cv2.bitwise_and(self._watermark_mask_warp, mask_warp)) / max(1, cv2.countNonZero(mask_warp)),
            "watermark_excluded_fraction": cv2.countNonZero(cv2.bitwise_and(watermark_exclusion_warp, mask_warp)) / max(1, cv2.countNonZero(mask_warp)),
        }

    def frames(self) -> Iterator[FrameBundle]:
        """
        Generator yielding FrameBundle for each camera frame.
        Engines can consume these bundles.
        """
        while True:
            if (getattr(self, "_recorder", None) is not None and self._checkpoint_kind and self.recording_state_provider
                    and (not getattr(self, "_source_is_phone", False)
                         or (self._camera_source_metadata or {}).get("freshFrame") is not False)):
                fo.record_checkpoint(self, self.recording_state_provider(), self._checkpoint_kind)
                self._checkpoint_kind = None
            self.frame_idx += 1

            try:
                if self.before_frame_callback is not None:
                    self.before_frame_callback(self.frame_idx)
                if self.marker_mode == "viewport" and fo is not None:
                    scene = fo.get_scene_params()
                    self.grid_w = int(scene["gridCols"])
                    self.grid_h = int(scene["gridRows"])
                ok, cam = self.cap.read()
                if not ok or cam is None:
                    if getattr(self, "_source_is_phone", False):
                        yield self._phone_wait_bundle()
                        continue
                    if self._source_is_file:
                        if not self.loop_video:
                            print("CV_CORE | TRUESIGHT_SOURCE: end of video — stopping.", flush=True)
                            break
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, cam = self.cap.read()
                        if not ok or cam is None:
                            print("CV_CORE | TRUESIGHT_SOURCE: cannot loop video — stopping.", flush=True)
                            break
                    else:
                        # macOS cameras occasionally drop a frame — retry before giving up
                        for _ in range(5):
                            time.sleep(0.05)
                            ok, cam = self.cap.read()
                            if ok and cam is not None:
                                break
                        if not ok or cam is None:
                            print("CV_CORE | Camera read failed after retries — stopping.", flush=True)
                            break

                if getattr(self, "_source_is_phone", False):
                    metadata = self._phone_frame_metadata(True)
                    self._camera_source_metadata = metadata
                    self._validate_phone_profile_metadata(metadata)
                cam = self._maybe_undistort(cam)
                if getattr(self, "_source_is_phone", False):
                    self._sync_phone_stream(cam, metadata)
                    self._phone_has_camera_frame = True
                    self._last_camera_frame = cam
                self.frame_time = (float(self.replay_frame_clock) if self.replay_frame_clock is not None
                                   else time.perf_counter())

                # Write frame to recorder if active
                if self._recorder is not None:
                    try:
                        self._recorder.write(cam)
                        if fo is not None and hasattr(fo, "record_timeline_frame"):
                            fo.record_timeline_frame(capture_time=self.frame_time, core_frame=self.frame_idx,
                                                     motion_thresh=self.warp_motion_thresh)
                    except Exception:
                        pass

                # Update lock state when due
                self._update_lock_if_due(cam)

                locked = (self.last_mask_cam is not None) and (self.last_mask_warp is not None)
                H_use = self.H_saved
                if getattr(self, "_source_is_phone", False):
                    self._publish_phone_feedback(cam, self._camera_source_metadata, locked=locked)

                if locked:
                    try:
                        shared = (self._compute_tracking_pixels(cam) if getattr(self, "tracking_pixels_only", False)
                                  else self._compute_shared_masks(cam))
                    except Exception:
                        print("CV_CORE | _compute_shared_masks error (frame "
                              f"{self.frame_idx}):", flush=True)
                        traceback.print_exc()
                        shared = {
                            "warp_blur": None, "warp_bgr": None, "bg_warp_u8": None,
                            "motion_warp": None, "motion_cam": None,
                            "shadowfree_cam": None, "final_mask_cam": None,
                            "raw_motion_ratio": 0.0, "largest_motion_area": 0.0,
                        }
                else:
                    shared = {
                        "warp_blur": None, "warp_bgr": None, "bg_warp_u8": None,
                        "motion_warp": None, "motion_cam": None,
                        "shadowfree_cam": None, "final_mask_cam": None,
                        "raw_motion_ratio": 0.0, "largest_motion_area": 0.0,
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
                    raw_motion_ratio=shared["raw_motion_ratio"],
                    largest_motion_area=shared["largest_motion_area"],
                    marker_mode=self.marker_mode,
                    watermark_mask_cam=shared.get("watermark_mask_cam"),
                    watermark_mask_warp=shared.get("watermark_mask_warp"),
                    watermark_exclusion_warp=shared.get("watermark_exclusion_warp"),
                    watermark_coverage=shared.get("watermark_coverage", 0.0),
                    watermark_excluded_fraction=shared.get("watermark_excluded_fraction", 0.0),
                    registration_metadata=self.registration_metadata_snapshot(),
                    camera_source_metadata=deepcopy(getattr(self, "_camera_source_metadata", None)),
                )

            except GeneratorExit:
                break
            except CameraCalibrationChanged:
                raise
            except Exception:
                print(f"CV_CORE | Unhandled error in frames() at frame {self.frame_idx}:", flush=True)
                traceback.print_exc()
                break
