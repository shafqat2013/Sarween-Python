# mini_calibration.py
#
# Full mini calibration phase for Sarween combo engine.
#
# Flow:
#   1. Fetch scene background image from Foundry (via HTTP)
#   2. Display it on TV with ArUco corner markers overlaid
#   3. CVCoreSession locks on markers — warp pipeline active
#   4. User moves each mini → blob appears → click blob → type name
#   5. Press SPACE when all minis labeled → capture phase begins
#   6. Background fades through 7 brightness steps (full → black)
#   7. At each step: wait for camera to stabilise, sample Lab color per blob
#   8. Save color curves to combo_profiles.json
#
# Runtime matching: v3_tracking calls min_lab_dist_to_profile() which tests
# the sampled color against all curve points and returns the minimum distance.

from __future__ import annotations

import json
import math
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import setup as s
import cv_core as core
import calibration as cal

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_PROFILES_PATH = Path(__file__).with_name("combo_profiles.json")

BRIGHTNESS_STEPS = [255, 200, 150, 100, 60, 30, 0]
SETTLE_TIME = 2.5

MIN_CIRCULARITY  = 0.1
MAX_ASPECT_RATIO = 2.5
MIN_AREA_FRAC    = 0.04
MAX_AREA_FRAC    = 9.0
SAMPLE_RADIUS_FRAC = 0.35
DEFAULT_MAX_LAB_DIST = 40.0
MORPH_CLOSE_FRAC = 0.30

WIN_TV   = "Sarween Calibration"
WIN_PREV = "Mini Calibration Preview"


# ──────────────────────────────────────────────────────────────────────────────
# Scene background fetch
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_scene_background(target_w: int, target_h: int) -> Optional[np.ndarray]:
    """
    Fetch the Foundry scene background image via HTTP and resize to target dims.
    Returns None if unavailable.
    """
    import foundryoutput as fo
    params = fo.get_scene_params()
    bg_path = params.get("background")
    if not bg_path:
        print("MINI_CAL | No background URL from Foundry — using white image")
        return None

    url = bg_path if bg_path.startswith("http") else f"http://localhost:30000/{bg_path.lstrip('/')}"
    print(f"MINI_CAL | Fetching scene background from {url}")
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read()
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print("MINI_CAL | Failed to decode background image")
            return None
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        print(f"MINI_CAL | Background loaded and resized to {target_w}x{target_h}")
        return img
    except Exception as e:
        print(f"MINI_CAL | Could not fetch background: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Calibration image generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_calibration_image(
    screen_w: int,
    screen_h: int,
    brightness: int = 255,
    base_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Generate calibration image with ArUco corner markers overlaid.
    If base_image provided: scale its brightness (255=full, 0=black).
    Otherwise: flat colour at given brightness.
    """
    if base_image is not None:
        alpha = brightness / 255.0
        img = (base_image.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    else:
        img = np.full((screen_h, screen_w, 3), brightness, dtype=np.uint8)

    cols = getattr(s, "grid_cols", 23)
    rows = getattr(s, "grid_rows", 16)
    cell_px  = cal._compute_cell_px(screen_w, screen_h, cols, rows)
    # ArUco markers need a white quiet zone on all sides. In the normal blended
    # display this comes from add_white_border() on the whole canvas. Here we
    # paint a white rect at each corner and inset the tile from the edge.
    # Equal white quiet zone on all four sides of each marker.
    # cell_px // 8 is one ArUco module width — minimum needed by the detector.
    quiet_px = max(8, cell_px // 8)
    qs = quiet_px  # shorthand

    # White rect covers corner area with equal padding on all sides of the tile.
    # (wx0, wy0, wx1, wy1, tile_x, tile_y)
    corners = [
        (0,                      0,                      cell_px+2*qs, cell_px+2*qs, qs,                      qs),
        (screen_w-cell_px-2*qs,  0,                      screen_w,     cell_px+2*qs, screen_w-cell_px-qs,     qs),
        (screen_w-cell_px-2*qs,  screen_h-cell_px-2*qs,  screen_w,     screen_h,     screen_w-cell_px-qs,     screen_h-cell_px-qs),
        (0,                      screen_h-cell_px-2*qs,  cell_px+2*qs, screen_h,     qs,                      screen_h-cell_px-qs),
    ]

    for i, (wx0, wy0, wx1, wy1, tx, ty) in enumerate(corners):
        img[wy0:wy1, wx0:wx1] = 255  # white quiet zone

        path = os.path.join(cal.MARKERS_DIR, f"marker_{i}.png")
        if not os.path.exists(path):
            print(f"MINI_CAL | Warning: marker tile {path} not found")
            continue
        tile = cv2.imread(path, cv2.IMREAD_COLOR)
        if tile is None:
            continue
        tile = cv2.resize(tile, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
        img[ty:ty+cell_px, tx:tx+cell_px] = tile

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Blob detection (motion mask — same as v3_tracking)
# ──────────────────────────────────────────────────────────────────────────────

def _find_blobs(
    bundle: "core.FrameBundle",
    grid_px: float,
) -> List[Tuple[np.ndarray, float, float, float, float]]:
    # During calibration labeling we want raw motion, not the shadow-filtered
    # final_mask_cam.  The shadow filter can produce an all-zero mask on the
    # first several frames (before any motion has been seen), which permanently
    # blocks the motion_warp fallback once final_mask_cam becomes non-None.
    # Using motion_warp directly here is intentional: we want to show the user
    # every blob so they can label it, not silently suppress shadow-like objects.
    if bundle.motion_warp is not None:
        mask = bundle.motion_warp.copy()
    else:
        return []

    if mask is None:
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
    cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    blobs = []
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < cell_area * MIN_AREA_FRAC or area > cell_area * MAX_AREA_FRAC:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        circ = float(4.0 * math.pi * area / (peri * peri))
        if circ < MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if max(w, h) / max(1.0, min(w, h)) > MAX_ASPECT_RATIO:
            continue
        M = cv2.moments(cnt)
        if abs(M["m00"]) < 1e-6:
            continue
        blobs.append((cnt, M["m10"]/M["m00"], M["m01"]/M["m00"], area, circ))

    return blobs


# ──────────────────────────────────────────────────────────────────────────────
# Lab color sampling
# ──────────────────────────────────────────────────────────────────────────────

def _sample_lab(
    warp_bgr: np.ndarray,
    cx: float,
    cy: float,
    area: float,
) -> Optional[Tuple[float, float, float]]:
    h, w = warp_bgr.shape[:2]
    r = max(2, int(round(math.sqrt(area / math.pi) * SAMPLE_RADIUS_FRAC)))
    x0, y0 = max(0, int(cx)-r), max(0, int(cy)-r)
    x1, y1 = min(w, int(cx)+r+1), min(h, int(cy)+r+1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = warp_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    m = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
    cv2.circle(m, (int(cx)-x0, int(cy)-y0), r, 255, -1)
    pixels = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)[m > 0]
    if pixels.shape[0] == 0:
        return None
    med = np.median(pixels, axis=0)
    return (float(med[0])*100.0/255.0, float(med[1])-128.0, float(med[2])-128.0)


# ──────────────────────────────────────────────────────────────────────────────
# Interactive blob labeling
# ──────────────────────────────────────────────────────────────────────────────

class BlobLabeler:
    def __init__(self):
        self.blobs: List = []
        self.labels: Dict[int, str] = {}
        self._pending_idx: Optional[int] = None
        self._input_text: str = ""
        self._input_active: bool = False
        self.done: bool = False

    def set_blobs(self, blobs):
        self.blobs = blobs

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self._input_active:
            return
        best_idx, best_dist = None, float("inf")
        for i, (cnt, cx, cy, area, circ) in enumerate(self.blobs):
            d = math.hypot(x - cx, y - cy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None and best_dist < 120:
            self._pending_idx = best_idx
            self._input_text = self.labels.get(best_idx, "")
            self._input_active = True

    def handle_key(self, key: int) -> bool:
        if key == 32:  # SPACE
            if not self._input_active:
                self.done = True
            return True
        if self._input_active:
            if key in (13, 10):  # Enter
                if self._pending_idx is not None:
                    name = self._input_text.strip()
                    if name:
                        self.labels[self._pending_idx] = name
                    elif self._pending_idx in self.labels:
                        del self.labels[self._pending_idx]
                self._input_active = False
                self._pending_idx = None
                self._input_text = ""
                return True
            elif key == 27:  # Escape
                self._input_active = False
                self._pending_idx = None
                self._input_text = ""
                return True
            elif key in (8, 127):
                self._input_text = self._input_text[:-1]
                return True
            elif 32 <= key <= 126:
                self._input_text += chr(key)
                return True
        return False

    def render(self, warp_bgr: np.ndarray) -> np.ndarray:
        vis = warp_bgr.copy()
        for i, (cnt, cx, cy, area, circ) in enumerate(self.blobs):
            is_selected = (i == self._pending_idx and self._input_active)
            is_labeled  = i in self.labels
            if is_selected:
                color, thickness = (0, 200, 255), 3
            elif is_labeled:
                color, thickness = (0, 255, 0), 2
            else:
                color, thickness = (180, 180, 0), 1
            cv2.drawContours(vis, [cnt.astype(np.int32)], -1, color, thickness)
            label = (self._input_text + "|") if is_selected else self.labels.get(i, "?")
            cv2.putText(vis, label, (int(cx)+6, int(cy)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, label, (int(cx)+6, int(cy)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        bh, bw = vis.shape[:2]
        if self._input_active:
            cv2.rectangle(vis, (10, bh-60), (bw-10, bh-10), (40,40,40), -1)
            cv2.rectangle(vis, (10, bh-60), (bw-10, bh-10), (200,200,200), 1)
            cv2.putText(vis, f"Name this mini: {self._input_text}|",
                        (20, bh-28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(vis, "Enter = confirm   Esc = cancel",
                        (20, bh-14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1, cv2.LINE_AA)
        else:
            status = f"{len(self.labels)}/{len(self.blobs)} labeled   SPACE = start capture   click blob to name"
            cv2.putText(vis, status, (12, bh-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,255,0), 1, cv2.LINE_AA)
        return vis


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_mini_calibration(camera_index: Optional[int] = None) -> bool:
    """
    Run full mini calibration. Returns True on success, False on cancel.
    Saves results to combo_profiles.json.
    """
    # ── Screen dimensions ────────────────────────────────────────────────────
    sel = s.load_last_selection() or {}
    display_idx = sel.get("display_index", 0)
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        mon = monitors[display_idx] if display_idx < len(monitors) else monitors[0]
        screen_w, screen_h = mon.width, mon.height
        screen_x, screen_y = mon.x, mon.y
    except Exception:
        screen_w, screen_h = 1920, 1080
        screen_x, screen_y = 0, 0

    print(f"MINI_CAL | Target display: {screen_w}x{screen_h} at ({screen_x},{screen_y})")

    # ── Fetch scene background ───────────────────────────────────────────────
    import foundryoutput as fo
    print("MINI_CAL | Waiting for Foundry scene info (make sure Foundry is open)...")
    deadline = time.perf_counter() + 15.0
    while time.perf_counter() < deadline:
        params = fo.get_scene_params()
        if params.get("background") or params.get("gridPx"):
            break
        time.sleep(0.25)
    else:
        print("MINI_CAL | Timed out waiting for Foundry — proceeding without background")

    # Use scene dimensions for the calibration image so ArUco marker positions
    # match exactly where Foundry places them. The window will scale to fill screen.
    params = fo.get_scene_params()
    scene_w = params.get("sceneW") or 1656
    scene_h = params.get("sceneH") or 1152
    print(f"MINI_CAL | Scene dimensions: {scene_w}x{scene_h}")

    scene_bg = _fetch_scene_background(scene_w, scene_h)

    # ── Show calibration image ───────────────────────────────────────────────
    cal_img = _generate_calibration_image(scene_w, scene_h, brightness=255,
                                          base_image=scene_bg)
    cv2.namedWindow(WIN_TV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TV, scene_w, scene_h)
    cv2.imshow(WIN_TV, cal_img)

    # Also create the preview window now — Qt will process both window
    # creation events during the waitKey(0) below, guaranteeing both
    # window handles exist before we attach the mouse callback.
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.namedWindow(WIN_PREV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREV, 1280, 720)
    cv2.imshow(WIN_PREV, blank)

    try:
        cv2.moveWindow(WIN_TV, screen_x, max(0, screen_y))
    except Exception:
        pass

    print("MINI_CAL | Calibration image displayed.")
    print("MINI_CAL | If it's on the wrong screen, drag it to the TV and make it fullscreen.")
    print("MINI_CAL | Press any key in that window when ready...")
    cv2.waitKey(0)

    # Destroy the pre-created preview window before opening the camera.
    # cv2.VideoCapture for an iPhone Continuity Camera blocks the main thread
    # for several seconds — Qt marks any existing window as unresponsive during
    # that time and invalidates its native handle, breaking setMouseCallback.
    # We create the labeling window AFTER the camera is open to avoid this.
    try:
        cv2.destroyWindow(WIN_PREV)
        cv2.waitKey(1)
    except Exception:
        pass

    # ── Start camera ─────────────────────────────────────────────────────────
    _cam_idx = camera_index if camera_index is not None else (
        (s.load_last_selection() or {}).get("webcam_index", 0))
    print(f"MINI_CAL | Starting camera (index {_cam_idx})...")
    sess = core.CVCoreSession(camera_index=camera_index)

    # Camera is open — now create the labeling window. Qt event loop is free
    # and will assign a valid native handle immediately.
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.namedWindow(WIN_PREV, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PREV, 1280, 720)
    cv2.imshow(WIN_PREV, blank)
    cv2.waitKey(1)

    labeler = BlobLabeler()
    _callback_set = False

    print("MINI_CAL | Waiting for ArUco lock...")
    print("MINI_CAL | Move each mini to trigger a blob, then click it to name it.")
    print("MINI_CAL | Press SPACE when all minis are labeled.")

    # ── Labeling phase ───────────────────────────────────────────────────────
    last_blob_update = 0.0

    try:
        for bundle in sess.frames():
            if not bundle.locked or bundle.warp_bgr is None:
                if bundle.cam_bgr is not None:
                    waiting = cv2.resize(bundle.cam_bgr, (1280, 720))
                else:
                    waiting = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(waiting,
                            f"Waiting for ArUco lock ({bundle.last_marker_count}/4)...",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(WIN_PREV, waiting)
                k = cv2.waitKey(30) & 0xFF
                if not _callback_set:
                    try:
                        cv2.setMouseCallback(WIN_PREV, labeler.on_mouse)
                        _callback_set = True
                        print("MINI_CAL | Mouse callback registered")
                    except cv2.error:
                        pass
                if k == 27:
                    sess.close()
                    cv2.destroyAllWindows()
                    return False
                continue

            grid_px = min(bundle.warp_w / float(bundle.grid_w),
                          bundle.warp_h / float(bundle.grid_h))

            now = time.perf_counter()
            if now - last_blob_update > 0.5:
                labeler.set_blobs(_find_blobs(bundle, grid_px))
                last_blob_update = now

            cv2.imshow(WIN_PREV, labeler.render(bundle.warp_bgr))
            key = cv2.waitKey(30) & 0xFF
            if not _callback_set:
                try:
                    cv2.setMouseCallback(WIN_PREV, labeler.on_mouse)
                    _callback_set = True
                    print("MINI_CAL | Mouse callback registered")
                except cv2.error:
                    pass
            labeler.handle_key(key)

            if key == 27:
                sess.close()
                cv2.destroyAllWindows()
                return False

            if labeler.done:
                break

        if not labeler.labels:
            print("MINI_CAL | No minis labeled. Aborting.")
            sess.close()
            cv2.destroyAllWindows()
            return False

        print(f"MINI_CAL | Labeled: {list(labeler.labels.values())}")
        print("MINI_CAL | Starting brightness capture...")

        # ── Capture phase ────────────────────────────────────────────────────
        curves: Dict[str, List] = {name: [] for name in labeler.labels.values()}
        blob_info = {idx: (cx, cy, area)
                     for idx, (cnt, cx, cy, area, circ) in enumerate(labeler.blobs)
                     if idx in labeler.labels}

        last_bundle = bundle

        for step_i, brightness in enumerate(BRIGHTNESS_STEPS):
            print(f"MINI_CAL | Step {step_i+1}/{len(BRIGHTNESS_STEPS)}: brightness={brightness}")

            step_img = _generate_calibration_image(scene_w, scene_h,
                                                   brightness=brightness,
                                                   base_image=scene_bg)
            cv2.imshow(WIN_TV, step_img)
            cv2.waitKey(1)

            # Progress display
            prog = np.full((720, 1280, 3), 40, dtype=np.uint8)
            cv2.putText(prog,
                        f"Step {step_i+1}/{len(BRIGHTNESS_STEPS)}  brightness={brightness}",
                        (40, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(prog, f"Waiting {SETTLE_TIME:.0f}s for camera...",
                        (40, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 1, cv2.LINE_AA)
            cv2.imshow(WIN_PREV, prog)
            cv2.waitKey(1)

            # Wait for settle
            settle_start = time.perf_counter()
            for bundle in sess.frames():
                cv2.waitKey(1)
                if bundle.locked and bundle.warp_bgr is not None:
                    last_bundle = bundle
                if time.perf_counter() - settle_start >= SETTLE_TIME:
                    break

            if last_bundle is None or last_bundle.warp_bgr is None:
                for name in curves:
                    curves[name].append(None)
                continue

            for idx, (cx, cy, area) in blob_info.items():
                name = labeler.labels[idx]
                lab = _sample_lab(last_bundle.warp_bgr, cx, cy, area)
                curves[name].append(lab)
                if lab:
                    print(f"MINI_CAL |   {name}: L={lab[0]:.1f} a={lab[1]:.1f} b={lab[2]:.1f}")
                else:
                    print(f"MINI_CAL |   {name}: sample failed")

        # ── Save profiles ────────────────────────────────────────────────────
        grid_px_final = min(
            last_bundle.warp_w / float(last_bundle.grid_w),
            last_bundle.warp_h / float(last_bundle.grid_h),
        ) if last_bundle else 50.0

        profiles = {}
        for name, lab_list in curves.items():
            valid = [l for l in lab_list if l is not None]
            if not valid:
                print(f"MINI_CAL | {name}: no valid samples, skipping")
                continue
            # Find blob info for this name
            area = next(
                (a for idx, (cx, cy, a) in blob_info.items()
                 if labeler.labels[idx] == name), grid_px_final**2)
            equiv_r = math.sqrt(area / math.pi)
            profiles[name] = {
                "lab_curve": [list(l) if l is not None else None for l in lab_list],
                "brightness_steps": BRIGHTNESS_STEPS,
                "expected_diameter_squares": float((equiv_r * 2.0) / grid_px_final),
                "max_lab_dist": DEFAULT_MAX_LAB_DIST,
                "lab": list(np.median(np.array(valid), axis=0).tolist()),
            }
            print(f"MINI_CAL | Saved '{name}' with {len(valid)} valid samples")

        _PROFILES_PATH.write_text(json.dumps(profiles, indent=2), encoding="utf-8")
        print(f"MINI_CAL | Profiles saved to {_PROFILES_PATH}")

        # Restore full brightness
        cv2.imshow(WIN_TV, _generate_calibration_image(scene_w, scene_h,
                                                        brightness=255,
                                                        base_image=scene_bg))
        cv2.waitKey(500)

    finally:
        try:
            sess.close()
        except Exception:
            pass
        try:
            cv2.destroyWindow(WIN_TV)
            cv2.destroyWindow(WIN_PREV)
        except Exception:
            pass

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Runtime helpers — imported by v3_tracking.py
# ──────────────────────────────────────────────────────────────────────────────

def load_profiles_with_curves(path: Path = _PROFILES_PATH) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"COMBO | Failed to load profiles: {e}")
        return {}


def min_lab_dist_to_profile(
    sampled_lab: Tuple[float, float, float],
    profile: Dict,
) -> float:
    def _dist(a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    curve = profile.get("lab_curve")
    if curve:
        valid = [c for c in curve if c is not None]
        if valid:
            return min(_dist(sampled_lab, tuple(c)) for c in valid)

    single = profile.get("lab")
    if single:
        return _dist(sampled_lab, tuple(single))

    return float("inf")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import asyncio
    import threading
    import foundryoutput as fo

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--camera", type=int, default=None,
                        help="Camera index to use (default: read from hardware_config.json)")
    args = parser.parse_args()

    def _run_foundry():
        asyncio.run(fo.main())
    threading.Thread(target=_run_foundry, daemon=True).start()

    success = run_mini_calibration(camera_index=args.camera)
    print("MINI_CAL | Done:", "success" if success else "cancelled")