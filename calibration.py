# calibration.py
import os
import cv2
import numpy as np
import time
import setup as s
import json
import hashlib
from app_paths import data_path, initialize_user_data

from screeninfo import get_monitors
from calibration_capture import CalibrationCaptureGate
from calibration_diagnostics import CalibrationDiagnostics
from foundry_service import validated_scene_info
from calibration_visuals import (
    choose_preview_display, close_calibration_windows, render_board_status,
    render_calibration_preview, render_charuco_board, show_calibration_preview,
    show_charuco_board,
)

try:
    import foundryoutput as fo
except Exception:
    fo = None

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 960
MARGIN_PX = 30
NUM_FRAMES = 10
_LAST_CAMERA_CALIBRATION_MESSAGE = ""

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

border_ratio = 0.1
alpha = 0.6

PROC_WARP_W = 1280
PROC_WARP_H = 720

PRINT_MODE = False

submitted_grid = {"cols": None, "rows": None}

_BLENDED_IMG_CACHE = None
_BLENDED_WINDOW_OPEN = False
BLENDED_WINDOW_NAME = "Blended ArUco Markers"

# Cache lookup is pure; runtime generation creates the required directories.
def _writable_dir() -> str:
    return str(data_path("Cache"))

BLENDED_OUTPUT_PATH = os.path.join(_writable_dir(), "blended_output.jpg")

# NEW: cache directory for blended outputs
BLENDED_CACHE_DIR = os.path.join(_writable_dir(), "blended_cache")
MARKERS_DIR = os.path.join(_writable_dir(), "markers")
_BLENDED_META_PATH = None

# NEW: window sizing helper (resize only once)
_WINDOW_SIZED = set()
def _ensure_window(name: str, w: int, h: int):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if name not in _WINDOW_SIZED:
        try:
            cv2.resizeWindow(name, int(w), int(h))
        except Exception:
            pass
        _WINDOW_SIZED.add(name)

def _get_mode() -> str:
    cfg = s.load_last_selection() or {}
    mode = (cfg.get("mode") or "self_hosted").strip().lower()
    return mode if mode in ("foundry", "self_hosted") else "self_hosted"

def _get_map_path() -> str:
    cfg = s.load_last_selection() or {}
    p = (cfg.get("map_path") or "maps/dnd1.jpg")
    # If it's just a filename or relative path, resolve via resource_path
    # so it works both in dev and inside the .app bundle.
    if not os.path.isabs(p):
        p = s.resource_path(p)
    return str(p)

def _make_blended_cache_key(map_path: str, cols: int, rows: int, screen_w: int, screen_h: int) -> str:
    try:
        st = os.stat(map_path)
        mtime = int(st.st_mtime)
        fsize = int(st.st_size)
    except Exception:
        mtime = 0
        fsize = 0
    payload = {
        "map_path": os.path.basename(map_path),
        "mtime": mtime,
        "size": fsize,
        "cols": int(cols),
        "rows": int(rows),
        "screen_w": int(screen_w),
        "screen_h": int(screen_h),
        "border_ratio": float(border_ratio),
        "alpha": float(alpha),
        "proc_w": int(PROC_WARP_W),
        "proc_h": int(PROC_WARP_H),
        "print_mode": bool(PRINT_MODE),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.md5(raw).hexdigest(), payload

def _set_blended_paths_for_key(key: str):
    global BLENDED_OUTPUT_PATH, _BLENDED_META_PATH
    os.makedirs(BLENDED_CACHE_DIR, exist_ok=True)
    BLENDED_OUTPUT_PATH = os.path.join(BLENDED_CACHE_DIR, f"blended_{key}.jpg")
    _BLENDED_META_PATH = os.path.join(BLENDED_CACHE_DIR, f"blended_{key}.json")

def _write_blended_meta(payload: dict):
    global _BLENDED_META_PATH
    if not _BLENDED_META_PATH:
        return
    try:
        with open(_BLENDED_META_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        pass


def get_blended_display_image(allow_generate=False):
    global _BLENDED_IMG_CACHE

    if _BLENDED_IMG_CACHE is not None:
        return _BLENDED_IMG_CACHE

    if os.path.exists(BLENDED_OUTPUT_PATH):
        img = cv2.imread(BLENDED_OUTPUT_PATH)
        if img is not None:
            _BLENDED_IMG_CACHE = img
            return _BLENDED_IMG_CACHE

    if allow_generate:
        try:
            generate_display()
            return _BLENDED_IMG_CACHE
        except Exception as e:
            print(f"⚠️ Could not generate blended display: {e}")
            return None

    return None


def show_blended_display_window():
    # In Foundry mode, never show this window
    if _get_mode() == "foundry":
        return

    global _BLENDED_WINDOW_OPEN
    img = get_blended_display_image(allow_generate=False)
    if img is None:
        print("⚠️ Blended display image not available.")
        return

    _ensure_window(BLENDED_WINDOW_NAME, 1280, 720)
    cv2.imshow(BLENDED_WINDOW_NAME, img)
    _BLENDED_WINDOW_OPEN = True


def hide_blended_display_window():
    global _BLENDED_WINDOW_OPEN
    if _BLENDED_WINDOW_OPEN:
        try:
            cv2.destroyWindow(BLENDED_WINDOW_NAME)
        except Exception:
            pass
        _BLENDED_WINDOW_OPEN = False


def is_blended_display_window_open():
    return bool(_BLENDED_WINDOW_OPEN)


def _foundry_wait_for_scene_grid(access=None):
    """
    Blocks until sceneInfo arrives. Cancel button or window close cancels.
    Returns (cols, rows, scene_info_dict)
    """
    if fo is None:
        raise RuntimeError("Foundry mode selected but foundryoutput.py cannot be imported.")

    print("[CAL] Foundry mode: waiting for Foundry sceneInfo...")

    from tk_camera_preview import TkFoundryWait
    import time as _t
    wait_win = TkFoundryWait()

    while True:
        if access is not None and not access.allowed:
            wait_win.close()
            access.require()
        try:
            fo.request_scene_info()
        except Exception:
            pass

        info = None
        try:
            info = fo.get_scene_params()
        except Exception:
            info = None

        ok = False
        w = h = grid_px = None
        if isinstance(info, dict):
            w = info.get("sceneW") or info.get("width")
            h = info.get("sceneH") or info.get("height")
            grid_px = info.get("gridPx") or info.get("gridSize") or info.get("grid")
            if w and h and grid_px:
                try:
                    w = int(w); h = int(h); grid_px = int(grid_px)
                    ok = (w > 0 and h > 0 and grid_px > 0)
                except Exception:
                    ok = False

        wait_win.update(ok=ok, scene_w=w or 0, scene_h=h or 0, grid_px=grid_px or 0)

        if ok:
            _t.sleep(0.5)  # let user see the success message
            wait_win.close()
            cols = int(round(w / float(grid_px)))
            rows = int(round(h / float(grid_px)))
            cols = max(1, cols)
            rows = max(1, rows)
            print(f"[CAL] Foundry sceneInfo received. cols={cols} rows={rows}")
            return cols, rows, info

        _t.sleep(0.25)


def _calibration_displays(display=None, displays=None):
    available = list(displays) if displays is not None else (
        [display] if display is not None else s.detect_displays_backup())
    if not available:
        raise ValueError("Connect a display for the calibration board.")
    if display is None:
        selected = getattr(s, "selected_display", None) or {}
        display = next((item for item in available if item.get("index") == selected.get("index")), None)
        display = display or next((item for item in available if not item.get("primary", item.get("index") == 0)), available[0])
    return dict(display), choose_preview_display(display, available)


def display_charuco(display=None, *, displays=None, board=None):
    """Generate and present the same board used by the calibration solver."""
    if board is None:
        board = cv2.aruco.CharucoBoard(
            (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)
    target, _preview = _calibration_displays(display, displays)
    return show_charuco_board(board, target, cv=cv2)


def save_camera_calibration_result(profile_id, camera_matrix, dist_coeffs, image_size, *, rms=None, profiles=None,
                                   expected_binding=None):
    """Persist one complete result to its named camera profile, never globals."""
    if profiles is None:
        from hardware_profiles import HardwareProfiles
        profiles = HardwareProfiles()
    if not profile_id:
        raise ValueError("Choose a named camera profile before calibrating.")
    return profiles.save_camera_calibration(profile_id, camera_matrix, dist_coeffs, image_size,
                                            rms=rms, expected_binding=expected_binding)


def last_camera_calibration_message():
    return _LAST_CAMERA_CALIBRATION_MESSAGE


def _save_calibration_diagnostics(diagnostics, outcome):
    global _LAST_CAMERA_CALIBRATION_MESSAGE
    _LAST_CAMERA_CALIBRATION_MESSAGE = diagnostics.summary()
    try:
        path = diagnostics.save(outcome)
        print(f"Camera calibration | Diagnostics saved to {path}", flush=True)
    except (OSError, ValueError) as exc:
        print(f"Camera calibration | Could not save diagnostics: {exc}", flush=True)


def calibrate_webcam(camera_index=None, *, profile_id=None, access=None,
                     display=None, displays=None, auto_capture=True, camera_source=None):
    global _LAST_CAMERA_CALIBRATION_MESSAGE
    _LAST_CAMERA_CALIBRATION_MESSAGE = ""
    if access is not None:
        access.require()
    from hardware_profiles import HardwareProfiles
    profiles = HardwareProfiles()
    profile = profiles.get_profile("camera", profile_id) if profile_id else profiles.active_profile("camera")
    if not profile:
        raise ValueError("Create and choose a camera profile before calibrating.")
    profile_id = profile["id"]
    diagnostics = CalibrationDiagnostics(profile_id, profile["name"], total=NUM_FRAMES)
    if camera_index is None:
        camera_index = s.load_last_selection().get("webcam_index", 0)
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    params = cv2.aruco.DetectorParameters()

    from camera_sources import (is_phone_camera_index, source_for_index,
                                open_camera_capture, profile_compatible_with_source, stable_stream_key)
    phone_source = is_phone_camera_index(camera_index)
    source_selection = ({"webcam_index": camera_index, "camera_source": camera_source}
                        if camera_source is not None else s.load_last_selection())
    source = source_for_index(camera_index, selection=source_selection)
    if not profile_compatible_with_source(profile, source):
        if phone_source:
            raise ValueError("Create and choose a Sarween Camera USB profile for this iPhone before calibrating.")
        raise ValueError("Choose a camera profile for this input before calibrating. Direct USB iPhone and native/OBS feeds need separate profiles.")
    cap = open_camera_capture(camera_index, source=source)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        cap.release()
        diagnostics.error = "The selected camera could not be opened."
        _save_calibration_diagnostics(diagnostics, "failed")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    captured = 0
    image_size = None
    board_window = preview_window = None
    gate = CalibrationCaptureGate(NUM_FRAMES, automatic=auto_capture,
                                  board_columns=SQUARES_VERTICALLY - 1,
                                  board_rows=SQUARES_HORIZONTALLY - 1)
    last_board_status = None
    timed_capture = False
    next_timed_capture = None
    failed_reads = 0
    camera_stream = None
    try:
        target, preview_display = _calibration_displays(display, displays)
        board_image = render_charuco_board(board, target["width"], target["height"])
        board_window = display_charuco(target, displays=displays, board=board)
        print("Camera calibration | Move or tilt the camera, then hold each new view still.")
        print("Camera calibration | Starts after five seconds. T: 5-second timer; C/Space: capture now; A: auto/manual; Q/Escape: cancel.")
        while captured < NUM_FRAMES:
            if access is not None:
                access.require()
            ret, frame = cap.read()
            if not ret or frame is None:
                failed_reads += 1
                if phone_source:
                    message = cap.status.get("message", "Waiting for the iPhone camera…")
                    cv2.imshow(board_window, render_board_status(board_image, message + "  Q: cancel",
                        count=captured, total=NUM_FRAMES, auto_capture=gate.automatic))
                    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
                        return None
                    continue
                if failed_reads >= 10:
                    raise ValueError("The camera feed stopped during calibration. Reconnect it and try again.")
                time.sleep(0.1)
                continue
            failed_reads = 0
            if phone_source:
                stream = stable_stream_key(cap.last_frame_metadata)
                if camera_stream is not None and stream != camera_stream:
                    raise ValueError("The iPhone stream restarted during calibration. Start a new calibration capture.")
                camera_stream = stream
            current_size = (frame.shape[1], frame.shape[0])
            if image_size is not None and current_size != image_size:
                raise ValueError("Camera resolution changed during calibration. Fix the feed settings and start again.")
            image_size = current_size

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
            charuco_corners = charuco_ids = None
            duplicate_markers = (marker_ids is not None
                                 and len(np.unique(marker_ids)) != len(marker_ids))
            if marker_ids is not None and not duplicate_markers:
                try:
                    found, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, board)
                    if not found:
                        charuco_corners = charuco_ids = None
                except cv2.error:
                    charuco_corners = charuco_ids = None
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                return None
            now = time.monotonic()
            if key in (ord('t'), ord('T')):
                timed_capture = not timed_capture
                gate.automatic = not timed_capture
                next_timed_capture = now + 5.0 if timed_capture else None
            if key in (ord('a'), ord('A')):
                timed_capture = False
                next_timed_capture = None
                gate.automatic = not gate.automatic
            timer_due = timed_capture and now >= next_timed_capture
            state = gate.observe(now, gray, charuco_corners, charuco_ids, current_size,
                                 manual=key in (ord('c'), ord('C'), 32) or timer_due)
            state["captureMode"] = "timed" if timed_capture else "auto" if gate.automatic else "manual"
            if timed_capture and state["reason"] == "manual":
                state["reason"] = "timed_wait"
                state["message"] = "Timer running. Move to a new angle; each frame is still checked for clarity."
            if timed_capture and (timer_due or state["capture"]):
                next_timed_capture = now + 5.0
            if state["capture"]:
                # A planar homography is useful for judging camera motion, but
                # real lens distortion bends that plane in the image. Retain
                # every valid detected corner so the solver can estimate it.
                all_charuco_corners.append(charuco_corners.copy())
                all_charuco_ids.append(charuco_ids.copy())
                captured += 1
                print(f"Camera calibration | Captured {captured}/{NUM_FRAMES}")
            detected_corners = 0 if charuco_ids is None else len(charuco_ids)
            diagnostics.observe(state, detected_corners=detected_corners, frame_size=current_size,
                                automatic=gate.automatic, duplicate_markers=duplicate_markers)
            message = ("Show only the calibration board; cover other ArUco markers."
                       if duplicate_markers else state["message"])
            countdown = None
            if timed_capture:
                countdown = max(0.0, next_timed_capture - now)
            elif state["reason"] == "starting":
                countdown = gate.start_delay * (1 - state["startProgress"])
            elif state["reason"] == "holding":
                countdown = gate.hold_seconds * (1 - state["stableProgress"])
            board_status = (message, captured, state["captureMode"],
                            round(countdown, 1) if countdown is not None else None,
                            detected_corners, state.get("poseCorners", state.get("inlierCorners")), state.get("holdResets"),
                            round(state.get("motionRatio") or 0.0, 3), state.get("sharedCorners"),
                            round(state.get("sharpness") or 0.0))
            if board_status != last_board_status:
                cv2.imshow(board_window, render_board_status(board_image, message, count=captured,
                    total=NUM_FRAMES, auto_capture=gate.automatic, countdown=countdown,
                    diagnostics=state, detected_corners=detected_corners))
                last_board_status = board_status
            preview = frame.copy()
            if marker_ids is not None and not duplicate_markers:
                cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
            preview = render_calibration_preview(preview, captured=captured, required=NUM_FRAMES,
                detected_corners=detected_corners, diagnostics=state,
                auto_capture=gate.automatic, prompt=message, countdown=countdown)
            if preview_window is None and preview_display is not None:
                preview_window = show_calibration_preview(preview, preview_display, board_display=target, cv=cv2)
            elif preview_window is not None:
                cv2.imshow(preview_window, preview)
    except BaseException as exc:
        diagnostics.error = str(exc) or type(exc).__name__
        raise
    finally:
        cap.release()
        close_calibration_windows(board_window, preview_window, cv=cv2)
        _save_calibration_diagnostics(diagnostics, "failed" if diagnostics.error else
                                      "captured" if captured == NUM_FRAMES else "cancelled")

    print("🔧 Calibrating camera...")
    try:
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None)
        if access is not None:
            access.require()
        saved_profile = save_camera_calibration_result(
            profile_id, camera_matrix, dist_coeffs, image_size, rms=float(retval), profiles=profiles,
            expected_binding=profile)
    except BaseException as exc:
        diagnostics.error = str(exc) or type(exc).__name__
        _save_calibration_diagnostics(diagnostics, "failed")
        raise
    _save_calibration_diagnostics(diagnostics, "saved")
    print(f"✅ Calibration complete. Saved camera profile: {saved_profile['name']} ({image_size[0]}×{image_size[1]}).")

    return saved_profile


def add_white_border(img, border_thickness):
    value = 255 if len(img.shape) == 2 else (255, 255, 255)
    return cv2.copyMakeBorder(
        img, border_thickness, border_thickness, border_thickness, border_thickness,
        borderType=cv2.BORDER_CONSTANT, value=value
    )


def generate_grid(img, grid_cols, grid_rows, color=(255, 0, 0), thickness=2, label_cells=True):
    img_with_grid = img.copy()
    h, w = img.shape[:2]

    col_width = w / grid_cols
    row_height = h / grid_rows

    cell_px = float(min(col_width, row_height))
    thickness = max(1, int(round(cell_px * 0.02)))
    font_scale = max(0.3, cell_px * 0.005)
    text_thickness = max(1, int(round(cell_px * 0.005)))

    pad_x = max(4, int(round(cell_px * 0.08)))
    pad_y = max(6, int(round(cell_px * 0.18)))

    for i in range(1, grid_cols):
        x = int(i * col_width)
        cv2.line(img_with_grid, (x, 0), (x, h), color, thickness)

    for j in range(1, grid_rows):
        y = int(j * row_height)
        cv2.line(img_with_grid, (0, y), (w, y), color, thickness)

    if label_cells:
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_x = int(col * col_width + pad_x)
                cell_y = int(row * row_height + pad_y)
                label = (
                    f"{chr(65 + col)}{row + 1}"
                    if col < 26
                    else f"{chr(65 + (col // 26) - 1)}{chr(65 + (col % 26))}{row + 1}"
                )
                cv2.putText(img_with_grid, label, (cell_x, cell_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), text_thickness, cv2.LINE_AA)

    return img_with_grid


def generate_grid_with_sliders(img):
    window_name = "Interactive Grid"
    _ensure_window(window_name, 1280, 720)

    submitted_grid = {"cols": None, "rows": None}
    init_cols, init_rows = 23, 16
    max_val = 50
    running = [True]

    h, w = img.shape[:2]
    submit_btn = (w - 200, h - 60, w - 100, h - 30)
    cancel_btn = (w - 90, h - 60, w - 10, h - 30)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if submit_btn[0] <= x <= submit_btn[2] and submit_btn[1] <= y <= submit_btn[3]:
                submitted_grid["cols"] = cv2.getTrackbarPos("Cols", window_name)
                submitted_grid["rows"] = cv2.getTrackbarPos("Rows", window_name)
                running[0] = False
            elif cancel_btn[0] <= x <= cancel_btn[2] and cancel_btn[1] <= y <= cancel_btn[3]:
                submitted_grid["cols"] = None
                submitted_grid["rows"] = None
                running[0] = False

    cv2.setMouseCallback(window_name, on_mouse)
    cv2.createTrackbar("Cols", window_name, init_cols, max_val, lambda x: None)
    cv2.createTrackbar("Rows", window_name, init_rows, max_val, lambda x: None)

    while running[0]:
        cols = max(1, cv2.getTrackbarPos("Cols", window_name))
        rows = max(1, cv2.getTrackbarPos("Rows", window_name))

        grid_img = generate_grid(img, cols, rows)
        cv2.rectangle(grid_img, submit_btn[:2], submit_btn[2:], (0, 255, 0), -1)
        cv2.putText(grid_img, "Submit", (submit_btn[0] + 10, submit_btn[3] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(grid_img, cancel_btn[:2], cancel_btn[2:], (0, 0, 255), -1)
        cv2.putText(grid_img, "Cancel", (cancel_btn[0] + 10, cancel_btn[3] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, grid_img)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            running[0] = False
            break

        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            running[0] = False

    cv2.destroyWindow(window_name)
    return submitted_grid["cols"], submitted_grid["rows"]


def _compute_cell_px(map_w, map_h, cols, rows):
    cell_w = map_w / float(cols)
    cell_h = map_h / float(rows)
    return max(10, int(round(min(cell_w, cell_h))))


def _compute_border_px(cell_px):
    return max(1, int(round(cell_px * border_ratio)))


def make_aruco_marker_tile(marker_id, cell_px, border_px):
    inner_px = max(10, cell_px - border_px)

    border_map = {
        0: (0,         border_px, 0,         border_px),
        1: (0,         border_px, border_px, 0),
        2: (border_px, 0,         border_px, 0),
        3: (border_px, 0,         0,         border_px),
    }

    try:
        marker_gray = cv2.aruco.generateImageMarker(aruco_dict, marker_id, inner_px)
    except AttributeError:
        marker_gray = np.zeros((inner_px, inner_px), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, marker_id, inner_px, marker_gray, 1)

    marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
    t, b, l, r = border_map[marker_id]
    tile = cv2.copyMakeBorder(marker_bgr, t, b, l, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    if tile.shape[0] != cell_px or tile.shape[1] != cell_px:
        tile = cv2.resize(tile, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
    return tile


def generate_aruco_marker_tiles(cell_px, border_px):
    os.makedirs(MARKERS_DIR, exist_ok=True)
    for i in range(4):
        tile = make_aruco_marker_tile(i, cell_px, border_px)
        cv2.imwrite(os.path.join(MARKERS_DIR, f"marker_{i}.png"), tile)


def prepare_map_asset(access=None, *, scene_info=None):
    mode = _get_mode()

    if mode == "foundry":
        if scene_info is None:
            cols, rows, info = _foundry_wait_for_scene_grid(access)
        else:
            if access is not None:
                access.require()
            info = validated_scene_info(scene_info)
            if info is None:
                raise ValueError("Foundry setup needs a connected scene with valid dimensions and grid size.")
            cols = max(1, int(round(info["sceneW"] / float(info["gridPx"]))))
            rows = max(1, int(round(info["sceneH"] / float(info["gridPx"]))))
        s.grid_cols = cols
        s.grid_rows = rows
        return None, None, None, cols, rows, None, None

    map_path = _get_map_path()
    background_img = cv2.imread(map_path)
    if background_img is None:
        raise FileNotFoundError(f"Could not load {map_path}")

    cols, rows = generate_grid_with_sliders(background_img)

    s.grid_cols = cols
    s.grid_rows = rows

    if getattr(s, "selected_display", None):
        screen_width = int(s.selected_display["width"])
        screen_height = int(s.selected_display["height"])
    else:
        screen_width = 1280
        screen_height = 720

    # NEW: configure cache paths for this run (self-hosted only)
    if mode == "self_hosted":
        key, meta = _make_blended_cache_key(map_path, cols, rows, screen_width, screen_height)
        _set_blended_paths_for_key(key)

        if os.path.exists(BLENDED_OUTPUT_PATH):
            img = cv2.imread(BLENDED_OUTPUT_PATH)
            if img is not None:
                global _BLENDED_IMG_CACHE
                _BLENDED_IMG_CACHE = img
                return img, img.shape[0], img.shape[1], cols, rows, None, None

    background_img = generate_grid(background_img, cols, rows)

    bg_h, bg_w = background_img.shape[:2]
    if PRINT_MODE:
        resized_bg = background_img.copy()
    else:
        scale = min(screen_width / bg_w, screen_height / bg_h)
        resized_bg = cv2.resize(background_img, (int(bg_w * scale), int(bg_h * scale)))

    map_h, map_w = resized_bg.shape[:2]
    cell_px = _compute_cell_px(map_w, map_h, cols, rows)
    border_px = _compute_border_px(cell_px)

    canvas = add_white_border(resized_bg, border_px)

    generate_aruco_marker_tiles(cell_px, border_px)

    img_h, img_w = canvas.shape[:2]

    s.display_img_w = int(img_w)
    s.display_img_h = int(img_h)

    scale_p = min(PROC_WARP_W / float(img_w), PROC_WARP_H / float(img_h))
    s.warp_w = int(round(img_w * scale_p))
    s.warp_h = int(round(img_h * scale_p))

    return canvas, img_h, img_w, cols, rows, cell_px, border_px


def generate_display():
    global _BLENDED_IMG_CACHE, _BLENDED_WINDOW_OPEN

    if _get_mode() == "foundry":
        return

    blended_img, img_h, img_w, cols, rows, cell_px, border_px = prepare_map_asset()

    # If prepare_map_asset() returned a cached image, it returned cell_px/border_px as None.
    # In that case, just show and return.
    if cell_px is None or border_px is None:
        _ensure_window(BLENDED_WINDOW_NAME, 1280, 720)
        cv2.imshow(BLENDED_WINDOW_NAME, blended_img)
        _BLENDED_WINDOW_OPEN = True
        return

    map_x0 = border_px
    map_y0 = border_px
    map_x1 = img_w - border_px
    map_y1 = img_h - border_px

    placements = [
        (0, map_x0,             map_y0),
        (1, map_x1 - cell_px,   map_y0),
        (2, map_x1 - cell_px,   map_y1 - cell_px),
        (3, map_x0,             map_y1 - cell_px),
    ]

    for i, x, y in placements:
        tile = cv2.imread(os.path.join(MARKERS_DIR, f"marker_{i}.png"), cv2.IMREAD_COLOR)
        if tile is None:
            continue

        if tile.shape[0] != cell_px or tile.shape[1] != cell_px:
            tile = cv2.resize(tile, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)

        roi = blended_img[y:y + cell_px, x:x + cell_px]
        if roi.shape[0] != cell_px or roi.shape[1] != cell_px:
            continue

        roi_f = roi.astype(np.float32)
        tile_f = tile.astype(np.float32)

        white_mask = np.all(tile >= 250, axis=2)
        nonwhite_mask = ~white_mask

        out = roi_f.copy()
        out[white_mask] = tile_f[white_mask]
        out[nonwhite_mask] = (1.0 - alpha) * roi_f[nonwhite_mask] + alpha * tile_f[nonwhite_mask]

        blended_img[y:y + cell_px, x:x + cell_px] = out.astype(np.uint8)

    _BLENDED_IMG_CACHE = blended_img
    cv2.imwrite(BLENDED_OUTPUT_PATH, blended_img)

    # Write cache metadata if available
    try:
        map_path = _get_map_path()
        if getattr(s, "selected_display", None):
            screen_width = int(s.selected_display["width"])
            screen_height = int(s.selected_display["height"])
        else:
            screen_width = 1280
            screen_height = 720
        key, meta = _make_blended_cache_key(map_path, cols, rows, screen_width, screen_height)
        _set_blended_paths_for_key(key)
        _write_blended_meta(meta)
        cv2.imwrite(BLENDED_OUTPUT_PATH, blended_img)
    except Exception:
        pass

    _ensure_window(BLENDED_WINDOW_NAME, 1280, 720)
    cv2.imshow(BLENDED_WINDOW_NAME, blended_img)
    _BLENDED_WINDOW_OPEN = True


def calibrate(access=None, *, scene_info=None):
    initialize_user_data()
    mode = _get_mode()

    if mode == "foundry":
        prepare_map_asset(access, scene_info=scene_info)
    else:
        generate_display()

    print("✅ Calibration Complete.")
