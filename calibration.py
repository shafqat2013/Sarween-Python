# calibration.py
import os
import cv2
import numpy as np
import time
import setup as s
import json
import hashlib

from screeninfo import get_monitors

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
BLENDED_OUTPUT_PATH = "blended_output.jpg"

# NEW: cache directory for blended outputs
BLENDED_CACHE_DIR = "blended_cache"
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


def _foundry_wait_for_scene_grid():
    """
    Blocks until sceneInfo arrives. ESC cancels.
    Returns (cols, rows, scene_info_dict)
    """
    if fo is None:
        raise RuntimeError("Foundry mode selected but foundryoutput.py cannot be imported.")

    print("[CAL] Foundry mode: waiting for Foundry sceneInfo... (press ESC in this window to cancel)")

    name = "Foundry Wait"
    _ensure_window(name, 520, 120)

    while True:
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

        canvas = np.zeros((120, 520, 3), dtype=np.uint8)
        msg = "Waiting for Foundry scene info..."
        if not ok:
            sub = "Make sure: foundryoutput.py server running + module connected."
        else:
            sub = f"Got scene {w}x{h}, grid {grid_px}px"

        cv2.putText(canvas, msg, (16, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(canvas, sub, (16, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

        cv2.imshow(name, canvas)
        key = cv2.waitKey(250) & 0xFF
        if key == 27:
            cv2.destroyWindow(name)
            raise SystemExit("Cancelled waiting for Foundry scene info.")

        if ok:
            cols = int(round(w / float(grid_px)))
            rows = int(round(h / float(grid_px)))
            cols = max(1, cols)
            rows = max(1, rows)
            cv2.destroyWindow(name)
            print(f"[CAL] Foundry sceneInfo received. cols={cols} rows={rows}")
            return cols, rows, info


def display_charuco():
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(
        board, (LENGTH_PX, int(LENGTH_PX * size_ratio)), marginSize=MARGIN_PX
    )

    monitors = get_monitors()
    if len(monitors) < 2:
        print("⚠️ Only one display detected. Showing on primary display.")
    cv2.namedWindow("CharucoBoard", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("CharucoBoard", img)
    cv2.waitKey(1)


def calibrate_webcam(camera_index=s.load_last_selection().get("webcam_index", 0)):
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    used_frames = []
    captured = 0

    print("📸 Press 'c' to capture a frame with a visible ChArUco board.")
    print("📷 Need 10 captures total. Press 'q' to quit early.")

    while captured < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        _ensure_window("Capture Preview", 1280, 720)
        cv2.imshow("Capture Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            if marker_ids is not None and len(marker_ids) > 0:
                charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, board
                )
                if charuco_retval:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    used_frames.append(frame.copy())
                    captured += 1
                    print(f"✅ Captured frame {captured}/10")
                else:
                    print("⚠️ Not enough ChArUco corners found. Try again.")
            else:
                print("⚠️ No ArUco markers detected. Try again.")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    print("🔧 Calibrating camera...")
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None
    )

    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("✅ Calibration complete. Saved camera_matrix.npy and dist_coeffs.npy.")

    for i, frame in enumerate(used_frames):
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)
        combined = np.hstack((frame, undistorted))
        _ensure_window(f"Frame {i+1}: Original (L) | Undistorted (R)", 1280, 720)
        cv2.imshow(f"Frame {i+1}: Original (L) | Undistorted (R)", combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


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


def generate_aruco_marker_tiles(cell_px, border_px):
    os.makedirs("markers", exist_ok=True)
    inner_px = max(10, cell_px - border_px)

    border_map = {
        0: (0,         border_px, 0,         border_px),
        1: (0,         border_px, border_px, 0),
        2: (border_px, 0,         border_px, 0),
        3: (border_px, 0,         0,         border_px),
    }

    for i in range(4):
        try:
            marker_gray = cv2.aruco.generateImageMarker(aruco_dict, i, inner_px)
        except AttributeError:
            marker_gray = np.zeros((inner_px, inner_px), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, i, inner_px, marker_gray, 1)

        marker_bgr = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        t, b, l, r = border_map[i]
        tile = cv2.copyMakeBorder(marker_bgr, t, b, l, r, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        if tile.shape[0] != cell_px or tile.shape[1] != cell_px:
            tile = cv2.resize(tile, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(f"markers/marker_{i}.png", tile)


def prepare_map_asset():
    mode = _get_mode()

    map_path = _get_map_path()
    background_img = cv2.imread(map_path)
    if background_img is None:
        raise FileNotFoundError(f"Could not load {map_path}")

    if mode == "foundry":
        cols, rows, info = _foundry_wait_for_scene_grid()
    else:
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
        tile = cv2.imread(f"markers/marker_{i}.png", cv2.IMREAD_COLOR)
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


def calibrate():
    mode = _get_mode()

    if mode == "foundry":
        prepare_map_asset()
    else:
        generate_display()

    print("✅ Calibration Complete.")