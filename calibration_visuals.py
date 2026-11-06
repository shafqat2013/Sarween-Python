"""ChArUco presentation and calibration previews, with no import-time windows."""

import math
import numbers
import sys

import cv2
import numpy as np


BOARD_WINDOW_NAME = "CharucoBoard"
PREVIEW_WINDOW_NAME = "Camera calibration"
MAX_CANVAS_PIXELS = 64 * 1024 * 1024
MAX_CANVAS_DIMENSION = 16384


def position_own_macos_window(window_name, x, y, *, appkit=None, cv=cv2):
    """Position this process's titled window in logical global top-left coordinates."""
    x = _integer(x, "Window x")
    y = _integer(y, "Window y")
    if appkit is None:
        import AppKit as appkit
    screens = list(appkit.NSScreen.screens())
    if not screens:
        raise RuntimeError("No macOS display is available for the calibration window.")
    primary_height = float(screens[0].frame().size.height)
    if not math.isfinite(primary_height) or primary_height <= 0:
        raise RuntimeError("macOS did not provide valid calibration display coordinates.")
    application = appkit.NSApplication.sharedApplication()
    for attempt in range(6):
        for window in application.windows():
            try:
                title = str(window.title())
                visible = getattr(window, "isVisible", None)
                if visible is not None and not visible():
                    continue
            except Exception:
                continue
            if title == window_name:
                try:
                    window.setFrameTopLeftPoint_((float(x), primary_height - y))
                except Exception as exc:
                    raise RuntimeError(f"Could not position the calibration window '{window_name}' on the selected display.") from exc
                return
        if attempt < 5:
            cv.waitKey(1)
    raise RuntimeError(f"Could not position the calibration window '{window_name}' on the selected display.")


def _position_window(window_name, x, y, *, cv, native_positioner=None):
    if native_positioner is not None:
        native_positioner(window_name, x, y)
    elif sys.platform == "darwin" and cv is cv2:
        # cvMoveWindow uses the window's current screen as its y reference on
        # Cocoa. The same-process AppKit window uses a stable global reference.
        position_own_macos_window(window_name, x, y, cv=cv)
    else:
        cv.moveWindow(window_name, x, y)


def _integer(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, numbers.Real)
            or not math.isfinite(value) or int(value) != value or (positive and value <= 0)):
        raise ValueError(f"{label} must be {'a positive' if positive else 'an'} integer.")
    return int(value)


def _display_rect(display):
    if not isinstance(display, dict):
        raise ValueError("Choose a display for the calibration board.")
    return (_integer(display.get("x", 0), "Display x"),
            _integer(display.get("y", 0), "Display y"),
            _integer(display.get("width"), "Display width", positive=True),
            _integer(display.get("height"), "Display height", positive=True))


def render_charuco_board(board, screen_width, screen_height, *, board_fraction=0.85):
    """Fit the shared board at one scale, retaining square cells and a white border."""
    width = _integer(screen_width, "Display width", positive=True)
    height = _integer(screen_height, "Display height", positive=True)
    if max(width, height) > MAX_CANVAS_DIMENSION or width * height > MAX_CANVAS_PIXELS:
        raise ValueError("The calibration display is too large to render safely.")
    if (isinstance(board_fraction, bool) or not isinstance(board_fraction, numbers.Real)
            or not math.isfinite(board_fraction) or not 0.1 <= board_fraction <= 0.95):
        raise ValueError("The board must leave a white border around the pattern.")
    columns, rows = board.getChessboardSize()
    columns = _integer(columns, "Board columns", positive=True)
    rows = _integer(rows, "Board rows", positive=True)
    cell_size = int(min(width * board_fraction / columns, height * board_fraction / rows))
    if cell_size < 24:
        raise ValueError("The selected display is too small for a readable ChArUco board.")
    board_width, board_height = columns * cell_size, rows * cell_size
    pattern = board.generateImage((board_width, board_height), marginSize=0, borderBits=1)
    if not isinstance(pattern, np.ndarray) or pattern.shape != (board_height, board_width):
        raise ValueError("The ChArUco board did not produce the expected image size.")
    canvas = np.full((height, width), 255, dtype=np.uint8)
    left, top = (width - board_width) // 2, (height - board_height) // 2
    canvas[top:top + board_height, left:left + board_width] = pattern
    return canvas


def show_charuco_board(board, display, *, cv=cv2, window_name=BOARD_WINDOW_NAME,
                       native_positioner=None):
    """Open the generated board only when explicitly called by calibration."""
    x, y, width, height = _display_rect(display)
    image = render_charuco_board(board, width, height)
    created = False
    try:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        created = True
        # Cocoa creates the native content on first imshow. Moving it before
        # that point can lose negative origins, and fullscreen picks its screen.
        cv.imshow(window_name, image)
        cv.waitKey(1)
        cv.resizeWindow(window_name, width, height)
        _position_window(window_name, x, y, cv=cv, native_positioner=native_positioner)
        cv.waitKey(1)
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow(window_name, image)
        cv.waitKey(1)
    except Exception:
        if created:
            close_calibration_windows(window_name, cv=cv)
        raise
    return window_name


def _overlap(first, second):
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    return x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1


def choose_preview_display(board_display, displays):
    """A fullscreen board keeps its display; a preview needs a separate surface."""
    board_rect = _display_rect(board_display)
    for display in displays:
        if not _overlap(board_rect, _display_rect(display)):
            return dict(display)
    return None


def show_calibration_preview(frame, preview_display, *, board_display,
                             cv=cv2, window_name=PREVIEW_WINDOW_NAME, native_positioner=None):
    """Never place a camera preview over the board, including one-monitor setups."""
    if preview_display is None:
        return None
    rect = _display_rect(preview_display)
    if _overlap(rect, _display_rect(board_display)):
        return None
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or min(frame.shape[:2]) <= 0:
        raise ValueError("The calibration preview needs a camera image.")
    x, y, screen_width, screen_height = rect
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 720 / height,
                screen_width * 0.85 / width, screen_height * 0.85 / height)
    window_width, window_height = max(1, int(width * scale)), max(1, int(height * scale))
    created = False
    try:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)
        created = True
        cv.imshow(window_name, frame)
        cv.waitKey(1)
        cv.resizeWindow(window_name, window_width, window_height)
        _position_window(window_name, x + (screen_width - window_width) // 2,
                         y + (screen_height - window_height) // 2,
                         cv=cv, native_positioner=native_positioner)
        cv.imshow(window_name, frame)
    except Exception:
        if created:
            close_calibration_windows(window_name, cv=cv)
        raise
    return window_name


def close_calibration_windows(*names, cv=cv2):
    """Close owned windows only; keep Foundry and other Sarween views intact."""
    names = names or (BOARD_WINDOW_NAME, PREVIEW_WINDOW_NAME)
    for name in dict.fromkeys(names):
        if not name:
            continue
        try:
            cv.destroyWindow(name)
        except Exception:
            pass


def _finite_metric(value):
    return float(value) if isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value) else None


def calibration_mode_label(auto_capture=False, diagnostics=None):
    mode = (diagnostics or {}).get("captureMode")
    if mode == "timed":
        return "TIMED (5s)"
    if mode == "manual":
        return "MANUAL"
    return "AUTO" if mode == "auto" or auto_capture else "MANUAL"


def calibration_diagnostic_lines(detected_corners, *, minimum_corners=10, diagnostics=None):
    """Short labels expose the exact capture blocker without camera or UI access."""
    diagnostics = diagnostics or {}
    detected = max(0, int(detected_corners or 0))
    minimum = max(1, int(minimum_corners))
    reason = diagnostics.get("reason")
    titles = {"no_board": "No board detected", "corners": f"{minimum} corners needed",
              "blur": "Too blurry", "coverage": "Show more of the board", "holding": "Hold steady",
              "moving": "Hold steady", "motion": "Hold steady", "starting": "Get ready",
              "duplicate": "Move to a new angle", "angle": "Move to a new angle",
              "cooldown": "Move to a new angle", "captured": "View captured",
              "complete": "All views captured", "manual": "Ready for manual capture",
              "frame_size": "Camera size changed", "timed_wait": "Timer running"}
    valid = _finite_metric(diagnostics.get("validCorners"))
    valid = max(0, int(valid)) if valid is not None else detected
    title = titles.get(reason) or ("No board detected" if valid == 0 else
                                  f"{minimum} corners needed" if valid < minimum else "Board detected")
    if reason == "manual" and diagnostics.get("captureMode") == "timed":
        title = "Timer running"
    corners = f"Calibration corners: {valid}/{minimum} needed"
    if valid != detected:
        corners += f" | detected: {detected}"
    lines = [title, corners]
    progress = _finite_metric(diagnostics.get("stableProgress"))
    motion = _finite_metric(diagnostics.get("motionRatio"))
    shared = _finite_metric(diagnostics.get("sharedCorners"))
    resets = _finite_metric(diagnostics.get("holdResets"))
    pose = _finite_metric(diagnostics.get("poseCorners"))
    metrics = []
    if progress is not None:
        metrics.append(f"Steady: {min(100, max(0, round(progress * 100)))}%")
    if motion is not None:
        metrics.append(f"Motion: {motion * 100:.1f}%")
    if shared is not None:
        metrics.append(f"Shared: {int(shared)}")
    if pose is not None:
        metrics.append(f"Pose model: {int(pose)}")
    if resets is not None:
        metrics.append(f"Hold resets: {int(resets)}")
    if metrics:
        lines.append(" | ".join(metrics))
    sharpness = _finite_metric(diagnostics.get("sharpness"))
    if sharpness is not None:
        lines.append(f"Sharpness: {sharpness:.0f}")
    return lines


def render_calibration_preview(frame, *, captured, required, detected_corners=0,
                               auto_capture=False, prompt="", countdown=None, status="",
                               diagnostics=None, minimum_corners=10):
    """Readable progress over a copy, leaving the actual calibration frame untouched."""
    if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3) or min(frame.shape[:2]) <= 0:
        raise ValueError("The calibration preview needs a camera image.")
    height, width = frame.shape[:2]
    scale = min(1.0, 960 / width, 720 / height)
    image = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = image.copy()
    width = image.shape[1]
    mode = calibration_mode_label(auto_capture, diagnostics)
    diagnostic_lines = calibration_diagnostic_lines(detected_corners, minimum_corners=minimum_corners,
                                                   diagnostics=diagnostics)
    heading = diagnostic_lines[0]
    lines = [f"Views: {captured}/{required} | {mode}", *diagnostic_lines[1:]]
    if countdown is not None:
        lines.append(f"Next capture in {max(0, countdown):.1f}s" if mode == "TIMED (5s)" else
                     f"Capture in {max(0, countdown):.1f}s - hold still")
    detail = status or prompt or (diagnostics or {}).get("message")
    if detail:
        lines.append(str(detail))
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, min(0.7, max(0.5, width / 1400)), 1
    wrapped = []
    for line in lines:
        current = ""
        for word in line.replace("·", "|").split():
            candidate = f"{current} {word}".strip()
            if current and cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] > width - 24:
                wrapped.append(current)
                current = word
            else:
                current = candidate
        if current:
            wrapped.append(current)
    line_height = max(23, cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] + 12)
    heading_scale = min(1.05, max(0.65, width / 1000))
    while heading_scale > 0.4 and cv2.getTextSize(heading, font, heading_scale, 2)[0][0] > width - 24:
        heading_scale -= 0.05
    heading_height = cv2.getTextSize("Ag", font, heading_scale, 2)[0][1] + 18
    overlay_height = min(image.shape[0], heading_height + 8 + line_height * len(wrapped))
    image[:overlay_height] = (24, 24, 24)
    cv2.putText(image, heading, (12, heading_height - 8), font, heading_scale,
                (255, 230, 120), 2, cv2.LINE_AA)
    for index, line in enumerate(wrapped):
        baseline = heading_height + 20 + index * line_height
        if baseline > overlay_height - 4:
            break
        cv2.putText(image, line, (12, baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    controls = "T 5s timer | A auto | C/SPACE now | Q/ESC cancel"
    controls_scale = min(0.55, max(0.3, width / 1800))
    cv2.rectangle(image, (0, image.shape[0] - 28), (width, image.shape[0]), (24, 24, 24), -1)
    cv2.putText(image, controls, (10, image.shape[0] - 9), font, controls_scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return image


def render_board_status(base_image, status, *, count, total, auto_capture=True, countdown=None,
                        diagnostics=None, detected_corners=None, minimum_corners=10):
    """Put progress below the board without changing its position or white quiet gap."""
    if (not isinstance(base_image, np.ndarray) or base_image.ndim not in (2, 3)
            or min(base_image.shape[:2]) <= 0):
        raise ValueError("Board status needs the original board image.")
    image = base_image.copy()
    height, width = image.shape[:2]
    dark = base_image < 128
    if dark.ndim == 3:
        dark = np.any(dark, axis=2)
    occupied_rows = np.flatnonzero(np.any(dark, axis=1))
    if not len(occupied_rows):
        return image
    quiet_gap = max(4, min(12, int(height * 0.01)))
    top = int(occupied_rows[-1]) + 1 + quiet_gap
    bottom = height - max(4, int(height * 0.006))
    available = bottom - top
    mode = calibration_mode_label(auto_capture, diagnostics)
    heading = f"{count}/{total} views saved | {mode}"
    diagnostic_lines = []
    if diagnostics is not None or detected_corners is not None:
        diagnostic_lines = calibration_diagnostic_lines(detected_corners if detected_corners is not None else
            (diagnostics or {}).get("detectedCorners", 0), minimum_corners=minimum_corners, diagnostics=diagnostics)
        heading += " | " + diagnostic_lines[1]
    if countdown is not None:
        heading += (f" | Next capture in {max(0, countdown):.1f}s" if mode == "TIMED (5s)" else
                    f" | capture in {max(0, countdown):.1f}s")
    lines = [heading]
    detail = " | ".join([*diagnostic_lines[:1], *diagnostic_lines[2:], str(status or "")]).strip(" |")
    if detail:
        lines.append(detail.replace("·", "|").replace("×", "x"))
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(0.85, max(0.3, width / 2400))
    while font_scale > 0.3:
        size, baseline = cv2.getTextSize("Ag", font, font_scale, 1)
        line_height = size[1] + baseline + 5
        if line_height * len(lines) - 5 <= available:
            break
        font_scale = max(0.3, font_scale - 0.05)
    size, baseline = cv2.getTextSize("Ag", font, font_scale, 1)
    line_height = size[1] + baseline + 5
    slots = max(0, (available + 5) // line_height)
    color = 0 if image.ndim == 2 else (0, 0, 0)
    footer = image[top:bottom]
    for index, line in enumerate(lines[:slots]):
        while line and cv2.getTextSize(line, font, font_scale, 1)[0][0] > width - 24:
            line = line[:-1]
        if not line:
            continue
        text_width = cv2.getTextSize(line, font, font_scale, 1)[0][0]
        y = size[1] + index * line_height
        cv2.putText(footer, line, ((width - text_width) // 2, y), font, font_scale, color, 1, cv2.LINE_AA)
    return image
