# setup.py
# all functions for hardware detection and selection
# ===========================
import subprocess
import cv2
import numpy as np
import os
import re
import json
import time
import tkinter as tk
from tkinter import ttk
from screeninfo import get_monitors
from app_paths import atomic_write_json, data_path, initialize_user_data, resource_path, saved_map_path


def _get_persistent_root() -> tk.Tk:
    if not hasattr(_get_persistent_root, "_root"):
        root = tk.Tk()
        root.withdraw()
        root.title("Sarween")
        _get_persistent_root._root = root
    return _get_persistent_root._root

# ===========================
# CONFIGURATION
# ===========================
CONFIG_FILE = str(data_path("hardware_config.json"))
MAPS_DIR = resource_path("maps")
MAP_EXTS = (".jpg", ".jpeg", ".png", ".webp")

MODE_SELF_HOSTED = "self_hosted"
MODE_FOUNDRY = "foundry"
MODE_LABELS = {
    MODE_SELF_HOSTED: "Self Hosted",
    MODE_FOUNDRY: "Foundry",
}
MODE_LABEL_TO_VALUE = {v: k for k, v in MODE_LABELS.items()}


def _list_map_files():
    try:
        if not os.path.isdir(MAPS_DIR):
            return []
        files = []
        for fn in sorted(os.listdir(MAPS_DIR)):
            if fn.startswith("."):
                continue
            if fn.lower().endswith(MAP_EXTS):
                files.append(os.path.join(MAPS_DIR, fn))
        return files
    except Exception:
        return []


def load_last_selection():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get("map_path"):
                    data["map_path"] = saved_map_path(data["map_path"])
                return data if isinstance(data, dict) else {}
        except Exception:
            pass
    return {}


def save_last_selection(display_index=None, webcam_index=None, mode=None, map_path=None):
    data = load_last_selection()
    if display_index is not None:
        data["display_index"] = int(display_index)
    if webcam_index is not None:
        data["webcam_index"] = int(webcam_index)
    if mode is not None:
        data["mode"] = str(mode)
    if map_path is not None:
        data["map_path"] = str(map_path)
    atomic_write_json(CONFIG_FILE, data)


config_data = load_last_selection()
last_display_index = config_data.get("display_index")
last_webcam_device_index = config_data.get("webcam_index")
last_mode = config_data.get("mode", MODE_SELF_HOSTED)
last_map_path = config_data.get("map_path")

# ===========================
# HARDWARE DETECTION
# ===========================
selected_display = None
selected_webcam = None
selected_mode = None


def detect_setup():
    displays = []
    webcams = []

    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        resolutions = re.findall(r"Resolution:\s+(\d+) x (\d+)", result.stdout)
        for i, (w, h) in enumerate(resolutions):
            displays.append({
                "index": i,
                "width": int(w),
                "height": int(h),
                "x": 0,
                "y": 0
            })

    except Exception:
        displays = detect_displays_backup()

    if not displays:
        displays = detect_displays_backup()
    webcams = detect_webcams_backup()
    return displays, webcams


def detect_displays_backup():
    displays = []
    for i, monitor in enumerate(get_monitors()):
        displays.append({
            "index": i,
            "width": monitor.width,
            "height": monitor.height,
            "x": monitor.x,
            "y": monitor.y
        })
    return displays


def detect_webcams_backup():
    webcams = []
    for i in range(6):
        cap = cv2.VideoCapture(i)
        try:
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                webcams.append({
                    "index": i,
                    "model": f"Webcam {i}",
                    "resolution": f"{width}x{height}"
                })
        finally:
            cap.release()
    return webcams


# ===========================
# GUI SELECTION
# ===========================

def preview_webcam_device(device_index: int):
    """
    Open a webcam preview on the primary (built-in) display.
    Stays open until the user presses any key or closes the window.
    Primary display is always at position (0, 0).
    """
    cap = cv2.VideoCapture(int(device_index))
    if not cap.isOpened():
        print(f"⚠️ Could not open webcam {device_index}")
        return

    win_name = "Webcam Preview — press any key to close"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    # Move to top-left of primary display (coordinates 0, 0)
    # This ensures it appears on the built-in screen, not the external TV.
    cv2.moveWindow(win_name, 0, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(win_name, frame)
        # Wait 1ms — any keypress closes the window
        if cv2.waitKey(1) & 0xFF != 255:
            break
        # Also close if the window was manually closed
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break

    cap.release()
    cv2.destroyWindow(win_name)


def unified_selection_window(displays, webcams, default_display_index=None,
                             default_webcam_device_index=None, default_mode=None, default_map_path=None):

    disp_options = [f"{d['index']}: {d['width']}x{d['height']}" for d in displays]
    cam_options = [f"{w['index']}: {w['model']} ({w['resolution']})" for w in webcams]
    mode_options = [MODE_LABELS[MODE_SELF_HOSTED], MODE_LABELS[MODE_FOUNDRY]]

    disp_index_by_pos = [d["index"] for d in displays]
    cam_device_by_pos = [w["index"] for w in webcams]

    map_files = _list_map_files()
    map_options = [os.path.basename(p) for p in map_files]

    selection = {"value": None}

    def submit():
        if display_combo.current() < 0 or webcam_combo.current() < 0:
            return
        sel_mode = MODE_LABEL_TO_VALUE[mode_combo.get()]
        if sel_mode == MODE_SELF_HOSTED and map_combo.current() < 0:
            return
        sel_map = None
        if sel_mode == MODE_SELF_HOSTED and map_files and map_combo.current() >= 0:
            sel_map = map_files[map_combo.current()]

        selection["value"] = {
            "display_index": disp_index_by_pos[display_combo.current()],
            "webcam_index": cam_device_by_pos[webcam_combo.current()],
            "mode": sel_mode,
            "map_path": sel_map
        }
        window.destroy()

    def cancel(event=None):
        window.destroy()

    def preview_webcam():
        if webcam_combo.current() >= 0:
            preview_webcam_device(cam_device_by_pos[webcam_combo.current()])

    def on_mode_change(event=None):
        sel_mode = MODE_LABEL_TO_VALUE.get(mode_combo.get(), MODE_SELF_HOSTED)
        if sel_mode == MODE_FOUNDRY:
            map_combo.configure(state="disabled")
            map_label.configure(state="disabled")
        else:
            map_combo.configure(state="readonly" if map_options else "disabled")
            map_label.configure(state="normal")
        available = display_combo.current() >= 0 and webcam_combo.current() >= 0
        map_available = sel_mode == MODE_FOUNDRY or map_combo.current() >= 0
        select_button.configure(state="normal" if available and map_available else "disabled")
        selection_status.set(
            "No camera detected" if not webcams else
            "No display detected" if not displays else
            "No self-hosted map available" if not map_available else "")

    root = _get_persistent_root()
    window = tk.Toplevel(root)
    window.title("Sarween Setup")
    window.geometry("520x360")
    window.resizable(False, False)
    window.bind("<Escape>", cancel)

    label = ttk.Label(window, text="Select your hardware + mode")
    label.pack(pady=10)

    frame = ttk.Frame(window)
    frame.pack(pady=10)

    ttk.Label(frame, text="Display").grid(row=0, column=0, sticky="e", padx=10)
    display_combo = ttk.Combobox(frame, values=disp_options, state="readonly", width=40)
    display_combo.grid(row=0, column=1)
    if default_display_index in disp_index_by_pos:
        display_combo.current(disp_index_by_pos.index(default_display_index))
    elif disp_options:
        display_combo.current(0)

    ttk.Label(frame, text="Webcam").grid(row=1, column=0, sticky="e", padx=10)
    webcam_combo = ttk.Combobox(frame, values=cam_options, state="readonly", width=40)
    webcam_combo.grid(row=1, column=1)
    if default_webcam_device_index in cam_device_by_pos:
        webcam_combo.current(cam_device_by_pos.index(default_webcam_device_index))
    elif cam_options:
        webcam_combo.current(0)

    ttk.Label(frame, text="Mode").grid(row=2, column=0, sticky="e", padx=10)
    mode_combo = ttk.Combobox(frame, values=mode_options, state="readonly", width=40)
    mode_combo.grid(row=2, column=1)
    mode_combo.set(MODE_LABELS.get(default_mode, MODE_LABELS[MODE_SELF_HOSTED]))
    mode_combo.bind("<<ComboboxSelected>>", on_mode_change)

    map_label = ttk.Label(frame, text="Map (self-hosted)")
    map_label.grid(row=3, column=0, sticky="e", padx=10)
    map_combo = ttk.Combobox(frame, values=map_options, state="readonly", width=40)
    map_combo.grid(row=3, column=1)

    default_idx = 0
    if default_map_path and map_files:
        try:
            base = os.path.basename(default_map_path)
            if base in map_options:
                default_idx = map_options.index(base)
        except Exception:
            pass
    if map_options:
        map_combo.current(default_idx)
    else:
        map_combo.configure(state="disabled")

    btns = ttk.Frame(window)
    btns.pack(pady=10)

    ttk.Button(btns, text="Preview Webcam", command=preview_webcam,
               state="normal" if webcams else "disabled").pack(side=tk.LEFT, padx=5)
    select_button = ttk.Button(btns, text="Select", command=submit)
    select_button.pack(side=tk.LEFT, padx=5)
    ttk.Button(btns, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)
    selection_status = tk.StringVar(master=window)
    ttk.Label(window, textvariable=selection_status).pack(pady=6)
    on_mode_change()

    root.wait_window(window)
    return selection["value"]


# ===========================
# INITIALIZE
# ===========================

def initialize():
    """
    Raise SystemExit on cancel; the application returns to its home window.
    """
    global selected_display, selected_webcam, selected_mode

    initialize_user_data()
    selection = load_last_selection()
    displays, webcams = detect_setup()

    sel = unified_selection_window(
        displays=displays,
        webcams=webcams,
        default_display_index=selection.get("display_index"),
        default_webcam_device_index=selection.get("webcam_index"),
        default_mode=selection.get("mode", MODE_SELF_HOSTED),
        default_map_path=selection.get("map_path")
    )

    if sel is None:
        print("Setup cancelled. Exiting program.")
        raise SystemExit(0)

    save_last_selection(
        display_index=sel["display_index"],
        webcam_index=sel["webcam_index"],
        mode=sel["mode"],
        map_path=sel.get("map_path")
    )

    selected_display = next(d for d in displays if d["index"] == sel["display_index"])
    selected_webcam = next(w for w in webcams if w["index"] == sel["webcam_index"])
    selected_mode = sel["mode"]

    print("Final selected display:", selected_display)
    print("Final selected webcam:", selected_webcam)
    print("Final selected mode:", selected_mode)
    print("Final selected map:", sel.get("map_path"))

    return selected_display, selected_webcam, selected_mode
