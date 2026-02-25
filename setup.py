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

# ===========================
# CONFIGURATION
# ===========================
CONFIG_FILE = "hardware_config.json"
MAPS_DIR = "maps"
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
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f)


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
            text=True
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
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            webcams.append({
                "index": i,
                "model": f"Webcam {i}",
                "resolution": f"{width}x{height}"
            })
            cap.release()
    return webcams


# ===========================
# GUI SELECTION
# ===========================

def preview_webcam_device(device_index: int):
    cap = cv2.VideoCapture(int(device_index))
    if not cap.isOpened():
        print(f"⚠️ Could not open webcam {device_index}")
        return

    start_time = time.time()
    cv2.namedWindow("Webcam Preview")

    while time.time() - start_time < 3:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


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
        sel_mode = MODE_LABEL_TO_VALUE[mode_combo.get()]
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
        preview_webcam_device(cam_device_by_pos[webcam_combo.current()])

    def on_mode_change(event=None):
        sel_mode = MODE_LABEL_TO_VALUE.get(mode_combo.get(), MODE_SELF_HOSTED)
        if sel_mode == MODE_FOUNDRY:
            map_combo.configure(state="disabled")
            map_label.configure(state="disabled")
        else:
            map_combo.configure(state="readonly" if map_options else "disabled")
            map_label.configure(state="normal")

    window = tk.Tk()
    window.title("Sarween Setup")
    window.geometry("520x320")
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

    ttk.Label(frame, text="Webcam").grid(row=1, column=0, sticky="e", padx=10)
    webcam_combo = ttk.Combobox(frame, values=cam_options, state="readonly", width=40)
    webcam_combo.grid(row=1, column=1)
    if default_webcam_device_index in cam_device_by_pos:
        webcam_combo.current(cam_device_by_pos.index(default_webcam_device_index))

    ttk.Label(frame, text="Mode").grid(row=2, column=0, sticky="e", padx=10)
    mode_combo = ttk.Combobox(frame, values=mode_options, state="readonly", width=40)
    mode_combo.grid(row=2, column=1)
    mode_combo.set(MODE_LABELS.get(default_mode, MODE_LABELS[MODE_SELF_HOSTED]))
    mode_combo.bind("<<ComboboxSelected>>", on_mode_change)

    # NEW: Map selection (self-hosted only)
    map_label = ttk.Label(frame, text="Map (self-hosted)")
    map_label.grid(row=3, column=0, sticky="e", padx=10)
    map_combo = ttk.Combobox(frame, values=map_options, state="readonly", width=40)
    map_combo.grid(row=3, column=1)

    # Default map selection
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

    on_mode_change()

    btns = ttk.Frame(window)
    btns.pack(pady=10)

    ttk.Button(btns, text="Preview Webcam", command=preview_webcam).pack(side=tk.LEFT, padx=5)
    ttk.Button(btns, text="Select", command=submit).pack(side=tk.LEFT, padx=5)
    ttk.Button(btns, text="Cancel", command=cancel).pack(side=tk.LEFT, padx=5)

    window.mainloop()
    return selection["value"]


# ===========================
# INITIALIZE
# ===========================

def initialize():
    """
    If the user cancels setup, EXIT THE PROGRAM.
    """
    global selected_display, selected_webcam, selected_mode

    displays, webcams = detect_setup()

    sel = unified_selection_window(
        displays=displays,
        webcams=webcams,
        default_display_index=last_display_index,
        default_webcam_device_index=last_webcam_device_index,
        default_mode=last_mode,
        default_map_path=last_map_path
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