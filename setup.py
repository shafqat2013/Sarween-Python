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

def load_last_selection():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_last_selection(display_index=None, webcam_index=None):
    data = load_last_selection()
    if display_index is not None:
        data["display_index"] = display_index
    if webcam_index is not None:
        data["webcam_index"] = webcam_index
    with open(CONFIG_FILE, "w") as f:
        json.dump(data, f)

config_data = load_last_selection()
last_display_index = config_data.get("display_index")
last_webcam_index = config_data.get("webcam_index")

# ===========================
# HARDWARE DETECTION
# ===========================
selected_display = None
selected_webcam = None

def detect_setup():
    displays = []
    webcams = []
    try:
        result = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True)

        # Built-in / External
        built_in_displays = re.findall(r"Built-In", result.stdout)
        print("Built-in displays detected." if built_in_displays else "No built-in displays detected.")
        external_displays = re.findall(r"External", result.stdout)
        print("External displays detected." if external_displays else "No external displays detected.")

        # Mirroring
        mirroring = re.findall(r"Mirror:\s+On", result.stdout)
        print("Display mirroring detected." if mirroring else "No display mirroring detected.")

        # Scaling
        scaling = re.findall(r"Scaling:\s+On", result.stdout)
        print("Display scaling detected." if scaling else "No display scaling detected.")

        # Rotation
        rotation = re.findall(r"Rotation:\s+(\d+)", result.stdout)
        print(f"Display rotation detected: {rotation[0]} degrees." if rotation else "No display rotation detected.")

        # Resolutions
        resolutions = re.findall(r"Resolution:\s+(\d+) x (\d+)", result.stdout)
        for i, (w, h) in enumerate(resolutions):
            displays.append({
                "index": i,
                "width": int(w),
                "height": int(h),
                "x": 0,
                "y": 0
            })
        if not displays:
            print("No displays found.")
    except Exception as e:
        print(f"Error detecting displays with system_profiler, using backup method: {e}")
        displays = detect_displays_backup()

    print('Displays:')
    print(displays)

    try:
        result = subprocess.run(["system_profiler", "SPCameraDataType"], capture_output=True, text=True)
        modelid = re.findall(r"Model ID:\s+(\w+)", result.stdout)
        for i, model in enumerate(modelid):
            webcams.append({
                "index": i,
                "model": model
            })
    except Exception as e:
        print(f"Error detecting webcams with system_profiler, using backup method: {e}")
        webcams = detect_webcams_backup()

    print('Webcams:')
    print(webcams)

    print('Switching to CV to get resolution:')
    webcams = detect_webcams_backup()
    print(webcams)
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
    webcam_count = 5
    for i in range(webcam_count):
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

def dropdown_selection_with_preview(title, prompt, options, preview_function=None, default_index=None):
    selection = {"value": None}

    def submit():
        selected_index = combo.current()
        if selected_index >= 0:
            selection["value"] = selected_index
            window.destroy()

    def preview():
        index = combo.current()
        if index >= 0 and preview_function:
            preview_function(index)

    def cancel(event=None):
        window.destroy()

    window = tk.Tk()
    window.title(title)
    window.geometry("400x180")
    window.resizable(False, False)
    window.bind('<Escape>', cancel)

    # ===========================
    # DARK MODE STYLING
    # ===========================
    style = ttk.Style()
    style.theme_use('default')  # Prevent Aqua override on macOS
    window.configure(bg="#2e2e2e")
    style.configure("TLabel", foreground="white", background="#2e2e2e")
    style.configure("TButton", foreground="white", background="#3c3c3c")
    style.configure("TCombobox",
                    foreground="white",
                    fieldbackground="#3c3c3c",
                    background="#2e2e2e")

    # ===========================
    # UI Elements
    # ===========================
    label = ttk.Label(window, text=prompt, style="TLabel")
    label.pack(pady=10)

    combo = ttk.Combobox(window, values=options, state="readonly", width=50, style="TCombobox")
    combo.pack()
    if default_index is not None and 0 <= default_index < len(options):
        combo.current(default_index)

    frame = ttk.Frame(window)
    frame.pack(pady=10)

    preview_btn = ttk.Button(frame, text="Preview", command=preview, style="TButton")
    preview_btn.pack(side=tk.LEFT, padx=10)

    select_btn = ttk.Button(frame, text="Select", command=submit, style="TButton")
    select_btn.pack(side=tk.LEFT, padx=10)

    window.mainloop()
    return selection["value"]

def preview_webcam(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"⚠️ Could not open webcam {index}")
        return

    print(f"📷 Previewing webcam {index}... (Press ESC to close early)")
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

def webcam_selection(webcams):
    global last_webcam_index
    if not webcams:
        print("No webcams found.")
        return None

    options = [f"{w['index']}: {w['model']} ({w['resolution']})" for w in webcams]
    index = dropdown_selection_with_preview(
        title="Select Webcam",
        prompt="Select the webcam you want to use:",
        options=options,
        preview_function=preview_webcam,
        default_index=last_webcam_index
    )
    if index is None:
        print("No webcam selected.")
        return None

    last_webcam_index = index
    save_last_selection(webcam_index=index)
    return webcams[index]

def display_selection(displays):
    global last_display_index
    if not displays:
        print("No displays found.")
        return None

    options = [f"{d['index']}: {d['width']}x{d['height']}" for d in displays]
    index = dropdown_selection_with_preview(
        title="Select Display",
        prompt="Select the display you want to use:",
        options=options,
        preview_function=None,
        default_index=last_display_index
    )
    if index is None:
        print("No display selected.")
        return None

    last_display_index = index
    save_last_selection(display_index=index)
    return displays[index]

def default_to_external_display(displays):
    if len(displays) > 1:
        external_displays = [d for d in displays if d['x'] != 0 or d['y'] != 0]
        if external_displays:
            external = max(external_displays, key=lambda d: d['width'] * d['height'])
            print(f"Using External Display: {external['width']}x{external['height']}")
            return external
    print("Using External Display.")
    return displays[1]

# ===========================
# INITIALIZE
# ===========================

def initialize():
    #login, detect attached hardware, select webcam, select display, select map, store in database
    
    global selected_display, selected_webcam
    displays, webcams = detect_setup()

    if len(displays) == 0:
        print('No displays detected.')
    elif len(displays) == 1:
        print('No external display detected, defaulting to primary display.')
        selected_display = displays[0]
    elif len(displays) == 2:
        print('1 external display detected, defaulting to external display.')
        selected_display = displays[1]
        #default_to_external_display(displays)
    else:
        print('Multiple displays detected, please select manually.')
        selected_display = display_selection(displays)

    if len(webcams) == 0:
        print('No cameras detected.')
    elif len(webcams) == 1:
        print('Defaulting to primary camera.')
        selected_webcam = webcams[0]
    else:
        print('Multiple cameras detected, please select manually.')
        selected_webcam = webcam_selection(webcams)
