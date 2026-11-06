# setup.py
# all functions for hardware detection and selection
# ===========================
import subprocess
import cv2
import numpy as np
import os
import sys
import re
import json
import time
import tkinter as tk
from tkinter import ttk
from screeninfo import get_monitors
from app_paths import atomic_write_json, data_path, initialize_user_data, resource_path, saved_map_path
from camera_sources import (PHONE_SOURCE, create_phone_profile, is_phone_camera_index,
                            open_camera_capture, phone_camera_inventory,
                            profile_compatible_with_source, source_for_index)


def _get_persistent_root() -> tk.Tk:
    """
    Return a single hidden Tk root that lives for the entire app lifetime.

    Why: on macOS, when a PyInstaller .app has no Tkinter windows open, the
    process loses its place in the window-server and the Dock icon disappears.
    Keeping one hidden root alive prevents that.
    """
    if not hasattr(_get_persistent_root, "_root"):
        root = tk.Tk()
        root.withdraw()          # hide — never shown to user
        root.title("Sarween")    # shown in Cmd+Tab switcher
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


_UNCHANGED_PROFILE = object()
_UNCHANGED_MAP = object()
_UNCHANGED_SOURCE = object()


def save_last_selection(display_index=None, webcam_index=None, mode=None, map_path=_UNCHANGED_MAP,
                        camera_profile_id=_UNCHANGED_PROFILE, camera_source=_UNCHANGED_SOURCE):
    data = load_last_selection()
    if display_index is not None:
        data["display_index"] = int(display_index)
    if webcam_index is not None:
        data["webcam_index"] = int(webcam_index)
        # Native selection must not retain a different phone's saved identity.
        if not is_phone_camera_index(data["webcam_index"]):
            data.pop("camera_source", None)
        elif camera_source is _UNCHANGED_SOURCE:
            previous_source = data.get("camera_source")
            if previous_source is not None:
                try:
                    source_for_index(data["webcam_index"], selection=data, inventory=[])
                except ValueError:
                    data.pop("camera_source", None)
    if camera_source is not _UNCHANGED_SOURCE:
        if camera_source is None or not is_phone_camera_index(data.get("webcam_index")):
            data.pop("camera_source", None)
        else:
            data["camera_source"] = source_for_index(data["webcam_index"], selection={
                "webcam_index": data["webcam_index"], "camera_source": camera_source}, inventory=[])
    if camera_profile_id is not _UNCHANGED_PROFILE:
        data["camera_profile_id"] = camera_profile_id
    if mode is not None:
        data["mode"] = str(mode)
    if map_path is not _UNCHANGED_MAP:
        data["map_path"] = str(map_path) if map_path is not None else None
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
selected_map_path = None


def detect_setup(*, passive=False):
    displays = []
    webcams = []

    # Use one inventory with real logical coordinates for both selection and
    # calibration placement. The profiler fallback has no desktop origins.
    try:
        displays = detect_displays_backup()
    except Exception:
        displays = []
    if passive:
        return displays, detect_webcams_passive()
    if displays:
        return displays, detect_webcams_backup()

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
    if sys.platform == "darwin":
        try:
            from foundry_display import MacFoundryDisplayBackend
            screens = MacFoundryDisplayBackend().displays()
            if screens:
                return [{"index": index, **screen} for index, screen in enumerate(screens)]
        except Exception:
            pass
    displays = []
    monitors = list(get_monitors())
    primary = next((monitor for monitor in monitors if monitor.is_primary), monitors[0] if monitors else None)
    for i, monitor in enumerate(monitors):
        displays.append({
            "index": i,
            "width": monitor.width,
            "height": monitor.height,
            "x": monitor.x,
            "y": (primary.height - monitor.y - monitor.height if sys.platform == "darwin" else monitor.y),
            "primary": bool(monitor.is_primary),
        })
    return displays


def _phone_inventory():
    """USB listing is passive; an unavailable daemon must not hide native inputs."""
    try:
        return phone_camera_inventory()
    except (ImportError, OSError, ValueError):
        return []


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
    return webcams + _phone_inventory()


# ===========================
# GUI SELECTION
# ===========================

def _preview_phone_camera(device_index, *, camera_source=None):
    """An explicit, cancellable preview; discovery never reaches this function."""
    cap = open_camera_capture(device_index, source=camera_source)
    window = "iPhone Preview — press any key to close"
    opened_window = False
    try:
        if not cap.isOpened():
            raise ValueError("The selected iPhone camera is unavailable.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        opened_window = True
        cv2.resizeWindow(window, 1280, 720)
        cv2.moveWindow(window, 0, 0)
        while True:
            ok, pixels = cap.read()
            if not ok:
                pixels = np.zeros((720, 1280, 3), np.uint8)
                status = cap.status
                message = str(status.get("message") or "Waiting for Sarween Camera on the iPhone…")
                cv2.putText(pixels, message[:105], (25, 345), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(pixels, "Press any key to cancel", (25, 385), cv2.FONT_HERSHEY_SIMPLEX,
                            .65, (180, 180, 180), 1, cv2.LINE_AA)
            cv2.imshow(window, pixels)
            if cv2.waitKey(1) & 0xFF != 255:
                break
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        cap.release()
        if opened_window:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass


def preview_webcam_device(device_index: int, *, camera_source=None):
    if is_phone_camera_index(device_index):
        return _preview_phone_camera(device_index, camera_source=camera_source)
    cap = open_camera_capture(int(device_index), source=camera_source)
    if not cap.isOpened():
        cap.release()
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


def camera_profile_pairing_confirmed(profile_id, confirmed):
    """Device indices cannot prove that the same physical/virtual feed is open."""
    return profile_id is None or confirmed is True


def unified_selection_window(displays, webcams, default_display_index=None,
                             default_webcam_device_index=None, default_mode=None, default_map_path=None,
                             default_camera_profile_id=None, access=None):
    """Compatibility wrapper; the main app embeds HardwareSelectionPanel instead."""
    root = _get_persistent_root()
    window = tk.Toplevel(root)
    window.title("Sarween Setup")
    window.geometry("660x560")
    result = {"value": None}
    buttons = ttk.Frame(window)
    panel = None

    def changed():
        if panel is not None:
            select_button.configure(state="normal" if panel.ready else "disabled")

    panel = HardwareSelectionPanel(window, access=access, changed=changed, defaults={
        "display_index": default_display_index, "webcam_index": default_webcam_device_index,
        "mode": default_mode or MODE_SELF_HOSTED, "map_path": default_map_path,
        "camera_profile_id": default_camera_profile_id})
    panel.pack(fill="both", expand=True, padx=16, pady=12)

    def submit():
        try:
            result["value"] = panel.selection()
        except (ValueError, OSError) as exc:
            panel.status.set(str(exc))
            return
        window.destroy()

    select_button = ttk.Button(buttons, text="Select", command=submit, state="disabled")
    select_button.pack(side="left", padx=5)
    ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=5)
    buttons.pack(pady=(0, 12))
    window.bind("<Escape>", lambda _event: window.destroy())
    panel.set_inventory(displays, webcams)
    root.wait_window(window)
    return result["value"]


# ===========================
# INITIALIZE
# ===========================

def _prime_camera_permission():
    """
    On first launch as a .app, macOS shows a camera permission prompt when
    cv2.VideoCapture is first called. This interrupts webcam detection and
    causes the dropdown to appear empty.

    Fix: open camera 0 briefly to trigger the prompt, wait up to 3 seconds
    for the user to grant permission, then release before the real scan runs.
    Only runs when frozen (inside .app bundle).
    """
    if not getattr(sys, "frozen", False):
        return
    try:
        cap = cv2.VideoCapture(0)
        # Wait up to 3 seconds for permission to be granted
        for _ in range(30):
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    break
            time.sleep(0.1)
        cap.release()
        time.sleep(0.3)  # small extra pause before scanning all indices
    except Exception:
        pass


def _active_camera_profile_id():
    from hardware_profiles import HardwareProfiles
    selected = HardwareProfiles().active_profile("camera")
    return selected["id"] if selected else None


def initialize(access=None, *, selection=None, displays=None, webcams=None):
    """Apply an embedded selection, or use the legacy chooser for older callers.

    Explicit choices require caller-supplied inventory and never discover/open
    hardware or create a setup window. A cancelled legacy chooser raises
    SystemExit so its caller can return to the app home window.
    """
    if access is not None:
        access.require()
    initialize_user_data()
    if selection is not None:
        if displays is None or webcams is None:
            raise ValueError("Supply the display and camera inventory for this selection.")
        return apply_selection(selection, displays, webcams, access=access)
    previous = load_last_selection()
    if displays is None or webcams is None:
        displays, webcams = detect_setup()
    selection = unified_selection_window(
        displays=displays, webcams=webcams,
        default_display_index=previous.get("display_index"),
        default_webcam_device_index=previous.get("webcam_index"),
        default_mode=previous.get("mode", MODE_SELF_HOSTED),
        default_map_path=previous.get("map_path"),
        default_camera_profile_id=_active_camera_profile_id(), access=access)
    if selection is None:
        print("Setup cancelled. Exiting program.")
        raise SystemExit(0)
    return apply_selection(selection, displays, webcams, access=access)


def _avfoundation_video_devices():
    """Enumerate video devices without creating a capture session or opening one."""
    import objc
    from Foundation import NSBundle
    bundle = NSBundle.bundleWithPath_(
        "/System/Library/Frameworks/AVFoundation.framework")
    if bundle is None or not bundle.load():
        raise RuntimeError("AVFoundation device inventory is unavailable.")
    device_class = objc.lookUpClass("AVCaptureDevice")
    # Match OpenCV's video + muxed inventory and uniqueID ordering without
    # creating AVCaptureDeviceInput/AVCaptureSession or requesting camera access.
    # https://github.com/opencv/opencv/blob/4.12.0/modules/videoio/src/cap_avfoundation_mac.mm#L339
    devices = list(device_class.devicesWithMediaType_("vide"))
    devices.extend(device_class.devicesWithMediaType_("muxx"))
    return sorted(devices, key=lambda device: str(device.uniqueID()))

def detect_webcams_passive():
    """Read the native camera inventory; never probe cameras with VideoCapture.

    A saved index is only a fallback choice, explicitly marked unverified. There
    is no portable passive OpenCV enumeration fallback, so no index is invented.
    """
    if sys.platform == "darwin":
        try:
            native = [{"index": index, "model": str(device.localizedName()),
                     "resolution": "Not opened", "enumeration_source": "AVFoundation",
                     "availability": "not_opened"}
                    for index, device in enumerate(_avfoundation_video_devices())]
            return native + _phone_inventory()
        except Exception:
            pass
    saved_index = load_last_selection().get("webcam_index")
    if type(saved_index) is not int or saved_index < 0:
        return _phone_inventory()
    return [{"index": saved_index, "model": f"Saved camera {saved_index}",
             "resolution": "Not opened", "enumeration_source": "saved-selection",
             "availability": "unverified",
             "inventory_note": "Camera inventory is unavailable. Preview the saved camera to confirm this input."}] + _phone_inventory()

def _profile_store(store=None):
    if store is None:
        from hardware_profiles import HardwareProfiles
        return HardwareProfiles()
    return store

def _validate_selection(selection, displays, webcams, *, store=None, map_files=None):
    """Validate an explicit choice without changing profiles, hardware or files."""
    if not isinstance(selection, dict):
        raise ValueError("Choose a camera, display and mode first.")
    for field, inventory, label in (("display_index", displays, "display"), ("webcam_index", webcams, "camera")):
        if type(selection.get(field)) is not int or not any(item["index"] == selection[field] for item in inventory):
            raise ValueError(f"Choose an available {label}.")
    mode = selection.get("mode")
    if not isinstance(mode, str) or mode not in MODE_LABELS:
        raise ValueError("Choose Foundry or Self Hosted mode.")
    source = source_for_index(selection["webcam_index"], inventory=webcams)
    profile_id = selection.get("camera_profile_id")
    if profile_id is not None:
        profile = _profile_store(store).get_profile("camera", profile_id) if isinstance(profile_id, str) else None
        if profile is None:
            raise ValueError("Choose a saved camera profile.")
        if not profile_compatible_with_source(profile, source):
            if source["type"] == PHONE_SOURCE:
                raise ValueError("Choose this iPhone's Sarween Camera profile or create a new one. OBS calibration belongs to a different feed.")
            raise ValueError("This profile belongs to the direct USB iPhone feed. Choose a native/OBS camera profile or no profile for this input.")
        if not camera_profile_pairing_confirmed(profile_id, selection.get("camera_profile_confirmed")):
            raise ValueError("Confirm that the selected camera profile matches this feed.")
    map_path = selection.get("map_path") if mode == MODE_SELF_HOSTED else None
    if mode == MODE_SELF_HOSTED:
        available = _list_map_files() if map_files is None else list(map_files)
        if map_path not in available or not os.path.isfile(map_path):
            raise ValueError("Choose an available self-hosted map.")
    result = {"display_index": selection["display_index"], "webcam_index": selection["webcam_index"],
            "camera_profile_id": profile_id,
            "camera_profile_confirmed": selection.get("camera_profile_confirmed") is True,
            "mode": mode, "map_path": map_path}
    if source["type"] == PHONE_SOURCE:
        result["camera_source"] = source
    return result

class HardwareSelectionPanel(ttk.Frame):
    """Embedded choices. Construction and selection never scan or open hardware.

    The caller supplies inventory and owns refresh. Only explicit Preview and
    Calibrate actions open the selected camera.
    """

    def __init__(self, parent, *, access=None, changed=None, store=None, defaults=None, map_files=None):
        super().__init__(parent)
        self.access, self._changed = access, changed
        self.store = _profile_store(store)
        self.displays, self.webcams = [], []
        self._has_inventory, self._enabled, self._notifying = False, True, False
        self.map_files = list(_list_map_files() if map_files is None else map_files)
        self._defaults = dict(defaults or {})
        self.status = tk.StringVar(self, "Refresh hardware to choose a camera and display.")
        self.profile_confirmed = tk.BooleanVar(self, False)
        self.columnconfigure(1, weight=1)
        for row, label in ((0, "Display"), (1, "Camera"), (2, "Camera profile"), (4, "Mode"), (5, "Map (self-hosted)")):
            widget = ttk.Label(self, text=label)
            widget.grid(row=row, column=0, sticky="nw", padx=(0, 10), pady=4)
            if row == 5:
                self.map_label = widget
        self.display_combo = ttk.Combobox(self, state="disabled", width=24)
        self.webcam_combo = ttk.Combobox(self, state="disabled", width=24)
        for row, combo in enumerate((self.display_combo, self.webcam_combo)):
            combo.grid(row=row, column=1, sticky="ew", pady=4)
        from hardware_profiles_ui import CameraProfilePicker
        default_id = self._defaults.get("camera_profile_id", (self.store.active_profile("camera") or {}).get("id"))
        self.profile_picker = CameraProfilePicker(self, selected_id=default_id, persist=False,
            changed=self._profile_changed, store=self.store)
        self.profile_picker.grid(row=2, column=1, sticky="ew", pady=4)
        self.profile_confirm_check = ttk.Checkbutton(self, variable=self.profile_confirmed,
            text="This profile matches the camera, lens, zoom and frame layout", command=self._update_state)
        self.profile_confirm_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        self.mode_combo = ttk.Combobox(self, state="readonly", values=tuple(MODE_LABELS.values()), width=24)
        self.mode_combo.set(MODE_LABELS.get(self._defaults.get("mode"), MODE_LABELS[MODE_FOUNDRY]))
        self.mode_combo.grid(row=4, column=1, sticky="ew", pady=4)
        self.map_combo = ttk.Combobox(self, state="disabled", width=24,
                                     values=[os.path.basename(path) for path in self.map_files])
        self.map_combo.grid(row=5, column=1, sticky="ew", pady=4)
        self._choose(self.map_combo, self.map_files, self._defaults.get("map_path"), fallback=True)
        actions = ttk.Frame(self)
        actions.grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self.preview_button = ttk.Button(actions, text="Preview Webcam", command=self.preview_camera)
        self.preview_button.pack(side="left", padx=(0, 6))
        self.calibrate_button = ttk.Button(actions, text="Calibrate camera", command=self.calibrate_camera)
        self.calibrate_button.pack(side="left")
        self.create_phone_profile_button = ttk.Button(actions, text="Create iPhone profile", command=self.create_iphone_profile)
        self.create_phone_profile_button.pack(side="left", padx=(6, 0))
        label = ttk.Label(self, textvariable=self.status, wraplength=540, justify="left")
        label.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
        self.display_combo.bind("<<ComboboxSelected>>", self._update_state)
        self.webcam_combo.bind("<<ComboboxSelected>>", self._camera_changed)
        self.mode_combo.bind("<<ComboboxSelected>>", self._update_state)
        self.map_combo.bind("<<ComboboxSelected>>", self._update_state)
        self._update_state()

    @staticmethod
    def _choose(combo, choices, value, *, fallback=False):
        if value in choices:
            combo.current(choices.index(value))
        elif fallback and choices and value is None:
            combo.current(0)
        else:
            combo.set("")

    @staticmethod
    def _index(combo, inventory):
        pos = combo.current()
        return inventory[pos]["index"] if 0 <= pos < len(inventory) else None

    def set_inventory(self, displays, webcams, defaults=None):
        old_camera = self._index(self.webcam_combo, self.webcams)
        previous_camera = next((item for item in self.webcams if item["index"] == old_camera), None)
        previous_profile = self.profile_picker.selected_profile()
        self.profile_picker.refresh()
        values = dict(self._defaults if not self._has_inventory else {
            "display_index": self._index(self.display_combo, self.displays), "webcam_index": old_camera})
        if defaults is not None:
            values.update(defaults)
        self.displays, self.webcams = [dict(item) for item in displays], [dict(item) for item in webcams]
        self.display_combo.configure(values=[f"{d['index']}: {d.get('name', 'Display')} ({d['width']:g}×{d['height']:g})" for d in self.displays])
        self.webcam_combo.configure(values=[
            ("" if is_phone_camera_index(w["index"]) else f"{w['index']}: ")
            + f"{w.get('model', 'Camera')} ({w.get('resolution', 'Not opened')})" for w in self.webcams])
        self._choose(self.display_combo, [d["index"] for d in self.displays], values.get("display_index"), fallback=not self._has_inventory)
        self._choose(self.webcam_combo, [w["index"] for w in self.webcams], values.get("webcam_index"), fallback=not self._has_inventory)
        if defaults is not None:
            if "mode" in defaults:
                self.mode_combo.set(MODE_LABELS.get(defaults["mode"], ""))
            if "map_path" in defaults:
                self._choose(self.map_combo, self.map_files, defaults["map_path"])
            if "camera_profile_id" in defaults:
                profile_id = defaults["camera_profile_id"]
                if profile_id is not None and self.store.get_profile("camera", profile_id) is None:
                    profile_id = None
                self.profile_picker.select_id(profile_id, persist=False, notify=False)
                self.profile_confirmed.set(False)
        current_camera = next((item for item in self.webcams
                               if item["index"] == self._index(self.webcam_combo, self.webcams)), None)
        if previous_camera != current_camera or previous_profile != self.profile_picker.selected_profile():
            self.profile_confirmed.set(False)
        self._has_inventory = True
        self._update_state()

    def selection(self):
        pos = self.map_combo.current()
        return _validate_selection({"display_index": self._index(self.display_combo, self.displays),
            "webcam_index": self._index(self.webcam_combo, self.webcams),
            "camera_profile_id": self.profile_picker.selected_id(),
            "camera_profile_confirmed": self.profile_confirmed.get() is True,
            "mode": MODE_LABEL_TO_VALUE.get(self.mode_combo.get()),
            "map_path": self.map_files[pos] if 0 <= pos < len(self.map_files) else None},
            self.displays, self.webcams, store=self.store, map_files=self.map_files)

    @property
    def ready(self):
        try:
            self.selection()
            return True
        except (ValueError, OSError):
            return False

    def _profile_changed(self, _profile=None):
        self.profile_confirmed.set(False)
        self._update_state()

    def _camera_changed(self, _event=None):
        self.profile_picker.select_id(None, persist=False, notify=False)
        self.profile_confirmed.set(False)
        self._update_state()

    def create_iphone_profile(self):
        try:
            self._require_access()
            index = self._index(self.webcam_combo, self.webcams)
            if not is_phone_camera_index(index):
                raise ValueError("Choose a USB iPhone first.")
            source = source_for_index(index, inventory=self.webcams)
            profile = create_phone_profile(self.store, source)
            self.profile_picker.refresh()
            self.profile_picker.select_id(profile["id"], persist=False, notify=False)
            self.profile_confirmed.set(False)
            self._update_state()
            self.status.set("New iPhone profile created. Calibrate camera for this feed, or confirm the profile to continue without lens correction.")
            return profile
        except (ValueError, OSError, PermissionError) as exc:
            self.status.set(str(exc))
            return False

    def refresh_profiles(self):
        previous = self.profile_picker.selected_profile()
        self.profile_picker.refresh()
        if previous != self.profile_picker.selected_profile():
            self.profile_confirmed.set(False)
        self._update_state()

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._update_state()

    def _update_state(self, _event=None):
        enabled = self._enabled
        self.display_combo.configure(state="readonly" if enabled and self.displays else "disabled")
        self.webcam_combo.configure(state="readonly" if enabled and self.webcams else "disabled")
        self.mode_combo.configure(state="readonly" if enabled else "disabled")
        self.profile_picker.set_enabled(enabled)
        self.profile_confirm_check.configure(state="normal" if enabled and self.profile_picker.selected_id() else "disabled")
        self_hosted = self.mode_combo.get() == MODE_LABELS[MODE_SELF_HOSTED]
        self.map_combo.configure(state="readonly" if enabled and self_hosted and self.map_files else "disabled")
        self.map_label.configure(state="normal" if self_hosted else "disabled")
        camera_ready = self._index(self.webcam_combo, self.webcams) is not None
        camera_index = self._index(self.webcam_combo, self.webcams)
        phone_selected = is_phone_camera_index(camera_index)
        profile_compatible = False
        if camera_ready:
            try:
                source = source_for_index(camera_index, inventory=self.webcams)
                profile_compatible = profile_compatible_with_source(self.profile_picker.selected_profile(), source)
            except ValueError:
                profile_compatible = False
        self.create_phone_profile_button.configure(state="normal" if enabled and phone_selected else "disabled")
        self.preview_button.configure(state="normal" if enabled and camera_ready else "disabled")
        self.calibrate_button.configure(state="normal" if enabled and camera_ready
                                        and self._index(self.display_combo, self.displays) is not None
                                        and self.profile_picker.selected_id() and profile_compatible else "disabled")
        try:
            self.selection()
            camera = next(item for item in self.webcams if item["index"] == self._index(self.webcam_combo, self.webcams))
            self.status.set(camera.get("inventory_note", "Ready to start."))
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))
        if self._changed is not None and not self._notifying:
            self._notifying = True
            try:
                self._changed()
            finally:
                self._notifying = False

    def _require_access(self):
        if not self._enabled:
            raise PermissionError("Sign in before using the camera.")
        if self.access is not None:
            self.access.require()

    def preview_camera(self):
        try:
            self._require_access()
            index = self._index(self.webcam_combo, self.webcams)
            if index is None:
                raise ValueError("Choose an available camera first.")
            if is_phone_camera_index(index):
                preview_webcam_device(index, camera_source=source_for_index(index, inventory=self.webcams))
            else:
                preview_webcam_device(index)
            return True
        except Exception as exc:
            self.status.set(str(exc))
            return False

    def calibrate_camera(self):
        try:
            self._require_access()
            profile_id = self.profile_picker.selected_id()
            camera_index = self._index(self.webcam_combo, self.webcams)
            display_index = self._index(self.display_combo, self.displays)
            if not profile_id or camera_index is None or display_index is None:
                raise ValueError("Choose a named camera profile, camera, and display first.")
            source = source_for_index(camera_index, inventory=self.webcams)
            if not profile_compatible_with_source(self.store.get_profile("camera", profile_id), source):
                raise ValueError("Choose this input's own camera profile before calibration. Direct USB and native/OBS feeds need separate profiles.")
            camera_options = {}
            if is_phone_camera_index(camera_index):
                camera_options["camera_source"] = source
            from calibration import calibrate_webcam, last_camera_calibration_message
            self.status.set("Move the camera, then hold still for auto capture. T: 5-second timer; Q: cancel.")
            self.update_idletasks()
            result = calibrate_webcam(camera_index, profile_id=profile_id, access=self.access,
                display=next(item for item in self.displays if item["index"] == display_index), displays=self.displays,
                **camera_options)
            self.profile_picker.refresh()
            self.profile_confirmed.set(bool(result))
            self._update_state()
            self.status.set("Camera profile updated. Keep the lens, zoom and frame layout fixed."
                            if result else f"Calibration was not saved. {last_camera_calibration_message()}")
            return bool(result)
        except Exception as exc:
            self.status.set(str(exc))
            return False

def apply_selection(selection, displays, webcams, *, access=None):
    """Apply already-chosen hardware without discovering or opening anything."""
    global selected_display, selected_webcam, selected_mode, selected_map_path
    if access is not None:
        access.require()
    store = _profile_store()
    sel = _validate_selection(selection, displays, webcams, store=store)
    previous_profile_id = (store.active_profile("camera") or {}).get("id")
    profile_changed = previous_profile_id != sel["camera_profile_id"]
    if profile_changed:
        store.select_profile("camera", sel["camera_profile_id"])
    try:
        source_options = {"camera_source": sel["camera_source"]} if "camera_source" in sel else {}
        save_last_selection(
            display_index=sel["display_index"], webcam_index=sel["webcam_index"],
            camera_profile_id=sel["camera_profile_id"], mode=sel["mode"], map_path=sel["map_path"], **source_options)
    except Exception:
        if profile_changed:
            # The files are individually atomic. Keep a failed setup save from
            # silently changing the active calibration where rollback is possible.
            try:
                store.select_profile("camera", previous_profile_id)
            except (ValueError, OSError):
                pass
        raise

    selected_display = next(d for d in displays if d["index"] == sel["display_index"])
    selected_webcam = next(w for w in webcams if w["index"] == sel["webcam_index"])
    selected_mode = sel["mode"]
    selected_map_path = sel["map_path"]

    print("Final selected display:", selected_display)
    print("Final selected webcam:", selected_webcam)
    print("Final selected mode:", selected_mode)
    print("Final selected map:", sel.get("map_path"))

    return selected_display, selected_webcam, selected_mode
