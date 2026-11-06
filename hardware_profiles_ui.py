"""Explicit camera and display profile selection for native Sarween windows."""

import math
import time
import tkinter as tk
from tkinter import ttk

from foundry_grid_controls import GRID_FIELDS


DISPLAY_SETTINGS = tuple(field[0] for field in GRID_FIELDS) + (
    "viewportMarkersEnabled", "viewportMarkerSize", "viewportMarkerInset", "physicalGridAutoApply")
CAMERA_FIELDS = (
    ("name", "Profile name"), ("deviceLabel", "Camera device label"),
    ("sourceHint", "Input source / connection"), ("lens", "Lens"), ("zoom", "Zoom"),
    ("transformNotes", "OBS crop, rotation and padding"),
)
_ACTIVE = object()


def _store(value):
    if value is None:
        from hardware_profiles import HardwareProfiles
        value = HardwareProfiles()
    return value


def _active_id(store, kind):
    try:
        return (store.active_profile(kind) or {}).get("id")
    except (ValueError, OSError):
        return None


def _vtt_status():
    import foundryoutput
    return foundryoutput.get_vtt_status()


def _send_profile(action, values):
    import foundryoutput
    return foundryoutput.request_vtt_command(action, values, return_request_id=True)


def camera_calibration_status(profile):
    if not profile:
        return "No camera profile selected. No lens calibration will be applied."
    if not profile.get("calibration"):
        return "No calibration saved for this profile. Frames stay uncorrected."
    if not profile.get("bindingVerified"):
        return "Unverified calibration. Frames stay uncorrected until this feed is calibrated."
    size = (profile.get("calibration") or {}).get("imageSize")
    suffix = f" ({size[0]} × {size[1]})" if size else ""
    return f"Calibration saved for this camera feed{suffix}."


def camera_form_values(values):
    result = {name: str(values.get(name, "")).strip() for name, _label in CAMERA_FIELDS}
    if not result["name"]:
        raise ValueError("Enter a camera profile name.")
    width, height = str(values.get("width", "")).strip(), str(values.get("height", "")).strip()
    if not width and not height:
        result["resolution"] = None
    else:
        try:
            resolution = [int(width), int(height)]
        except ValueError:
            raise ValueError("Enter both final output dimensions in whole pixels.") from None
        if not all(1 <= value <= 32768 for value in resolution):
            raise ValueError("Final output dimensions must be positive pixel counts.")
        result["resolution"] = resolution
    return result


def display_settings(values, *, complete=False):
    if not values or set(values) - set(DISPLAY_SETTINGS):
        raise ValueError("Save at least one known display setting.")
    if complete and set(values) != set(DISPLAY_SETTINGS):
        raise ValueError("Foundry has not supplied all display settings yet.")
    result = {}
    for name, label, low, high in (
            *((name, label, low, high) for name, label, low, high, *_rest in GRID_FIELDS),
            ("viewportMarkerSize", "Marker size", 72, 240), ("viewportMarkerInset", "Marker inset", 0, 80)):
        if name not in values:
            continue
        try:
            if isinstance(values[name], bool):
                raise ValueError()
            value = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Enter a number for {label.lower()}.") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}.")
        result[name] = value
    for name in ("viewportMarkersEnabled", "physicalGridAutoApply"):
        if name in values:
            if not isinstance(values[name], bool):
                raise ValueError("Choose on or off for the display options.")
            result[name] = values[name]
    return result


class CameraProfilePicker(ttk.Frame):
    """Selection is explicit and affects the next camera session, never a live feed."""

    def __init__(self, parent, *, selected_id=_ACTIVE, changed=None, store=None, persist=True):
        super().__init__(parent)
        self.store = _store(store)
        self._changed = changed
        self._persist = bool(persist)
        self._profiles = {}
        self._selected_id = _active_id(self.store, "camera") if selected_id is _ACTIVE else selected_id
        self.choice = tk.StringVar(self)
        self.status = tk.StringVar(self)
        self.combo = ttk.Combobox(self, textvariable=self.choice, state="readonly")
        self.combo.pack(fill="x")
        self.combo.bind("<<ComboboxSelected>>", self._selected)
        label = ttk.Label(self, textvariable=self.status, wraplength=420)
        label.pack(fill="x", pady=(3, 0))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width)))
        self.refresh()

    def refresh(self):
        try:
            profiles = self.store.list_profiles("camera")
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))
            return
        self._profiles = {profile["id"]: profile for profile in profiles}
        self._labels = {profile["name"]: profile["id"] for profile in profiles}
        self.combo.configure(values=("No camera profile (no calibration)", *self._labels))
        if self._selected_id not in self._profiles:
            self._selected_id = None
        profile = self.selected_profile()
        self.choice.set(profile["name"] if profile else "No camera profile (no calibration)")
        self.status.set(camera_calibration_status(profile))

    def _selected(self, _event=None):
        try:
            self.select_id(self._labels.get(self.choice.get()))
        except (ValueError, OSError) as exc:
            self.status.set(str(exc))

    def select_id(self, profile_id, *, persist=None, notify=True):
        if profile_id is not None and profile_id not in self._profiles:
            raise ValueError("Choose a saved camera profile.")
        should_persist = self._persist if persist is None else persist
        if should_persist:
            self.store.select_profile("camera", profile_id)
        self._selected_id = profile_id
        self.refresh()
        if notify and self._changed is not None:
            self._changed(self.selected_profile())

    def selected_id(self):
        return self._selected_id

    def selected_profile(self):
        return self._profiles.get(self._selected_id)

    def set_enabled(self, enabled):
        self.combo.configure(state="readonly" if enabled else "disabled")


class DisplayProfilePicker(ttk.LabelFrame):
    def __init__(self, parent, *, require_access=None, store=None, read_status=None,
                 request=None, display_controller=None, can_apply=None):
        super().__init__(parent, text="Display profile", padding=10)
        self.store = _store(store)
        self._require_access = require_access
        self._read_status = read_status or _vtt_status
        self._request = request or _send_profile
        self._display = display_controller
        self._can_apply = can_apply
        self._enabled = False
        self._profiles = {}
        self._labels = {}
        self._pending = None
        self._last_status = {}
        self._last_profiles = float("-inf")
        self._selected_id = _active_id(self.store, "display")
        self.choice = tk.StringVar(self)
        self.message = tk.StringVar(self, "Choose a saved profile, then Apply profile.")
        self.columnconfigure(0, weight=1)
        self.combo = ttk.Combobox(self, textvariable=self.choice, state="disabled")
        self.combo.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.combo.bind("<<ComboboxSelected>>", self._selected)
        self.apply_button = ttk.Button(self, text="Apply profile", command=self.apply_profile)
        self.apply_button.grid(row=0, column=1)
        label = ttk.Label(self, textvariable=self.message, wraplength=350)
        label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width)))
        self.refresh_profiles()
        self._update_enabled()

    def refresh_profiles(self):
        try:
            profiles = self.store.list_profiles("display")
        except (ValueError, OSError) as exc:
            self.message.set(str(exc))
            self._last_profiles = time.monotonic()
            return
        self._profiles = {profile["id"]: profile for profile in profiles}
        self._labels = {profile["name"]: profile["id"] for profile in profiles}
        if self._selected_id not in self._profiles:
            self._selected_id = None
        self.combo.configure(values=tuple(self._labels))
        self.choice.set(self._profiles.get(self._selected_id, {}).get("name", ""))
        self._last_profiles = time.monotonic()

    def _selected(self, _event=None):
        self._selected_id = self._labels.get(self.choice.get())
        self.message.set("Apply this profile to use its grid and marker settings.")
        self._update_enabled()

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._update_enabled()

    def _available(self):
        status = self._last_status
        capture = status.get("capture") or {}
        active = (capture.get("active") or capture.get("recording")
                  or capture.get("phase") in {"Starting", "Saving", "Restoring"})
        return bool(self._enabled and not self._pending and status.get("connected")
                    and status.get("supported") and not status.get("pending") and not status.get("blocked")
                    and not active and "settings.set" in status.get("capabilities", [])
                    and (self._can_apply is None or self._can_apply()))

    def _update_enabled(self):
        self.combo.configure(state="readonly" if self._enabled and not self._pending else "disabled")
        self.apply_button.configure(state="normal" if self._selected_id and self._available() else "disabled")

    def apply_profile(self):
        try:
            if self._require_access is not None:
                self._require_access()
            status = self._read_status()
            self.refresh(status)
            if not self._available():
                return False
            profile = self.store.get_profile("display", self._selected_id)
            if profile is None:
                raise ValueError("Choose a saved display profile.")
            settings = display_settings(profile["settings"])
            request_id = self._request("settings.set", {"settings": settings})
            if not isinstance(request_id, str) or not request_id:
                self.message.set("Foundry did not accept the profile. Check the connection and try again.")
                return False
            self._pending = {"requestId": request_id, "profileId": profile["id"], "name": profile["name"],
                             "sceneId": status.get("sceneId"), "settings": dict(settings),
                             "targetDisplayId": profile.get("targetDisplayId")}
            self.message.set(f"Applying {profile['name']}…")
            self.refresh(self._read_status())
            return True
        except Exception as exc:
            self.message.set(str(exc))
            return False

    def refresh(self, status=None):
        if time.monotonic() - self._last_profiles >= 2:
            self.refresh_profiles()
        self._last_status = dict(self._read_status() if status is None else status)
        if self._pending:
            result = self._last_status.get("lastResult") or {}
            request = self._last_status.get("pendingRequest") or {}
            pending = self._pending
            if result.get("requestId") == pending["requestId"]:
                self._pending = None
                if result.get("success") is True and result.get("sceneId") == pending["sceneId"]:
                    try:
                        current = self.store.get_profile("display", pending["profileId"])
                        if (current is None or current.get("settings") != pending["settings"]
                                or current.get("targetDisplayId") != pending["targetDisplayId"]):
                            self.message.set("Prior saved settings applied; the profile changed. Apply it again.")
                            self._update_enabled()
                            return
                        self.store.select_profile("display", pending["profileId"])
                        self.message.set(f"Applied {pending['name']}.")
                        if pending.get("targetDisplayId") and self._display is not None:
                            if self._require_access is not None:
                                self._require_access()
                            self._display.configure(display_id=pending["targetDisplayId"])
                    except Exception as exc:
                        self.message.set(f"Grid and markers applied. {exc}")
                else:
                    self.message.set(str(result.get("message") or "Foundry could not apply this profile."))
            elif request.get("requestId") != pending["requestId"] and not self._last_status.get("pending"):
                self._pending = None
                self.message.set(str(self._last_status.get("message") or "Profile application was not confirmed."))
        self._update_enabled()


def _page(notebook, title):
    page = ttk.Frame(notebook)
    notebook.add(page, text=title)
    canvas = tk.Canvas(page, highlightthickness=0, height=260)
    bar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
    bar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=bar.set)
    frame = ttk.Frame(canvas, padding=10)
    item = canvas.create_window(0, 0, window=frame, anchor="nw")
    frame.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
    frame.columnconfigure(1, weight=1)
    frame._scroll_canvas = canvas
    return frame


def _bind_scroll(frame):
    tag = f"HardwareProfileScroll{id(frame)}"

    def scroll(event):
        direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else 1
        frame._scroll_canvas.yview_scroll(direction, "units")
        return "break"

    for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        frame.bind_class(tag, event, scroll)
        frame._scroll_canvas.bind(event, scroll)

    def include(widget):
        widget.bindtags((tag, *widget.bindtags()))
        for child in widget.winfo_children():
            include(child)

    include(frame)


class HardwareProfilesPanel(ttk.Frame):
    def __init__(self, parent, *, require_access=None, store=None, read_vtt_status=None,
                 display_controller=None, changed=None, calibrate_camera=None):
        super().__init__(parent)
        self.store = _store(store)
        self._require_access = require_access
        self._read_vtt = read_vtt_status or _vtt_status
        self._display = display_controller
        self._changed = changed
        self._calibrate_camera = calibrate_camera
        self._enabled = False
        self._last_refresh = float("-inf")
        self._camera_id = None
        self._display_id = None
        self._camera_profiles = {}
        self._display_profiles = {}
        self._display_saved_keys = set()
        self._camera_snapshot = {}
        self._display_snapshot = {}
        self._widgets = []
        self.message = tk.StringVar(self)
        tabs = ttk.Notebook(self)
        tabs.pack(fill="both", expand=True)
        camera = _page(tabs, "Cameras")
        display = _page(tabs, "Displays")
        self.camera_choice = tk.StringVar(self)
        self.display_choice = tk.StringVar(self)
        self.camera_combo = ttk.Combobox(camera, textvariable=self.camera_choice, state="disabled")
        self.camera_combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.camera_combo.bind("<<ComboboxSelected>>", self.load_camera)
        self._widgets.append(self.camera_combo)
        self.camera_values = {}
        for row, (name, label) in enumerate(CAMERA_FIELDS, 1):
            self._entry(camera, self.camera_values, name, label, row)
        self._entry(camera, self.camera_values, "width", "Final output width (pixels)", 7)
        self._entry(camera, self.camera_values, "height", "Final output height (pixels)", 8)
        self.camera_status = tk.StringVar(self, camera_calibration_status(None))
        self._label(camera, self.camera_status, 9)
        buttons = ttk.Frame(camera)
        buttons.grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self._button(buttons, "New", self.new_camera)
        self._button(buttons, "Save profile", self.save_camera)
        self._button(buttons, "Use profile", self.use_camera)
        if calibrate_camera is not None:
            self._button(buttons, "Calibrate profile", self.calibrate_camera)
        instructions = ("Camera choices take effect in the next live session. Keep the lens, zoom, "
                        "output size and OBS transforms fixed after calibration.")
        if calibrate_camera is None:
            instructions += (" To calibrate, start a live session, select the camera and profile, "
                             "then choose Calibrate camera in Setup.")
        ttk.Label(camera, text=instructions, wraplength=500).grid(
                      row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))
        legacy = ttk.Frame(camera)
        legacy.grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self._button(legacy, "Import old calibration as unverified", self.import_legacy_camera)
        self.display_combo = ttk.Combobox(display, textvariable=self.display_choice, state="disabled")
        self.display_combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.display_combo.bind("<<ComboboxSelected>>", self.load_display)
        self._widgets.append(self.display_combo)
        self.display_values = {}
        self._entry(display, self.display_values, "name", "Profile name", 1)
        for row, (_name, label, _lo, _hi, _step, _default, _places) in enumerate(GRID_FIELDS, 2):
            self._entry(display, self.display_values, _name, label, row)
        self._entry(display, self.display_values, "viewportMarkerSize", "Marker size (pixels)", 5)
        self._entry(display, self.display_values, "viewportMarkerInset", "Marker inset (pixels)", 6)
        for row, (name, label) in enumerate((("viewportMarkersEnabled", "Show ArUco markers"),
                                           ("physicalGridAutoApply", "Keep physical grid size when the view changes")), 7):
            ttk.Label(display, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
            self.display_values[name] = tk.StringVar(self, "Not saved")
            combo = ttk.Combobox(display, textvariable=self.display_values[name],
                                 values=("Not saved", "On", "Off"), state="readonly", width=12)
            combo.grid(row=row, column=1, sticky="ew", pady=3)
            self._widgets.append(combo)
        self.include_display = tk.BooleanVar(self, False)
        self._target_display_id = None
        check = ttk.Checkbutton(display, text="Save the preferred monitor with this profile", variable=self.include_display)
        check.grid(row=9, column=0, columnspan=2, sticky="w", pady=3)
        self._widgets.append(check)
        buttons = ttk.Frame(display)
        buttons.grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self._button(buttons, "New", self.new_display)
        self._button(buttons, "Read current VTT settings", self.read_current_display)
        self._button(buttons, "Save profile", self.save_display)
        ttk.Label(display, text="Apply saved grid and marker settings from VTT Controls. The saved monitor "
                  "sets a preference; use Move Foundry to place the window.", wraplength=500).grid(
                      row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))
        label = ttk.Label(self, textvariable=self.message, wraplength=600)
        label.pack(fill="x", pady=(6, 0))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width)))
        for page in (camera, display):
            _bind_scroll(page)
        self.refresh(force=True)
        camera_id = _active_id(self.store, "camera")
        display_id = _active_id(self.store, "display")
        if camera_id:
            self.camera_choice.set(next((name for name, profile in self._camera_profiles.items() if profile["id"] == camera_id), ""))
            self.load_camera()
        if display_id:
            self.display_choice.set(next((name for name, profile in self._display_profiles.items() if profile["id"] == display_id), ""))
            self.load_display()
        self.set_enabled(False)

    def _entry(self, parent, variables, name, label, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        variables[name] = tk.StringVar(self)
        entry = ttk.Entry(parent, textvariable=variables[name], width=32)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        self._widgets.append(entry)

    def _button(self, parent, label, command):
        button = ttk.Button(parent, text=label, command=command)
        button.pack(side="left", padx=(0, 6))
        self._widgets.append(button)

    def _label(self, parent, variable, row):
        label = ttk.Label(parent, textvariable=variable, wraplength=500)
        label.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(180, event.width)))

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        for widget in self._widgets:
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox) else "normal") if enabled else "disabled")

    def refresh(self, *, force=False):
        if not force and time.monotonic() - self._last_refresh < 2:
            return
        try:
            self._camera_profiles = {profile["name"]: profile for profile in self.store.list_profiles("camera")}
            self._display_profiles = {profile["name"]: profile for profile in self.store.list_profiles("display")}
        except (ValueError, OSError) as exc:
            self.message.set(str(exc))
            self._last_refresh = time.monotonic()
            return
        self.camera_combo.configure(values=tuple(self._camera_profiles))
        self.display_combo.configure(values=tuple(self._display_profiles))
        camera = next((profile for profile in self._camera_profiles.values() if profile["id"] == self._camera_id), None)
        if camera is not None:
            if self._camera_form_snapshot() == self._camera_snapshot:
                self.camera_choice.set(camera["name"])
                self.load_camera()
            else:
                self.camera_status.set(camera_calibration_status(camera))
        display = next((profile for profile in self._display_profiles.values() if profile["id"] == self._display_id), None)
        if display is not None and self._display_form_snapshot() == self._display_snapshot:
            self.display_choice.set(display["name"])
            self.load_display()
        self._last_refresh = time.monotonic()

    def _camera_form_snapshot(self):
        return {key: str(variable.get()) for key, variable in self.camera_values.items()}

    def _display_form_snapshot(self):
        return {**{key: str(variable.get()) for key, variable in self.display_values.items()},
                "includeDisplay": self.include_display.get(), "targetDisplayId": self._target_display_id}

    def _access(self):
        if self._require_access is not None:
            self._require_access()
        if not self._enabled:
            raise PermissionError("Sign in before changing hardware profiles.")

    def new_camera(self):
        self._camera_id = None
        self.camera_choice.set("")
        for variable in self.camera_values.values():
            variable.set("")
        self.camera_status.set("New camera profile. No calibration is associated with it.")
        self._camera_snapshot = self._camera_form_snapshot()

    def load_camera(self, _event=None):
        profile = self._camera_profiles.get(self.camera_choice.get())
        if profile is None:
            return
        self._camera_id = profile["id"]
        for name, _label in CAMERA_FIELDS:
            self.camera_values[name].set(profile.get(name, ""))
        size = profile.get("resolution") or ("", "")
        self.camera_values["width"].set(size[0])
        self.camera_values["height"].set(size[1])
        self.camera_status.set(camera_calibration_status(profile))
        self._camera_snapshot = self._camera_form_snapshot()

    def save_camera(self):
        try:
            self._access()
            values = camera_form_values({key: value.get() for key, value in self.camera_values.items()})
            profile = self.store.upsert_profile("camera", values, profile_id=self._camera_id)
            self._camera_id = profile["id"]
            self.camera_choice.set(profile["name"])
            self.camera_status.set(camera_calibration_status(profile))
            self._saved(f"Saved camera profile {profile['name']}.")
            self.load_camera()
        except Exception as exc:
            self.message.set(str(exc))

    def use_camera(self):
        try:
            self._access()
            if self._camera_id is None:
                raise ValueError("Save or choose a camera profile first.")
            if self._camera_form_snapshot() != self._camera_snapshot:
                raise ValueError("Save these profile changes before selecting it.")
            self.store.select_profile("camera", self._camera_id)
            self.message.set("Camera profile selected for the next live session.")
        except Exception as exc:
            self.message.set(str(exc))

    def calibrate_camera(self):
        try:
            self._access()
            profile = self.store.get_profile("camera", self._camera_id) if self._camera_id else None
            if profile is None:
                raise ValueError("Save or choose a camera profile first.")
            self._calibrate_camera(profile)
        except Exception as exc:
            self.message.set(str(exc))

    def import_legacy_camera(self):
        try:
            self._access()
            profile = self.store.import_legacy_camera()
            if profile is None:
                self.message.set("No old calibration files were found.")
                return
            self._saved("Old calibration imported as unverified. Original files were preserved.")
            self.camera_choice.set(profile["name"])
            self.load_camera()
        except Exception as exc:
            self.message.set(str(exc))

    def new_display(self):
        self._display_id = None
        self.display_choice.set("")
        for name, variable in self.display_values.items():
            variable.set("Not saved" if name in {"viewportMarkersEnabled", "physicalGridAutoApply"} else "")
        self.include_display.set(False)
        self._target_display_id = None
        self._display_saved_keys = set()
        self._display_snapshot = self._display_form_snapshot()

    def load_display(self, _event=None):
        profile = self._display_profiles.get(self.display_choice.get())
        if profile is None:
            return
        self.new_display()
        self._display_id = profile["id"]
        self.display_choice.set(profile["name"])
        self.display_values["name"].set(profile["name"])
        for name, value in profile["settings"].items():
            self.display_values[name].set(("On" if value else "Off") if isinstance(value, bool) else value)
        self._display_saved_keys = set(profile["settings"])
        self._target_display_id = profile.get("targetDisplayId")
        self.include_display.set(self._target_display_id is not None)
        self._display_snapshot = self._display_form_snapshot()

    def read_current_display(self):
        try:
            self._access()
            status = self._read_vtt()
            if not status.get("connected") or not status.get("supported") or status.get("blocked"):
                raise ValueError("Connect Foundry and open a scene before reading its display settings.")
            settings = display_settings({key: value for key, value in (status.get("settings") or {}).items()
                                         if key in DISPLAY_SETTINGS}, complete=True)
            for name, value in settings.items():
                self.display_values[name].set(("On" if value else "Off") if isinstance(value, bool) else value)
            if self._display is not None:
                self._target_display_id = self._display.status().get("selectedDisplay")
            self.message.set("Current grid and marker settings loaded. Name the profile and save it.")
        except Exception as exc:
            self.message.set(str(exc))

    def save_display(self):
        try:
            self._access()
            name = self.display_values["name"].get().strip()
            if not name:
                raise ValueError("Enter a display profile name.")
            values = {}
            for key in DISPLAY_SETTINGS:
                value = self.display_values[key].get()
                if key in {"viewportMarkersEnabled", "physicalGridAutoApply"}:
                    if value in {"On", "Off"}:
                        values[key] = value == "On"
                elif str(value).strip():
                    values[key] = value
            if self._display_saved_keys - values.keys():
                raise ValueError("Keep the saved settings, or create a new profile with fewer settings.")
            settings = display_settings(values)
            target = self._target_display_id if self.include_display.get() else None
            if self.include_display.get() and target is None and self._display is not None:
                target = self._display.status().get("selectedDisplay")
            if self.include_display.get() and target is None:
                raise ValueError("Choose a preferred monitor in VTT Controls first.")
            profile = self.store.upsert_profile("display", {"name": name, "settings": settings,
                "targetDisplayId": target}, profile_id=self._display_id)
            self._display_id = profile["id"]
            self._display_saved_keys = set(profile["settings"])
            self.display_choice.set(profile["name"])
            self._saved(f"Saved display profile {profile['name']}.")
            self.load_display()
        except Exception as exc:
            self.message.set(str(exc))

    def _saved(self, message):
        self.refresh(force=True)
        self.message.set(message)
        if self._changed is not None:
            self._changed()
