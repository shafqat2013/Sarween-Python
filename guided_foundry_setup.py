"""One session-start transaction for Foundry, the selected TV and its overlays.

No camera is opened here. The main window owns polling/cancellation; an isolated
startup controller prevents the settings tab from consuming this flow's work.
"""
from __future__ import annotations

from copy import deepcopy
import json
import time
import tkinter as tk
from tkinter import ttk


def load_setup_preferences(path=None):
    from app_paths import data_path
    try:
        values = json.loads((path or data_path("session_setup.json")).read_text(encoding="utf-8"))
        if isinstance(values, dict) and type(values.get("gridLabelsVisible")) is bool:
            return {"gridLabelsVisible": values["gridLabelsVisible"]}
    except (OSError, ValueError):
        pass
    return {}


class GuidedFoundryCoordinator:
    """Main-thread coordinator; Foundry launch and placement retain their workers."""

    def __init__(self, *, service, store, controller, display_controller,
                 require_access=None, read_status=None, request=None, clock=None):
        self.service, self.store, self.controller = service, store, controller
        self.display = display_controller
        self.require_access = require_access
        if read_status is None or request is None:
            import foundryoutput as fo
            read_status = read_status or fo.get_vtt_status
            request = request or (lambda action, values: fo.request_vtt_command(
                action, values, return_request_id=True))
        self.read_status, self.request = read_status, request
        self.clock = clock or time.monotonic
        self._state = {"phase": "idle", "ready": False, "message": "Ready to start Foundry.",
                       "error": None, "warning": ""}
        self._profile = self._settings = self._pending = None
        self._placement_started = False
        self._deadline = 0

    def _set(self, phase, message, **values):
        self._state.update(phase=phase, message=message, ready=phase == "ready", **values)
        return dict(self._state)

    def validate(self, choices):
        values = deepcopy(choices)
        from foundry_startup import normalize_preferences
        values["startup"] = normalize_preferences(values["startup"])
        profile_id = values.get("profileId")
        profile = self.store.get_profile("display", profile_id) if profile_id else None
        if profile_id and profile is None:
            raise ValueError("The display profile was removed. Choose a saved profile again.")
        marker_profile = profile or self.store.active_profile("display")
        if values.get("printed"):
            if marker_profile is None or not marker_profile.get("printedMarkerLayout"):
                raise ValueError("Printed markers need a registered display profile. Choose screen markers for now; align printed markers in VTT Controls → Display.")
            from printed_markers import validate_layout
            validate_layout(marker_profile["printedMarkerLayout"])
        display = values.get("display") or {}
        if values["startup"]["placeWindow"]:
            connected = self.display.status().get("displays", [])
            # Never substitute a disconnected numeric ID or another same-name TV.
            target = next((row for row in connected if str(row["id"]) == str(display.get("id"))
                           and (not display.get("name") or row["name"] == display["name"])), None)
            if target is None:
                matches = [row for row in connected if display.get("name") and row["name"] == display["name"]]
                target = matches[0] if len(matches) == 1 else None
            if target is None:
                raise ValueError("The selected TV is no longer connected. Go Back, refresh hardware and choose the TV by name; or turn off automatic window placement.")
            values["display"] = dict(target)
        values["profile"] = profile
        values["markerProfile"] = marker_profile
        return values

    def begin(self, choices):
        try:
            if self.require_access:
                self.require_access()
            if self._state["phase"] not in {"idle", "failed", "cancelled", "ready"}:
                return False
            values = self.validate(choices)
            self._profile = deepcopy(values["profile"])
            self._marker_profile = deepcopy(values["markerProfile"])
            from hardware_profiles_ui import display_settings
            settings = (display_settings(self._profile["settings"]) if self._profile else {})
            settings.update(viewportMarkersEnabled=not values.get("printed", False),
                            gridLabelsVisible=bool(values.get("gridLabels", False)))
            self._settings = settings
            self._printed = bool(values.get("printed", False))
            self._pending = None
            self._placement_started = False
            self._state.update(error=None, warning="", ready=False)
            if values["startup"]["placeWindow"]:
                self.display.configure(display_id=values["display"]["id"], auto=False)
            else:
                # The guided checkbox owns placement for this startup; a saved
                # watcher must not move windows after the user turns it off.
                self.display.configure(auto=False)
            self.service.start()
            self.controller.set_enabled(True)
            if not self.controller.start(values["startup"], password=values.get("password", "")):
                raise ValueError("Foundry startup is already busy. Wait for it to finish or cancel it, then try again.")
            self._set("foundry", "Starting Foundry and connecting its Sarween module…")
            return True
        except Exception as exc:
            self._set("failed", str(exc), error=str(exc))
            return False

    def poll(self):
        try:
            if self.require_access:
                self.require_access()
            phase = self._state["phase"]
            if phase == "foundry":
                status = self.controller.status()
                if status["phase"] in {"failed", "cancelled"}:
                    return self._set(status["phase"], status["message"], error=status["message"])
                if status["phase"] != "ready":
                    return self._set("foundry", status["message"])
                if self.controller.take_placement_request():
                    if self.display.place():
                        self._placement_started = True
                        return self._set("placement", "Moving Foundry to the selected TV and entering full screen…")
                    self._state["warning"] = self.display.status().get("message") or "Place Foundry on the selected TV manually."
                    return self._set("placement_attention", self._state["warning"])
                self._deadline = self.clock() + 45
                self._set("settings", "Applying the TV profile, markers and grid labels…")
            elif phase == "placement":
                placement = self.display.status()
                if placement.get("busy"):
                    return dict(self._state)
                self._placement_started = False
                message = placement.get("message", "")
                if not message.startswith("Foundry is full screen on "):
                    self._state["warning"] = message or "Place Foundry on the selected TV manually."
                    return self._set("placement_attention", self._state["warning"])
                self._deadline = self.clock() + 45
                self._set("settings", "Applying the TV profile, markers and grid labels…")
            if self._state["phase"] == "settings":
                return self._poll_settings()
        except Exception as exc:
            self.controller.cancel()
            self._cancel_placement()
            self._set("failed", str(exc), error=str(exc))
        return dict(self._state)

    def confirm_placement(self):
        if self._state["phase"] != "placement_attention":
            return False
        if self.require_access:
            self.require_access()
        # The user's confirmation refers to the final TV viewport. Apply the
        # physical grid only now, rather than sizing against the laptop screen.
        self._state["warning"] = ""
        self._deadline = self.clock() + 45
        self._set("settings", "Applying the TV profile, markers and grid labels…")
        return True

    def _poll_settings(self):
        status = self.read_status()
        if self.clock() > self._deadline:
            return self._set("failed", "The TV settings were not confirmed. Check the Foundry connection, then retry Start session.", error="TV settings timed out.")
        if self._pending:
            if not status.get("connected") or not status.get("supported"):
                return self._set("failed", "Foundry disconnected while applying TV settings. Retry Start session.", error="Foundry disconnected.")
            result = status.get("lastResult") or {}
            if status.get("sceneId") != self._pending["sceneId"]:
                return self._set("failed", "The Foundry scene changed during setup. Check the scene and retry Start session.", error="Scene changed.")
            if result.get("requestId") == self._pending["requestId"]:
                if result.get("success") is not True or result.get("sceneId") != self._pending["sceneId"]:
                    message = result.get("message") or "Foundry could not apply the TV settings. Retry Start session."
                    return self._set("failed", message, error=message)
                if status.get("blocked") or status.get("pending"):
                    return dict(self._state)
                if self._marker_profile:
                    current = self.store.get_profile("display", self._marker_profile["id"])
                    if current != self._marker_profile:
                        raise ValueError("The display profile changed during setup. Review it and retry Start session.")
                    if self._profile:
                        self.store.select_profile("display", self._profile["id"])
                    elif self.store.active_profile("display") != self._marker_profile:
                        raise ValueError("The active TV profile changed during setup. Review it and retry Start session.")
                    if current.get("printedMarkersEnabled", False) != self._printed:
                        self.store.set_printed_markers_enabled(current["id"], self._printed, expected_profile=current)
                warning = self._state["warning"]
                return self._set("ready", "Foundry and TV settings are ready." + (" " + warning if warning else ""))
            if (not status.get("connected") or not status.get("supported")):
                return self._set("failed", "Foundry disconnected while applying TV settings. Retry Start session.", error="Foundry disconnected.")
            return dict(self._state)
        if not status.get("connected") or not status.get("supported") or status.get("blocked") or status.get("pending"):
            return self._set("settings", status.get("message") or "Waiting for Foundry to accept TV settings…")
        if "settings.set" not in (status.get("capabilities") or []):
            raise ValueError("Reload Foundry with the updated Sarween module to apply the TV settings.")
        request_id = self.request("settings.set", {"settings": deepcopy(self._settings)})
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Foundry did not accept the TV settings. Check the connection and retry Start session.")
        self._pending = {"requestId": request_id, "sceneId": status.get("sceneId")}
        return dict(self._state)

    def _cancel_placement(self):
        if self._placement_started:
            # Changing the display generation cancels this flow's in-flight AX
            # work. No native window is closed and auto-placement stays off.
            self.display.set_enabled(False)
            self.display.set_enabled(True)
            self._placement_started = False

    def cancel(self):
        self.controller.cancel()
        self._cancel_placement()
        self._set("cancelled", "Session setup cancelled. Foundry stays open.")


class GuidedFoundrySetup(ttk.Frame):
    """Compact configuration step; its parent owns the single Start session button."""

    def __init__(self, parent, *, service, access=None, require_access=None,
                 display_controller=None, store=None, controller=None, changed=None):
        super().__init__(parent, padding=4)
        if store is None:
            from hardware_profiles import HardwareProfiles
            store = HardwareProfiles()
        if controller is None:
            from foundry_startup import FoundryStartupController
            controller = FoundryStartupController()
        if display_controller is None:
            from foundry_display import get_foundry_display_controller
            display_controller = get_foundry_display_controller()
        self.store, self.controller = store, controller
        self._require_access = require_access or (access.require if access is not None else None)
        self.coordinator = GuidedFoundryCoordinator(service=service, store=store, controller=controller,
            display_controller=display_controller, require_access=self._require_access)
        self._changed = changed
        self._display, self._displays = {}, []
        self._enabled, self._closed = True, False
        self._preferences_saved = False
        self._widgets = []
        self.columnconfigure(1, weight=1)
        self.target = tk.StringVar(self, "TV: choose a display in Hardware.")
        self._label(self, textvariable=self.target).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        defaults = controller.defaults()
        self.values = {key: tk.StringVar(self, defaults[key]) for key in
                       ("appPath", "dataPath", "baseUrl", "worldId", "username")}
        self.password = tk.StringVar(self)
        self.unpause = tk.BooleanVar(self, defaults["unpause"])
        self.place_window = tk.BooleanVar(self, defaults["placeWindow"])
        self.profile_choice = tk.StringVar(self)
        self.marker_choice = tk.StringVar(self, "Screen markers")
        labels_visible = load_setup_preferences().get("gridLabelsVisible", False)
        try:
            status = self.coordinator.read_status()
            if status.get("connected") and type((status.get("settings") or {}).get("gridLabelsVisible")) is bool:
                labels_visible = status["settings"]["gridLabelsVisible"]
        except Exception:
            pass
        self.grid_labels = tk.BooleanVar(self, labels_visible)
        self.world_choice = tk.StringVar(self)
        self._profiles, self._worlds = {}, {}
        fields = (("TV profile", self.profile_choice), ("Markers", self.marker_choice),
                  ("Foundry world", self.world_choice))
        for row, (label, variable) in enumerate(fields, 1):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            combo = ttk.Combobox(self, textvariable=variable, state="readonly")
            combo.grid(row=row, column=1, sticky="ew", pady=4)
            self._widgets.append(combo)
            if row == 1:
                self.profile_combo = combo
            elif row == 2:
                self.marker_combo = combo
                combo.configure(values=("Screen markers", "Printed markers (already aligned)"))
            else:
                self.world_combo = combo
        for row, (label, variable, secret) in enumerate((
                ("Foundry user", self.values["username"], False),
                ("Password (optional)", self.password, True)), 4):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            widget = ttk.Entry(self, textvariable=variable, show="•" if secret else "")
            widget.grid(row=row, column=1, sticky="ew", pady=4)
            self._widgets.append(widget)
        for row, (label, variable) in enumerate((("Unpause after joining", self.unpause),
                ("Move Foundry to this TV + full screen", self.place_window),
                ("Show grid cell labels", self.grid_labels)), 6):
            widget = ttk.Checkbutton(self, text=label, variable=variable)
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
            self._widgets.append(widget)
        self.note = tk.StringVar(self, "Start session will open or reuse Foundry, connect the module and apply these TV settings. Passwords are never saved.")
        self._label(self, textvariable=self.note).grid(row=9, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self._advanced = ttk.LabelFrame(self, text="Application and connection", padding=6)
        self._advanced.columnconfigure(1, weight=1)
        self.advanced_button = ttk.Button(self, text="Show advanced connection settings", command=self._toggle_advanced)
        self.advanced_button.grid(row=10, column=0, columnspan=2, sticky="w", pady=(8, 2))
        self._widgets.append(self.advanced_button)
        for row, (key, label) in enumerate((("appPath", "Foundry app"), ("dataPath", "Data folder"), ("baseUrl", "Local address"))):
            ttk.Label(self._advanced, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(self._advanced, textvariable=self.values[key])
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            self._widgets.append(entry)
        refresh = ttk.Button(self._advanced, text="Refresh worlds", command=self.refresh_choices)
        refresh.grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        self._widgets.append(refresh)
        self.refresh_choices()
        self.profile_combo.bind("<<ComboboxSelected>>", self._profile_selected)
        for variable in (self.profile_choice, self.marker_choice, self.world_choice,
                         self.unpause, self.place_window, self.grid_labels, self.values["username"]):
            variable.trace_add("write", lambda *_args: self._changed() if self._changed else None)

    @staticmethod
    def _label(parent, **options):
        label = ttk.Label(parent, wraplength=500, justify="left", **options)
        def wrap(event):
            if event.width > 1:
                width = max(100, event.width - 4)
                if int(label.cget("wraplength")) != width:
                    label.configure(wraplength=width)
        label.bind("<Configure>", wrap)
        return label

    def refresh_choices(self):
        try:
            active = self.store.active_profile("display")
            current = self.profile_choice.get()
            self._profiles = {row["name"]: row for row in self.store.list_profiles("display")}
            self.profile_combo.configure(values=("Keep current TV grid settings", *self._profiles))
            self.profile_choice.set(current if current in self._profiles else
                                    active["name"] if active else "Keep current TV grid settings")
            self._worlds = {row["name"] + " (" + row["id"] + ")": row["id"]
                            for row in self.controller.worlds(self.values["dataPath"].get())}
            self.world_combo.configure(values=tuple(self._worlds))
            self.world_choice.set(next((name for name, identity in self._worlds.items()
                                        if identity == self.values["worldId"].get()), ""))
            if not self._worlds:
                self.note.set("No Foundry worlds found. Expand advanced connection settings and check the data folder.")
            self._profile_selected()
        except (OSError, ValueError) as exc:
            self.note.set(str(exc))

    def _profile_selected(self, _event=None):
        profile = self._profiles.get(self.profile_choice.get())
        self.marker_choice.set("Printed markers (already aligned)" if profile and profile.get("printedMarkersEnabled") else "Screen markers")

    def _toggle_advanced(self):
        if self._advanced.winfo_manager():
            self._advanced.grid_remove()
            self.advanced_button.configure(text="Show advanced connection settings")
        else:
            self._advanced.grid(row=11, column=0, columnspan=2, sticky="ew", pady=(4, 0))
            self.advanced_button.configure(text="Hide advanced connection settings")

    def set_display(self, display):
        self._display = dict(display or {})
        name = self._display.get("name") or "Display " + str(self._display.get("index", ""))
        self.target.set("TV: " + name)

    def set_inventory(self, displays, selected_display_index):
        self._displays = list(displays)
        self.set_display(next((row for row in displays if row["index"] == selected_display_index), {}))

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        for widget in self._widgets:
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox) else "normal") if enabled else "disabled")

    def validate(self):
        values = {key: variable.get() for key, variable in self.values.items()}
        values["worldId"] = self._worlds.get(self.world_choice.get(), "")
        values.update(unpause=self.unpause.get(), placeWindow=self.place_window.get())
        profile = self._profiles.get(self.profile_choice.get())
        return self.coordinator.validate({"startup": values, "password": self.password.get(),
            "profileId": profile["id"] if profile else None, "display": self._display,
            "printed": self.marker_choice.get() == "Printed markers (already aligned)",
            "gridLabels": self.grid_labels.get()})

    def begin(self):
        if not self._enabled or self._closed:
            return False
        try:
            choices = self.validate()
        except Exception as exc:
            self.coordinator._set("failed", str(exc), error=str(exc))
            self.note.set(str(exc))
            return False
        accepted = self.coordinator.begin(choices)
        if accepted:
            self._preferences_saved = False
            self.password.set("")
            self.set_enabled(False)
        self.note.set(self.coordinator._state["message"])
        return accepted

    def start(self, display_index=None):
        if display_index is not None:
            self.set_inventory(self._displays, display_index)
        return self.begin()

    def poll(self):
        status = self.coordinator.poll()
        if status.get("ready") and not self._preferences_saved:
            from app_paths import atomic_write_json, data_path
            try:
                atomic_write_json(data_path("session_setup.json"), {"gridLabelsVisible": bool(self.grid_labels.get())})
                self._preferences_saved = True
            except OSError as exc:
                status = {**status, "message": status["message"] + " Could not save overlay preferences: " + str(exc)}
        self.note.set(status["message"])
        return status

    status = poll

    def confirm_placement(self):
        return self.coordinator.confirm_placement()

    def cancel(self):
        self.password.set("")
        self.coordinator.cancel()

    def close(self):
        if not self._closed:
            self.cancel()
            self._closed = True
