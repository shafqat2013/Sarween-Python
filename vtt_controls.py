"""Native desktop controls for the connected Sarween Foundry module."""

import math
import time
import tkinter as tk
from tkinter import ttk

from foundry_grid_controls import FoundryGridControls
from guided_capture import MAX_CAPTURE_MINIS


NUMBER_SETTINGS = {
    "viewportMarkerSize": ("Marker size (pixels)", 72, 240, 4, 120),
    "viewportMarkerInset": ("Marker inset (pixels)", 0, 80, 2, 8),
    "defaultMovementSpeed": ("Fallback speed (feet)", 0, 120, 5, 30),
}
BOOL_SETTINGS = {
    "viewportMarkersEnabled": ("Show ArUco corner markers", True),
    "gridLabelsVisible": ("Show grid cell labels on the map", False),
    "animateTokenMovement": ("Animate token movement", True),
    "physicalTapSelection": ("Physical tap selects a mini", False),
    "physicalGridAutoApply": ("Keep physical grid size when the view changes", True),
    "debug": ("Write Foundry debug logs", True),
    "foundryUIHidden": ("Hide Foundry interface on the TV", False),
}


def parse_settings(values, names):
    result = {}
    for name in names:
        value = values[name]
        if name in BOOL_SETTINGS:
            if not isinstance(value, bool):
                raise ValueError(f"Choose on or off for {BOOL_SETTINGS[name][0].lower()}.")
        elif name in NUMBER_SETTINGS:
            label, low, high, _step, _default = NUMBER_SETTINGS[name]
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Enter a number for {label.lower()}.") from None
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{label} must be between {low:g} and {high:g}.")
        else:
            raise ValueError("Unknown Foundry setting.")
        result[name] = value
    return result


def capture_participants(rows, token_ids):
    selected = []
    for mini_id, enabled, token_id in rows:
        if not enabled:
            continue
        if token_id not in token_ids:
            raise ValueError("Choose a Foundry token for each selected mini.")
        selected.append({"miniId": mini_id, "tokenId": token_id})
    if not 1 <= len(selected) <= MAX_CAPTURE_MINIS:
        raise ValueError("Choose one to six minis for capture.")
    if len({row["tokenId"] for row in selected}) != len(selected):
        raise ValueError("Choose a different Foundry token for every mini.")
    return selected


def capture_active(capture):
    return bool(capture.get("active") or capture.get("recording")
                or capture.get("phase") in {"Starting", "Saving", "Restoring"})


def _read_status():
    import foundryoutput
    return foundryoutput.get_vtt_status()


def _request(action, values=None):
    import foundryoutput
    return foundryoutput.request_vtt_command(action, values)


def _scroll_page(notebook, title):
    page = ttk.Frame(notebook)
    notebook.add(page, text=title)
    canvas = tk.Canvas(page, highlightthickness=0, height=340, width=460)
    scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    canvas.configure(yscrollcommand=scrollbar.set)
    content = ttk.Frame(canvas, padding=(6, 10, 10, 8))
    window = canvas.create_window(0, 0, window=content, anchor="nw")
    content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=event.width))
    content.columnconfigure(0, weight=1)
    content._scroll_canvas = canvas
    return content


def _wrapped_label(parent, **options):
    """Use allocated width, including when a token name or status is very long."""
    label = ttk.Label(parent, anchor="w", justify="left", wraplength=480, **options)
    label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
    return label


def _compact_notebook_style(parent):
    """Keep nested tabs from spending the small VTT pane on Aqua gutters."""
    style = ttk.Style(parent)
    name = "Sarween.VTT.TNotebook"
    try:
        if style.theme_use() == "aqua":
            element = "Sarween.VTT.Notebook.client"
            if element not in style.element_names():
                style.element_create(element, "from", "clam", "Notebook.client")
            style.layout(name, [(element, {"sticky": "nswe"})])
        options = {"padding": 0, "borderwidth": 0, "tabmargins": 0}
        background = style.lookup("TFrame", "background")
        if background:
            options["background"] = background
        style.configure(name, **options)
    except tk.TclError:
        style.configure(name, padding=0)
    return name


def compact_status_message(connection, message, limit=160):
    """One-line summary; the complete action message stays on Startup."""
    text = " · ".join(part for part in (str(connection).strip(), " ".join(str(message).split())) if part)
    return text if len(text) <= limit else text[:limit - 1] + "…"


class VTTControls(ttk.Frame):
    def __init__(self, parent, *, require_access=None, foundry_service=None,
                 read_status=None, request=None, display_controller=None, live_session=False,
                 printed_marker_controller=None, profile_store=None):
        super().__init__(parent)
        self._require_access = require_access
        self._service = foundry_service
        self._live_session = bool(live_session)
        self._printed_marker_controller = printed_marker_controller
        self._profile_store = profile_store
        self._read_status = read_status or _read_status
        self._request = request or _request
        self._enabled = False
        self._last_refresh = float("-inf")
        self._last_status = {}
        self._syncing = False
        self._dirty = set()
        self._submitted = {}
        self._remote_settings = {}
        self._notice = ""
        self._actions = []
        self._setting_widgets = []
        self._tokens = {}
        self._token_labels = {}
        self._mini_labels = {}
        self._library_rows = []
        self._capture_initialized = set()
        self._capture_rows = []
        self._choice_scene_id = None
        self._choice_token_ids = {}
        self._assignment_request = None
        self._display_busy = False
        self.variables = {}
        self.connection = tk.StringVar(self, "Foundry: disconnected")
        self.relay = tk.StringVar(self, "Local relay: stopped")
        self.scene = tk.StringVar(self, "Scene: unavailable")
        self.message = tk.StringVar(self)
        self.footer_message = tk.StringVar(self)
        self.pages = ttk.Notebook(self, style=_compact_notebook_style(self))
        self.pages.pack(fill="both", expand=True)
        startup = _scroll_page(self.pages, "Startup")
        display = _scroll_page(self.pages, "Display")
        movement = _scroll_page(self.pages, "Movement")
        capture = _scroll_page(self.pages, "Capture")
        assignment = _scroll_page(self.pages, "Assignment")
        advanced = _scroll_page(self.pages, "Advanced")
        # Relay commands and detailed connection information belong with launch
        # controls, inside the scrollable Startup page. Other pages retain a
        # short connection/action summary without losing form space.
        connection = ttk.Frame(startup)
        connection.pack(fill="x", pady=(0, 6))
        buttons = ttk.Frame(connection)
        buttons.pack(fill="x", pady=(0, 4))
        self.connect_button = ttk.Button(buttons, text="Start relay", command=self.toggle_connection)
        self.connect_button.pack(side="left")
        self._button(buttons, "Reload Foundry", "lifecycle.reload").pack(side="left", padx=(6, 0))
        for variable in (self.relay, self.scene, self.message):
            _wrapped_label(connection, textvariable=variable).pack(fill="x", anchor="w")
        self.grid_controls = FoundryGridControls(display, require_access=require_access)
        self.grid_controls.set_enabled(self._enabled)
        self.grid_controls.pack(fill="x", pady=(0, 8))
        self._build_display(display, display_controller)
        self.display_controller = self._display
        from foundry_startup_ui import FoundryStartupControls
        self.foundry_startup = FoundryStartupControls(startup, require_access=require_access,
            foundry_service=foundry_service, display_controller=self._display)
        self.foundry_startup.pack(fill="x")
        from hardware_profiles_ui import DisplayProfilePicker
        self.display_profiles = DisplayProfilePicker(display, require_access=require_access,
            display_controller=self._display,
            can_apply=lambda: self._allowed("settings.set") and not self._display_busy)
        self.display_profiles.pack(fill="x", pady=(0, 8), before=self.grid_controls)
        self._build_movement(movement)
        self._build_capture(capture)
        self._build_assignment(assignment)
        from token_creation_controls import TokenCreationControls
        self.token_creation = TokenCreationControls(assignment, require_access=require_access,
            read_status=self._read_status, can_create=lambda: not self._display_busy)
        self.token_creation.pack(fill="x", pady=(8, 0))
        self._build_advanced(advanced)
        for page in (startup, display, movement, capture, assignment, advanced):
            self._bind_scroll(page)
        self.footer = ttk.Label(self, textvariable=self.footer_message, width=1, anchor="w")
        # Reserve this line before the notebook takes its expanding parcel.
        self.footer.pack(side="bottom", before=self.pages, fill="x", pady=(2, 0))
        for variable in (self.connection, self.message):
            variable.trace_add("write", self._sync_status_summary)
        self._sync_status_summary()
        self.refresh()

    def _sync_status_summary(self, *_args):
        self.footer_message.set(compact_status_message(self.connection.get(), self.message.get()))

    def _button(self, parent, text, action, callback=None):
        button = ttk.Button(parent, text=text, command=callback or (lambda: self.command(action)))
        self._actions.append((button, action))
        return button

    def _setting(self, parent, name, row):
        if name in BOOL_SETTINGS:
            label, default = BOOL_SETTINGS[name]
            variable = tk.BooleanVar(self, default)
            widget = ttk.Checkbutton(parent, text=label, variable=variable)
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
        else:
            label, low, high, step, default = NUMBER_SETTINGS[name]
            variable = tk.StringVar(self, str(default))
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            widget = ttk.Spinbox(parent, textvariable=variable, from_=low, to=high,
                                 increment=step, width=8)
            widget.grid(row=row, column=1, sticky="e", pady=3)
        variable.trace_add("write", lambda *_args, name=name: self._edited(name))
        self.variables[name] = variable
        self._setting_widgets.append(widget)

    def _section(self, parent, title):
        section = ttk.LabelFrame(parent, text=title, padding=10)
        section.pack(fill="x", pady=(0, 8))
        section.columnconfigure(0, weight=1)
        return section

    def _bind_scroll(self, page):
        tag = f"VTTScroll{id(page)}"

        def scroll(event):
            direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else 1
            page._scroll_canvas.yview_scroll(direction, "units")
            return "break"

        self.bind_class(tag, "<MouseWheel>", scroll)
        self.bind_class(tag, "<Button-4>", scroll)
        self.bind_class(tag, "<Button-5>", scroll)

        def include(widget):
            widget.bindtags((tag, *widget.bindtags()))
            for child in widget.winfo_children():
                include(child)

        include(page)

    def _settings_button(self, section, names, row):
        self._button(section, "Apply settings", "settings.set",
                     lambda: self.apply_settings(names)).grid(row=row, column=0, columnspan=2,
                                                             sticky="w", pady=(6, 0))

    def _build_display(self, parent, controller):
        overlays = self._section(parent, "Map overlays")
        overlays.pack(before=self.grid_controls, fill="x", pady=(0, 8))
        names = ("viewportMarkersEnabled", "gridLabelsVisible")
        for row, name in enumerate(names):
            self._setting(overlays, name, row)
        self._settings_button(overlays, names, 2)
        markers = self._section(parent, "ArUco markers")
        markers.pack(before=self.grid_controls, fill="x", pady=(0, 8))
        names = ("viewportMarkerSize", "viewportMarkerInset")
        for row, name in enumerate(names):
            self._setting(markers, name, row)
        self._settings_button(markers, names, 2)
        from printed_marker_controls import PrintedMarkerControls
        self.printed_markers = PrintedMarkerControls(parent, require_access=self._require_access,
            controller=self._printed_marker_controller, store=self._profile_store,
            can_change=lambda: not (capture_active(self._last_status.get("capture") or {})
                                    or self._display_busy or self._last_status.get("pending")
                                    or self._last_status.get("blocked")))
        self.printed_markers.pack(before=self.grid_controls, fill="x", pady=(0, 8))
        self._display = controller
        if self._display is None:
            from foundry_display import get_foundry_display_controller
            self._display = get_foundry_display_controller()
        self._display.set_enabled(self._enabled)
        placement = self._section(parent, "Foundry window")
        self.display_choice = tk.StringVar(self)
        self.display_auto = tk.BooleanVar(self, False)
        self.display_message = tk.StringVar(self)
        self._display_names = {}
        self.display_combo = ttk.Combobox(placement, textvariable=self.display_choice,
                                         state="readonly", width=26)
        self.display_combo.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self.display_combo.bind("<<ComboboxSelected>>", lambda _event: self.configure_display())
        self.display_place = ttk.Button(placement, text="Move Foundry here + full screen",
                                        command=self.place_display)
        self.display_place.grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        self.display_auto_check = ttk.Checkbutton(placement, text="Automatically place newly opened Foundry windows",
                                                 variable=self.display_auto, command=self.configure_display)
        self.display_auto_check.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        label = _wrapped_label(placement, textvariable=self.display_message)
        label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=3)

    def _build_movement(self, parent):
        section = self._section(parent, "Selected mini")
        self.movement_message = tk.StringVar(self, "No mini selected")
        _wrapped_label(section, textvariable=self.movement_message).grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.movement_choice = tk.StringVar(self)
        self.movement_combo = ttk.Combobox(section, textvariable=self.movement_choice, state="readonly")
        self.movement_combo.grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        buttons = ttk.Frame(section)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        self._button(buttons, "Select mini", "movement.select", self.select_movement).pack(side="left")
        self._button(buttons, "Clear", "movement.select",
                     lambda: self.command("movement.select", {"tokenId": None})).pack(side="left", padx=4)
        buttons = ttk.Frame(section)
        buttons.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        self.undo_button = self._button(buttons, "Undo last move", "movement.undo")
        self.undo_button.pack(side="left")
        self.reset_button = self._button(buttons, "Reset movement", "movement.reset")
        self.reset_button.pack(side="left", padx=4)
        settings = self._section(parent, "Movement settings")
        names = ("animateTokenMovement", "defaultMovementSpeed")
        for row, name in enumerate(names):
            self._setting(settings, name, row)
        self._settings_button(settings, names, 2)
        ttk.Label(settings, text="Physical tap selection: paused").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_capture(self, parent):
        section = self._section(parent, "Guided camera capture")
        self.capture_message = tk.StringVar(self, "Choose one to six minis, then start recording.")
        section.columnconfigure(1, weight=1)
        label = _wrapped_label(section, textvariable=self.capture_message)
        label.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._capture_setup_widgets = []
        for index in range(MAX_CAPTURE_MINIS):
            enabled = tk.BooleanVar(self, False)
            name = tk.StringVar(self, f"Mini {index + 1}")
            token = tk.StringVar(self)
            choice = ttk.Frame(section)
            choice.grid(row=index + 1, column=0, sticky="ew", padx=(0, 10), pady=4)
            choice.columnconfigure(1, weight=1)
            check = ttk.Checkbutton(choice, variable=enabled)
            check.grid(row=0, column=0, sticky="nw")
            # A Checkbutton textvariable cannot wrap. Keep the check itself
            # separate so long mini names never push the token choice offscreen.
            name_label = _wrapped_label(choice, textvariable=name)
            name_label.configure(wraplength=220)
            name_label.grid(row=0, column=1, sticky="ew")
            combo = ttk.Combobox(section, textvariable=token, state="readonly", width=24)
            combo.grid(row=index + 1, column=1, sticky="ew", pady=3)
            self._capture_rows.append({"miniId": None, "enabled": enabled,
                                       "name": name, "token": token, "combo": combo, "check": check})
            check.configure(command=lambda row=self._capture_rows[-1]: self._capture_edited(row))
            name_label.bind("<Button-1>", lambda _event, check=check: check.invoke())
            combo.bind("<<ComboboxSelected>>", lambda _event, row=self._capture_rows[-1]: self._capture_edited(row))
            self._capture_setup_widgets.extend((check, combo))
        self.scenario = tk.StringVar(self, "Fog on")
        self.notes = tk.StringVar(self)
        form_row = MAX_CAPTURE_MINIS + 1
        ttk.Label(section, text="Fog / lighting label").grid(row=form_row, column=0, sticky="w", pady=(8, 3))
        scenario = ttk.Combobox(section, textvariable=self.scenario, values=("Fog on", "Fog off"),
                                state="readonly", width=20)
        scenario.grid(row=form_row, column=1, sticky="ew", pady=(8, 3))
        ttk.Label(section, text="Notes").grid(row=form_row + 1, column=0, sticky="w", pady=3)
        notes = ttk.Entry(section, textvariable=self.notes)
        notes.grid(row=form_row + 1, column=1, sticky="ew", pady=3)
        self._capture_setup_widgets.extend((scenario, notes))
        buttons = ttk.Frame(section)
        buttons.grid(row=form_row + 2, column=0, columnspan=2, sticky="w", pady=(8, 3))
        self.start_capture_button = self._button(buttons, "Start recording", "capture.start", self.start_capture)
        self.start_capture_button.pack(side="left")
        self._button(buttons, "New capture", "capture.new").pack(side="left", padx=4)
        buttons = ttk.Frame(section)
        buttons.grid(row=form_row + 3, column=0, columnspan=2, sticky="w", pady=3)
        self.placed_button = self._button(buttons, "Placed", "capture.confirm")
        self.placed_button.pack(side="left")
        self.stop_capture_button = self._button(buttons, "Stop + save", "capture.stop")
        self.stop_capture_button.pack(side="left", padx=4)
        self.resume_button = self._button(buttons, "Resume tracking", "capture.resume")
        self.resume_button.pack(side="left")

    def _build_assignment(self, parent):
        section = self._section(parent, "Mini → Foundry token")
        section.columnconfigure(0, weight=0)
        section.columnconfigure(1, weight=1)
        self.assignment_mini = tk.StringVar(self)
        self.assignment_token = tk.StringVar(self)
        self.assignment_message = tk.StringVar(self, "Put the token on the active Foundry scene first.")
        for row, (label, variable) in enumerate((("Mini", self.assignment_mini), ("Foundry token", self.assignment_token))):
            ttk.Label(section, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            combo = ttk.Combobox(section, textvariable=variable, state="readonly")
            combo.grid(row=row, column=1, sticky="ew", pady=3)
            if row == 0:
                self.assignment_mini_combo = combo
            else:
                self.assignment_token_combo = combo
        actions = ttk.Frame(section)
        actions.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 4))
        self._button(actions, "Apply assignment", "assignment.save", self.save_assignment).pack(side="left")
        self._button(actions, "Cancel assignment", "assignment.cancel").pack(side="left", padx=(6, 0))
        _wrapped_label(section, textvariable=self.assignment_message).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=3)

    def _build_advanced(self, parent):
        section = self._section(parent, "Foundry interface")
        self._setting(section, "foundryUIHidden", 0)
        self._button(section, "Apply interface", "ui.setHidden", self.apply_hidden).grid(
            row=1, column=0, sticky="w", pady=4)
        section = self._section(parent, "Module settings")
        names = ("physicalGridAutoApply", "debug")
        for row, name in enumerate(names):
            self._setting(section, name, row)
        self._settings_button(section, names, 2)

    def _edited(self, name):
        if self._syncing:
            return
        self._notice = ""
        try:
            value = parse_settings({name: self.variables[name].get()}, (name,))[name]
        except (ValueError, tk.TclError):
            self._dirty.add(name)
            return
        if value == self._remote_settings.get(name):
            self._dirty.discard(name)
        else:
            self._dirty.add(name)

    def set_mini_rows(self, rows):
        self._library_rows = [dict(row) for row in rows]

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self._display.set_enabled(self._native_allowed())
        self.grid_controls.set_enabled(enabled and not capture_active(self._last_status.get("capture") or {}))
        self.display_profiles.set_enabled(enabled)
        if hasattr(self, "foundry_startup"):
            self.foundry_startup.set_enabled(enabled)
        if hasattr(self, "printed_markers"):
            self.printed_markers.set_enabled(enabled)
        if hasattr(self, "token_creation"):
            self.token_creation.set_enabled(enabled)
        self._update_enabled()

    def _allowed(self, action):
        status = self._last_status
        if not (self._enabled and status.get("connected") and status.get("supported")
                and not status.get("pending") and not status.get("blocked")):
            return False
        if action not in status.get("capabilities", []):
            return False
        capture = status.get("capture") or {}
        active = capture_active(capture)
        if action == "capture.confirm":
            return bool(capture.get("recording") and capture.get("phase") == "placement" and not capture.get("waiting"))
        if action == "capture.stop":
            return bool(active and capture.get("phase") not in {"Saving", "Restoring"})
        if action == "capture.start" and (not self._live_session or self._display_busy):
            return False
        if action in {"capture.new", "capture.resume"}:
            return not active
        if active:
            return False
        if action == "movement.undo":
            return bool((status.get("movement") or {}).get("canUndo"))
        if action == "movement.reset":
            return bool(status.get("movement"))
        return True

    def _update_enabled(self):
        for widget, action in self._actions:
            widget.configure(state="normal" if self._allowed(action) else "disabled")
        settings_enabled = self._allowed("settings.set")
        for widget in self._setting_widgets:
            widget.configure(state="normal" if settings_enabled else "disabled")
        setup_enabled = self._allowed("capture.start")
        for widget in self._capture_setup_widgets:
            widget.configure(state=("readonly" if isinstance(widget, ttk.Combobox) else "normal")
                             if setup_enabled else "disabled")
        for row in self._capture_rows:
            if row["miniId"] is None:
                row["check"].configure(state="disabled")
                row["combo"].configure(state="disabled")
        for widget, action in ((self.movement_combo, "movement.select"),
                               (self.assignment_mini_combo, "assignment.save"),
                               (self.assignment_token_combo, "assignment.save")):
            widget.configure(state="readonly" if self._allowed(action) else "disabled")
        self.connect_button.configure(state="normal" if self._service is not None and
                                      (self._enabled or self._service.running) else "disabled")

    def _sync_settings(self, settings, pending):
        self._syncing = True
        try:
            for name, variable in self.variables.items():
                if name not in settings:
                    continue
                try:
                    value = parse_settings(settings, (name,))[name]
                except ValueError:
                    continue
                self._remote_settings[name] = value
                if not pending and value == self._submitted.get(name):
                    try:
                        if parse_settings({name: variable.get()}, (name,))[name] == value:
                            self._dirty.discard(name)
                    except (ValueError, tk.TclError):
                        pass
                if name not in self._dirty:
                    variable.set(value)
        finally:
            self._syncing = False

    def _sync_choices(self, status):
        selections = {"movement": self.movement_choice, "assignment": self.assignment_token,
                      **{f"capture:{index}": row["token"] for index, row in enumerate(self._capture_rows)}}
        for key, variable in selections.items():
            label = variable.get()
            if label in self._token_labels:
                self._choice_token_ids[key] = self._token_labels[label]
            elif not label:
                self._choice_token_ids.pop(key, None)
        scene_id = status.get("sceneId") or (status.get("scene") or {}).get("id")
        scene_changed = bool(scene_id and self._choice_scene_id and scene_id != self._choice_scene_id)
        if scene_id:
            self._choice_scene_id = scene_id
        if scene_changed:
            self._capture_initialized.clear()
            for index, row in enumerate(self._capture_rows):
                row["token"].set("")
                row["enabled"].set(False)
                self._choice_token_ids.pop(f"capture:{index}", None)
        self._tokens = {token["tokenId"]: token for token in status.get("tokens", [])}
        self._token_labels = {f"{token.get('tokenName', token_id)} ({token_id})": token_id
                              for token_id, token in self._tokens.items()}
        choices = tuple(self._token_labels)
        labels_by_id = {token_id: label for label, token_id in self._token_labels.items()}
        if scene_id and status.get("connected") and status.get("supported"):
            for key, variable in selections.items():
                selected_id = self._choice_token_ids.get(key)
                if selected_id:
                    variable.set(labels_by_id.get(selected_id, ""))
                    if key.startswith("capture:") and selected_id not in self._tokens:
                        index = int(key.split(":")[1])
                        self._capture_initialized.discard(self._capture_rows[index]["miniId"])
        for combo in (self.movement_combo, self.assignment_token_combo,
                      *(row["combo"] for row in self._capture_rows)):
            combo.configure(values=choices)
        minis = {mini["miniId"]: dict(mini) for mini in status.get("minis", [])}
        for row in self._library_rows:
            mini_id = row.get("id")
            if mini_id:
                minis.setdefault(mini_id, {"miniId": mini_id, "tokenName": row.get("name", mini_id)})
        pending = status.get("assignment") or {}
        if pending.get("miniId"):
            minis.setdefault(pending["miniId"], {"miniId": pending["miniId"], "tokenName": pending["miniId"]})
        self._mini_labels = {f"{mini.get('tokenName', mini_id)} ({mini_id})": mini_id
                             for mini_id, mini in minis.items()}
        self.assignment_mini_combo.configure(values=tuple(self._mini_labels))
        request = (pending.get("miniId"), pending.get("sceneId")) if pending else None
        if request and request != self._assignment_request:
            self.assignment_mini.set(next((label for label, mini_id in self._mini_labels.items()
                                           if mini_id == request[0]), request[0]))
        self._assignment_request = request
        for row, mini in zip(self._capture_rows, status.get("minis", [])):
            mini_id = mini["miniId"]
            row["miniId"] = mini_id
            row["name"].set(f"{mini.get('route', '')} · {mini.get('tokenName', mini_id)}")
            if mini_id not in self._capture_initialized:
                token_id = mini.get("tokenId")
                if token_id not in self._tokens:
                    token_id = next((key for key, token in self._tokens.items()
                                     if str(token.get("tokenName", "")).lower() == str(mini.get("tokenName", "")).lower()), None)
                if token_id:
                    row["token"].set(next(label for label, key in self._token_labels.items() if key == token_id))
                    row["enabled"].set(True)
                    self._capture_initialized.add(mini_id)
        for row in self._capture_rows[len(status.get("minis", [])):]:
            row["miniId"] = None
        queued = f" · {pending['pendingCount']} waiting" if pending.get("pendingCount", 0) > 1 else ""
        self.assignment_message.set(f"Assignment requested for {pending['miniId']}{queued}" if pending else
                                    "Put the token on the active Foundry scene first.")

    def refresh(self, status=None):
        now = time.monotonic()
        if status is None:
            if now - self._last_refresh < 0.25:
                return
            status = self._read_status()
        self._last_refresh = now
        self._last_status = dict(status)
        self._sync_settings(status.get("settings") or {}, status.get("pending"))
        self._sync_choices(status)
        self.connection.set("Foundry: connected" if status.get("connected") else "Foundry: disconnected")
        scene = status.get("scene") or {}
        self.scene.set(f"Scene: {scene.get('name') or scene.get('id')}" if scene else "Scene: unavailable")
        if self._service is not None:
            self.relay.set(f"Local relay: {self._service.state.lower()}")
            self.connect_button.configure(text="Stop relay" if self._service.running else "Start relay")
        else:
            self.relay.set("Local relay: managed by the live session")
        movement = status.get("movement") or {}
        self.movement_message.set((f"{movement.get('label') or movement.get('miniId') or 'Selected token'}\n"
                                   f"{movement.get('usedFeet', 0):g} ft used · {movement.get('remainingFeet', 0):g} ft left"
                                   + (f" · {movement['overageFeet']:g} ft over" if movement.get("overageFeet", 0) else ""))
                                  if movement else "No mini selected")
        capture = status.get("capture") or {}
        phase = capture.get("phase") or capture.get("state") or "Ready"
        countdown = f" · {capture['seconds']}s" if capture.get("seconds") else ""
        target = capture.get("target") or {}
        prompt = (f"\nPlace {target.get('tokenName') or target.get('miniId')} at {target.get('label', '')} ({target.get('cell', '')})"
                  if target and phase == "placement" else "")
        capture_text = (f"{phase}{countdown} · {capture.get('index', 0)}/{capture.get('total', 0)} placements"
                        f"{prompt}\n{capture.get('message') or ''}").rstrip()
        if not self._live_session and not capture_active(capture):
            capture_text += "\nStart a live session before recording."
        self.capture_message.set(capture_text)
        unavailable = not status.get("connected") or not status.get("supported") or status.get("pending") or status.get("blocked")
        self.message.set(("" if unavailable else self._notice) or str(status.get("message") or "Connect Foundry to use its controls."))
        self.grid_controls.set_enabled(self._enabled and not capture_active(capture))
        self.grid_controls.refresh()
        self._refresh_display()
        self.display_profiles.refresh(status)
        if hasattr(self, "foundry_startup"):
            self.foundry_startup.refresh(status)
        if hasattr(self, "printed_markers"):
            self.printed_markers.refresh()
        if hasattr(self, "token_creation"):
            self.token_creation.refresh(status)
        self._update_enabled()

    def command(self, action, values=None):
        try:
            if self._require_access is not None:
                self._require_access()
            self.refresh(self._read_status())
            if not self._allowed(action):
                return False
            accepted = self._request(action, values or {})
            self._notice = ""
            self.refresh(self._read_status())
            return bool(accepted)
        except Exception as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def apply_settings(self, names):
        try:
            settings = parse_settings({name: self.variables[name].get() for name in names}, names)
            self._submitted.update(settings)
            self._dirty.update(settings)
            if not self.command("settings.set", {"settings": settings}):
                return False
            return True
        except (ValueError, tk.TclError) as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def apply_hidden(self):
        hidden = self.variables["foundryUIHidden"].get()
        self._submitted["foundryUIHidden"] = hidden
        self._dirty.add("foundryUIHidden")
        if not self.command("ui.setHidden", {"hidden": hidden}):
            return False
        return True

    def start_capture(self):
        try:
            participants = capture_participants([(row["miniId"], row["enabled"].get(),
                self._token_labels.get(row["token"].get())) for row in self._capture_rows if row["miniId"]], self._tokens)
            notes = self.notes.get()
            if len(notes) > 2000:
                raise ValueError("Keep capture notes to 2,000 characters or fewer.")
            return self.command("capture.start", {"participants": participants,
                "scenario": "fog-on" if self.scenario.get() == "Fog on" else "fog-off", "notes": notes})
        except ValueError as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def select_movement(self):
        token_id = self._token_labels.get(self.movement_choice.get())
        if token_id is None:
            self._notice = "Choose a Foundry token to select."
            self.message.set(self._notice)
            return False
        return self.command("movement.select", {"tokenId": token_id})

    def save_assignment(self):
        mini_id = self._mini_labels.get(self.assignment_mini.get())
        token_id = self._token_labels.get(self.assignment_token.get())
        if mini_id is None or token_id is None:
            self._notice = "Choose a mini and a Foundry token."
            self.message.set(self._notice)
            return False
        return self.command("assignment.save", {"miniId": mini_id, "tokenId": token_id})

    def _capture_edited(self, row):
        self._capture_initialized.add(row["miniId"])
        self._notice = ""

    def close(self):
        if hasattr(self, "foundry_startup") and hasattr(self.foundry_startup, "close"):
            self.foundry_startup.close()
        self._display.close()

    def toggle_connection(self):
        try:
            if self._service is None:
                return False
            if self._service.running:
                self._service.stop()
            else:
                if self._require_access is not None:
                    self._require_access()
                self._service.start()
            self.refresh(self._read_status())
            return True
        except Exception as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def _refresh_display(self):
        self._display.set_enabled(self._native_allowed())
        status = self._display.status()
        self._display_busy = bool(status.get("busy"))
        self._display_names = {f"{display.get('name', display['id'])}{' (main)' if display.get('primary') else ''}": display["id"]
                               for display in status.get("displays", [])}
        self.display_combo.configure(values=tuple(self._display_names),
                                     state="readonly" if status.get("supported") and self._native_allowed() else "disabled")
        selected = status.get("selectedDisplay")
        self.display_choice.set(next((name for name, display_id in self._display_names.items() if display_id == selected), ""))
        self.display_auto.set(bool(status.get("auto")))
        self.display_message.set("Finish guided capture before moving the Foundry window." if self._enabled and not self._native_allowed()
                                 else str(status.get("message") or ""))
        enabled = bool(self._native_allowed() and status.get("supported") and not status.get("busy"))
        self.display_place.configure(state="normal" if enabled and status.get("permission") and selected is not None else "disabled")
        self.display_auto_check.configure(state="normal" if enabled else "disabled")

    def configure_display(self):
        try:
            selected = self._display_names.get(self.display_choice.get())
            automatic = bool(self.display_auto.get())
            if self._require_access is not None:
                self._require_access()
            self.refresh(self._read_status())
            if not self._native_allowed():
                return
            self._display.configure(display_id=selected, auto=automatic)
            self._refresh_display()
        except Exception as exc:
            self.display_message.set(str(exc))

    def place_display(self):
        try:
            if self._require_access is not None:
                self._require_access()
            self.refresh(self._read_status())
            if not self._native_allowed():
                return
            self._display.place()
            self._refresh_display()
            self._update_enabled()
        except Exception as exc:
            self.display_message.set(str(exc))

    def _native_allowed(self):
        capture = self._last_status.get("capture") or {}
        return bool(self._enabled and not capture_active(capture))
