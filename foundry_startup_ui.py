"""Foundry startup controls inside Sarween's existing VTT tab."""

import tkinter as tk
from tkinter import ttk, filedialog


class FoundryStartupControls(ttk.LabelFrame):
    def __init__(self, parent, *, require_access=None, foundry_service=None,
                 display_controller=None, controller=None):
        super().__init__(parent, text="Start Foundry", padding=10)
        if controller is None:
            from foundry_startup import get_foundry_startup_controller
            controller = get_foundry_startup_controller()
        self.controller, self._access = controller, require_access
        self._service, self._display = foundry_service, display_controller
        self._enabled = False
        self._placement_notice = ""
        self._placement_pending = False
        self._widgets = []
        self.columnconfigure(1, weight=1)
        defaults = controller.defaults()
        self.values = {key: tk.StringVar(self, defaults[key]) for key in
                       ("appPath", "dataPath", "baseUrl", "worldId", "username")}
        self.unpause = tk.BooleanVar(self, defaults["unpause"])
        self.place_window = tk.BooleanVar(self, defaults["placeWindow"])
        self.password = tk.StringVar(self)
        self.message = tk.StringVar(self, "Choose your world, then start Foundry.")
        self._worlds = {}
        for row, (key, label) in enumerate((("worldId", "World"), ("username", "Foundry user"))):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            if key == "worldId":
                self.world_choice = tk.StringVar(self)
                widget = self.world_combo = ttk.Combobox(self, textvariable=self.world_choice, state="readonly")
            else:
                widget = ttk.Entry(self, textvariable=self.values[key])
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            self._widgets.append(widget)
        ttk.Label(self, text="Password (optional)").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        password = ttk.Entry(self, textvariable=self.password, show="•")
        password.grid(row=2, column=1, sticky="ew", pady=3)
        self._widgets.append(password)
        for row, (label, variable) in enumerate((("Unpause after joining", self.unpause),
                ("Move to the selected TV + full screen", self.place_window)), start=3):
            widget = ttk.Checkbutton(self, text=label, variable=variable)
            widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=3)
            self._widgets.append(widget)
        buttons = ttk.Frame(self)
        buttons.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 5))
        self.start_button = ttk.Button(buttons, text="Start Foundry", command=self.start)
        self.start_button.pack(side="left")
        self.cancel_button = ttk.Button(buttons, text="Cancel startup", command=self.controller.cancel)
        self.cancel_button.pack(side="left", padx=6)
        self.refresh_button = ttk.Button(buttons, text="Refresh worlds", command=self.refresh_worlds)
        self.refresh_button.pack(side="left")
        label = ttk.Label(self, textvariable=self.message, wraplength=450, justify="left")
        label.grid(row=6, column=0, columnspan=2, sticky="ew", pady=4)
        label.bind("<Configure>", lambda event, label=label: label.configure(wraplength=max(100, event.width)))
        note = ttk.Label(self, text="Uses your normal Foundry login. Passwords are never saved. "
                         "For automatic login, close Foundry once and launch it here.", wraplength=450, justify="left")
        note.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 8))
        note.bind("<Configure>", lambda event, note=note: note.configure(wraplength=max(100, event.width)))
        advanced = ttk.LabelFrame(self, text="Application and local connection", padding=6)
        advanced.grid(row=8, column=0, columnspan=2, sticky="ew")
        advanced.columnconfigure(1, weight=1)
        for row, (key, label) in enumerate((("appPath", "Foundry app"), ("dataPath", "Data folder"),
                                          ("baseUrl", "Local address"))):
            ttk.Label(advanced, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
            widget = ttk.Entry(advanced, textvariable=self.values[key], width=24)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            self._widgets.append(widget)
        self.refresh_worlds()
        self.refresh()
        self.bind("<Destroy>", lambda event: self.close() if event.widget is self else None, add="+")

    def refresh_worlds(self):
        try:
            worlds = self.controller.worlds(self.values["dataPath"].get())
            self._worlds = {row["name"] + " (" + row["id"] + ")": row["id"] for row in worlds}
            self.world_combo.configure(values=tuple(self._worlds))
            identity = self.values["worldId"].get()
            self.world_choice.set(next((label for label, value in self._worlds.items() if value == identity), ""))
            if not worlds:
                self.message.set("No worlds found. Check the Foundry data folder, then refresh.")
        except Exception as exc:
            self.message.set(str(exc))

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self.controller.set_enabled(enabled)
        self.refresh()

    def start(self):
        try:
            if self._access is not None:
                self._access()
            if not self._enabled or self.controller.status()["busy"]:
                return False
            values = {key: variable.get() for key, variable in self.values.items()}
            values["worldId"] = self._worlds.get(self.world_choice.get(), "")
            values.update(unpause=self.unpause.get(), placeWindow=self.place_window.get())
            if self._service is None:
                raise ValueError("The local Foundry relay is unavailable in this panel.")
            self._service.start()
            accepted = self.controller.start(values, password=self.password.get())
            if accepted:
                self.password.set("")
                self._placement_notice = ""
                self._placement_pending = False
            self.refresh()
            return accepted
        except Exception as exc:
            self.message.set(str(exc))
            return False

    def refresh(self, _vtt_status=None):
        status = self.controller.status()
        if self._placement_pending and self._display is not None:
            placement = self._display.status()
            if not placement.get("busy"):
                self._placement_pending = False
                self._placement_notice = "Foundry is ready. " + placement.get("message", "Check its window on the TV.")
        busy = bool(status["busy"])
        state = "normal" if self._enabled and not busy else "disabled"
        for widget in self._widgets:
            widget.configure(state=state)
        self.world_combo.configure(state="readonly" if self._enabled and not busy else "disabled")
        self.start_button.configure(state=state)
        self.refresh_button.configure(state=state)
        self.cancel_button.configure(state="normal" if self._enabled and busy else "disabled")
        if status.get("phase") != "idle":
            self.message.set(self._placement_notice or status["message"])
        if self.controller.take_placement_request():
            if self._display is None or not self._display.place():
                self._placement_notice = "Foundry is ready. Select a TV and allow optional window placement under Display."
            else:
                self._placement_pending = True
                self._placement_notice = "Foundry is ready. Moving its window to the selected TV…"
            self.message.set(self._placement_notice)

    def close(self):
        self.controller.cancel()
