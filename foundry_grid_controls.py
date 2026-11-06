"""Shared desktop controls for Foundry's physical display scale."""

import math
import time
import tkinter as tk
from tkinter import ttk


GRID_FIELDS = (
    ("targetGridInches", "Square size (inches)", 0.25, 3.0, 0.05, 1.0, 2),
    ("displayDiagonalInches", "TV diagonal (inches)", 10.0, 150.0, 0.5, 55.0, 1),
    ("physicalGridCalibration", "Fine adjustment", 0.5, 1.5, 0.005, 1.0, 3),
)


def parse_grid_settings(values):
    settings = {}
    for name, label, minimum, maximum, _step, _default, _places in GRID_FIELDS:
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Enter a number for {label.lower()}.") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}.")
        settings[name] = value
    return settings


def _read_grid_status():
    import foundryoutput
    return foundryoutput.get_physical_grid_status()


def _request_grid(settings, *, apply=True):
    import foundryoutput
    return foundryoutput.request_physical_grid(settings, apply=apply)


class FoundryGridControls(ttk.LabelFrame):
    def __init__(self, parent, *, require_access=None, read_status=None, request=None):
        super().__init__(parent, text="TV grid size", padding=10)
        self._require_access = require_access
        self._read_status = read_status or _read_grid_status
        self._request = request or _request_grid
        self._access_enabled = True
        self._editing_enabled = None
        self._last_refresh = float("-inf")
        self._last_status = {}
        self._remote_settings = {}
        self._dirty = set()
        self._submitted = None
        self._syncing = False
        self._notice = ""
        self.variables = {}
        self._fields = []
        self.columnconfigure(0, weight=1)
        for row, (name, label, minimum, maximum, step, default, places) in enumerate(GRID_FIELDS):
            ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
            variable = tk.StringVar(self, value=f"{default:.{places}f}")
            variable.trace_add("write", lambda *_args, name=name: self._field_edited(name))
            self.variables[name] = variable
            field = ttk.Spinbox(self, textvariable=variable, from_=minimum, to=maximum,
                                increment=step, width=7, format=f"%.{places}f")
            field.grid(row=row, column=1, sticky="e", pady=2)
            field.bind("<Return>", lambda _event: self.apply())
            self._fields.append(field)
        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 4))
        self.apply_button = ttk.Button(buttons, text="Apply size", command=self.apply)
        self.apply_button.pack(side="left")
        self.one_inch_button = ttk.Button(buttons, text="1-inch squares", command=self.use_one_inch)
        self.one_inch_button.pack(side="left", padx=(6, 0))
        hint = ttk.Label(self, text="Measure a square on the TV with a ruler. Increase fine adjustment "
                         "if it is too small. Resize before placing minis.", wraplength=280)
        hint.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(4, 4))
        hint.bind("<Configure>", lambda event: hint.configure(wraplength=max(180, event.width)))
        self.message = tk.StringVar(self)
        status = ttk.Label(self, textvariable=self.message, wraplength=280)
        status.grid(row=5, column=0, columnspan=2, sticky="ew")
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(180, event.width)))
        self.refresh()

    def _field_edited(self, name):
        if self._syncing:
            return
        self._notice = ""
        try:
            unchanged = float(self.variables[name].get()) == self._remote_settings.get(name)
        except ValueError:
            unchanged = False
        if unchanged:
            self._dirty.discard(name)
        else:
            self._dirty.add(name)

    def set_enabled(self, enabled):
        self._access_enabled = bool(enabled)
        self._update_enabled()

    def _update_enabled(self):
        status = self._last_status
        enabled = bool(self._access_enabled and status.get("connected") and status.get("supported")
                       and not status.get("pending") and not status.get("blocked"))
        if enabled == self._editing_enabled:
            return
        self._editing_enabled = enabled
        state = "normal" if enabled else "disabled"
        for widget in (*self._fields, self.apply_button, self.one_inch_button):
            widget.configure(state=state)

    def refresh(self, status=None):
        now = time.monotonic()
        if status is None:
            if now - self._last_refresh < 0.25:
                return
            status = self._read_status()
        self._last_refresh = now
        self._last_status = dict(status)
        remote = status.get("settings") or {}
        self._syncing = True
        try:
            for name, _label, minimum, maximum, _step, _default, places in GRID_FIELDS:
                value = remote.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if not math.isfinite(value) or not minimum <= value <= maximum:
                    continue
                self._remote_settings[name] = value
                # Acknowledged values can replace the submitted draft. A failed
                # or timed-out request leaves edits intact so Apply can retry.
                if (self._submitted and not status.get("pending")
                        and value == self._submitted.get(name)):
                    try:
                        matches_draft = float(self.variables[name].get()) == value
                    except ValueError:
                        matches_draft = False
                    if matches_draft:
                        self._dirty.discard(name)
                if name not in self._dirty:
                    self.variables[name].set(f"{value:.{places}f}")
        finally:
            self._syncing = False
        unavailable = (not status.get("connected") or not status.get("supported")
                       or status.get("pending") or status.get("blocked"))
        notice = "" if unavailable else self._notice
        self.message.set(notice or str(status.get("message") or "Connect Foundry to resize the TV grid."))
        self._update_enabled()

    def apply(self):
        try:
            if self._require_access is not None:
                self._require_access()
            status = self._read_status()
            self.refresh(status)
            if not self._editing_enabled:
                return False
            settings = parse_grid_settings({name: variable.get() for name, variable in self.variables.items()})
            if not self._request(settings, apply=True):
                self.refresh(self._read_status())
                return False
            self._notice = ""
            self._submitted = settings
            self._dirty.update(settings)
            self.refresh(self._read_status())
            return True
        except Exception as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def use_one_inch(self):
        self.variables["targetGridInches"].set("1.00")
        return self.apply()
