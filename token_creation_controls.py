"""Manual actor-to-token creation in Sarween's existing VTT controls."""
import tkinter as tk
from tkinter import ttk

from token_creation import TokenCreationController, blocked_reason, parse_cell


def _request(action, values):
    import foundryoutput
    return foundryoutput.request_vtt_command(action, values, return_request_id=True)


class TokenCreationControls(ttk.LabelFrame):
    def __init__(self, parent, *, require_access=None, read_status=None, request=None, can_create=None):
        super().__init__(parent, text="Add a Foundry token", padding=10)
        self._require_access, self._read_status = require_access, read_status
        self._can_create = can_create
        self._enabled = False
        self._status = {}
        self._actors = {}
        self._selected_actor_id = None
        self.controller = TokenCreationController(request or _request)
        self.actor = tk.StringVar(self)
        self.name = tk.StringVar(self)
        self.placement = tk.StringVar(self, "Viewport center")
        self.cell = tk.StringVar(self)
        self.message = tk.StringVar(self, self.controller.message)
        self.columnconfigure(1, weight=1)
        for row, text in enumerate(("World actor", "Token name (optional)", "Place at", "Grid cell")):
            ttk.Label(self, text=text).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
        self.actor_combo = ttk.Combobox(self, textvariable=self.actor, state="disabled", width=24)
        self.actor_combo.grid(row=0, column=1, sticky="ew", pady=3)
        self.name_entry = ttk.Entry(self, textvariable=self.name)
        self.name_entry.grid(row=1, column=1, sticky="ew", pady=3)
        self.placement_combo = ttk.Combobox(self, textvariable=self.placement,
            values=("Viewport center", "Grid cell"), state="disabled", width=24)
        self.placement_combo.grid(row=2, column=1, sticky="ew", pady=3)
        self.cell_entry = ttk.Entry(self, textvariable=self.cell, width=12)
        self.cell_entry.grid(row=3, column=1, sticky="ew", pady=3)
        self.create_button = ttk.Button(self, text="Create token", command=self.create)
        self.create_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 3))
        label = ttk.Label(self, text=("Uses the actor's prototype appearance and vision. Cell placement puts the "
            "token's top-left in that square; viewport placement snaps near the current view's center."),
            wraplength=480, justify="left")
        label.grid(row=5, column=0, columnspan=2, sticky="ew", pady=3)
        label.bind("<Configure>", lambda event, label=label: label.configure(wraplength=max(100, event.width)))
        label = ttk.Label(self, textvariable=self.message, wraplength=480, justify="left")
        label.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        label.bind("<Configure>", lambda event, label=label: label.configure(wraplength=max(100, event.width)))
        for variable in (self.actor, self.name, self.placement, self.cell):
            variable.trace_add("write", lambda *_args: self._update_enabled())
        self._update_enabled()

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._update_enabled()

    def _draft(self):
        actor_id = self._actors.get(self.actor.get())
        if actor_id is None:
            raise ValueError("Choose an available world actor.")
        values = {"actorId": actor_id, "name": self.name.get().strip(),
                  "placement": "cell" if self.placement.get() == "Grid cell" else "viewport"}
        if values["placement"] == "cell":
            values["row"], values["col"] = parse_cell(self.cell.get())
        return values

    def _available(self):
        return bool(self._enabled and not blocked_reason(self._status) and not self.controller.pending_request
                    and (self._can_create is None or self._can_create()))

    def _update_enabled(self):
        if self.actor.get() in self._actors:
            self._selected_actor_id = self._actors[self.actor.get()]
        available = self._available()
        self.actor_combo.configure(state="readonly" if available else "disabled")
        self.name_entry.configure(state="normal" if available else "disabled")
        self.placement_combo.configure(state="readonly" if available else "disabled")
        self.cell_entry.configure(state="normal" if available and self.placement.get() == "Grid cell" else "disabled")
        try:
            self._draft()
            valid = True
        except ValueError:
            valid = False
        self.create_button.configure(state="normal" if available and valid else "disabled",
            text="Creating…" if self.controller.pending_request else "Create token")

    def refresh(self, status=None):
        if status is None:
            status = self._read_status() if self._read_status else {}
        self._status = dict(status)
        previous_actor = self._actors.get(self.actor.get()) or self._selected_actor_id
        self._actors = {f"{actor.get('actorName') or actor['actorId']} ({actor['actorId']})": actor["actorId"]
                        for actor in status.get("actors", []) if isinstance(actor, dict) and actor.get("actorId")}
        self.actor_combo.configure(values=tuple(self._actors))
        self.actor.set(next((label for label, actor_id in self._actors.items() if actor_id == previous_actor), ""))
        if status.get("connected") and previous_actor not in self._actors.values():
            self._selected_actor_id = None
        self.controller.refresh(status)
        reason = blocked_reason(status)
        self.message.set(reason or self.controller.message)
        self._update_enabled()

    def create(self):
        try:
            if not self._enabled:
                raise PermissionError("Sign in before creating Foundry tokens.")
            if self._require_access is not None:
                self._require_access()
            status = self._read_status() if self._read_status else self._status
            self.refresh(status)
            if not self._available():
                return False
            accepted = self.controller.create(self._draft(), status)
            self.message.set(self.controller.message)
            self._update_enabled()
            return accepted
        except (ValueError, OSError, PermissionError) as exc:
            self.message.set(str(exc))
            return False
