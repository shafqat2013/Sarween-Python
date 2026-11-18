"""Scene and placed-token vision controls for the laptop, before mini binding."""
import math
import tkinter as tk
from tkinter import ttk


SCENE_FIELDS = ("tokenVision", "fogExploration", "globalIllumination")


def sight_range(value):
    if isinstance(value, bool):
        raise ValueError("Enter a sight range of zero or more scene units.")
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError("Enter a sight range of zero or more scene units.") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError("Enter a sight range of zero or more scene units.")
    return value


def _label(parent, **options):
    label = ttk.Label(parent, justify="left", anchor="w", wraplength=440, **options)
    label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
    return label


class VisionFogControls(ttk.Frame):
    def __init__(self, parent, *, read_status, request, require_access=None, can_change=None):
        super().__init__(parent)
        self._read_status, self._request = read_status, request
        self._require_access, self._can_change = require_access, can_change
        self._enabled = False
        self._status = {}
        self._context = None
        self._token_id = None
        self._tokens, self._labels = {}, {}
        self._dirty, self._submitted, self._remote = set(), {}, {}
        self._syncing = False
        self._pending = None
        self._notice = ""
        self.variables = {name: tk.BooleanVar(self, False) for name in SCENE_FIELDS}
        self.variables.update(enabled=tk.BooleanVar(self, False), range=tk.StringVar(self, "0"))
        self.token = tk.StringVar(self)
        self.scene_name = tk.StringVar(self, "Scene: unavailable")
        self.tv_view = tk.StringVar(self, "TV view: no token selected")
        self.range_label = tk.StringVar(self, "Sight range (scene units)")
        self.message = tk.StringVar(self)
        _label(self, textvariable=self.scene_name).pack(fill="x", pady=(0, 6))
        scene = ttk.LabelFrame(self, text="Scene vision and fog", padding=10)
        scene.pack(fill="x", pady=(0, 8))
        self._scene_widgets = []
        for name, text in (("tokenVision", "Token vision"), ("fogExploration", "Fog exploration (remember explored areas)"),
                           ("globalIllumination", "Global illumination (daylight)")):
            widget = ttk.Checkbutton(scene, text=text, variable=self.variables[name])
            widget.pack(anchor="w", pady=3)
            self._scene_widgets.append(widget)
        _label(scene, text="For the first fog test: turn token vision and explored fog on, and daylight off.").pack(
            fill="x", pady=(5, 3))
        self.apply_scene_button = ttk.Button(scene, text="Apply scene settings", command=self.apply_scene)
        self.apply_scene_button.pack(anchor="w", pady=(5, 0))
        token = ttk.LabelFrame(self, text="Token vision", padding=10)
        token.pack(fill="x", pady=(0, 8))
        token.columnconfigure(1, weight=1)
        _label(token, text="Foundry token").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.token_combo = ttk.Combobox(token, textvariable=self.token, state="disabled", width=24)
        self.token_combo.grid(row=0, column=1, sticky="ew", pady=3)
        self.token_combo.bind("<<ComboboxSelected>>", lambda _event: self._choose_token())
        self.vision_check = ttk.Checkbutton(token, text="Vision enabled", variable=self.variables["enabled"])
        self.vision_check.grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        _label(token, textvariable=self.range_label).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.range_entry = ttk.Entry(token, textvariable=self.variables["range"], width=10)
        self.range_entry.grid(row=2, column=1, sticky="ew", pady=3)
        _label(token, text="Use a finite range, such as 30, for the test. Zero relies on other lighting."
               ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(3, 5))
        self.apply_token_button = ttk.Button(token, text="Apply token settings", command=self.apply_token)
        self.apply_token_button.grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        self.show_view_button = ttk.Button(token, text="Show this token's view on TV", command=self.show_view)
        self.show_view_button.grid(row=5, column=0, columnspan=2, sticky="w", pady=3)
        _label(token, textvariable=self.tv_view).grid(row=6, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        _label(self, text="Configure vision and choose the TV view before starting a session and binding minis.").pack(
            fill="x", pady=(0, 6))
        _label(self, textvariable=self.message).pack(fill="x")
        for name, variable in self.variables.items():
            variable.trace_add("write", lambda *_args, name=name: self._edited(name))
        self._update_enabled()

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._update_enabled()

    def _reason(self, action):
        status = self._status
        if not self._enabled:
            return "Sign in to use Foundry controls."
        if self._can_change is not None and not self._can_change():
            return "Stop the session or mini observer before changing vision, then bind the minis again."
        if status.get("outputPaused"):
            return "Stop the mini observer or finish capture before changing vision."
        if not status.get("connected"):
            return "Connect Foundry to configure vision and fog."
        if not status.get("supported") or action not in status.get("capabilities", []):
            return "Reload Foundry with the updated Sarween module to use Vision & Fog."
        if not status.get("sceneId"):
            return "Open a Foundry scene first."
        capture = status.get("capture") or {}
        if (capture.get("active") or capture.get("recording")
                or capture.get("phase") in {"Starting", "Saving", "Restoring"}):
            return "Finish recording before changing vision."
        if status.get("pending") or status.get("blocked") or self._pending:
            return "Waiting for Foundry to confirm the current action."
        return ""

    def _value(self, name):
        value = self.variables[name].get()
        if name == "range":
            return sight_range(value)
        if not isinstance(value, bool):
            raise ValueError("Choose on or off for vision and fog settings.")
        return value

    def _edited(self, name):
        if self._syncing:
            return
        self._notice = ""
        try:
            same = self._value(name) == self._remote.get(name)
        except (ValueError, tk.TclError):
            same = False
        self._dirty.discard(name) if same else self._dirty.add(name)
        self._update_enabled()

    def _sync_fields(self, values):
        self._syncing = True
        try:
            for name, value in values.items():
                if name == "range":
                    try:
                        value = sight_range(value)
                    except ValueError:
                        continue
                elif not isinstance(value, bool):
                    continue
                self._remote[name] = value
                if name in self._submitted and self._submitted[name] == value:
                    try:
                        if self._value(name) == value:
                            self._dirty.discard(name)
                            self._submitted.pop(name, None)
                    except (ValueError, tk.TclError):
                        pass
                if name not in self._dirty:
                    self.variables[name].set(f"{value:g}" if name == "range" else value)
        finally:
            self._syncing = False

    def _choose_token(self, *, clear_notice=True):
        token_id = self._labels.get(self.token.get())
        if token_id != self._token_id:
            self._token_id = token_id
            for name in ("enabled", "range"):
                self._dirty.discard(name)
                self._submitted.pop(name, None)
                self._remote.pop(name, None)
            if clear_notice:
                self._notice = ""
        row = self._tokens.get(token_id, {})
        self._sync_fields(row.get("vision") or {})
        self._update_enabled()

    def _update_enabled(self):
        vision = self._status.get("vision") or {}
        scene_known = all(isinstance(vision.get(name), bool) for name in SCENE_FIELDS)
        token_vision = (self._tokens.get(self._token_id) or {}).get("vision") or {}
        token_known = isinstance(token_vision.get("enabled"), bool)
        try:
            sight_range(token_vision.get("range"))
        except ValueError:
            token_known = False
        scene_ready = not self._reason("vision.scene.set") and scene_known
        token_ready = (not self._reason("vision.token.set") and self._token_id in self._tokens
                       and token_known)
        choose_ready = not self._reason("vision.token.select") and bool(self._tokens)
        for widget in self._scene_widgets:
            widget.configure(state="normal" if scene_ready else "disabled")
        self.apply_scene_button.configure(state="normal" if scene_ready else "disabled")
        self.token_combo.configure(state="readonly" if choose_ready else "disabled")
        for widget in (self.vision_check, self.range_entry):
            widget.configure(state="normal" if token_ready else "disabled")
        self.apply_token_button.configure(state="normal" if token_ready else "disabled")
        self.show_view_button.configure(state="normal" if choose_ready and self._token_id in self._tokens else "disabled")
        self.message.set(self._reason("vision.scene.set") or self._notice
                         or ("Waiting for Foundry's current vision settings." if not scene_known else "")
                         or "Choose settings, apply them, then show a token's view before binding minis.")

    def refresh(self, status=None):
        status = self._read_status() if status is None else status
        self._status = dict(status)
        context = ((status.get("world") or {}).get("id"), status.get("sceneId"))
        if status.get("connected") and context[1]:
            if self._context is not None and context != self._context:
                self._dirty.clear()
                self._submitted.clear()
                self._remote.clear()
                self._pending = self._token_id = None
                self._notice = "Scene changed. Check its vision settings before applying."
            self._context = context
        result = status.get("lastResult") or {}
        if self._pending:
            if result.get("requestId") == self._pending:
                self._pending = None
                self._notice = str(result.get("message") or (
                    "Foundry applied the settings." if result.get("success") is True else "Foundry could not apply the settings."))
            elif not status.get("connected") or not status.get("pending"):
                self._pending = None
                self._notice = "Foundry did not confirm the action. Check the current settings before trying again."
        scene = status.get("scene") or {}
        self.scene_name.set(f"Scene: {scene.get('name') or status.get('sceneId') or 'unavailable'}")
        vision = status.get("vision") or {}
        self._sync_fields({name: vision.get(name) for name in SCENE_FIELDS})
        self.range_label.set(f"Sight range ({vision.get('distanceUnits') or 'scene units'})")
        self._tokens = {row["tokenId"]: row for row in status.get("tokens", [])
                        if isinstance(row, dict) and row.get("tokenId")}
        self._labels = {f"{row.get('tokenName') or token_id} ({token_id})": token_id
                        for token_id, row in self._tokens.items()}
        self.token_combo.configure(values=tuple(self._labels))
        selected_id = self._token_id if self._token_id in self._tokens else vision.get("selectedTokenId")
        if selected_id not in self._tokens:
            selected_id = next(iter(self._tokens), None)
        self.token.set(next((label for label, token_id in self._labels.items() if token_id == selected_id), ""))
        self._choose_token(clear_notice=False)
        shown = self._tokens.get(vision.get("selectedTokenId"))
        self.tv_view.set(f"TV view: {shown.get('tokenName') or shown['tokenId']}" if shown else "TV view: no single token selected")
        self._update_enabled()

    def _send(self, action, values, *, token_id=None):
        try:
            context = self._context
            if self._require_access is not None:
                self._require_access()
            self.refresh(self._read_status())
            if context != self._context or (token_id is not None and token_id != self._token_id):
                self._notice = "Scene or token changed. Check the current choices before applying."
                self._update_enabled()
                return False
            reason = self._reason(action)
            if reason:
                self.message.set(reason)
                return False
            if action == "vision.scene.set" and not all(
                    isinstance((self._status.get("vision") or {}).get(name), bool) for name in SCENE_FIELDS):
                raise ValueError("Waiting for Foundry's current vision settings.")
            if action == "vision.token.set":
                vision = (self._tokens.get(token_id) or {}).get("vision") or {}
                if not isinstance(vision.get("enabled"), bool):
                    raise ValueError("Waiting for this token's current vision settings.")
                sight_range(vision.get("range"))
            accepted = self._request(action, values)
            if not accepted:
                self._notice = "Foundry could not accept the action. Check its connection and try again."
                self._update_enabled()
                return False
            if action != "vision.token.select":
                self._submitted.update({name: value for name, value in values.items() if name != "tokenId"})
                self._dirty.update(self._submitted)
            self._pending = accepted
            self._notice = ""
            self._update_enabled()
            return True
        except Exception as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def apply_scene(self):
        try:
            return self._send("vision.scene.set", {name: self._value(name) for name in SCENE_FIELDS})
        except (ValueError, tk.TclError) as exc:
            self.message.set(str(exc))
            return False

    def apply_token(self):
        try:
            if self._token_id not in self._tokens:
                raise ValueError("Choose a Foundry token first.")
            return self._send("vision.token.set", {"tokenId": self._token_id,
                "enabled": self._value("enabled"), "range": self._value("range")}, token_id=self._token_id)
        except (ValueError, tk.TclError) as exc:
            self.message.set(str(exc))
            return False

    def show_view(self):
        if self._token_id not in self._tokens:
            self.message.set("Choose a Foundry token first.")
            return False
        return self._send("vision.token.select", {"tokenId": self._token_id}, token_id=self._token_id)
