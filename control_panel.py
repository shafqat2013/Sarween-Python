# control_panel.py
# Simple Tkinter control panel for Sarween runtime toggles/actions.
#
# Design goals:
# - Works in both Self-hosted and Foundry mode
# - Non-blocking: NO tkinter mainloop() — you call panel.pump() from your CV loop
# - Thread-safe-ish: button presses set flags; CV loop polls and clears one-shot actions
# - Provides toggles (show windows) + actions (capture/exit/recapture bg)
#
# Usage:
#   panel = ControlPanel(mode="self_hosted")
#   panel.hide()   # optional
#   ...
#   panel.show()
#   ...
#   panel.set_status(foundry_connected=True, locked=True, marker_count=4, missing_ids=[])
#   ...
#   if not panel.pump(): break  # user closed panel
#   toggles = panel.get_toggles()
#   actions = panel.pop_actions()

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List


class ControlPanel:
    def __init__(self, mode: str = "self_hosted"):
        self.mode = (mode or "self_hosted").strip().lower()
        if self.mode not in ("self_hosted", "foundry"):
            self.mode = "self_hosted"

        self._lock = threading.Lock()

        # Toggle state (persistent)
        self._toggles: Dict[str, bool] = {
            "show_live_camera": False,
            "show_homography": False,
            "show_masks": False,
            "show_identify": False,
            "show_blended": False,   # self-hosted only; safe to keep in both
        }

        # One-shot actions (cleared after pop_actions)
        self._actions: Dict[str, bool] = {
            "capture": False,
            "exit": False,
            "recapture_bg": False,
        }

        # Tk root
        self.root = tk.Tk()
        self.root.title("Sarween Control Panel")
        self.root.geometry("520x340")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Theme / style
        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass

        self.root.configure(bg="#2e2e2e")
        style.configure("TFrame", background="#2e2e2e")
        style.configure("TLabel", foreground="white", background="#2e2e2e")
        style.configure("TButton", foreground="white")
        style.configure("TCheckbutton", foreground="white", background="#2e2e2e")

        # Status vars
        self.var_mode = tk.StringVar(value=f"Mode: {self.mode}")
        self.var_lock = tk.StringVar(value="Lock: (waiting)")
        self.var_markers = tk.StringVar(value="Markers: --")
        self.var_missing = tk.StringVar(value="Missing: --")
        self.var_foundry = tk.StringVar(value="Foundry: (n/a)")
        self.var_hint = tk.StringVar(value="")

        # Checkbutton vars
        self._var_show_live = tk.BooleanVar(value=self._toggles["show_live_camera"])
        self._var_show_h = tk.BooleanVar(value=self._toggles["show_homography"])
        self._var_show_masks = tk.BooleanVar(value=self._toggles["show_masks"])
        self._var_show_ident = tk.BooleanVar(value=self._toggles["show_identify"])
        self._var_show_blended = tk.BooleanVar(value=self._toggles["show_blended"])

        # Layout
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        title = ttk.Label(outer, text="Sarween Control Panel", font=("Helvetica", 16, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        status_box = ttk.Frame(outer)
        status_box.pack(fill="x", pady=(0, 10))

        ttk.Label(status_box, textvariable=self.var_mode).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_lock).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_markers).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_missing).pack(anchor="w")

        # Foundry status only meaningful in foundry mode, but safe to show always
        ttk.Label(status_box, textvariable=self.var_foundry).pack(anchor="w", pady=(6, 0))

        ttk.Separator(outer).pack(fill="x", pady=10)

        # Toggles
        toggles_box = ttk.Frame(outer)
        toggles_box.pack(fill="x")

        ttk.Label(toggles_box, text="Windows").pack(anchor="w")

        ttk.Checkbutton(
            toggles_box,
            text="Live Camera View (Camera + Detections)",
            variable=self._var_show_live,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w", pady=(6, 0))

        ttk.Checkbutton(
            toggles_box,
            text="Homography View",
            variable=self._var_show_h,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Checkbutton(
            toggles_box,
            text="Masks Debug (2x2)",
            variable=self._var_show_masks,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Checkbutton(
            toggles_box,
            text="Identify Debug",
            variable=self._var_show_ident,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Checkbutton(
            toggles_box,
            text="Blended ArUco Markers (self-hosted)",
            variable=self._var_show_blended,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Separator(outer).pack(fill="x", pady=10)

        # Actions
        actions_box = ttk.Frame(outer)
        actions_box.pack(fill="x")

        ttk.Label(actions_box, text="Actions").pack(anchor="w")

        btn_row = ttk.Frame(actions_box)
        btn_row.pack(fill="x", pady=(8, 0))

        ttk.Button(btn_row, text="Capture", command=self._act_capture).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Recapture BG", command=self._act_recapture_bg).pack(side="left", padx=(0, 8))
        ttk.Button(btn_row, text="Exit", command=self._act_exit).pack(side="right")

        ttk.Label(outer, textvariable=self.var_hint, foreground="#cccccc").pack(anchor="w", pady=(10, 0))

        # Start hidden? caller can show/hide
        # Default: show (so you can see it immediately if desired)
        # You can call panel.hide() initially in tracking until lock is acquired.

    # ─────────────────────────────────────────────────────────
    # UI event handlers
    # ─────────────────────────────────────────────────────────

    def _on_close(self):
        # Treat closing the window as Exit
        with self._lock:
            self._actions["exit"] = True
        try:
            self.root.withdraw()
        except Exception:
            pass

    def _sync_toggles_from_ui(self):
        with self._lock:
            self._toggles["show_live_camera"] = bool(self._var_show_live.get())
            self._toggles["show_homography"] = bool(self._var_show_h.get())
            self._toggles["show_masks"] = bool(self._var_show_masks.get())
            self._toggles["show_identify"] = bool(self._var_show_ident.get())
            self._toggles["show_blended"] = bool(self._var_show_blended.get())

    def _act_capture(self):
        with self._lock:
            self._actions["capture"] = True

    def _act_exit(self):
        with self._lock:
            self._actions["exit"] = True

    def _act_recapture_bg(self):
        with self._lock:
            self._actions["recapture_bg"] = True

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def show(self):
        try:
            self.root.deiconify()
            self.root.lift()
        except Exception:
            pass

    def hide(self):
        try:
            self.root.withdraw()
        except Exception:
            pass

    def set_hint(self, text: str):
        try:
            self.var_hint.set(text or "")
        except Exception:
            pass

    def set_status(
        self,
        *,
        foundry_connected: bool | None = None,
        locked: bool | None = None,
        marker_count: int | None = None,
        missing_ids: List[int] | None = None,
    ):
        """
        Update the status labels.
        Safe to call frequently from the CV loop.
        """
        if locked is None:
            lock_txt = "Lock: (waiting)"
        else:
            lock_txt = "Lock: ✅ acquired" if locked else "Lock: (waiting)"
        try:
            self.var_lock.set(lock_txt)
        except Exception:
            pass

        if marker_count is not None:
            try:
                self.var_markers.set(f"Markers: {int(marker_count)}/4")
            except Exception:
                pass

        if missing_ids is not None:
            try:
                if len(missing_ids) == 0:
                    self.var_missing.set("Missing: none")
                else:
                    self.var_missing.set(f"Missing: {missing_ids}")
            except Exception:
                pass

        if foundry_connected is None:
            ftxt = "Foundry: (n/a)" if self.mode != "foundry" else "Foundry: (unknown)"
        else:
            ftxt = "Foundry: ✅ connected" if foundry_connected else "Foundry: ❌ disconnected"
        try:
            self.var_foundry.set(ftxt)
        except Exception:
            pass

    def get_toggles(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._toggles)

    def pop_actions(self) -> Dict[str, bool]:
        """
        Return current one-shot actions and clear them.
        """
        with self._lock:
            out = dict(self._actions)
            for k in self._actions.keys():
                self._actions[k] = False
        return out

    def pump(self) -> bool:
        """
        Non-blocking UI update.
        Returns False if user requested exit (via close or Exit button).
        """
        # Update Tk events
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            # Window destroyed unexpectedly
            return False

        # If exit was requested, signal caller
        with self._lock:
            if self._actions.get("exit"):
                return False
        return True