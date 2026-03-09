# control_panel.py
# Simple Tkinter control panel for Sarween runtime toggles/actions.
#
# Design goals:
# - Works in both Self-hosted and Foundry mode
# - Non-blocking: NO tkinter mainloop() — you call panel.pump() from your CV loop
# - Thread-safe-ish: button presses set flags; CV loop polls and clears one-shot actions
# - Provides toggles (show windows) + actions (capture/exit/recapture bg) + engine switch
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

import json
import os
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional


CONFIG_FILE = "hardware_config.json"


def rc_to_a1(row: int, col: int) -> str:
    """
    Convert zero-based (row, col) to chess-style A1 notation.
    Columns are letters (col 0 → 'A'), rows are 1-based numbers (row 0 → '1').
    e.g. rc_to_a1(0, 0) → 'A1',  rc_to_a1(2, 3) → 'D3'
    """
    col_letter = ""
    c = col
    while True:
        col_letter = chr(ord('A') + (c % 26)) + col_letter
        c = c // 26 - 1
        if c < 0:
            break
    return f"{col_letter}{row + 1}"


def _load_config() -> Dict[str, object]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _save_config_patch(patch: Dict[str, object]) -> None:
    data = _load_config()
    try:
        data.update(patch)
        with open(CONFIG_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        # best-effort only
        pass


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
            "show_identify": False,
            "show_blended": False,   # self-hosted only; safe to keep in both

            # split mask views
            "show_motion_warp": False,
            "show_motion_cam": False,
            "show_shadowfree": False,
            "show_final_mask": False,
            "show_calib_preview": True,  # on by default — useful during setup
        }

        # One-shot actions (cleared after pop_actions)
        # switch_engine stores target engine string ("blob" or "band") when requested
        self._actions: Dict[str, object] = {
            "capture": False,
            "exit": False,
            "recapture_bg": False,
            "switch_engine": None,
            "calibrate_band": None,
        }

        # Tk root
        self.root = tk.Tk()
        self.root.title("Sarween Control Panel")
        self.root.geometry("620x820")
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

        # Plain tk.Button colour constants (ttk ignores bg on macOS)
        self._btn_bg  = "#4a4a4a"
        self._btn_fg  = "white"
        self._btn_abg = "#666666"   # active (hover/press) background
        style.configure("TCheckbutton", foreground="white", background="#2e2e2e")

        # Status vars
        self.var_mode = tk.StringVar(value=f"Mode: {self.mode}")
        self.var_lock = tk.StringVar(value="Lock: (waiting)")
        self.var_markers = tk.StringVar(value="Markers: --")
        self.var_missing = tk.StringVar(value="Missing: --")
        self.var_foundry = tk.StringVar(value="Foundry: (n/a)")
        self.var_fps = tk.StringVar(value="FPS: --")
        self.var_hint = tk.StringVar(value="")

        # Engine selection (persistent via hardware_config.json)
        cfg = _load_config()
        initial_engine = str(cfg.get("engine") or "blob").strip().lower()
        if initial_engine not in ("blob", "band"):
            initial_engine = "blob"
        self._engine_var = tk.StringVar(value=initial_engine)

        # Checkbutton vars
        self._var_show_live = tk.BooleanVar(value=self._toggles["show_live_camera"])
        self._var_show_h = tk.BooleanVar(value=self._toggles["show_homography"])
        self._var_show_ident = tk.BooleanVar(value=self._toggles["show_identify"])
        self._var_show_blended = tk.BooleanVar(value=self._toggles["show_blended"])

        self._var_show_motion_warp = tk.BooleanVar(value=self._toggles["show_motion_warp"])
        self._var_show_motion_cam = tk.BooleanVar(value=self._toggles["show_motion_cam"])
        self._var_show_shadowfree = tk.BooleanVar(value=self._toggles["show_shadowfree"])
        self._var_show_final_mask = tk.BooleanVar(value=self._toggles["show_final_mask"])
        self._var_show_calib_preview = tk.BooleanVar(value=self._toggles["show_calib_preview"])

        # Layout
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, padx=14, pady=14)

        title = ttk.Label(outer, text="Sarween Control Panel", font=("Helvetica", 16, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        # Engine row
        engine_box = ttk.Frame(outer)
        engine_box.pack(fill="x", pady=(0, 10))

        ttk.Label(engine_box, text="Engine").pack(side="left")
        engine_combo = ttk.Combobox(
            engine_box,
            values=["blob", "band"],
            textvariable=self._engine_var,
            state="readonly",
            width=10
        )
        engine_combo.pack(side="left", padx=(10, 8))

        self._btn(engine_box, text="Switch (restart)", command=self._act_switch_engine).pack(side="left")

        # Band Calibration
        band_box = ttk.Frame(outer)
        band_box.pack(fill="x", pady=(0, 10))

        ttk.Label(band_box, text="Band name").pack(side="left")
        self._band_name_var = tk.StringVar(value="")
        band_entry = ttk.Entry(band_box, textvariable=self._band_name_var, width=18)
        band_entry.pack(side="left", padx=(10, 8))

        self._btn(band_box, text="Calibrate band", command=self._act_calibrate_band).pack(side="left")
        self._btn(band_box, text="Detect & calibrate (auto)", command=self._act_calibrate_band_auto).pack(side="left", padx=(8, 0))

        status_box = ttk.Frame(outer)
        status_box.pack(fill="x", pady=(0, 10))

        ttk.Label(status_box, textvariable=self.var_mode).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_lock).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_markers).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_missing).pack(anchor="w")
        ttk.Label(status_box, textvariable=self.var_fps).pack(anchor="w")

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

        ttk.Checkbutton(
            toggles_box,
            text="Calibration Preview (band engine)",
            variable=self._var_show_calib_preview,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Separator(outer).pack(fill="x", pady=10)

        # Masks
        masks_box = ttk.Frame(outer)
        masks_box.pack(fill="x")

        ttk.Label(masks_box, text="Masks").pack(anchor="w")

        ttk.Checkbutton(
            masks_box,
            text="Motion (warp)",
            variable=self._var_show_motion_warp,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w", pady=(6, 0))

        ttk.Checkbutton(
            masks_box,
            text="Motion (camera)",
            variable=self._var_show_motion_cam,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Checkbutton(
            masks_box,
            text="Shadow-free mask",
            variable=self._var_show_shadowfree,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Checkbutton(
            masks_box,
            text="Final mask (annotated)",
            variable=self._var_show_final_mask,
            command=self._sync_toggles_from_ui
        ).pack(anchor="w")

        ttk.Separator(outer).pack(fill="x", pady=10)

        # Actions
        actions_box = ttk.Frame(outer)
        actions_box.pack(fill="x")

        ttk.Label(actions_box, text="Actions").pack(anchor="w")

        btn_row = ttk.Frame(actions_box)
        btn_row.pack(fill="x", pady=(8, 0))

        self._btn(btn_row, text="Capture", command=self._act_capture).pack(side="left", padx=(0, 8))
        self._btn(btn_row, text="Recapture BG", command=self._act_recapture_bg).pack(side="left", padx=(0, 8))
        self._btn(btn_row, text="Exit", command=self._act_exit).pack(side="right")

        # Motion threshold slider
        thresh_box = ttk.Frame(outer)
        thresh_box.pack(fill="x", pady=(10, 0))

        ttk.Label(thresh_box, text="Motion threshold (lower = more sensitive):").pack(anchor="w")

        thresh_row = ttk.Frame(thresh_box)
        thresh_row.pack(fill="x", pady=(4, 0))

        self._motion_thresh_var = tk.IntVar(value=30)  # matches cv_core WARP_MOTION_THRESH default
        self._thresh_label_var = tk.StringVar(value="30")

        thresh_slider = ttk.Scale(
            thresh_row,
            from_=5, to=80,
            orient="horizontal",
            variable=self._motion_thresh_var,
            command=self._on_thresh_change,
        )
        thresh_slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(thresh_row, textvariable=self._thresh_label_var, width=4).pack(side="left")

        ttk.Label(outer, textvariable=self.var_hint, foreground="#cccccc").pack(anchor="w", pady=(10, 0))

        ttk.Separator(outer).pack(fill="x", pady=10)

        # Mini positions table
        pos_box = ttk.Frame(outer)
        pos_box.pack(fill="both", expand=True)

        ttk.Label(pos_box, text="Mini Positions", foreground="#aaaaaa").pack(anchor="w", pady=(0, 4))

        self._pos_table = ttk.Frame(pos_box)
        self._pos_table.pack(fill="both", expand=True)
        self._pos_rows: dict = {}  # mini_id -> {"name_var": ..., "pos_var": ...}

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

    def _on_thresh_change(self, val):
        v = int(float(val))
        try:
            self._thresh_label_var.set(str(v))
        except Exception:
            pass

    def _sync_toggles_from_ui(self):
        with self._lock:
            self._toggles["show_live_camera"] = bool(self._var_show_live.get())
            self._toggles["show_homography"] = bool(self._var_show_h.get())
            self._toggles["show_identify"] = bool(self._var_show_ident.get())
            self._toggles["show_blended"] = bool(self._var_show_blended.get())
            self._toggles["show_calib_preview"] = bool(self._var_show_calib_preview.get())

            self._toggles["show_motion_warp"] = bool(self._var_show_motion_warp.get())
            self._toggles["show_motion_cam"] = bool(self._var_show_motion_cam.get())
            self._toggles["show_shadowfree"] = bool(self._var_show_shadowfree.get())
            self._toggles["show_final_mask"] = bool(self._var_show_final_mask.get())

    def _btn(self, parent, text: str, command) -> tk.Button:
        """Create a dark-styled tk.Button (ttk.Button ignores bg on macOS)."""
        return tk.Button(
            parent, text=text, command=command,
            bg=self._btn_bg, fg=self._btn_fg,
            activebackground=self._btn_abg, activeforeground="white",
            relief="flat", padx=8, pady=3,
            cursor="hand2",
        )

    def _act_capture(self):
        with self._lock:
            self._actions["capture"] = True

    def _act_exit(self):
        with self._lock:
            self._actions["exit"] = True

    def _act_recapture_bg(self):
        with self._lock:
            self._actions["recapture_bg"] = True

    def _act_switch_engine(self):
        eng = (self._engine_var.get() or "blob").strip().lower()
        if eng not in ("blob", "band"):
            eng = "blob"
        # persist
        _save_config_patch({"engine": eng})
        with self._lock:
            self._actions["switch_engine"] = eng

    def _act_calibrate_band(self):
        name = (getattr(self, "_band_name_var", None).get() if getattr(self, "_band_name_var", None) else "").strip()
        payload = {"name": name} if name else {"name": None}
        with self._lock:
            self._actions["calibrate_band"] = payload

    def _act_calibrate_band_auto(self):
        with self._lock:
            self._actions["calibrate_band"] = {"name": None, "auto": True}

    def update_positions(self, positions: dict) -> None:
        """
        Update the mini positions table.
        positions: dict of {mini_id: grid_coord_str} e.g. {"yellow": "C4", "blue": None}
        Safe to call from the CV loop (runs on the Tk thread via after()).
        """
        def _update():
            try:
                for mini_id, coord in positions.items():
                    if mini_id not in self._pos_rows:
                        # Create a new row for this mini
                        row = len(self._pos_rows)
                        name_var = tk.StringVar(value=mini_id)
                        pos_var = tk.StringVar(value="—")
                        ttk.Label(
                            self._pos_table,
                            textvariable=name_var,
                            foreground="#aaaaaa",
                            width=12,
                            anchor="w",
                        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
                        ttk.Label(
                            self._pos_table,
                            textvariable=pos_var,
                            foreground="#ffffff",
                            font=("Courier", 11, "bold"),
                            anchor="w",
                        ).grid(row=row, column=1, sticky="w", pady=1)
                        self._pos_rows[mini_id] = {"name_var": name_var, "pos_var": pos_var}

                    display = coord if coord else "—"
                    self._pos_rows[mini_id]["pos_var"].set(display)
            except Exception:
                pass

        try:
            self.root.after(0, _update)
        except Exception:
            pass

    # keep log_movement as a no-op so old call sites don't crash
    def log_movement(self, mini_id: str, grid_coord: str) -> None:
        pass

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
        foundry_connected: Optional[bool] = None,
        locked: Optional[bool] = None,
        marker_count: Optional[int] = None,
        missing_ids: Optional[List[int]] = None,
        fps: Optional[float] = None,
    ):
        """
        Update the status labels.
        Safe to call frequently from the CV loop.
        """
        # Only update lock label if caller provides locked=...
        if locked is not None:
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

        if fps is not None:
            try:
                self.var_fps.set(f"FPS: {float(fps):.1f}")
            except Exception:
                pass

    def get_toggles(self) -> Dict[str, bool]:
        with self._lock:
            return dict(self._toggles)

    def get_motion_thresh(self) -> int:
        """Return the current motion threshold slider value."""
        try:
            return int(self._motion_thresh_var.get())
        except Exception:
            return 30

    def pop_actions(self) -> Dict[str, object]:
        """
        Return current one-shot actions and clear them.
        """
        with self._lock:
            out = dict(self._actions)
            self._actions["capture"] = False
            self._actions["exit"] = False
            self._actions["recapture_bg"] = False
            self._actions["switch_engine"] = None
            self._actions["calibrate_band"] = None
        return out

    def pump(self) -> bool:
        """
        Non-blocking UI update.
        Returns False if user requested exit (via close or Exit button).
        """
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            return False

        with self._lock:
            if self._actions.get("exit"):
                return False
        return True