# control_panel.py
# Sarween runtime control panel — combo engine only.
# Pure native ttk, matches Sarween Setup window appearance.

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional


CONFIG_FILE = "hardware_config.json"


def rc_to_a1(row: int, col: int) -> str:
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
        pass


class ControlPanel:
    def __init__(self, mode: str = "self_hosted", tk_root: tk.Tk = None):
        self.mode = (mode or "self_hosted").strip().lower()
        if self.mode not in ("self_hosted", "foundry"):
            self.mode = "self_hosted"

        self._lock = threading.Lock()

        self._toggles: Dict[str, bool] = {
            "show_live_camera": False,
            "show_homography": False,
            "show_identify": False,
            "show_blended": False,
            "show_motion_warp": False,
            "show_motion_cam": False,
            "show_shadowfree": False,
            "show_final_mask": False,
            "show_hsv_overlap": False,
            "show_calib_preview": False,
            "show_timing": False,
            "verbose_tracking": False,
        }

        self._actions: Dict[str, object] = {
            "exit": False,
            "recapture_bg": False,
            "calibrate_band": None,
            "calibrate_minis": False,
            "toggle_recording": False,
            "dump_state": False,
        }

        # ── Tk root / window ─────────────────────────────────────────────────
        if tk_root is not None:
            self._owns_root = False
            self._tk_root = tk_root
            self.root = tk.Toplevel(tk_root)
        else:
            self._owns_root = True
            self._tk_root = tk.Tk()
            self.root = self._tk_root

        self.root.title("Sarween Control Panel")
        self.root.geometry("480x640")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Status vars ───────────────────────────────────────────────────────
        self.var_mode    = tk.StringVar(value=f"Mode: {self.mode}")
        self.var_lock    = tk.StringVar(value="Lock: (waiting)")
        self.var_markers = tk.StringVar(value="Markers: --")
        self.var_missing = tk.StringVar(value="Missing: --")
        self.var_foundry = tk.StringVar(value="Foundry: (n/a)")
        self.var_fps     = tk.StringVar(value="FPS: --")
        self.var_hint    = tk.StringVar(value="")

        # ── Checkbutton vars ──────────────────────────────────────────────────
        self._var_show_live          = tk.BooleanVar(value=self._toggles["show_live_camera"])
        self._var_show_h             = tk.BooleanVar(value=self._toggles["show_homography"])
        self._var_show_ident         = tk.BooleanVar(value=self._toggles["show_identify"])
        self._var_show_blended       = tk.BooleanVar(value=self._toggles["show_blended"])
        self._var_show_motion_warp   = tk.BooleanVar(value=self._toggles["show_motion_warp"])
        self._var_show_motion_cam    = tk.BooleanVar(value=self._toggles["show_motion_cam"])
        self._var_show_shadowfree    = tk.BooleanVar(value=self._toggles["show_shadowfree"])
        self._var_show_final_mask    = tk.BooleanVar(value=self._toggles["show_final_mask"])
        self._var_show_hsv_overlap   = tk.BooleanVar(value=self._toggles["show_hsv_overlap"])
        self._var_show_calib_preview = tk.BooleanVar(value=self._toggles["show_calib_preview"])
        self._var_show_timing        = tk.BooleanVar(value=self._toggles["show_timing"])
        self._var_verbose_tracking   = tk.BooleanVar(value=self._toggles["verbose_tracking"])

        # ── Layout ────────────────────────────────────────────────────────────
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        ttk.Label(outer, text="Sarween Control Panel",
                  font=("Helvetica", 16, "bold")).pack(anchor="w", pady=(0, 10))

        # Calibration row
        cal_box = ttk.Frame(outer)
        cal_box.pack(fill="x", pady=(0, 10))
        ttk.Label(cal_box, text="Mini name").pack(side="left")
        self._band_name_var = tk.StringVar(value="")
        ttk.Entry(cal_box, textvariable=self._band_name_var, width=16).pack(
            side="left", padx=(8, 8))
        ttk.Button(cal_box, text="Calibrate",
                   command=self._act_calibrate_band).pack(side="left", padx=(0, 4))
        ttk.Button(cal_box, text="Auto",
                   command=self._act_calibrate_band_auto).pack(side="left")

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        # Status
        status_box = ttk.Frame(outer)
        status_box.pack(fill="x", pady=(0, 6))
        for var in (self.var_mode, self.var_lock, self.var_markers,
                    self.var_missing, self.var_fps, self.var_foundry):
            ttk.Label(status_box, textvariable=var).pack(anchor="w")

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        # Windows toggles
        ttk.Label(outer, text="Windows").pack(anchor="w")
        for text, var in [
            ("Live Camera View",                   self._var_show_live),
            ("Homography View",                    self._var_show_h),
            ("Identify / Warp Debug",              self._var_show_ident),
            ("Calibration Preview",                self._var_show_calib_preview),
            ("Show Timing (terminal)",             self._var_show_timing),
            ("Verbose Tracking (terminal)",        self._var_verbose_tracking),
        ]:
            ttk.Checkbutton(outer, text=text, variable=var,
                            command=self._sync_toggles_from_ui).pack(anchor="w")

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        # Masks toggles
        ttk.Label(outer, text="Masks").pack(anchor="w")
        for text, var in [
            ("Motion (warp)",          self._var_show_motion_warp),
            ("Motion (camera)",        self._var_show_motion_cam),
            ("Shadow-free mask",       self._var_show_shadowfree),
            ("Final mask",             self._var_show_final_mask),
        ]:
            ttk.Checkbutton(outer, text=text, variable=var,
                            command=self._sync_toggles_from_ui).pack(anchor="w")

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        # Actions
        ttk.Label(outer, text="Actions").pack(anchor="w")
        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="Recapture BG",
                   command=self._act_recapture_bg).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Calibrate Minis",
                   command=self._act_calibrate_minis).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Exit",
                   command=self._act_exit).pack(side="right")

        rec_row = ttk.Frame(outer)
        rec_row.pack(fill="x", pady=(4, 0))
        self._rec_btn_var = tk.StringVar(value="⏺  Record")
        ttk.Button(rec_row, textvariable=self._rec_btn_var,
                   command=self._act_toggle_recording).pack(side="left")
        self.var_recording = tk.StringVar(value="")
        ttk.Label(rec_row, textvariable=self.var_recording,
                  foreground="red").pack(side="left", padx=(10, 0))

        debug_row = ttk.Frame(outer)
        debug_row.pack(fill="x", pady=(4, 0))
        ttk.Button(debug_row, text="Dump State",
                   command=self._act_dump_state).pack(side="left")

        # Motion threshold
        thresh_box = ttk.Frame(outer)
        thresh_box.pack(fill="x", pady=(10, 0))
        ttk.Label(thresh_box, text="Motion threshold:").pack(anchor="w")
        thresh_row = ttk.Frame(thresh_box)
        thresh_row.pack(fill="x", pady=(4, 0))
        self._motion_thresh_var = tk.IntVar(value=30)
        self._thresh_label_var  = tk.StringVar(value="30")
        ttk.Scale(
            thresh_row, from_=5, to=80, orient="horizontal",
            variable=self._motion_thresh_var,
            command=self._on_thresh_change,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(thresh_row, textvariable=self._thresh_label_var, width=4).pack(side="left")

        # Hint
        ttk.Label(outer, textvariable=self.var_hint).pack(anchor="w", pady=(8, 0))

        ttk.Separator(outer, orient="horizontal").pack(fill="x", pady=8)

        # Mini positions
        ttk.Label(outer, text="Mini Positions").pack(anchor="w", pady=(0, 4))
        self._pos_table = ttk.Frame(outer)
        self._pos_table.pack(fill="both", expand=True)
        self._pos_rows: dict = {}

    # ── UI handlers ───────────────────────────────────────────────────────────

    def _on_close(self):
        with self._lock:
            self._actions["exit"] = True
        try:
            self.root.withdraw()
        except Exception:
            pass

    def _on_thresh_change(self, val):
        try:
            self._thresh_label_var.set(str(int(float(val))))
        except Exception:
            pass

    def _sync_toggles_from_ui(self):
        with self._lock:
            self._toggles["show_live_camera"]   = bool(self._var_show_live.get())
            self._toggles["show_homography"]    = bool(self._var_show_h.get())
            self._toggles["show_identify"]      = bool(self._var_show_ident.get())
            self._toggles["show_blended"]       = bool(self._var_show_blended.get())
            self._toggles["show_calib_preview"] = bool(self._var_show_calib_preview.get())
            self._toggles["show_motion_warp"]   = bool(self._var_show_motion_warp.get())
            self._toggles["show_motion_cam"]    = bool(self._var_show_motion_cam.get())
            self._toggles["show_shadowfree"]    = bool(self._var_show_shadowfree.get())
            self._toggles["show_final_mask"]    = bool(self._var_show_final_mask.get())
            self._toggles["show_hsv_overlap"]   = bool(self._var_show_hsv_overlap.get())
            self._toggles["show_timing"]        = bool(self._var_show_timing.get())
            self._toggles["verbose_tracking"]   = bool(self._var_verbose_tracking.get())

    def _act_exit(self):
        with self._lock:
            self._actions["exit"] = True

    def _act_recapture_bg(self):
        with self._lock:
            self._actions["recapture_bg"] = True

    def _act_calibrate_band(self):
        name = (self._band_name_var.get() if self._band_name_var else "").strip()
        with self._lock:
            self._actions["calibrate_band"] = {"name": name or None}

    def _act_calibrate_band_auto(self):
        with self._lock:
            self._actions["calibrate_band"] = {"name": None, "auto": True}

    def _act_calibrate_minis(self):
        with self._lock:
            self._actions["calibrate_minis"] = True

    def _act_toggle_recording(self):
        with self._lock:
            self._actions["toggle_recording"] = True

    def _act_dump_state(self):
        with self._lock:
            self._actions["dump_state"] = True

    # ── Public API ────────────────────────────────────────────────────────────

    def update_positions(self, positions: dict) -> None:
        def _update():
            try:
                for mini_id, coord in positions.items():
                    if mini_id not in self._pos_rows:
                        row = len(self._pos_rows)
                        name_var = tk.StringVar(value=mini_id)
                        pos_var  = tk.StringVar(value="—")
                        ttk.Label(
                            self._pos_table,
                            textvariable=name_var,
                            width=12,
                            anchor="w",
                        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=1)
                        ttk.Label(
                            self._pos_table,
                            textvariable=pos_var,
                            font=("Courier", 11, "bold"),
                            anchor="w",
                        ).grid(row=row, column=1, sticky="w", pady=1)
                        self._pos_rows[mini_id] = {"name_var": name_var, "pos_var": pos_var}
                    self._pos_rows[mini_id]["pos_var"].set(coord if coord else "—")
            except Exception:
                pass
        try:
            self.root.after(0, _update)
        except Exception:
            pass

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

    def set_recording_status(self, recording: bool, path: str = "") -> None:
        """Update the Record button label and status text."""
        try:
            if recording:
                self._rec_btn_var.set("⏹  Stop Rec")
                import os as _os
                self.var_recording.set(f"● REC  {_os.path.basename(path)}")
            else:
                self._rec_btn_var.set("⏺  Record")
                self.var_recording.set("")
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
        if locked is not None:
            try:
                self.var_lock.set("Lock: ✅ acquired" if locked else "Lock: (waiting)")
            except Exception:
                pass
        if marker_count is not None:
            try:
                self.var_markers.set(f"Markers: {int(marker_count)}/4")
            except Exception:
                pass
        if missing_ids is not None:
            try:
                self.var_missing.set(
                    "Missing: none" if len(missing_ids) == 0
                    else f"Missing: {missing_ids}"
                )
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
        try:
            return int(self._motion_thresh_var.get())
        except Exception:
            return 30

    def pop_actions(self) -> Dict[str, object]:
        with self._lock:
            out = dict(self._actions)
            self._actions["exit"]             = False
            self._actions["recapture_bg"]     = False
            self._actions["calibrate_band"]   = None
            self._actions["calibrate_minis"]  = False
            self._actions["toggle_recording"] = False
            self._actions["dump_state"]       = False
        return out

    def pump(self) -> bool:
        try:
            self._tk_root.update_idletasks()
            self._tk_root.update()
        except tk.TclError:
            return False
        with self._lock:
            if self._actions.get("exit"):
                return False
        return True
