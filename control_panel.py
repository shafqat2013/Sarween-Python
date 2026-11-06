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
from app_paths import atomic_write_json, data_path


CONFIG_FILE = str(data_path("hardware_config.json"))


def control_panel_window_bounds(screen_width, screen_height):
    """Return initial/minimum sizes in Tk's logical screen coordinates."""
    available_width = max(1, int(screen_width) - 64)
    available_height = max(1, int(screen_height) - 100)
    width, height = min(800, available_width), min(760, available_height)
    return width, height, min(680, width), min(520, height)


def _wrap_label(label, initial_width=640):
    label.configure(wraplength=initial_width, justify="left")

    def resized(event):
        # Ignore the temporary one-pixel geometry before Tk lays out the page.
        if event.width > 1:
            label.configure(wraplength=event.width)

    label.bind("<Configure>", resized)
    return label


def _scrollable_content(parent):
    """Scroll only a long page, leaving the notebook and footer stationary."""
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    canvas = tk.Canvas(parent, highlightthickness=0, width=1, height=1)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.grid(row=0, column=0, sticky="nsew")
    scrollbar.grid(row=0, column=1, sticky="ns")
    canvas.configure(yscrollcommand=scrollbar.set)
    content = ttk.Frame(canvas, padding=12)
    window = canvas.create_window(0, 0, window=content, anchor="nw")
    content.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>", lambda event: canvas.itemconfigure(window, width=max(1, event.width)))
    content._scroll_canvas = canvas
    return content


def _bind_page_scroll(content):
    canvas = content._scroll_canvas
    tag = f"ControlPanelScroll{id(content)}"

    def scroll(event):
        direction = -1 if getattr(event, "delta", 0) > 0 or getattr(event, "num", None) == 4 else 1
        canvas.yview_scroll(direction, "units")
        return "break"

    for event in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        content.bind_class(tag, event, scroll)

    def include(widget):
        widget.bindtags((tag, *widget.bindtags()))
        for child in widget.winfo_children():
            include(child)

    include(content)


def rc_to_a1(row: int, col: int) -> str:
    col_letter = ""
    c = col
    while True:
        col_letter = chr(ord('A') + (c % 26)) + col_letter
        c = c // 26 - 1
        if c < 0:
            break
    return f"{col_letter}{row + 1}"


def format_camera_calibration_status(status):
    status = status if isinstance(status, dict) else {}
    name = status.get("profileName") or "no profile"
    if status.get("applied"):
        text = f"Camera: {name} · lens correction on"
    else:
        reason = str(status.get("reason") or "no matching calibration")
        text = f"Camera: {name} · lens correction off: {reason}"
    size = status.get("frameSize")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        text += f" ({size[0]} × {size[1]})"
    return text


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
        atomic_write_json(CONFIG_FILE, data)
    except Exception:
        pass


class ControlPanel:
    def set_camera_calibration_status(self, status):
        message = format_camera_calibration_status(status)
        if message == getattr(self, "_camera_calibration_text", None):
            return
        self._camera_calibration_text = message
        try:
            self.var_camera_calibration.set(message)
        except tk.TclError:
            pass

    def set_access_status(self, message):
        if not hasattr(self, "_access_label"):
            self._access_label = _wrap_label(ttk.Label(self._footer))
            self._access_label.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self._access_label.configure(text=message)

    def __init__(self, mode: str = "self_hosted", tk_root: tk.Tk = None, *, access=None, foundry_service=None,
                 parent=None):
        self.mode = (mode or "self_hosted").strip().lower()
        if self.mode not in ("self_hosted", "foundry"):
            self.mode = "self_hosted"
        self._access = access
        self.grid_controls = None
        self.vtt_controls = None

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
            "scan_mini": None,
            "retry_moves": False,
            "open_observer": False,
        }
        self._library_rows: List[Dict[str, object]] = []
        self._library_window = None
        self._library_tree = None
        self._ring_scan_window = None

        # ── Tk root / window ─────────────────────────────────────────────────
        self._embedded = parent is not None
        if self._embedded:
            self._owns_root = False
            self._tk_root = parent.winfo_toplevel()
            self.root = ttk.Frame(parent)
        elif tk_root is not None:
            self._owns_root = False
            self._tk_root = tk_root
            self.root = tk.Toplevel(tk_root)
        else:
            self._owns_root = True
            self._tk_root = tk.Tk()
            self.root = self._tk_root

        if not self._embedded:
            self.root.title("Sarween Control Panel")
            width, height, min_width, min_height = control_panel_window_bounds(
                self.root.winfo_screenwidth(), self.root.winfo_screenheight())
            self.root.geometry(f"{width}x{height}")
            self.root.minsize(min_width, min_height)
            self.root.resizable(True, True)
            self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Status vars ───────────────────────────────────────────────────────
        self.var_mode    = tk.StringVar(value=f"Mode: {self.mode}")
        self.var_camera_calibration = tk.StringVar(value=format_camera_calibration_status(None))
        self.var_lock    = tk.StringVar(value="Lock: (waiting)")
        self.var_markers = tk.StringVar(value="Markers: --")
        self.var_missing = tk.StringVar(value="Missing: --")
        self.var_foundry = tk.StringVar(value="Foundry: (n/a)")
        self.var_delivery = tk.StringVar(value="")
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
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        ttk.Label(outer, text="Sarween Control Panel",
                  font=("Helvetica", 16, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.tabs = ttk.Notebook(outer)
        self.tabs.grid(row=1, column=0, sticky="nsew")
        session_tab = ttk.Frame(self.tabs, padding=12)
        scans_tab = ttk.Frame(self.tabs, padding=12)
        diagnostics_page = ttk.Frame(self.tabs)
        diagnostics_tab = _scrollable_content(diagnostics_page)
        self.tabs.add(session_tab, text="Session")
        self.tabs.add(scans_tab, text="Scans")
        self.tabs.add(diagnostics_page, text="Diagnostics")
        if self.mode == "foundry":
            from vtt_controls import VTTControls
            vtt_tab = ttk.Frame(self.tabs, padding=12)
            self.tabs.add(vtt_tab, text="VTT Controls")
            self.vtt_controls = VTTControls(vtt_tab,
                require_access=access.require if access is not None else None,
                foundry_service=foundry_service, live_session=True)
            self.vtt_controls.pack(fill="both", expand=True)
            self.grid_controls = self.vtt_controls.grid_controls
        from map_view import MapView
        self.map_view = MapView(self.tabs)
        self.tabs.add(self.map_view, text="Map")

        self._footer = ttk.Frame(outer)
        self._footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        self._footer.columnconfigure(0, weight=1)
        _wrap_label(ttk.Label(self._footer, textvariable=self.var_hint)).grid(
            row=0, column=0, sticky="ew")

        # Calibration row
        cal_box = ttk.Frame(scans_tab)
        cal_box.pack(fill="x", pady=(0, 10))
        cal_box.columnconfigure(1, weight=1)
        ttk.Label(cal_box, text="Mini name").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self._band_name_var = tk.StringVar(value="")
        ttk.Entry(cal_box, textvariable=self._band_name_var, width=16).grid(
            row=0, column=1, sticky="ew", padx=(0, 8))
        ttk.Button(cal_box, text="Calibrate",
                   command=self._act_calibrate_band).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(cal_box, text="Auto",
                   command=self._act_calibrate_band_auto).grid(row=0, column=3)

        # Status
        status_box = ttk.Frame(session_tab)
        status_box.pack(fill="x")
        for column in (0, 1):
            status_box.columnconfigure(column, weight=1, uniform="status")
        for row, variables in enumerate(((self.var_mode, self.var_fps),
                                         (self.var_lock, self.var_foundry),
                                         (self.var_markers, self.var_missing))):
            for column, variable in enumerate(variables):
                _wrap_label(ttk.Label(status_box, textvariable=variable), 320).grid(
                    row=row, column=column, sticky="new", padx=(0, 12) if column == 0 else 0, pady=2)
        _wrap_label(ttk.Label(status_box, textvariable=self.var_camera_calibration)).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        if self.mode == "foundry":
            delivery = ttk.Frame(status_box)
            delivery.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
            delivery.columnconfigure(0, weight=1)
            _wrap_label(ttk.Label(delivery, textvariable=self.var_delivery), 480).grid(
                row=0, column=0, sticky="ew", padx=(0, 8))
            self._retry_moves_btn = ttk.Button(delivery, text="Retry moves",
                command=self._act_retry_moves, state="disabled")
            self._retry_moves_btn.grid(row=0, column=1, sticky="e")

        ttk.Separator(session_tab, orient="horizontal").pack(fill="x", pady=12)

        # Windows toggles
        ttk.Label(diagnostics_tab, text="Windows").pack(anchor="w")
        for text, var in [
            ("Live Camera View",                   self._var_show_live),
            ("Homography View",                    self._var_show_h),
            ("Identify / Warp Debug",              self._var_show_ident),
            ("Calibration Preview",                self._var_show_calib_preview),
            ("Show Timing (terminal)",             self._var_show_timing),
            ("Verbose Tracking (terminal)",        self._var_verbose_tracking),
        ]:
            ttk.Checkbutton(diagnostics_tab, text=text, variable=var,
                            command=self._sync_toggles_from_ui).pack(anchor="w")

        ttk.Separator(diagnostics_tab, orient="horizontal").pack(fill="x", pady=8)

        # Masks toggles
        ttk.Label(diagnostics_tab, text="Masks").pack(anchor="w")
        for text, var in [
            ("Motion (warp)",          self._var_show_motion_warp),
            ("Motion (camera)",        self._var_show_motion_cam),
            ("Shadow-free mask",       self._var_show_shadowfree),
            ("Final mask",             self._var_show_final_mask),
        ]:
            ttk.Checkbutton(diagnostics_tab, text=text, variable=var,
                            command=self._sync_toggles_from_ui).pack(anchor="w")

        ttk.Separator(diagnostics_tab, orient="horizontal").pack(fill="x", pady=8)

        # Actions
        btn_row = ttk.Frame(session_tab)
        btn_row.pack(fill="x")
        btn_row.columnconfigure(2, weight=1)
        ttk.Button(btn_row, text="Recapture BG",
                   command=self._act_recapture_bg).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(scans_tab, text="Calibrate Minis",
                   command=self._act_calibrate_minis).pack(anchor="w", pady=(8, 0))
        ttk.Button(btn_row, text="Stop session",
                   command=self._act_exit).grid(row=0, column=3, sticky="e")

        rec_row = ttk.Frame(session_tab)
        rec_row.pack(fill="x", pady=(8, 0))
        rec_row.columnconfigure(1, weight=1)
        self._rec_btn_var = tk.StringVar(value="⏺  Record")
        ttk.Button(rec_row, textvariable=self._rec_btn_var,
                   command=self._act_toggle_recording).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.var_recording = tk.StringVar(value="")
        _wrap_label(ttk.Label(rec_row, textvariable=self.var_recording,
                  foreground="red"), 480).grid(row=0, column=1, sticky="ew")

        observer_row = ttk.Frame(session_tab)
        observer_row.pack(fill="x", pady=(10, 0))
        ttk.Button(observer_row, text="Open six-mini observer",
                   command=self._act_open_observer).pack(side="left", padx=(0, 8))
        _wrap_label(ttk.Label(observer_row, text="Experimental · Foundry movement off while observing"),
                    400).pack(side="left", fill="x", expand=True)

        debug_row = ttk.Frame(diagnostics_tab)
        debug_row.pack(fill="x", pady=(4, 0))
        ttk.Button(debug_row, text="Dump State",
                   command=self._act_dump_state).pack(side="left")
        ttk.Button(btn_row, text="Mini Library",
                   command=self._show_mini_library).grid(row=0, column=1, padx=(0, 8))

        # Motion threshold
        thresh_box = ttk.Frame(diagnostics_tab)
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

        ttk.Separator(session_tab, orient="horizontal").pack(fill="x", pady=12)

        # Mini positions
        ttk.Label(session_tab, text="Mini Positions").pack(anchor="w", pady=(0, 8))
        self._pos_table = ttk.Frame(session_tab)
        self._pos_table.pack(fill="both", expand=True)
        self._pos_tree = ttk.Treeview(self._pos_table, columns=("mini", "cell"),
                                      show="headings", height=7, selectmode="browse")
        self._pos_tree.heading("mini", text="Mini")
        self._pos_tree.heading("cell", text="Cell")
        self._pos_tree.column("mini", width=300, minwidth=140, stretch=True, anchor="w")
        self._pos_tree.column("cell", width=120, minwidth=80, stretch=False, anchor="w")
        position_scroll = ttk.Scrollbar(self._pos_table, orient="vertical", command=self._pos_tree.yview)
        position_scroll.pack(side="right", fill="y")
        self._pos_tree.pack(side="left", fill="both", expand=True)
        self._pos_tree.configure(yscrollcommand=position_scroll.set)
        self._pos_rows: dict = {}
        _bind_page_scroll(diagnostics_tab)

    # ── UI handlers ───────────────────────────────────────────────────────────

    def _on_close(self):
        with self._lock:
            self._actions["exit"] = True
        try:
            if not self._embedded:
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

    def _act_open_observer(self):
        with self._lock:
            self._actions["open_observer"] = True

    def _act_dump_state(self):
        with self._lock:
            self._actions["dump_state"] = True

    def _show_mini_library(self):
        if self._library_window is not None:
            try:
                if self._library_window.winfo_exists():
                    self._library_window.deiconify()
                    self._library_window.lift()
                    return
            except Exception:
                pass

        window = tk.Toplevel(self.root)
        window.title("Sarween Mini Library")
        window.geometry("760x340")
        window.minsize(680, 300)
        self._library_window = window

        outer = ttk.Frame(window, padding=14)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Mini Library", font=("Helvetica", 16, "bold")).pack(
            anchor="w", pady=(0, 10)
        )

        columns = ("ring", "scan", "foundry", "confidence", "cell")
        tree = ttk.Treeview(outer, columns=columns, show="headings", height=7)
        headings = {
            "ring": "Mini / ring",
            "scan": "Scan portfolio",
            "foundry": "Foundry token",
            "confidence": "Recognition",
            "cell": "Last cell",
        }
        widths = {
            "ring": 125,
            "scan": 150,
            "foundry": 150,
            "confidence": 120,
            "cell": 85,
        }
        for column in columns:
            tree.heading(column, text=headings[column])
            tree.column(column, width=widths[column], minwidth=70, anchor="w")
        tree.pack(fill="both", expand=True)
        self._library_tree = tree

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(10, 0))
        ttk.Button(actions, text="Scan selected", command=self._act_scan_selected).pack(
            side="left"
        )
        ttk.Button(
            actions,
            text="Full brightness scan",
            command=self._act_library_brightness_scan,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Close", command=window.destroy).pack(side="right")

        window.protocol("WM_DELETE_WINDOW", window.destroy)
        self._render_mini_library()

    def _act_scan_selected(self):
        tree = self._library_tree
        selected = tree.selection() if tree is not None else ()
        if not selected:
            self.set_hint("Choose a mini in the Mini Library first")
            return
        mini_id = str(selected[0])
        with self._lock:
            self._actions["scan_mini"] = mini_id
        self.set_hint(f"Move {mini_id} now; Sarween is waiting for its ring scan")

    def _act_library_brightness_scan(self):
        with self._lock:
            self._actions["calibrate_minis"] = True
        self.set_hint("Starting full brightness scan")

    def _render_mini_library(self):
        tree = self._library_tree
        if tree is None:
            return
        try:
            selected = set(tree.selection())
            for item in tree.get_children():
                tree.delete(item)
            color_tags = {
                "red": "#b42318",
                "blue": "#175cd3",
                "yellow": "#8a6500",
                "green": "#067647",
                "white": "#667085",
            }
            for color, foreground in color_tags.items():
                tree.tag_configure(color, foreground=foreground)
            for row in self._library_rows:
                mini_id = str(row["id"])
                color = str(row.get("ringColor") or "unknown").lower()
                count = int(row.get("sampleCount") or 0)
                scan = f"{row.get('scanStatus', 'Needs scan')} ({count} samples)"
                confidence = row.get("confidence")
                confidence_text = (
                    f"Seen {float(confidence) * 100:.0f}%"
                    if confidence is not None
                    else "Not seen"
                )
                tree.insert(
                    "",
                    "end",
                    iid=mini_id,
                    values=(
                        f"{row.get('name', mini_id)} / {color}",
                        scan,
                        row.get("token") or "Not assigned",
                        confidence_text,
                        row.get("position") or "--",
                    ),
                    tags=(color,) if color in color_tags else (),
                )
            for mini_id in selected:
                if tree.exists(mini_id):
                    tree.selection_add(mini_id)
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────

    def update_map(self, state: dict) -> None:
        self.map_view.set_state(state)

    def update_positions(self, positions: dict) -> None:
        def _update():
            try:
                for mini_id in self._pos_tree.get_children():
                    if mini_id not in positions:
                        self._pos_tree.delete(mini_id)
                for mini_id, coord in positions.items():
                    values = (mini_id, coord or "—")
                    if self._pos_tree.exists(mini_id):
                        self._pos_tree.item(mini_id, values=values)
                    else:
                        self._pos_tree.insert("", "end", iid=mini_id, values=values)
                self._pos_rows = dict(positions)
            except Exception:
                pass
        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def update_mini_library(self, rows: List[Dict[str, object]]) -> None:
        def _update():
            self._library_rows = [dict(row) for row in rows]
            if self.vtt_controls is not None:
                self.vtt_controls.set_mini_rows(self._library_rows)
            self._render_mini_library()

        try:
            self.root.after(0, _update)
        except Exception:
            pass

    def log_movement(self, mini_id: str, grid_coord: str) -> None:
        pass

    def show(self):
        try:
            if self._embedded:
                self.root.pack(fill="both", expand=True)
            else:
                self.root.deiconify()
            self.root.lift()
        except Exception:
            pass

    def hide(self):
        try:
            if self._embedded:
                self.root.pack_forget()
            else:
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

    def choose_ring_sample(self, name, image):
        from ring_scan_dialog import show_ring_scan
        if self._ring_scan_window is not None and self._ring_scan_window.winfo_exists():
            self._ring_scan_window.destroy()

        def confirmed(sample):
            with self._lock:
                self._actions["calibrate_band"] = sample

        self._ring_scan_window = show_ring_scan(self.root, name, image, confirmed)

    def _act_retry_moves(self):
        with self._lock:
            self._actions["retry_moves"] = True

    def set_delivery_status(self, status):
        if self.mode != "foundry":
            return
        self.var_delivery.set(str(status["message"]))
        self._retry_moves_btn.configure(state="normal" if status["retryAvailable"] else "disabled")

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
        if foundry_connected is not None:
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
            self._actions["scan_mini"]        = None
            self._actions["retry_moves"]      = False
            self._actions["open_observer"]    = False
        return out

    def pump(self) -> bool:
        try:
            if self.vtt_controls is not None:
                self.vtt_controls.set_enabled(self._access is None or self._access.allowed)
                self.vtt_controls.refresh()
            self._tk_root.update_idletasks()
            self._tk_root.update()
        except tk.TclError:
            return False
        with self._lock:
            if self._actions.get("exit"):
                return False
        return True
