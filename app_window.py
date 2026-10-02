"""The real, hardware-optional Sarween home window."""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app_paths import data_dir, data_path
import mini_library as ml


def _read_object(path):
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def saved_roster():
    """Read portfolios without importing samples into the user's saved files."""
    _read_object(data_path("mini_library.json"))
    library = ml.load_library(data_path("mini_library.json"))
    profiles = _read_object(data_path("combo_profiles.json"))
    ml.sync_profile_samples(library, profiles)
    mappings = _read_object(data_path("mini_token_map.json"))
    return library, ml.library_rows(library, profiles=profiles, mappings=mappings)


class AppWindow:
    def __init__(self, root, *, service, run_session, auth_panel_factory=None):
        self.root = root
        self.service = service
        self.run_session = run_session
        self.closed = False
        self.busy = False
        self.replay = None
        self._poll_id = None
        self.library = {"minis": {}}
        root.title("Sarween")
        root.geometry(f"860x{min(620, root.winfo_screenheight() - 100)}")
        root.minsize(700, 460)
        root.protocol("WM_DELETE_WINDOW", self.close)

        # Keep the home controls out of both the layout and keyboard navigation
        # until approval has been checked (including during session restoration).
        self.login = ttk.Frame(root)
        self.login.pack(fill="both", expand=True)
        self.home = outer = ttk.Frame(root, padding=16)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="Sarween", font=("Helvetica", 20, "bold")).pack(side="left")
        self.start_button = ttk.Button(header, text="Start live session", command=self.start_session)
        self.start_button.pack(side="right")
        self.recording_button = ttk.Button(header, text="Open your video", command=self.open_recording)
        self.recording_button.pack(side="right", padx=6)
        self.demo_button = ttk.Button(header, text="Watch real example", command=self.open_demo)
        self.demo_button.pack(side="right")
        self.status = tk.StringVar(root, value="Idle - camera off")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(0, 12))

        from auth_ui import AuthPanel
        self.auth = (auth_panel_factory or AuthPanel)(
            root, self.login, account_parent=outer, changed=self._access_changed)
        # The live session runner and relay share this same gate.
        self.service.access = self.auth

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        library_tab = ttk.Frame(self.tabs, padding=12)
        connections = ttk.Frame(self.tabs, padding=12)
        diagnostics = ttk.Frame(self.tabs, padding=12)
        for frame, label in ((library_tab, "Mini Library"), (connections, "Connections"),
                             (diagnostics, "Diagnostics")):
            self.tabs.add(frame, text=label)

        columns = ("mini", "ring", "profile", "samples", "token")
        table = ttk.Frame(library_tab)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(table, columns=columns, show="headings", height=6, selectmode="browse")
        for column, title, width in zip(columns,
                ("Mini", "Ring", "Tracking profile", "Verified samples", "Saved token ID"),
                (105, 70, 120, 110, 155)):
            self.tree.heading(column, text=title)
            self.tree.column(column, width=width, minwidth=60, anchor="w")
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._select_mini)
        self.detail = tk.StringVar(root, value="No mini selected")
        detail_label = ttk.Label(library_tab, textvariable=self.detail, justify="left", wraplength=700)
        detail_label.pack(fill="x", pady=(12, 8))
        detail_label.bind("<Configure>", lambda event: detail_label.configure(wraplength=max(100, event.width)))
        ttk.Button(library_tab, text="Refresh library", command=self.refresh_library).pack(anchor="w")

        self.connection = tk.StringVar(root, value="Foundry: disconnected")
        self.relay = tk.StringVar(root, value="Local relay: stopped")
        self.scene = tk.StringVar(root, value="Scene: unavailable")
        for variable in (self.connection, self.relay, self.scene):
            ttk.Label(connections, textvariable=variable, wraplength=610).pack(anchor="w", pady=(0, 8))
        ttk.Label(connections, text="Camera: off").pack(anchor="w", pady=(0, 8))
        ttk.Label(connections, text="Marker registration: inactive").pack(anchor="w", pady=(0, 12))
        self.connect_button = ttk.Button(connections, text="Connect Foundry", command=self.toggle_connection)
        self.connect_button.pack(anchor="w")

        self.diagnostic = tk.StringVar(root, value="No active session")
        ttk.Label(diagnostics, textvariable=self.diagnostic, wraplength=610, justify="left").pack(anchor="w")
        ttk.Separator(diagnostics).pack(fill="x", pady=12)
        path_label = ttk.Label(diagnostics, text=f"User data\n{data_dir()}", wraplength=610, justify="left")
        path_label.pack(anchor="w", fill="x")
        path_label.bind("<Configure>", lambda event: path_label.configure(wraplength=max(100, event.width)))
        self.refresh_library()
        self._access_changed()
        self._poll()
        root.deiconify()
        if not self.auth.allowed:
            self.auth.email_entry.focus_set()

    def refresh_library(self):
        try:
            library, rows = saved_roster()
        except Exception as exc:
            self.status.set("Saved mini data could not be loaded")
            self.diagnostic.set(str(exc))
            return
        self.library = library
        selected = self.tree.selection()
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in rows:
            self.tree.insert("", "end", iid=row["id"], values=(
                row["name"], row["ringColor"],
                "Available" if row["scanStatus"] == "Ready" else "Needs scan",
                row["sampleCount"], row["token"] or "Not assigned"))
        children = self.tree.get_children()
        if children:
            self.tree.selection_set(selected[0] if selected and selected[0] in children else children[0])
        self._select_mini()

    def _select_mini(self, _event=None):
        selected = self.tree.selection()
        if not selected:
            self.detail.set("No mini selected")
            return
        mini_id = selected[0]
        entry = self.library["minis"][mini_id]
        samples = entry.get("samples", [])
        verified = sum(bool(sample.get("verified")) for sample in samples)
        dates = sorted(str(sample["capturedAt"]) for sample in samples if sample.get("capturedAt"))
        self.detail.set(f"{entry.get('name', mini_id)} ({mini_id})\n"
                        f"{verified} verified samples / {len(samples) - verified} unverified samples\n"
                        f"Last capture: {dates[-1] if dates else 'Not recorded'}")

    def toggle_connection(self):
        try:
            if self.service.running:
                self.service.stop()
            else:
                self.auth.require()
                self.service.start()
        except Exception as exc:
            self.status.set("Foundry connection failed")
            self.diagnostic.set(str(exc))
        self._refresh_connection()

    def _refresh_connection(self):
        import foundryoutput as fo
        state = self.service.state
        self.relay.set(f"Local relay: {state.lower()}")
        self.connect_button.configure(text="Disconnect Foundry" if self.service.running else "Connect Foundry")
        delivery = fo.get_delivery_status()
        connected = self.service.running and delivery["connected"]
        if hasattr(self.auth, "usage_state"):
            self.auth.usage_state("foundry", connected)
        self.connection.set(f"Foundry: {delivery['message']}" if connected else "Foundry: disconnected")
        scene = fo.get_scene_params() if connected else {}
        self.scene.set(f"Scene: {scene['gridCols']} columns x {scene['gridRows']} rows"
                       if scene.get("sceneId") else "Scene: unavailable")
        if self.service.error:
            self.status.set("Foundry connection failed")
            self.diagnostic.set(self.service.error)

    def _poll(self):
        if self.closed:
            return
        self._refresh_connection()
        self._access_changed()
        self._poll_id = self.root.after(500, self._poll)

    def start_session(self):
        if self.busy or self.closed:
            return
        if not self.auth.allowed:
            self.status.set(self.auth.status())
            return
        if self.replay:
            self.replay.close()
        self.busy = True
        self.start_button.state(["disabled"])
        self.status.set("Opening live setup")
        self.root.update_idletasks()
        self.root.withdraw()
        try:
            self.run_session(self.service)
            self.status.set("Session stopped - camera off" if self.auth.allowed else self.auth.status())
        except SystemExit:
            self.status.set("Setup or session cancelled - camera off")
        except Exception as exc:
            self.status.set("Live session could not continue - camera off")
            if hasattr(self.auth, "usage_event"):
                self.auth.usage_event("live_errors")
            self.diagnostic.set(str(exc))
            self.tabs.select(2)
            import traceback
            traceback.print_exc()
        finally:
            self.busy = False
            if not self.closed:
                self._access_changed()
                self.refresh_library()
                self.root.deiconify()
                self.root.lift()

    def open_recording(self, path=None, *, demo=False):
        if self.busy or self.closed:
            return
        if not self.auth.allowed:
            self.status.set(self.auth.status())
            return
        path = path or filedialog.askopenfilename(parent=self.root, title="Open your video",
                    filetypes=[("Video", "*.mp4 *.mov *.avi *.mkv"), ("All files", "*")])
        if not path:
            return
        try:
            from replay_window import ReplayWindow
            if self.replay:
                self.replay.close()
            self.replay = ReplayWindow(self.root, path, demo=demo)
            if hasattr(self.auth, "usage_event"):
                self.auth.usage_event("demo_opens" if demo else "replay_opens")
        except Exception as exc:
            messagebox.showerror("Open recording", str(exc), parent=self.root)

    def open_demo(self):
        from pathlib import Path
        import sys
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        self.open_recording(base / "demo" / "tabletop.mp4", demo=True)

    def close(self):
        if self.closed:
            return
        if self.replay:
            self.replay.close()
        self.auth.close()
        self.service.stop()
        self.closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        self.root.destroy()

    def _access_changed(self):
        # AuthPanel can notify before the remainder of the window exists.
        if not hasattr(self, "auth") or not hasattr(self, "connect_button"):
            return
        allowed = self.auth.allowed
        if allowed and not self.home.winfo_manager():
            self.login.pack_forget()
            self.home.pack(fill="both", expand=True)
            self.demo_button.focus_set()
        elif not allowed and not self.login.winfo_manager():
            self.home.pack_forget()
            self.login.pack(fill="both", expand=True)
            self.auth.email_entry.focus_set()
        for button in (self.start_button, self.recording_button, self.demo_button):
            button.state(["!disabled"] if allowed and not self.busy else ["disabled"])
        self.connect_button.state(["!disabled"] if allowed or self.service.running else ["disabled"])
        if not allowed:
            if self.replay:
                self.replay.close()
                self.replay = None
            if self.service.running:
                self.service.stop()
