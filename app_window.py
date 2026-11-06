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
    def __init__(self, root, *, service, run_session, auth_panel_factory=None, run_observer=None,
                 inline_startup=True, startup_factory=None):
        self.root = root
        self.service = service
        self.run_session = run_session
        self.run_observer = run_observer
        self.closed = False
        self.busy = False
        self.replay = None
        self._poll_id = None
        self.inline_startup = inline_startup
        self._startup_factory = startup_factory
        self._startup = self._runtime_panel = None
        self._close_requested = False
        self._observer_startup = False
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
        self.runtime_host = ttk.Frame(root)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 12))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Sarween", font=("Helvetica", 20, "bold")).grid(
            row=0, column=0, rowspan=2, sticky="nw")
        self.start_button = ttk.Button(header, text="Start live session", command=self.start_session)
        self.start_button.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.observer_button = None
        if run_observer is not None:
            self.observer_button = ttk.Button(header, text="Start mini observer",
                command=lambda: self.start_session(observer=True))
            self.observer_button.grid(row=0, column=2, sticky="ew")
        self.recording_button = ttk.Button(header, text="Open your video", command=self.open_recording)
        self.recording_button.grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=(6, 0))
        self.demo_button = ttk.Button(header, text="Watch real example", command=self.open_demo)
        self.demo_button.grid(row=1, column=2, sticky="ew", pady=(6, 0))
        self.status = tk.StringVar(root, value="Idle - camera off")
        ttk.Label(outer, textvariable=self.status).pack(anchor="w", pady=(0, 12))

        from auth_ui import AuthPanel
        self.auth = (auth_panel_factory or AuthPanel)(
            root, self.login, account_parent=outer, changed=self._access_changed)
        # The live session runner and relay share this same gate.
        self.service.access = self.auth

        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill="both", expand=True)
        self.session_tab = ttk.Frame(self.tabs, padding=4)
        self.tabs.add(self.session_tab, text="Session")
        self.session_intro = ttk.Label(self.session_tab,
            text="Start a live session or mini observer to choose hardware and connect here.",
            wraplength=640, justify="left")
        self.session_intro.pack(anchor="w", padx=8, pady=12)
        library_tab = ttk.Frame(self.tabs, padding=12)
        connections = ttk.Frame(self.tabs, padding=12)
        hardware = ttk.Frame(self.tabs, padding=12)
        diagnostics = ttk.Frame(self.tabs, padding=12)
        self.diagnostics_tab = diagnostics
        for frame, label in ((library_tab, "Mini Library"), (connections, "VTT Controls"),
                             (hardware, "Hardware Profiles"),
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

        from vtt_controls import VTTControls
        self.vtt_controls = VTTControls(connections, require_access=self.auth.require,
                                        foundry_service=self.service)
        self.vtt_controls.pack(fill="both", expand=True)
        self.connection = self.vtt_controls.connection
        self.relay = self.vtt_controls.relay
        self.scene = self.vtt_controls.scene
        self.connect_button = self.vtt_controls.connect_button
        self.grid_controls = self.vtt_controls.grid_controls
        from hardware_profiles_ui import HardwareProfilesPanel
        self.hardware_profiles = HardwareProfilesPanel(hardware, require_access=self.auth.require,
            display_controller=self.vtt_controls.display_controller,
            changed=self.vtt_controls.display_profiles.refresh_profiles)
        self.hardware_profiles.pack(fill="both", expand=True)

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
        self.vtt_controls.set_mini_rows(rows)
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
        self.vtt_controls.set_enabled(self.auth.allowed)
        self.vtt_controls.refresh()
        self.hardware_profiles.refresh()
        if self.service.error:
            self.status.set("Foundry connection failed")
            self.diagnostic.set(self.service.error)

    def _poll(self):
        if self.closed:
            return
        self._refresh_connection()
        self._access_changed()
        self._poll_id = self.root.after(500, self._poll)

    def start_session(self, *, observer=False):
        if getattr(self, "inline_startup", False):
            self._start_inline_session(observer=observer)
            return
        if self.busy or self.closed:
            return
        if not self.auth.allowed:
            self.status.set(self.auth.status())
            return
        runner = self.run_observer if observer else self.run_session
        if runner is None:
            self.status.set("Mini observer is unavailable in this build")
            return
        if self.replay:
            self.replay.close()
        self.busy = True
        self.start_button.state(["disabled"])
        self.status.set("Opening live setup")
        self.root.update_idletasks()
        self.root.withdraw()
        try:
            runner(self.service)
            self.status.set("Session stopped - camera off" if self.auth.allowed else self.auth.status())
        except SystemExit:
            self.status.set("Setup or session cancelled - camera off")
        except Exception as exc:
            self.status.set("Live session could not continue - camera off")
            if hasattr(self.auth, "usage_event"):
                self.auth.usage_event("live_errors")
            self.diagnostic.set(str(exc))
            self.tabs.select(self.diagnostics_tab)
            import traceback
            traceback.print_exc()
        finally:
            self.busy = False
            if not self.closed:
                self._access_changed()
                self.refresh_library()
                self.root.deiconify()
                self.root.lift()

    def _start_inline_session(self, *, observer=False):
        if self.busy or self.closed:
            return
        if not self.auth.allowed:
            self.status.set(self.auth.status())
            return
        runner = self.run_observer if observer else self.run_session
        if runner is None:
            self.status.set("Mini observer is unavailable in this build")
            return
        if self.replay:
            self.replay.close()
        self.busy = True
        self._observer_startup = observer
        if observer:
            import foundryoutput as fo
            fo.set_observation_only(True)
        self.status.set("Choose hardware")
        self._access_changed()
        self.tabs.select(self.session_tab)
        self.session_intro.pack_forget()
        from session_startup import SessionStartupUI
        factory = self._startup_factory or SessionStartupUI
        try:
            self._startup = factory(self.session_tab, service=self.service, access=self.auth,
                runtime_parent=self.runtime_host, observer=observer,
                on_ready=lambda startup: self._run_inline_session(runner, startup),
                on_cancel=self._cancel_inline_session, on_runtime=self._show_runtime_panel)
        except Exception as exc:
            self.diagnostic.set(str(exc))
            self._finish_inline_session("Hardware setup could not open - camera off")

    def _show_runtime_panel(self, panel):
        self._runtime_panel = panel
        self.home.pack_forget()
        self.login.pack_forget()
        self.runtime_host.pack(fill="both", expand=True)
        self.root.minsize(700, 520)
        if self.root.winfo_height() < 520:
            self.root.geometry(f"{max(700, self.root.winfo_width())}x520")
        self.status.set("Mini observer running" if self._observer_startup else "Live session running")
        self.root.update_idletasks()

    def _run_inline_session(self, runner, startup):
        if self.closed or startup is not self._startup or not self.auth.allowed:
            return
        from session_startup import StartupPending
        self.status.set("Starting camera")
        self.root.update_idletasks()
        message = None
        try:
            runner(self.service, startup=startup)
            message = "Session stopped - camera off" if self.auth.allowed else self.auth.status()
        except StartupPending:
            startup.resume_wait()
        except SystemExit:
            message = "Setup or session cancelled - camera off"
        except Exception as exc:
            self.diagnostic.set(str(exc))
            self.status.set("Session needs attention - camera off")
            if hasattr(self.auth, "usage_event"):
                self.auth.usage_event("live_errors")
            if not self._close_requested and self.auth.allowed:
                self.runtime_host.pack_forget()
                self._runtime_panel = None
                self._access_changed()
                startup.show_error(str(exc))
            else:
                message = self.auth.status()
            import traceback
            traceback.print_exc()
        finally:
            if message is not None:
                self._finish_inline_session(message)

    def _cancel_inline_session(self, startup):
        if startup is self._startup:
            self._finish_inline_session("Setup cancelled - camera off")

    def _finish_inline_session(self, message):
        startup, self._startup = self._startup, None
        if startup is not None:
            startup.dispose()
        self.runtime_host.pack_forget()
        self._runtime_panel = None
        self.root.minsize(700, 460)
        if self._observer_startup:
            import foundryoutput as fo
            fo.set_observation_only(False)
        self._observer_startup = False
        self.busy = False
        self.status.set(message)
        if self._close_requested:
            self.close()
        elif not self.closed:
            self.session_intro.pack(anchor="w", padx=8, pady=12)
            self._access_changed()
            self.refresh_library()

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
        if getattr(self, "inline_startup", False) and self.busy:
            self._close_requested = True
            if self._runtime_panel is not None:
                self._runtime_panel._on_close()
            elif self._startup is not None:
                self._startup.cancel()
            return
        if self.replay:
            self.replay.close()
        self.auth.close()
        self.service.stop()
        self.vtt_controls.close()
        self.closed = True
        if self._poll_id is not None:
            self.root.after_cancel(self._poll_id)
        self.root.destroy()

    def _access_changed(self):
        # AuthPanel can notify before the remainder of the window exists.
        if not hasattr(self, "auth") or not hasattr(self, "vtt_controls"):
            return
        allowed = self.auth.allowed
        self.vtt_controls.set_enabled(allowed)
        self.hardware_profiles.set_enabled(allowed)
        if allowed and not self.home.winfo_manager():
            self.login.pack_forget()
            self.home.pack(fill="both", expand=True)
            self.demo_button.focus_set()
        elif not allowed and not self.login.winfo_manager():
            self.home.pack_forget()
            self.login.pack(fill="both", expand=True)
            self.auth.email_entry.focus_set()
        buttons = [self.start_button, self.recording_button, self.demo_button]
        if self.observer_button is not None:
            buttons.append(self.observer_button)
        for button in buttons:
            button.state(["!disabled"] if allowed and not self.busy else ["disabled"])
        self.connect_button.state(["!disabled"] if allowed or self.service.running else ["disabled"])
        if not allowed:
            if getattr(self, "inline_startup", False) and self.busy:
                if self._runtime_panel is not None:
                    self._runtime_panel._on_close()
                    self.runtime_host.pack_forget()
                elif self._startup is not None:
                    self._startup.cancel()
            if self.replay:
                self.replay.close()
                self.replay = None
            if self.service.running:
                self.service.stop()
