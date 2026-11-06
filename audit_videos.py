"""A small, standalone review desk. Opens saved files, never a camera."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app_paths import atomic_write_json, data_path
from audit_catalog import AuditRecord, discover_videos, review_status
from audit_video import AuditVideo
from tracking_evaluation import format_time


BACKGROUND = "#141b22"
SURFACE = "#1c2630"
TEXT = "#e7edf4"
MUTED = "#a4b2bf"
ACCENT = "#87d4bd"
STATE_FILE = "video_audit_state.json"


def load_state():
    import json
    try:
        value = json.loads(data_path(STATE_FILE).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def short_time(seconds):
    return format_time(max(0, float(seconds)))


def recording_label(video):
    try:
        stamp = datetime.strptime(video.stem.removeprefix("sarween_rec_"), "%Y%m%d_%H%M%S")
        return f"{stamp:%b} {stamp.day} · {stamp:%H:%M:%S}"
    except ValueError:
        return video.stem


class AuditWindow:
    def __init__(self, root, *, folders=None, initial_video=None):
        self.root = root
        self.state = load_state()
        saved_folders = self.state.get("folders", [])
        saved_videos = self.state.get("videos", [])
        self.extra_folders = list(folders or (saved_folders if isinstance(saved_folders, list) else []))
        self.extra_videos = list(saved_videos if isinstance(saved_videos, list) else [])
        self.extra_folders = [str(p) for p in self.extra_folders if isinstance(p, (str, Path))]
        self.extra_videos = [str(p) for p in self.extra_videos if isinstance(p, (str, Path))]
        if initial_video:
            self.extra_videos.append(str(Path(initial_video).expanduser().resolve()))
        self.paths = []
        self.visible = []
        self.record = None
        self.reader = None
        self.playing = False
        self.closed = False
        self.photo = None
        self._pixels = None
        self._row_map = {}
        self._undo = []
        self._selecting = False
        self._scrubbing = False
        self._segment_end = None
        self._last_saved = 0
        self._clock_started = 0
        self._clock_origin = 0
        self._tick_id = None
        self._save_after = None
        self.root.title("Sarween Video Review")
        width = min(1320, max(900, root.winfo_screenwidth() - 70))
        height = min(820, max(580, root.winfo_screenheight() - 90))
        root.geometry(f"{width}x{height}")
        root.minsize(min(950, width), min(590, height))
        root.configure(background=BACKGROUND)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.tk.call("tk", "appname", "Sarween Video Review")
        if sys.platform == "darwin":
            root.createcommand("::tk::mac::Quit", self.close)
        self._style()
        self._build()
        self._keys()
        self.refresh_queue()
        preferred = (str(Path(initial_video).expanduser().resolve()) if initial_video
                     else str(self.state.get("last_video", "")))
        selected = next((p for p in self.visible if str(p) == preferred), None)
        self.open_video(selected or (self.visible[0] if self.visible else None))
        self._tick_id = root.after(40, self.tick)

    def _style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Helvetica", 12), background=BACKGROUND,
                        foreground=TEXT, fieldbackground=SURFACE)
        style.configure("TFrame", background=BACKGROUND)
        style.configure("TLabel", background=BACKGROUND, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Helvetica", 23, "bold"))
        style.configure("Heading.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("TButton", padding=(10, 7), background=SURFACE, foreground=TEXT)
        style.map("TButton", background=[("active", "#344352")],
                  foreground=[("disabled", "#687684")])
        style.configure("Accent.TButton", background="#275c50", foreground="#effff9")
        style.map("Accent.TButton", background=[("active", "#357b69")])
        style.configure("Treeview", background=SURFACE, fieldbackground=SURFACE,
                        foreground=TEXT, rowheight=32, borderwidth=0)
        style.configure("Treeview.Heading", background=BACKGROUND, foreground=MUTED,
                        relief="flat", font=("Helvetica", 11))
        style.map("Treeview", background=[("selected", "#344c5c")],
                  foreground=[("selected", "#ffffff")])
        style.configure("TCombobox", fieldbackground=SURFACE, foreground=TEXT)
        style.configure("TCheckbutton", background=BACKGROUND, foreground=MUTED)

    def _build(self):
        shell = ttk.Frame(self.root, padding=18)
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)
        header = ttk.Frame(shell)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(header, text="Video review", style="Title.TLabel").pack(side="left")
        self.progress = tk.StringVar(value="")
        ttk.Label(header, textvariable=self.progress, style="Muted.TLabel").pack(side="right")
        body = ttk.Frame(shell)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(body, width=225)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        sidebar.columnconfigure(0, weight=1)
        sidebar.rowconfigure(2, weight=1)
        ttk.Label(sidebar, text="Your recordings", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        saved_filter = self.state.get("filter", "To review")
        self.filter = tk.StringVar(value=saved_filter if saved_filter in
                                  ("To review", "Reviewed", "All recordings") else "To review")
        box = ttk.Combobox(sidebar, textvariable=self.filter,
                           values=("To review", "Reviewed", "All recordings"),
                           state="readonly", width=22)
        box.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        box.bind("<<ComboboxSelected>>", self.change_filter)
        listing = ttk.Frame(sidebar)
        listing.grid(row=2, column=0, sticky="nsew")
        listing.columnconfigure(0, weight=1)
        listing.rowconfigure(0, weight=1)
        self.queue = ttk.Treeview(listing, show="tree", selectmode="browse")
        self.queue.column("#0", width=215, minwidth=170)
        self.queue.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(listing, orient="vertical", command=self.queue.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.queue.configure(yscrollcommand=scroll.set)
        self.queue.tag_configure("done", foreground=ACCENT)
        self.queue.tag_configure("unavailable", foreground="#788591")
        for gesture in ("<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>",
                        "<KeyRelease-Home>", "<KeyRelease-End>", "<KeyRelease-Prior>", "<KeyRelease-Next>"):
            self.queue.bind(gesture, self.pick_video)
        add = ttk.Frame(sidebar)
        add.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(add, text="+ Video", command=self.add_video).pack(side="left")
        ttk.Button(add, text="+ Folder", command=self.add_folder).pack(side="right")
        ttk.Label(sidebar, text="Pick one. Check its moves.\nYou can stop at any time.",
                  style="Muted.TLabel", justify="left").grid(
                      row=4, column=0, sticky="w", pady=(12, 0))
        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=3, minsize=340)
        main.columnconfigure(1, weight=2, minsize=310)
        main.rowconfigure(1, weight=1)
        self.title = tk.StringVar(value="Choose a recording")
        ttk.Label(main, textvariable=self.title, style="Heading.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))
        player = ttk.Frame(main)
        player.grid(row=1, column=0, sticky="nsew", padx=(0, 16))
        player.columnconfigure(0, weight=1)
        player.rowconfigure(1, weight=1)
        view_bar = ttk.Frame(player)
        view_bar.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        self.board = tk.BooleanVar(value=True)
        self.grid_labels = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_bar, text="Board view", variable=self.board,
                        command=self.render).pack(side="left")
        ttk.Checkbutton(view_bar, text="Grid labels", variable=self.grid_labels,
                        command=self.render).pack(side="left", padx=(8, 0))
        self.zoom = tk.StringVar(value="Fit")
        zoom = ttk.Combobox(view_bar, textvariable=self.zoom, values=("Fit", "2×", "3×"),
                            width=4, state="readonly")
        zoom.pack(side="right")
        zoom.bind("<<ComboboxSelected>>", self.change_zoom)
        self.canvas = tk.Canvas(player, background="#080d12", highlightthickness=0,
                                width=600, height=370)
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", lambda _e: self.render())
        self.canvas.bind("<ButtonPress-1>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B1-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.view_hint = tk.StringVar(value="")
        ttk.Label(player, textvariable=self.view_hint, style="Muted.TLabel",
                  wraplength=480).grid(row=2, column=0, sticky="w", pady=(6, 2))
        seek_bar = ttk.Frame(player)
        seek_bar.grid(row=3, column=0, sticky="ew", pady=(6, 2))
        seek_bar.columnconfigure(0, weight=1)
        self.seek_value = tk.DoubleVar(value=0)
        self.seek_scale = ttk.Scale(seek_bar, from_=0, to=1, variable=self.seek_value,
                                    command=self.scrub)
        self.seek_scale.grid(row=0, column=0, sticky="ew")
        self.clock = tk.StringVar(value="00:00 / 00:00")
        ttk.Label(seek_bar, textvariable=self.clock, style="Muted.TLabel").grid(
            row=0, column=1, padx=(8, 0))
        transport = ttk.Frame(player)
        transport.grid(row=4, column=0, sticky="ew", pady=(4, 8))
        self.play_button = ttk.Button(transport, text="Play", command=self.toggle_play)
        self.play_button.pack(side="left")
        ttk.Button(transport, text="−1 sec", command=lambda: self.step(-1)).pack(side="left", padx=5)
        ttk.Button(transport, text="+1 sec", command=lambda: self.step(1)).pack(side="left")
        self.speed = tk.StringVar(value="1×")
        speed = ttk.Combobox(transport, textvariable=self.speed,
                             values=("0.25×", "0.5×", "1×", "2×"), width=6, state="readonly")
        speed.pack(side="right")
        speed.bind("<<ComboboxSelected>>", lambda _e: self._reset_clock())
        self.notes = tk.StringVar(value="")
        ttk.Label(player, text="Notes (optional)", style="Muted.TLabel").grid(
            row=5, column=0, sticky="w", pady=(12, 4))
        entry = ttk.Entry(player, textvariable=self.notes)
        entry.grid(row=6, column=0, sticky="ew")
        entry.bind("<KeyRelease>", self.notes_changed)
        ttk.Label(player, text="Space: play/pause   ← →: seek   C: correct   X: wrong",
                  style="Muted.TLabel", wraplength=470).grid(
                      row=7, column=0, sticky="w", pady=(10, 0))
        right = ttk.Frame(main)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        ttk.Label(right, text="What moved, and when", style="Heading.TLabel").grid(
            row=0, column=0, sticky="w")
        self.source = tk.StringVar(value="")
        self.source_note = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.source, style="Muted.TLabel",
                  wraplength=350).grid(row=1, column=0, sticky="ew", pady=(5, 2))
        ttk.Label(right, textvariable=self.source_note, style="Muted.TLabel",
                  wraplength=350).grid(row=2, column=0, sticky="ew", pady=(0, 10))
        transcript = ttk.Frame(right)
        transcript.grid(row=3, column=0, sticky="nsew")
        transcript.columnconfigure(0, weight=1)
        transcript.rowconfigure(0, weight=1)
        self.tree = ttk.Treeview(transcript, columns=("time", "move", "check"),
                                show="headings", selectmode="browse")
        for name, label, width in (("time", "Time", 68), ("move", "Movement", 200),
                                   ("check", "Check", 86)):
            self.tree.heading(name, text=label)
            self.tree.column(name, width=width, minwidth=55, stretch=name == "move")
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(transcript, orient="vertical", command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.tag_configure("Confirmed", foreground=ACCENT)
        self.tree.tag_configure("Wrong", foreground="#f1a1a1")
        for gesture in ("<ButtonRelease-1>", "<KeyRelease-Up>", "<KeyRelease-Down>",
                        "<KeyRelease-Home>", "<KeyRelease-End>", "<KeyRelease-Prior>", "<KeyRelease-Next>"):
            self.tree.bind(gesture, self.select_move)
        self.tree.bind("<Button-2>", self.context_menu)
        self.tree.bind("<Button-3>", self.context_menu)
        self.move_hint = tk.StringVar(value="Click a movement to jump just before it.")
        ttk.Label(right, textvariable=self.move_hint, style="Muted.TLabel",
                  wraplength=350).grid(row=4, column=0, sticky="ew", pady=(8, 6))
        check_bar = ttk.Frame(right)
        check_bar.grid(row=5, column=0, sticky="ew")
        self.correct_button = ttk.Button(check_bar, text="✓ Correct", style="Accent.TButton",
                                          command=lambda: self.decide(True))
        self.correct_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.wrong_button = ttk.Button(check_bar, text="✕ Wrong", command=lambda: self.decide(False))
        self.wrong_button.pack(side="left", fill="x", expand=True)
        actions = ttk.Frame(right)
        actions.grid(row=6, column=0, sticky="ew", pady=(7, 0))
        ttk.Button(actions, text="+ Missed move", command=self.add_missed).pack(side="left")
        ttk.Button(actions, text="Watch move", command=self.watch_move).pack(side="left", padx=6)
        self.undo_button = ttk.Button(actions, text="Undo", command=self.undo)
        self.undo_button.pack(side="right")
        ttk.Separator(right).grid(row=7, column=0, sticky="ew", pady=14)
        self.done_button = ttk.Button(right, text="Done with this video →",
                                      style="Accent.TButton", command=self.finish_video)
        self.done_button.grid(row=8, column=0, sticky="ew")
        ttk.Button(right, text="Export my review…", command=self.export).grid(
            row=9, column=0, sticky="ew", pady=(7, 0))
        self.status = tk.StringVar(value="Your progress saves automatically.")
        ttk.Label(shell, textvariable=self.status, style="Muted.TLabel").grid(
            row=2, column=0, sticky="ew", pady=(14, 0))

    def _keys(self):
        for key, action in (("<space>", self.toggle_play),
                            ("<Left>", lambda: self.step(-1)),
                            ("<Right>", lambda: self.step(1)),
                            ("<Shift-Left>", lambda: self.step_frame(-1)),
                            ("<Shift-Right>", lambda: self.step_frame(1)),
                            ("c", lambda: self.decide(True)),
                            ("x", lambda: self.decide(False)),
                            ("m", self.add_missed)):
            self.root.bind(key, lambda e, fn=action: self.shortcut(e, fn))

    def shortcut(self, event, action):
        if event.widget.winfo_class() in ("Entry", "TEntry", "Text", "TCombobox", "Spinbox"):
            return None
        if getattr(event, "state", 0) & 0xC:
            return None
        action()
        return "break"

    def save_state(self):
        self.state.update(folders=self.extra_folders, videos=self.extra_videos,
                          filter=self.filter.get())
        if self.record:
            self.state["last_video"] = str(self.record.video)
        atomic_write_json(data_path(STATE_FILE), self.state)

    def remember_selection(self):
        if not self.record:
            return
        if not isinstance(self.state.get("selected_rows"), dict):
            self.state["selected_rows"] = {}
        row = self.selected_row()
        if row:
            self.state["selected_rows"][str(self.record.video)] = row["id"]

    def refresh_queue(self):
        self.paths = [*discover_videos(), *discover_videos(self.extra_folders)]
        self.paths = list(dict.fromkeys([*self.paths, *[Path(v).expanduser().resolve()
                               for v in self.extra_videos if Path(v).is_file()]]))
        self.paths.sort(key=lambda path: (path.name.lower(), str(path)), reverse=True)
        statuses = {p: review_status(p) for p in self.paths}
        done = sum(value == "Reviewed" for value in statuses.values())
        unavailable = sum(value == "Unavailable" for value in statuses.values())
        self.progress.set(f"{done} of {len(self.paths) - unavailable} reviewed"
                          + (f" · {unavailable} unavailable" if unavailable else ""))
        mode = self.filter.get()
        self.visible = [p for p in self.paths if mode == "All recordings"
                        or (mode == "Reviewed" and statuses[p] == "Reviewed")
                        or (mode == "To review" and statuses[p] == "To review")]
        self._selecting = True
        self.queue.delete(*self.queue.get_children())
        for index, video in enumerate(self.visible):
            state = statuses[video]
            prefix = "✓ " if state == "Reviewed" else ("! " if state == "Unavailable" else "")
            self.queue.insert("", "end", iid=f"video-{index}", text=prefix + recording_label(video),
                              tags=("done" if state == "Reviewed" else
                                    "unavailable" if state == "Unavailable" else "pending",))
            if self.record and video == self.record.video:
                self.queue.selection_set(f"video-{index}")
        self._selecting = False

    def change_filter(self, _event=None):
        self.refresh_queue()
        if not self.record or self.record.video not in self.visible:
            self.open_video(self.visible[0] if self.visible else None)

    def pick_video(self, _event=None):
        if self._selecting or not self.queue.selection():
            return
        index = int(self.queue.selection()[0].split("-")[-1])
        self.open_video(self.visible[index])

    def open_video(self, video):
        if video is not None and self.record and self.reader and self.record.video == Path(video).resolve():
            try:
                stat = Path(video).stat()
                if (stat.st_size == self.record.review.identity["bytes"]
                        and stat.st_mtime_ns == self.record.review.identity["mtime_ns"]):
                    return
            except OSError:
                pass
        self.pause()
        if not self.flush_notes() or not self.save_progress():
            return
        if self.reader:
            self.reader.close()
        self.reader = self.record = None
        self._undo.clear()
        self._pixels = None
        self.photo = None
        self.canvas.itemconfigure(self._image_item, image="")
        self.title.set(Path(video).name if video else "No recordings in this list")
        self.source.set("")
        self.source_note.set("")
        self.notes.set("")
        self.view_hint.set("")
        self.clock.set("00:00 / 00:00")
        self.refresh_rows()
        if video is None:
            self.status.set("Your work is saved. Choose another list or add a video.")
            return
        try:
            self.record = AuditRecord(video)
            self.title.set(self.record.title)
            self.source.set(self.record.source_label)
            self.source_note.set(self.record.source_note)
            self.notes.set(self.record.notes)
            self.refresh_rows()
            selections = self.state.get("selected_rows", {})
            saved_row = selections.get(str(self.record.video)) if isinstance(selections, dict) else None
            for iid, row in self._row_map.items():
                if row["id"] == saved_row:
                    self.tree.selection_set(iid)
                    self.tree.see(iid)
            self.selection_actions()
            if self.record.unavailable_reason:
                self.view_hint.set(self.record.unavailable_reason)
                self.status.set("This recording is unavailable; choose another video.")
                self.done_button.state(["disabled"])
                return
            self.reader = AuditVideo(video, timeline=self.record.review.timeline)
            self.seek_scale.configure(to=max(1, self.reader.count - 1))
            self.zoom.set("Fit")
            if not self.seek(round(self.record.position_seconds * self.reader.fps)):
                return
            self.save_state()
            self.refresh_queue()
            self.status.set("Pick a move and check it. Your place saves automatically.")
        except Exception as exc:
            self.view_hint.set("Could not open this recording.")
            self.status.set(str(exc))
            if self.reader:
                self.reader.close()
                self.reader = None

    def refresh_rows(self, select=None):
        self._selecting = True
        self.tree.delete(*self.tree.get_children())
        self._row_map = {}
        rows = self.record.rows if self.record else []
        for index, row in enumerate(rows):
            iid = f"move-{index}"
            self._row_map[iid] = row
            movement = f"{row['mini']}   {row.get('from_cell') or '…'} → {row['to_cell']}"
            if row.get("kind") == "manual":
                movement = "+ " + movement
            self.tree.insert("", "end", iid=iid,
                             values=(short_time(row["time_seconds"]), movement, row["status"]),
                             tags=(row["status"],))
            if row["id"] == select:
                self.tree.selection_set(iid)
        if not self.tree.selection() and rows:
            first = next((iid for iid, row in self._row_map.items()
                          if row["status"] == "Unreviewed"), next(iter(self._row_map)))
            self.tree.selection_set(first)
        self._selecting = False
        pending = sum(row["status"] == "Unreviewed" for row in rows)
        if not rows:
            self.move_hint.set("No saved movement list. You can add the moves you see."
                               if self.record and self.record.source_label == "No saved transcript"
                               else "No moves in this list. Watch for any missed movements.")
        else:
            self.move_hint.set(f"{len(rows) - pending} of {len(rows)} moves checked. "
                               "Watch the whole video to catch missed moves.")
        self.done_button.state(["disabled"] if pending or not self.record else ["!disabled"])
        self.undo_button.state(["!disabled"] if self._undo else ["disabled"])
        self.correct_button.state(["!disabled"] if rows else ["disabled"])
        self.wrong_button.state(["!disabled"] if rows else ["disabled"])
        self.selection_actions()

    def selection_actions(self):
        row = self.selected_row()
        available = row is not None and row.get("kind") != "manual"
        self.correct_button.state(["!disabled"] if available else ["disabled"])
        self.wrong_button.state(["!disabled"] if available else ["disabled"])

    def selected_row(self):
        ids = self.tree.selection()
        return self._row_map.get(ids[0]) if ids else None

    def select_move(self, _event=None):
        if self._selecting:
            return
        self.selection_actions()
        row = self.selected_row()
        if row and self.reader:
            self.seek(round(max(0, row["time_seconds"] - 2) * self.reader.fps))
            self.focus_cell(row["to_cell"])

    def watch_move(self):
        row = self.selected_row()
        if not row or not self.reader:
            return
        if not self.seek(round(max(0, row["time_seconds"] - 2) * self.reader.fps)):
            return
        self._segment_end = min(self.reader.count - 1,
                                round((row["time_seconds"] + 2) * self.reader.fps))
        self.playing = True
        self._reset_clock()
        self.play_button.configure(text="Pause")

    def seek(self, frame, *, pause=True):
        if not self.reader:
            return False
        if pause:
            self.pause()
        try:
            self.reader.seek(int(frame))
            self._scrubbing = True
            self.seek_value.set(self.reader.frame)
            self._scrubbing = False
            self.clock.set(f"{short_time(self.reader.frame / self.reader.fps)} / "
                           f"{short_time(self.reader.duration)}")
            if not self.render():
                self.pause()
                self.status.set("Couldn't display this video frame.")
                return False
            return True
        except Exception as exc:
            self._scrubbing = False
            self.pause()
            self.status.set(str(exc))
            return False

    def render(self):
        if not self.reader or self.closed:
            return False
        from PIL import Image, ImageTk
        try:
            pixels, hint = self.reader.display(board=self.board.get(), grid=self.grid_labels.get())
            self._pixels = pixels
            image = Image.fromarray(pixels[:, :, ::-1])
            width, height = max(2, self.canvas.winfo_width()), max(2, self.canvas.winfo_height())
            factor = min(width / image.width, height / image.height)
            multiplier = {"Fit": 1, "2×": 2, "3×": 3}.get(self.zoom.get(), 1)
            draw_width = max(1, round(image.width * factor * multiplier))
            draw_height = max(1, round(image.height * factor * multiplier))
            image = image.resize((draw_width, draw_height), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image, master=self.root)
            self._draw_offset = (max(0, (width - draw_width) / 2),
                                 max(0, (height - draw_height) / 2))
            self._draw_scale = draw_width / pixels.shape[1]
            self.canvas.coords(self._image_item, *self._draw_offset)
            self.canvas.itemconfigure(self._image_item, image=self.photo)
            self.canvas.configure(scrollregion=(0, 0, max(width, draw_width), max(height, draw_height)))
            self.view_hint.set(hint + (" · Drag to pan" if multiplier > 1 else ""))
            return True
        except Exception as exc:
            self.canvas.itemconfigure(self._image_item, image="")
            self.photo = None
            self.view_hint.set(str(exc))
            return False

    def change_zoom(self, _event=None):
        self.render()
        row = self.selected_row()
        if row:
            self.focus_cell(row["to_cell"])

    def focus_cell(self, cell):
        if not self.reader or self.zoom.get() == "Fit":
            self.canvas.xview_moveto(0)
            self.canvas.yview_moveto(0)
            return
        try:
            point = self.reader.cell_center(cell)
            if point is None or not self.board.get():
                return
            x = self._draw_offset[0] + point[0] * self._draw_scale
            y = self._draw_offset[1] + point[1] * self._draw_scale
            region = [float(v) for v in self.canvas.cget("scrollregion").split()]
            self.canvas.xview_moveto(max(0, (x - self.canvas.winfo_width() / 2) / max(1, region[2])))
            self.canvas.yview_moveto(max(0, (y - self.canvas.winfo_height() / 2) / max(1, region[3])))
        except (AttributeError, ValueError):
            pass

    def scrub(self, value):
        if not self._scrubbing:
            self.seek(round(float(value)))

    def pause(self):
        self.playing = False
        self._segment_end = None
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play")

    def _reset_clock(self):
        self._clock_started = time.monotonic()
        self._clock_origin = self.reader.frame if self.reader else 0

    def toggle_play(self):
        if not self.reader:
            return
        if self.playing:
            self.pause()
            self.save_progress()
        else:
            if self.reader.frame >= self.reader.count - 1:
                if not self.seek(0):
                    return
            self.playing = True
            self._segment_end = None
            self._reset_clock()
            self.play_button.configure(text="Pause")

    def step(self, seconds):
        if self.reader:
            self.seek(round(self.reader.frame + seconds * self.reader.fps))

    def step_frame(self, frames):
        if self.reader:
            self.seek(self.reader.frame + frames)

    def tick(self):
        if self.closed:
            return
        if self.playing and self.reader:
            speed = float(self.speed.get().rstrip("×"))
            target = min(self.reader.count - 1,
                         self._clock_origin + int((time.monotonic() - self._clock_started)
                                                  * self.reader.fps * speed))
            limit = self._segment_end if self._segment_end is not None else self.reader.count - 1
            target = min(target, limit)
            if target != self.reader.frame:
                self.seek(target, pause=False)
            if target >= limit:
                self.pause()
                self.save_progress()
        if self.reader and time.monotonic() - self._last_saved >= 2:
            self.save_progress()
        self._tick_id = self.root.after(40, self.tick)

    def save_progress(self):
        if self.reader and self.record:
            self.remember_selection()
            self._last_saved = time.monotonic()
            try:
                self.record.set_position(self.reader.frame / self.reader.fps)
            except (OSError, ValueError) as exc:
                self.status.set(f"Couldn't save your place: {exc}")
                return False
        return True

    def notes_changed(self, _event=None):
        if self._save_after:
            self.root.after_cancel(self._save_after)
        self._save_after = self.root.after(600, self.flush_notes)

    def flush_notes(self):
        if self._save_after:
            self.root.after_cancel(self._save_after)
            self._save_after = None
        if self.record and self.notes.get() != self.record.notes:
            try:
                self.record.save_notes(self.notes.get())
            except (OSError, ValueError) as exc:
                self.status.set(f"Couldn't save your notes: {exc}")
                return False
        return True

    def grid(self):
        if not self.record:
            return None
        timeline = self.record.review.timeline
        cols, rows = timeline.get("gridCols"), timeline.get("gridRows")
        try:
            if any(isinstance(v, bool) for v in (cols, rows)):
                return None
            result = (int(cols), int(rows))
            if any(float(raw) != value or not 0 < value <= 10000
                   for raw, value in zip((cols, rows), result)):
                return None
            return result
        except (ValueError, TypeError, OverflowError):
            return None

    def mutate(self, action):
        if not self.record:
            return False
        before = copy.deepcopy(self.record.review.data)
        try:
            action()
        except (OSError, ValueError) as exc:
            self.record.review.data = before
            messagebox.showerror("Review not saved", str(exc), parent=self.root)
            return False
        self._undo.append(before)
        self._undo = self._undo[-20:]
        self.status.set("Saved automatically.")
        self.refresh_rows()
        self.refresh_queue()
        return True

    def decide(self, accepted):
        row = self.selected_row()
        if not row or not self.reader or row.get("kind") == "manual":
            return
        self.mutate(lambda: self.record.decide(row["id"], accepted,
                    duration=self.reader.duration, grid=self.grid()))

    def undo(self):
        if not self.record or not self._undo:
            return
        previous = self.record.review.data
        restored = copy.deepcopy(self._undo[-1])
        for key in ("position_seconds", "notes"):
            if key in previous:
                restored[key] = copy.deepcopy(previous[key])
        try:
            self.record.restore(restored)
            self._undo.pop()
            self.refresh_rows()
            self.refresh_queue()
            self.status.set("Last change undone.")
        except (OSError, ValueError) as exc:
            self.status.set(f"Couldn't undo: {exc}")

    def add_missed(self):
        if not self.reader or not self.record:
            return
        self.pause()
        window = tk.Toplevel(self.root)
        window.title("Add a missed move")
        window.transient(self.root)
        panel = ttk.Frame(window, padding=18)
        panel.pack(fill="both", expand=True)
        panel.columnconfigure(1, weight=1)
        selected = self.selected_row()
        profiles = self.record.review.timeline.get("profiles", {})
        names = sorted({row["mini"] for row in self.record.rows}
                       | (set(profiles) if isinstance(profiles, dict) else set()))
        mini = tk.StringVar(value=(selected or {}).get("mini") or (names[0] if names else "red10"))
        when = tk.StringVar(value=short_time(self.reader.frame / self.reader.fps))
        destination, origin, note = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for index, (label, variable) in enumerate((("Mini", mini), ("Time", when),
                ("Moved to cell", destination), ("From cell (optional)", origin),
                ("Note (optional)", note))):
            ttk.Label(panel, text=label).grid(row=index, column=0, sticky="w", padx=(0, 12), pady=5)
            widget = (ttk.Combobox(panel, textvariable=variable, values=names, width=24)
                      if index == 0 else ttk.Entry(panel, textvariable=variable, width=26))
            widget.grid(row=index, column=1, sticky="ew", pady=5)
            if index == 2:
                destination_entry = widget
        error = tk.StringVar(value="")
        ttk.Label(panel, textvariable=error, wraplength=350).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))

        def save():
            from tracking_evaluation import parse_time_seconds
            from movement_transcript import validate_label
            try:
                seconds = parse_time_seconds(when.get())
                validate_label({"mini": mini.get(), "time_seconds": seconds,
                                "to_cell": destination.get(), "from_cell": origin.get() or None},
                               duration=self.reader.duration, grid=self.grid())
                action = lambda: self.record.add_missed(seconds, mini.get(), destination.get(),
                    from_cell=origin.get() or None, duration=self.reader.duration,
                    grid=self.grid(), notes=note.get())
                if self.mutate(action):
                    window.destroy()
            except ValueError as exc:
                error.set(str(exc))
        buttons = ttk.Frame(panel)
        buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save missed move", style="Accent.TButton",
                   command=save).pack(side="left")
        window.bind("<Return>", lambda _e: save())
        window.bind("<Escape>", lambda _e: window.destroy())
        window.grab_set()
        destination_entry.focus_set()

    def context_menu(self, event):
        iid = self.tree.identify_row(event.y)
        if not iid or not self.record:
            return
        self.tree.selection_set(iid)
        row = self._row_map[iid]
        menu = tk.Menu(self.root, tearoff=False)
        if row.get("kind") == "manual":
            menu.add_command(label="Remove missed move",
                             command=lambda: self.mutate(lambda: self.record.remove_label(row["id"])))
        else:
            menu.add_command(label="Reset this check",
                             command=lambda: self.mutate(lambda: self.record.reset_decision(row["id"])))
        menu.tk_popup(event.x_root, event.y_root)

    def finish_video(self):
        if not self.reader or not self.record:
            return
        if any(row["status"] == "Unreviewed" for row in self.record.rows):
            self.status.set("Check the remaining listed moves first.")
            return
        self.pause()
        if not self.flush_notes() or not self.save_progress():
            return
        if self.mutate(lambda: self.record.mark_complete(True)):
            self.status.set("Video reviewed. Your work is saved.")
            if self.filter.get() == "To review":
                self.open_video(self.visible[0] if self.visible else None)

    def export(self):
        if not self.record:
            return
        folder = filedialog.askdirectory(parent=self.root, title="Save movement review")
        if folder:
            try:
                self.record.export(folder)
                self.status.set("Review exported as readable text, CSV and JSON.")
            except (OSError, ValueError) as exc:
                messagebox.showerror("Export failed", str(exc), parent=self.root)

    def add_video(self):
        video = filedialog.askopenfilename(parent=self.root, title="Add a recording",
            filetypes=[("Video", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"), ("All files", "*")])
        if video:
            self.extra_videos.append(str(Path(video).resolve()))
            self.filter.set("To review")
            self.refresh_queue()
            self.open_video(Path(video))
            self.save_state()

    def add_folder(self):
        folder = filedialog.askdirectory(parent=self.root, title="Add a recordings folder")
        if folder:
            if folder not in self.extra_folders:
                self.extra_folders.append(folder)
            self.refresh_queue()
            self.save_state()
            self.status.set("Folder added. Your existing reviews are kept.")

    def close(self):
        self.pause()
        if not self.flush_notes() or not self.save_progress():
            return
        try:
            self.save_state()
        except OSError as exc:
            self.status.set(f"Your review is saved, but the app couldn't save its last-opened video: {exc}")
            return
        self.closed = True
        if self._tick_id:
            self.root.after_cancel(self._tick_id)
        if self.reader:
            self.reader.close()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--folder", action="append", type=Path, help="Additional recording folder")
    args = parser.parse_args(argv)
    root = tk.Tk()
    try:
        AuditWindow(root, folders=[str(p.expanduser().resolve()) for p in args.folder or []],
                    initial_video=args.video)
    except Exception as exc:
        messagebox.showerror("Video review couldn't start", str(exc), parent=root)
        root.destroy()
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
