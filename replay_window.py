"""Hardware-free video playback, tracking analysis and human review."""

import bisect
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from app_paths import data_path, atomic_write_json
from map_view import MapView
from movement_transcript import export_rows, review_case
from replay_jobs import ReplayJob
from replay_review import ReviewSession
from tracking_evaluation import format_time


def tool_button(parent, symbol, name, command):
    button = ttk.Button(parent, text=symbol, width=3, command=command)
    tip = [None]
    def show(_event):
        if tip[0] is None:
            tip[0] = tk.Toplevel(button)
            tip[0].overrideredirect(True)
            tip[0].geometry(f"+{button.winfo_rootx()}+{button.winfo_rooty()+button.winfo_height()+3}")
            ttk.Label(tip[0], text=name, padding=4).pack()
    def hide(_event):
        if tip[0] is not None:
            tip[0].destroy()
            tip[0] = None
    button.bind("<Enter>", show)
    button.bind("<Leave>", hide)
    button.bind("<Destroy>", hide)
    return button


class ReplayWindow:
    def __init__(self, parent, video, *, demo=False):
        import cv2
        self.cv2 = cv2
        self.review = ReviewSession(video)
        timeline = self.review.timeline
        self.cap = cv2.VideoCapture(str(video))
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        if not self.cap.isOpened() or self.count < 1 or not 0 < self.fps < 1000:
            self.cap.release()
            raise ValueError("This video could not be opened")
        self.duration = (self.count - 1) / self.fps
        self.frame = 0
        self.playing = False
        self.demo = demo
        self.demo_ready = False
        self.preparing = False
        self.analysis_visible = bool(self.review.data.get("analysis"))
        self._demo_start_id = None
        self.closed = False
        self.job = None
        self._tick_id = None
        self.image = None
        self.last_pixels = None
        self._seek_updating = False
        self.root = tk.Toplevel(parent)
        title = (timeline.get("title") or ("Synthetic test fixture" if timeline.get("synthetic")
                 else "Real tabletop example")) if demo else Path(video).name
        self.root.title("Sarween - " + title)
        self.root.geometry(f"1100x{min(640, parent.winfo_screenheight()-80)}")
        self.root.minsize(780, 460)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1, minsize=80)
        header = ttk.Frame(outer)
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text=title,
                  font=("Helvetica", 13, "bold")).pack(side="left")
        ttk.Label(header, text="Processed on this Mac").pack(side="right")
        settings = ttk.Frame(outer)
        settings.grid(row=1, column=0, sticky="ew", pady=8)
        timeline = self.review.timeline
        self.cols = tk.StringVar(value=str(timeline.get("gridCols", 23)))
        self.rows = tk.StringVar(value=str(timeline.get("gridRows", 16)))
        self.mode = tk.StringVar(value=timeline.get("markerMode", "legacy"))
        self.limit = tk.StringVar(value="180")
        for text, variable in (("Columns", self.cols), ("Rows", self.rows), ("Time limit (s)", self.limit)):
            ttk.Label(settings, text=text).pack(side="left", padx=(0, 4))
            ttk.Spinbox(settings, from_=1, to=900, width=5, textvariable=variable).pack(side="left", padx=(0, 10))
        ttk.Combobox(settings, textvariable=self.mode, values=("legacy", "viewport"), width=10, state="readonly").pack(side="left")
        self.analyze_button = ttk.Button(settings, text="Analyze", command=self.analyze)
        self.analyze_button.pack(side="right")
        self.cancel_button = ttk.Button(settings, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right", padx=4)
        self.cancel_button.state(["disabled"])
        profiles = Path(video).with_suffix(".profiles.json")
        saved_options = self.review.data.get("options", {})
        self.profiles = Path(saved_options.get("profiles_path") or (profiles if profiles.exists() else data_path("combo_profiles.json")))
        # Opening a reviewed recording is read-only. Only explicit reanalysis
        # may reset its run-specific decisions and completion flag.
        self.auto_analyze = demo or bool(not self.analysis_visible and timeline.get("events") and self.profiles.is_file())
        self.needs_setup = not (self.auto_analyze or self.analysis_visible)
        self.preparing = self.auto_analyze
        if self.auto_analyze:
            self.analysis_visible = False
        profile_bar = ttk.Frame(outer)
        profile_bar.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(profile_bar, text="Choose profiles", command=self.choose_profiles).pack(side="left")
        self.profile_label = ttk.Label(profile_bar, text=self.profiles.name)
        self.profile_label.pack(side="left", padx=8)
        self.restore_profiles = tk.BooleanVar(value=True)
        ttk.Checkbutton(profile_bar, text="Recorded profile changes", variable=self.restore_profiles).pack(side="right")
        self.views = ttk.Frame(outer)
        self.views.columnconfigure((0, 1), weight=1, uniform="video")
        self.views.rowconfigure(1, weight=1)
        ttk.Label(self.views, text="What the camera sees", font=("Helvetica", 13, "bold")).grid(
            row=0, column=0, sticky="w", pady=(12, 8))
        ttk.Label(self.views, text="What the software sees", font=("Helvetica", 13, "bold")).grid(
            row=0, column=1, sticky="w", padx=(8, 0), pady=(12, 8))
        video_view = ttk.Frame(self.views)
        video_view.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.video_canvas = tk.Canvas(video_view, background="#171d20", highlightthickness=0,
                                      width=500, height=260)
        caption = ttk.Label(video_view, text=timeline.get("description") or
                            ("Synthetic test footage" if timeline.get("synthetic") else "Your video stays on this Mac"), wraplength=460)
        caption.pack(side="bottom", fill="x", pady=4)
        caption.bind("<Configure>", lambda e: caption.configure(wraplength=max(100, e.width)))
        self.video_canvas.pack(fill="both", expand=True)
        self.map = MapView(self.views)
        self.map.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        self.views.grid(row=3, column=0, sticky="nsew")
        self.video_canvas.bind("<Configure>", lambda _event: self.render())
        transport = ttk.Frame(outer)
        transport.grid(row=4, column=0, sticky="ew", pady=5)
        tool_button(transport, "\u23ee", "Previous frame", lambda: self.seek(self.frame - 1)).pack(side="left")
        self.play_button = tool_button(transport, "\u25b6", "Play / Pause", self.toggle_play)
        self.play_button.pack(side="left")
        tool_button(transport, "\u23ed", "Next frame", lambda: self.seek(self.frame + 1)).pack(side="left")
        self.seek_value = tk.DoubleVar(value=0)
        ttk.Scale(transport, from_=0, to=max(1, self.count-1), variable=self.seek_value, command=self.scrub).pack(side="left", fill="x", expand=True, padx=8)
        self.clock = tk.StringVar(value="")
        ttk.Label(transport, textvariable=self.clock, width=23).pack(side="left")
        self.speed = tk.StringVar(value="1x")
        speed_box = ttk.Combobox(transport, textvariable=self.speed, values=("0.25x", "0.5x", "1x", "2x"), width=5, state="readonly")
        speed_box.pack(side="left")
        speed_box.bind("<<ComboboxSelected>>", self.change_speed)
        self.tools_button = ttk.Button(transport, text="Hide settings" if self.needs_setup else "Show settings",
                                       command=self.toggle_review_tools)
        self.tools_button.pack(side="left", padx=(8, 0))
        review_bar = ttk.Frame(outer)
        review_bar.grid(row=5, column=0, sticky="ew", pady=4)
        self.track = tk.StringVar(value="Detected")
        for name in ("Detected", "Reviewed", "Foundry"):
            ttk.Radiobutton(review_bar, text=name, value=name, variable=self.track, command=self.refresh_rows).pack(side="left")
        self.filter = tk.StringVar(value="All minis")
        self.filter_box = ttk.Combobox(review_bar, textvariable=self.filter, values=("All minis",), width=13, state="readonly")
        self.filter_box.pack(side="left", padx=8)
        self.filter_box.bind("<<ComboboxSelected>>", lambda _event: self.refresh_rows())
        self.complete = tk.BooleanVar(value=self.review.data["complete"])
        ttk.Checkbutton(review_bar, text="Review complete", variable=self.complete, command=self.complete_review).pack(side="right")
        table = ttk.Frame(outer)
        table.grid(row=6, column=0, sticky="ew")
        self.tree = ttk.Treeview(table, columns=("time", "mini", "from", "to", "status", "source"), show="headings", height=5, selectmode="browse")
        for key, label, width in (("time", "Time", 90), ("mini", "Mini", 110), ("from", "From", 70), ("to", "To", 70), ("status", "Review", 100), ("source", "Cause", 140)):
            self.tree.heading(key, text=label)
            self.tree.column(key, width=width, minwidth=50)
        self.tree.pack(side="left", fill="x", expand=True)
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        scroll.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.bind("<<TreeviewSelect>>", self.select_row)
        actions = ttk.Frame(outer)
        actions.grid(row=7, column=0, sticky="ew", pady=6)
        for symbol, name, action in (("+", "Add missed movement", self.add), ("\u270e", "Edit movement", self.edit),
                                     ("\u2713", "Confirm detection", lambda: self.decide(True)),
                                     ("\u00d7", "Reject detection / remove label", lambda: self.decide(False))):
            tool_button(actions, symbol, name, action).pack(side="left", padx=(0, 4))
        ttk.Button(actions, text="Export transcript", command=self.export).pack(side="left", padx=4)
        ttk.Button(actions, text="Export regression case", command=self.export_case).pack(side="left", padx=4)
        ttk.Button(actions, text="Score", command=self.score).pack(side="left")
        ttk.Button(actions, text="Diagnostics", command=self.diagnostics).pack(side="right")
        self.status = tk.StringVar(value="Saved tracking results. Press Play to compare both views." if self.analysis_visible else
                                  "Set the grid and marker mode, choose mini profiles, then click Analyze. "
                                  "Tracking needs visible ArUco markers and colored mini rings.")
        label = ttk.Label(outer, textvariable=self.status, wraplength=1000)
        label.grid(row=8, column=0, sticky="ew")
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
        self.review_tools = (settings, profile_bar, review_bar, table, actions)
        self.tools_visible = self.needs_setup
        if not self.needs_setup:
            for widget in self.review_tools:
                widget.grid_remove()
        if self.auto_analyze:
            self.play_button.state(["disabled"])
            self.status.set("Reading the video and tracking its minis…")
        elif self.needs_setup:
            self.root.minsize(780, 580)
        self.refresh_rows()
        self.seek(0)
        self._tick_id = self.root.after(30, self.tick)
        if self.auto_analyze:
            self._demo_start_id = self.root.after(100, self.analyze)

    def toggle_review_tools(self):
        self.tools_visible = not self.tools_visible
        for widget in self.review_tools:
            widget.grid() if self.tools_visible else widget.grid_remove()
        self.tools_button.configure(text="Hide settings" if self.tools_visible else "Show settings")
        minimum = 580 if self.tools_visible else 460
        self.root.minsize(780, minimum)
        if self.root.winfo_height() < minimum:
            self.root.geometry(f"{self.root.winfo_width()}x{minimum}")

    def change_speed(self, _event=None):
        self.play_started, self.play_origin = time.monotonic(), self.frame

    def choose_profiles(self):
        path = filedialog.askopenfilename(parent=self.root, title="Mini profiles", filetypes=[("JSON profiles", "*.json")])
        if path:
            self.profiles = Path(path)
            self.profile_label.configure(text=self.profiles.name)
            self.restore_profiles.set(False)

    def options(self):
        cols, rows = int(self.cols.get()), int(self.rows.get())
        if not 1 <= cols <= 500 or not 1 <= rows <= 500:
            raise ValueError("Grid dimensions must be between 1 and 500")
        data = self.review.timeline
        return {"profiles_path": str(self.profiles), "grid_w": cols, "grid_h": rows,
                "warp_w": data.get("warpWidth", 1280), "warp_h": data.get("warpHeight", 720),
                "marker_mode": self.mode.get(), "restore_recorded_profiles": self.restore_profiles.get()}

    def analyze(self):
        if self._demo_start_id is not None:
            self.root.after_cancel(self._demo_start_id)
            self._demo_start_id = None
        if self.closed or self.job and not self.job.closed:
            return
        try:
            self.run_options = self.options()
            self.job = ReplayJob(self.review.video, self.run_options, timeout=float(self.limit.get()))
            self.preparing = True
            self.demo_ready = self.analysis_visible = False
            self.play_button.state(["disabled"])
            self.seek(0)
            self.analyze_button.state(["disabled"])
            self.cancel_button.state(["!disabled"])
            self.status.set("Analyzing saved pixels - one worker")
        except Exception as exc:
            self.preparing = False
            self.play_button.state(["!disabled"])
            self.error(exc)

    def cancel(self):
        if self.job:
            self.job.cancel()

    def tick(self):
        if self.closed:
            return
        if self.job and not getattr(self.job, "delivered", False):
            if self.job.poll():
                self.job.delivered = True
                self.preparing = False
                self.play_button.state(["!disabled"])
                self.analyze_button.state(["!disabled"])
                self.cancel_button.state(["disabled"])
                if self.job.error:
                    self.status.set(self.job.error)
                else:
                    try:
                        self.review.accept_analysis(self.job.result, self.run_options)
                        d = self.job.result["diagnostics"]
                        self.complete.set(False)
                        self.refresh_rows()
                        self.analysis_visible = bool(d.get("locked_frames") and d.get("tracked_frames") and d.get("map_states"))
                        self.demo_ready = self.analysis_visible
                        self.render_map()
                        warning = "No marker lock; results are not valid." if not d["locked_frames"] else (
                            "No tracking frames; results are not valid." if not d["tracked_frames"] else "Analysis complete")
                        self.status.set(f"{warning} | {len(self.job.result['events'])} detections | "
                                        f"{d['processed_frames']} frames | {len(d['replay_limitations'])} fidelity notes")
                        if self.auto_analyze and self.analysis_visible:
                            self.seek(0)
                            self.status.set("Tracking calculated from this video. Move through the timeline to compare both views."
                                            if self.job.result["events"] else
                                            "No minis detected. Check the selected mini profiles and ring visibility.")
                            self.toggle_play()
                    except Exception as exc:
                        self.error(exc)
            else:
                p = self.job.progress
                self.status.set(f"Analyzing: {p.get('frame', 0)}/{p.get('total', self.count)} frames | "
                                f"{p.get('moves', 0)} detections | {int(time.monotonic()-self.job.started)}s")
        if self.playing:
            speed = float(self.speed.get().removesuffix("x"))
            now = time.monotonic()
            target = min(self.count - 1, int(self.play_origin + (now - self.play_started) * self.fps * speed))
            if target != self.frame:
                self.seek(target, pause=False)
            if target == self.count - 1:
                self.playing = False
                self.play_button.configure(text="\u25b6")
        self._tick_id = self.root.after(30, self.tick)

    def toggle_play(self):
        if self.closed or self.preparing:
            return
        self.playing = not self.playing
        if self.playing and self.frame == self.count - 1:
            self.seek(0, pause=False)
        self.play_started, self.play_origin = time.monotonic(), self.frame
        self.play_button.configure(text="\u23f8" if self.playing else "\u25b6")

    def scrub(self, value):
        if not self._seek_updating:
            self.seek(round(float(value)))

    def seek(self, frame, *, pause=True):
        if self.closed:
            return
        if pause:
            self.playing = False
            self.play_button.configure(text="\u25b6")
        frame = max(0, min(self.count - 1, int(frame)))
        if int(self.cap.get(self.cv2.CAP_PROP_POS_FRAMES)) != frame:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame)
        ok, pixels = self.cap.read()
        if not ok:
            self.status.set(f"Could not decode frame {frame}")
            return
        self.frame, self.last_pixels = frame, pixels
        self._seek_updating = True
        self.seek_value.set(frame)
        self._seek_updating = False
        self.clock.set(f"{format_time(frame/self.fps)} / {format_time(self.duration)}")
        self.render()
        self.render_map()

    def render(self):
        if self.last_pixels is None or self.closed:
            return
        from PIL import Image, ImageTk
        frame = Image.fromarray(self.cv2.cvtColor(self.last_pixels, self.cv2.COLOR_BGR2RGB))
        w, h = max(2, self.video_canvas.winfo_width()), max(2, self.video_canvas.winfo_height())
        frame.thumbnail((w, h))
        self.image = ImageTk.PhotoImage(frame, master=self.root)
        self.video_canvas.delete("all")
        self.video_canvas.create_image(w/2, h/2, image=self.image)

    def render_map(self):
        result = (self.review.data.get("analysis") or {}) if self.analysis_visible else {}
        states = result.get("diagnostics", {}).get("map_states", [])
        times = [item["time_seconds"] for item in states]
        index = bisect.bisect_right(times, self.frame/self.fps) - 1
        try:
            grid = [int(self.cols.get()), int(self.rows.get())]
        except ValueError:
            grid = [23, 16]
        state = states[index] if index >= 0 else {"grid": grid, "positions": {}}
        self.map.set_state(state)

    def refresh_rows(self):
        self.display_rows = {row["id"]: row for row in self.review.rows(self.track.get())}
        names = sorted({row["mini"] for track in ("Detected", "Reviewed", "Foundry") for row in self.review.rows(track)})
        self.filter_box.configure(values=["All minis"] + names)
        if self.filter.get() not in ["All minis"] + names:
            self.filter.set("All minis")
        self.tree.delete(*self.tree.get_children())
        for key, row in self.display_rows.items():
            if self.filter.get() not in ("All minis", row["mini"]):
                continue
            self.tree.insert("", "end", iid=key, values=(format_time(row["time_seconds"]), row["mini"],
                row.get("from_cell") or "?", row["to_cell"], row["decision"], row.get("source", "detection")))

    def selected(self):
        keys = self.tree.selection()
        return self.display_rows.get(keys[0]) if keys else None

    def select_row(self, _event=None):
        row = self.selected()
        if row:
            self.seek(round(row["time_seconds"] * self.fps))

    def add(self):
        self.edit_dialog({"time_seconds": self.frame/self.fps, "source": "detection"})

    def edit(self):
        row = self.selected()
        if row:
            self.edit_dialog(row, replace=row["id"] if self.track.get() == "Reviewed" else None)

    def edit_dialog(self, row, replace=None):
        dialog = tk.Toplevel(self.root)
        dialog.title("Movement label")
        dialog.transient(self.root)
        fields = {}
        for index, (key, title, default) in enumerate((("mini", "Mini", ""), ("time_seconds", "Video time (seconds or mm:ss)", ""),
                ("from_cell", "From cell (optional)", ""), ("to_cell", "To cell", ""), ("source", "Cause", "detection"))):
            ttk.Label(dialog, text=title).grid(row=index, column=0, sticky="w", padx=12, pady=6)
            fields[key] = tk.StringVar(value=str(row.get(key) or default))
            if key == "source":
                ttk.Combobox(dialog, textvariable=fields[key], values=("detection", "viewportTransform"), state="readonly").grid(row=index, column=1, padx=12)
            else:
                ttk.Entry(dialog, textvariable=fields[key]).grid(row=index, column=1, padx=12)
        def save():
            try:
                self.review.label({**{key: value.get() for key, value in fields.items()}, "provenance": "manual-video-review"},
                    duration=self.duration, grid=(int(self.cols.get()), int(self.rows.get())), replace=replace)
                self.complete.set(False)
                self.track.set("Reviewed")
                self.refresh_rows()
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("Movement label", str(exc), parent=dialog)
        ttk.Button(dialog, text="Save", command=save).grid(row=5, column=1, sticky="e", padx=12, pady=12)
        dialog.grab_set()

    def decide(self, accepted):
        row = self.selected()
        if not row:
            return
        try:
            if self.track.get() == "Detected":
                self.review.decide(row, accepted, duration=self.duration, grid=(int(self.cols.get()), int(self.rows.get())))
            elif self.track.get() == "Reviewed" and not accepted:
                self.review.remove(row["id"])
            else:
                return
            self.complete.set(False)
            self.refresh_rows()
        except Exception as exc:
            self.error(exc)

    def complete_review(self):
        self.review.data["complete"] = self.complete.get()
        self.review.save()

    def export(self):
        path = filedialog.asksaveasfilename(parent=self.root, title="Export transcript", initialfile=self.review.video.stem + ".movements.csv", defaultextension=".csv")
        if path:
            try:
                export_rows([row for track in ("Detected", "Reviewed", "Foundry") for row in self.review.rows(track)], Path(path).with_suffix(""))
                self.status.set("Exported CSV, text and JSON transcripts")
            except Exception as exc:
                self.error(exc)

    def export_case(self):
        try:
            options = self.options()
            sidecar = self.review.video.with_suffix(".tracking.json")
            if sidecar.exists():
                options["timeline_path"] = str(sidecar)
            case = review_case(self.review.video, self.review.data["labels"], duration=self.duration,
                               options=options, complete=self.complete.get())
            path = filedialog.asksaveasfilename(parent=self.root, title="Export regression case", initialfile=self.review.video.stem + ".case.json", defaultextension=".json")
            if path:
                atomic_write_json(path, case)
                self.status.set("Regression case exported")
        except Exception as exc:
            self.error(exc)

    def diagnostics(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Replay diagnostics")
        text = tk.Text(dialog, wrap="word", width=80, height=24)
        text.pack(fill="both", expand=True)
        result = self.review.data.get("analysis") or {}
        d = {key: value for key, value in result.get("diagnostics", {}).items() if key != "map_states"}
        text.insert("1.0", json.dumps(d, indent=2) + "\n\n" + (self.job.log if self.job else ""))
        text.configure(state="disabled")

    def score(self):
        try:
            from tracking_evaluation import MovementEvent, evaluate_events
            result = self.review.data.get("analysis")
            if not result or not result["diagnostics"].get("tracked_frames"):
                raise ValueError("Run a valid analysis before scoring")
            case = review_case(self.review.video, self.review.data["labels"], duration=self.duration,
                              options=self.options(), complete=self.complete.get())["cases"][0]
            scored = evaluate_events(case, [MovementEvent(**row) for row in result["events"]], video=str(self.review.video))
            self.status.set(f"{'PASS' if scored.ok else 'FAIL'} | {scored.matched_expectations}/{scored.total_expectations} matched | "
                            f"{len(scored.missing)} missed | {len(scored.unexpected)} unexpected | {scored.label_source}")
        except Exception as exc:
            self.error(exc)

    def error(self, exc):
        self.status.set(str(exc))
        messagebox.showerror("Recording", str(exc), parent=self.root)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.job:
            self.job.cancel()
        if self._tick_id is not None:
            self.root.after_cancel(self._tick_id)
        if self._demo_start_id is not None:
            self.root.after_cancel(self._demo_start_id)
        self.cap.release()
        self.root.destroy()
