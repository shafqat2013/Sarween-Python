"""Native observation UI; it owns no camera, relay, or token movement code.

The session supplies rectified board pixels and measurements on Tk's thread.
Starting-position clicks all use one copied frame and opaque context, so clicks
cannot silently refer to different camera frames or geometry revisions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


@dataclass(frozen=True)
class ImageFit:
    image_width: int
    image_height: int
    display_width: int
    display_height: int
    left: float
    top: float

    def image_point(self, x, y):
        """Return a source coordinate, rejecting clicks in the letterbox."""
        x, y = float(x) - self.left, float(y) - self.top
        if not (0 <= x < self.display_width and 0 <= y < self.display_height):
            return None
        return (x * self.image_width / self.display_width,
                y * self.image_height / self.display_height)

    def canvas_point(self, x, y):
        return (self.left + float(x) * self.display_width / self.image_width,
                self.top + float(y) * self.display_height / self.image_height)


def fit_image(image_width, image_height, canvas_width, canvas_height):
    sizes = tuple(map(int, (image_width, image_height, canvas_width, canvas_height)))
    if min(sizes) <= 0:
        raise ValueError("Image and preview dimensions must be positive")
    width, height, available_width, available_height = sizes
    scale = min(available_width / width, available_height / height)
    display_width = max(1, min(available_width, round(width * scale)))
    display_height = max(1, min(available_height, round(height * scale)))
    return ImageFit(width, height, display_width, display_height,
                    (available_width - display_width) / 2,
                    (available_height - display_height) / 2)


def _board_image(image):
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 2):
        raise ValueError("Observer preview needs a uint8 BGR board image")
    return image


class StartingPositions:
    """Frozen, first-frame initialization state independent of native widgets."""

    def __init__(self, count=6, minimum_separation=8.0):
        if isinstance(count, bool) or int(count) != count or not 1 <= count <= 12:
            raise ValueError("Choose between one and twelve minis")
        self.count = int(count)
        self.minimum_separation = float(minimum_separation)
        if not math.isfinite(self.minimum_separation) or self.minimum_separation <= 0:
            raise ValueError("Starting positions need a positive separation")
        self.clear()

    def clear(self):
        self.frame = None
        self.context = None
        self.points = []
        self.selecting = False

    def start(self, image, context=None):
        self.frame = _board_image(image).copy()
        self.context = deepcopy(context)
        self.points = []
        self.selecting = True

    def choose(self, point):
        if not self.selecting or len(self.points) >= self.count:
            return False
        x, y = map(float, point)
        height, width = self.frame.shape[:2]
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
            return False
        if any(math.hypot(x - px, y - py) < self.minimum_separation for px, py in self.points):
            return False
        self.points.append((x, y))
        return True

    def undo(self):
        if self.selecting and self.points:
            self.points.pop()
            return True
        return False

    @property
    def complete(self):
        return self.selecting and len(self.points) == self.count

    @property
    def positions(self):
        return {f"mini{index}": point for index, point in enumerate(self.points, 1)}


def movement_text(event):
    mini = str(event.get("mini", "Mini"))
    origin = event.get("from_cell") or "?"
    destination = event.get("to_cell")
    if not destination:
        cells = event.get("destination_cells") or []
        destination = "/".join(map(str, cells)) if cells else "uncertain"
        if cells:
            destination += " (uncertain)"
    return f"{mini}: {origin} → {destination}"


class ObserverWindow:
    """One Tk window, driven by the live session's existing event loop.

    ``on_initialize(positions, frozen_board, frozen_context)`` must validate the
    still-current camera geometry. Return ``False`` to refuse initialization.
    ``context`` is opaque and may contain arrays; this UI never serializes it.
    ``on_record`` toggles recording in the session and ``on_reset`` clears the
    observer. All callbacks execute synchronously on Tk's thread.
    """

    def __init__(self, parent, on_initialize, on_reset=None, on_record=None,
                 on_close=None, mini_count=6):
        self.parent = parent
        self.on_initialize = on_initialize
        self.on_reset = on_reset
        self.on_record = on_record
        self.on_close = on_close
        self.selection = StartingPositions(mini_count)
        self.closed = False
        self.initialized = False
        self._latest_board = None
        self._latest_context = None
        self._latest_result = {}
        self._fit = None
        self._photo = None
        self._canvas_size = (760, 410)
        self._recording = False
        self._record_path = ""
        self._event_keys = set()
        self._events = []

        self.window = tk.Toplevel(parent)
        self.window.title("Sarween — Mini observer")
        width = min(860, max(620, self.window.winfo_screenwidth() - 80))
        height = min(760, max(500, self.window.winfo_screenheight() - 120))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(min(width, 620), min(height, 500))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(3, weight=1)

        ttk.Label(outer, text="Mini observer", font=("Helvetica", 16, "bold")).grid(
            row=0, column=0, sticky="w")
        ttk.Label(outer, text="Observation only — Foundry token movement OFF",
                  foreground="#d59c25").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(outer, text="Experimental · fog behavior unverified").grid(
            row=2, column=0, sticky="w", pady=(2, 8))
        self.canvas = tk.Canvas(outer, width=760, height=410, background="#111111",
                                highlightthickness=0)
        self.canvas.grid(row=3, column=0, sticky="nsew")
        self._image_item = self.canvas.create_image(0, 0, anchor="nw")
        self.canvas.bind("<Configure>", self._resize)
        self.canvas.bind("<Button-1>", self._choose)

        buttons = ttk.Frame(outer)
        buttons.grid(row=4, column=0, sticky="ew", pady=(8, 4))
        self.choose_button = ttk.Button(buttons, text="Choose starting positions", command=self.start_selection,
                                        state="disabled")
        self.choose_button.pack(side="left")
        self.undo_button = ttk.Button(buttons, text="Undo last click", command=self.undo,
                                      state="disabled")
        self.undo_button.pack(side="left", padx=(6, 0))
        self.record_button = ttk.Button(buttons, text="Record observer inputs", command=self._record,
                                        state="disabled")
        self.record_button.pack(side="right")

        self.status = tk.StringVar(self.window, "Waiting for the camera and marker lock.")
        status = ttk.Label(outer, textvariable=self.status, justify="left", wraplength=720)
        status.grid(row=5, column=0, sticky="ew", pady=(0, 4))
        status.bind("<Configure>", lambda event: status.configure(wraplength=max(100, event.width)))
        self.record_status = tk.StringVar(self.window, "Not recording")
        label = ttk.Label(outer, textvariable=self.record_status, justify="left", wraplength=720)
        label.grid(row=6, column=0, sticky="ew", pady=(0, 5))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
        ttk.Label(outer, text="Green: last confirmed · Amber: held/uncertain · Gray: candidates").grid(
            row=7, column=0, sticky="w", pady=(0, 5))
        log_frame = ttk.Frame(outer)
        log_frame.grid(row=8, column=0, sticky="ew")
        self.transcript = tk.Text(log_frame, height=4, wrap="word", state="disabled")
        self.transcript.pack(side="left", fill="both", expand=True)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.transcript.yview)
        scrollbar.pack(side="right", fill="y")
        self.transcript.configure(yscrollcommand=scrollbar.set)

    @property
    def selecting(self):
        return self.selection.selecting

    def _resize(self, event):
        # The next supplied frame repaints even while selection is frozen.
        # No Tk update or geometry operation is nested in Cocoa resize events.
        self._canvas_size = (max(1, int(event.width)), max(1, int(event.height)))

    def set_status(self, message):
        if not self.closed:
            self.status.set(str(message))

    def update_frame(self, board_bgr, result=None, *, context=None, status=None):
        if self.closed:
            return
        self._latest_board = _board_image(board_bgr).copy()
        self._latest_context = deepcopy(context)
        self._latest_result = result or {}
        selectable = not isinstance(context, dict) or context.get("locked", True)
        self.choose_button.configure(state="normal" if selectable else "disabled")
        if status is not None and not self.selecting:
            self.set_status(status)
        elif not self.selecting:
            state = self._latest_result.get("state") or {}
            reason = state.get("reason")
            if reason:
                self.set_status(str(reason).replace("_", " "))
        self._remember_events(self._latest_result)
        self._paint()

    def start_selection(self):
        if self.closed or self._latest_board is None:
            return
        try:
            if self.on_reset:
                self.on_reset()
            self.selection.start(self._latest_board, self._latest_context)
        except Exception as exc:
            self.set_status(f"Could not choose positions: {exc}")
            return
        self.initialized = False
        self._clear_transcript()
        self.undo_button.configure(state="normal")
        self.record_button.configure(state="normal" if self._recording else "disabled")
        self.choose_button.configure(text="Start over")
        self._selection_status()
        self._paint()

    def _selection_status(self):
        done, count = len(self.selection.points), self.selection.count
        if done < count:
            self.set_status(f"Frozen view · click the base center of Mini {done + 1} ({done}/{count} chosen). Keep the minis still.")
        else:
            self.set_status("Starting positions chosen. Initializing observer…")

    def _choose(self, event):
        if not self.selecting or self._fit is None:
            return
        point = self._fit.image_point(event.x, event.y)
        if point is None:
            return
        if not self.selection.choose(point):
            self.set_status("Choose a different base center; that click is too close to an existing position.")
            return
        self._selection_status()
        self._paint()
        if self.selection.complete:
            self._initialize()

    def _initialize(self):
        try:
            accepted = self.on_initialize(self.selection.positions, self.selection.frame.copy(),
                                           deepcopy(self.selection.context))
        except Exception as exc:
            self.set_status(f"Could not start observer: {exc}. Undo a click or start over.")
            return
        if accepted is False:
            self.set_status("Starting view is no longer valid. Start over with the current camera view.")
            return
        self.selection.selecting = False
        self.initialized = True
        self.undo_button.configure(state="disabled")
        self.record_button.configure(state="normal" if self.on_record else "disabled")
        self.choose_button.configure(text="Choose positions again")
        self.set_status("Starting positions saved. Checking that every base is visible; keep the minis still.")

    def undo(self):
        if self.selection.undo():
            self._selection_status()
            self._paint()

    def invalidate_initialization(self, message="Camera geometry changed. Choose starting positions again."):
        """Session-owned lock/geometry invalidation without callback recursion."""
        self.selection.clear()
        self.initialized = False
        self.undo_button.configure(state="disabled")
        self.record_button.configure(state="normal" if self._recording else "disabled")
        self.choose_button.configure(text="Choose starting positions")
        self.set_status(message)

    def _record(self):
        if self.on_record:
            try:
                self.on_record()
            except Exception as exc:
                self.set_status(f"Observer recording could not change: {exc}")

    def set_recording(self, recording, path=""):
        if self.closed:
            return
        self._recording, self._record_path = bool(recording), str(path)
        self.record_button.configure(text="Stop recording" if recording else "Record observer inputs",
                                     state="normal" if self.on_record and (recording or self.initialized) else "disabled")
        self.record_status.set(("Recording observer inputs: " if recording else "Observer recording folder: ") + str(path)
                               if path else ("Recording observer inputs" if recording else "Not recording"))

    def _clear_transcript(self):
        self._event_keys.clear()
        self._events.clear()
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    def _remember_events(self, result):
        for event in result.get("events", result.get("emitted", [])):
            key = (event.get("sequence"), event.get("mini"), event.get("confirmed_time"),
                   tuple(event.get("to_xy") or ()), event.get("to_cell"))
            if key in self._event_keys:
                continue
            self._event_keys.add(key)
            self._events.append(dict(event))
            self.transcript.configure(state="normal")
            self.transcript.insert("end", f"{len(self._events)}. {movement_text(event)}\n")
            self.transcript.see("end")
            self.transcript.configure(state="disabled")

    def _paint(self):
        board = self.selection.frame if self.selecting else self._latest_board
        if board is None or self.closed:
            return
        height, width = board.shape[:2]
        fit = self._fit = fit_image(width, height, *self._canvas_size)
        resized = cv2.resize(board, (fit.display_width, fit.display_height),
                             interpolation=cv2.INTER_AREA if fit.display_width < width else cv2.INTER_LINEAR)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
                                        master=self.window)
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.coords(self._image_item, fit.left, fit.top)
        self.canvas.delete("observation")

        def mark(xy, radius, color, label=None):
            try:
                x, y = map(float, xy)
                if not (math.isfinite(x) and math.isfinite(y)):
                    return
                cx, cy = fit.canvas_point(x, y)
                r = max(4, float(radius) * fit.display_width / width)
            except (TypeError, ValueError):
                return
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r, outline=color, width=2,
                                    tags="observation")
            if label:
                # A dark label background remains readable over arbitrary maps.
                text = self.canvas.create_text(cx + r + 4, cy - r, text=label, anchor="sw",
                                                fill=color, font=("Helvetica", 11, "bold"),
                                                tags="observation")
                bounds = self.canvas.bbox(text)
                if bounds:
                    # Keep the label readable when a physical base is near a
                    # preview edge. The measured circle stays at its exact
                    # location; only the explanatory text is shifted inward.
                    x0, y0, x1, y1 = bounds
                    canvas_width, canvas_height = self._canvas_size
                    dx = 3-x0 if x0 < 3 else min(0, canvas_width-3-x1)
                    dy = 3-y0 if y0 < 3 else min(0, canvas_height-3-y1)
                    if dx or dy:
                        self.canvas.move(text, dx, dy)
                        bounds = self.canvas.bbox(text)
                    backing = self.canvas.create_rectangle(*bounds, fill="#111111", outline="",
                                                             tags="observation")
                    self.canvas.tag_lower(backing, text)

        if self.selecting:
            for index, xy in enumerate(self.selection.points, 1):
                mark(xy, 16, "#ffd166", f"Mini {index}")
            return
        result = self._latest_result
        state = result.get("state") or {}
        for item in result.get("candidates", []):
            mark((item.get("cx"), item.get("cy")), item.get("radius", 12), "#919191")
        held = set(state.get("missing") or []) | set(state.get("unavailable") or [])
        held.update(result.get("unavailable") or [])
        held_result = result.get("held")
        if isinstance(held_result, (list, tuple, set)):
            held.update(held_result)
        active = state.get("active_mini")
        if active:
            held.add(active)
        hold_all = bool(result.get("obstructed") or result.get("needs_reset")
                        or result.get("acquired") is False
                        or held_result is True
                        or state.get("phase") in {"held", "blocked", "invalid", "uninitialized"})
        for mini, value in (result.get("positions") or {}).items():
            description = value if isinstance(value, dict) else (result.get("current_cells") or {}).get(mini, {})
            xy = description.get("xy") if description else value
            uncertain = bool(description.get("boundary_uncertain"))
            is_held = hold_all or mini in held or uncertain
            suffix = " (held)" if hold_all or mini in held else " (?)" if uncertain else ""
            mark(xy, 18, "#ffd166" if is_held else "#75ef8d", f"{mini}{suffix}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._photo = None
        try:
            self.window.destroy()
        except tk.TclError:
            pass
        if self.on_close:
            self.on_close()
