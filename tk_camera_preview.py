# tk_camera_preview.py
# Tkinter-based camera preview window.
#
# Replaces cv2.imshow("Camera ...") during the ArUco lock phase so the app
# works as a double-clicked .app on macOS without triggering the Qt/Cocoa
# NSPersistentUI crash that cv2.waitKey causes.
#
# Usage (from tracking engine or calibration):
#
#   from tk_camera_preview import TkCameraPreview
#
#   preview = TkCameraPreview()          # creates window immediately
#   ...
#   preview.update(cam_bgr, locked=False, marker_count=2, missing_ids=[1, 3])
#   ...
#   preview.close()                      # call once ArUco lock is achieved
#
# The window pumps its own Tk event loop via root.update() — call
# preview.update() once per frame from your existing frame loop.
# No separate thread needed.

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import sys


def _get_root() -> tk.Tk:
    """Reuse the persistent root from setup.py if available, else create one."""
    try:
        import setup as s
        return s._get_persistent_root()
    except Exception:
        pass
    if not hasattr(_get_root, "_root"):
        root = tk.Tk()
        root.withdraw()
        _get_root._root = root
    return _get_root._root


class TkCameraPreview:
    """
    A Toplevel window that shows a live camera feed with an ArUco status overlay.

    Call update(cam_bgr, ...) once per frame from your frame loop.
    Call close() to destroy the window (e.g. once locked).
    """

    PREVIEW_W = 640
    PREVIEW_H = 480

    def __init__(self, title: str = "Camera Preview — Position your map"):
        self._root = _get_root()
        self._closed = False

        self._win = tk.Toplevel(self._root)
        self._win.title(title)
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close_button)

        # Canvas for the video frame
        self._canvas = tk.Canvas(
            self._win,
            width=self.PREVIEW_W,
            height=self.PREVIEW_H,
            bg="black",
            highlightthickness=0,
        )
        self._canvas.pack()

        # Status bar below the frame
        self._status_var = tk.StringVar(value="Searching for ArUco markers…")
        status_frame = tk.Frame(self._win, bg="#1e1e1e", pady=6)
        status_frame.pack(fill=tk.X)

        self._status_label = tk.Label(
            status_frame,
            textvariable=self._status_var,
            bg="#1e1e1e",
            fg="#dddddd",
            font=("Helvetica", 13),
            anchor="w",
            padx=12,
        )
        self._status_label.pack(fill=tk.X)

        self._marker_var = tk.StringVar(value="")
        self._marker_label = tk.Label(
            status_frame,
            textvariable=self._marker_var,
            bg="#1e1e1e",
            fg="#aaaaaa",
            font=("Helvetica", 11),
            anchor="w",
            padx=12,
        )
        self._marker_label.pack(fill=tk.X)

        self._photo = None  # keep reference to prevent GC

        # Bring to front
        self._win.lift()
        self._win.focus_force()
        try:
            if getattr(sys, "frozen", False):
                from AppKit import NSApplication
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass

        self._root.update()

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        cam_bgr: np.ndarray,
        locked: bool = False,
        marker_count: int = 0,
        missing_ids: list = None,
    ) -> None:
        """
        Push a new frame to the preview window and update the status text.
        Call once per frame from your frame loop.
        """
        if self._closed:
            return

        # Update status text
        if locked:
            self._status_var.set("✅ ArUco lock acquired — tracking started")
            self._marker_label.config(fg="#44cc44")
            self._marker_var.set("All 4 markers found")
        else:
            self._status_var.set(
                f"Searching for ArUco markers… ({marker_count}/4 found)"
            )
            missing_str = (
                f"Missing marker IDs: {missing_ids}" if missing_ids else ""
            )
            self._marker_var.set(missing_str)
            self._marker_label.config(fg="#aaaaaa")

        # Resize frame to fit canvas
        try:
            frame = self._prepare_frame(cam_bgr)
            img = Image.fromarray(frame)
            self._photo = ImageTk.PhotoImage(image=img)
            self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)
        except Exception:
            pass

        # Pump the Tk event loop
        try:
            self._root.update()
        except Exception:
            self._closed = True

    def close(self) -> None:
        """Destroy the preview window."""
        if self._closed:
            return
        self._closed = True
        try:
            self._win.destroy()
        except Exception:
            pass
        try:
            self._root.update()
        except Exception:
            pass

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ── Internal ───────────────────────────────────────────────────────────────

    def _on_close_button(self) -> None:
        """User clicked the window's X button — treat as cancel."""
        self._closed = True
        try:
            self._win.destroy()
        except Exception:
            pass
        # Signal main loop to exit cleanly
        raise SystemExit("Camera preview closed by user.")

    def _prepare_frame(self, cam_bgr: np.ndarray) -> np.ndarray:
        """Resize + convert BGR→RGB for display."""
        h, w = cam_bgr.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(cam_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to canvas size with black letterbox
        canvas = np.zeros((self.PREVIEW_H, self.PREVIEW_W, 3), dtype=np.uint8)
        y0 = (self.PREVIEW_H - new_h) // 2
        x0 = (self.PREVIEW_W - new_w) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = resized

        # BGR → RGB
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ── Foundry wait window ────────────────────────────────────────────────────────

class TkFoundryWait:
    """
    A Toplevel window that shows "waiting for Foundry scene info" status.
    Replaces the cv2.imshow loop in calibration._foundry_wait_for_scene_grid().

    Usage:
        wait_win = TkFoundryWait()
        while True:
            wait_win.update(ok=False)
            # ... poll Foundry ...
            if ok:
                wait_win.close()
                break
    """

    def __init__(self):
        self._root = _get_root()
        self._closed = False

        self._win = tk.Toplevel(self._root)
        self._win.title("Sarween — Waiting for Foundry")
        self._win.geometry("480x140")
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        frame = tk.Frame(self._win, bg="#1e1e1e", padx=20, pady=20)
        frame.pack(fill=tk.BOTH, expand=True)

        self._title_label = tk.Label(
            frame,
            text="Waiting for Foundry scene info…",
            bg="#1e1e1e",
            fg="#ffffff",
            font=("Helvetica", 15, "bold"),
            anchor="w",
        )
        self._title_label.pack(fill=tk.X, pady=(0, 8))

        self._sub_var = tk.StringVar(
            value="Make sure: Sarween module is active and a scene is open."
        )
        self._sub_label = tk.Label(
            frame,
            textvariable=self._sub_var,
            bg="#1e1e1e",
            fg="#aaaaaa",
            font=("Helvetica", 12),
            anchor="w",
            wraplength=440,
            justify="left",
        )
        self._sub_label.pack(fill=tk.X)

        cancel_frame = tk.Frame(self._win, bg="#1e1e1e", pady=8)
        cancel_frame.pack(fill=tk.X)
        ttk.Button(cancel_frame, text="Cancel", command=self._on_close).pack(side=tk.RIGHT, padx=12)

        self._win.lift()
        self._win.focus_force()
        try:
            if getattr(sys, "frozen", False):
                from AppKit import NSApplication
                NSApplication.sharedApplication().activateIgnoringOtherApps_(True)
        except Exception:
            pass

        self._root.update()

    def update(self, ok: bool = False, scene_w: int = 0, scene_h: int = 0, grid_px: int = 0) -> None:
        if self._closed:
            return
        if ok:
            self._title_label.config(text="✅ Scene info received!", fg="#44cc44")
            self._sub_var.set(f"Scene: {scene_w}×{scene_h}, grid: {grid_px}px")
        else:
            self._title_label.config(
                text="Waiting for Foundry scene info…", fg="#ffffff"
            )
            self._sub_var.set(
                "Make sure: Sarween module is active and a scene is open in Foundry."
            )
        try:
            self._root.update()
        except Exception:
            self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._win.destroy()
        except Exception:
            pass
        try:
            self._root.update()
        except Exception:
            pass

    def _on_close(self) -> None:
        self._closed = True
        try:
            self._win.destroy()
        except Exception:
            pass
        raise SystemExit("Cancelled waiting for Foundry scene info.")
