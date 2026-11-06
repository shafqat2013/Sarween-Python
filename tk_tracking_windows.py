"""Tracking preview windows with one native event loop on macOS.

OpenCV's Qt event pump can dispatch Tk callbacks outside Tk's Python bridge.
Live Mac tracking therefore uses Tk for every preview and keyboard event.
Image processing stays in OpenCV; this module never changes source frames.
"""

from collections import deque
from dataclasses import dataclass
import sys
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageTk


class NoTrackingWindows:
    def ensure_window(self, *_args, **_kwargs):
        pass

    def imshow(self, *_args, **_kwargs):
        pass

    def waitKey(self, _delay=1):
        return -1

    def destroyWindow(self, _name):
        pass

    def destroyAllWindows(self):
        pass


class OpenCvTrackingWindows:
    def ensure_window(self, name, width=1280, height=720):
        from cv_core import ensure_window
        ensure_window(name, width, height)

    def imshow(self, name, frame):
        cv2.imshow(name, frame)

    def waitKey(self, delay=1):
        return cv2.waitKey(delay)

    def destroyWindow(self, name):
        cv2.destroyWindow(name)

    def destroyAllWindows(self):
        cv2.destroyAllWindows()


@dataclass
class _Preview:
    window: object
    canvas: object
    item: object
    size: tuple[int, int]
    photo: object = None


class TkTrackingWindows:
    def __init__(self, root):
        if root is None:
            raise ValueError("Tk previews require the existing application root")
        self.root = root
        self._windows = {}
        self._user_closed = set()
        self._keys = deque()
        self._pumping = False
        self._quit_requested = False

    def ensure_window(self, name, width=1280, height=720):
        if name in self._windows or name in self._user_closed:
            return
        window = tk.Toplevel(self.root)
        window.title(name)
        window.resizable(True, True)
        window.minsize(160, 90)
        canvas = tk.Canvas(window, width=width, height=height,
                           background="black", highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        item = canvas.create_image(width // 2, height // 2, anchor=tk.CENTER)
        preview = _Preview(window, canvas, item, (width, height))
        self._windows[name] = preview

        def resized(event):
            # Painting on the next video frame avoids recursive geometry work
            # during Cocoa's live-resize callbacks.
            preview.size = (max(1, int(event.width)), max(1, int(event.height)))

        canvas.bind("<Configure>", resized)
        window.bind("<KeyPress>", self._key)
        window.protocol("WM_DELETE_WINDOW", lambda: self._close(name, user=True))

    def _key(self, event):
        char = getattr(event, "char", "")
        if char == "q":
            # Some live-view refreshes poll only to pump events. Preserve quit
            # until the session's final keyboard check can consume it.
            self._quit_requested = True
        elif len(char) == 1:
            self._keys.append(ord(char))
        elif getattr(event, "keysym", "") == "Escape":
            self._keys.append(27)

    def imshow(self, name, frame):
        if frame is None or name in self._user_closed:
            return
        if name not in self._windows:
            self.ensure_window(name)
        preview = self._windows[name]
        if not isinstance(frame, np.ndarray) or frame.ndim not in (2, 3):
            raise ValueError("Preview requires a grayscale or BGR image")
        height, width = frame.shape[:2]
        if height == 0 or width == 0:
            return
        canvas_width, canvas_height = preview.size
        scale = min(canvas_width / width, canvas_height / height)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        if resized.ndim == 3:
            if resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            elif resized.shape[2] == 4:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGBA)
            else:
                raise ValueError("Preview requires three or four color channels")
        photo = ImageTk.PhotoImage(image=Image.fromarray(resized), master=self.root)
        preview.canvas.itemconfigure(preview.item, image=photo)
        preview.canvas.coords(preview.item, canvas_width // 2, canvas_height // 2)
        preview.photo = photo

    def waitKey(self, _delay=1):
        if not self._pumping:
            self._pumping = True
            try:
                self.root.update()
            except tk.TclError:
                return ord("q")
            finally:
                self._pumping = False
        if self._quit_requested:
            return ord("q")
        return self._keys.popleft() if self._keys else -1

    def _close(self, name, *, user=False):
        preview = self._windows.pop(name, None)
        if user:
            self._user_closed.add(name)
        else:
            self._user_closed.discard(name)
        if preview is not None:
            try:
                preview.window.destroy()
            except tk.TclError:
                pass

    def destroyWindow(self, name):
        self._close(name)

    def destroyAllWindows(self):
        for name in tuple(self._windows):
            self._close(name)
        self._user_closed.clear()
        self._keys.clear()
        self._quit_requested = False


def create_tracking_windows(root=None, *, enabled=True, platform=None):
    if not enabled:
        return NoTrackingWindows()
    if (sys.platform if platform is None else platform) == "darwin":
        return TkTrackingWindows(root)
    return OpenCvTrackingWindows()
