"""Opt-in placement of the native Foundry Mac window, without launching apps.

The native backend uses Accessibility only after an explicit request or saved
auto-placement preference. It checks existing permission without requesting it.
No Apple Events, screen capture, keyboard simulation, or Foundry file edits are
needed. AppKit and the system frameworks are loaded lazily.
"""

from __future__ import annotations

import ctypes
import json
import sys
import threading
import time
from pathlib import Path

from app_paths import atomic_write_json, data_path


FOUNDRY_BUNDLE_ID = "com.electron.foundryvtt"
PERMISSION_MESSAGE = (
    "To move Foundry, allow Sarween in System Settings → Privacy & Security → "
    "Accessibility. For a source run, macOS may list Terminal or your Python app. "
    "This optional feature does not request permission automatically."
)


def screen_coordinates(frame, primary_height):
    """Convert Cocoa bottom-left screen points to AX top-left screen points."""
    x, y, width, height = map(float, frame)
    return {"x": x, "y": float(primary_height) - y - height,
            "width": width, "height": height}


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _Accessibility:
    """Small owned-reference wrapper around the public Accessibility C API."""

    def __init__(self):
        self.ax = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        pointer, integer = ctypes.c_void_p, ctypes.c_int32
        declarations = (
            (self.ax, "AXIsProcessTrusted", [], ctypes.c_bool),
            (self.ax, "AXUIElementCreateApplication", [ctypes.c_int], pointer),
            (self.ax, "AXUIElementSetMessagingTimeout", [pointer, ctypes.c_float], integer),
            (self.ax, "AXUIElementCopyAttributeValue", [pointer, pointer, ctypes.POINTER(pointer)], integer),
            (self.ax, "AXUIElementSetAttributeValue", [pointer, pointer, pointer], integer),
            (self.ax, "AXUIElementIsAttributeSettable", [pointer, pointer, ctypes.POINTER(ctypes.c_bool)], integer),
            (self.ax, "AXValueCreate", [ctypes.c_uint32, pointer], pointer),
            (self.ax, "AXValueGetValue", [pointer, ctypes.c_uint32, pointer], ctypes.c_bool),
            (self.cf, "CFStringCreateWithCString", [pointer, ctypes.c_char_p, ctypes.c_uint32], pointer),
            (self.cf, "CFStringGetCString", [pointer, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32], ctypes.c_bool),
            (self.cf, "CFArrayGetCount", [pointer], ctypes.c_long),
            (self.cf, "CFArrayGetValueAtIndex", [pointer, ctypes.c_long], pointer),
            (self.cf, "CFBooleanGetValue", [pointer], ctypes.c_bool),
            (self.cf, "CFHash", [pointer], ctypes.c_ulong),
            (self.cf, "CFRelease", [pointer], None),
        )
        for library, name, arguments, result in declarations:
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, result
        self.true = pointer.in_dll(self.cf, "kCFBooleanTrue")
        self.false = pointer.in_dll(self.cf, "kCFBooleanFalse")

    def release(self, value):
        if value:
            self.cf.CFRelease(value)

    def _attribute(self, name):
        return self.cf.CFStringCreateWithCString(None, name.encode(), 0x08000100)

    def copy(self, element, name):
        attribute, result = self._attribute(name), ctypes.c_void_p()
        try:
            error = self.ax.AXUIElementCopyAttributeValue(element, attribute, ctypes.byref(result))
            return result.value if not error else None
        finally:
            self.release(attribute)

    def text(self, element, name):
        value = self.copy(element, name)
        if not value:
            return ""
        try:
            buffer = ctypes.create_string_buffer(512)
            return buffer.value.decode("utf-8", "replace") if self.cf.CFStringGetCString(
                value, buffer, len(buffer), 0x08000100) else ""
        finally:
            self.release(value)

    def boolean(self, element, name):
        value = self.copy(element, name)
        if not value:
            return False
        try:
            return bool(self.cf.CFBooleanGetValue(value))
        finally:
            self.release(value)

    def settable(self, element, name):
        attribute, result = self._attribute(name), ctypes.c_bool()
        try:
            return not self.ax.AXUIElementIsAttributeSettable(element, attribute, ctypes.byref(result)) and result.value
        finally:
            self.release(attribute)

    def set(self, element, name, value):
        attribute = self._attribute(name)
        try:
            error = self.ax.AXUIElementSetAttributeValue(element, attribute, value)
            if error:
                raise RuntimeError(f"Foundry could not change {name.removeprefix('AX')} (macOS error {error}).")
        finally:
            self.release(attribute)

    def geometry(self, element, name, kind, value):
        encoded = self.ax.AXValueCreate(kind, ctypes.byref(value))
        if not encoded:
            raise RuntimeError("macOS could not encode the target display position.")
        try:
            self.set(element, name, encoded)
        finally:
            self.release(encoded)

    def point(self, element):
        value = self.copy(element, "AXPosition")
        if not value:
            return None
        try:
            result = _Point()
            return (result.x, result.y) if self.ax.AXValueGetValue(value, 1, ctypes.byref(result)) else None
        finally:
            self.release(value)


class MacFoundryDisplayBackend:
    supported = sys.platform == "darwin"

    def __init__(self):
        self._accessibility = None
        self._pids = []

    @property
    def ax(self):
        if self._accessibility is None:
            self._accessibility = _Accessibility()
        return self._accessibility

    def trusted(self):
        return self.supported and bool(self.ax.ax.AXIsProcessTrusted())

    def displays(self):
        if not self.supported:
            return []
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Refresh display controls from the main application thread.")
        from AppKit import NSScreen
        self._pids = [int(app.processIdentifier()) for app in self._applications()]
        screens = list(NSScreen.screens())
        if not screens:
            return []
        primary_height = screens[0].frame().size.height
        result = []
        for index, screen in enumerate(screens):
            frame = screen.frame()
            identifier = str(screen.deviceDescription()["NSScreenNumber"])
            bounds = screen_coordinates((frame.origin.x, frame.origin.y, frame.size.width, frame.size.height), primary_height)
            result.append({"id": identifier, "name": str(screen.localizedName()),
                           "primary": index == 0, **bounds})
        return result

    def _applications(self):
        from AppKit import NSWorkspace
        return sorted((app for app in NSWorkspace.sharedWorkspace().runningApplications()
                       if app.bundleIdentifier() == FOUNDRY_BUNDLE_ID and not app.isTerminated()),
                      key=lambda app: int(app.processIdentifier()), reverse=True)

    def _visit_windows(self, callback):
        """Keep borrowed window references alive through each callback."""
        if not self.trusted():
            raise PermissionError(PERMISSION_MESSAGE)
        ax = self.ax
        # The UI refreshes this inventory. Worker threads use only PID snapshots
        # and the C Accessibility API, never AppKit display/process enumeration.
        for pid in tuple(self._pids):
            element = ax.ax.AXUIElementCreateApplication(pid)
            array = None
            try:
                ax.ax.AXUIElementSetMessagingTimeout(element, 0.5)
                array = ax.copy(element, "AXWindows")
                if not array:
                    continue
                for index in range(ax.cf.CFArrayGetCount(array)):
                    window = ax.cf.CFArrayGetValueAtIndex(array, index)
                    if ax.text(window, "AXRole") != "AXWindow" or ax.text(window, "AXSubrole") != "AXStandardWindow":
                        continue
                    key = (pid, int(ax.cf.CFHash(window)))
                    result = callback(key, window)
                    if result is not None:
                        return result
            finally:
                ax.release(array)
                ax.release(element)
        return None

    def windows(self):
        result = []
        self._visit_windows(lambda key, _window: result.append(key))
        return result

    def place(self, key, display, cancelled):
        def move(candidate, window):
            if key is not None and candidate != key:
                return None
            ax = self.ax
            # A native full-screen window commonly reports its frame as read
            # only. Check the full-screen control first, then wait for the
            # normal window before asking whether its frame can be changed.
            if not ax.settable(window, "AXFullScreen"):
                raise RuntimeError("Foundry's window cannot be moved or made full screen through macOS Accessibility.")
            if cancelled():
                return False
            if ax.boolean(window, "AXFullScreen"):
                ax.set(window, "AXFullScreen", ax.false)
                deadline = time.monotonic() + 4
                while ax.boolean(window, "AXFullScreen"):
                    if cancelled():
                        return False
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Foundry is still leaving full screen. Try Move Foundry again.")
                    time.sleep(0.05)
            if cancelled():
                return False
            if ax.boolean(window, "AXMinimized"):
                ax.set(window, "AXMinimized", ax.false)
            # Native full-screen transitions can keep the frame unchangeable
            # briefly after AXFullScreen becomes false. Retry only this window.
            deadline = time.monotonic() + 3
            while True:
                try:
                    if not all(ax.settable(window, name) for name in ("AXPosition", "AXSize")):
                        raise RuntimeError("Foundry's window cannot be moved or made full screen through macOS Accessibility.")
                    for name, kind, value in (
                            ("AXPosition", 1, _Point(display["x"], display["y"])),
                            ("AXSize", 2, _Size(display["width"], display["height"])),
                            ("AXPosition", 1, _Point(display["x"], display["y"]))):
                        if cancelled():
                            return False
                        ax.geometry(window, name, kind, value)
                    break
                except RuntimeError:
                    if cancelled():
                        return False
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.1)
            position = ax.point(window)
            if position is None or not (display["x"] - 2 <= position[0] < display["x"] + display["width"]
                                       and display["y"] - 2 <= position[1] < display["y"] + display["height"]):
                raise RuntimeError("Foundry did not move onto the selected display. Try again after its window finishes opening.")
            if cancelled():
                return False
            ax.set(window, "AXFullScreen", ax.true)
            deadline = time.monotonic() + 4
            while not ax.boolean(window, "AXFullScreen"):
                if cancelled():
                    return False
                if time.monotonic() >= deadline:
                    raise RuntimeError("Foundry moved, but macOS has not confirmed full screen. Try Move Foundry again.")
                time.sleep(0.05)
            return True

        result = self._visit_windows(move)
        if result is None:
            raise RuntimeError("Open the native Foundry app first, then click Move Foundry. Browser windows are not moved.")
        return result


class FoundryDisplayController:
    """One shared, opt-in worker for both desktop VTT panels."""

    def __init__(self, *, backend=None, path=None, poll_interval=1.0):
        self.backend = backend or MacFoundryDisplayBackend()
        self.path = Path(path) if path is not None else data_path("vtt_display.json")
        try:
            preferences = json.loads(self.path.read_text())
            if not isinstance(preferences, dict):
                preferences = {}
        except (OSError, ValueError):
            preferences = {}
        self._selected = str(preferences.get("displayId") or "")
        self._auto = preferences.get("auto") is True
        self._lock = threading.RLock()
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        self._enabled = True
        self._busy = False
        self._job = None
        self._generation = 0
        self._seen = None
        self._displays = []
        self._permission = False
        self._message = "Choose a display, then move the native Foundry window."
        self._last_refresh = float("-inf")
        self._poll_interval = max(0.01, float(poll_interval))

    def _refresh(self):
        now = time.monotonic()
        with self._lock:
            if now - self._last_refresh < 2:
                return
            self._last_refresh = now
        try:
            displays = self.backend.displays()
            permission = bool(self.backend.trusted()) if self.backend.supported else False
            with self._lock:
                self._displays, self._permission = displays, permission
                if not self._selected and displays:
                    self._selected = next((item["id"] for item in displays if not item["primary"]), displays[0]["id"])
        except Exception as exc:
            with self._lock:
                self._message = f"Display controls unavailable: {exc}"

    def status(self):
        self._refresh()
        with self._lock:
            if not self.backend.supported:
                message = "Foundry window placement is available on macOS."
            elif not self._permission:
                message = PERMISSION_MESSAGE
            elif not self._displays:
                message = "Connect the TV as an extended display first."
            elif self._selected not in {item["id"] for item in self._displays}:
                message = "The saved display is disconnected. Choose a connected display."
            else:
                message = self._message
            if self._auto and self._enabled and self._permission and not self._stop.is_set():
                self._ensure_worker()
            return {"supported": self.backend.supported,
                    "displays": [{key: item[key] for key in ("id", "name", "primary")} for item in self._displays],
                    "selectedDisplay": self._selected, "auto": self._auto,
                    "busy": self._busy, "message": message, "permission": self._permission}

    def configure(self, display_id=None, auto=None):
        self._refresh()
        with self._lock:
            if display_id is not None:
                display_id = str(display_id)
                if display_id not in {item["id"] for item in self._displays}:
                    raise ValueError("Choose a connected display.")
            if auto is not None and not isinstance(auto, bool):
                raise ValueError("Automatic placement must be on or off.")
            selected = self._selected if display_id is None else display_id
            automatic = self._auto if auto is None else auto
            atomic_write_json(self.path, {"displayId": selected, "auto": automatic})
            if selected != self._selected or automatic != self._auto:
                self._generation += 1
            if automatic != self._auto:
                self._seen = None
            self._selected, self._auto = selected, automatic
            self._message = "Automatic placement enabled for newly opened Foundry windows." if automatic else "Automatic placement is off."
            if automatic and self._enabled and self._permission:
                self._ensure_worker()
            self._wake.set()

    def set_enabled(self, enabled):
        with self._lock:
            enabled = bool(enabled)
            if enabled != self._enabled:
                self._generation += 1
            self._enabled = enabled
            if not self._enabled:
                if self._job is not None:
                    self._busy = False
                self._job = None
                self._seen = None
            self._wake.set()

    def place(self):
        self._refresh()
        with self._lock:
            if self._stop.is_set() or not self._enabled or self._busy:
                return False
            if not self.backend.supported or not self._permission:
                return False
            display = next((dict(item) for item in self._displays if item["id"] == self._selected), None)
            if display is None:
                self._message = "Choose a connected display."
                return False
            self._job = (display, self._generation)
            self._busy = True
            self._message = "Moving Foundry and entering full screen…"
            self._ensure_worker()
            self._wake.set()
            return True

    def _ensure_worker(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="sarween-foundry-display", daemon=True)
            self._thread.start()

    def _cancelled(self, generation=None):
        with self._lock:
            return (self._stop.is_set() or not self._enabled
                    or generation is not None and generation != self._generation)

    def _move(self, key, display, generation):
        cancelled = lambda: self._cancelled(generation)
        try:
            if not cancelled() and self.backend.place(key, display, cancelled):
                message = f"Foundry is full screen on {display['name']}."
            else:
                message = "Foundry window placement cancelled."
        except Exception as exc:
            message = str(exc)
        with self._lock:
            self._busy = False
            self._message = message

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                job, self._job = self._job, None
                auto = self._auto and self._enabled
                generation = self._generation
                display = next((dict(item) for item in self._displays if item["id"] == self._selected), None)
            if job is not None:
                self._move(None, *job)
            elif auto and display is not None:
                try:
                    keys = self.backend.windows()
                    with self._lock:
                        new = [] if self._seen is None else [key for key in keys if key not in self._seen]
                        self._seen = set(keys)
                    for key in new:
                        with self._lock:
                            if (not self._auto or not self._enabled or self._stop.is_set()
                                    or self._job is not None):
                                break
                            self._busy = True
                            self._message = "Moving a newly opened Foundry window…"
                        self._move(key, display, generation)
                except Exception as exc:
                    with self._lock:
                        self._message = str(exc)
            self._wake.wait(self._poll_interval if auto else None)
            self._wake.clear()

    def close(self):
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)


_controller = None
_controller_lock = threading.Lock()


def get_foundry_display_controller():
    global _controller
    with _controller_lock:
        if _controller is None or _controller._stop.is_set():
            _controller = FoundryDisplayController()
        return _controller
