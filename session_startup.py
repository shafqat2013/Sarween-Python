"""Cancellable setup in the existing main window, before camera capture."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import setup as hardware


class StartupPending(Exception):
    """The connection changed between readiness and camera startup."""


class SessionStartupUI(ttk.Frame):
    def __init__(self, parent, *, service, access, runtime_parent, on_ready,
                 on_cancel, on_runtime, observer=False, inventory_loader=None,
                 display_controller=None, foundry_setup_factory=None):
        super().__init__(parent, padding=8)
        self.service, self.access = service, access
        self.runtime_parent = runtime_parent
        self.on_ready, self.on_cancel, self.on_runtime = on_ready, on_cancel, on_runtime
        self.observer = observer
        self.inventory_loader = inventory_loader
        self.displays, self.webcams = [], []
        self.selection_data = self.scene_info = None
        self._closed = self._launching = self._owns_relay = False
        self._after_id = None
        self._inventory_generation = 0
        self._inventory_results = queue.Queue()
        self._last_scene_request = float("-inf")
        self._camera_capture = None
        self._guided_ready = False
        self._placement_warning = None
        self.phase = "hardware"
        self.step = tk.StringVar(self, "Step 1 of 2 · TV and camera")
        ttk.Label(self, textvariable=self.step).pack(anchor="w", pady=(0, 4))
        self.heading = tk.StringVar(self, "Choose your TV and camera")
        self.message = tk.StringVar(self, "Finding displays and cameras…")
        ttk.Label(self, textvariable=self.heading, font=("Helvetica", 14, "bold")).pack(anchor="w")
        label = ttk.Label(self, textvariable=self.message, wraplength=650, justify="left")
        label.pack(fill="x", pady=(6, 8))
        label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 8))
        row = ttk.Frame(self)
        row.pack(side="bottom", fill="x", pady=(8, 0))
        self.refresh_button = ttk.Button(row, text="Refresh hardware", command=self.refresh_inventory)
        self.refresh_button.pack(side="left")
        self.continue_button = ttk.Button(row, text="Next: Foundry and display", command=self.continue_setup)
        self.continue_button.pack(side="right")
        self.cancel_button = ttk.Button(row, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right", padx=6)
        self.edit_button = ttk.Button(row, text="Back", command=self.edit_hardware)
        self.edit_button.state(["disabled"])
        from control_panel import _scrollable_content, _bind_page_scroll
        selection_page = ttk.Frame(self)
        selection_page.pack(fill="both", expand=True)
        content = _scrollable_content(selection_page)
        self.selector = hardware.HardwareSelectionPanel(content, access=access, changed=self._selection_changed)
        self.selector.pack(fill="both", expand=True)
        from guided_foundry_setup import GuidedFoundrySetup
        self.foundry_setup = (foundry_setup_factory or GuidedFoundrySetup)(content,
            service=service, require_access=access.require, display_controller=display_controller)
        if observer:
            self.test_note = ttk.Label(self,
                text="Tracking test · Observe and record minis without moving Foundry tokens.",
                wraplength=640, justify="left")
            self.test_note.pack(fill="x", before=selection_page, pady=(0, 8))
        _bind_page_scroll(content)
        self.pack(fill="both", expand=True)
        self.refresh_inventory()

    def _schedule(self, callback, delay=150):
        self._cancel_timer()
        if not self._closed:
            self._after_id = self.after(delay, callback)

    def _cancel_timer(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _selection_changed(self):
        if not hasattr(self, "continue_button") or not hasattr(self, "selector"):
            return
        enabled = self.phase == "hardware" and self.selector.ready and not self._closed
        self.continue_button.state(["!disabled"] if enabled else ["disabled"])
        if self.phase == "hardware":
            mode = getattr(getattr(self.selector, "mode_combo", None), "get", lambda: "")()
            self.continue_button.configure(text="Start session" if mode == hardware.MODE_LABELS[hardware.MODE_SELF_HOSTED]
                                           else "Next: Foundry and display")

    def continue_setup(self):
        """Move through the guided pages without launching any devices early."""
        if self._closed or self._launching:
            return
        if self.phase == "placement":
            self._placement_warning = None
            self.continue_button.state(["disabled"])
            self.progress.start(60)
            self.foundry_setup.confirm_placement()
            self.phase = "foundry"
            self._poll_foundry()
        elif self.phase == "hardware":
            try:
                self.access.require()
                selection = self.selector.selection()
            except Exception as exc:
                self.message.set(str(exc))
                return
            if selection["mode"] != hardware.MODE_FOUNDRY:
                self.begin_connection()
                return
            self.selection_data = selection
            display = next((item for item in self.displays if item["index"] == selection["display_index"]), {})
            self.foundry_setup.set_display(display)
            self.selector.set_enabled(False)
            self.selector.pack_forget()
            self.foundry_setup.set_enabled(True)
            self.foundry_setup.pack(fill="both", expand=True)
            self.phase = "foundry_setup"
            self.step.set("Step 2 of 2 · Foundry and display")
            self.heading.set("Review your session settings")
            self.message.set("Start Foundry, apply your TV settings and connect the camera in one step.")
            self.refresh_button.state(["disabled"])
            self.refresh_button.pack_forget()
            self.edit_button.configure(text="Back")
            self.edit_button.pack(side="right", padx=6)
            self.edit_button.state(["!disabled"])
            self.continue_button.configure(text="Start tracking test" if self.observer else "Start session")
            self.continue_button.state(["!disabled"])
            self._scroll_to_top()
        else:
            self.begin_connection()

    def _scroll_to_top(self):
        # Each page starts at its heading rather than inheriting the previous
        # page's scroll position. This is Sarween's own content canvas.
        try:
            self.selector.master.master.yview_moveto(0)
        except (AttributeError, tk.TclError):
            pass

    def refresh_inventory(self):
        if self._closed or self._launching:
            return
        self._cancel_timer()
        self.phase = "inventory"
        self._inventory_generation += 1
        generation = self._inventory_generation
        self.heading.set("Choose your TV and camera")
        self.message.set("Finding displays and cameras…")
        self.selector.set_enabled(False)
        self.continue_button.state(["disabled"])
        self.refresh_button.state(["disabled"])
        self.progress.start(60)
        self.update_idletasks()
        # NSScreen inventory belongs to the Tk/main thread. Camera discovery
        # only enumerates devices; it never opens a VideoCapture for probing.
        try:
            displays = hardware.detect_displays_backup() if self.inventory_loader is None else None
        except Exception as exc:
            self._inventory_results.put((generation, None, str(exc)))
            self._schedule(self._poll_inventory)
            return

        def scan():
            try:
                result = (self.inventory_loader() if self.inventory_loader is not None
                          else (displays, hardware.detect_webcams_passive()))
                self._inventory_results.put((generation, result, None))
            except Exception as exc:
                self._inventory_results.put((generation, None, str(exc)))

        threading.Thread(target=scan, name="Sarween-Hardware-Inventory", daemon=True).start()
        self._schedule(self._poll_inventory)

    def _poll_inventory(self):
        self._after_id = None
        if self._closed:
            return
        if not self.access.allowed:
            self.cancel()
            return
        result = None
        while True:
            try:
                item = self._inventory_results.get_nowait()
            except queue.Empty:
                break
            if item[0] == self._inventory_generation:
                result = item
        if result is None:
            self._schedule(self._poll_inventory)
            return
        self.progress.stop()
        self.refresh_button.state(["!disabled"])
        self.phase = "hardware"
        if result[2]:
            self.message.set("Hardware could not be listed: " + result[2])
            return
        self.displays, self.webcams = result[1]
        defaults = hardware.load_last_selection()
        # AirPlay can change both its ID and inventory position. Restore a
        # unique saved name; an absent/ambiguous TV needs an explicit choice.
        if defaults.get("display_name"):
            matches = [item for item in self.displays if item.get("name") == defaults["display_name"]]
            defaults["display_index"] = matches[0]["index"] if len(matches) == 1 else -1
        defaults["camera_profile_id"] = hardware._active_camera_profile_id()
        self.selector.set_inventory(self.displays, self.webcams, defaults)
        self.selector.set_enabled(True)
        self.message.set("Saved choices are filled in. Preview or recalibrate only when needed.")
        self._selection_changed()

    def begin_connection(self):
        if self._closed or self._launching:
            return
        try:
            self.access.require()
            if self.phase not in ("foundry_setup", "failed") or not self.selection_data:
                self.selection_data = self.selector.selection()
            if not self.selection_data:
                return
        except Exception as exc:
            self.message.set(str(exc))
            return
        self._cancel_timer()
        self.selector.set_enabled(False)
        self.refresh_button.state(["disabled"])
        self.continue_button.state(["disabled"])
        self.edit_button.state(["!disabled"])
        self.phase = "connection"
        self.progress.start(60)
        if self.selection_data["mode"] != hardware.MODE_FOUNDRY:
            self._queue_launch()
            return
        guide = getattr(self, "foundry_setup", None)
        if guide is not None and not self._guided_ready:
            try:
                self._owns_relay = self._owns_relay or not self.service.running
                guide.set_enabled(True)
                if not guide.begin():
                    self.show_error(guide.status().get("error") or "Check your Foundry settings.")
                    guide.set_enabled(True)
                    return
            except Exception as exc:
                self.show_error(str(exc))
                return
            self.phase = "foundry"
            self.step.set("Starting your table")
            self.edit_button.configure(text="Change setup")
            self._poll_foundry()
            return
        self._begin_relay_connection()

    def _poll_foundry(self):
        self._after_id = None
        if self._closed:
            return
        if not self.access.allowed:
            self.cancel()
            return
        try:
            snapshot = self.foundry_setup.status()
        except Exception as exc:
            self.show_error(str(exc))
            return
        self.heading.set("Preparing Foundry and your TV")
        self.message.set(snapshot["message"])
        if snapshot.get("phase") == "placement_attention":
            self._placement_warning = snapshot.get("warning") or snapshot["message"]
            self.phase = "placement"
            self.progress.stop()
            self.heading.set("Foundry is ready · TV placement needs attention")
            self.message.set(self._placement_warning + "\n\nMove Foundry to the TV and enter full screen, then continue.")
            self.continue_button.configure(text="Continue")
            self.continue_button.state(["!disabled"])
        elif snapshot.get("ready"):
            self._guided_ready = True
            self._begin_relay_connection()
        elif snapshot.get("error") or snapshot.get("phase") in ("failed", "cancelled"):
            self.show_error(snapshot.get("error") or snapshot["message"])
        else:
            self._schedule(self._poll_foundry)

    def _begin_relay_connection(self):
        self.phase = "connection"
        self.progress.start(60)
        try:
            already_running = self.service.running
            self.service.start()
            self._owns_relay = self._owns_relay or not already_running
        except Exception as exc:
            self.show_error(str(exc))
            return
        self._poll_connection()

    def _poll_connection(self):
        self._after_id = None
        if self._closed:
            return
        if not self.access.allowed:
            self.cancel()
            return
        try:
            snapshot = self.service.startup_status()
        except Exception as exc:
            self.show_error("Could not read the Foundry connection: " + str(exc))
            return
        if snapshot["phase"] == "waiting_scene" and time.monotonic() - self._last_scene_request >= 1:
            import foundryoutput as fo
            try:
                fo.request_scene_info()
            except Exception:
                # Connection state can change between the snapshot and send.
                # The next poll can resume without another Start click.
                pass
            self._last_scene_request = time.monotonic()
        self.heading.set({"starting_relay": "Starting the local connection",
                          "waiting_module": "Waiting for the Foundry module",
                          "waiting_scene": "Waiting for the Foundry scene",
                          "ready": "Foundry connected",
                          "failed": "Connection needs attention"}[snapshot["phase"]])
        self.message.set(snapshot["message"])
        if snapshot["phase"] == "ready":
            self.scene_info = snapshot["scene_info"]
            self._queue_launch()
        elif snapshot["phase"] == "failed":
            self.show_error(snapshot.get("error") or snapshot["message"])
        else:
            # A module that has not connected yet is a waiting state, with no
            # startup deadline and no requirement to click Start a second time.
            self._schedule(self._poll_connection)

    def _queue_launch(self):
        if self._closed or self._launching:
            return
        from camera_sources import is_phone_camera_index, open_camera_capture, require_phone_profile_metadata
        if is_phone_camera_index(self.selection_data.get("webcam_index")):
            if getattr(self, "_camera_capture", None) is None:
                try:
                    self.access.require()
                    self._camera_capture = open_camera_capture(
                        self.selection_data["webcam_index"], selection=self.selection_data)
                except Exception as exc:
                    self.show_error(str(exc))
                    return
            if not self._camera_capture.status.get("fresh"):
                self.phase = "camera"
                self.heading.set("Connecting to your iPhone")
                self.message.set(self._camera_capture.status.get("message") or
                                 "Unlock your iPhone and open Sarween Camera. Keep the app in the foreground.")
                self._schedule(self._poll_camera)
                return
            profile_id = self.selection_data.get("camera_profile_id")
            if profile_id:
                try:
                    store = getattr(self.selector, "store", None) or hardware._profile_store()
                    profile = store.get_profile("camera", profile_id)
                    require_phone_profile_metadata(profile, self.selection_data["camera_source"],
                                                   self._camera_capture.status)
                except (ValueError, OSError) as exc:
                    self.show_error(str(exc))
                    return
        self._launching = True
        self.phase = "launching"
        self.heading.set("Starting the camera")
        self.message.set("Hardware is selected. Opening the live control panel…")
        self.cancel_button.state(["disabled"])
        self.edit_button.state(["disabled"])
        self.edit_button.pack_forget()
        self.refresh_button.pack(side="left")
        self._schedule(self._launch, delay=0)

    def _poll_camera(self):
        self._after_id = None
        if self._closed:
            return
        if not self.access.allowed:
            self.cancel()
            return
        # The reader owns connection/reconnection. This poll never blocks Tk
        # on a frame read and never starts a second camera reader.
        self._queue_launch()

    def take_camera_capture(self):
        """Transfer the already-connected source to the runtime exactly once."""
        capture = getattr(self, "_camera_capture", None)
        self._camera_capture = None
        return capture

    def _release_camera(self):
        capture = self.take_camera_capture()
        if capture is not None:
            capture.release()

    def _launch(self):
        self._after_id = None
        if self._closed:
            return
        if not self.access.allowed:
            self.cancel()
            return
        self.on_ready(self)

    def resume_wait(self):
        self._launching = False
        self.phase = "connection"
        self.cancel_button.state(["!disabled"])
        self.edit_button.state(["!disabled"])
        self.progress.start(60)
        self._schedule(self._poll_connection, delay=0)

    def runtime_started(self, panel):
        self.progress.stop()
        self.on_runtime(panel)

    def show_error(self, message):
        if self._closed:
            return
        self._cancel_timer()
        self._release_camera()
        self._launching = False
        self.phase = "failed"
        guide = getattr(self, "foundry_setup", None)
        if guide is not None and not self._guided_ready:
            guide.set_enabled(True)
        self.progress.stop()
        self.heading.set("Session needs attention")
        self.message.set(str(message))
        self.continue_button.configure(text="Retry")
        self.continue_button.state(["!disabled"])
        self.cancel_button.state(["!disabled"])
        self.edit_button.state(["!disabled"])

    def edit_hardware(self):
        if self._closed or self._launching:
            return
        self._cancel_timer()
        self._release_camera()
        guide = getattr(self, "foundry_setup", None)
        if guide is not None:
            guide.cancel()
            guide.pack_forget()
            self.selector.pack(fill="both", expand=True)
            self._guided_ready = False
            self.step.set("Step 1 of 2 · TV and camera")
        self.phase = "hardware"
        self.progress.stop()
        self.selector.set_enabled(True)
        self.refresh_button.state(["!disabled"])
        self.edit_button.state(["disabled"])
        self.heading.set("Choose your TV and camera")
        self.message.set("Update your choices, then continue.")
        self._selection_changed()
        self._scroll_to_top()

    def dispose(self, *, stop_owned_relay=False):
        if self._closed:
            return
        self._closed = True
        self._inventory_generation += 1
        self._cancel_timer()
        self._release_camera()
        guide = getattr(self, "foundry_setup", None)
        if guide is not None:
            guide.close()
        self.progress.stop()
        try:
            if stop_owned_relay and self._owns_relay and self.service.running:
                self.service.stop()
        finally:
            self.destroy()

    def cancel(self):
        if self._closed:
            return
        try:
            self.dispose(stop_owned_relay=True)
        finally:
            self.on_cancel(self)
