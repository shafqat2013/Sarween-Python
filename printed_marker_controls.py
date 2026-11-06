"""Paper-marker alignment controls embedded in the existing VTT Display page."""
from pathlib import Path
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import webbrowser

from app_paths import data_path


INSTRUCTIONS = (
    "Save and select the calibrated display profile. Put Foundry full screen on that TV before "
    "creating the matching print sheet, then print at actual size (100%). "
    "Place each paper card directly over its matching screen guide: "
    "20 over 10, 21 over 11, 22 over 12, 23 over 13. Align the edges, clear your hands, then align. "
    "This uses your declared placement; keep the cards fixed afterward."
)
BEZEL_INSTRUCTIONS = (
    "For paper beside the LCD instead, use Calibrate bezel placement. Keep all four paper markers "
    "and all four screen markers visible together, with the screen bright through the polarizer "
    "(or remove it) during this one-time alignment."
)


def _controller():
    from printed_marker_runtime import get_printed_marker_controller
    return get_printed_marker_controller()


def _open_sheet(path):
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif os.name == "nt":
        os.startfile(str(path))
    else:
        webbrowser.open(path.as_uri())


def _label(parent, **options):
    label = ttk.Label(parent, wraplength=480, justify="left", **options)
    label.bind("<Configure>", lambda event: label.configure(wraplength=max(100, event.width)))
    return label


class PrintedMarkerControls(ttk.LabelFrame):
    """Commands are explicit; polling never changes profiles or marker mode."""

    def __init__(self, parent, *, require_access=None, controller=None, store=None,
                 can_change=None, open_sheet=None, sheet_builder=None, sheet_parameters=None):
        super().__init__(parent, text="Printed ArUco markers", padding=10)
        if store is None:
            from hardware_profiles import HardwareProfiles
            store = HardwareProfiles()
        self.store, self._controller = store, controller
        self._require_access, self._can_change = require_access, can_change
        self._open_sheet = open_sheet or _open_sheet
        self._sheet_builder, self._sheet_parameters = sheet_builder, sheet_parameters
        self._enabled = False
        self._dirty = False
        self._syncing = False
        self._profile_id = None
        self._last_status = {}
        self._notice = ""
        self.use_paper = tk.BooleanVar(self, False)
        self.profile = tk.StringVar(self)
        self.message = tk.StringVar(self)
        self.columnconfigure(0, weight=1)
        _label(self, text=INSTRUCTIONS).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        _label(self, textvariable=self.profile).grid(row=1, column=0, columnspan=2, sticky="ew", pady=3)
        self.sheet_button = ttk.Button(self, text="Create matching print sheet", command=self.open_printable_sheet)
        self.sheet_button.grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        buttons = ttk.Frame(self)
        buttons.grid(row=3, column=0, columnspan=2, sticky="w", pady=3)
        self.register_button = ttk.Button(buttons, text="Align with screen guides", command=self.register)
        self.register_button.pack(side="left", padx=(0, 6))
        self.cancel_button = ttk.Button(buttons, text="Cancel alignment", command=self.cancel_registration)
        self.cancel_button.pack(side="left")
        self.use_check = ttk.Checkbutton(self, text="Use printed markers", variable=self.use_paper,
                                         command=self._edited)
        self.use_check.grid(row=4, column=0, sticky="w", pady=3)
        self.apply_button = ttk.Button(self, text="Apply", command=self.apply_mode)
        self.apply_button.grid(row=4, column=1, sticky="e", pady=3)
        self.bezel_button = ttk.Button(self, text="Calibrate bezel placement", command=self.register_bezel)
        self.bezel_button.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 3))
        _label(self, text=BEZEL_INSTRUCTIONS).grid(row=6, column=0, columnspan=2, sticky="ew", pady=3)
        _label(self, textvariable=self.message).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.refresh()

    def _get_controller(self):
        if self._controller is None:
            self._controller = _controller()
        return self._controller

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        self._update_enabled()

    def _context_available(self):
        return self._can_change is None or bool(self._can_change())

    def _available(self):
        status = self._last_status
        return bool(self._enabled and self._profile_id and status.get("available")
                    and not status.get("pending") and self._context_available()
                    and (not status.get("profileId") or status["profileId"] == self._profile_id))

    def _update_enabled(self):
        status = self._last_status
        available = self._available()
        can_choose = available and (status.get("calibrated") or status.get("mode") == "printed")
        self.sheet_button.configure(state="normal" if self._enabled else "disabled")
        self.register_button.configure(state="normal" if available else "disabled")
        self.bezel_button.configure(state="normal" if available else "disabled")
        self.cancel_button.configure(state="normal" if self._enabled and status.get("pending") else "disabled")
        self.use_check.configure(state="normal" if can_choose else "disabled")
        can_apply = available and self._dirty and (not self.use_paper.get() or status.get("calibrated"))
        self.apply_button.configure(state="normal" if can_apply else "disabled")

    def _edited(self):
        if self._syncing:
            return
        self._dirty = True
        self._notice = ""
        self._update_enabled()

    def refresh(self):
        try:
            profile = self.store.active_profile("display")
            status = dict(self._get_controller().status())
            self._last_status = status
            profile_id = profile["id"] if profile else None
            if profile_id != self._profile_id:
                self._dirty, self._notice = False, ""
            self._profile_id = profile_id
            self.profile.set(f"Display profile: {profile['name']}" if profile else "No display profile selected.")
            self._syncing = True
            try:
                if not self._dirty:
                    self.use_paper.set(status.get("mode") == "printed")
            finally:
                self._syncing = False
            if not profile:
                message = "Save a display profile in Hardware Profiles, then select it in Display profile above."
            elif not status.get("available"):
                message = status.get("message") or "Start a live camera session to align or use printed markers."
            elif status.get("profileId") and status["profileId"] != profile_id:
                message = "The camera is updating to this display profile. Wait a moment."
            elif status.get("pending"):
                message = status.get("message") or "Aligning printed markers… Keep all eight markers visible."
            elif not self._context_available():
                message = "Finish the current recording or display change before changing markers."
            else:
                message = self._notice or status.get("message") or (
                    "Printed markers registered. Choose the marker mode and Apply."
                    if status.get("calibrated") else "Ready for one-time alignment.")
            self.message.set(message)
        except (ValueError, OSError, ImportError, RuntimeError) as exc:
            self._last_status = {"available": False, "pending": False}
            self.message.set(str(exc))
        self._update_enabled()

    def _access(self):
        if not self._enabled:
            raise PermissionError("Sign in before changing printed markers.")
        if self._require_access is not None:
            self._require_access()

    def _act(self, operation, *, cancel=False):
        try:
            self._access()
            previous_profile_id = self._profile_id
            self.refresh()
            if cancel:
                if not self._last_status.get("pending"):
                    return False
            else:
                if previous_profile_id != self._profile_id:
                    raise ValueError("The display profile changed. Check your choices and try again.")
                if not self._available():
                    return False
            operation(self._get_controller())
            self._notice = ""
            self.refresh()
            return True
        except (ValueError, OSError, PermissionError, RuntimeError) as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False

    def register(self):
        return self._act(lambda controller: controller.request_calibration(aligned_to_guides=True))

    def register_bezel(self):
        return self._act(lambda controller: controller.request_calibration())

    def cancel_registration(self):
        return self._act(lambda controller: controller.cancel_calibration(), cancel=True)

    def apply_mode(self):
        wanted = self.use_paper.get()
        if type(wanted) is not bool:
            self.message.set("Choose on or off for printed markers.")
            return False

        def apply(controller):
            if wanted and not self._last_status.get("calibrated"):
                raise ValueError("Register printed markers before using them.")
            controller.use_printed(wanted)
            self._dirty = False
        return self._act(apply)

    def open_printable_sheet(self):
        try:
            self._access()
            profile = self.store.active_profile("display")
            if not profile:
                raise ValueError("Save and select a display profile before creating its print sheet.")
            context = self._get_controller().view_context()
            if not context.get("ready"):
                raise ValueError("Connect Foundry full screen on the profile's TV before creating its print sheet.")
            builder, parameters = self._sheet_builder, self._sheet_parameters
            if builder is None or parameters is None:
                from printed_marker_sheet import build_sheet, parameters_from_context
                builder, parameters = builder or build_sheet, parameters or parameters_from_context
            options = parameters(profile, context)
            result = builder(data_path("Printouts/sarween-printed-markers-a4.pdf"), **options)
            path = Path(result["path"]).resolve()
            if not path.is_file():
                raise OSError("The matching print sheet could not be created.")
            if self.store.active_profile("display") != profile:
                raise ValueError("The display profile changed. Create its matching print sheet again.")
            self._open_sheet(path)
            return True
        except (OSError, ValueError, PermissionError, ImportError, RuntimeError) as exc:
            self._notice = str(exc)
            self.message.set(self._notice)
            return False
