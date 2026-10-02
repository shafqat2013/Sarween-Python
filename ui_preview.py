"""Preview the actual Sarween panels with sample data and no hardware services."""

from __future__ import annotations

import argparse
import math
import tkinter as tk
from tkinter import ttk

from control_panel import ControlPanel


SCENARIOS = ("Ready", "Missing markers", "Move pending", "Move failed", "Disconnected")


def sample_roster():
    return [
        {"id": color.lower(), "name": color, "ringColor": color.lower(),
         "scanStatus": "Sample profile" if index < 4 else "Needs scan",
         "sampleCount": 3 if index < 4 else 0, "token": f"{color} (sample)",
         "confidence": .95 - index * .04 if index < 4 else None,
         "position": cell if index < 4 else None}
        for index, (color, cell) in enumerate(zip(
            ("Red", "Blue", "Yellow", "Green", "White"),
            ("J7", "M9", "Y14", "K21", "AL20")))
    ]


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


class PreviewPanel(ControlPanel):
    def __init__(self, *, tk_root=None):
        super().__init__(mode="foundry", tk_root=tk_root)
        self.root.title("Sarween Control Panel - Offline Preview")
        self.var_mode.set("Mode: Offline preview / sample data")
        self._scenario = tk.StringVar(master=self.root, value="Ready")
        menu = tk.Menu(self.root)
        states = tk.Menu(menu, tearoff=False)
        for scenario in SCENARIOS:
            states.add_radiobutton(label=scenario, value=scenario, variable=self._scenario,
                                   command=lambda name=scenario: self.set_scenario(name))
        menu.add_cascade(label="Preview", menu=states)
        self.root.configure(menu=menu)
        self._disable_hardware_controls(self.root)
        self.update_mini_library(sample_roster())
        self.update_positions({row["name"]: row["position"] for row in sample_roster()})
        self.set_scenario("Ready")

    @staticmethod
    def _disable_hardware_controls(root):
        for widget in descendants(root):
            if isinstance(widget, ttk.Checkbutton):
                widget.state(["disabled"])
            elif isinstance(widget, ttk.Button) and widget.cget("text") not in (
                    "Exit", "Stop session", "Close", "Mini Library", "Retry moves"):
                widget.state(["disabled"])

    def set_scenario(self, name):
        if name not in SCENARIOS:
            raise ValueError(f"Unknown preview state: {name}")
        self._scenario.set(name)
        missing = name == "Missing markers"
        disconnected = name == "Disconnected"
        self.var_lock.set("Lock: waiting (sample)" if missing else "Lock: acquired (sample)")
        self.var_markers.set("Markers: 2/4 (sample)" if missing else "Markers: 4/4 (sample)")
        self.var_missing.set("Missing: [10, 11] (sample)" if missing else "Missing: none (sample)")
        self.var_fps.set("FPS: -- (no camera)")
        self.var_foundry.set("Foundry: disconnected (sample)" if disconnected else "Foundry: connected (sample)")
        messages = {"Ready": "All moves acknowledged (sample)",
                    "Missing markers": "Tracking paused: registration unavailable (sample)",
                    "Move pending": "Red -> M9: waiting for acknowledgement (sample)",
                    "Move failed": "Red -> M9: no acknowledgement after 3 attempts (sample)",
                    "Disconnected": "Latest move retained for reconnection (sample)"}
        self.set_delivery_status({"message": messages[name], "retryAvailable": name == "Move failed"})

    def _act_retry_moves(self):
        self.set_scenario("Move pending")

    def _show_mini_library(self):
        super()._show_mini_library()
        self._library_window.title("Sarween Mini Library - Offline Preview")
        self._disable_hardware_controls(self._library_window)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--close-after", type=float, help="Automatically close after this many seconds (UI smoke tests)")
    args = parser.parse_args(argv)
    if args.close_after is not None and (not math.isfinite(args.close_after) or not 0 < args.close_after <= 120):
        parser.error("--close-after must be between 0 and 120 seconds")
    panel = PreviewPanel()
    root = panel.root

    def check_exit():
        # The preview consumes exit locally; no live action dispatcher is started.
        if panel.pop_actions()["exit"]:
            root.destroy()
        else:
            root.after(100, check_exit)

    root.after(100, check_exit)
    if args.close_after is not None:
        root.after(round(args.close_after * 1000), root.destroy)
    print("Sarween offline UI preview: sample data only; no camera, Foundry server, tracking worker, or file saves.")
    try:
        root.mainloop()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    main()
