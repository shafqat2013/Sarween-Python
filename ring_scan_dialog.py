"""A frozen camera-warp picker for explicit ring identity and base appearance."""

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from ring_calibration import parse_base_diameter_squares, sample_ring_patch


def show_ring_scan(parent, name, image, on_confirm):
    window = tk.Toplevel(parent)
    window.title(f"Ring scan: {name}")
    image = image.copy()
    height, width = image.shape[:2]
    scale = min(1.0, (parent.winfo_screenwidth() - 80) / width,
                (parent.winfo_screenheight() - 260) / height)
    display_width, display_height = int(width * scale), int(height * scale)
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    photo = ImageTk.PhotoImage(rgb.resize((display_width, display_height)), master=window)
    view = tk.Canvas(window, width=display_width, height=display_height, highlightthickness=0)
    view.pack(padx=8, pady=8)
    view.create_image(0, 0, image=photo, anchor="nw")
    view.image = photo
    footer = ttk.Frame(window, padding=8)
    footer.pack(fill="x")
    zoom = tk.Canvas(footer, width=125, height=125, highlightthickness=0)
    zoom.pack(side="left", padx=(0, 12))
    status = tk.StringVar(value="Ring sample: none")
    ttk.Label(footer, textvariable=status).pack(anchor="w")
    sample_kind = tk.StringVar(value="identity")
    modes = ttk.Frame(footer)
    modes.pack(anchor="w", pady=(4, 4))
    for label, value in (("Ring color", "identity"), ("Base appearance", "base_appearance")):
        ttk.Radiobutton(modes, text=label, variable=sample_kind, value=value,
                        command=lambda: mode_changed()).pack(side="left", padx=(0, 8))
    instruction = tk.StringVar(value="Click a solid part of the colored ring.")
    ttk.Label(footer, textvariable=instruction, wraplength=360).pack(anchor="w")
    diameter = tk.StringVar(value="")
    diameter_row = ttk.Frame(footer)
    ttk.Label(diameter_row, text="Base diameter (grid squares)").pack(side="left", padx=(0, 8))
    ttk.Entry(diameter_row, textvariable=diameter, width=7).pack(side="left")
    ttk.Label(diameter_row, text="Optional; blank keeps the saved size").pack(side="left", padx=(8, 0))
    replace = tk.BooleanVar(value=False)
    replace_check = ttk.Checkbutton(footer, text="Replace active colors", variable=replace)
    replace_check.pack(anchor="w")
    selection = {"lab": None}

    def confirm():
        if sample_kind.get() not in ("identity", "base_appearance"):
            status.set("Choose Ring color or Base appearance")
            return
        values = {}
        if sample_kind.get() == "base_appearance":
            try:
                values["base_diameter_squares"] = parse_base_diameter_squares(diameter.get().strip() or None)
            except ValueError as exc:
                status.set(str(exc))
                return
        if selection["lab"] is not None:
            on_confirm({"name": name, "sample_lab": selection["lab"],
                        "replace_colors": replace.get(), "sample_kind": sample_kind.get(), **values})
            window.destroy()

    buttons = ttk.Frame(footer)
    buttons.pack(anchor="e", pady=(8, 0))
    ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)
    use = ttk.Button(buttons, text="Use ring sample", command=confirm, state="disabled")
    use.pack(side="left")

    def mode_changed():
        base = sample_kind.get() == "base_appearance"
        # Require a fresh click for a different cue, so a ring selection cannot
        # accidentally become a base sample merely by changing the mode.
        selection["lab"] = None
        replace.set(False)
        use.configure(text="Use base sample" if base else "Use ring sample", state="disabled")
        replace_check.configure(text="Replace base appearance colors" if base else "Replace active colors")
        if base:
            diameter_row.pack(anchor="w", pady=(4, 4), before=replace_check)
        else:
            diameter_row.pack_forget()
        instruction.set("Click a solid part of the visible circular base." if base
                        else "Click a solid part of the colored ring.")
        status.set("Base sample: none" if base else "Ring sample: none")
        view.delete("selection")
        zoom.delete("all")

    def choose(event):
        x, y = round(event.x * width / display_width), round(event.y * height / display_height)
        lab = sample_ring_patch(image, (x, y))
        selection["lab"] = lab
        use.configure(state="normal" if lab else "disabled")
        label = "Base sample" if sample_kind.get() == "base_appearance" else "Ring sample"
        status.set(f"{label}: ({x}, {y})" if lab else "Mixed or edge pixels; sample not accepted")
        view.delete("selection")
        view.create_rectangle(event.x - 5, event.y - 5, event.x + 5, event.y + 5,
                              outline="#ff00ff", width=2, tags="selection")
        patch = rgb.crop((x - 12, y - 12, x + 13, y + 13)).resize((125, 125), Image.Resampling.NEAREST)
        preview = ImageTk.PhotoImage(patch, master=window)
        zoom.delete("all")
        zoom.create_image(0, 0, image=preview, anchor="nw")
        zoom.image = preview

    view.bind("<Button-1>", choose)
    return window
