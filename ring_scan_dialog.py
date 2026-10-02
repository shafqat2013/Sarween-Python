"""A frozen camera-warp picker: the user, not a motion blob, identifies the ring."""

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from ring_calibration import sample_ring_patch


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
    replace = tk.BooleanVar(value=False)
    ttk.Checkbutton(footer, text="Replace active colors", variable=replace).pack(anchor="w")
    selection = {"lab": None}

    def confirm():
        if selection["lab"] is not None:
            on_confirm({"name": name, "sample_lab": selection["lab"],
                        "replace_colors": replace.get()})
            window.destroy()

    buttons = ttk.Frame(footer)
    buttons.pack(anchor="e", pady=(8, 0))
    ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="left", padx=4)
    use = ttk.Button(buttons, text="Use ring sample", command=confirm, state="disabled")
    use.pack(side="left")

    def choose(event):
        x, y = round(event.x * width / display_width), round(event.y * height / display_height)
        lab = sample_ring_patch(image, (x, y))
        selection["lab"] = lab
        use.configure(state="normal" if lab else "disabled")
        status.set(f"Ring sample: ({x}, {y})" if lab else "Mixed or edge pixels; sample not accepted")
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
