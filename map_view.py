"""A lightweight faceted, top-down view of measured board state."""

import tkinter as tk
from tkinter import ttk
from tracking_evaluation import a1_to_row_col


COLORS = {"red": "#e75f67", "blue": "#58a9e8", "yellow": "#e9ca57", "green": "#6dc896", "white": "#e5e8ed"}
PALETTE = list(COLORS.values()) + ["#cc8cc9"]


class MapView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.state = {"grid": [23, 16], "positions": {}}
        self.canvas = tk.Canvas(self, background="#192125", highlightthickness=0, width=500, height=310)
        self.caption = tk.StringVar(value="No tracked positions")
        caption = ttk.Label(self, textvariable=self.caption, wraplength=460)
        caption.pack(side="bottom", fill="x", pady=4)
        self.canvas.pack(fill="both", expand=True)
        caption.bind("<Configure>", lambda event: caption.configure(wraplength=max(100, event.width)))
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def set_state(self, state):
        self.state = state or {"grid": [23, 16], "positions": {}}
        self.draw()

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        cols, rows = self.state.get("grid", [23, 16])
        cols, rows = max(1, int(cols)), max(1, int(rows))
        width, height = canvas.winfo_width(), canvas.winfo_height()
        unit = min(max(1, width - 50) / cols, max(1, height - 38) / rows)
        left, top = (width - cols * unit) / 2, (height - rows * unit) / 2
        canvas.create_rectangle(left, top, left + cols * unit, top + rows * unit, fill="#253135", outline="#647276")
        stride = max(1, int(10 / unit))
        for col in range(0, cols + 1, stride):
            x = left + col * unit
            canvas.create_line(x, top, x, top + rows * unit, fill="#374649")
        for row in range(0, rows + 1, stride):
            y = top + row * unit
            canvas.create_line(left, y, left + cols * unit, y, fill="#374649")
        names = []
        for index, (name, position) in enumerate(sorted(self.state.get("positions", {}).items())):
            try:
                row, col = a1_to_row_col(position["cell"])
            except (ValueError, KeyError):
                continue
            if not (0 <= col < cols and 0 <= row < rows):
                continue
            lost = position.get("status") != "tracked"
            color = next((value for key, value in COLORS.items() if key in name.lower()), PALETTE[index % len(PALETTE)])
            x, y, r = left + (col + .5) * unit, top + (row + .5) * unit, max(3, unit * .38)
            canvas.create_polygon(x, y-r, x+r, y-r*.35, x+r*.7, y+r, x-r*.7, y+r, x-r, y-r*.35,
                                  fill="#596467" if lost else color, outline="#f5f5f5", dash=(2, 2) if lost else ())
            canvas.create_polygon(x, y-r, x, y+r*.3, x-r, y-r*.35, fill="#ffffff", stipple="gray50", outline="")
            if unit >= 19:
                canvas.create_text(x, y, text=name[0].upper(), fill="#13191c", font=("Helvetica", 9, "bold"))
            names.append(f"{name}: {position['cell']}" + (" (lost)" if lost else ""))
        lock = "Locked" if self.state.get("locked") else "No camera lock"
        self.caption.set(f"{cols} x {rows} | {lock}\n" + ("   ".join(names) or "No tracked positions"))
