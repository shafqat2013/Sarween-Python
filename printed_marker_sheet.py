"""Print cards at the physical size of Foundry's fullscreen LCD markers.

Marker cells are exported from OpenCV's actual 6x6/250 dictionary. ReportLab is
used when available; the small vector-only fallback keeps printing available
in the normal desktop runtime without adding a PDF dependency to tracking.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from app_paths import resource_path

MM = 72 / 25.4
PAGE = (210 * MM, 297 * MM)
IDS = (20, 21, 22, 23)
LABELS = {20: "TOP LEFT", 21: "TOP RIGHT", 22: "BOTTOM RIGHT", 23: "BOTTOM LEFT"}


def parameters_from_context(profile, context):
    """Read explicit TV size and current same-connection fullscreen geometry."""
    if not profile or not isinstance(profile.get("settings"), dict):
        raise ValueError("Choose a saved display profile before printing matching cards.")
    if not context.get("ready") or not context.get("view"):
        raise ValueError("Connect Foundry fullscreen on the selected TV before creating a matching print sheet.")
    view, settings = context["view"], context.get("settings") or {}
    parameters = {"diagonal_inches": profile["settings"].get("displayDiagonalInches"),
                  "viewport_width": view.get("viewportWidth"), "viewport_height": view.get("viewportHeight"),
                  "marker_size_px": settings.get("viewportMarkerSize"),
                  "marker_inset_px": settings.get("viewportMarkerInset")}
    dimensions(**parameters)
    size, inset = parameters["marker_size_px"], parameters["marker_inset_px"]
    expected = {"left": inset + size / 6, "top": inset + size / 6,
                "right": parameters["viewport_width"] - inset - size / 6,
                "bottom": parameters["viewport_height"] - inset - size / 6}
    actual = view.get("registration") or {}
    if any(abs(float(actual.get(key, float("inf"))) - value) > .1 for key, value in expected.items()):
        raise ValueError("Foundry marker settings and viewport disagree. Wait for the current view, then print again.")
    return parameters


def dimensions(diagonal_inches, viewport_width, viewport_height, marker_size_px, marker_inset_px=8):
    values = (diagonal_inches, viewport_width, viewport_height, marker_size_px, marker_inset_px)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("Marker printing needs finite display dimensions and marker settings.")
    if not (10 <= diagonal_inches <= 150 and min(viewport_width, viewport_height) >= 100
            and 72 <= marker_size_px <= 240 and 0 <= marker_inset_px <= 80):
        raise ValueError("Choose a display profile and valid fullscreen marker settings.")
    per_pixel = diagonal_inches * 25.4 / math.hypot(viewport_width, viewport_height)
    plate = marker_size_px * per_pixel
    if plate > 170:
        raise ValueError("These markers are too large for A4. Reduce their onscreen size before printing.")
    return {"diagonalInches": float(diagonal_inches), "viewportWidth": int(viewport_width),
            "viewportHeight": int(viewport_height), "markerSizePx": float(marker_size_px),
            "markerInsetPx": float(marker_inset_px), "plateMm": plate,
            "blackSquareMm": plate * 2 / 3, "whiteBorderMm": plate / 6,
            "screenInsetMm": marker_inset_px * per_pixel}


def _cells():
    value = json.loads(Path(resource_path("assets/printed_markers/dictionary-cells.json")).read_text())
    if value.get("dictionary") != "DICT_6X6_250" or set(value.get("markers", {})) != set(map(str, IDS)):
        raise ValueError("The printable marker dictionary is unavailable.")
    for rows in value["markers"].values():
        if len(rows) != 8 or any(len(row) != 8 or any(type(c) is not int or c not in (0, 1) for c in row) for row in rows):
            raise ValueError("The printable marker cells are invalid.")
    return value["markers"]


class _VectorPDF:
    """Minimal uncompressed PDF canvas for text and opaque vector rectangles."""
    def __init__(self, path, pagesize):
        self.path, self.size = Path(path), pagesize
        self.pages, self.commands = [], []

    def setFont(self, name, size):
        self.font = (name, size)

    def setFillColorRGB(self, r, g, b):
        self.commands.append(f"{r:g} {g:g} {b:g} rg")

    def setStrokeColorRGB(self, r, g, b):
        self.commands.append(f"{r:g} {g:g} {b:g} RG")

    def setLineWidth(self, width):
        self.commands.append(f"{width:g} w")

    def rect(self, x, y, width, height, stroke=0, fill=0):
        self.commands.append(f"{x:.5f} {y:.5f} {width:.5f} {height:.5f} re " + ("B" if stroke and fill else "f" if fill else "S"))

    def line(self, x, y, end_x, end_y):
        self.commands.append(f"{x:.5f} {y:.5f} m {end_x:.5f} {end_y:.5f} l S")

    def drawString(self, x, y, text):
        safe = str(text).encode("ascii", "replace").decode().replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        name, size = self.font
        self.commands.append(f"BT /{'F2' if 'Bold' in name else 'F1'} {size:g} Tf 1 0 0 1 {x:.5f} {y:.5f} Tm ({safe}) Tj ET")

    def showPage(self):
        self.pages.append("\n".join(self.commands).encode("ascii"))
        self.commands = []

    def save(self):
        if self.commands:
            self.showPage()
        kids = " ".join(f"{5 + i * 2} 0 R" for i in range(len(self.pages)))
        objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
                   f"<< /Type /Pages /Kids [{kids}] /Count {len(self.pages)} >>".encode(),
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                   b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"]
        for i, content in enumerate(self.pages):
            objects.extend([f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.size[0]:.5f} {self.size[1]:.5f}] /Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> /Contents {6+i*2} 0 R >>".encode(),
                            f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"])
        output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, value in enumerate(objects, 1):
            offsets.append(len(output))
            output.extend(f"{i} 0 obj\n".encode() + value + b"\nendobj\n")
        xref = len(output)
        output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
        output.extend("".join(f"{position:010d} 00000 n \n" for position in offsets[1:]).encode())
        output.extend(f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        self.path.write_bytes(output)


def build_sheet(path, *, diagonal_inches, viewport_width, viewport_height,
                marker_size_px, marker_inset_px=8, engine="auto"):
    spec = dimensions(diagonal_inches, viewport_width, viewport_height, marker_size_px, marker_inset_px)
    cells = _cells()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if engine not in {"auto", "reportlab", "vector"}:
        raise ValueError("Unknown printable sheet writer.")
    canvas_type = _VectorPDF
    if engine != "vector":
        try:
            from reportlab.pdfgen.canvas import Canvas
            canvas_type = Canvas
        except ImportError:
            if engine == "reportlab":
                raise
    canvas = canvas_type(str(path), pagesize=PAGE)
    plate = spec["plateMm"]
    groups = [IDS] if plate <= 78 else [(identity,) for identity in IDS]
    for identities in groups:
        canvas.setFillColorRGB(0, 0, 0)
        canvas.setFont("Helvetica-Bold", 20)
        canvas.drawString(18 * MM, 280 * MM, "Sarween printed corner markers")
        canvas.setFont("Helvetica", 10)
        lines = [f"Matches {diagonal_inches:g}-inch LCD | fullscreen {viewport_width} x {viewport_height} | {marker_size_px:g}px markers",
                 f"Paper card {plate:.2f}mm | black square {spec['blackSquareMm']:.2f}mm | dictionary 6x6/250",
                 "Print A4 at 100% / Actual Size. Turn off Fit to Page. Check the ruler below.",
                 "Use matte paper. Cut along the outline and keep the complete white border.",
                 "Place each card over its matching screen guide, facing the camera, then align in Sarween."]
        for i, line in enumerate(lines):
            canvas.drawString(18 * MM, (268 - i * 6) * MM, line)
        positions = {20: (20, 144), 21: (190 - plate, 144),
                     23: (20, 51), 22: (190 - plate, 51)}
        for identity in identities:
            x, y = positions[identity] if len(identities) == 4 else ((210 - plate) / 2, 140 - plate / 2)
            canvas.setFillColorRGB(0, 0, 0)
            canvas.setFont("Helvetica-Bold", 11)
            canvas.drawString(x * MM, (y + plate + 4) * MM, f"{identity} - {LABELS[identity]}")
            canvas.setStrokeColorRGB(.65, .65, .65)
            canvas.setLineWidth(.25)
            canvas.rect(x * MM, y * MM, plate * MM, plate * MM, stroke=1, fill=0)
            black, quiet = spec["blackSquareMm"], spec["whiteBorderMm"]
            module = black / 8
            canvas.setFillColorRGB(0, 0, 0)
            for row, columns in enumerate(cells[str(identity)]):
                for col, white in enumerate(columns):
                    if not white:
                        canvas.rect((x + quiet + col * module) * MM,
                                    (y + quiet + (7-row) * module) * MM,
                                    module * MM, module * MM, stroke=0, fill=1)
        canvas.setFillColorRGB(0, 0, 0)
        canvas.setStrokeColorRGB(0, 0, 0)
        canvas.setLineWidth(.6)
        canvas.line(20 * MM, 25 * MM, 70 * MM, 25 * MM)
        for x in (20, 70):
            canvas.line(x * MM, 23 * MM, x * MM, 27 * MM)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(20 * MM, 18 * MM, "This line must measure exactly 50mm after printing.")
        canvas.drawString(20 * MM, 10 * MM, "If the fullscreen viewport, display size or marker size changes, generate a new sheet.")
        canvas.showPage()
    canvas.save()
    return {**spec, "path": str(path.resolve()), "pages": len(groups), "markerIds": list(IDS)}
