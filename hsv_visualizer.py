#!/usr/bin/env python3
"""
hsv_visualizer.py — Standalone HSV range visualizer for Sarween band profiles.

Loads band_profiles.json from the same directory and shows an OpenCV window
visualizing each band's HSV range, with overlapping hue regions highlighted in red.

No camera, no ArUco, no CV pipeline required.

Usage:
    python hsv_visualizer.py
    python hsv_visualizer.py --profiles /path/to/band_profiles.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Data structures (duplicated here so this file is fully standalone)
# ──────────────────────────────────────────────────────────────────────────────

class HSVRange:
    def __init__(self, lower, upper):
        self.lower = tuple(int(x) for x in lower)
        self.upper = tuple(int(x) for x in upper)

class ColorProfile:
    def __init__(self, ranges: List[HSVRange], expected_diameter_squares: float = 1.0):
        self.ranges = ranges
        self.expected_diameter_squares = expected_diameter_squares


# ──────────────────────────────────────────────────────────────────────────────
# Profile loading
# ──────────────────────────────────────────────────────────────────────────────

def load_profiles(path: Path) -> Dict[str, ColorProfile]:
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        print(f"ERROR: {path} is empty or not a JSON object.")
        sys.exit(1)

    out: Dict[str, ColorProfile] = {}
    for color_name, cfg in data.items():
        if not isinstance(cfg, dict):
            continue
        ranges_cfg = cfg.get("ranges")
        if not isinstance(ranges_cfg, list) or not ranges_cfg:
            continue
        ranges = []
        for r in ranges_cfg:
            lo = r.get("lower")
            hi = r.get("upper")
            if isinstance(lo, list) and len(lo) == 3 and isinstance(hi, list) and len(hi) == 3:
                ranges.append(HSVRange(lo, hi))
        if not ranges:
            continue
        exp = float(cfg.get("expected_diameter_squares", 1.0))
        out[str(color_name)] = ColorProfile(ranges=ranges, expected_diameter_squares=exp)

    if not out:
        print(f"ERROR: No valid profiles found in {path}.")
        sys.exit(1)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────────────

def render(profiles: Dict[str, ColorProfile], width: int = 1280) -> np.ndarray:
    BAR_H    = 48
    SWATCH_W = 64
    LABEL_W  = 120
    ROW_PAD  = 12
    HEADER_H = 56
    FOOTER_H = 36
    FONT     = cv2.FONT_HERSHEY_SIMPLEX

    bar_w = width - LABEL_W - SWATCH_W - 40

    color_order = list(profiles.keys())
    n_rows  = len(color_order)
    total_h = HEADER_H + n_rows * (BAR_H + ROW_PAD) + FOOTER_H

    canvas = np.zeros((total_h, width, 3), dtype=np.uint8)
    canvas[:] = (28, 28, 28)

    # ── Header: full hue spectrum strip ──────────────────────────────────────
    bar_x0 = LABEL_W + SWATCH_W + 20
    for x in range(bar_w):
        h = int(x / bar_w * 179)
        bgr = cv2.cvtColor(np.array([[[h, 220, 210]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0][0]
        canvas[10:10+28, bar_x0 + x] = bgr.tolist()
    # Tick marks + labels every 30 hue units
    for hval in range(0, 180, 30):
        x = int(hval / 179 * (bar_w - 1)) + bar_x0
        cv2.line(canvas, (x, 8), (x, 40), (220, 220, 220), 1)
        cv2.putText(canvas, str(hval), (x - 8, HEADER_H - 4),
                    FONT, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Hue (0–179)", (bar_x0, HEADER_H - 4),
                FONT, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    # ── Build overlap map ─────────────────────────────────────────────────────
    hue_owners: List[set] = [set() for _ in range(180)]
    for name, prof in profiles.items():
        for r in prof.ranges:
            for h in range(max(0, r.lower[0]), min(179, r.upper[0]) + 1):
                hue_owners[h].add(name)

    # ── One row per band ──────────────────────────────────────────────────────
    for row_i, name in enumerate(color_order):
        prof = profiles[name]
        y0 = HEADER_H + row_i * (BAR_H + ROW_PAD)
        y1 = y0 + BAR_H

        # Grey base bar
        canvas[y0:y1, bar_x0: bar_x0 + bar_w] = (50, 50, 50)

        # Swatch from median of first range
        r0 = prof.ranges[0]
        med_h = (r0.lower[0] + r0.upper[0]) // 2
        med_s = (r0.lower[1] + r0.upper[1]) // 2
        med_v = (r0.lower[2] + r0.upper[2]) // 2
        swatch_bgr = cv2.cvtColor(
            np.array([[[med_h, med_s, med_v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
        )[0][0].tolist()
        sw_x = LABEL_W + 6
        canvas[y0 + 5: y1 - 5, sw_x: sw_x + SWATCH_W - 10] = swatch_bgr
        cv2.rectangle(canvas, (sw_x, y0 + 5), (sw_x + SWATCH_W - 11, y1 - 6), (200, 200, 200), 1)

        # Band name label
        cv2.putText(canvas, name, (8, y0 + BAR_H // 2 + 6),
                    FONT, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        # Hue range bars
        for r in prof.ranges:
            lo_h, hi_h = r.lower[0], r.upper[0]
            lo_s, hi_s = r.lower[1], r.upper[1]
            lo_v, hi_v = r.lower[2], r.upper[2]

            x_lo = int(lo_h / 179 * (bar_w - 1)) + bar_x0
            x_hi = int(hi_h / 179 * (bar_w - 1)) + bar_x0

            for x in range(x_lo, min(x_hi + 1, bar_x0 + bar_w)):
                hpx = int((x - bar_x0) / bar_w * 179)
                if len(hue_owners[hpx]) > 1:
                    bgr = (30, 30, 220)   # red = overlap
                else:
                    bgr = cv2.cvtColor(
                        np.array([[[hpx, med_s, med_v]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
                    )[0][0].tolist()
                canvas[y0 + 3: y1 - 3, x] = bgr

            # White border around range
            cv2.rectangle(canvas, (x_lo, y0 + 3), (x_hi, y1 - 4), (255, 255, 255), 1)

            # S/V info to right of bar
            sv_text = f"S: {lo_s}–{hi_s}   V: {lo_v}–{hi_v}"
            tx = bar_x0 + bar_w + 8
            if tx + 180 < width:
                cv2.putText(canvas, sv_text, (tx, y0 + BAR_H // 2 + 5),
                            FONT, 0.40, (160, 160, 160), 1, cv2.LINE_AA)

    # ── Footer ────────────────────────────────────────────────────────────────
    fy = total_h - FOOTER_H + 22
    cv2.putText(canvas, "RED = overlapping hue ranges   |   Press Q or Esc to quit",
                (LABEL_W, fy), FONT, 0.44, (100, 100, 200), 1, cv2.LINE_AA)

    n = len(color_order)
    cv2.putText(canvas, f"{n} band{'s' if n != 1 else ''} loaded",
                (8, fy), FONT, 0.44, (120, 120, 120), 1, cv2.LINE_AA)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sarween HSV band profile visualizer")
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(__file__).with_name("band_profiles.json"),
        help="Path to band_profiles.json (default: same directory as this script)",
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Window width in pixels (default: 1280)",
    )
    args = parser.parse_args()

    profiles = load_profiles(args.profiles)
    print(f"Loaded {len(profiles)} profile(s): {', '.join(profiles.keys())}")

    img = render(profiles, width=args.width)

    win = "HSV Band Profiles"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.width, img.shape[0])
    cv2.imshow(win, img)

    print("Press Q or Esc to quit.")
    while True:
        k = cv2.waitKey(100) & 0xFF
        # Re-render on each loop in case window was resized
        try:
            _, _, ww, wh = cv2.getWindowImageRect(win)
            if ww > 100:
                img = render(profiles, width=ww)
                cv2.imshow(win, img)
        except Exception:
            pass
        if k in (ord('q'), ord('Q'), 27):  # Q or Esc
            break
        # Check if window was closed
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()