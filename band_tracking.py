# band_tracking.py
import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HSVRange:
    lower: Tuple[int, int, int]  # (H,S,V)
    upper: Tuple[int, int, int]

@dataclass
class ColorProfile:
    # Some colors (notably red) may need two ranges
    ranges: List[HSVRange]
    expected_diameter_squares: float = 1.0

@dataclass
class BandDetection:
    color_name: str
    cx: float
    cy: float
    contour_area: float
    circularity: float
    ellipse_eccentricity: Optional[float]
    score: float
    bbox: Tuple[int, int, int, int]  # x,y,w,h in WARP space

# ──────────────────────────────────────────────────────────────────────────────
# Core helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mask_for_profile(hsv_img: np.ndarray, profile: ColorProfile) -> np.ndarray:
    mask = None
    for r in profile.ranges:
        m = cv2.inRange(
            hsv_img,
            np.array(r.lower, dtype=np.uint8),
            np.array(r.upper, dtype=np.uint8),
        )
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    if mask is None:
        mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    return mask

def _cleanup_mask(mask: np.ndarray, grid_px: float) -> np.ndarray:
    # scale morphology with warp scale
    k = max(3, int(round(grid_px * 0.08)) | 1)  # ~8% of a square, odd
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return mask

def _contour_centroid(cnt: np.ndarray) -> Optional[Tuple[float, float]]:
    M = cv2.moments(cnt)
    if abs(M["m00"]) < 1e-6:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return float(cx), float(cy)

def _contour_circularity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    if peri <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (peri * peri))

def _ellipse_eccentricity(cnt: np.ndarray) -> Optional[float]:
    if len(cnt) < 5:
        return None
    ellipse = cv2.fitEllipse(cnt)
    (cx, cy), (MA, ma), angle = ellipse
    a = max(MA, ma) / 2.0
    b = min(MA, ma) / 2.0
    if a <= 1e-6:
        return None
    e = math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))
    return float(e)

def _score_candidate(
    area: float,
    circularity: float,
    ecc: Optional[float],
    expected_area: float,
    prev_xy: Optional[Tuple[float, float]],
    xy: Tuple[float, float],
    grid_px: float,
) -> float:
    # area closeness (log scale-ish)
    area_ratio = area / max(1.0, expected_area)
    area_term = math.exp(-abs(math.log(max(1e-6, area_ratio))) * 1.2)

    # circularity preference
    circ_term = max(0.0, min(1.0, (circularity - 0.25) / 0.55))

    # eccentricity: tolerate some tilt
    ecc_term = 1.0
    if ecc is not None:
        ecc_term = math.exp(-max(0.0, ecc - 0.55) * 3.0)

    # distance to previous (prevents jumping to noise)
    dist_term = 1.0
    if prev_xy is not None:
        dx = xy[0] - prev_xy[0]
        dy = xy[1] - prev_xy[1]
        d = math.hypot(dx, dy)
        dist_term = math.exp(-d / max(1.0, grid_px * 1.2))

    return float(0.45 * area_term + 0.35 * circ_term + 0.15 * ecc_term + 0.05 * dist_term)

def warp_centroid_to_cell(cx: float, cy: float, grid_w: int, grid_h: int, warp_w: int, warp_h: int) -> Tuple[int, int]:
    cell_w = warp_w / float(grid_w)
    cell_h = warp_h / float(grid_h)
    col = int(np.clip(cx / cell_w, 0, grid_w - 1))
    row = int(np.clip(cy / cell_h, 0, grid_h - 1))
    return col, row

def warp_point_to_cam(cx: float, cy: float, H_inv: np.ndarray) -> Tuple[int, int]:
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    cam_pt = cv2.perspectiveTransform(pt, H_inv).reshape(-1)
    return int(round(cam_pt[0])), int(round(cam_pt[1]))

def warp_bbox_to_cam_poly(bbox_xywh: Tuple[int, int, int, int], H_inv: np.ndarray) -> np.ndarray:
    x, y, w, h = bbox_xywh
    pts = np.array([[[x, y],
                     [x + w, y],
                     [x + w, y + h],
                     [x, y + h]]], dtype=np.float32)
    cam_pts = cv2.perspectiveTransform(pts, H_inv).astype(np.int32)
    return cam_pts.reshape(-1, 2)

# ──────────────────────────────────────────────────────────────────────────────
# Main detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_bands(
    warp_bgr: np.ndarray,
    color_profiles: Dict[str, ColorProfile],
    grid_px: float,
    prev_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Optional[BandDetection]], Dict[str, Any]]:
    if prev_state is None:
        prev_state = {}

    hsv = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2HSV)

    detections: Dict[str, Optional[BandDetection]] = {}
    new_state: Dict[str, Any] = {}

    expected_radius = 0.5 * grid_px
    expected_area = math.pi * expected_radius * expected_radius * 0.55

    for color_name, profile in color_profiles.items():
        prev_xy = None
        if color_name in prev_state:
            prev_xy = prev_state[color_name].get("last_xy")

        mask = _mask_for_profile(hsv, profile)
        mask = _cleanup_mask(mask, grid_px)

        cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

        best: Optional[BandDetection] = None

        # size gates in warp px relative to grid square
        min_area = (grid_px * grid_px) * 0.12
        max_area = (grid_px * grid_px) * 1.60

        for cnt in cnts:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue

            cxy = _contour_centroid(cnt)
            if cxy is None:
                continue
            cx, cy = cxy

            circ = _contour_circularity(cnt)
            if circ < 0.18:
                continue

            ecc = _ellipse_eccentricity(cnt)
            if ecc is not None and ecc > 0.92:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            ar = max(w, h) / max(1.0, min(w, h))
            if ar > 3.2:
                continue

            score = _score_candidate(
                area=area,
                circularity=circ,
                ecc=ecc,
                expected_area=expected_area * (profile.expected_diameter_squares ** 2),
                prev_xy=prev_xy,
                xy=(cx, cy),
                grid_px=grid_px,
            )

            cand = BandDetection(
                color_name=color_name,
                cx=cx, cy=cy,
                contour_area=area,
                circularity=circ,
                ellipse_eccentricity=ecc,
                score=score,
                bbox=(int(x), int(y), int(w), int(h)),
            )

            if best is None or cand.score > best.score:
                best = cand

        detections[color_name] = best
        if best is not None:
            new_state[color_name] = {"last_xy": (best.cx, best.cy), "last_score": float(best.score)}
        else:
            new_state[color_name] = {"last_xy": None, "last_score": 0.0}

    return detections, new_state
