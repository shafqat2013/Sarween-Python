import cv2
import os
import csv
import json
import numpy as np
import time
import setup as s
from datetime import datetime

# ── Canonical DB path ─────────────────────────────────────────────────────────
DB_CSV = os.path.abspath(os.path.join("mini_captures", "..", "mini_database.csv"))

# ── Lazy camera calibration (avoid I/O at import) ─────────────────────────────
_camera_params_cache = None
def get_camera_params():
    """
    Returns (camera_matrix, dist_coeffs), loading once on first use.
    """
    global _camera_params_cache
    if _camera_params_cache is None:
        _camera_params_cache = (
            np.load("camera_matrix.npy"),
            np.load("dist_coeffs.npy")
        )
    return _camera_params_cache


# ──────────────────────────────────────────────────────────────────────────────
# DB schema helpers (mini_id + view_id + name) with auto-migration
# ──────────────────────────────────────────────────────────────────────────────
DB_HEADER_V3 = [
    "mini_id", "view_id", "name",
    "id", "timestamp",
    "image", "mask", "contour_npy", "hist_npy",
    "area", "perimeter", "cx", "cy",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "aspect_ratio", "circularity", "hull_area", "solidity"
]

def _read_csv_header(path):
    try:
        with open(path, "r", newline="") as f:
            first = f.readline()
        if not first:
            return None
        return [h.strip() for h in first.strip().split(",")]
    except Exception:
        return None

def _migrate_db_to_v3_if_needed(db_csv_path=DB_CSV):
    """
    Ensure DB has mini_id/view_id/name columns.
    Supports migration from:
      - legacy (no mini_id/view_id): set mini_id=view_id=id, name=""
      - v2 (mini_id/view_id but no name): keep ids, add name=""
    """
    if not os.path.exists(db_csv_path):
        return

    header = _read_csv_header(db_csv_path)
    if not header:
        return

    has_mini = ("mini_id" in header and "view_id" in header)
    has_name = ("name" in header)

    # Already v3
    if has_mini and has_name:
        return

    rows = []
    try:
        with open(db_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        print(f"⚠️ DB migrate read failed: {e}")
        return

    tmp_path = db_csv_path + ".tmp_v3"
    try:
        with open(tmp_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DB_HEADER_V3)
            writer.writeheader()

            for r in rows:
                rid = (r.get("id") or r.get("timestamp") or "").strip()

                if has_mini:
                    mini_id = (r.get("mini_id") or rid).strip()
                    view_id = (r.get("view_id") or rid).strip()
                else:
                    # legacy: old rows become mini_id=id, view_id=id
                    mini_id = rid
                    view_id = rid

                name = (r.get("name") or "").strip() if has_name else ""

                out = {
                    "mini_id": mini_id,
                    "view_id": view_id,
                    "name": name,

                    "id": r.get("id", rid),
                    "timestamp": r.get("timestamp", rid),
                    "image": r.get("image", ""),
                    "mask": r.get("mask", ""),
                    "contour_npy": r.get("contour_npy", ""),
                    "hist_npy": r.get("hist_npy", ""),
                    "area": r.get("area", ""),
                    "perimeter": r.get("perimeter", ""),
                    "cx": r.get("cx", ""),
                    "cy": r.get("cy", ""),
                    "bbox_x": r.get("bbox_x", ""),
                    "bbox_y": r.get("bbox_y", ""),
                    "bbox_w": r.get("bbox_w", ""),
                    "bbox_h": r.get("bbox_h", ""),
                    "aspect_ratio": r.get("aspect_ratio", ""),
                    "circularity": r.get("circularity", ""),
                    "hull_area": r.get("hull_area", ""),
                    "solidity": r.get("solidity", ""),
                }
                writer.writerow(out)

        os.replace(tmp_path, db_csv_path)
        print(f"ℹ️ Migrated mini DB to v3 schema (adds name + ensures mini_id/view_id): {db_csv_path}")
    except Exception as e:
        print(f"⚠️ DB migrate write failed: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _append_db_row_v3(db_csv_path, row_dict):
    _migrate_db_to_v3_if_needed(db_csv_path)
    write_header = not os.path.exists(db_csv_path)
    os.makedirs(os.path.dirname(db_csv_path), exist_ok=True)
    try:
        with open(db_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DB_HEADER_V3)
            if write_header:
                writer.writeheader()
            writer.writerow(row_dict)
    except Exception as e:
        print(f"⚠️ Failed to append to CSV: {e}")

def _next_mini_id(db_csv_path=DB_CSV):
    """
    Returns next numeric mini_id as zero-padded 6-digit string: 000001, 000002, ...
    """
    _migrate_db_to_v3_if_needed(db_csv_path)
    if not os.path.exists(db_csv_path):
        return "000001"

    max_id = 0
    try:
        with open(db_csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mid = (row.get("mini_id") or "").strip()
                if mid.isdigit():
                    max_id = max(max_id, int(mid))
    except Exception:
        # If read fails, fall back safely
        return "000001"

    return f"{max_id + 1:06d}"

def set_mini_name(mini_id, name, db_csv_path=DB_CSV):
    """
    Update name for all rows with given mini_id (rewrite CSV).
    """
    _migrate_db_to_v3_if_needed(db_csv_path)
    if not os.path.exists(db_csv_path):
        return False

    name = (name or "").strip()

    tmp_path = db_csv_path + ".tmp_rename"
    changed = False
    try:
        with open(db_csv_path, "r", newline="") as f_in:
            reader = csv.DictReader(f_in)
            rows = list(reader)
            header = reader.fieldnames or DB_HEADER_V3

        # Ensure header matches v3
        header = DB_HEADER_V3

        with open(tmp_path, "w", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=header)
            writer.writeheader()
            for r in rows:
                r_mid = (r.get("mini_id") or "").strip()
                if r_mid == str(mini_id):
                    r["name"] = name
                    changed = True
                else:
                    if "name" not in r:
                        r["name"] = ""
                writer.writerow({k: r.get(k, "") for k in header})

        os.replace(tmp_path, db_csv_path)
        return changed
    except Exception as e:
        print(f"⚠️ Failed to set mini name: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return False


# ── Background capture helpers ────────────────────────────────────────────────
def capture_background(camera_index=None):
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]
    camera_matrix, dist_coeffs = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background (blur)...")
    time.sleep(1)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ Could not capture background.")
        return None
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur

def capture_background_full(camera_index=None):
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]
    camera_matrix, dist_coeffs = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background (bgr+blur)...")
    time.sleep(1)
    for _ in range(10):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ Could not capture background.")
        return None
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21,21), 0)
    return {"bgr": frame, "blur": blur}


# ── Shadow-aware mask (ignores shadows) ───────────────────────────────────────
def shadow_free_mask(bg_bgr, live_bgr,
                     h_thresh=12, s_thresh=25, v_shadow_drop=45,
                     k_open=3, k_close=5):
    """
    Return a binary mask (uint8 0/255) of foreground that ignores shadows.
    """
    bg_hsv   = cv2.cvtColor(bg_bgr,   cv2.COLOR_BGR2HSV)
    live_hsv = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2HSV)

    Hb,Sb,Vb = cv2.split(bg_hsv)
    Hl,Sl,Vl = cv2.split(live_hsv)

    dH = cv2.absdiff(Hl, Hb)
    dH = np.minimum(dH.astype(np.uint16), (180 - dH).astype(np.uint16)).astype(np.uint8)
    dS = cv2.absdiff(Sl, Sb)

    v_darker   = cv2.subtract(Vb, Vl)
    v_brighter = cv2.subtract(Vl, Vb)

    chroma_change   = (dH > h_thresh) | (dS > s_thresh)
    strong_brighten = (v_brighter > (v_shadow_drop + 10))
    shadow          = (v_darker  > v_shadow_drop) & (dH <= h_thresh) & (dS <= s_thresh)

    fg = (chroma_change | strong_brighten) & (~shadow)
    mask = (fg.astype(np.uint8) * 255)

    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_o, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c, iterations=1)
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Mask combining (used by tracking.py)
# ──────────────────────────────────────────────────────────────────────────────
def combine_masks_componentwise(motion_mask, shadowfree_mask,
                                keep_ratio=0.25,
                                min_comp_area=400,
                                k_open=3, k_close=5):
    """
    Keep connected components from motion_mask if enough of their pixels also pass
    the shadow-free mask. Returns a clean binary mask (0/255).
    """
    mot = (motion_mask > 0).astype(np.uint8) * 255
    sh  = (shadowfree_mask > 0).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mot, connectivity=8)
    out = np.zeros_like(mot)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_comp_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(comp, sh)
        ratio = cv2.countNonZero(overlap) / float(area)
        if ratio >= keep_ratio:
            out = cv2.bitwise_or(out, comp)

    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k1, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k2, iterations=1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Histogram helpers (HS-only for lighting robustness)
# ──────────────────────────────────────────────────────────────────────────────
def flatten_hist_HS_from_HSV(hist_hsv):
    h = hist_hsv
    if h.ndim == 3:
        h = h.sum(axis=2)
    h = h.astype(np.float32)
    s_ = h.sum()
    if s_ > 0:
        h /= s_
    return h.ravel().reshape(-1, 1)

def contour_hist_HS(frame_bgr, contour, bins_h=32, bins_s=32):
    x, y, w, h = cv2.boundingRect(contour)
    roi_bgr = frame_bgr[y:y+h, x:x+w]
    if roi_bgr.size == 0:
        return None

    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tmp, [(contour - [x, y]).astype(np.int32)], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    roi_mask = cv2.erode(tmp, kernel, iterations=1)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist2 = cv2.calcHist([roi_hsv], [0, 1], roi_mask, [bins_h, bins_s], [0, 180, 0, 256])
    return flatten_hist_HS_from_HSV(hist2)


# ──────────────────────────────────────────────────────────────────────────────
# ORB helpers
# ──────────────────────────────────────────────────────────────────────────────
def compute_orb_descriptors(frame_bgr, contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi_bgr = frame_bgr[y:y+h, x:x+w]
    if roi_bgr.size == 0:
        return None

    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tmp, [(contour - [x, y]).astype(np.int32)], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    roi_mask = cv2.erode(tmp, kernel, iterations=1)

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(
        nfeatures=800, scaleFactor=1.2, nlevels=8,
        edgeThreshold=31, firstLevel=0, WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20
    )
    kp, des = orb.detectAndCompute(roi_gray, roi_mask)
    return des

def orb_similarity(desc_q, desc_db, ratio_thresh=0.75):
    """
    Lowe ratio-tested ORB match score in [0,1].
    """
    if desc_q is None or desc_db is None:
        return None, {"good": 0, "nq": 0, "ndb": 0}
    if len(desc_q) < 8 or len(desc_db) < 8:
        return None, {"good": 0, "nq": len(desc_q), "ndb": len(desc_db)}

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc_q, desc_db, k=2)

    good = 0
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good += 1

    denom = max(1, min(len(desc_q), len(desc_db)))
    score = np.clip(good / denom, 0.0, 1.0)
    return float(score), {"good": good, "nq": len(desc_q), "ndb": len(desc_db)}

def derive_orb_path_from_hist(hist_path):
    if hist_path.endswith("_hist.npy"):
        return hist_path[:-9] + "_orb.npz"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Save from frame+contour, supports multi-view via mini_id + optional name
# ──────────────────────────────────────────────────────────────────────────────
def save_mini_from_frame_and_contour(
    frame_bgr,
    contour,
    save_dir="mini_captures",
    min_area=1500,
    return_info=False,
    mini_id=None,
    name=None
):
    """
    Save a mini view from a provided frame + contour.

    If mini_id is None: creates a NEW numeric mini_id (000001...) and first view.
    If mini_id is provided: appends a NEW view under that mini_id.
    If name is provided and mini_id exists: the DB name for that mini_id is set/updated.
    """
    if frame_bgr is None or contour is None or len(contour) == 0:
        return None

    os.makedirs(save_dir, exist_ok=True)

    area = float(cv2.contourArea(contour))
    if area < float(min_area):
        print(f"⚠️ Selected contour too small (area={area:.0f} < {min_area}).")
        return None

    view_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # numeric mini_id (000001...) on first creation
    if mini_id is None:
        mini_id = _next_mini_id(DB_CSV)

    # Normalize name
    name = (name or "").strip()

    # features
    perimeter = float(cv2.arcLength(contour, True))
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
    else:
        cx = cy = -1.0

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 0.0
    circularity  = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    hull         = cv2.convexHull(contour)
    hull_area    = float(cv2.contourArea(hull)) if hull is not None and hull.size > 0 else 0.0
    solidity     = area / hull_area if hull_area > 0 else 0.0

    # mask from contour
    mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
    cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, thickness=-1)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # tighter mask for hist
    roi_mask_full = np.zeros_like(mask)
    cv2.drawContours(roi_mask_full, [contour.astype(np.int32)], -1, 255, thickness=-1)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    roi_mask_full = cv2.erode(roi_mask_full, kernel5, iterations=1)

    H_img, W_img = mask.shape[:2]
    x0 = max(0, int(x)); y0 = max(0, int(y))
    x1 = min(W_img, int(x + w)); y1 = min(H_img, int(y + h))
    if x1 <= x0 or y1 <= y0:
        print("⚠️ Invalid ROI bounds for selected contour.")
        return None

    roi_mask_tight = roi_mask_full[y0:y1, x0:x1]
    roi_bgr = frame_bgr[y0:y1, x0:x1]

    # HSV hist (3D, but we’ll collapse to HS at load time)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([roi_hsv], [0, 1, 2], roi_mask_tight, [32, 32, 8], [0, 180, 0, 256, 0, 256])
    hs = hist.sum()
    if hs > 0:
        hist = hist / hs

    # ORB
    orb_des = compute_orb_descriptors(frame_bgr, contour)

    base = os.path.join(save_dir, f"mini_{view_id}")
    img_path  = f"{base}.png"
    mask_path = f"{base}_mask.png"
    cnt_path  = f"{base}_contour.npy"
    hist_path = f"{base}_hist.npy"
    orb_path  = f"{base}_orb.npz"
    json_path = f"{base}.json"

    cv2.imwrite(img_path, frame_bgr)
    cv2.imwrite(mask_path, mask)
    np.save(cnt_path, contour)
    np.save(hist_path, hist)
    try:
        if orb_des is not None:
            np.savez_compressed(orb_path, des=orb_des.astype(np.uint8))
        else:
            np.savez_compressed(orb_path, des=np.empty((0, 32), dtype=np.uint8))
    except Exception as e:
        print(f"⚠️ Failed to save ORB descriptors: {e}")
        orb_path = ""

    meta = {
        "mini_id": mini_id,
        "name": name,
        "view_id": view_id,
        "id": view_id,
        "timestamp": view_id,
        "files": {
            "image": img_path,
            "mask": mask_path,
            "contour_npy": cnt_path,
            "hist_npy": hist_path,
            "orb_npz": orb_path
        },
        "features": {
            "area": area,
            "perimeter": perimeter,
            "centroid": {"x": cx, "y": cy},
            "bbox": {"x": int(x0), "y": int(y0), "w": int(x1-x0), "h": int(y1-y0)},
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "hull_area": hull_area,
            "solidity": solidity,
            "min_area_threshold": min_area
        }
    }
    try:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write JSON metadata: {e}")

    # If a name is provided, set/update it for the mini_id across the DB
    if name:
        set_mini_name(mini_id, name, DB_CSV)

    row_v3 = {
        "mini_id": str(mini_id),
        "view_id": view_id,
        "name": name,

        "id": view_id,
        "timestamp": view_id,
        "image": img_path,
        "mask": mask_path,
        "contour_npy": cnt_path,
        "hist_npy": hist_path,
        "area": f"{area:.4f}",
        "perimeter": f"{perimeter:.4f}",
        "cx": f"{cx:.2f}",
        "cy": f"{cy:.2f}",
        "bbox_x": int(x0),
        "bbox_y": int(y0),
        "bbox_w": int(x1-x0),
        "bbox_h": int(y1-y0),
        "aspect_ratio": f"{aspect_ratio:.4f}",
        "circularity": f"{circularity:.4f}",
        "hull_area": f"{hull_area:.4f}",
        "solidity": f"{solidity:.4f}",
    }
    _append_db_row_v3(DB_CSV, row_v3)

    print(f"✅ Mini saved -> {img_path}")
    print(f"   • mini_id -> {mini_id}")
    print(f"   • name    -> {name if name else '(none)'}")
    print(f"   • view_id -> {view_id}")
    print(f"   • DB row ->  {DB_CSV}")

    if not return_info:
        return mini_id

    return {
        "mini_id": mini_id,
        "name": name,
        "view_id": view_id,
        "id": view_id,
        "image": img_path,
        "mask": mask_path,
        "contour_npy": cnt_path,
        "hist_npy": hist_path,
        "json": json_path,
        "csv": DB_CSV,
        "features": meta["features"],
        "orb_npz": orb_path
    }


# ──────────────────────────────────────────────────────────────────────────────
# Legacy capture (largest blob from a fresh camera read)
# (kept for compatibility; not used by tracking.py's picker)
# ──────────────────────────────────────────────────────────────────────────────
def capture_mini(
    camera_index=None,
    background_blur=None,
    background_bgr=None,
    save_dir="mini_captures",
    min_area=1500,
    return_info=False,
    mini_id=None,
    name=None
):
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]

    camera_matrix, dist_coeffs = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    for _ in range(5):
        cap.read()
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret or frame_bgr is None:
        print("❌ Could not capture mini.")
        return None

    frame_bgr = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    if background_blur is not None:
        diff = cv2.absdiff(background_blur, blur)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if background_bgr is not None:
        sf = shadow_free_mask(background_bgr, frame_bgr)
        mask_comb = combine_masks_componentwise(mask, sf, keep_ratio=0.35, min_comp_area=400)
        if cv2.countNonZero(mask_comb) > 0:
            mask = mask_comb

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        print("⚠️ No contours found.")
        return None

    contour = max(cnts, key=cv2.contourArea)
    return save_mini_from_frame_and_contour(
        frame_bgr=frame_bgr,
        contour=contour,
        save_dir=save_dir,
        min_area=min_area,
        return_info=return_info,
        mini_id=mini_id,
        name=name
    )


# ──────────────────────────────────────────────────────────────────────────────
# DB loader (HS-collapsed hist + ORB + name)
# ──────────────────────────────────────────────────────────────────────────────
def load_mini_database(db_csv_path=DB_CSV):
    entries = []
    if not os.path.exists(db_csv_path):
        print(f"ℹ️ DB not found at {db_csv_path}.")
        return entries

    _migrate_db_to_v3_if_needed(db_csv_path)

    with open(db_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("id") or row.get("timestamp") or "").strip()
            mini_id = (row.get("mini_id") or rid).strip()
            view_id = (row.get("view_id") or rid).strip()
            name = (row.get("name") or "").strip()

            try:
                hist_hsv = np.load(row["hist_npy"])
                hist_hs_flat = flatten_hist_HS_from_HSV(hist_hsv)
            except Exception as e:
                print(f"⚠️ Skip {mini_id}/{view_id} (hist load failed): {e}")
                continue

            contour = None
            try:
                contour = np.load(row["contour_npy"], allow_pickle=False)
            except Exception:
                pass

            orb_path = derive_orb_path_from_hist(row["hist_npy"])
            orb_des = None
            if orb_path and os.path.exists(orb_path):
                try:
                    data = np.load(orb_path, allow_pickle=False)
                    orb_des = data["des"]
                    if orb_des.size == 0:
                        orb_des = None
                except Exception:
                    orb_des = None

            entries.append({
                "mini_id": mini_id,
                "view_id": view_id,
                "name": name,
                "hist_hs_flat": hist_hs_flat,
                "contour": contour,
                "orb_des": orb_des
            })
    return entries


# ──────────────────────────────────────────────────────────────────────────────
# Identify minis: match_id is stable mini_id; include name when known
# ──────────────────────────────────────────────────────────────────────────────
def identify_minis(
    live_frame_bgr,
    diff_mask=None,
    live_blur=None,
    background_blur=None,
    db_entries=None,
    db_csv_path=DB_CSV,
    min_area=1500,
    known_threshold=0.50,
    min_fill=0.35,
    min_solidity=0.85,
    keep_top_k=3,
    skip_border=True,
    draw=True,
    show_components=True
):
    if db_entries is None:
        db_entries = load_mini_database(db_csv_path)

    if diff_mask is None:
        if live_blur is not None and background_blur is not None:
            diff = cv2.absdiff(background_blur, live_blur)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(live_frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            _, diff_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # CHANGED: remove extra erosion/dilation here; use the provided diff_mask directly.
    fg = (diff_mask > 0).astype(np.uint8) * 255

    cnts_info = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep_top_k]

    detections = []
    annotated = live_frame_bgr.copy() if draw else None
    H, W = fg.shape[:2]

    def shape_similarity(c1, c2):
        if c1 is None or c2 is None:
            return None
        d = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)
        return max(0.0, 1.0 - min(d, 1.0)), float(d)

    def hist_similarity_flat(h1_flat, h2_flat):
        chisq = cv2.compareHist(h1_flat, h2_flat, cv2.HISTCMP_CHISQR)
        bh    = cv2.compareHist(h1_flat, h2_flat, cv2.HISTCMP_BHATTACHARYYA)
        corr  = cv2.compareHist(h1_flat, h2_flat, cv2.HISTCMP_CORREL)
        chisq_sim = 1.0 / (1.0 + min(chisq, 10.0))
        bh_sim    = max(0.0, 1.0 - min(bh, 1.0))
        corr_sim  = (corr + 1.0) / 2.0
        return 0.50*chisq_sim + 0.30*bh_sim + 0.20*corr_sim, {
            "chisq": float(chisq), "bhatt": float(bh), "corr": float(corr)
        }

    for contour in cnts:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if skip_border and (x <= 0 or y <= 0 or x + w >= W - 1 or y + h >= H - 1):
            continue

        fill = area / float(w * h) if w > 0 and h > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and hull.size > 0 else 0.0
        solidity = area / hull_area if hull_area > 0 else 0.0
        if fill < min_fill or solidity < min_solidity:
            continue

        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = cy = -1.0

        q_hist_hs_flat = contour_hist_HS(live_frame_bgr, contour)
        q_orb_des = compute_orb_descriptors(live_frame_bgr, contour)

        best = {
            "score": -1.0,
            "mini_id": None,
            "view_id": None,
            "name": "",
            "shape_dist": None,
            "hist_metrics": None,
            "orb_metrics": None,
            "h_sim": None,
            "o_sim": None,
            "s_sim": None
        }

        for entry in db_entries:
            h_sim, h_metrics = hist_similarity_flat(q_hist_hs_flat, entry["hist_hs_flat"]) if q_hist_hs_flat is not None else (0.0, None)

            s_sim = None; s_dist = None
            if entry.get("contour") is not None and entry["contour"] is not None and getattr(entry["contour"], "size", 0) > 0:
                s_pair = shape_similarity(contour, entry["contour"])
                if s_pair is not None:
                    s_sim, s_dist = s_pair

            o_sim = None; o_metrics = None
            if q_orb_des is not None and entry.get("orb_des") is not None:
                o_sim, o_metrics = orb_similarity(q_orb_des, entry["orb_des"])

            final = 0.33 * h_sim \
                  + 0.33 * (o_sim if o_sim is not None else h_sim) \
                  + 0.33 * (s_sim if s_sim is not None else h_sim)

            if final > best["score"]:
                best.update({
                    "score": float(final),
                    "mini_id": entry.get("mini_id"),
                    "view_id": entry.get("view_id"),
                    "name": (entry.get("name") or "").strip(),
                    "shape_dist": s_dist,
                    "hist_metrics": h_metrics,
                    "orb_metrics": o_metrics,
                    "h_sim": h_sim, "o_sim": o_sim, "s_sim": s_sim
                })

        is_known = best["score"] >= known_threshold and best["mini_id"] is not None
        label = "known" if is_known else "unknown"
        match_id = best["mini_id"] if is_known else None

        det = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "area": area,
            "label": label,
            "match_id": match_id,  # stable mini_id
            "name": (best["name"] if is_known else ""),
            "best_view_id": (best["view_id"] if is_known else None),
            "score": best["score"],
            "shape_dist": best["shape_dist"],
            "hist_metrics": best["hist_metrics"],
            "orb_metrics": best["orb_metrics"],
            "h_sim": best["h_sim"],
            "o_sim": best["o_sim"],
            "s_sim": best["s_sim"],
            "contour": contour.astype(np.int32)
        }
        detections.append(det)

        if draw:
            color = (0, 200, 0) if label == "known" else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            name_prefix = (best["name"] + " ") if (best["name"] and is_known) else ""
            tag1 = f"{name_prefix}{label}{'' if match_id is None else ':'+str(match_id)[-6:]}  s={best['score']:.2f}"
            y1 = max(0, y-8)
            cv2.putText(annotated, tag1, (x, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(annotated, tag1, (x, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

            if show_components:
                h_txt  = "--" if best["h_sim"] is None else f"{best['h_sim']:.2f}"
                o_txt  = "--" if best["o_sim"] is None else f"{best['o_sim']:.2f}"
                sh_txt = "--" if best["s_sim"] is None else f"{best['s_sim']:.2f}"
                tag2 = f"h={h_txt}  o={o_txt}  sh={sh_txt}"
                y2 = max(0, y1 - 16)
                cv2.putText(annotated, tag2, (x, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 3, cv2.LINE_AA)
                cv2.putText(annotated, tag2, (x, y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)

    result = {"detections": detections}
    if draw:
        result["annotated"] = annotated
    return result
