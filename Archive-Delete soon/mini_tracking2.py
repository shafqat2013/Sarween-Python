import cv2
import os, csv, json
import numpy as np
import time
import setup as s
import calibration as c  # keeping your import even if unused here
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
            np.load('camera_matrix.npy'),
            np.load('dist_coeffs.npy')
        )
    return _camera_params_cache

# ── Background capture helpers ────────────────────────────────────────────────
def capture_background(camera_index=None):
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]
    camera_matrix, dist_coeffs = get_camera_params()

    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background (blur)...")
    time.sleep(1)
    for _ in range(10):
        cap.read()  # warm-up
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

    # Hue wrap-safe difference
    dH = cv2.absdiff(Hl, Hb)
    dH = np.minimum(dH.astype(np.uint16), (180 - dH).astype(np.uint16)).astype(np.uint8)
    dS = cv2.absdiff(Sl, Sb)

    v_darker   = cv2.subtract(Vb, Vl)  # likely shadow
    v_brighter = cv2.subtract(Vl, Vb)  # specular/lighting increase

    chroma_change   = (dH > h_thresh) | (dS > s_thresh)
    strong_brighten = (v_brighter > (v_shadow_drop + 10))
    shadow          = (v_darker  > v_shadow_drop) & (dH <= h_thresh) & (dS <= s_thresh)

    # Final FG = chroma change OR strong brighten, but not shadow
    fg = (chroma_change | strong_brighten) & (~shadow)
    mask = (fg.astype(np.uint8) * 255)

    # Clean up
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_o, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c, iterations=1)
    return mask

def final_foreground_mask(motion_mask, shadowfree_mask,
                          keep_ratio=0.25, min_comp_area=400):
    return combine_masks_componentwise(motion_mask, shadowfree_mask,
                                       keep_ratio=keep_ratio, min_comp_area=min_comp_area)

# ── NEW: unified helper for per-frame final mask (used by tracking.py) ───────
def build_final_mask_for_frame(live_frame_bgr, BG,
                               keep_ratio=0.25,
                               min_comp_area=150,
                               motion_thresh=30):
    """
    Build the final foreground mask used for mini detection, given:
      - live_frame_bgr: current BGR frame
      - BG: dict from capture_background_full() with keys 'bgr' and 'blur'

    This mirrors the logic that was previously in tracking.begin_session:
      - absdiff on blurred grayscale
      - threshold
      - erode + dilate (2,2)
      - shadow_free_mask
      - combine_masks_componentwise
    """
    bg_blur = BG["blur"]
    bg_bgr  = BG["bgr"]

    live_gray = cv2.cvtColor(live_frame_bgr, cv2.COLOR_BGR2GRAY)
    live_blur = cv2.GaussianBlur(live_gray, (21, 21), 0)

    diff = cv2.absdiff(bg_blur, live_blur)
    _, motion_mask = cv2.threshold(diff, motion_thresh, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.erode(motion_mask, None, iterations=2)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)

    shadowfree = shadow_free_mask(bg_bgr, live_frame_bgr)
    final_mask = combine_masks_componentwise(
        motion_mask, shadowfree,
        keep_ratio=keep_ratio,
        min_comp_area=min_comp_area
    )
    return final_mask

# ── Histogram helpers (HS-only for lighting robustness) ───────────────────────
def flatten_hist_HS_from_HSV(hist_hsv):
    """
    hist_hsv: 3D (H,S,V) histogram or 2D (H,S).
    Returns normalized 1D float32 vector for compareHist.
    """
    h = hist_hsv
    if h.ndim == 3:  # collapse V
        h = h.sum(axis=2)
    h = h.astype(np.float32)
    s_ = h.sum()
    if s_ > 0:
        h /= s_
    return h.ravel().reshape(-1,1)

def contour_hist_HS(frame_bgr, contour, bins_h=32, bins_s=32):
    x,y,w,h = cv2.boundingRect(contour)
    roi_bgr = frame_bgr[y:y+h, x:x+w]
    if roi_bgr.size == 0:
        return None
    # shrink mask slightly to avoid edges/shadows bleeding into color stats
    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tmp, [(contour - [x, y]).astype(np.int32)], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    roi_mask = cv2.erode(tmp, kernel, iterations=1)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist2 = cv2.calcHist([roi_hsv], [0,1], roi_mask,
                         [bins_h, bins_s], [0,180, 0,256])
    return flatten_hist_HS_from_HSV(hist2)

# ── ORB feature helpers (rotation/scale tolerant) ─────────────────────────────
def compute_orb_descriptors(frame_bgr, contour):
    """
    Compute ORB descriptors within the contour ROI (masked).
    Returns descriptors (Nx32 uint8) or None.
    """
    x,y,w,h = cv2.boundingRect(contour)
    roi_bgr = frame_bgr[y:y+h, x:x+w]
    if roi_bgr.size == 0:
        return None
    tmp = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(tmp, [(contour - [x, y]).astype(np.int32)], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    roi_mask = cv2.erode(tmp, kernel, iterations=1)

    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=800, scaleFactor=1.2, nlevels=8,
                         edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20)
    kp, des = orb.detectAndCompute(roi_gray, roi_mask)
    return des  # may be None

def orb_similarity(desc_q, desc_db, ratio_thresh=0.75):
    """
    Lowe's ratio-tested ORB match score in [0,1].
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
    """
    Given '..._hist.npy', return '..._orb.npz' if that naming convention holds.
    """
    if hist_path.endswith("_hist.npy"):
        return hist_path[:-9] + "_orb.npz"  # remove '_hist.npy' (9 chars) -> add '_orb.npz'
    return None

# ── Capture & persist a mini (blob, contour, HSV hist, ORB) ───────────────────
def capture_mini(
    camera_index=None,
    background_blur=None,
    background_bgr=None,
    save_dir="mini_captures",
    min_area=1500
):
    """
    Capture frame, extract largest blob, compute features, and save artifacts + metadata.
    """
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]
    camera_matrix, dist_coeffs = get_camera_params()

    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    for _ in range(5):  # warm up
        cap.read()
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret or frame_bgr is None:
        print("❌ Could not capture mini.")
        return None

    # Undistort and prep grayscale/blur
    frame_bgr = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # Motion mask (preferred) or Otsu fallback
    if background_blur is not None:
        diff = cv2.absdiff(background_blur, blur)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If we have a BGR background, suppress shadows
    if background_bgr is not None:
        sf = shadow_free_mask(background_bgr, frame_bgr)
        # Component-wise gate: keep motion blobs that survive the shadow test enough
        mask_comb = combine_masks_componentwise(mask, sf, keep_ratio=0.35, min_comp_area=400)
        # Fallback: if combiner kills everything (e.g., thresholds too strict), keep motion mask
        if cv2.countNonZero(mask_comb) > 0:
            mask = mask_comb

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find largest contour (blob)
    cnts_info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]
    if not cnts:
        print("⚠️ No contours found.")
        return None
    contour = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        print(f"⚠️ Largest contour too small (area={area:.0f} < {min_area}).")
        return None

    # Contour features
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
    hull_area    = float(cv2.contourArea(hull))
    solidity     = area / hull_area if hull_area > 0 else 0.0

    # ROI mask slightly eroded
    roi_mask_full = np.zeros_like(mask)
    cv2.drawContours(roi_mask_full, [contour], -1, 255, thickness=-1)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    roi_mask_full = cv2.erode(roi_mask_full, kernel5, iterations=1)
    roi_mask_tight = roi_mask_full[y:y+h, x:x+w]
    roi_bgr = frame_bgr[y:y+h, x:x+w]

    # HSV histogram inside the contour region (3D, keep for DB; HS used for matching)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([roi_hsv], [0, 1, 2], roi_mask_tight,
                        [32, 32, 8], [0, 180, 0, 256, 0, 256])
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum  # L1 normalize

    # ORB descriptors (rotation/scale tolerant)
    # Compute on full-frame with contour mask to keep coordinates consistent
    orb_des = compute_orb_descriptors(frame_bgr, contour)

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(save_dir, f"mini_{ts}")
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
            # Create an empty placeholder so loader can detect "exists but empty"
            np.savez_compressed(orb_path, des=np.empty((0,32), dtype=np.uint8))
    except Exception as e:
        print(f"⚠️ Failed to save ORB descriptors: {e}")
        orb_path = ""

    # Metadata
    meta = {
        "id": ts,
        "timestamp": ts,
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
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "hull_area": hull_area,
            "solidity": solidity,
            "min_area_threshold": min_area
        },
        "histogram": {"space": "HSV", "bins": [32, 32, 8],
                      "ranges": {"H": [0, 180], "S": [0, 256], "V": [0, 256]},
                      "normalized": True}
    }

    try:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write JSON metadata: {e}")

    # Append to CSV "database" (keep old schema for back-compat; we infer ORB path later)
    csv_path = DB_CSV
    header = [
        "id", "timestamp", "image", "mask", "contour_npy", "hist_npy",
        "area", "perimeter", "cx", "cy", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "aspect_ratio", "circularity", "hull_area", "solidity"
    ]
    row = [
        ts, ts, img_path, mask_path, cnt_path, hist_path,
        f"{area:.4f}", f"{perimeter:.4f}", f"{cx:.2f}", f"{cy:.2f}",
        x, y, w, h, f"{aspect_ratio:.4f}", f"{circularity:.4f}",
        f"{hull_area:.4f}", f"{solidity:.4f}"
    ]
    try:
        write_header = not os.path.exists(csv_path)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
    except Exception as e:
        print(f"⚠️ Failed to append to CSV: {e}")

    print(f"✅ Mini captured -> {img_path}")
    print(f"   • Mask:      {mask_path}")
    print(f"   • Contour:   {cnt_path} (area={area:.0f})")
    print(f"   • Histogram: {hist_path}")
    print(f"   • ORB:       {orb_path if orb_path else '(save failed)'}")
    print(f"   • JSON:      {json_path}")
    print(f"   • DB row ->  {csv_path}")

    return {
        "image": img_path, "mask": mask_path, "contour_npy": cnt_path, "hist_npy": hist_path,
        "json": json_path, "csv": csv_path, "features": meta["features"], "orb_npz": orb_path
    }

# ── DB loader (HS-collapsed hist + ORB for robust matching) ───────────────────
def load_mini_database(db_csv_path=DB_CSV):
    entries = []
    if not os.path.exists(db_csv_path):
        print(f"ℹ️ DB not found at {db_csv_path}.")
        return entries
    with open(db_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hist_hsv = np.load(row["hist_npy"])  # saved 32x32x8
                hist_hs_flat = flatten_hist_HS_from_HSV(hist_hsv)  # collapse V
            except Exception as e:
                print(f"⚠️ Skip {row.get('id','?')} (hist load failed): {e}")
                continue

            contour = None
            try:
                contour = np.load(row["contour_npy"], allow_pickle=False)
            except Exception:
                pass

            # Derive ORB path from hist path (back-compat)
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
                "id": row["id"],
                "hist_hs_flat": hist_hs_flat,
                "contour": contour,
                "orb_des": orb_des
            })
    return entries

# ── Identify minis with ORB + color + shape, plus strong geometric filters ────
def identify_minis(
    live_frame_bgr,
    diff_mask=None,
    live_blur=None,
    background_blur=None,
    db_entries=None,
    db_csv_path=DB_CSV,
    min_area=1500,
    known_threshold=0.50,
    min_fill=0.35,         # area / (w*h)
    min_solidity=0.85,     # area / hull_area
    keep_top_k=3,
    skip_border=True,
    draw=True,
    show_components=True    # NEW: draw h/o/shape components on-frame
):
    if db_entries is None:
        db_entries = load_mini_database(db_csv_path)

    # Build a mask if none provided
    if diff_mask is None:
        if live_blur is not None and background_blur is not None:
            diff = cv2.absdiff(background_blur, live_blur)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.cvtColor(live_frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            _, diff_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = cv2.erode(diff_mask, None, iterations=1)
    fg = cv2.dilate(fg, None, iterations=1)

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

        # Geometric filters
        fill = area / float(w * h) if w > 0 and h > 0 else 0.0
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 0 else 0.0
        if fill < min_fill or solidity < min_solidity:
            continue

        # Centroid
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = cy = -1.0

        # HS histogram for matching
        q_hist_hs_flat = contour_hist_HS(live_frame_bgr, contour)
        # ORB descriptors for matching
        q_orb_des = compute_orb_descriptors(live_frame_bgr, contour)

        best = {
            "score": -1.0, "shape_dist": None, "hist_metrics": None,
            "match_id": None, "orb_metrics": None,
            "h_sim": None, "o_sim": None, "s_sim": None  # NEW: component scores
        }

        for entry in db_entries:
            h_sim, h_metrics = hist_similarity_flat(q_hist_hs_flat, entry["hist_hs_flat"]) if q_hist_hs_flat is not None else (0.0, None)

            s_sim = None; s_dist = None
            if entry.get("contour") is not None and entry["contour"] is not None and entry["contour"].size > 0:
                s_pair = shape_similarity(contour, entry["contour"])
                if s_pair is not None:
                    s_sim, s_dist = s_pair

            o_sim = None; o_metrics = None
            if q_orb_des is not None and entry.get("orb_des") is not None:
                o_sim, o_metrics = orb_similarity(q_orb_des, entry["orb_des"])

            # Same weighting as before (fallback to h_sim if missing)
            final = 0.33 * h_sim \
                  + 0.33 * (o_sim if o_sim is not None else h_sim) \
                  + 0.33 * (s_sim if s_sim is not None else h_sim)

            if final > best["score"]:
                best.update({
                    "score": float(final),
                    "shape_dist": s_dist,
                    "hist_metrics": h_metrics,
                    "match_id": entry["id"],
                    "orb_metrics": o_metrics,
                    "h_sim": h_sim, "o_sim": o_sim, "s_sim": s_sim  # NEW
                })

        is_known = best["score"] >= known_threshold and best["match_id"] is not None
        label = "known" if is_known else "unknown"
        match_id = best["match_id"] if is_known else None

        det = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "area": area,
            "label": label,
            "match_id": match_id,
            "score": best["score"],
            "shape_dist": best["shape_dist"],
            "hist_metrics": best["hist_metrics"],
            "orb_metrics": best["orb_metrics"],
            # expose components
            "h_sim": best["h_sim"],
            "o_sim": best["o_sim"],
            "s_sim": best["s_sim"],
            # NEW: For integration with map tracking
            "contour": contour.astype(np.int32)   # <-- required by tracking.py integration
        }
        detections.append(det)

        if draw:
            color = (0, 200, 0) if label == "known" else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Line 1: id + s
            tag1 = f"{label}{'' if match_id is None else ':'+str(match_id[-6:])}  s={best['score']:.2f}"
            y1 = max(0, y-8)
            cv2.putText(annotated, tag1, (x, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(annotated, tag1, (x, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)

            # Line 2: components (h/o/shape)
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

def combine_masks_componentwise(motion_mask, shadowfree_mask,
                                keep_ratio=0.25,      
                                min_comp_area=400,
                                k_open=3, k_close=5):
    """
    Keep connected components from motion_mask if enough of their pixels
    also pass the shadow-free mask. Returns a clean binary mask (0/255).
    """
    mot = (motion_mask > 0).astype(np.uint8) * 255
    sh  = (shadowfree_mask > 0).astype(np.uint8) * 255

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mot, connectivity=8)
    out = np.zeros_like(mot)

    for i in range(1, num):  # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_comp_area:
            continue
        comp = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(comp, sh)
        ratio = cv2.countNonZero(overlap) / float(area)
        if ratio >= keep_ratio:
            out = cv2.bitwise_or(out, comp)

    # tidy
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k1, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k2, iterations=1)
    return out

def combine_masks_componentwise_debug(motion_mask, shadowfree_mask,
                                      keep_ratio=0.25,
                                      min_comp_area=400,
                                      k_open=3, k_close=5,
                                      min_fill=None,
                                      min_solidity=None,
                                      min_area_id=None):
    """
    Returns (final_mask, debug_bgr):
      - final_mask: binary mask used for ID, built component-wise
      - debug_bgr: annotated visualization explaining why comps were kept/dropped
    """
    mot = ((motion_mask > 0).astype(np.uint8) * 255)
    sh  = ((shadowfree_mask > 0).astype(np.uint8) * 255)

    H, W = mot.shape[:2]
    out  = np.zeros_like(mot)

    # Base debug canvas: grayscale motion mask -> BGR
    dbg = cv2.cvtColor(mot, cv2.COLOR_GRAY2BGR)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mot, connectivity=8)

    for i in range(1, num):  # 0 = background
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area <= 0 or w <= 0 or h <= 0:
            continue

        comp = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(comp, sh)
        ov = int(cv2.countNonZero(overlap))
        ratio = (ov / float(area)) if area > 0 else 0.0

        # Component-wise shadow test
        pass_area_comp  = (area >= min_comp_area)
        pass_ratio      = (ratio >= keep_ratio)
        kept_comp       = pass_area_comp and pass_ratio

        # Geometric filters (mirror identify_minis)
        cnts = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        fill, solidity = 0.0, 0.0
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            a = float(cv2.contourArea(c))
            xx, yy, ww, hh = cv2.boundingRect(c)
            fill = a / float(ww * hh) if ww > 0 and hh > 0 else 0.0
            hull = cv2.convexHull(c)
            ha = float(cv2.contourArea(hull)) if hull is not None and hull.size > 0 else 0.0
            solidity = (a / ha) if ha > 0 else 0.0

        pass_fill      = True if min_fill      is None else (fill     >= min_fill)
        pass_solidity  = True if min_solidity  is None else (solidity >= min_solidity)
        pass_area_id   = True if min_area_id   is None else (area     >= min_area_id)

        pass_for_id = kept_comp and pass_fill and pass_solidity and pass_area_id

        # Add to final mask if passes component-wise shadow test
        if kept_comp:
            out = cv2.bitwise_or(out, comp)

        # Colors/labels
        if pass_for_id:
            color = (0, 200, 0)    # green
            tagL  = "KEPT"
        elif kept_comp:
            color = (0, 215, 255)  # yellow
            tagL  = "DROP(geom)"
        else:
            color = (0, 0, 255)    # red
            tagL  = "DROP(shadow)"

        cv2.rectangle(dbg, (x, y), (x+w, y+h), color, 2)

        # ----- SAFE string building (no nested f-strings) -----
        r_ok   = 'Y' if pass_ratio else 'N'
        fill_str = "" if min_fill is None else f">={min_fill:.2f}? {'Y' if pass_fill else 'N'}"
        sol_str  = "" if min_solidity is None else f">={min_solidity:.2f}? {'Y' if pass_solidity else 'N'}"
        aid_thr  = '-' if min_area_id is None else str(min_area_id)
        aid_ok   = 'Y' if pass_area_id else 'N'

        l1 = f"#{i} {tagL}  A={area}  ov={ov}  r={ratio:.2f} (>= {keep_ratio:.2f}? {r_ok})"
        l2 = f"fill={fill:.2f}{fill_str}  sol={solidity:.2f}{sol_str}  Aid>={aid_thr}? {aid_ok}"
        # ------------------------------------------------------

        y1 = max(12, y - 6)
        y2 = y1 - 18
        cv2.putText(dbg, l1, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(dbg, l1, (x, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0),   1, cv2.LINE_AA)
        cv2.putText(dbg, l2, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(dbg, l2, (x, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0),   1, cv2.LINE_AA)

    # Tidy final mask
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  k1, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k2, iterations=1)

    # Header with thresholds
    header1 = "FinalMask: component-wise keep"
    header2 = f"keep_ratio≥{keep_ratio:.2f}  min_comp_area≥{min_comp_area}"
    header3 = f"min_fill≥{min_fill if min_fill is not None else '-'}  " \
              f"min_solidity≥{min_solidity if min_solidity is not None else '-'}  " \
              f"min_area_id≥{min_area_id if min_area_id is not None else '-'}"
    cv2.putText(dbg, header1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 3, cv2.LINE_AA)
    cv2.putText(dbg, header1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(dbg, header2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 3, cv2.LINE_AA)
    cv2.putText(dbg, header2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(dbg, header3, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 3, cv2.LINE_AA)
    cv2.putText(dbg, header3, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return out, dbg

# ── Main loop ─────────────────────────────────────────────────────────────────
def mini_capture(camera_index=None):
    if camera_index is None:
        camera_index = s.load_last_selection()["webcam_index"]
    camera_matrix, dist_coeffs = get_camera_params()

    background_frame = None   # blurred grayscale
    BG = None                 # dict with 'bgr' and 'blur'
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    os.makedirs("mini_captures", exist_ok=True)
    print("▶️  Press 'r' to reset background, 'c' to save a mini snapshot, 'i' to identify, 'q' to quit.")

    while True:
        if background_frame is None:
            background_frame = capture_background(camera_index)
            if background_frame is None:
                print("❌ Exiting because background (blur) could not be captured.")
                break

        if BG is None:
            BG = capture_background_full(camera_index)
            if BG is None:
                print("❌ Exiting because background (bgr+blur) could not be captured.")
                break

        # Read a live frame
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            print("❌ Failed to read from webcam.")
            print(ret)
            print(live_frame)
            break

        # Undistort and preprocess current frame
        live_frame = cv2.undistort(live_frame, camera_matrix, dist_coeffs)
        live_gray  = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        live_blur  = cv2.GaussianBlur(live_gray, (21, 21), 0)

        # Motion mask (precise)
        diff = cv2.absdiff(background_frame, live_blur)
        _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.erode(diff_mask, None, iterations=2)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        # Shadow-free mask (permissive)
        diff_mask2 = shadow_free_mask(BG["bgr"], live_frame)

        # Final combined mask + annotated debug view
        final_mask, final_dbg = combine_masks_componentwise_debug(
            diff_mask, diff_mask2,
            keep_ratio=0.25,          # tune here…
            min_comp_area=400,        # tune here…
            min_fill=0.35,            # mirror identify_minis()
            min_solidity=0.75,
            min_area_id=1500          # same as identify_minis min_area
        )

        # Show views
        cv2.imshow("Final Mask (used for ID)", final_dbg)   # annotated view

        # Handle keys (single waitKey per loop)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            background_frame = capture_background(camera_index)
            BG = capture_background_full(camera_index)
            if background_frame is not None:
                print("🔄 Background (blur) reset.")
            else:
                print("❌ Background reset failed; keeping old background.")
            if BG is not None:
                print("🔄 BG (bgr+blur) reset.")
            else:
                print("❌ BG reset failed; keeping old BG.")
        elif key == ord('c'):
            result = capture_mini(camera_index,
                                  background_blur=background_frame,
                                  background_bgr=BG["bgr"])
            if result is None:
                print("⚠️ Mini capture failed.")
            else:
                f = result["features"]
                print(f"🧩 Features: area={f['area']:.0f}, circ={f['circularity']:.3f}, "
                      f"solidity={f['solidity']:.3f}, bbox={f['bbox']}")
        elif key == ord('i'):
            result = identify_minis(
                live_frame,
                diff_mask=final_mask,
                db_csv_path=DB_CSV,
                min_area=1500,
                min_fill=0.35,        # ← match overlay
                min_solidity=0.75,    # ← match overlay
                known_threshold=0.50,
                draw=True
            )

            print(f"Found {len(result['detections'])} blobs:")
            for d in result['detections']:
                h_txt  = "--" if d.get("h_sim") is None else f"{d['h_sim']:.2f}"
                o_txt  = "--" if d.get("o_sim") is None else f"{d['o_sim']:.2f}"
                sh_txt = "--" if d.get("s_sim") is None else f"{d['s_sim']:.2f}"
                print(f"  • {d['label']:7s} s={d['score']:.2f} "
                      f"(h={h_txt}, o={o_txt}, sh={sh_txt})  id={d['match_id']}  bbox={d['bbox']}")
            if 'annotated' in result:
                cv2.imshow("Identified Minis", result['annotated'])
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── Safe module entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    s.initialize()
    mini_capture()
