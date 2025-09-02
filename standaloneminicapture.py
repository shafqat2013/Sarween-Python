import cv2
import os, csv, json
import numpy as np
import time
import setup as s
import calibration as c  # keeping your import even if unused here
from datetime import datetime

# ── Load camera calibration ────────────────────────────────────────────────────
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs   = np.load('dist_coeffs.npy')

def capture_background(camera_index=s.load_last_selection()["webcam_index"]):
    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background...")
    time.sleep(1)
    for _ in range(10):
        cap.read()  # warm-up
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ Could not capture background.")
        return None
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    return blur

def capture_mini(
    camera_index=s.load_last_selection()["webcam_index"],
    background_blur=None, 
    background_bgr=None, 
    save_dir="mini_captures", 
    min_area=1500
):
    """
    Capture a frame, extract largest foreground blob/contour (using background_blur if provided),
    compute HSV color histogram inside the contour, and save all artifacts + metadata.
    Returns a dict with paths and features, or None on failure.
    """
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(camera_index)
    # Small warm-up helps exposure settle
    for _ in range(5):
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

    # Foreground mask: prefer diff against supplied background; otherwise fallback to Otsu
    if background_blur is not None:
        diff = cv2.absdiff(background_blur, blur)
        _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    else:
        # Fallback when no background is available
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If we have a BGR background, suppress shadows
    if background_bgr is not None:
        sf = shadow_free_mask(background_bgr, frame_bgr)
        mask = cv2.bitwise_and(mask, sf)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find largest contour (blob)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
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
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Build a tight ROI + mask just around the contour
    roi_mask_full = np.zeros_like(mask)
    cv2.drawContours(roi_mask_full, [contour], -1, 255, thickness=-1)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    roi_mask_full = cv2.erode(roi_mask_full, kernel5, iterations=1)
    roi_mask_tight = roi_mask_full[y:y+h, x:x+w]
    roi_bgr = frame_bgr[y:y+h, x:x+w]

    # HSV histogram inside the contour region
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [roi_hsv], [0, 1, 2], roi_mask_tight,
        [32, 32, 8], [0, 180, 0, 256, 0, 256]
    )
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum  # L1 normalize

    # Save artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.join(save_dir, f"mini_{ts}")
    img_path = f"{base}.png"
    mask_path = f"{base}_mask.png"
    cnt_path = f"{base}_contour.npy"
    hist_path = f"{base}_hist.npy"
    json_path = f"{base}.json"

    # Save full frame and mask
    cv2.imwrite(img_path, frame_bgr)
    cv2.imwrite(mask_path, mask)
    # Save arrays
    np.save(cnt_path, contour)
    np.save(hist_path, hist)

    # Metadata
    meta = {
        "id": ts,
        "timestamp": ts,
        "files": {
            "image": img_path,
            "mask": mask_path,
            "contour_npy": cnt_path,
            "hist_npy": hist_path
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
        "histogram": {
            "space": "HSV",
            "bins": [32, 32, 8],
            "ranges": {"H": [0, 180], "S": [0, 256], "V": [0, 256]},
            "normalized": True
        }
    }

    # Save JSON metadata
    try:
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        print(f"⚠️ Failed to write JSON metadata: {e}")

    # Append to CSV "database"
    csv_path = os.path.join(save_dir, "..", "mini_database.csv")
    csv_path = os.path.abspath(csv_path)
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
    print(f"   • JSON:      {json_path}")
    print(f"   • DB row ->  {csv_path}")

    return {
        "image": img_path,
        "mask": mask_path,
        "contour_npy": cnt_path,
        "hist_npy": hist_path,
        "json": json_path,
        "csv": csv_path,
        "features": meta["features"]
    }

# ── Helper: load DB (histograms + contours) ────────────────────────────────────
def load_mini_database(db_csv_path="mini_database.csv"):
    """
    Loads your mini DB from CSV and returns a list of entries with:
      { 'id', 'hist', 'contour', 'image', 'mask', 'hist_npy', 'contour_npy', 'json' }
    Missing files are skipped with a warning.
    """
    entries = []
    if not os.path.exists(db_csv_path):
        print(f"ℹ️ DB not found at {db_csv_path}.")
        return entries
    with open(db_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hist_hsv = np.load(row["hist_npy"])  # your saved 32x32x8
                hist_hs_flat = flatten_hist_HS_from_HSV(hist_hsv)  # collapse V
            except Exception as e:
                print(f"⚠️ Skip {row.get('id','?')} (hist load failed): {e}")
                continue
            contour = None
            try:
                contour = np.load(row["contour_npy"], allow_pickle=False)
            except Exception as e:
                pass
            entries.append({
                "id": row["id"],
                "hist_hs_flat": hist_hs_flat,
                "contour": contour
            })
    return entries

# ── Identify Minis ─────────────────────────────────────────────────────────────
def identify_minis(
    live_frame_bgr,
    diff_mask=None,
    live_blur=None,
    background_blur=None,
    db_entries=None,
    db_csv_path="mini_database.csv",
    min_area=1500,
    known_threshold=0.50,
    draw=True
):
    """
    Detect foreground blobs and try to match each to known minis in DB.

    Inputs:
      live_frame_bgr   : current UNDISTORTED BGR frame
      diff_mask        : optional binary mask (255 = foreground). If None, will compute
                         from live_blur vs background_blur (preferred) or fall back to Otsu.
      live_blur        : optional current blurred grayscale (to build diff if needed)
      background_blur  : optional blurred grayscale background (to build diff if needed)
      db_entries       : optional preloaded DB from load_mini_database(); if None, will load.
      db_csv_path      : CSV path to DB if db_entries not provided
      min_area         : contour area threshold to ignore tiny blobs
      known_threshold  : 0..1 similarity above which a blob is considered a known mini
      draw             : if True, returns annotated frame in result['annotated']

    Returns:
      {
        'detections': [
           {'bbox': (x,y,w,h),
            'centroid': (cx,cy),
            'area': area,
            'label': 'known' or 'unknown',
            'match_id': id_or_None,
            'score': final_similarity_0_to_1,
            'shape_dist': float_or_None,
            'hist_metrics': {'chisq':..., 'bhatt':..., 'corr':...}
           }, ...
        ],
        'annotated': (optional) annotated_frame_bgr
      }
    """
    if db_entries is None:
        db_entries = load_mini_database(db_csv_path)

    # Build/clean a foreground mask if not provided
    if diff_mask is None:
        if live_blur is not None and background_blur is not None:
            diff = cv2.absdiff(background_blur, live_blur)
            _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        else:
            # Fallback to Otsu over grayscale if blur not provided
            gray = cv2.cvtColor(live_frame_bgr, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            _, diff_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = cv2.erode(diff_mask, None, iterations=2)
    fg = cv2.dilate(fg, None, iterations=2)

    # Find blobs
    cnts_info = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts_info[0] if len(cnts_info) == 2 else cnts_info[1]

    detections = []
    annotated = live_frame_bgr.copy() if draw else None

    # Helper: shape similarity via matchShapes (smaller is better)
    def shape_similarity(c1, c2):
        if c1 is None or c2 is None:
            return None
        d = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I1, 0.0)  # 0 = identical
        # Convert to a 0..1 "similarity": 1 at 0 distance, linearly decays to 0 by d=1.0
        sim = max(0.0, 1.0 - min(d, 1.0))
        return sim, float(d)
    
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

        # Features
        perimeter = float(cv2.arcLength(contour, True))
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = cy = -1.0
        x, y, w, h = cv2.boundingRect(contour)

        #q_hist = contour_hist(live_frame_bgr, contour)
        q_hist_hs_flat = contour_hist_HS(live_frame_bgr, contour)
        
        # if q_hist is None:
        #     continue

        if q_hist_hs_flat is None:
             continue

        # Match against DB
        best = {
            "score": -1.0, "shape_dist": None, "hist_metrics": None,
            "match_id": None
        }
        for entry in db_entries:
            #h_sim, h_metrics = hist_similarity(q_hist, entry["hist"])
            h_sim, h_metrics = hist_similarity_flat(q_hist_hs_flat, entry["hist_hs_flat"])
            s_sim = None
            s_dist = None
            if entry.get("contour") is not None and entry["contour"].size > 0:
                s_pair = shape_similarity(contour, entry["contour"])
                if s_pair is not None:
                    s_sim, s_dist = s_pair

            # Blend histogram + shape (favor color; shape helps disambiguate)
            if s_sim is not None:
                final = 0.75 * h_sim + 0.25 * s_sim
            else:
                final = h_sim

            if final > best["score"]:
                best.update({
                    "score": float(final),
                    "shape_dist": s_dist,
                    "hist_metrics": h_metrics,
                    "match_id": entry["id"]
                })

        # Decide label
        is_known = best["score"] >= known_threshold
        label = "known" if is_known and best["match_id"] is not None else "unknown"
        match_id = best["match_id"] if is_known else None

        det = {
            "bbox": (int(x), int(y), int(w), int(h)),
            "centroid": (float(cx), float(cy)),
            "area": area,
            "label": label,
            "match_id": match_id,
            "score": best["score"],
            "shape_dist": best["shape_dist"],
            "hist_metrics": best["hist_metrics"]
        }
        detections.append(det)

        # Draw
        if draw:
            color = (0, 200, 0) if label == "known" else (0, 165, 255)  # green / orange
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
            tag = f"{label}{'' if match_id is None else ':'+str(match_id[-6:])}  s={best['score']:.2f}"
            # black text with white outline for readability
            cv2.putText(annotated, tag, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated, tag, (x, max(0, y-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    result = {"detections": detections}
    if draw:
        result["annotated"] = annotated
    return result

def shadow_free_mask(bg_bgr, live_bgr,
                     h_thresh=12, s_thresh=25, v_shadow_drop=35,
                     k_open=3, k_close=5):
    """
    Return a binary mask (uint8 0/255) of foreground that ignores shadows.
    Logic:
      - Pixel is 'shadow' if V gets darker a lot but H,S barely change.
      - Pixel is 'foreground' if H or S change enough (or gets much brighter).
    """
    bg_hsv   = cv2.cvtColor(bg_bgr,   cv2.COLOR_BGR2HSV)
    live_hsv = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2HSV)

    Hb,Sb,Vb = cv2.split(bg_hsv)
    Hl,Sl,Vl = cv2.split(live_hsv)

    # Hue is circular on [0,180) in OpenCV HSV
    dH = cv2.absdiff(Hl, Hb)
    dH = cv2.min(dH, 180 - dH)
    dS = cv2.absdiff(Sl, Sb)

    # Positive when the live pixel is darker than background (likely shadow)
    v_darker = cv2.subtract(Vb, Vl)
    # Positive when the live pixel is brighter than background (specular/lighting change)
    v_brighter = cv2.subtract(Vl, Vb)

    # Base "something changed in chroma"
    chroma_change = (dH > h_thresh) | (dS > s_thresh)
    # Strong brightening (e.g., shiny piece) should still count foreground
    strong_brighten = (v_brighter > (v_shadow_drop + 10))

    # Shadow definition: darker a lot, without much chroma change
    shadow = (v_darker > v_shadow_drop) & (dH <= h_thresh) & (dS <= s_thresh)

    # Final FG = chroma change OR strong brighten, but not shadow
    fg = (chroma_change | strong_brighten) & (~shadow)
    mask = (fg.astype(np.uint8) * 255)

    # Clean up
    kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,  k_open))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_o, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_c, iterations=1)
    return mask

def capture_background_full(camera_index=s.load_last_selection()["webcam_index"]):
    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background...")
    time.sleep(1)
    for _ in range(10): cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("❌ Could not capture background.")
        return None
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (21,21), 0)
    return {"bgr": frame, "blur": blur}

def flatten_hist_HS_from_HSV(hist_hsv):
    """
    hist_hsv: 3D (H,S,V) histogram or 2D (H,S).
    Returns normalized 1D float32 vector for compareHist.
    """
    h = hist_hsv
    if h.ndim == 3:  # collapse V
        h = h.sum(axis=2)
    h = h.astype(np.float32)
    s = h.sum()
    if s > 0: h /= s
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
    # Normalize and flatten
    return flatten_hist_HS_from_HSV(hist2)


def standaloneminicapture(camera_index=s.load_last_selection()["webcam_index"]):
    background_frame = None
    BG = None
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # Optional: set a stable resolution (comment out if you prefer defaults)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    os.makedirs("mini_captures", exist_ok=True)

    print("▶️  Press 'r' to reset background, 'c' to save a mini snapshot, 'q' to quit.")

    while True:
        if background_frame is None:
            background_frame = capture_background(camera_index)
            if background_frame is None:
                print("❌ Exiting because background could not be captured.")
                break

        if BG is None:
            BG = capture_background_full(camera_index)
            if BG is None:
                print("❌ Exiting because background could not be captured.")
                break

        # Read a live frame
        ret, live_frame = cap.read()
        if not ret or live_frame is None:
            print("❌ Failed to read from webcam.")
            break

        # Undistort and preprocess current frame
        live_frame = cv2.undistort(live_frame, camera_matrix, dist_coeffs)
        live_gray  = cv2.cvtColor(live_frame, cv2.COLOR_BGR2GRAY)
        live_blur  = cv2.GaussianBlur(live_gray, (21, 21), 0)

        # Difference mask
        diff = cv2.absdiff(background_frame, live_blur)
        _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.erode(diff_mask, None, iterations=2)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        #Shadow free mask (testing)
        diff_mask2 = shadow_free_mask(BG["bgr"], live_frame)

        # Show views
        cv2.imshow("Live (undistorted)", live_frame)
        cv2.imshow("Difference Mask", diff_mask)
        cv2.imshow("Shadow-free Mask", diff_mask2)   

        # Handle keys (single waitKey per loop)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            background_frame = capture_background(camera_index)
            BG = capture_background_full(camera_index)
            if background_frame is not None:
                print("🔄 Background reset.")
            else:
                print("❌ Background reset failed; keeping old background.")
            if BG is not None:
                print("🔄 BG reset.")
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
                live_frame,                # already undistorted
                #diff_mask=diff_mask,       # reuse your current mask
                diff_mask = diff_mask2,      # use the shadow-free mask
                db_csv_path="mini_database.csv",  # adjust if yours lives elsewhere
                min_area=1500,
                known_threshold=0.58,
                draw=True
            )
            print(f"Found {len(result['detections'])} blobs:")
            for d in result['detections']:
                print(f"  • {d['label']:7s}  score={d['score']:.2f}  id={d['match_id']}  bbox={d['bbox']}")
            if 'annotated' in result:
                cv2.imshow("Identified Minis", result['annotated'])
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Init + run
s.initialize()
standaloneminicapture()
