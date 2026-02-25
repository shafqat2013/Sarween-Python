# all real-time tracking code. Split into 3 sections:
# For the TV (homography, aruco detection, camera calibration)
# For the miniatures (grid mapping, color_mask/diff_mask/motion_mask)
# For hand gestures (spell_casting, tbd)
import time
import cv2
import numpy as np
import setup as s
import calibration as c
from mini_tracking2 import identify_minis

from collections import deque

persistent_cells = set()
cell_history = deque(maxlen=5)  # Adjust length for how many frames to remember
mini_last_cell = {}   # { mini_id: "K8" }
UNKNOWN_COUNTER = 0


# ────── HELPERS FOR GRID CELL COVERAGE CALCULATION ────────────────────────────────

def _cell_labels(grid_cols, grid_rows):
    labels = []
    for r in range(grid_rows):
        row = []
        for c in range(grid_cols):
            if c < 26:
                col_label = chr(ord('A') + c)
            else:
                col_label = f"C{c}"
            row.append(f"{col_label}{r+1}")
        labels.append(row)
    return labels  # [row][col]

def _coverage_by_cell(mask_warped, grid_cols, grid_rows, warp_w, warp_h):
    """Return a grid of coverage ratios [row][col] in [0,1]."""
    cell_w = warp_w // grid_cols
    cell_h = warp_h // grid_rows
    cov = [[0.0]*grid_cols for _ in range(grid_rows)]
    full = cell_w * cell_h
    for r in range(grid_rows):
        y0, y1 = r*cell_h, min((r+1)*cell_h, warp_h)
        for c in range(grid_cols):
            x0, x1 = c*cell_w, min((c+1)*cell_w, warp_w)
            cell = mask_warped[y0:y1, x0:x1]
            covered = int(cv2.countNonZero(cell))
            cov[r][c] = covered / max(1, full)
    return cov

def pick_base_cell_from_contour(contour_xy, H, warp_w, warp_h, grid_cols=23, grid_rows=16,
                                min_coverage=0.50):
    """
    Project a mini's contour to the warped map space using H, compute per-cell coverage,
    and pick ONE grid square:
      - Only consider cells with coverage >= min_coverage
      - Prefer the lowest row (max r)
      - Break ties by 'most central' in X relative to projected centroid
    Return: string label like 'K8' or None if nothing clears the threshold.
    """
    if contour_xy is None or len(contour_xy) == 0:
        return None

    # 1) project contour to warped space
    pts = contour_xy.reshape(-1, 1, 2).astype(np.float32)
    proj = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    # 2) rasterize the projected polygon into a mask
    mask = np.zeros((warp_h, warp_w), dtype=np.uint8)
    cv2.fillPoly(mask, [proj.astype(np.int32)], 255)

    # 3) compute coverage per cell
    cover = _coverage_by_cell(mask, grid_cols, grid_rows, warp_w, warp_h)

    # projected centroid for tie-breaking
    M = cv2.moments(proj.astype(np.float32))
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
    else:
        cx = float(np.mean(proj[:,0]))

    # 4) pick candidates
    labels = _cell_labels(grid_cols, grid_rows)
    candidates = []
    cell_w = warp_w // grid_cols
    for r in range(grid_rows):
        for c in range(grid_cols):
            if cover[r][c] >= min_coverage:
                # centrality = distance from column center
                col_center_x = (c + 0.5) * cell_w
                centrality = abs(cx - col_center_x)
                candidates.append((r, c, cover[r][c], centrality))

    if not candidates:
        return None

    # 5) choose: lowest row first (max r), then smallest centrality, then highest coverage
    candidates.sort(key=lambda t: (-t[0], t[3], -t[2]))
    r, c, _, _ = candidates[0]
    return labels[r][c]


# ── HOMOGRAPHY & REAL-TIME LOOP WITH TRACKING ─────────────────────────────────
def homography(camera_index=s.load_last_selection()["webcam_index"], dictionary=cv2.aruco.DICT_6X6_250):
    global persistent_cells
    background_frame = None
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # Load reference image to get aspect ratio
    image_ref = cv2.imread("maps/dnd1.jpg")
    if image_ref is None:
        print("❌ Could not load reference image for aspect ratio.")
        return

    img_h, img_w = image_ref.shape[:2]
    aspect_img = img_w / img_h

    # Set base height and compute width to preserve aspect ratio
    warp_h = 400
    warp_w = int(warp_h * aspect_img)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    parameters = cv2.aruco.DetectorParameters()

    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_modern_detector = True
    except AttributeError:
        use_modern_detector = False

    #undistort frame with saved camera matrix and distortion coefficients
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs = np.load('dist_coeffs.npy')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #undistort frame with saved camera matrix and distortion coefficients
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # # Preprocess current frame for diff
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)

        if background_frame is None:
            background_frame = capture_background(camera_index)
        
        if cv2.waitKey(1) & 0xFF == ord('r'):
            print("🔄 Resetting background...")
            background_frame = cv2.GaussianBlur(gray, (21, 21), 0)    

        # Compute absolute difference
        diff = cv2.absdiff(background_frame, gray_blur)

        # Threshold the difference to create binary mask of moving/different areas
        _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Optional: remove noise
        diff_mask = cv2.erode(diff_mask, None, iterations=2)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        # Optional: view the diff mask
        cv2.namedWindow("Difference Mask", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Difference Mask", 1280, 720)
        cv2.imshow("Difference Mask", diff_mask)

        

        if use_modern_detector:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters)

        marker_centers = []
        warped = None

        if ids is not None and len(ids) >= 4:
            # Draw detected marker borders and centers
            for i in range(len(ids)):
                pts = corners[i][0].astype(int)
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                marker_centers.append((cx, cy))

            try:
                src_pts = get_extreme_corners(corners, ids)
                dst_pts = np.array([
                    [0, 0],
                    [warp_w, 0],
                    [warp_w, warp_h],
                    [0, warp_h]
                ], dtype=np.float32)

                H, _ = cv2.findHomography(src_pts, dst_pts)

                if H is not None:
                    warped = cv2.warpPerspective(frame, H, (warp_w, warp_h))

                    # 1) run ID on the ORIGINAL frame (not warped), but provide your motion mask
                    detections = []
                    try:
                        det_result = identify_minis(
                            live_frame_bgr=frame,
                            diff_mask=diff_mask,                 # the one you already compute
                            background_blur=background_frame,    # from your existing capture
                            db_entries=None,                     # it will load DB itself
                            draw=False,
                            show_components=False
                        )
                        # det_result is a list (because draw=False), not a dict
                        detections = det_result if isinstance(det_result, list) else det_result.get('detections', [])
                    except Exception as e:
                        # don't crash the loop if ID hiccups
                        # print or log if you like
                        pass

                    # 2) for each detected mini, pick exactly ONE grid cell
                    for d in detections:
                        contour = d.get("contour", None)           # from our tweak
                        match_id = d.get("match_id")
                        label = d.get("label", "unknown")          # "known" / "unknown"

                        # stable mini_id: if known use match_id; otherwise assign a temporary unique ID
                        if label == "known" and match_id:
                            mini_id = match_id
                        else:
                            # assign sticky unknown id based on bbox center to reduce flicker
                            # (simple approach; you can replace with proper tracker later)
                            x,y,w,h = d.get("bbox", (0,0,0,0))
                            cx, cy = int(x + w/2), int(y + h/2)
                            mini_id = f"unknown_{cx}_{cy}"

                        # 3) project contour to warped map and choose one 'base' cell
                        base_cell = pick_base_cell_from_contour(
                            contour_xy=contour,
                            H=H,
                            warp_w=warp_w,
                            warp_h=warp_h,
                            grid_cols=23,
                            grid_rows=16,
                            min_coverage=0.50
                        )

                        if base_cell is None:
                            continue

                        # 4) movement output: only when the cell changes
                        last = mini_last_cell.get(mini_id)
                        if last != base_cell:
                            mini_last_cell[mini_id] = base_cell
                            print(f"{mini_id}, {base_cell}")   # <-- your required output

                    # covered_cells = get_grid_cells_covered_by_mask(
                    # cv2.warpPerspective(diff_mask, H, (warp_w, warp_h)),
                    # grid_cols=23, grid_rows=16, warp_w=warp_w, warp_h=warp_h
                    # )        
                    # # Print covered cells
                    # if covered_cells:
                    # print(f"Mini overlaps: {', '.join(covered_cells)}")

                    current_cells = get_grid_cells_covered_by_mask(
                        cv2.warpPerspective(diff_mask, H, (warp_w, warp_h)),
                        grid_cols=23, grid_rows=16, warp_w=warp_w, warp_h=warp_h
                    )

                    cell_history.append(set(current_cells))
                    persistent_cells = set.union(*cell_history)

                    # Print covered cells
                    if persistent_cells:
                        print(f"Miniatures detected at: {', '.join(persistent_cells)}")

                    # Grid and labeling
                    # Currently hardcoded for 23x16 grid, but needs to be adjustable by user
                    grid_cols = 23
                    grid_rows = 16
                    col_w = warp_w // grid_cols
                    row_h = warp_h // grid_rows

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.2
                    font_thickness = 1
                    text_color = (0, 255, 0)

                    for row in range(grid_rows):
                        for col in range(grid_cols):
                            if col > 0:
                                x = col * col_w
                                cv2.line(warped, (x, 0), (x, warp_h), color=(0, 255, 0), thickness=1)
                            if row > 0:
                                y = row * row_h
                                cv2.line(warped, (0, y), (warp_w, y), color=(0, 255, 0), thickness=1)

                            # Label each cell like A1, B2, etc.
                            col_label = chr(ord('A') + col)
                            row_label = str(row + 1)
                            label = f"{col_label}{row_label}"

                            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
                            text_x = col * col_w + 4
                            text_y = row * row_h + text_size[1] + 4

                            cv2.putText(warped, label, (text_x, text_y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

                    # Draw homography quad on original frame
                    cv2.polylines(frame, [np.int32(src_pts)], isClosed=True, color=(0, 255, 0), thickness=2)

            except Exception as e:
                print(f"⚠️ Homography computation failed: {e}")

        # Show marker count on original frame
        h, w = frame.shape[:2]
        cv2.putText(frame, str(len(marker_centers)), (w // 2 - 10, h // 2 + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=(255, 255, 255),
                    thickness=4,
                    lineType=cv2.LINE_AA)

        # Display both live and warped output
        cv2.namedWindow("Live ArUco Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Live ArUco Detection", 1280, 720)
        cv2.imshow("Live ArUco Detection", frame)
        if warped is not None:
            cv2.imshow("Warped Homography View", warped)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_grid_cells_covered_by_mask(mask, grid_cols=23, grid_rows=16, warp_w=400, warp_h=400):
    col_w = warp_w // grid_cols
    row_h = warp_h // grid_rows

    covered_cells = set()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Get the grid cell range covered by this bounding box
        col_start = max(0, x // col_w)
        col_end = min(grid_cols - 1, (x + w) // col_w)
        row_start = max(0, y // row_h)
        row_end = min(grid_rows - 1, (y + h) // row_h)

        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                if col < 26:
                    col_label = chr(ord('A') + col)
                else:
                    col_label = f"C{col}"
                label = f"{col_label}{row + 1}"
                covered_cells.add(label)

    return sorted(covered_cells)

def get_extreme_corners(corners, ids):
    # Flatten into (id, [4 corner pts]) tuples
    marker_data = [(ids[i][0], corners[i][0]) for i in range(len(ids))]

    # Compute center of each marker
    marker_centers = {
        m_id: np.mean(pts, axis=0)
        for m_id, pts in marker_data
    }

    # Sort by vertical (y) position, then horizontal (x)
    sorted_markers = sorted(marker_centers.items(), key=lambda item: (item[1][1], item[1][0]))

    # Take top 2 and bottom 2
    top_two = sorted_markers[:2]
    bottom_two = sorted_markers[2:4]

    top_left_marker = min(top_two, key=lambda item: item[1][0])[0]
    top_right_marker = max(top_two, key=lambda item: item[1][0])[0]
    bottom_left_marker = min(bottom_two, key=lambda item: item[1][0])[0]
    bottom_right_marker = max(bottom_two, key=lambda item: item[1][0])[0]

    id_to_corners = {m_id: pts for m_id, pts in marker_data}

    #outside corners (when webcam is correctly oriented)
    src_pts = np.array([
        id_to_corners[top_left_marker][0],      # top-left corner
        id_to_corners[top_right_marker][1],     # top-right corner
        id_to_corners[bottom_right_marker][2],  # bottom-right corner
        id_to_corners[bottom_left_marker][3],   # bottom-left corner
    ], dtype=np.float32)

    #inside corners (outside corners when webcam is upside down)
    # src_pts = np.array([
    #     id_to_corners[top_left_marker][2],  # bottom-right corner
    #     id_to_corners[top_right_marker][3],   # bottom-left corner
    #     id_to_corners[bottom_right_marker][0],      # top-left corner
    #     id_to_corners[bottom_left_marker][1],     # top-right corner
    # ], dtype=np.float32)

    return src_pts

def capture_background(camera_index=s.load_last_selection()["webcam_index"]):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    #Capture background reference frame
    print("📸 Capturing background...")
    time.sleep(1)  # Give camera time to stabilize
    for _ in range(10):  # Read a few frames to flush camera buffer
        cap.read()

    ret, background_frame = cap.read()
    if not ret:
        print("❌ Failed to capture background frame.")
        return
    background_frame = cv2.GaussianBlur(cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
    print("✅ Background captured.")

    cap.release()
    return background_frame

def begin_session():
    print("🔄 Starting tracking session...")
    homography()
