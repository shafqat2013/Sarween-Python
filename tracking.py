import time
import cv2
import numpy as np
import setup as s
from collections import deque
from scipy.spatial import distance as dist

# ── GRID CELL PERSISTENCE ─────────────────────────────────────────────────────
persistent_cells = set()
cell_history = deque(maxlen=5)  # how many recent frames of cells to remember

# ── CENTROID TRACKER FOR MINIATURE ID & PATH ───────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.next_object_id = 0
        self.objects = {}         # object_id -> deque of centroids
        self.disappeared = {}     # object_id -> consecutive missing frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = deque([centroid], maxlen=64)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        # If no detections, mark all as disappeared
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # First frame, register everything
        if not self.objects:
            for c in input_centroids:
                self.register(c)
            return self.objects

        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid][-1] for oid in object_ids]
        D = dist.cdist(np.array(object_centroids), np.array(input_centroids))

        # Match by minimal distance
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols or D[row, col] > self.max_distance:
                continue
            oid = object_ids[row]
            self.objects[oid].append(input_centroids[col])
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Check for disappeared
        for row in set(range(D.shape[0])) - used_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # Register new
        for col in set(range(len(input_centroids))) - used_cols:
            self.register(input_centroids[col])

        return self.objects

def classify_path(pts):
    """Label the movement as static / straight / curved / unusual."""
    if len(pts) < 3:
        return "static"
    arr = np.array(pts)
    dx = np.diff(arr[:, 0])
    dy = np.diff(arr[:, 1])
    angles = np.arctan2(dy, dx)
    angle_deltas = np.diff(angles)
    total_curve = np.sum(np.abs(angle_deltas))
    if total_curve < np.pi/4:
        return "straight"
    elif total_curve < np.pi:
        return "curved"
    else:
        return "unusual"

# ── GRID COVERAGE FUNCTION (UNCHANGED) ─────────────────────────────────────────
def get_grid_cells_covered_by_mask(mask, grid_cols=23, grid_rows=16, warp_w=400, warp_h=400):
    col_w = warp_w // grid_cols
    row_h = warp_h // grid_rows
    covered = set()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        col_start = max(0, x // col_w)
        col_end   = min(grid_cols - 1, (x + w) // col_w)
        row_start = max(0, y // row_h)
        row_end   = min(grid_rows - 1, (y + h) // row_h)
        for r in range(row_start, row_end + 1):
            for c in range(col_start, col_end + 1):
                label = chr(ord('A')+c) + str(r+1) if c<26 else f"C{c}{r+1}"
                covered.add(label)
    return sorted(covered)

# ── HOMOGRAPHY & REAL-TIME LOOP WITH TRACKING ─────────────────────────────────
def homography(camera_index=s.load_last_selection()["webcam_index"], dictionary=cv2.aruco.DICT_6X6_250):
    global persistent_cells
    background_frame = None
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    # aspect ratio setup from reference image
    image_ref = cv2.imread("maps/dnd1.jpg")
    if image_ref is None:
        print("❌ Could not load reference image.")
        return
    img_h, img_w = image_ref.shape[:2]
    warp_h = 400
    warp_w = int(warp_h * (img_w/img_h))

    # ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    params = cv2.aruco.DetectorParameters()
    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        modern = True
    except AttributeError:
        modern = False

    # load camera calibration
    camera_matrix = np.load('camera_matrix.npy')
    dist_coeffs   = np.load('dist_coeffs.npy')

    # ── instantiate tracker ──────────────────────────────────────────────────
    tracker = CentroidTracker(max_disappeared=30, max_distance=80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21,21), 0)

        if background_frame is None:
            background_frame = capture_background(camera_index)

        # reset background on 'r'
        if cv2.waitKey(1) & 0xFF == ord('r'):
            background_frame = blur.copy()
            print("🔄 Background reset.")

        diff = cv2.absdiff(background_frame, blur)
        _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.erode(diff_mask, None, iterations=2)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        cv2.imshow("Difference Mask", diff_mask)

        if modern:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, params)

        warped = None
        if ids is not None and len(ids) >= 4:
            # draw ArUco
            for i in range(len(ids)):
                pts = corners[i][0].astype(int)
                cv2.polylines(frame, [pts], True, (255,0,0), 2)
                cx, cy = int(pts[:,0].mean()), int(pts[:,1].mean())
                cv2.circle(frame, (cx,cy), 5, (0,0,255), -1)

            try:
                src = get_extreme_corners(corners, ids)
                dst = np.array([[0,0],[warp_w,0],[warp_w,warp_h],[0,warp_h]],dtype=np.float32)
                H, _ = cv2.findHomography(src, dst)
                if H is not None:
                    warped = cv2.warpPerspective(frame, H, (warp_w,warp_h))
                    warp_mask = cv2.warpPerspective(diff_mask, H, (warp_w,warp_h))

                    # ── GRID CELL UPDATE ─────────────────────────────────
                    current_cells = get_grid_cells_covered_by_mask(
                        warp_mask, grid_cols=23, grid_rows=16, warp_w=warp_w, warp_h=warp_h
                    )
                    cell_history.append(set(current_cells))
                    persistent_cells.clear()
                    for s in cell_history:
                        persistent_cells |= s
                    if persistent_cells:
                        print("Miniatures at:", ", ".join(sorted(persistent_cells)))

                    # ── MINIATURE TRACKING ─────────────────────────────
                    cnts, _ = cv2.findContours(warp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    centroids = []
                    for cnt in cnts:
                        x,y,w,h = cv2.boundingRect(cnt)
                        centroids.append((x+w//2, y+h//2))
                    objects = tracker.update(centroids)
                    for oid, pts in objects.items():
                        x,y = pts[-1]
                        cv2.circle(warped, (x, y), 5, (0, 0, 0), -1)
                        cv2.putText(warped, f"ID{oid}", (x+5, y-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(warped, path_type, (x+5, y+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    # ── DRAW GRID & LABELS ────────────────────────────
                    for r in range(16):
                        for c in range(23):
                            if c>0: cv2.line(warped, (c*(warp_w//23),0), (c*(warp_w//23),warp_h), (0,255,0),1)
                            if r>0: cv2.line(warped, (0,r*(warp_h//16)), (warp_w,r*(warp_h//16)), (0,255,0),1)
                            lbl = chr(ord('A')+c) + str(r+1)
                            sz = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.2, 1)[0]
                            cv2.putText(warped, lbl,
                                        (c*(warp_w//23)+4, r*(warp_h//16)+sz[1]+4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0,255,0), 1)

                    # homography quad
                    cv2.polylines(frame, [np.int32(src)], True, (0,255,0), 2)

            except Exception as e:
                print("⚠️ Homography failed:", e)

        # show count of markers
        h, w = frame.shape[:2]
        cv2.putText(frame, f"{len(marker_centers) if 'marker_centers' in locals() else 0}",
                    (w//2-10, h//2+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 4)

        cv2.imshow("Live ArUco Detection", frame)
        if warped is not None:
            cv2.imshow("Warped Homography View", warped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ── SUPPORTING FUNCTIONS ───────────────────────────────────────────────────────
def get_extreme_corners(corners, ids):
    marker_data = [(ids[i][0], corners[i][0]) for i in range(len(ids))]
    centers = {mid: pts.mean(axis=0) for mid, pts in marker_data}
    sorted_m = sorted(centers.items(), key=lambda t: (t[1][1], t[1][0]))
    top2, bot2 = sorted_m[:2], sorted_m[2:4]
    tl = min(top2, key=lambda t: t[1][0])[0]
    tr = max(top2, key=lambda t: t[1][0])[0]
    bl = min(bot2, key=lambda t: t[1][0])[0]
    br = max(bot2, key=lambda t: t[1][0])[0]
    by_id = {mid: pts for mid, pts in marker_data}
    return np.array([
        by_id[tl][0], by_id[tr][1], by_id[br][2], by_id[bl][3]
    ], dtype=np.float32)

def capture_background(camera_index=s.load_last_selection()["webcam_index"]):
    cap = cv2.VideoCapture(camera_index)
    print("📸 Capturing background...")
    time.sleep(1)
    for _ in range(10):
        cap.read()
    ret, bg = cap.read()
    cap.release()
    if not ret:
        print("❌ Could not capture background.")
        return None
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21,21), 0)

def begin_session():
    print("🔄 Starting tracking session...")
    homography()

if __name__ == "__main__":
    begin_session()
