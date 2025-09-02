# mini_tracking.py
import cv2
import numpy as np
from collections import deque
from tv_tracking import TVTracker, get_grid_cells_covered_by_mask
from scipy.spatial import distance as dist

# ── centroid tracker & path classifier ──────────────────────────────────────
class CentroidTracker:
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}       # id -> deque of centroids
        self.disappeared = {}   # id -> missing frames
        self.max_disappeared = max_disappeared

    def register(self, c):
        self.objects[self.next_id] = deque([c], maxlen=64)
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, input_centroids):
        if not input_centroids:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        if not self.objects:
            for c in input_centroids:
                self.register(c)
            return self.objects

        oids = list(self.objects)
        prev_centroids = [self.objects[oid][-1] for oid in oids]
        D = dist.cdist(np.array(prev_centroids), np.array(input_centroids))
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols or D[r,c] > 1e6:
                continue
            oid = oids[r]
            self.objects[oid].append(input_centroids[c])
            self.disappeared[oid] = 0
            used_rows.add(r)
            used_cols.add(c)

        for r in set(range(D.shape[0])) - used_rows:
            oid = oids[r]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        for c in set(range(len(input_centroids))) - used_cols:
            self.register(input_centroids[c])

        return self.objects

def classify_path(pts):
    if len(pts) < 3:
        return "static"
    arr = np.array(pts)
    dx, dy = np.diff(arr[:,0]), np.diff(arr[:,1])
    ang = np.arctan2(dy, dx)
    curve = np.sum(np.abs(np.diff(ang)))
    if curve < np.pi/4:  return "straight"
    if curve < np.pi:    return "curved"
    return "unusual"

def main():
    tv = TVTracker()
    tv.capture_background()

    tracker = CentroidTracker(max_disappeared=30)
    cell_history = deque(maxlen=5)
    persistent_cells = set()

    while True:
        orig, warped, H = tv.process_frame()
        if orig is None:
            break

        # compute diff mask on the original frame
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21,21), 0)
        diff = cv2.absdiff(tv.background_frame, blur)
        _, diff_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        diff_mask = cv2.erode(diff_mask, None, iterations=2)
        diff_mask = cv2.dilate(diff_mask, None, iterations=2)

        # if we got a valid homography, warp the mask & track
        if warped is not None and H is not None:
            warp_mask = cv2.warpPerspective(diff_mask, H, (tv.warp_w, tv.warp_h))

            # update grid occupancy
            cells = get_grid_cells_covered_by_mask(
                warp_mask, grid_cols=23, grid_rows=16,
                warp_w=tv.warp_w, warp_h=tv.warp_h
            )
            cell_history.append(set(cells))
            persistent_cells.clear()
            for s in cell_history:
                persistent_cells |= s

            if persistent_cells:
                print("Cells:", ", ".join(sorted(persistent_cells)))

            # detect mini centroids
            cnts, _ = cv2.findContours(warp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            centroids = [(x + w//2, y + h//2)
                         for c in cnts
                         for x,y,w,h in [cv2.boundingRect(c)]]

            objects = tracker.update(centroids)
            for oid, pts in objects.items():
                x,y = pts[-1]
                cv2.circle(warped, (x,y), 5, (0,0,0), -1)
                cv2.putText(warped, f"ID{oid}", (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                cv2.putText(warped, classify_path(list(pts)), (x+5, y+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            # show grid & labels (reuse your helper or inline draw here)
            # … your grid‐drawing code …

            cv2.imshow("Warped Homography View", warped)

        cv2.imshow("Live View", orig)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            tv.reset_background()
        elif key == ord('q'):
            break

    tv.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
