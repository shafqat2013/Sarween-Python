# tv_tracking.py
import time
import cv2
import numpy as np

def get_extreme_corners(corners, ids):
    """Pick the four outer corners from four detected markers."""
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
        by_id[tl][0],   # top-left
        by_id[tr][1],   # top-right
        by_id[br][2],   # bottom-right
        by_id[bl][3],   # bottom-left
    ], dtype=np.float32)

class TVTracker:
    def __init__(self, camera_index=0, map_image_path="maps/dnd1.jpg", warp_h=400,
                 aruco_dict_type=cv2.aruco.DICT_6X6_250):
        # open camera
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("Could not open webcam.")
        # load map reference to get aspect ratio
        image_ref = cv2.imread(map_image_path)
        if image_ref is None:
            raise IOError(f"Could not load map image at {map_image_path}")
        h, w = image_ref.shape[:2]
        self.warp_h = warp_h
        self.warp_w = int(w/h * warp_h)
        # camera calibration
        self.camera_matrix = np.load('camera_matrix.npy')
        self.dist_coeffs   = np.load('dist_coeffs.npy')
        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.parameters = cv2.aruco.DetectorParameters()
        try:
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.modern = True
        except AttributeError:
            self.modern = False
        # background reference (gray-blurred)
        self.background_frame = None

    def capture_background(self):
        """Grab a clean background reference for diff."""
        time.sleep(1)
        for _ in range(10):
            self.cap.read()
        ret, raw = self.cap.read()
        if not ret:
            raise IOError("Failed to capture background frame.")
        und = cv2.undistort(raw, self.camera_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        self.background_frame = cv2.GaussianBlur(gray, (21,21), 0)

    def reset_background(self):
        self.capture_background()

    def process_frame(self):
        """
        Reads one frame, undistorts, finds ArUco, computes homography.
        Returns: (orig_frame, warped_frame, H) or (orig_frame, None, None)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None
        frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect markers
        if self.modern:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, self.parameters)

        if ids is None or len(ids) < 4:
            return frame, None, None

        # compute homography
        src = get_extreme_corners(corners, ids)
        dst = np.array([
            [0,         0],
            [self.warp_w, 0],
            [self.warp_w, self.warp_h],
            [0,         self.warp_h],
        ], dtype=np.float32)

        H, _ = cv2.findHomography(src, dst)
        warped = cv2.warpPerspective(frame, H, (self.warp_w, self.warp_h))
        return frame, warped, H

    def release(self):
        self.cap.release()
