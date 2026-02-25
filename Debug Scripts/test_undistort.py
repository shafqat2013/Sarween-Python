import cv2
import numpy as np

cam_mtx = np.load("camera_matrix.npy")
dist    = np.load("dist_coeffs.npy")

cap = cv2.VideoCapture(1)  # or your camera index

while True:
    ok, frame = cap.read()
    if not ok:
        break

    undist = cv2.undistort(frame, cam_mtx, dist)

    # Show side by side
    both = cv2.hconcat([frame, undist])
    cv2.putText(both, "RAW", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(both, "UNDIST", (frame.shape[1] + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.namedWindow("Raw vs Undistorted", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Raw vs Undistorted", 1280, 720)
    cv2.imshow("Raw vs Undistorted", both)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
