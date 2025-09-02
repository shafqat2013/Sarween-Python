import cv2
import numpy as np
import os
from screeninfo import get_monitors

# Parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 960
MARGIN_PX = 30
NUM_FRAMES = 10

def display_charuco_board_fullscreen():
    # Generate ChArUco board image
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
                                    SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(
        board, (LENGTH_PX, int(LENGTH_PX * size_ratio)), marginSize=MARGIN_PX)

    # Detect monitors
    monitors = get_monitors()
    if len(monitors) < 2:
        print("⚠️ Only one display detected. Showing on primary display.")
        x, y = 0, 0
    else:
        # Use the second monitor (usually index 1)
        second = monitors[1]
        x, y = second.x, second.y

    # Display fullscreen window at the chosen monitor location
    cv2.namedWindow("CharucoBoard", cv2.WINDOW_AUTOSIZE)
    #cv2.moveWindow("CharucoBoard", x, y)
    #cv2.setWindowProperty("CharucoBoard", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("CharucoBoard", img)

    cv2.waitKey(1)  # keep window open until user starts capture


def calibrate_and_save_parameters():
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    used_frames = []  # Store frames that were used
    captured = 0

    print("📸 Capturing 10 frames with visible ChArUco board...")

    while captured < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        if marker_ids is not None and len(marker_ids) > 0:
            charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, board
            )
            if charuco_retval:
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)
                used_frames.append(frame.copy())
                captured += 1
                print(f"✅ Captured frame {captured}/10")

        # Optional preview
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        cv2.imshow("Capture Preview", frame)
        cv2.waitKey(100)

    cap.release()
    cv2.destroyAllWindows()

    # Calibrate
    print("🔧 Calibrating camera...")
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners, all_charuco_ids, board, gray.shape[::-1], None, None
    )

    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
    print("✅ Calibration complete. Saved camera_matrix.npy and dist_coeffs.npy.")

    # Display original vs undistorted frames
    for i, frame in enumerate(used_frames):
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

        combined = np.hstack((frame, undistorted))
        cv2.imshow(f"Frame {i+1}: Original (L) | Undistorted (R)", combined)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

# === MAIN ===
display_charuco_board_fullscreen()
input("📷 Press Enter to start capturing webcam frames for calibration...")
calibrate_and_save_parameters()
