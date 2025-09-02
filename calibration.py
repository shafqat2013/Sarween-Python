import os
import cv2
import numpy as np
import setup as s

from screeninfo import get_monitors #for charuco, but should be moved to hardware.py

# ===========================
# CAMERA DISTORTION CALIBRATION (ONCE PER WEBCAM)
# ===========================

# Parameters for charuco
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
SQUARES_VERTICALLY = 7
SQUARES_HORIZONTALLY = 5
SQUARE_LENGTH = 0.03
MARKER_LENGTH = 0.015
LENGTH_PX = 960
MARGIN_PX = 30
NUM_FRAMES = 10

# Parameters for NEW display + map setup (should be run every time a new map or display is selected)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 100  # inner black marker size
border_ratio = 0.1  # white border around marker and background
alpha = 0.6  # marker transparency
border_thickness = int(marker_size * border_ratio)

# Use your screen resolution as maximum allowed dimensions, should come from setup.py
screen_width = 3024
screen_height = 1964
#screen_width = s.selected_display["width"]
#screen_height = s.selected_display["height"]

# Store submitted grid size
submitted_grid = {"cols": None, "rows": None}


def display_charuco():
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

def calibrate_webcam(camera_index=s.load_last_selection()["webcam_index"]):
    dictionary = ARUCO_DICT
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    all_charuco_corners = []
    all_charuco_ids = []
    used_frames = []
    captured = 0

    print("📸 Press 'c' to capture a frame with a visible ChArUco board.")
    print("📷 Need 10 captures total. Press 'q' to quit early.")

    while captured < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        # Optional preview
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        cv2.imshow("Capture Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
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
                else:
                    print("⚠️ Not enough ChArUco corners found. Try again.")
            else:
                print("⚠️ No ArUco markers detected. Try again.")

        elif key == ord('q'):
            print("❌ Quit by user before completing all captures.")
            cap.release()
            cv2.destroyAllWindows()
            return

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

# ===========================
# DISPLAY AND MAP CALIBRATION (ONCE PER DISPLAY + MAP COMBO)
# ===========================

# === Add white border to asset ===
def add_white_border(img, border_thickness):
    if len(img.shape) == 2:
        value = 255
    else:
        value = (255, 255, 255)
    return cv2.copyMakeBorder(
        img,
        top=border_thickness,
        bottom=border_thickness,
        left=border_thickness,
        right=border_thickness,
        borderType=cv2.BORDER_CONSTANT,
        value=value
    )

def generate_aruco_markers(marker_size, border_thickness):
    for i in range(4):
        try:
            marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
        except AttributeError:
            marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
            cv2.aruco.drawMarker(aruco_dict, i, marker_size, marker_img, 1)

        marker_with_border = add_white_border(marker_img, border_thickness)
        cv2.imwrite(f"markers/marker_{i}.png", marker_with_border)
        print(f"✅ Saved marker_{i}.png with white border")

def generate_grid(img, grid_cols, grid_rows, color=(255, 0, 0), thickness=2, label_cells=True):
    img_with_grid = img.copy()
    h, w = img.shape[:2]

    col_width = w / grid_cols
    row_height = h / grid_rows

    # Draw vertical lines
    for i in range(1, grid_cols):
        x = int(i * col_width)
        cv2.line(img_with_grid, (x, 0), (x, h), color, thickness)

    # Draw horizontal lines
    for j in range(1, grid_rows):
        y = int(j * row_height)
        cv2.line(img_with_grid, (0, y), (w, y), color, thickness)

    # Add cell labels (A1, B2, ...)
    if label_cells:
        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_x = int(col * col_width + 5)
                cell_y = int((row + 1) * row_height - 5)
                label = (
                    f"{chr(65 + col)}{row + 1}"
                    if col < 26
                    else f"{chr(65 + (col // 26) - 1)}{chr(65 + (col % 26))}{row + 1}"
                )
                cv2.putText(img_with_grid, label, (cell_x, cell_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img_with_grid

def generate_grid_with_sliders(img):
    window_name = "Interactive Grid"
    cv2.namedWindow(window_name)

    submitted_grid = {"cols": None, "rows": None}
    init_cols, init_rows = 23, 17
    max_val = 50
    running = [True]

    h, w = img.shape[:2]
    submit_btn = (w - 200, h - 60, w - 100, h - 30)
    cancel_btn = (w - 90, h - 60, w - 10, h - 30)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if submit_btn[0] <= x <= submit_btn[2] and submit_btn[1] <= y <= submit_btn[3]:
                submitted_grid["cols"] = cv2.getTrackbarPos("Cols", window_name)
                submitted_grid["rows"] = cv2.getTrackbarPos("Rows", window_name)
                print(f"✅ Submitted: {submitted_grid['cols']} cols, {submitted_grid['rows']} rows")
                running[0] = False
            elif cancel_btn[0] <= x <= cancel_btn[2] and cancel_btn[1] <= y <= cancel_btn[3]:
                print("❌ Cancelled grid selection.")
                submitted_grid["cols"] = None
                submitted_grid["rows"] = None
                running[0] = False

    cv2.setMouseCallback(window_name, on_mouse)

    cv2.createTrackbar("Cols", window_name, init_cols, max_val, lambda x: None)
    cv2.createTrackbar("Rows", window_name, init_rows, max_val, lambda x: None)

    while running[0]:
        cols = max(1, cv2.getTrackbarPos("Cols", window_name))
        rows = max(1, cv2.getTrackbarPos("Rows", window_name))

        grid_img = generate_grid(img, cols, rows)

        # Draw submit button
        cv2.rectangle(grid_img, submit_btn[:2], submit_btn[2:], (0, 255, 0), -1)
        cv2.putText(grid_img, "Submit", (submit_btn[0] + 10, submit_btn[3] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw cancel button
        cv2.rectangle(grid_img, cancel_btn[:2], cancel_btn[2:], (0, 0, 255), -1)
        cv2.putText(grid_img, "Cancel", (cancel_btn[0] + 10, cancel_btn[3] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow(window_name, grid_img)

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(30)
        if key == 27:  # ESC to quit
            running[0] = False

    cv2.destroyWindow(window_name)
    return submitted_grid["cols"], submitted_grid["rows"]

def prepare_map_asset(border_thickness):
    background_img = cv2.imread('maps/dnd1.jpg')

    # Let user define the grid
    cols, rows = generate_grid_with_sliders(background_img)
    background_img = generate_grid(background_img, cols, rows)

    # You can store this for later use if needed
    s.grid_cols = cols
    s.grid_rows = rows

    bg_h, bg_w = background_img.shape[:2]
    available_width = screen_width - 2 * border_thickness
    available_height = screen_height - 2 * border_thickness

    # Resize while maintaining aspect ratio
    scale = min(available_width / bg_w, available_height / bg_h)
    resized_bg = cv2.resize(background_img, (int(bg_w * scale), int(bg_h * scale)))

    # Add white border to match marker border
    resized_bg_with_border = add_white_border(resized_bg, border_thickness)

    # Final image for blending
    blended_img = resized_bg_with_border.copy()
    img_h, img_w = blended_img.shape[:2]
    
    return blended_img, img_h, img_w

def generate_display():
    blended_img, img_h, img_w = prepare_map_asset(border_thickness)

    for i in range(4):
        marker = cv2.imread(f'markers/marker_{i}.png', cv2.IMREAD_GRAYSCALE)
        marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        h, w = marker.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        # Target corners
        top = 0
        bottom = img_h
        left = 0
        right = img_w

        if i == 0:  # top-left
            dst_pts = np.array([[left, top], [left + w, top], [left + w, top + h], [left, top + h]], dtype=np.float32)
        elif i == 1:  # top-right
            dst_pts = np.array([[right - w, top], [right, top], [right, top + h], [right - w, top + h]], dtype=np.float32)
        elif i == 2:  # bottom-right
            dst_pts = np.array([[right - w, bottom - h], [right, bottom - h], [right, bottom], [right - w, bottom]], dtype=np.float32)
        elif i == 3:  # bottom-left
            dst_pts = np.array([[left, bottom - h], [left + w, bottom - h], [left + w, bottom], [left, bottom]], dtype=np.float32)

        marker_mask = np.ones(marker.shape, dtype=np.uint8) * 255
        H, _ = cv2.findHomography(src_pts, dst_pts)
        warped_marker = cv2.warpPerspective(marker_bgr, H, (img_w, img_h), flags=cv2.INTER_NEAREST)
        warped_mask = cv2.warpPerspective(marker_mask, H, (img_w, img_h), flags=cv2.INTER_NEAREST)

        warped_mask_bool = warped_mask > 0
        for c in range(3):
            blended_img[..., c][warped_mask_bool] = (
                (1 - alpha) * blended_img[..., c][warped_mask_bool] +
                alpha * warped_marker[..., c][warped_mask_bool]
            ).astype(np.uint8)

    # === Step 4: Show/save final image ===
    cv2.imwrite("blended_output.jpg", blended_img)
    cv2.namedWindow("Blended ArUco Markers", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Blended ArUco Markers", blended_img)
    cv2.waitKey(0)

# def shrink_aruco_markers(initial_size=200, min_size=20, camera_index=0):
#     print("🔍 Starting auto-shrink ArUco process...")
#     marker_size = initial_size

#     success_start_time = None
#     stable_duration = 2.0  # seconds required for successful detection
#     window_name = "Calibration Display"

#     while marker_size >= min_size:
#         padding = int(marker_size * 0.1)
#         padded_marker_size = marker_size + 2 * padding

#         # Generate and display markers
#         generate_aruco_markers(marker_count=4, marker_size=marker_size, padding=padding)

#         # Display fullscreen calibration image
#         width = s.selected_display["width"]
#         height = s.selected_display["height"]
#         x_offset = s.selected_display["x"]
#         y_offset = s.selected_display["y"]

#         image = np.full((height, width, 3), 255, dtype=np.uint8)

#         # Load and place markers
#         markers = []
#         for i in range(4):
#             path = f"markers/marker_{i}.png"
#             marker = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             markers.append(marker)

#         positions = [
#             (0, 0),
#             (width - padded_marker_size, 0),
#             (0, height - padded_marker_size),
#             (width - padded_marker_size, height - padded_marker_size)
#         ]
#         for marker, (x, y) in zip(markers, positions):
#             image[y:y + padded_marker_size, x:x + padded_marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

#         #cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
#         #cv2.moveWindow(window_name, x_offset, y_offset)
#         #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#         cv2.imshow(window_name, image)

#         # Start webcam detection
#         cap = cv2.VideoCapture(camera_index)
#         if not cap.isOpened():
#             print("❌ Failed to open webcam.")
#             return None

#         aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#         parameters = cv2.aruco.DetectorParameters()

#         try:
#             detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
#             use_modern_detector = True
#         except AttributeError:
#             use_modern_detector = False

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             if use_modern_detector:
#                 corners, ids, _ = detector.detectMarkers(gray)
#             else:
#                 corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters)

#             marker_centers = []
#             if ids is not None and len(corners) >= 4:
#                 for corner in corners:
#                     pts = corner[0]
#                     cx = int(np.mean(pts[:, 0]))
#                     cy = int(np.mean(pts[:, 1]))
#                     marker_centers.append((cx, cy))

#                 if len(marker_centers) == 4:
#                     if success_start_time is None:
#                         success_start_time = time.time()
#                     elif time.time() - success_start_time >= stable_duration:
#                         print(f"✅ Quadrilateral stable for {stable_duration} seconds. Shrinking marker...")
#                         marker_size -= 20
#                         cap.release()
#                         cv2.destroyWindow(window_name)
#                         break  # Restart outer loop with smaller marker
                    
#                 else:
#                     success_start_time = None
#             else:
#                 success_start_time = None

#             # Feedback
#             cv2.putText(frame, f"Size: {marker_size} | Detected: {len(marker_centers)}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#             cv2.imshow("Live ArUco", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 cap.release()
#                 cv2.destroyAllWindows()
#                 return marker_size

#         # Cleanup
#         cap.release()
#         cv2.destroyWindow("Live ArUco")

#     print(f"🛑 Finished shrinking. Final marker size: {marker_size}")
#     return marker_size

# ===========================
# FULL PIPELINE
# ===========================

def calibrate():

   #new webcam calibration, new display and map calibration, generate grid, store image + grid data for this map/display combo

    #should only be run if new webcam selected
    #print("new webcam detected, calibrating...")
    #display_charuco()
    #input("📷 Press Enter to start capturing webcam frames for calibration...")
    #calibrate_webcam()
    #currently, this is overwriting the camera_matrix.npy and dist_coeffs.npy files, but we should save them with a unique name based on the webcam index or timestamp
    
    #should be run every time a new map image is loaded
    generate_aruco_markers(marker_size, border_thickness)
    #print("🧪 Calibrating ArUco Marker Size...")
    # marker_size = shrink_aruco_markers(camera_index=s.selected_webcam["index"])
    # if marker_size is None:
    #     print("❌ No detectable marker size found. Calibration aborted.")
    #     return
    generate_display()
    #currently, this is overwriting the blended_output.jpg file, but we should save it with a unique name based on the map name and display index
    print("✅ Calibration Complete.")