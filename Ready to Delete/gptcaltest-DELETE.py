import os
import time
import math
import cv2
import numpy as np
import setup as h

def generate_aruco_markers(marker_count=4, marker_size=100, padding=10):
    os.makedirs("markers", exist_ok=True)

    final_size = marker_size + 2 * padding  # size including padding

    try:
        if hasattr(cv2.aruco, "getPredefinedDictionary"):  # OpenCV 4.7+
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
            for i in range(marker_count):
                # Create white background
                marker_image = np.full((final_size, final_size), 255, dtype=np.uint8)
                # Create marker only
                marker_only = np.zeros((marker_size, marker_size), dtype=np.uint8)
                cv2.aruco.generateImageMarker(aruco_dict, i, marker_size, marker_only)
                # Paste marker into the center of the white image
                marker_image[padding:padding+marker_size, padding:padding+marker_size] = marker_only
                cv2.imwrite(f"markers/marker_{i}.png", marker_image)
        else:
            print("Error. This shouldn't happen, I've commented out the fallback method.")
        #     aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        #     for i in range(marker_count):
        #         marker_image = np.full((final_size, final_size), 255, dtype=np.uint8)
        #         marker_only = np.zeros((marker_size, marker_size), dtype=np.uint8)
        #         cv2.aruco.drawMarker(aruco_dict, i, marker_size, marker_only, 1)
        #         marker_image[padding:padding+marker_size, padding:padding+marker_size] = marker_only
        #         cv2.imwrite(f"markers/marker_{i}.png", marker_image)

        print(f"✅ {marker_count} ArUco markers with padding saved in 'markers/'.")

    except AttributeError as e:
        print("❌ Error: OpenCV ArUco module not found! Install with 'opencv-contrib-python'.")
        print(e)

def generate_display(marker_size=100):
    print("🛠 Starting Calibration (Markers Just Outside Image Corners)...")

    padding = int(marker_size * 0.1)
    padded_marker_size = marker_size + 2 * padding

    screen_w = h.selected_display["width"]
    screen_h = h.selected_display["height"]
    screen_x = h.selected_display["x"]
    screen_y = h.selected_display["y"]

    generate_aruco_markers(marker_count=4, marker_size=marker_size, padding=padding)

    # Space for content (inset from screen to leave room for markers)
    content_w = screen_w - 2 * padded_marker_size
    content_h = screen_h - 2 * padded_marker_size

    content_image = cv2.imread("img/dnd1.jpg")
    if content_image is None:
        print("❌ Could not load dnd1.jpg")
        return

    img_h, img_w = content_image.shape[:2]
    aspect_img = img_w / img_h
    aspect_target = content_w / content_h

    # Resize image with preserved aspect ratio
    if aspect_img > aspect_target:
        new_w = content_w
        new_h = int(content_w / aspect_img)
    else:
        new_h = content_h
        new_w = int(content_h * aspect_img)

    resized_img = cv2.resize(content_image, (new_w, new_h))

    # Draw 23x16 blue grid on resized image
    cols = 23
    rows = 16
    col_w = new_w // cols
    row_h = new_h // rows

    for i in range(1, cols):
        x = i * col_w
        cv2.line(resized_img, (x, 0), (x, new_h), color=(255, 0, 0), thickness=1)

    for j in range(1, rows):
        y = j * row_h
        cv2.line(resized_img, (0, y), (new_w, y), color=(255, 0, 0), thickness=1)

    # Create white canvas to hold resized image in center
    content_canvas = np.full((content_h, content_w, 3), 255, dtype=np.uint8)
    offset_x = (content_w - new_w) // 2
    offset_y = (content_h - new_h) // 2
    content_canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_img

    # Create full screen white image
    full_image = np.full((screen_h, screen_w, 3), 255, dtype=np.uint8)
    content_x = padded_marker_size
    content_y = padded_marker_size
    full_image[content_y:content_y + content_h, content_x:content_x + content_w] = content_canvas

    # Load ArUco markers
    markers = []
    for i in range(4):
        marker_path = f"markers/marker_{i}.png"
        marker = cv2.imread(marker_path, cv2.IMREAD_GRAYSCALE)
        markers.append(cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR))

    # ✅ Position ArUco markers just outside the corners of the resized image
    image_x = content_x + offset_x
    image_y = content_y + offset_y

    positions = [
        (image_x - padded_marker_size, image_y - padded_marker_size),             # top-left
        (image_x + new_w,             image_y - padded_marker_size),             # top-right
        (image_x - padded_marker_size, image_y + new_h),                         # bottom-left
        (image_x + new_w,             image_y + new_h)                           # bottom-right
    ]

    for marker, (x, y) in zip(markers, positions):
        x0 = max(x, 0)
        y0 = max(y, 0)
        x1 = min(x + padded_marker_size, screen_w)
        y1 = min(y + padded_marker_size, screen_h)

        marker_x0 = x0 - x
        marker_y0 = y0 - y
        marker_x1 = marker_x0 + (x1 - x0)
        marker_y1 = marker_y0 + (y1 - y0)

        full_image[y0:y1, x0:x1] = marker[marker_y0:marker_y1, marker_x0:marker_x1]

    # Save the result
    cv2.imwrite("calibration_display.jpg", full_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print("🖼 Saved calibration image as 'calibration_display.jpg'")

    # Display fullscreen
    window_name = "Calibration Display"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow(window_name, screen_x, screen_y)
    cv2.imshow(window_name, full_image)
    cv2.waitKey(1)  # Let it render first frame

    # Force fullscreen after display
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



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

    # #outside corners
    # src_pts = np.array([
    #     id_to_corners[top_left_marker][0],      # top-left corner
    #     id_to_corners[top_right_marker][1],     # top-right corner
    #     id_to_corners[bottom_right_marker][2],  # bottom-right corner
    #     id_to_corners[bottom_left_marker][3],   # bottom-left corner
    # ], dtype=np.float32)

    #inside corners
    src_pts = np.array([
        id_to_corners[top_left_marker][2],  # bottom-right corner
        id_to_corners[top_right_marker][3],   # bottom-left corner
        id_to_corners[bottom_right_marker][0],      # top-left corner
        id_to_corners[bottom_left_marker][1],     # top-right corner
    ], dtype=np.float32)

    return src_pts

def process_aruco_with_homography(camera_index=0, dictionary=cv2.aruco.DICT_6X6_250):
    # Load reference image to get aspect ratio
    image_ref = cv2.imread("img/dnd1.jpg")
    if image_ref is None:
        print("❌ Could not load reference image for aspect ratio.")
        return

    img_h, img_w = image_ref.shape[:2]
    aspect_img = img_w / img_h

    # Set base height and compute width to preserve aspect ratio
    warp_h = 400
    warp_w = int(warp_h * aspect_img)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("❌ Failed to open webcam.")
        return

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    parameters = cv2.aruco.DetectorParameters()

    try:
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        use_modern_detector = True
    except AttributeError:
        use_modern_detector = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
                    # Detect red regions in the warped view
                    color_mask = find_color_regions(warped)
                    covered_cells = get_grid_cells_covered_by_mask(color_mask, grid_cols=23, grid_rows=16, warp_w=warp_w, warp_h=warp_h)

                    # Visualize red mask (optional)
                    cv2.imshow("Red Mask", _mask)

                    # Print covered cells
                    if covered_cells:
                        print(f"🔴 Red disk overlaps: {', '.join(covered_cells)}")

                    # Grid and labeling
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
        cv2.imshow("Live ArUco Detection", frame)
        if warped is not None:
            cv2.imshow("Warped Homography View", warped)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def shrink_aruco_markers(initial_size=200, min_size=20, camera_index=0):
    print("🔍 Starting auto-shrink ArUco process...")
    marker_size = initial_size

    success_start_time = None
    stable_duration = 2.0  # seconds required for successful detection
    window_name = "Calibration Display"

    while marker_size >= min_size:
        padding = int(marker_size * 0.1)
        padded_marker_size = marker_size + 2 * padding

        # Generate and display markers
        generate_aruco_markers(marker_count=4, marker_size=marker_size, padding=padding)

        # Display fullscreen calibration image
        width = h.selected_display["width"]
        height = h.selected_display["height"]
        x_offset = h.selected_display["x"]
        y_offset = h.selected_display["y"]

        image = np.full((height, width, 3), 255, dtype=np.uint8)

        # Load and place markers
        markers = []
        for i in range(4):
            path = f"markers/marker_{i}.png"
            marker = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            markers.append(marker)

        positions = [
            (0, 0),
            (width - padded_marker_size, 0),
            (0, height - padded_marker_size),
            (width - padded_marker_size, height - padded_marker_size)
        ]
        for marker, (x, y) in zip(markers, positions):
            image[y:y + padded_marker_size, x:x + padded_marker_size] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.moveWindow(window_name, x_offset, y_offset)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(window_name, image)

        # Start webcam detection
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Failed to open webcam.")
            return None

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        try:
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            use_modern_detector = True
        except AttributeError:
            use_modern_detector = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if use_modern_detector:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters)

            marker_centers = []
            if ids is not None and len(corners) >= 4:
                for corner in corners:
                    pts = corner[0]
                    cx = int(np.mean(pts[:, 0]))
                    cy = int(np.mean(pts[:, 1]))
                    marker_centers.append((cx, cy))

                if len(marker_centers) == 4:
                    if success_start_time is None:
                        success_start_time = time.time()
                    elif time.time() - success_start_time >= stable_duration:
                        print(f"✅ Quadrilateral stable for {stable_duration} seconds. Shrinking marker...")
                        marker_size -= 20
                        cap.release()
                        cv2.destroyWindow(window_name)
                        break  # Restart outer loop with smaller marker
                    
                else:
                    success_start_time = None
            else:
                success_start_time = None

            # Feedback
            cv2.putText(frame, f"Size: {marker_size} | Detected: {len(marker_centers)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Live ArUco", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return marker_size

        # Cleanup
        cap.release()
        cv2.destroyWindow("Live ArUco")

    print(f"🛑 Finished shrinking. Final marker size: {marker_size}")
    return marker_size

def find_color_regions(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue
    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])

    color_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    return color_mask

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
                label = f"{chr(ord('A') + col)}{row + 1}"
                covered_cells.add(label)

    return sorted(covered_cells)

# ===========================
# FULL PIPELINE
# ===========================

def calibrate():
    print("🧪 Calibrating ArUco Marker Size...")
    marker_size = 100
    # marker_size = shrink_aruco_markers(camera_index=h.selected_webcam["index"])
    # if marker_size is None:
    #     print("❌ No detectable marker size found. Calibration aborted.")
    #     return
    generate_display(marker_size=marker_size)
    print("✅ Calibration Complete.")

h.initialize()
calibrate()
process_aruco_with_homography()