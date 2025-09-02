
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

    screen_w = s.selected_display["width"]
    screen_h = s.selected_display["height"]
    screen_x = s.selected_display["x"]
    screen_y = s.selected_display["y"]

    generate_aruco_markers(marker_count=4, marker_size=marker_size, padding=padding)

    # Space for content (inset from screen to leave room for markers)
    content_w = screen_w - 2 * padded_marker_size
    content_h = screen_h - 2 * padded_marker_size

    content_image = cv2.imread("maps/dnd1.jpg")
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



