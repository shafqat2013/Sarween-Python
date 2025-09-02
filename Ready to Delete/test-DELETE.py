import cv2
import numpy as np
import os

# === CONFIG ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
marker_size = 50  # inner black marker size
border_ratio = 0.1  # white border around marker and background
alpha = 0.6  # marker transparency

# Use your screen resolution as maximum allowed dimensions
screen_width = 3024
screen_height = 1964

os.makedirs("markers", exist_ok=True)

# === Step 1: Generate ArUco markers with white border ===
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

border_thickness = int(marker_size * border_ratio)

for i in range(4):
    try:
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
    except AttributeError:
        marker_img = np.zeros((marker_size, marker_size), dtype=np.uint8)
        cv2.aruco.drawMarker(aruco_dict, i, marker_size, marker_img, 1)

    marker_with_border = add_white_border(marker_img, border_thickness)
    cv2.imwrite(f"markers/marker_{i}.png", marker_with_border)
    print(f"✅ Saved marker_{i}.png with white border")

# === Step 2: Load and resize background image with white border ===
background_img = cv2.imread('maps/dnd1.jpg')
cv2.namedWindow("background", cv2.WINDOW_AUTOSIZE)
cv2.imshow("background", background_img)

bg_h, bg_w = background_img.shape[:2]
available_width = screen_width - 2 * border_thickness
available_height = screen_height - 2 * border_thickness

# Resize while maintaining aspect ratio
scale = min(available_width / bg_w, available_height / bg_h)
resized_bg = cv2.resize(background_img, (int(bg_w * scale), int(bg_h * scale)))
cv2.namedWindow("resized background", cv2.WINDOW_AUTOSIZE)
cv2.imshow("resized background", resized_bg)

# Add white border to match marker border
resized_bg_with_border = add_white_border(resized_bg, border_thickness)

# Final image for blending
blended_img = resized_bg_with_border.copy()
img_h, img_w = blended_img.shape[:2]

# === Step 3: Warp and alpha-blend markers into each corner ===
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

# === Step 5: ArUco Detection via Webcam ===
cap = cv2.VideoCapture(0)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

print("📷 Press 'q' to quit webcam view.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to capture frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray_frame)

    if corners and ids is not None:
        for i, pts in enumerate(corners):
            pts = pts[0].astype(int)
            center = tuple(np.mean(pts, axis=0).astype(int))
            pts_poly = pts.reshape((-1, 1, 2)).astype(np.int32)

            cv2.polylines(frame, [pts_poly], True, (255, 0, 0), 3)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {ids[i][0]}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print("Detected IDs:", ids.ravel().tolist())

    cv2.putText(frame, "Press 'q' to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Webcam Marker Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
