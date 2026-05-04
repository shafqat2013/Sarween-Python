import cv2


def list_cameras(max_tested=10):
    available = []
    print("Available cameras:")
    for i in range(max_tested):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[{i}] Camera index {i}")
                available.append(i)
        cap.release()
    return available


def main():
    # --- pick camera ---
    cams = list_cameras()
    if not cams:
        raise RuntimeError("No cameras found.")

    cam_index = int(input("Select camera index: "))
    cap = cv2.VideoCapture(cam_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {cam_index}")

    # --- aruco setup ---
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    print("Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                pts = marker_corners.reshape((4, 2)).astype(int)

                # blue outline
                cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

                # label at top-left corner (more stable than center)
                x, y = pts[0]
                cv2.putText(
                    frame,
                    f"ID {marker_id}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("ArUco Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()