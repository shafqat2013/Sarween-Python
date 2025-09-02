import cv2
cap = cv2.VideoCapture(1)
cv2.namedWindow('Camera Test', cv2.WINDOW_AUTOSIZE)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break
    cv2.imshow('Camera Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break