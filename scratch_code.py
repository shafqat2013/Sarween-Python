import os
import json
import cv2
import numpy as np


def add_mini_to_database(cap, apply_homography, get_diff_mask, data_dir='mini_data', db_path='mini_database.json'):
    """
    Interactive function to add a new miniature to the database.

    Parameters:
    - cap: cv2.VideoCapture object for reading frames
    - apply_homography: function that takes a frame and returns the warped map-aligned image
    - get_diff_mask: function that takes (warped_background, warped_current) and returns a binary mask
    - data_dir: directory to save vision data files
    - db_path: path to the JSON database file

    Steps:
    1. Prompt user to clear the map and press 'r' to reset the background frame.
    2. Prompt user to place the mini and press 'c' to capture vision data.
    3. Save contour data, color histogram, and Hu moments to files.
    4. Prompt user for mini metadata (name, size, type).
    5. Append entry to database JSON and return the new record.
    """
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Load or initialize database
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            db = json.load(f)
    else:
        db = {'minis': []}

    # Determine next mini ID
    mini_id = len(db['minis']) + 1

    # Step 1: Reset background frame
    print("Clear the map, then press 'r' to reset the background frame.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Add Mini', frame)
        if cv2.waitKey(1) & 0xFF == ord('r'):
            background = frame.copy()
            print("Background frame reset.")
            break

    # Step 2: Capture mini vision data
    print("Place the mini on the map, then press 'c' to capture vision data.")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Add Mini', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            current = frame.copy()
            print("Capturing vision data...")
            break
    cv2.destroyWindow('Add Mini')

    # Warp images to map coordinates
    warped_bg = apply_homography(background)
    warped_current = apply_homography(current)

    # Compute difference mask
    mask = get_diff_mask(warped_bg, warped_current)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save contour data
    contour_file = os.path.join(data_dir, f"contours_{mini_id}.npy")
    np.save(contour_file, contours)

    # Compute and save color histogram (HSV)
    hsv = cv2.cvtColor(warped_current, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1,2], mask, [8,8,8], [0,180,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    hist_file = os.path.join(data_dir, f"hist_{mini_id}.npy")
    np.save(hist_file, hist)

    # Compute and save Hu moments for the largest contour
    if contours:
        c_max = max(contours, key=cv2.contourArea)
        moments = cv2.moments(c_max)
        hu = cv2.HuMoments(moments).flatten()
    else:
        hu = np.zeros(7)
    hu_file = os.path.join(data_dir, f"hu_{mini_id}.npy")
    np.save(hu_file, hu)

    # Step 3: Prompt for mini metadata
    name = input("Enter mini name: ")
    size = input("Enter mini size: ")
    type_ = input("Enter mini type: ")

    # Create database entry
    entry = {
        'mini_id': mini_id,
        'contour_file': contour_file,
        'hist_file': hist_file,
        'hu_file': hu_file,
        'name': name,
        'size': size,
        'type': type_
    }

    # Append and save database
    db['minis'].append(entry)
    with open(db_path, 'w') as f:
        json.dump(db, f, indent=2)

    print(f"Mini '{name}' saved with ID {mini_id}.")
    return entry


import cv2
from your_module import add_mini_to_database, apply_homography, get_diff_mask

cap = cv2.VideoCapture(0)
entry = add_mini_to_database(cap, apply_homography, get_diff_mask)
