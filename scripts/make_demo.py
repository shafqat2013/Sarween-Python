"""Generate deterministic, synthetic footage. No personal or third-party assets."""
import json
from pathlib import Path
import cv2
import numpy as np


def generate(folder):
    folder.mkdir(parents=True, exist_ok=True)
    fps, seconds = 10, 24
    colors = {"Red": (60, 60, 215), "Blue": (215, 105, 45), "Yellow": (40, 205, 225),
              "Green": (90, 190, 55), "White": (212, 212, 212)}
    starts = {"Red": (2, 3), "Blue": (6, 3), "Yellow": (10, 3), "Green": (14, 3), "White": (17, 3)}
    moves = [(4 + i * 3, name, (starts[name][0], 8)) for i, name in enumerate(colors)]
    moves.append((20, "Red", (4, 8)))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    markers = [cv2.aruco.generateImageMarker(dictionary, i, 48) for i in range(4)]
    background = np.full((520, 840, 3), 245, dtype=np.uint8)
    background[20:500, 20:820] = (44, 48, 47)
    for col in range(21):
        cv2.line(background, (20 + col*40, 20), (20 + col*40, 499), (68, 75, 70), 1)
    for row in range(13):
        cv2.line(background, (20, 20 + row*40), (819, 20 + row*40), (68, 75, 70), 1)
    for marker, (x, y) in zip(markers, ((20, 20), (772, 20), (772, 452), (20, 452))):
        background[y-6:y+54, x-6:x+54] = 255
        background[y:y+48, x:x+48] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    writer = cv2.VideoWriter(str(folder / "five_minis.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (840, 520))
    if not writer.isOpened():
        raise RuntimeError("Demo video encoder unavailable")
    try:
        for frame in range(fps * seconds):
            pixels = background.copy()
            positions = starts.copy()
            for at, name, destination in moves:
                if frame / fps >= at:
                    positions[name] = destination
            if frame >= fps:
                for name, (col, row) in positions.items():
                    x, y = 40 + col*40, 40 + row*40
                    cv2.circle(pixels, (x, y), 16, colors[name], -1, cv2.LINE_AA)
                    cv2.circle(pixels, (x, y), 9, (24, 28, 32), -1, cv2.LINE_AA)
                    cv2.fillConvexPoly(pixels, np.array([(x, y-10), (x-6, y+6), (x+7, y+5)]), (115, 124, 130))
            writer.write(pixels)
    finally:
        writer.release()
    profiles = {}
    for name, bgr in colors.items():
        lab = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        profiles[name] = {"lab": [lab[0]*100/255, lab[1]-128, lab[2]-128], "presence_lab_dist": 12}
    def write(name, value):
        (folder / name).write_text(json.dumps(value, indent=2) + "\n")
    write("five_minis.profiles.json", profiles)
    write("five_minis.tracking.json", {"schemaVersion": 1, "synthetic": True, "fps": fps,
        "gridCols": 20, "gridRows": 12, "warpWidth": 800, "warpHeight": 480, "markerMode": "legacy",
        "framesUndistorted": True, "events": [{"frame": 0, "sceneInfo": {"sceneId": "synthetic-demo",
            "width": 800, "height": 480, "gridSize": 40, "gridType": 1},
            "trackingControls": {"paused": False, "selectedMini": None}}]})
    def cell(position):
        return chr(65 + position[0]) + str(position[1] + 1)
    expectations = [{"mini": name, "to": cell(position), "at": 1.6} for name, position in starts.items()]
    previous = starts.copy()
    for at, name, destination in moves:
        expectations.append({"mini": name, "from": cell(previous[name]), "to": cell(destination), "at": at + .6})
        previous[name] = destination
    write("five_minis.case.json", {"cases": [{"name": "synthetic-five-minis", "video": "five_minis.mp4",
        "profiles": "five_minis.profiles.json", "label_source": "synthetic-scripted", "expectations": expectations,
        "tolerance_seconds": 1.0, "allow_unexpected": False}]})


if __name__ == "__main__":
    generate(Path(__file__).resolve().parents[1] / "demo")
