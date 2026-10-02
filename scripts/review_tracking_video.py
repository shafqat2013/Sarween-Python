"""Extract independent visual-review evidence, without loading mini predictions.

Only viewport recordings with already-undistorted frames are supported. Each
frame must detect all four markers; missing registration is never filled in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np


def geometry_at(timeline: dict, frame: int) -> tuple[dict, dict]:
    scene, view = {}, {}
    for event in sorted(timeline.get("events", []), key=lambda e: e["frame"]):
        if event["frame"] > frame:
            break
        if "sceneInfo" in event:
            scene = event["sceneInfo"] or {}
        if "viewTransform" in event:
            view = event["viewTransform"] or {}
    if not scene or not view or scene.get("sceneId") != view.get("sceneId"):
        raise ValueError(f"No matching scene/viewport geometry at frame {frame}")
    return scene, view


def scene_matrix(view: dict, width: int, height: int) -> np.ndarray:
    r, t = view["registration"], view["clientToCanvasTransform"]
    client = np.array([[(r["right"] - r["left"]) / (width - 1), 0, r["left"]],
                       [0, (r["bottom"] - r["top"]) / (height - 1), r["top"]],
                       [0, 0, 1]], dtype=float)
    canvas = np.array([[t["a"], t["c"], t["tx"]],
                       [t["b"], t["d"], t["ty"]], [0, 0, 1]], dtype=float)
    matrix = canvas @ client
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError("Invalid viewport transform")
    return matrix


def cell_at(view: dict, scene: dict, width: int, height: int,
            x: float, y: float) -> str | None:
    if not (0 <= x < width and 0 <= y < height):
        return None
    sx, sy, _ = scene_matrix(view, width, height) @ [x, y, 1]
    if not (0 <= sx < scene["width"] and 0 <= sy < scene["height"]):
        return None
    size = float(view["gridSize"])
    col = math.floor((sx - view.get("gridOriginX", 0)) / size)
    row = math.floor((sy - view.get("gridOriginY", 0)) / size)
    if col < 0 or row < 0:
        return None
    return f"{column_name(col)}{row + 1}"


def column_name(col: int) -> str:
    result = ""
    col += 1
    while col:
        col, digit = divmod(col - 1, 26)
        result = chr(65 + digit) + result
    return result


def registration(frame: np.ndarray, detector, width: int, height: int):
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    markers = {} if ids is None else {int(i): c[0] for i, c in zip(ids.ravel(), corners)}
    seen = sorted(markers)
    if not all(i in markers for i in (10, 11, 12, 13)):
        return None, seen
    source = np.float32([markers[10][0], markers[11][1], markers[12][2], markers[13][3]])
    dest = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    return cv2.getPerspectiveTransform(source, dest), seen


def grid_overlay(frame: np.ndarray, scene: dict, view: dict) -> np.ndarray:
    height, width = frame.shape[:2]
    inverse = np.linalg.inv(scene_matrix(view, width, height))
    size = float(view["gridSize"])
    if not math.isfinite(size) or size <= 0:
        raise ValueError("Invalid grid size")
    ox, oy = view.get("gridOriginX", 0), view.get("gridOriginY", 0)
    cols = math.ceil((scene["width"] - ox) / size)
    rows = math.ceil((scene["height"] - oy) / size)
    if not (0 < cols <= 500 and 0 < rows <= 500):
        raise ValueError("Review supports up to 500 rows/columns")
    result = frame.copy()

    def point(x, y):
        px, py, _ = inverse @ [x, y, 1]
        return int(round(px)), int(round(py))

    for col in range(cols + 1):
        cv2.line(result, point(ox + col * size, oy),
                 point(ox + col * size, oy + rows * size), (90, 90, 90), 1)
    for row in range(rows + 1):
        cv2.line(result, point(ox, oy + row * size),
                 point(ox + cols * size, oy + row * size), (90, 90, 90), 1)
    for row in range(rows):
        for col in range(cols):
            x, y = point(ox + (col + .08) * size, oy + (row + .40) * size)
            if 0 <= x < width and 0 <= y < height:
                cv2.putText(result, f"{column_name(col)}{row + 1}", (x, y),
                            cv2.FONT_HERSHEY_PLAIN, .55, (180, 180, 180), 1)
    return result


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def extract(args) -> Path:
    cv2.setNumThreads(0)
    video = args.video.resolve()
    timeline_path = video.with_suffix(".tracking.json")
    timeline = json.loads(timeline_path.read_text())
    if timeline.get("markerMode") != "viewport" or timeline.get("framesUndistorted") is not True:
        raise ValueError("Review requires a viewport recording with framesUndistorted=true")
    width, height = int(timeline["warpWidth"]), int(timeline["warpHeight"])
    if not (2 <= width <= 4096 and 2 <= height <= 4096):
        raise ValueError("Invalid warp dimensions")
    if not math.isfinite(args.deadline) or args.deadline <= 0:
        raise ValueError("Deadline must be finite and positive")
    deadline = time.monotonic() + args.deadline
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    cap = cv2.VideoCapture(str(video))
    try:
        fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError("Cannot read video metadata")
        if args.frames:
            indices = sorted(set(int(n) for n in args.frames.split(",")))
        else:
            if not math.isfinite(args.step) or args.step <= 0:
                raise ValueError("Step must be finite and positive")
            indices = range(0, count, max(1, round(args.step * fps)))
        if not indices or len(indices) > 120 or min(indices) < 0 or max(indices) >= count:
            raise ValueError("Choose 1-120 valid zero-based video frames")
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        records, tiles = [], []
        for index in indices:
            if time.monotonic() > deadline:
                raise TimeoutError("Review extraction deadline exceeded")
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, raw = cap.read()
            if not ok or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != index + 1:
                raise ValueError(f"Failed exact seek to frame {index}")
            homography, seen = registration(raw, detector, width, height)
            record = {"frame": index, "seconds": index / fps, "markers": seen,
                      "registered": homography is not None}
            if homography is None:
                display = raw
            else:
                scene, view = geometry_at(timeline, index)
                display = cv2.warpPerspective(raw, homography, (width, height))
                record.update(homography=homography.tolist(), scene=scene, view=view)
                cv2.imwrite(str(output / f"frame-{index:05d}-grid.jpg"), grid_overlay(display, scene, view))
                if args.point:
                    record["point"] = args.point
                    record["cell"] = cell_at(view, scene, width, height, *args.point)
            cv2.imwrite(str(output / f"frame-{index:05d}.jpg"), display)
            tile = np.zeros((384, 640, 3), np.uint8)
            tile[:360] = cv2.resize(display, (640, 360))
            caption = f"frame {index} | {index / fps:.3f}s | " + ("registered" if homography is not None else "UNREGISTERED")
            cv2.putText(tile, caption, (8, 377), cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
            tiles.append(tile)
            records.append(record)
        for start in range(0, len(tiles), 9):
            page = tiles[start:start + 9]
            while len(page) % 3:
                page.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(page[i:i + 3]) for i in range(0, len(page), 3)])
            cv2.imwrite(str(output / f"sheet-{start // 9 + 1:02d}.jpg"), sheet)
        manifest = {"video": str(video), "video_sha256": digest(video),
                    "timeline_sha256": digest(timeline_path), "fps": fps,
                    "frame_convention": "zero-based; seconds = frame / nominal video FPS",
                    "method": "Independent manual visual review; only marker and scene geometry used. No mini detector, profile, expected move, or tracking event consulted.",
                    "frames": records}
        path = output / "evidence.json"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        return path
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--step", type=float, default=1, help="Nominal seconds between overview frames")
    parser.add_argument("--frames", help="Comma-separated zero-based frames, instead of --step")
    parser.add_argument("--point", type=float, nargs=2, metavar=("X", "Y"), help="Map a manually identified rectified-frame point to a cell")
    parser.add_argument("--deadline", type=float, default=45, help="Cooperative extraction deadline in seconds")
    args = parser.parse_args()
    print(extract(args))


if __name__ == "__main__":
    main()
