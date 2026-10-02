"""Restore the camera angle of approved, privacy-edited clips for real replay.

The tabletop pixels come from the approved clean exports. Original pixels are
used only around the four physical markers so their cropped borders remain
detectable. Everything outside the table/marker regions is masked. No tracking
positions, rendered markers, synthetic frames or timing changes are introduced.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def prepare(source, clips, analyses, profile, output):
    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=True)
    segments = []
    for stem in ("10-two-moves", "11-three-moves"):
        meta = json.loads((analyses / (stem + "-analysis.json")).read_text())
        video = clips / (stem + "-soft-reflection-clean.mp4")
        cap = cv2.VideoCapture(str(video))
        first = round(meta["in"] * meta["fps"])
        segments.append((first, first + int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                         np.linalg.inv(np.array(meta["homography"])), cap, video))
    raw = cv2.VideoCapture(str(source))
    fps = raw.get(cv2.CAP_PROP_FPS)
    width, height = int(raw.get(3)), int(raw.get(4))
    first, end = segments[0][0], segments[-1][1]
    raw.set(cv2.CAP_PROP_POS_FRAMES, first)
    ok, initial = raw.read()
    if not ok:
        raise ValueError("Cannot decode source footage")
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250))
    corners, ids, _ = detector.detectMarkers(initial)
    found = {int(i): p[0] for p, i in zip(corners, ids.flatten())} if ids is not None else {}
    if not all(i in found for i in (10, 11, 12, 13)):
        raise ValueError("The first source frame must contain the four original markers")
    marker_mask = np.zeros((height, width), np.uint8)
    for marker_id in (10, 11, 12, 13):
        points = found[marker_id]
        expanded = points.mean(axis=0) + (points - points.mean(axis=0)) * 1.4
        cv2.fillConvexPoly(marker_mask, np.round(expanded).astype(np.int32), 255)
    masks = [cv2.warpPerspective(np.full((720, 1280), 255, np.uint8), s[2], (width, height))
             for s in segments]
    bounds = np.maximum(marker_mask, np.maximum.reduce(masks))
    x, y, w, h = cv2.boundingRect(bounds)
    x, y = max(0, x - 8), max(0, y - 8)
    w, h = min(width-x, w+16), min(height-y, h+16)
    w, h = w//2*2, h//2*2
    destination = output / "tabletop.mp4"
    encoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2", "-preset", "fast",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], stdin=subprocess.PIPE)
    raw.set(cv2.CAP_PROP_POS_FRAMES, first)
    try:
        for frame in range(first, end):
            ok, original = raw.read()
            if not ok:
                raise ValueError(f"Missing source frame {frame}")
            index = 0 if frame < segments[0][1] else 1
            begin, _, inverse, cap, _ = segments[index]
            offset = frame - begin
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != offset:
                cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
            ok, clean = cap.read()
            if not ok:
                raise ValueError(f"Missing approved clip frame {offset}")
            table = cv2.warpPerspective(clean, inverse, (width, height))
            image = np.full_like(original, (32, 29, 26))
            image[masks[index] > 250] = table[masks[index] > 250]
            image[marker_mask > 0] = original[marker_mask > 0]
            encoder.stdin.write(image[y:y+h, x:x+w].tobytes())
    finally:
        encoder.stdin.close()
        raw.release()
        for segment in segments:
            segment[3].release()
    if encoder.wait(timeout=60):
        raise RuntimeError("Video encoding failed")
    timeline = json.loads(source.with_suffix(".tracking.json").read_text())
    previous = max((e for e in timeline["events"] if e["frame"] <= first), key=lambda e:e["frame"])
    events = [dict(previous, frame=first)] + [e for e in timeline["events"] if first < e["frame"] < end]
    cleaned = []
    for event in events:
        event = copy.deepcopy({k:v for k,v in event.items() if k in (
            "frame", "sceneInfo", "viewTransform", "sceneVisualRevision", "sceneVisualReason")})
        event["frame"] -= first
        for key in ("sceneInfo", "viewTransform"):
            if event.get(key):
                event[key]["sceneId"] = "tabletop-example"
                event[key].pop("background", None)
        cleaned.append(event)
    metadata = {k:timeline[k] for k in ("schemaVersion", "markerMode", "framesUndistorted",
        "warpWidth", "warpHeight", "gridCols", "gridRows")}
    metadata.update(video=destination.name, title="Real tabletop example", synthetic=False,
        description="Real recording · surroundings masked and reflection blurred",
        fps=fps, frameWidth=w, frameHeight=h, framesRecorded=end-first, events=cleaned)
    (output / "tabletop.tracking.json").write_text(json.dumps(metadata, indent=2)+"\n")
    profiles = json.loads(profile.read_text())
    (output / "tabletop.profiles.json").write_text(json.dumps({"Red mini": profiles["red10"]}, indent=2)+"\n")
    (output / "tabletop.provenance.json").write_text(json.dumps({
        "source": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "first_source_frame": first, "end_source_frame_exclusive": end, "fps": fps,
        "edits": "Approved clean clip pixels restored to the camera angle; original marker borders retained; surroundings masked; silent; original speed.",
        "approved_exports": {s[4].name:hashlib.sha256(s[4].read_bytes()).hexdigest() for s in segments},
        "tracking": "Recomputed from tabletop.mp4 by the normal replay worker. No saved positions or synthetic movement."}, indent=2)+"\n")
    print(destination, f"{end-first} frames, {w}x{h}, {fps:.3f} fps")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "clips", "analyses", "profile", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.clips, args.analyses, args.profile, args.output)
