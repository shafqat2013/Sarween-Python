"""Offline, observational comparison of recorded camera and Foundry frames.

The reference JPEGs are extracted from Foundry's canvas while video recording.
They are aligned to the same marker-center rectangle as the camera warp. This
prototype reports evidence only; it never changes tracking or sends moves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import cv_core
import foundryoutput as fo
import v3_tracking as tracking
from mini_calibration import load_profiles_with_curves
from app_paths import data_path
from tracking_regression import FoundryTimelineReplay
from reference_health import MAX_REFERENCE_AGE_SECONDS, image_quality, matching_view, validated_view


def difference_mask(camera: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compare low-frequency color after compensating for TV/camera gain."""
    height, width = reference.shape[:2]
    camera = cv2.resize(camera, (width, height), interpolation=cv2.INTER_AREA)
    camera = cv2.GaussianBlur(camera, (5, 5), 0).astype(np.float32)
    reference = cv2.GaussianBlur(reference, (5, 5), 0).astype(np.float32)
    corrected = np.empty_like(reference)
    for channel in range(3):
        ref_channel = reference[:, :, channel]
        cam_channel = camera[:, :, channel]
        ref_low, ref_high = np.percentile(ref_channel, (10, 90))
        cam_low, cam_high = np.percentile(cam_channel, (10, 90))
        gain = np.clip((cam_high - cam_low) / max(ref_high - ref_low, 1), 0.5, 2.0)
        corrected[:, :, channel] = (ref_channel - np.median(ref_channel)) * gain + np.median(cam_channel)
    residual = np.max(np.abs(camera - corrected), axis=2)
    threshold = max(25.0, float(np.median(residual)) * 3.0)
    return np.where(residual > threshold, 255, 0).astype(np.uint8)


def load_references(timeline_path: Path) -> dict[int, tuple[dict, np.ndarray]]:
    data = json.loads(timeline_path.read_text(encoding="utf-8"))
    references = {}
    for entry in data.get("referenceFrames", []):
        path = (timeline_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(timeline_path.parent.resolve()):
            raise ValueError(f"Reference path escapes recording directory: {path}")
        image = cv2.imread(str(path))
        if image is None or image.shape[:2] != (360, 640):
            raise ValueError(f"Missing or invalid rendered reference: {path}")
        # Recompute content statistics rather than trust old or edited metadata.
        references[int(entry["frame"])] = ({**entry, "quality": image_quality(image)}, image)
    return references


def reference_for_camera(references, capture_time, view, *, max_age=MAX_REFERENCE_AGE_SECONDS):
    """Choose a past, recent, informative reference with captured geometry.

    Receipt and sender wall clocks only estimate camera alignment. This is
    sufficient for inspecting a diagnostic, never for authorizing token moves.
    Legacy snapshots cannot establish timing and are deliberately excluded.
    """
    choices = []
    for entry, image in references.values():
        provenance = entry.get("provenance") or {}
        if not provenance.get("verified") or not entry.get("quality", {}).get("informative"):
            continue
        if not matching_view(entry.get("viewTransform"), view):
            continue
        try:
            validated_view(entry.get("viewTransform"), entry.get("sceneId"))
            source_time = float(entry["receivedAtMonotonic"]) - float(provenance["ageAtReceiptSeconds"])
            age = float(capture_time) - source_time
            if not np.isfinite([source_time, age]).all() or not 0 <= age <= max_age:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        choices.append((source_time, entry, image, age))
    if not choices:
        return None
    _source_time, entry, image, age = max(choices, key=lambda item: item[0])
    return entry, image, age


def candidate_evidence(mask: np.ndarray, detections: dict, warp_size: tuple[int, int]) -> list[dict]:
    evidence = []
    sx = mask.shape[1] / warp_size[0]
    sy = mask.shape[0] / warp_size[1]
    for name, detection in detections.items():
        if detection is None:
            continue
        x, y, w, h = detection.bbox
        x0, y0 = max(0, int(x * sx)), max(0, int(y * sy))
        x1, y1 = min(mask.shape[1], int((x + w) * sx) + 1), min(mask.shape[0], int((y + h) * sy) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = mask[y0:y1, x0:x1]
        evidence.append({
            "mini": name,
            "warpXY": [round(detection.cx, 1), round(detection.cy, 1)],
            "candidateChangedPercent": round(100.0 * cv2.countNonZero(crop) / crop.size, 2),
        })
    return evidence


def compare_video(video_path: Path, timeline_path: Path, max_frames: int | None = None) -> list[dict]:
    references = load_references(timeline_path)
    if not references:
        raise ValueError(
            "This recording has no rendered references. Older videos cannot reconstruct "
            "the actual fog/lighting; record a new session with the updated module."
        )
    if not any(entry.get("provenance", {}).get("verified") and entry["quality"]["informative"]
               for entry, _image in references.values()):
        raise ValueError(
            "No usable rendered reference evidence: exports are low-information or lack "
            "explicit-render provenance and capture clocks. Uniform black may be legitimate fog; "
            "it cannot establish export correctness or alignment. Reload the updated module and record again."
        )
    last_reference_frame = max(references) + 1
    max_frames = min(max_frames, last_reference_frame) if max_frames is not None else last_reference_frame
    timeline = FoundryTimelineReplay(timeline_path)
    timeline.apply_through(0)
    profiles_path = video_path.with_suffix(".profiles.json")
    if not profiles_path.exists():
        profiles_path = data_path("combo_profiles.json")
    profiles = load_profiles_with_curves(profiles_path)
    prev_state = {}
    sess = cv_core.CVCoreSession(
        source_path=str(video_path),
        marker_mode=str(timeline.data.get("markerMode") or "viewport"),
        warp_w=int(timeline.data["warpWidth"]),
        warp_h=int(timeline.data["warpHeight"]),
        grid_w=int(timeline.data["gridCols"]),
        grid_h=int(timeline.data["gridRows"]),
        before_frame_callback=timeline.apply_through,
        source_frames_undistorted=bool(timeline.data.get("framesUndistorted", True)),
    )
    results = []
    capture_times = {int(row["frame"]): row.get("captureTime") for row in timeline.data.get("frameClocks", [])}
    try:
        for bundle in sess.frames():
            frame = max(0, int(sess.cap.get(cv2.CAP_PROP_POS_FRAMES) or bundle.frame_idx) - 1)
            if frame > max_frames:
                break
            reference = reference_for_camera(references, capture_times.get(frame), fo.get_view_transform_payload())
            if reference is None or not bundle.locked or bundle.warp_bgr is None:
                for name in profiles:
                    previous = prev_state.setdefault(name, {})
                    previous["change_hist"] = tracking._updated_change_history(
                        previous, float(bundle.raw_motion_ratio or 0)
                    )
                continue
            entry, image, age = reference
            if entry["sceneId"] != fo.SCENE_ID or not matching_view(
                entry.get("viewTransform"), fo.get_view_transform_payload()
            ):
                continue
            grid_px = tracking._grid_px_for_bundle(bundle)
            if profiles and grid_px is not None:
                detections, prev_state = tracking.detect_minis(
                    bundle, profiles, grid_px, prev_state
                )
            else:
                detections = {}
            mask = difference_mask(bundle.warp_bgr, image)
            results.append({
                "cameraFrame": frame,
                "referenceFrame": int(entry["frame"]),
                "changedPercent": round(100.0 * cv2.countNonZero(mask) / mask.size, 2),
                "viewRevision": entry["viewRevision"],
                "referenceAgeSeconds": round(age, 6),
                "synchronizationVerified": False,
                "diagnosticOnly": True,
                "candidates": candidate_evidence(
                    mask, detections, (bundle.warp_w, bundle.warp_h)
                ),
            })
    finally:
        sess.close()
    if not results:
        raise ValueError("No locked camera frames matched a reference scene and viewport.")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare recorded camera frames to clean Foundry canvas snapshots.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    video = args.video.expanduser().resolve()
    timeline = (args.timeline or video.with_suffix(".tracking.json")).expanduser().resolve()
    try:
        print(json.dumps(compare_video(video, timeline, args.max_frames), indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"rendered_reference: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
