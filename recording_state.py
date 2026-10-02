"""Portable replay checkpoints: numeric arrays and JSON only, never pickle."""

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

CORE_ARRAYS = ("H_saved", "_H_inv", "last_mask_cam", "last_mask_warp", "BG_warp_f32",
               "_heal_acc", "_last_shadowfree_cam", "_last_final_mask_cam")
CORE_SCALARS = ("frame_idx", "warp_w", "warp_h", "grid_w", "grid_h", "aruco_every_n",
                "aruco_every_n_fast", "lock_drop_after", "warp_motion_thresh", "fog_change_ratio",
                "bg_alpha_slow", "bg_alpha_fast", "view_settle_seconds", "_view_settle_frames",
                "_view_transform_revision", "_scene_visual_revision", "_bg_seeded", "_shadow_frame_count",
                "_SHADOW_EVERY_N", "_heal_ms", "_fps_estimate", "_last_frame_time", "last_marker_count",
                "last_seen_ids", "last_missing_ids", "lock_miss_streak", "lock_lost_reason")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def save_checkpoint(path, session, tracking=None):
    arrays = {name: getattr(session, name) for name in CORE_ARRAYS
              if getattr(session, name, None) is not None}
    arrays.update({"BG_cam_" + key: value for key, value in session.BG_cam.items()})
    state = {name: json_value(getattr(session, name)) for name in CORE_SCALARS}
    state["tracking"] = json_value(tracking or {})
    arrays["metadata"] = np.frombuffer(json.dumps(state, allow_nan=False).encode(), dtype=np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_path(timeline_path, entry):
    root = Path(timeline_path).resolve().parent
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Replay checkpoint escapes the recording folder")
    if path.stat().st_size > 100_000_000:
        raise ValueError("Replay checkpoint is too large")
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Replay checkpoint checksum does not match")
    with zipfile.ZipFile(path) as archive:
        if sum(info.file_size for info in archive.infolist()) > 200_000_000:
            raise ValueError("Replay checkpoint expands beyond the safety limit")
    return path


def restore_checkpoint(timeline_path, entry, session, engine):
    path = checkpoint_path(timeline_path, entry)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        background_only = entry.get("kind") == "background"
        names = ("BG_warp_f32",) if background_only else CORE_ARRAYS
        for name in names:
            setattr(session, name, archive[name].copy() if name in archive else None)
        session.BG_cam = {key: archive["BG_cam_" + key].copy() for key in ("bgr", "blur")}
        if background_only:
            session._bg_seeded = metadata["_bg_seeded"]
            return None
        for name in CORE_SCALARS:
            if name != "frame_idx":
                setattr(session, name, copy.deepcopy(metadata[name]))
        session._frame_phase_offset = int(metadata["frame_idx"]) + 1 - int(entry["frame"])
        tracking = metadata.get("tracking", {})
        if tracking.get("engine"):
            engine.restore(tracking["engine"])
        return tracking.get("profiles")
