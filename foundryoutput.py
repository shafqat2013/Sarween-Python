import asyncio
import base64
import binascii
import copy
import json
import math
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import websockets
from guided_capture import GuidedCapture, regression_case

# ──────────────────────────────────────────────────────────────────────────────
# Foundry configuration
# ──────────────────────────────────────────────────────────────────────────────

# SCENE_ID is not hardcoded — it is populated when Foundry sends sceneInfo on
# connect (see recv_loop / set_scene_params). Moves queued before sceneInfo
# arrives will use whatever value is set here at that point; in practice the
# module.js sends sceneInfo proactively on open so the window is very small.
SCENE_ID = ""

# Default token (used if we can't resolve a specific mapping)
DEFAULT_TOKEN_ID = "1"  # fallback

# Map from mini_id (or logical mini key) -> Foundry tokenId.
# Populated at runtime from mini_token_map.json via _load_mapping().
# Do not hardcode IDs here — use the assignment flow in-game instead.
MINI_TO_TOKEN = {}
SCENE_TOKEN_NAMES = {}

# Persist mappings across runs
MAP_PATH = Path(__file__).with_name("mini_token_map.json")

# Foundry scene size (fallback defaults; can be overridden by sceneInfo)
SCENE_W = 1656   # dnd1.jpg width
SCENE_H = 1152   # dnd1.jpg height
SCENE_BACKGROUND = None  # background image path/URL from Foundry

# Foundry grid metadata (NEW; prefer these if present)
GRID_PX = None   # pixels per grid square
GRID_TYPE = None
SHIFT_X = 0      # grid offset in px
SHIFT_Y = 0

# Grid settings (override from tracking if needed)
_grid_cols = 23
_grid_rows = 16

# Async bits
_move_queue: asyncio.Queue | None = None
_assign_queue: asyncio.Queue | None = None
_ctrl_queue: asyncio.Queue | None = None  # NEW: control messages (getSceneInfo, etc.)
_loop: asyncio.AbstractEventLoop | None = None

# Live Foundry viewport registration. The browser sends this whenever the GM
# pans or zooms, allowing a fixed physical point on the TV to be mapped back to
# the correct cell on an arbitrarily large Foundry scene.
_view_transform: dict | None = None
_view_transform_revision = 0
_view_transform_lock = threading.Lock()
_scene_visual_revision = 0
_scene_visual_reason = "initial"
_test_sequence: dict | None = None

# Recording metadata is global so it survives CV session handoffs during mini
# calibration while the same VideoWriter remains open.
_timeline_lock = threading.Lock()
_timeline_recording: dict | None = None
_timeline_path: Path | None = None
_timeline_last_signature: str | None = None
_capture_lock = threading.Lock()
_capture_session: GuidedCapture | None = None
_capture_commands = deque()
_capture_ready = False
_tracking_output_paused = False
_camera_lock_state = {"locked": False, "missingIds": []}
_capture_status = {"type": "captureStatus", "state": "idle"}
_selected_mini_id = None


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def _load_mapping() -> None:
    global MINI_TO_TOKEN
    if MAP_PATH.exists():
        try:
            data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                MINI_TO_TOKEN.update({str(k): str(v) for k, v in data.items() if v})
                print(f"FoundryOutput | Loaded {len(data)} mini→token mappings from {MAP_PATH}")
        except Exception as e:
            print("FoundryOutput | Failed to load mapping:", e)

def _save_mapping() -> None:
    try:
        MAP_PATH.write_text(json.dumps(MINI_TO_TOKEN, indent=2), encoding="utf-8")
        print(f"FoundryOutput | Saved mini→token mappings to {MAP_PATH}")
    except Exception as e:
        print("FoundryOutput | Failed to save mapping:", e)


def _remove_stale_token_mapping(token_id: str) -> list[str]:
    """Remove and persist every mini mapping that points at a missing token."""
    token_id = str(token_id or "").strip()
    if not token_id:
        return []
    stale_minis = [
        mini_id
        for mini_id, mapped_token_id in MINI_TO_TOKEN.items()
        if str(mapped_token_id) == token_id
    ]
    for mini_id in stale_minis:
        MINI_TO_TOKEN.pop(mini_id, None)
    if stale_minis:
        _save_mapping()
    return stale_minis


# ──────────────────────────────────────────────────────────────────────────────
# Scene/grid params (NEW)
# ──────────────────────────────────────────────────────────────────────────────

def set_scene_params(scene_id=None, scene_w=None, scene_h=None, grid_px=None, shift_x=0, shift_y=0, grid_type=None, background=None):
    """
    Update scene metadata based on Foundry 'sceneInfo' message.
    Only prints when values actually change to avoid log spam.
    """
    global SCENE_ID, SCENE_W, SCENE_H, GRID_PX, GRID_TYPE, SHIFT_X, SHIFT_Y, SCENE_BACKGROUND
    global _grid_cols, _grid_rows

    previous_scene_id = SCENE_ID
    new_scene_id = str(scene_id) if scene_id else SCENE_ID
    new_w        = int(scene_w)  if scene_w  else SCENE_W
    new_h        = int(scene_h)  if scene_h  else SCENE_H
    new_grid_px  = int(grid_px)  if grid_px  else GRID_PX
    new_shift_x  = int(shift_x or 0)
    new_shift_y  = int(shift_y or 0)

    changed = (
        new_scene_id != SCENE_ID or
        new_w        != SCENE_W  or
        new_h        != SCENE_H  or
        new_grid_px  != GRID_PX  or
        new_shift_x  != SHIFT_X  or
        new_shift_y  != SHIFT_Y
    )

    SCENE_ID = new_scene_id
    SCENE_W  = new_w
    SCENE_H  = new_h
    GRID_PX  = new_grid_px
    if grid_type is not None:
        GRID_TYPE = int(grid_type)
    SHIFT_X  = new_shift_x
    SHIFT_Y  = new_shift_y
    if GRID_PX and int(GRID_PX) > 0:
        _grid_cols = max(1, int(math.ceil(SCENE_W / float(GRID_PX))))
        _grid_rows = max(1, int(math.ceil(SCENE_H / float(GRID_PX))))
    if new_scene_id != previous_scene_id:
        SCENE_BACKGROUND = str(background) if background else None
    elif background:
        SCENE_BACKGROUND = str(background)

    if previous_scene_id and SCENE_ID != previous_scene_id:
        clear_view_transform()
        with _capture_lock:
            if _capture_session is not None:
                _capture_commands.append({"action": "stop", "reason": "sceneChanged"})

    if changed:
        print(
            "FoundryOutput | Scene params updated: "
            f"sceneId={SCENE_ID} size={SCENE_W}x{SCENE_H} "
            f"gridPx={GRID_PX} shift=({SHIFT_X},{SHIFT_Y})"
            + (f" gridType={grid_type}" if grid_type is not None else "")
        )

def get_scene_params() -> dict:
    return {
        "sceneId": SCENE_ID,
        "sceneW": SCENE_W,
        "sceneH": SCENE_H,
        "gridPx": GRID_PX,
        "gridType": GRID_TYPE,
        "shiftX": SHIFT_X,
        "shiftY": SHIFT_Y,
        "gridCols": _grid_cols,
        "gridRows": _grid_rows,
        "background": SCENE_BACKGROUND,
        "viewTransformRevision": get_view_transform_revision(),
    }


def reconcile_scene_bindings(payload: dict) -> None:
    """Named player tokens can be reassigned automatically across scenes."""
    global SCENE_TOKEN_NAMES
    if "tokens" not in payload:
        return
    SCENE_TOKEN_NAMES = {
        str(token["id"]): str(token.get("name") or token["id"])
        for token in payload["tokens"]
        if token.get("id")
    }
    valid_ids = {str(token["id"]) for token in payload["tokens"] if token.get("id")}
    bindings = payload.get("miniBindings") or {}
    before = dict(MINI_TO_TOKEN)
    for mini_id, token_id in list(MINI_TO_TOKEN.items()):
        if token_id not in valid_ids:
            MINI_TO_TOKEN.pop(mini_id, None)
    for mini_id, token_id in bindings.items():
        if mini_id not in MINI_TO_TOKEN and mini_id in {"red10", "blue", "yellow", "green", "white"} and str(token_id) in valid_ids:
            MINI_TO_TOKEN[mini_id] = str(token_id)
    if MINI_TO_TOKEN != before:
        _save_mapping()


def get_mini_assignments() -> tuple[dict, dict]:
    """Return copies of live mini mappings and scene token labels for UI use."""
    return dict(MINI_TO_TOKEN), dict(SCENE_TOKEN_NAMES)


def get_selected_mini() -> str | None:
    return _selected_mini_id


def clear_view_transform() -> None:
    global _view_transform, _view_transform_revision
    with _view_transform_lock:
        if _view_transform is not None:
            _view_transform = None
            _view_transform_revision += 1


def set_view_transform(data: dict) -> bool:
    """Validate and store a Foundry canvas-to-browser transform."""
    global _view_transform, _view_transform_revision

    try:
        registration = data["registration"]
        matrix = data["canvasTransform"]
        client_matrix = data.get("clientToCanvasTransform")
        candidate = {
            "sceneId": str(data.get("sceneId") or SCENE_ID),
            "viewportWidth": float(data["viewportWidth"]),
            "viewportHeight": float(data["viewportHeight"]),
            "left": float(registration["left"]),
            "top": float(registration["top"]),
            "right": float(registration["right"]),
            "bottom": float(registration["bottom"]),
            "a": float(matrix["a"]),
            "b": float(matrix["b"]),
            "c": float(matrix["c"]),
            "d": float(matrix["d"]),
            "tx": float(matrix["tx"]),
            "ty": float(matrix["ty"]),
            "gridOriginX": float(data["gridOriginX"]),
            "gridOriginY": float(data["gridOriginY"]),
            "gridSize": float(data["gridSize"]),
        }
        if client_matrix:
            candidate.update({
                "clientA": float(client_matrix["a"]),
                "clientB": float(client_matrix["b"]),
                "clientC": float(client_matrix["c"]),
                "clientD": float(client_matrix["d"]),
                "clientTx": float(client_matrix["tx"]),
                "clientTy": float(client_matrix["ty"]),
            })
        else:
            matrix_det = candidate["a"] * candidate["d"] - candidate["b"] * candidate["c"]
            candidate.update({
                "clientA": candidate["d"] / matrix_det,
                "clientB": -candidate["b"] / matrix_det,
                "clientC": -candidate["c"] / matrix_det,
                "clientD": candidate["a"] / matrix_det,
                "clientTx": (
                    candidate["c"] * candidate["ty"]
                    - candidate["d"] * candidate["tx"]
                ) / matrix_det,
                "clientTy": (
                    candidate["b"] * candidate["tx"]
                    - candidate["a"] * candidate["ty"]
                ) / matrix_det,
            })
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        print("FoundryOutput | Ignoring invalid viewTransform payload.")
        return False

    numeric_values = [value for key, value in candidate.items() if key != "sceneId"]
    determinant = candidate["a"] * candidate["d"] - candidate["b"] * candidate["c"]
    valid = (
        all(math.isfinite(value) for value in numeric_values)
        and candidate["viewportWidth"] > 0
        and candidate["viewportHeight"] > 0
        and candidate["right"] > candidate["left"]
        and candidate["bottom"] > candidate["top"]
        and candidate["gridSize"] > 0
        and abs(determinant) > 1e-9
    )
    if not valid:
        print("FoundryOutput | Ignoring unusable viewTransform payload.")
        return False

    with _view_transform_lock:
        changed = candidate != _view_transform
        if changed:
            _view_transform = candidate
            _view_transform_revision += 1
            revision = _view_transform_revision
        else:
            revision = _view_transform_revision

    if changed:
        print(
            "FoundryOutput | View transform updated: "
            f"revision={revision} scale=({candidate['a']:.4f},{candidate['d']:.4f}) "
            f"offset=({candidate['tx']:.1f},{candidate['ty']:.1f})"
        )
    return True


def get_view_transform_revision() -> int:
    with _view_transform_lock:
        return _view_transform_revision


def mark_scene_visual_changed(reason: str = "unknown") -> int:
    global _scene_visual_revision, _scene_visual_reason
    with _view_transform_lock:
        _scene_visual_revision += 1
        _scene_visual_reason = str(reason)
        revision = _scene_visual_revision
    print(
        f"FoundryOutput | Scene visual changed: revision={revision} reason={reason}"
    )
    return revision


def get_scene_visual_revision() -> int:
    with _view_transform_lock:
        return _scene_visual_revision


def get_scene_info_payload() -> dict:
    return {
        "type": "sceneInfo",
        "sceneId": SCENE_ID,
        "width": SCENE_W,
        "height": SCENE_H,
        "gridSize": GRID_PX,
        "gridType": GRID_TYPE,
        "shiftX": SHIFT_X,
        "shiftY": SHIFT_Y,
        "background": SCENE_BACKGROUND,
    }


def get_view_transform_payload() -> dict | None:
    view = _view_transform_snapshot()
    if view is None:
        return None
    return {
        "type": "viewTransform",
        "sceneId": view["sceneId"],
        "viewportMarkerMode": True,
        "viewportWidth": view["viewportWidth"],
        "viewportHeight": view["viewportHeight"],
        "registration": {
            "left": view["left"],
            "top": view["top"],
            "right": view["right"],
            "bottom": view["bottom"],
        },
        "canvasTransform": {
            "a": view["a"],
            "b": view["b"],
            "c": view["c"],
            "d": view["d"],
            "tx": view["tx"],
            "ty": view["ty"],
        },
        "clientToCanvasTransform": {
            "a": view["clientA"],
            "b": view["clientB"],
            "c": view["clientC"],
            "d": view["clientD"],
            "tx": view["clientTx"],
            "ty": view["clientTy"],
        },
        "gridOriginX": view["gridOriginX"],
        "gridOriginY": view["gridOriginY"],
        "gridSize": view["gridSize"],
    }


def get_recording_state_snapshot() -> dict:
    with _view_transform_lock:
        visual_revision = _scene_visual_revision
        visual_reason = _scene_visual_reason
        test_sequence = copy.deepcopy(_test_sequence)
    return {
        "sceneInfo": get_scene_info_payload(),
        "viewTransform": get_view_transform_payload(),
        "sceneVisualRevision": visual_revision,
        "sceneVisualReason": visual_reason,
        "testSequence": test_sequence,
        "cameraLock": dict(_camera_lock_state),
    }


def set_test_sequence(payload: dict) -> None:
    global _test_sequence
    targets = []
    for item in payload.get("targets") or []:
        try:
            targets.append(
                {
                    "step": int(item["step"]),
                    "row": int(item["row"]),
                    "column": int(item["column"]),
                    "cell": str(item["cell"]),
                    **{key: item[key] for key in ("route", "ringColor", "label", "miniId") if key in item},
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    with _view_transform_lock:
        _test_sequence = {
            "sceneId": payload.get("sceneId"),
            "targets": targets,
        } if targets else None
    labels = [target["cell"] for target in targets]
    print(f"FoundryOutput | Test sequence updated: {labels}", flush=True)


def timeline_path_for_video(video_path: str | Path) -> Path:
    return Path(video_path).with_suffix(".tracking.json")


def _write_timeline_file() -> None:
    with _timeline_lock:
        if _timeline_recording is None or _timeline_path is None:
            return
        data = copy.deepcopy(_timeline_recording)
        path = _timeline_path
        # Camera and WebSocket threads can both persist annotations.
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temp_path.replace(path)


def start_timeline_recording(
    video_path: str | Path,
    *,
    fps: float,
    frame_width: int,
    frame_height: int,
    marker_mode: str,
    warp_width: int,
    warp_height: int,
    grid_cols: int,
    grid_rows: int,
    frames_undistorted: bool = True,
) -> Path:
    global _timeline_recording, _timeline_path, _timeline_last_signature
    stop_timeline_recording()
    path = timeline_path_for_video(video_path)
    snapshot = get_recording_state_snapshot()
    signature = json.dumps(snapshot, sort_keys=True)
    with _timeline_lock:
        _timeline_path = path
        _timeline_last_signature = signature
        _timeline_recording = {
            "schemaVersion": 1,
            "video": Path(video_path).name,
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "markerMode": str(marker_mode),
            "fps": float(fps),
            "frameWidth": int(frame_width),
            "frameHeight": int(frame_height),
            "framesUndistorted": bool(frames_undistorted),
            "warpWidth": int(warp_width),
            "warpHeight": int(warp_height),
            "gridCols": int(grid_cols),
            "gridRows": int(grid_rows),
            "framesRecorded": 0,
            "trackingEvents": [],
            "groundTruth": [],
            "referenceFrames": [],
            "events": [{"frame": 0, **snapshot}],
        }
    _write_timeline_file()
    queue_control({"type": "referenceCapture", "enabled": marker_mode == "viewport"})
    print(f"FoundryOutput | Recording Foundry timeline -> {path}", flush=True)
    return path


def record_timeline_frame() -> None:
    global _timeline_last_signature
    snapshot = get_recording_state_snapshot()
    signature = json.dumps(snapshot, sort_keys=True)
    should_write = False
    with _timeline_lock:
        if _timeline_recording is None:
            return
        frame = int(_timeline_recording["framesRecorded"])
        if signature != _timeline_last_signature:
            _timeline_recording["events"].append({"frame": frame, **snapshot})
            _timeline_last_signature = signature
            should_write = True
        _timeline_recording["framesRecorded"] = frame + 1
    if should_write:
        _write_timeline_file()


def record_rendered_reference(payload: dict) -> bool:
    """Persist a low-rate, clean Foundry canvas image beside the video."""
    encoded = payload.get("image")
    if not isinstance(encoded, str) or not encoded.startswith("data:image/jpeg;base64,"):
        return False
    if len(encoded) > 500_000:
        return False
    try:
        image = base64.b64decode(encoded.split(",", 1)[1], validate=True)
    except (ValueError, binascii.Error):
        return False
    if len(image) > 350_000 or not image.startswith(b"\xff\xd8") or not image.endswith(b"\xff\xd9"):
        return False

    with _timeline_lock:
        if _timeline_recording is None or _timeline_path is None:
            return False
        if str(payload.get("sceneId")) != str(SCENE_ID):
            return False
        frame = int(_timeline_recording["framesRecorded"])
        references = _timeline_recording["referenceFrames"]
        min_interval = max(1, int(float(_timeline_recording["fps"])))
        if references and frame - int(references[-1]["frame"]) < min_interval:
            return False
        folder = _timeline_path.parent / (_timeline_path.stem.removesuffix(".tracking") + "_references")
        folder.mkdir(exist_ok=True)
        path = folder / f"frame_{frame:06d}.jpg"
        path.write_bytes(image)
        references.append({
            "frame": frame,
            "sceneId": SCENE_ID,
            "viewRevision": get_view_transform_revision(),
            "viewTransform": get_view_transform_payload(),
            "path": str(path.relative_to(_timeline_path.parent)),
        })
    _write_timeline_file()
    return True


def record_tracking_event(
    mini_id: str,
    cell_label: str,
    *,
    source: str = "detection",
) -> None:
    """Record a live tracker emission separately from Foundry acknowledgements."""
    with _timeline_lock:
        if _timeline_recording is None:
            return
        frame = max(0, int(_timeline_recording["framesRecorded"]) - 1)
        _timeline_recording["trackingEvents"].append(
            {
                "frame": frame,
                "mini": str(mini_id),
                "cell": str(cell_label),
                "source": str(source),
            }
        )
    _write_timeline_file()


def stop_timeline_recording() -> Path | None:
    global _timeline_recording, _timeline_path, _timeline_last_signature
    with _timeline_lock:
        if _timeline_recording is None or _timeline_path is None:
            return None
        path = _timeline_path
    _write_timeline_file()
    with _timeline_lock:
        _timeline_recording = None
        _timeline_path = None
        _timeline_last_signature = None
    queue_control({"type": "referenceCapture", "enabled": False})
    print(f"FoundryOutput | Foundry timeline saved -> {path}", flush=True)
    return path


def queue_control(payload: dict) -> None:
    if _loop is not None and _ctrl_queue is not None:
        _loop.call_soon_threadsafe(_ctrl_queue.put_nowait, payload)


def tracking_output_paused() -> bool:
    with _capture_lock:
        return _tracking_output_paused


def guided_capture_active() -> bool:
    with _capture_lock:
        return _capture_session is not None


def update_camera_lock(locked: bool, missing_ids: list) -> None:
    global _camera_lock_state
    _camera_lock_state = {"locked": bool(locked), "missingIds": list(missing_ids)}


def send_capture_status(state: str, **values) -> None:
    global _capture_status
    _capture_status = {"type": "captureStatus", "state": state, **values}
    queue_control(_capture_status)


def pop_capture_command() -> dict | None:
    with _capture_lock:
        return _capture_commands.popleft() if _capture_commands else None


def handle_capture_control(payload: dict) -> None:
    global _capture_session, _tracking_output_paused
    action = payload.get("action")
    try:
        with _capture_lock:
            if action == "start":
                if _capture_session is not None:
                    raise ValueError("A capture is already active")
                _capture_session = GuidedCapture(payload, SCENE_ID)
                _tracking_output_paused = True
                _capture_commands.append({"action": "start"})
                return
            if action == "resume":
                if _capture_session is not None:
                    raise ValueError("Finish the capture before resuming tracking")
                _tracking_output_paused = False
                send_capture_status("idle")
                return
            if _capture_session is None or payload.get("sessionId") != _capture_session.session_id:
                raise ValueError("No matching active capture")
            if action == "stop":
                _capture_commands.append({"action": "stop", "reason": str(payload.get("reason") or "userStopped")})
                return
            if action not in {"prompt", "confirm"}:
                raise ValueError("Unknown capture action")
            if not _capture_ready:
                raise ValueError("Camera recording is not ready")
            if _capture_session.scene_id != SCENE_ID:
                raise ValueError("Scene changed during capture")
            if action == "confirm" and not _camera_lock_state["locked"]:
                raise ValueError("Marker lock was lost; restore lock before confirming")
            with _timeline_lock:
                if _timeline_recording is None:
                    raise ValueError("Video recording has stopped")
                frame = max(0, int(_timeline_recording["framesRecorded"]) - 1)
                if action == "prompt":
                    _capture_session.prompt(payload["index"], frame)
                    return
                event = _capture_session.confirm(payload["index"], frame)
                _timeline_recording["groundTruth"].append(event)
        _write_timeline_file()
        queue_control({"type": "captureStatus", "state": "confirmed", "sessionId": payload["sessionId"], "index": event["index"], "frame": frame})
    except (ValueError, KeyError, TypeError) as exc:
        queue_control({"type": "captureStatus", "state": "error", "message": str(exc)})


def ready_guided_capture(video_path: str, profiles: dict) -> None:
    global _capture_ready
    with _capture_lock:
        metadata = _capture_session.metadata()
        _capture_ready = True
    with _timeline_lock:
        _timeline_recording["capture"] = metadata
        _timeline_recording["profilesSnapshot"] = copy.deepcopy(profiles)
    _write_timeline_file()
    Path(video_path).with_suffix(".profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    send_capture_status("recording", sessionId=metadata["sessionId"], video=Path(video_path).name)


def finish_guided_capture(reason: str) -> None:
    global _capture_session, _capture_ready
    with _capture_lock:
        session = _capture_session
        was_ready = _capture_ready
        _capture_session = None
        _capture_ready = False
        _capture_commands.clear()
    if session is None:
        return
    case_path = None
    with _timeline_lock:
        if was_ready and _timeline_recording is not None:
            _timeline_recording["captureComplete"] = session.next_index == len(session.targets)
            _timeline_recording["captureEndReason"] = reason
            data = copy.deepcopy(_timeline_recording)
            case_path = _timeline_path.with_suffix(".json").with_name(Path(data["video"]).stem + ".case.json")
            case_path.write_text(json.dumps(regression_case(data, data["video"]), indent=2), encoding="utf-8")
    _write_timeline_file()
    send_capture_status("saved", sessionId=session.session_id, reason=reason, case=case_path.name if case_path else None)


def has_view_transform() -> bool:
    with _view_transform_lock:
        return _view_transform is not None


def _view_transform_snapshot() -> dict | None:
    with _view_transform_lock:
        return dict(_view_transform) if _view_transform is not None else None


def get_view_transform_info() -> dict:
    view = _view_transform_snapshot()
    if view is None:
        return {"available": False, "revision": get_view_transform_revision()}
    return {
        "available": True,
        "revision": get_view_transform_revision(),
        "sceneId": view["sceneId"],
        "registration": (view["left"], view["top"], view["right"], view["bottom"]),
        "clientToCanvas": (
            view["clientA"], view["clientB"], view["clientC"],
            view["clientD"], view["clientTx"], view["clientTy"],
        ),
        "gridOrigin": (view["gridOriginX"], view["gridOriginY"]),
        "gridSize": view["gridSize"],
    }


def _canvas_to_warp_point(
    view: dict, canvas_x: float, canvas_y: float, warp_w: int, warp_h: int
) -> tuple[float, float]:
    determinant = (
        view["clientA"] * view["clientD"]
        - view["clientB"] * view["clientC"]
    )
    dx = canvas_x - view["clientTx"]
    dy = canvas_y - view["clientTy"]
    client_x = (view["clientD"] * dx - view["clientC"] * dy) / determinant
    client_y = (-view["clientB"] * dx + view["clientA"] * dy) / determinant
    warp_x = (
        (client_x - view["left"])
        / (view["right"] - view["left"])
        * float(warp_w - 1)
    )
    warp_y = (
        (client_y - view["top"])
        / (view["bottom"] - view["top"])
        * float(warp_h - 1)
    )
    return warp_x, warp_y


def warp_grid_segments(
    warp_w: int, warp_h: int
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return actual Foundry grid lines projected into the camera warp."""
    view = _view_transform_snapshot()
    if view is None or warp_w <= 1 or warp_h <= 1:
        return []

    grid = view["gridSize"]
    cols = max(1, int(math.ceil(SCENE_W / grid)))
    rows = max(1, int(math.ceil(SCENE_H / grid)))
    x0 = view["gridOriginX"]
    y0 = view["gridOriginY"]
    x1 = x0 + cols * grid
    y1 = y0 + rows * grid
    segments = []

    def add_segment(start, end):
        points = (
            _canvas_to_warp_point(view, *start, warp_w, warp_h),
            _canvas_to_warp_point(view, *end, warp_w, warp_h),
        )
        xs = (points[0][0], points[1][0])
        ys = (points[0][1], points[1][1])
        if max(xs) < 0 or min(xs) >= warp_w or max(ys) < 0 or min(ys) >= warp_h:
            return
        segments.append(points)

    for col in range(cols + 1):
        x = x0 + col * grid
        add_segment((x, y0), (x, y1))
    for row in range(rows + 1):
        y = y0 + row * grid
        add_segment((x0, y), (x1, y))
    return segments


def warp_to_grid_cell(cx: float, cy: float, warp_w: int, warp_h: int) -> tuple[int, int] | None:
    """Map a point in the camera's marker warp to a Foundry (column, row)."""
    view = _view_transform_snapshot()
    if view is None or warp_w <= 1 or warp_h <= 1:
        return None
    if view["sceneId"] and SCENE_ID and view["sceneId"] != SCENE_ID:
        return None

    client_x = view["left"] + (float(cx) / float(warp_w - 1)) * (view["right"] - view["left"])
    client_y = view["top"] + (float(cy) / float(warp_h - 1)) * (view["bottom"] - view["top"])

    canvas_x = (
        view["clientA"] * client_x
        + view["clientC"] * client_y
        + view["clientTx"]
    )
    canvas_y = (
        view["clientB"] * client_x
        + view["clientD"] * client_y
        + view["clientTy"]
    )

    col = math.floor((canvas_x - view["gridOriginX"]) / view["gridSize"])
    row = math.floor((canvas_y - view["gridOriginY"]) / view["gridSize"])
    max_cols = max(1, int(math.ceil(SCENE_W / view["gridSize"])))
    max_rows = max(1, int(math.ceil(SCENE_H / view["gridSize"])))
    if col < 0 or row < 0 or col >= max_cols or row >= max_rows:
        return None
    return int(col), int(row)


def warp_grid_dimensions(warp_w: int, warp_h: int) -> tuple[float, float] | None:
    """Return the displayed Foundry grid size along the camera warp's axes."""
    view = _view_transform_snapshot()
    if view is None or warp_w <= 1 or warp_h <= 1:
        return None

    sx = float(warp_w - 1) / (view["right"] - view["left"])
    sy = float(warp_h - 1) / (view["bottom"] - view["top"])
    grid = view["gridSize"]
    determinant = (
        view["clientA"] * view["clientD"]
        - view["clientB"] * view["clientC"]
    )
    canvas_x_client = (view["clientD"] / determinant, -view["clientB"] / determinant)
    canvas_y_client = (-view["clientC"] / determinant, view["clientA"] / determinant)
    x_axis = math.hypot(canvas_x_client[0] * grid * sx, canvas_x_client[1] * grid * sy)
    y_axis = math.hypot(canvas_y_client[0] * grid * sx, canvas_y_client[1] * grid * sy)
    if not math.isfinite(x_axis) or not math.isfinite(y_axis) or min(x_axis, y_axis) <= 0:
        return None
    return x_axis, y_axis


def warp_grid_size(warp_w: int, warp_h: int) -> float | None:
    dimensions = warp_grid_dimensions(warp_w, warp_h)
    return min(dimensions) if dimensions is not None else None

def request_scene_info() -> None:
    """
    Ask Foundry module to send active scene info.
    Requires module.js handling for {type:"getSceneInfo"}.
    """
    global _loop, _ctrl_queue
    if _loop is None or _ctrl_queue is None:
        print("FoundryOutput | Event loop not ready yet; cannot request scene info.")
        return

    def _enqueue():
        try:
            _ctrl_queue.put_nowait({"type": "getSceneInfo"})
        except Exception as e:
            print("FoundryOutput | Failed to enqueue getSceneInfo:", e)

    _loop.call_soon_threadsafe(_enqueue)


# ──────────────────────────────────────────────────────────────────────────────
# Public API for tracking/main
# ──────────────────────────────────────────────────────────────────────────────

def set_grid_params(warp_w: int, warp_h: int, grid_cols: int, grid_rows: int) -> None:
    """
    tracking.py calls this each run.
    """
    global _grid_cols, _grid_rows
    _grid_cols = int(grid_cols)
    _grid_rows = int(grid_rows)
    print(
        f"FoundryOutput | Grid params set: "
        f"{_grid_cols}x{_grid_rows} on scene {SCENE_W}x{SCENE_H}px"
    )

def has_mapping(mini_id: str) -> bool:
    mini_id = str(mini_id).strip()
    return mini_id in MINI_TO_TOKEN and bool(MINI_TO_TOKEN[mini_id])

def request_assignment(mini_id: str) -> None:
    """
    Ask Foundry to prompt the GM to assign this mini_id to a token.
    """
    global _loop, _assign_queue
    if tracking_output_paused():
        return
    if _loop is None or _assign_queue is None:
        print("FoundryOutput | Event loop not ready yet; cannot request assignment.")
        return

    mini_id = str(mini_id).strip()

    def _enqueue():
        try:
            _assign_queue.put_nowait(mini_id)
        except Exception as e:
            print("FoundryOutput | Failed to enqueue assignment request:", e)

    _loop.call_soon_threadsafe(_enqueue)

def queue_cell_move(mini_id: str, cell_label: str) -> None:
    """
    Called when a mini moves into a new cell.
    We queue (mini_id, cell_label) so the send loop can pick the correct token.
    """
    global _loop, _move_queue
    if tracking_output_paused():
        return
    if _loop is None or _move_queue is None:
        print("FoundryOutput | Event loop not ready yet; cannot queue move.")
        return

    mini_id = str(mini_id).strip()
    cell_label = cell_label.strip()
    scene_id = SCENE_ID

    def _enqueue():
        try:
            _move_queue.put_nowait((mini_id, cell_label, scene_id))
        except Exception as e:
            print("FoundryOutput | Failed to enqueue move:", e)

    _loop.call_soon_threadsafe(_enqueue)

def move_token_to_grid(mini_id: str, cell_label: str) -> None:
    """
    API used by main.py:on_mini_moved.
    """
    queue_cell_move(mini_id, cell_label)


# ──────────────────────────────────────────────────────────────────────────────
# Mini → token resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_token_id_for_mini(mini_id: str) -> str:
    """
    Resolve which Foundry tokenId should be moved for a given mini_id.
    """
    if mini_id in MINI_TO_TOKEN:
        return MINI_TO_TOKEN[mini_id]

    if "_" in mini_id:
        base = mini_id.split("_", 1)[0]
        if base in MINI_TO_TOKEN:
            return MINI_TO_TOKEN[base]

    if len(mini_id) >= 6:
        tail = mini_id[-6:]
        if tail in MINI_TO_TOKEN:
            return MINI_TO_TOKEN[tail]

    return DEFAULT_TOKEN_ID


# ──────────────────────────────────────────────────────────────────────────────
# Grid → pixel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _grid_to_pixels(cell_str: str) -> tuple[int, int]:
    """
    Convert a grid cell like 'B14' or 'r7c11' into (x, y) pixel coordinates
    in Foundry scene space, using the TOP-LEFT corner of the cell.

    If GRID_PX is known from Foundry, prefer:
      x = SHIFT_X + col * GRID_PX
      y = SHIFT_Y + row * GRID_PX

    Otherwise fallback to SCENE_W/_grid_cols and SCENE_H/_grid_rows.
    """
    s = cell_str.strip()

    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    if m:
        col_letters = m.group(1).upper()
        row_num = int(m.group(2))

        col = 0
        for ch in col_letters:
            col = col * 26 + (ord(ch) - ord('A') + 1)
        col -= 1
        row = row_num - 1
    else:
        m2 = re.match(r"^r(\d+)c(\d+)$", s.lower())
        if not m2:
            raise ValueError(f"Unrecognized cell format: {cell_str}")
        row = int(m2.group(1))
        col = int(m2.group(2))

    #adj_row = max(row - 1, 0)
    adj_row = max(row, 0)


    view = _view_transform_snapshot()
    if view is not None and view["sceneId"] == SCENE_ID:
        px = view["gridOriginX"] + col * view["gridSize"]
        py = view["gridOriginY"] + adj_row * view["gridSize"]
    elif GRID_PX is not None and int(GRID_PX) > 0:
        px = SHIFT_X + col * int(GRID_PX)
        py = SHIFT_Y + adj_row * int(GRID_PX)
    else:
        cell_w = SCENE_W / float(_grid_cols)
        cell_h = SCENE_H / float(_grid_rows)
        px = col * cell_w
        py = adj_row * cell_h

    return int(round(px)), int(round(py))


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket loops
# ──────────────────────────────────────────────────────────────────────────────

async def send_loop(websocket):
    """
    Waits for (mini_id, cell_label) queued by queue_cell_move(), converts to pixels,
    resolves the correct token, and sends move commands to Foundry.
    """
    global _move_queue
    if _move_queue is None:
        _move_queue = asyncio.Queue()

    print("FoundryOutput | Ready to send moves to Foundry (queue_cell_move from tracking/main).")

    while True:
        mini_id, cell_label, queued_scene = await _move_queue.get()
        if tracking_output_paused() or queued_scene != SCENE_ID:
            continue

        if not SCENE_ID:
            print(f"FoundryOutput | sceneInfo not yet received from Foundry; "
                  f"dropping move for mini {mini_id}. "
                  f"This should resolve once Foundry connects and sends sceneInfo.")
            continue

        token_id = _resolve_token_id_for_mini(mini_id)
        if token_id == DEFAULT_TOKEN_ID:
            print(f"FoundryOutput | mini {mini_id} has no mapping yet; request assignment and skip move.")
            request_assignment(mini_id)
            continue

        try:
            x, y = _grid_to_pixels(cell_label)
        except Exception as e:
            print(f"FoundryOutput | Failed to convert cell '{cell_label}': {e}")
            continue

        payload = {
            "type": "moveToken",
            "sceneId": SCENE_ID,
            "miniId": mini_id,
            "cell": cell_label,
            "tokenId": token_id,
            "x": x,
            "y": y,
        }

        print(
            f"FoundryOutput | mini={mini_id} cell={cell_label} "
            f"→ token={token_id} → ({x}, {y}) | Sending move command."
        )

        try:
            await websocket.send(json.dumps(payload))
        except websockets.ConnectionClosed:
            print("FoundryOutput | Connection closed while sending.")
            break

async def assign_request_loop(websocket):
    global _assign_queue
    if _assign_queue is None:
        _assign_queue = asyncio.Queue()

    while True:
        mini_id = await _assign_queue.get()
        if tracking_output_paused():
            continue
        msg = {"type": "assignMini", "miniId": str(mini_id)}
        print(f"FoundryOutput | Requesting assignment for mini {mini_id}")
        try:
            await websocket.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            print("FoundryOutput | Connection closed while requesting assignment.")
            break

async def ctrl_loop(websocket):
    """
    NEW: sends control messages like {type:"getSceneInfo"}.
    """
    global _ctrl_queue
    if _ctrl_queue is None:
        _ctrl_queue = asyncio.Queue()

    while True:
        msg = await _ctrl_queue.get()
        try:
            await websocket.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            print("FoundryOutput | Connection closed while sending control message.")
            break

async def recv_loop(websocket):
    """
    Receives messages from Foundry (assignment results, scene info).
    """
    global MINI_TO_TOKEN, _selected_mini_id
    while True:
        try:
            raw = await websocket.recv()
        except websockets.ConnectionClosed:
            print("FoundryOutput | Connection closed (recv).")
            break

        try:
            data = json.loads(raw)
        except Exception:
            continue

        msg_type = data.get("type")

        if msg_type in ("hello", "ping"):
            if msg_type == "hello":
                queue_control(_capture_status)
                with _timeline_lock:
                    recording = _timeline_recording
                    enabled = bool(recording and recording["markerMode"] == "viewport")
                queue_control({"type": "referenceCapture", "enabled": enabled})
            continue

        if msg_type == "assignMiniResult":
            mini_id = str(data.get("miniId", "")).strip()
            token_id = data.get("tokenId")
            cancelled = bool(data.get("cancelled", False))

            if cancelled:
                print(f"FoundryOutput | Assignment cancelled for mini {mini_id}")
                continue

            if mini_id and token_id:
                MINI_TO_TOKEN[mini_id] = str(token_id)
                print(f"FoundryOutput | Assigned mini {mini_id} -> token {token_id}")
                _save_mapping()
            else:
                print(f"FoundryOutput | Invalid assignMiniResult: {data}")

        elif msg_type == "sceneInfo":
            # Expected from module.js:
            # {type:"sceneInfo", sceneId, width, height, gridSize, shiftX, shiftY, gridType}
            scene_id = data.get("sceneId") or SCENE_ID
            w = data.get("width")
            h = data.get("height")
            grid_size = data.get("gridSize") or data.get("gridPx") or data.get("grid")
            sx = data.get("shiftX", 0)
            sy = data.get("shiftY", 0)
            gt = data.get("gridType", None)

            set_scene_params(scene_id, w, h, grid_size, sx, sy, gt,
                             background=data.get("background"))
            reconcile_scene_bindings(data)

        elif msg_type == "viewTransform":
            set_view_transform(data)

        elif msg_type == "renderedReference":
            if data.get("viewTransform"):
                set_view_transform(data["viewTransform"])
            try:
                record_rendered_reference(data)
            except OSError as exc:
                print(f"FoundryOutput | Reference snapshot save error: {exc}", flush=True)

        elif msg_type == "sceneVisualChanged":
            mark_scene_visual_changed(str(data.get("reason") or "foundry"))

        elif msg_type == "movementSelection":
            mini_id = str(data.get("miniId") or "").strip()
            _selected_mini_id = mini_id if data.get("selected") and mini_id else None
            print(
                f"FoundryOutput | Movement selection: "
                f"{_selected_mini_id or 'none'}",
                flush=True,
            )

        elif msg_type == "tokenMissing":
            token_id = str(data.get("tokenId") or "").strip()
            stale_minis = _remove_stale_token_mapping(token_id)
            for mini_id in stale_minis:
                print(
                    f"FoundryOutput | Token {token_id} no longer exists; "
                    f"requesting a new assignment for mini {mini_id}."
                )
                request_assignment(mini_id)

        elif msg_type == "testSequence":
            set_test_sequence(data)

        elif msg_type == "captureControl":
            handle_capture_control(data)

        else:
            # ignore unknown message types
            pass


async def handler(websocket):
    print("FoundryOutput | Foundry connected via WebSocket")

    tasks = [
        asyncio.create_task(send_loop(websocket)),
        asyncio.create_task(assign_request_loop(websocket)),
        asyncio.create_task(ctrl_loop(websocket)),   # NEW
        asyncio.create_task(recv_loop(websocket)),
    ]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    with _capture_lock:
        if _capture_session is not None:
            _capture_commands.append({"action": "stop", "reason": "disconnected"})

    print("FoundryOutput | Handler finished.")


async def main():
    global _loop, _move_queue, _assign_queue, _ctrl_queue
    _loop = asyncio.get_running_loop()
    _move_queue = asyncio.Queue()
    _assign_queue = asyncio.Queue()
    _ctrl_queue = asyncio.Queue()

    _load_mapping()

    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("FoundryOutput | WebSocket server listening on ws://127.0.0.1:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
