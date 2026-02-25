import asyncio
import json
import re
from pathlib import Path

import websockets

# ──────────────────────────────────────────────────────────────────────────────
# Foundry configuration
# ──────────────────────────────────────────────────────────────────────────────

SCENE_ID = "pKZNLeOehA3WJuYC"

# Default token (used if we can't resolve a specific mapping)
DEFAULT_TOKEN_ID = "1"  # fallback

# Map from mini_id (or logical mini key) -> Foundry tokenId.
# Many different mini_ids can point to the same tokenId.
MINI_TO_TOKEN = {
    "936478": "MH95ZnoQkuIIIqoL",
    "863792": "MH95ZnoQkuIIIqoL",
    "626666": "MH95ZnoQkuIIIqoL",
    "353367": "MH95ZnoQkuIIIqoL",
    "511420": "MH95ZnoQkuIIIqoL",
    "111111": "MoAbjfUrvJEkD4IK",
}

# Persist mappings across runs
MAP_PATH = Path(__file__).with_name("mini_token_map.json")

# Foundry scene size (fallback defaults; can be overridden by sceneInfo)
SCENE_W = 1656   # dnd1.jpg width
SCENE_H = 1152   # dnd1.jpg height

# Foundry grid metadata (NEW; prefer these if present)
GRID_PX = None   # pixels per grid square
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


# ──────────────────────────────────────────────────────────────────────────────
# Scene/grid params (NEW)
# ──────────────────────────────────────────────────────────────────────────────

def set_scene_params(scene_id=None, scene_w=None, scene_h=None, grid_px=None, shift_x=0, shift_y=0, grid_type=None):
    """
    Update scene metadata based on Foundry 'sceneInfo' message.
    """
    global SCENE_ID, SCENE_W, SCENE_H, GRID_PX, SHIFT_X, SHIFT_Y

    if scene_id:
        SCENE_ID = str(scene_id)

    if scene_w:
        SCENE_W = int(scene_w)
    if scene_h:
        SCENE_H = int(scene_h)

    GRID_PX = int(grid_px) if grid_px else None
    SHIFT_X = int(shift_x or 0)
    SHIFT_Y = int(shift_y or 0)

    print(
        "FoundryOutput | Scene params updated: "
        f"sceneId={SCENE_ID} size={SCENE_W}x{SCENE_H} "
        f"gridPx={GRID_PX} shift=({SHIFT_X},{SHIFT_Y})"
        + (f" gridType={grid_type}" if grid_type is not None else "")
    )

def get_scene_params() -> dict:
    """
    Used by calibration.py (Foundry mode) to compute cols/rows from width/height/gridPx.
    """
    return {
        "sceneId": SCENE_ID,
        "sceneW": SCENE_W,
        "sceneH": SCENE_H,
        "gridPx": GRID_PX,
        "shiftX": SHIFT_X,
        "shiftY": SHIFT_Y,
        "gridCols": _grid_cols,
        "gridRows": _grid_rows,
    }

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
    if _loop is None or _move_queue is None:
        print("FoundryOutput | Event loop not ready yet; cannot queue move.")
        return

    mini_id = str(mini_id).strip()
    cell_label = cell_label.strip()

    def _enqueue():
        try:
            _move_queue.put_nowait((mini_id, cell_label))
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


    if GRID_PX is not None and int(GRID_PX) > 0:
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
        mini_id, cell_label = await _move_queue.get()

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
            "sceneId": SCENE_ID,
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
    global MINI_TO_TOKEN
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

            set_scene_params(scene_id, w, h, grid_size, sx, sy, gt)

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
