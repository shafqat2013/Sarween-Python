import asyncio
import json
import re
import websockets

# ──────────────────────────────────────────────────────────────────────────────
# Foundry configuration
# ──────────────────────────────────────────────────────────────────────────────

SCENE_ID = "pKZNLeOehA3WJuYC"

# Default token (used if we can't resolve a specific mapping)
DEFAULT_TOKEN_ID = "1" #"MH95ZnoQkuIIIqoL"

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

# Foundry scene size (your map image size)
SCENE_W = 1656   # dnd1.jpg width
SCENE_H = 1152   # dnd1.jpg height

# Grid settings (override from tracking if needed)
_grid_cols = 23
_grid_rows = 16

# Async bits
_move_queue: asyncio.Queue | None = None
_loop: asyncio.AbstractEventLoop | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Public API for tracking/main
# ──────────────────────────────────────────────────────────────────────────────

def set_grid_params(warp_w: int, warp_h: int, grid_cols: int, grid_rows: int) -> None:
    """
    We ignore warp_w/warp_h for pixel math, because Foundry cares about SCENE_W/SCENE_H.
    We only care about the grid dimensions here.
    """
    global _grid_cols, _grid_rows
    _grid_cols = int(grid_cols)
    _grid_rows = int(grid_rows)
    print(
        f"FoundryOutput | Grid params set: "
        f"{_grid_cols}x{_grid_rows} on scene {SCENE_W}x{SCENE_H}px"
    )


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
    mini_id: the ID from tracking (e.g., DB key / capture ID).
    cell_label: 'r7c11', 'B14', etc.
    """
    queue_cell_move(mini_id, cell_label)


# ──────────────────────────────────────────────────────────────────────────────
# Mini → token resolution
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_token_id_for_mini(mini_id: str) -> str:
    """
    Resolve which Foundry tokenId should be moved for a given mini_id.

    Many mini_ids can map to the same tokenId via MINI_TO_TOKEN.

    Resolution order:
      1) Exact mini_id match
      2) Prefix before first underscore (e.g., 'batman' in 'batman_2025_...').
      3) Last 6 characters (to match the on-screen 'known:123456' short code).
      4) DEFAULT_TOKEN_ID fallback.
    """
    # 1) Exact match
    if mini_id in MINI_TO_TOKEN:
        return MINI_TO_TOKEN[mini_id]

    # 2) Prefix before first underscore (allows grouping multiple scans)
    if "_" in mini_id:
        base = mini_id.split("_", 1)[0]
        if base in MINI_TO_TOKEN:
            return MINI_TO_TOKEN[base]

    # 3) Last 6 characters (short code shown on-screen as known:xxxxxx)
    if len(mini_id) >= 6:
        tail = mini_id[-6:]
        if tail in MINI_TO_TOKEN:
            return MINI_TO_TOKEN[tail]

    # 4) Fallback
    return DEFAULT_TOKEN_ID



# ──────────────────────────────────────────────────────────────────────────────
# Grid → pixel helpers
# ──────────────────────────────────────────────────────────────────────────────

def _grid_to_pixels(cell_str: str) -> tuple[int, int]:
    """
    Convert a grid cell like 'B14' or 'r7c11' into (x, y) pixel coordinates
    in Foundry scene space, using the TOP-LEFT corner of the cell.

    For now we follow this logic:
      - A1  → (0, 0)
      - B1  → (72, 0)
      - A2  → (0, 72)
    i.e. origin is the top-left of A1 and each cell is 72x72 pixels.

    - Scene size: SCENE_W x SCENE_H (e.g. 1656 x 1152)
    - Grid: _grid_cols x _grid_rows (e.g. 23 x 16)
    """

    s = cell_str.strip()

    # 1) Parse "B14" / "AA3" / "Z10"
    m = re.match(r"^([A-Za-z]+)(\d+)$", s)
    if m:
        col_letters = m.group(1).upper()
        row_num = int(m.group(2))

        # Excel-style letters → numeric column index (A=0, B=1, ..., AA=26, etc.)
        col = 0
        for ch in col_letters:
            col = col * 26 + (ord(ch) - ord('A') + 1)
        col -= 1          # 1-based → 0-based
        row = row_num - 1 # 1-based → 0-based

    else:
        # 2) Parse "r7c11" from tracking.py
        #    r = row index (0-based), c = column index (0-based)
        m2 = re.match(r"^r(\d+)c(\d+)$", s.lower())
        if not m2:
            raise ValueError(f"Unrecognized cell format: {cell_str}")
        row = int(m2.group(1))
        col = int(m2.group(2))

    # Scene-based cell size (this should be 72x72 with your settings)
    cell_w = SCENE_W / float(_grid_cols)
    cell_h = SCENE_H / float(_grid_rows)

    # Apply a -1 row offset: tracking's row N maps visually to Foundry's row N-1.
    # Clamp at 0 so we don't go negative on the first row.
    adj_row = max(row - 1, 0)

    # TOP-LEFT corner of the cell:
    # A1 (row=0, col=0) -> (0, 0)
    # B1 (row=0, col=1) -> (cell_w, 0) -> (72, 0) when cell_w=72
    px = col * cell_w
    py = adj_row * cell_h

    return int(round(px)), int(round(py))


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket send loop (no user input; driven by queue_cell_move)
# ──────────────────────────────────────────────────────────────────────────────

async def send_loop(websocket):
    """
    Waits for (mini_id, cell_label) pushed by queue_cell_move(), converts to pixels,
    resolves the correct token, and sends move commands to Foundry.
    """
    global _move_queue
    if _move_queue is None:
        _move_queue = asyncio.Queue()

    print("FoundryOutput | Ready to send moves to Foundry (queue_cell_move from tracking/main).")

    while True:
        mini_id, cell_label = await _move_queue.get()
        try:
            x, y = _grid_to_pixels(cell_label)
        except Exception as e:
            print(f"FoundryOutput | Failed to convert cell '{cell_label}': {e}")
            continue

        token_id = _resolve_token_id_for_mini(mini_id)

        payload = {
            "sceneId": SCENE_ID,
            "tokenId": token_id,
            "x": x,
            "y": y,
        }

        print(
            f"FoundryOutput | mini={mini_id} cell={cell_label} "
            f"→ token={token_id} → ({x}, {y}) | Sending move command: {payload}"
        )
        try:
            await websocket.send(json.dumps(payload))
        except websockets.ConnectionClosed:
            print("FoundryOutput | Connection closed while sending.")
            break


async def handler(websocket):
    print("FoundryOutput | Foundry connected via WebSocket")

    # Optional hello from the Foundry module
    try:
        hello_msg = await asyncio.wait_for(websocket.recv(), timeout=2)
        print("FoundryOutput | Received:", hello_msg)
    except asyncio.TimeoutError:
        print("FoundryOutput | No hello received (that's fine).")

    # Start the move-sending loop (driven by queue_cell_move)
    await send_loop(websocket)

    print("FoundryOutput | Handler finished.")


async def main():
    global _loop, _move_queue
    _loop = asyncio.get_running_loop()
    _move_queue = asyncio.Queue()

    async with websockets.serve(handler, "127.0.0.1", 8765):
        print("FoundryOutput | WebSocket server listening on ws://127.0.0.1:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
