import asyncio
import threading
from datetime import datetime

import setup as s
import calibration as c
import foundryoutput as fo
from control_panel import rc_to_a1

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _rc_str_to_a1(cell: str) -> str:
    """Convert internal 'r0c0' string to A1 display notation."""
    try:
        parts = cell[1:].split('c')
        return rc_to_a1(int(parts[0]), int(parts[1]))
    except Exception:
        return cell


def start_foundry_server_in_background():
    """
    Start the FoundryOutput WebSocket server (fo.main) in a background thread
    so tracking + CV can run in the main thread.
    """

    def runner():
        # This runs the async foundryoutput.main() in its own event loop
        asyncio.run(fo.main())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    print("MAIN | Started FoundryOutput server in background thread")


def on_mini_moved(mini_id, grid_coord):
    """
    Callback passed into tracking.begin_session.
    Called whenever a known mini's consensus grid cell changes.
    """
    print(f"[MAIN] Mini {mini_id} moved to {_rc_str_to_a1(grid_coord)}")
    # Forward the grid coordinate to the Foundry output module
    fo.move_token_to_grid(mini_id, grid_coord)


def _get_engine_from_config() -> str:
    cfg = s.load_last_selection() or {}
    eng = (cfg.get("engine") or "blob").strip().lower()
    return eng if eng in ("blob", "band") else "blob"


def main():
    # Start the Foundry WebSocket server in the background
    start_foundry_server_in_background()

    try:
        print("running s.initialize. Timestamp: " + timestamp)
        s.initialize()

        print("running c.calibrate. Timestamp: " + timestamp)
        c.calibrate()

        while True:
            engine = _get_engine_from_config()
            print(f"MAIN | Engine selected: {engine}")

            if engine == "band":
                import band_tracking as t
            else:
                import blob_tracking as t

            print("running begin_session. Timestamp: " + timestamp)
            result = t.begin_session(on_mini_moved)
            if isinstance(result, dict) and result.get("switch_to") in ("blob", "band"):
                # main loop will restart and pick up the newly saved engine
                continue
            break

    except SystemExit as e:
        # User cancelled setup or cancelled waiting for Foundry scene info
        print(f"MAIN | Exiting: {e}")
        return


if __name__ == "__main__":
    main()