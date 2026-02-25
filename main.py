import asyncio
import threading
from datetime import datetime

import setup as s
import calibration as c
import tracking as t
import foundryoutput as fo

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


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
    print(f"[MAIN] Mini {mini_id} moved to {grid_coord}")
    # Forward the grid coordinate to the Foundry output module
    fo.move_token_to_grid(mini_id, grid_coord)


def main():
    # Start the Foundry WebSocket server in the background
    start_foundry_server_in_background()

    try:
        print("running s.initialize. Timestamp: " + timestamp)
        s.initialize()

        print("running c.calibrate. Timestamp: " + timestamp)
        c.calibrate()

        print("running t.begin_session. Timestamp: " + timestamp)
        t.begin_session(on_mini_moved)

    except SystemExit as e:
        # User cancelled setup or cancelled waiting for Foundry scene info
        print(f"MAIN | Exiting: {e}")
        return


if __name__ == "__main__":
    main()
