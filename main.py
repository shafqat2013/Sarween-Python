import asyncio
import threading
import sys
import os
from datetime import datetime
from pathlib import Path

# ── Redirect stdout/stderr to log file when running as a frozen .app ──────────
if getattr(sys, "frozen", False):
    _log_dir = os.path.expanduser("~/Library/Logs/Sarween")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, "sarween.log")
    _log_file = open(_log_path, "w", buffering=1)
    sys.stdout = _log_file
    sys.stderr = _log_file
    print(f"Sarween log started: {datetime.now()}", flush=True)

import setup as s
import calibration as c
import foundryoutput as fo
from control_panel import rc_to_a1

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def _rc_str_to_a1(cell: str) -> str:
    try:
        parts = cell[1:].split('c')
        return rc_to_a1(int(parts[0]), int(parts[1]))
    except Exception:
        return cell


def start_foundry_server_in_background():
    def runner():
        asyncio.run(fo.main())
    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    print("MAIN | Started FoundryOutput server in background thread")


def on_mini_moved(mini_id, grid_coord):
    print(f"[MAIN] Mini {mini_id} moved to {_rc_str_to_a1(grid_coord)}")
    fo.move_token_to_grid(mini_id, grid_coord)


def main():
    start_foundry_server_in_background()

    try:
        print("running s.initialize. Timestamp: " + timestamp)
        s.initialize()

        print("running c.calibrate. Timestamp: " + timestamp)
        c.calibrate()

        import v3_tracking as t
        from mini_calibration import run_mini_calibration

        _profiles_path = Path(__file__).with_name("combo_profiles.json")
        if not _profiles_path.exists():
            print("MAIN | No mini profiles found — running mini calibration")
            run_mini_calibration()

        print("running begin_session. Timestamp: " + timestamp)
        while True:
            result = t.begin_session(on_mini_moved)
            if result == "recalibrate":
                print("MAIN | Re-running mini calibration")
                run_mini_calibration()
            else:
                break

    except SystemExit as e:
        print(f"MAIN | Exiting: {e}")
        return


if __name__ == "__main__":
    main()
