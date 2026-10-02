import sys
import os

if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--replay-worker":
    from pathlib import Path
    from tracking_regression import _worker
    raise SystemExit(_worker(Path(sys.argv[2]), Path(sys.argv[3])))
from datetime import datetime
from app_paths import initialize_user_data
from foundry_service import FoundryService

# ── Redirect stdout/stderr to log file when running as a frozen .app ──────────
if getattr(sys, "frozen", False):
    _log_dir = os.path.expanduser("~/Library/Logs/Sarween")
    os.makedirs(_log_dir, exist_ok=True)
    _log_path = os.path.join(_log_dir, "sarween.log")
    _log_file = open(_log_path, "a", buffering=1)
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
    service = FoundryService()
    service.start()
    return service


def on_mini_moved(mini_id, grid_coord, *, source="detection"):
    print(f"[MAIN] Mini {mini_id} moved to {_rc_str_to_a1(grid_coord)} ({source})")
    fo.move_token_to_grid(mini_id, grid_coord, source=source)


def run_live_session(service):
    service.access.require()
    s.initialize()
    service.access.require()
    if s.selected_mode == s.MODE_FOUNDRY:
        service.start()
        # Binding is quick, but errors must surface before a Foundry wait window.
        import time
        deadline = time.monotonic() + 3
        while service.state == "Starting" and time.monotonic() < deadline:
            time.sleep(0.02)
        if service.state != "Listening":
            raise RuntimeError(service.error or "Foundry relay could not start")
    c.calibrate(access=service.access)
    service.access.require()

    import v3_tracking as t
    from mini_calibration import run_mini_calibration

    try:
        while t.begin_session(on_mini_moved, access=service.access) == "recalibrate":
            service.access.require()
            run_mini_calibration()
    finally:
        t.close_pending_recording()


def main(*, smoke_report=None):
    initialize_user_data()
    from app_window import AppWindow
    root = s._get_persistent_root()
    service = FoundryService()
    app = AppWindow(root, service=service, run_session=run_live_session)
    smoke_result = {}
    if smoke_report is not None:
        # Exercise the packaged UI and lazy imports, then leave no app running.
        app.start_button.state(["disabled"])
        app.connect_button.state(["disabled"])

        def inspect_replay(deadline):
            try:
                import time
                replay = app.replay
                if time.monotonic() > deadline:
                    raise RuntimeError("Packaged demo timed out")
                if not replay.job or not getattr(replay.job, "delivered", False):
                    root.after(100, lambda: inspect_replay(deadline))
                    return
                if replay.job.error:
                    raise RuntimeError(replay.job.error + "\n" + replay.job.log)
                if len(replay.review.rows("Detected")) != 5:
                    raise RuntimeError("Packaged real example did not detect all five placements")
                replay.seek(180)
                if replay.map.state["positions"].get("Red mini", {}).get("cell") != "K21":
                    raise RuntimeError("Packaged real example did not show the tracked mini at K21")
                smoke_result.update(replayDemo=True, replayEvents=5, mapMinis=1,
                                    replayWorkerReaped=replay.job.process.returncode == 0)
                root.quit()
            except Exception as exc:
                smoke_result.update(ok=False, error=str(exc))
                root.quit()

        def inspect_startup():
            try:
                import platform
                import v3_tracking
                import ring_scan_dialog
                from app_paths import data_dir, resource_path
                from pathlib import Path
                if service.running or app.busy:
                    raise RuntimeError("Smoke test unexpectedly started a service")
                if app.status.get() != "Idle - camera off":
                    raise RuntimeError(app.status.get())
                if not root.winfo_viewable():
                    raise RuntimeError("Home window is not visible")
                if not Path(resource_path("maps/dnd1.jpg")).is_file():
                    raise RuntimeError("Bundled demo map is missing")
                # Check native CV/NumPy dependencies without opening a camera.
                import cv2
                import numpy as np
                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                marker = cv2.aruco.generateImageMarker(dictionary, 10, 80)
                padded = cv2.copyMakeBorder(marker, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
                _corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary).detectMarkers(padded)
                if ids is None or ids.flatten().tolist() != [10]:
                    raise RuntimeError("Packaged ArUco detector failed its synthetic check")
                if not np.allclose(np.linalg.inv(np.eye(3)), np.eye(3)):
                    raise RuntimeError("Packaged NumPy linear algebra failed")
                smoke_result.update(
                    ok=True, frozen=bool(getattr(sys, "frozen", False)),
                    architecture=platform.machine(), dataDirectory=str(data_dir()),
                    miniIds=list(app.tree.get_children()),
                    windowSize=[root.winfo_width(), root.winfo_height()],
                    relayRunning=service.running,
                    nativeCvCheck=True,
                )
                # A smoke check must not bypass authentication to start a replay.
                if not app.auth.allowed:
                    smoke_result.update(authLocked=True, replayDemo=False)
                    root.quit()
                    return
                from replay_window import ReplayWindow
                import time
                app.replay = ReplayWindow(root, resource_path("demo/tabletop.mp4"), demo=True)
                root.after(100, lambda: inspect_replay(time.monotonic() + 45))
            except Exception as exc:
                smoke_result.update(ok=False, error=str(exc))
                root.quit()

        root.after(1500, inspect_startup)
    try:
        root.mainloop()
    finally:
        app.close()
    if smoke_report is not None:
        from app_paths import atomic_write_json
        smoke_result["relayStopped"] = not service.running
        atomic_write_json(smoke_report, smoke_result, backup=False)
        if not smoke_result.get("ok"):
            raise RuntimeError(f"Packaged startup check failed: {smoke_result}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Sarween miniature tracker")
    parser.add_argument("--smoke-test-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    main(smoke_report=args.smoke_test_report)
