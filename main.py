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


def run_live_session(service, *, observer_only=False, startup=None):
    service.access.require()
    if startup is None:
        s.initialize(access=service.access)
    else:
        s.initialize(access=service.access, selection=startup.selection_data,
                     displays=startup.displays, webcams=startup.webcams)
    service.access.require()
    if s.selected_mode == s.MODE_FOUNDRY:
        service.start()
        if startup is not None:
            from session_startup import StartupPending
            state = service.startup_status()
            if state["phase"] != "ready":
                raise StartupPending()
            startup.scene_info = state["scene_info"]
        # A cold macOS startup can take longer than the socket bind itself.
        # Let the existing relay finish starting before entering camera setup.
        else:
            import time
            deadline = time.monotonic() + 10
            while service.state == "Starting" and time.monotonic() < deadline:
                time.sleep(0.02)
            if service.state != "Listening":
                raise RuntimeError(service.error or
                    f"Foundry relay is {service.state.lower()}. Wait a moment, then retry Start session.")
    if startup is not None and s.selected_mode == s.MODE_FOUNDRY:
        c.calibrate(access=service.access, scene_info=startup.scene_info)
    else:
        c.calibrate(access=service.access)
    service.access.require()

    import v3_tracking as t
    from mini_calibration import run_mini_calibration

    try:
        options = {"access": service.access, "foundry_service": service,
                   "observation_only": observer_only}
        if startup is not None:
            options.update(control_parent=startup.runtime_parent, control_started=startup.runtime_started)
            from camera_sources import is_phone_camera_index
            if is_phone_camera_index(startup.selection_data.get("webcam_index")):
                capture = startup.take_camera_capture()
                if capture is not None:
                    options["camera_capture"] = capture
        while True:
            result = t.begin_session(on_mini_moved, **options)
            options.pop("camera_capture", None)
            if result != "recalibrate":
                break
            service.access.require()
            run_mini_calibration()
    finally:
        # Idempotent release also covers constructor failures before ownership
        # reaches the session's normal cleanup path.
        if "options" in locals() and options.get("camera_capture") is not None:
            options["camera_capture"].release()
        t.close_pending_recording()


def run_observer_session(service, *, startup=None):
    # Inhibit output before setup/camera startup, rather than waiting for the
    # first observer window to be drawn. Cancellation must also clear the guard.
    fo.set_observation_only(True)
    try:
        return (run_live_session(service, observer_only=True) if startup is None else
                run_live_session(service, observer_only=True, startup=startup))
    finally:
        # Inline setup owns the guard across a delayed connection or an
        # inline retry. AppWindow releases it on cancel/session completion.
        if startup is None:
            fo.set_observation_only(False)


def main(*, smoke_report=None, dev_no_login=False):
    if dev_no_login and (getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS")):
        raise ValueError("Developer login bypass is only available when running from source.")
    auth_panel_factory = None
    if dev_no_login:
        from dev_access import DeveloperAuthPanel
        auth_panel_factory = DeveloperAuthPanel
        print("[DEV] Developer mode — login bypassed for this source session.", flush=True)
    initialize_user_data()
    from app_window import AppWindow
    root = s._get_persistent_root()
    service = FoundryService()
    app = AppWindow(root, service=service, run_session=run_live_session,
                    auth_panel_factory=auth_panel_factory, run_observer=run_observer_session)
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
    if not getattr(sys, "frozen", False) and not hasattr(sys, "_MEIPASS"):
        parser.add_argument("--dev-no-login", action="store_true",
                            help="Bypass login for this local development session.")
    args = parser.parse_args()
    main(smoke_report=args.smoke_test_report,
         dev_no_login=getattr(args, "dev_no_login", False))
