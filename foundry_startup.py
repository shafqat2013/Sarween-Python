"""Explicit native Foundry launch and login, scoped to one local application.

No work happens on import. The controller starts only at a user's request.
Passwords stay in that request, never in preferences, logs or command arguments.
Foundry's own --world flag starts the world; a loopback debugging channel to the
instance launched here submits its ordinary login form. No Accessibility grant,
browser profile edits, Foundry file edits or simulated keystrokes are required.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import plistlib
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from app_paths import atomic_write_json, data_path, resource_path


APP_PATH = "/Applications/Foundry Virtual Tabletop.app"
BUNDLE_ID = "com.electron.foundryvtt"
DEFAULT_DATA = str(Path.home() / "Library/Application Support/FoundryVTT")
DEFAULTS = {"appPath": APP_PATH, "dataPath": DEFAULT_DATA,
            "baseUrl": "http://localhost:30000", "worldId": "testing",
            "username": "Gamemaster", "unpause": True, "placeWindow": False}
_WORLD = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")


def normalize_preferences(values):
    if not isinstance(values, dict) or set(values) - set(DEFAULTS):
        raise ValueError("Unknown Foundry startup preference.")
    result = {**DEFAULTS, **values}
    for key in ("appPath", "dataPath"):
        value = result[key]
        if not isinstance(value, str) or not value or "\0" in value:
            raise ValueError("Choose a valid Foundry application and data folder.")
        result[key] = str(Path(value).expanduser().absolute())
    if not isinstance(result["worldId"], str) or not _WORLD.fullmatch(result["worldId"]):
        raise ValueError("Choose an installed Foundry world.")
    name = result["username"]
    if not isinstance(name, str) or not name.strip() or len(name) > 100 or any(ord(c) < 32 for c in name):
        raise ValueError("Enter the Foundry user's name.")
    result["username"] = name.strip()
    for key in ("unpause", "placeWindow"):
        if type(result[key]) is not bool:
            raise ValueError("Choose on or off for startup options.")
    url = urllib.parse.urlsplit(result["baseUrl"])
    if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username or url.password or url.query or url.fragment
            or ".." in urllib.parse.unquote(url.path).split("/")):
        raise ValueError("Automatic login currently requires a local HTTP Foundry address.")
    try:
        port = url.port
    except ValueError:
        raise ValueError("Enter a valid local Foundry port.") from None
    if port is None or not 1 <= port <= 65535:
        raise ValueError("Include Foundry's port in its local address.")
    result["baseUrl"] = result["baseUrl"].rstrip("/")
    return result


def installed_worlds(data_folder):
    root = Path(data_folder).expanduser() / "Data/worlds"
    result = []
    for manifest in sorted(root.glob("*/world.json")):
        try:
            if manifest.is_symlink() or manifest.stat().st_size > 2_000_000:
                continue
            value = json.loads(manifest.read_text())
            identity = value.get("id")
            if not isinstance(identity, str) or not _WORLD.fullmatch(identity) or identity != manifest.parent.name:
                continue
            result.append({"id": identity, "name": str(value.get("title") or identity)[:200]})
        except (OSError, ValueError, TypeError):
            continue
    return result


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


class NativeFoundryLauncher:
    """Small fixed-purpose adapter; it never attaches to another browser."""

    supported = sys.platform == "darwin"

    def __init__(self):
        self._process = None
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def running_pids(self):
        if not self.supported:
            return []
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Start Foundry from the main application thread.")
        from AppKit import NSRunningApplication
        return [int(app.processIdentifier()) for app in
                NSRunningApplication.runningApplicationsWithBundleIdentifier_(BUNDLE_ID)
                if not app.isTerminated()]

    def launch(self, config, debug_port):
        if not self.supported:
            raise RuntimeError("Automatic native Foundry launch is currently available on macOS.")
        bundle = Path(config["appPath"])
        try:
            with (bundle / "Contents/Info.plist").open("rb") as stream:
                info = plistlib.load(stream)
        except (OSError, ValueError) as exc:
            raise ValueError("Choose the installed Foundry Virtual Tabletop application.") from exc
        executable_name = info.get("CFBundleExecutable", "")
        if (info.get("CFBundleIdentifier") != BUNDLE_ID or not executable_name
                or Path(executable_name).name != executable_name):
            raise ValueError("The selected application is not Foundry Virtual Tabletop.")
        executable = bundle / "Contents/MacOS" / executable_name
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Foundry's application executable is unavailable.")
        if config["worldId"] not in {row["id"] for row in installed_worlds(config["dataPath"])}:
            raise ValueError("The selected world is not installed in that Foundry data folder.")
        # These are argument values, never shell text. Credentials are absent.
        self._process = subprocess.Popen([
            str(executable), "--world=" + config["worldId"],
            "--dataPath=" + config["dataPath"],
            "--port=" + str(urllib.parse.urlsplit(config["baseUrl"]).port),
            "--remote-debugging-address=127.0.0.1",
            "--remote-debugging-port=" + str(debug_port),
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        return self._process.pid

    def alive(self):
        return self._process is not None and self._process.poll() is None

    def _json(self, port, path):
        with self._opener.open(f"http://127.0.0.1:{int(port)}{path}", timeout=1) as response:
            content = response.read(2_000_001)
        if len(content) > 2_000_000:
            raise RuntimeError("Foundry returned an oversized startup response.")
        return json.loads(content)

    @staticmethod
    def _same_foundry_origin(url, base):
        actual, expected = urllib.parse.urlsplit(url), urllib.parse.urlsplit(base)
        prefix = expected.path.rstrip("/")
        return (actual.scheme == expected.scheme == "http"
                and actual.hostname in {"localhost", "127.0.0.1", "::1"}
                and actual.port == expected.port
                and actual.path in {prefix + suffix for suffix in ("/", "", "/join", "/game", "/setup")})

    def _evaluate(self, port, target, expression):
        url = target.get("webSocketDebuggerUrl", "")
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "ws" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.port != port or not parsed.path.startswith("/devtools/page/")):
            raise RuntimeError("Foundry's startup channel is not local to this launch.")
        from websockets.sync.client import connect
        with connect(url, open_timeout=2, close_timeout=.2, max_size=2_000_000, proxy=None) as channel:
            channel.send(json.dumps({"id": 1, "method": "Runtime.evaluate", "params": {
                "expression": expression, "returnByValue": True, "awaitPromise": True}}))
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline:
                reply = json.loads(channel.recv(timeout=max(.01, deadline - time.monotonic())))
                if reply.get("id") != 1:
                    continue
                if reply.get("error") or reply.get("result", {}).get("exceptionDetails"):
                    # Never include raw remote exceptions: expressions can hold
                    # an in-memory password while submitting the login form.
                    raise RuntimeError("Foundry could not complete its normal login request.")
                return reply.get("result", {}).get("result", {}).get("value") or {}
        raise TimeoutError("Waiting for Foundry's startup page.")

    def prepare_login(self, port, config, password):
        targets = self._json(port, "/json/list")
        if not isinstance(targets, list):
            raise RuntimeError("Foundry's startup page is unavailable.")
        pages = [target for target in targets if target.get("type") == "page"
                 and self._same_foundry_origin(target.get("url", ""), config["baseUrl"])]
        if len(pages) != 1:
            return {"phase": "loading", "message": "Waiting for Foundry's local game window…"}
        # Fixed script, with data JSON-encoded separately. No caller-provided
        # JavaScript, arbitrary URL, file path or browser target is accepted.
        page_url = urllib.parse.urlsplit(pages[0]["url"])
        # localhost and 127.0.0.1 are equivalent for selecting the owned page,
        # but browser cookies and fetch credentials still require its exact host.
        page_base = urllib.parse.urlunsplit((page_url.scheme, page_url.netloc,
            urllib.parse.urlsplit(config["baseUrl"]).path.rstrip("/"), "", ""))
        arguments = json.dumps({"world": config["worldId"], "user": config["username"],
                                "password": password, "base": page_base})
        script = Path(resource_path("foundry_startup_login.js")).read_text(encoding="utf-8")
        expression = "(" + script + ")(" + arguments + ")"
        return self._evaluate(port, pages[0], expression)


class FoundryStartupController:
    """Cancellable worker; Tk polls status and performs native placement itself."""

    def __init__(self, *, backend=None, path=None, timeout=150, clock=None, pause=None,
                 read_status=None, request=None, connection_grace=10):
        self.backend = backend or NativeFoundryLauncher()
        self.path = Path(path) if path is not None else data_path("foundry_startup.json")
        try:
            self.preferences = normalize_preferences(json.loads(self.path.read_text()))
        except (OSError, ValueError, TypeError):
            self.preferences = deepcopy(DEFAULTS)
        self._clock = clock or time.monotonic
        self._pause = pause
        self._read_status = read_status
        self._request = request
        self.timeout = float(timeout)
        self.connection_grace = max(0, float(connection_grace))
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._thread = None
        self._owned = None
        self._enabled = True
        self._state = {"phase": "idle", "busy": False,
                       "message": "Choose your world, then start Foundry from Sarween.",
                       "placementRequested": False}

    def defaults(self):
        return deepcopy(self.preferences)

    def worlds(self, data_folder=None):
        return installed_worlds(data_folder or self.preferences["dataPath"])

    def status(self):
        with self._lock:
            return deepcopy(self._state)

    def _state_set(self, phase, message, **values):
        with self._lock:
            self._state.update(phase=phase, message=message, **values)

    def set_enabled(self, enabled):
        self._enabled = bool(enabled)
        if not enabled:
            self.cancel()

    def cancel(self):
        self._cancel.set()
        with self._lock:
            self._state["placementRequested"] = False

    def _stopped(self):
        return self._cancel.is_set() or not self._enabled

    def _wait(self):
        if self._pause is not None:
            self._pause(.2)
        else:
            self._cancel.wait(.2)

    def start(self, values, *, password=""):
        config = normalize_preferences(values)
        if not isinstance(password, str) or len(password) > 4096:
            raise ValueError("Enter a valid Foundry password.")
        if self.status()["busy"] or not self._enabled:
            return False
        # Inventory belongs to the initiating/main thread, before the worker.
        pids = self.backend.running_pids()
        if config["worldId"] not in {world["id"] for world in self.worlds(config["dataPath"])}:
            raise ValueError("Choose a world installed in the selected Foundry data folder.")
        atomic_write_json(self.path, config)
        self.preferences = config
        self._cancel.clear()
        self._state_set("starting", "Starting Foundry…", busy=True, placementRequested=False)
        self._thread = threading.Thread(target=self._run, args=(config, password, pids),
                                        name="Sarween-Foundry-Startup", daemon=True)
        self._thread.start()
        return True

    def _status(self):
        if self._read_status is not None:
            return self._read_status()
        import foundryoutput
        return foundryoutput.get_vtt_status()

    def _command(self, action, values):
        if self._request is not None:
            return self._request(action, values)
        import foundryoutput
        return foundryoutput.request_vtt_command(action, values, return_request_id=True)

    @staticmethod
    def _new_debug_port():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            return listener.getsockname()[1]

    @staticmethod
    def _check_requested_session(status, config):
        if not status.get("connected"):
            return
        world, user = status.get("world") or {}, status.get("user") or {}
        if world.get("id") and world["id"] != config["worldId"]:
            raise RuntimeError("A different Foundry world is connected. Select that world in Sarween, or switch "
                               f"Foundry to '{config['worldId']}' and join as '{config['username']}', then try again. "
                               "Foundry stays open.")
        if ((user.get("name") and user["name"] != config["username"])
                or user.get("isGM") is False):
            raise RuntimeError(f"Foundry is connected as a different user. Join '{config['worldId']}' as "
                               f"'{config['username']}' with Gamemaster permissions, then try again. "
                               "Foundry stays open.")

    def _run(self, config, password, pids):
        try:
            deadline = self._clock() + self.timeout
            remote = self._status()
            unmanaged = bool(pids and (not self._owned or self._owned["pid"] not in pids))
            if unmanaged:
                # Existing native windows have no startup channel owned here.
                # Wait for their ordinary module connection; never attach to
                # their browser/debugging channel or launch a second instance.
                password = ""
            # Starting the relay and receiving the already-open world's module
            # handshake are separate asynchronous steps. Give that native window
            # a quiet reconnect grace before showing manual-join guidance.
            if unmanaged and not remote.get("connected"):
                self._state_set("waiting_module", "Foundry is already open. Waiting for its Sarween module…")
                reconnect_deadline = min(deadline, self._clock() + self.connection_grace)
                while not self._stopped() and not remote.get("connected") and self._clock() < reconnect_deadline:
                    self._wait()
                    remote = self._status()
            if self._stopped():
                return
            self._check_requested_session(remote, config)
            connected = remote.get("connected") and (remote.get("world") or {}).get("id") == config["worldId"]
            if not connected and not remote.get("connected") and not unmanaged:
                if pids:
                    port = self._owned["port"]
                else:
                    if self._stopped():
                        return
                    port = self._new_debug_port()
                    pid = self.backend.launch(config, port)
                    self._owned = {"pid": pid, "port": port}
                while not self._stopped() and self._clock() < deadline:
                    if not self.backend.alive():
                        raise RuntimeError("Foundry exited before startup completed. Open it normally to check its startup message.")
                    try:
                        step = self.backend.prepare_login(port, config, password)
                    except (OSError, ValueError, TimeoutError, urllib.error.URLError):
                        step = {"phase": "loading", "message": "Waiting for Foundry's local game window…"}
                    phase = step.get("phase")
                    if phase == "error":
                        raise RuntimeError(step.get("message") or "Foundry could not log in.")
                    if phase == "joined":
                        break
                    self._state_set(phase or "loading", step.get("message") or "Loading Foundry…")
                    self._wait()
                else:
                    if self._stopped():
                        return
                    raise TimeoutError("Foundry did not finish starting. Check its window for an update, license, migration or login message, then retry.")
            password = ""
            waiting_message = (f"Foundry is already open. Join '{config['worldId']}' as '{config['username']}', "
                               "enable or reload the Sarween module, and open a scene. "
                               "Sarween will continue automatically when connected."
                               if unmanaged else "Waiting for the Sarween module and active scene…")
            self._state_set("waiting_module", waiting_message)
            request_id = None
            while not self._stopped() and self._clock() < deadline:
                status = self._status()
                if self._stopped():
                    return
                self._check_requested_session(status, config)
                world = status.get("world") or {}
                ready = (status.get("connected") and status.get("supported") and not status.get("blocked")
                         and world.get("id") == config["worldId"]
                         and (status.get("user") or {}).get("name") == config["username"]
                         and (status.get("user") or {}).get("isGM") is True)
                if not ready:
                    self._wait()
                    continue
                if config["unpause"]:
                    if request_id:
                        result = status.get("lastResult") or {}
                        if result.get("requestId") == request_id:
                            if not result.get("success"):
                                raise RuntimeError(result.get("message") or "Foundry did not confirm unpausing.")
                            if status.get("paused") is not False:
                                self._wait()
                                continue
                            break
                    elif not status.get("pending"):
                        if self._stopped():
                            return
                        self._state_set("unpausing", "Unpausing the Foundry game…")
                        request_id = self._command("lifecycle.setPaused", {"paused": False})
                        if not isinstance(request_id, str) or not request_id:
                            raise RuntimeError("Foundry could not accept the unpause request. Check the connection and try again.")
                else:
                    break
                self._wait()
            else:
                if self._stopped():
                    return
                if unmanaged:
                    raise TimeoutError(f"Still waiting for the already-open Foundry. Join '{config['worldId']}' as "
                                       f"'{config['username']}', enable or reload the Sarween module, and open a scene, "
                                       "then click Start Foundry again. Foundry stays open.")
                raise TimeoutError("Foundry opened, but its Sarween module or active scene is not ready. Enable the module, open a scene, then retry.")
            if self._stopped():
                return
            self._state_set("ready", "Foundry is ready." + (" Moving its window next…" if config["placeWindow"] else ""),
                            placementRequested=config["placeWindow"])
        except Exception as exc:
            self._state_set("failed", str(exc), placementRequested=False)
        finally:
            password = ""
            if self._stopped():
                self._state_set("cancelled", "Startup automation cancelled. Foundry stays open.", placementRequested=False)
            with self._lock:
                self._state["busy"] = False

    def take_placement_request(self):
        with self._lock:
            value = self._state["placementRequested"] and not self._stopped()
            self._state["placementRequested"] = False
            return value


_controller = None


def get_foundry_startup_controller():
    global _controller
    if _controller is None:
        _controller = FoundryStartupController()
    return _controller
