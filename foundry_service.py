"""Explicit, stoppable ownership of the local Foundry relay."""

import asyncio
from collections.abc import Mapping
import math
import threading


def validated_scene_info(info):
    """Return complete, JSON-safe scene geometry, never fallback defaults.

    This validates one received snapshot. Whether it belongs to the current
    connection is a separate condition maintained by foundryoutput.
    """
    if not isinstance(info, Mapping) or info.get("error"):
        return None
    scene_id = info.get("sceneId")
    if not isinstance(scene_id, str) or not scene_id.strip():
        return None
    result = {"sceneId": scene_id.strip()}
    for name, aliases in (("sceneW", ("sceneW", "width")),
                          ("sceneH", ("sceneH", "height")),
                          ("gridPx", ("gridPx", "gridSize", "grid"))):
        value = next((info[key] for key in aliases if info.get(key) is not None), None)
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number) or number <= 0:
            return None
        result[name] = int(number) if number.is_integer() else number
    return result


class FoundryService:
    def __init__(self, serve=None):
        self._serve = serve
        self._thread = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.error = ""
        self.access = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    @property
    def state(self):
        if self.error:
            return "Failed"
        if not self.running:
            return "Stopped"
        return "Listening" if self._ready.is_set() else "Starting"

    def startup_status(self):
        """Nonblocking startup snapshot for a main-window ``after`` poll.

        A bound relay socket is not a module handshake, and retained geometry
        is not a scene received on this connection. No widgets, waits, control
        commands or timeout-driven retries are created by this method.
        """
        state = self.state
        status = {"phase": "starting_relay", "message": "Starting the Foundry connection…",
                  "scene_info": None, "error": ""}
        if state in ("Failed", "Stopped"):
            error = str(self.error or "The Foundry relay stopped before setup completed.")
            return {**status, "phase": "failed", "message": error, "error": error}
        if state != "Listening":
            return status
        try:
            from foundryoutput import get_connection_status
            connection = get_connection_status()
        except Exception as exc:
            error = f"Could not read the Foundry connection: {exc}"
            return {**status, "phase": "failed", "message": error, "error": error}
        if not connection.get("connected"):
            return {**status, "phase": "waiting_module",
                    "message": "Waiting for Foundry. Open your world with the Sarween module enabled."}
        if not connection.get("handshake_ready"):
            return {**status, "phase": "waiting_module",
                    "message": connection.get("error") or
                        "Foundry connected. Waiting for the Sarween module handshake…"}
        scene = validated_scene_info(connection.get("scene_info"))
        if not connection.get("scene_ready") or scene is None:
            return {**status, "phase": "waiting_scene",
                    "message": "Waiting for an active Foundry scene and its grid. Open a scene in Foundry."}
        return {**status, "phase": "ready", "message": "Foundry scene connected. Starting your session…",
                "scene_info": scene}

    def start(self):
        if self.access is not None:
            self.access.require()
        if self.running:
            return
        self.error = ""
        self._stop.clear()
        self._ready.clear()
        if self._serve is None:
            from foundryoutput import main
            self._serve = main

        def run():
            try:
                asyncio.run(self._serve(stop_event=self._stop, ready_event=self._ready))
            except Exception as exc:
                self.error = str(exc)
            finally:
                self._ready.clear()

        self._thread = threading.Thread(target=run, name="Sarween-Foundry", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
            if self._thread.is_alive():
                raise RuntimeError("Foundry relay did not stop within 4 seconds")
