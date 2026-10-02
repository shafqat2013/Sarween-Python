"""Explicit, stoppable ownership of the local Foundry relay."""

import asyncio
import threading


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
