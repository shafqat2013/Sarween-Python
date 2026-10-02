"""One owned, cancellable replay process. No camera or Foundry connections."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time


class ReplayJob:
    def __init__(self, video, options=None, *, timeout=180, command_factory=None):
        if not math.isfinite(timeout) or not 0 < timeout <= 900:
            raise ValueError("Replay timeout must be between 0 and 900 seconds")
        from tracking_regression import worker_command
        self.directory = tempfile.TemporaryDirectory(prefix="sarween-replay-")
        folder = Path(self.directory.name)
        self.response = folder / "response.json"
        self.progress_file = folder / "progress.json"
        self.log_path = folder / "worker.log"
        self.result = None
        self.error = None
        self.log = ""
        self.progress = {}
        self.closed = False
        self.started = time.monotonic()
        self.timeout = timeout
        request = folder / "request.json"
        opts = {**(options or {}), "progress_path": str(self.progress_file)}
        request.write_text(json.dumps({"video_path": str(video), "options": opts, "threads": 1,
                                      "timeout_seconds": timeout, "parent_pid": os.getpid()}, default=str))
        env = os.environ.copy()
        env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
        try:
            with self.log_path.open("w") as log:
                self.process = subprocess.Popen((command_factory or worker_command)(request, self.response),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env, start_new_session=True)
        except BaseException:
            self.directory.cleanup()
            raise

    def poll(self):
        if self.closed:
            return True
        if self.progress_file.exists():
            try:
                self.progress = json.loads(self.progress_file.read_text())
            except (ValueError, OSError):
                pass
        if self.process.poll() is None:
            if time.monotonic() - self.started >= self.timeout:
                self.cancel("Analysis reached its time limit. No worker was left running.")
                return True
            return False
        self.process.wait()
        try:
            result = json.loads(self.response.read_text())
            if self.process.returncode or result.get("error"):
                self.error = result.get("error") or f"Replay exited with code {self.process.returncode}"
            else:
                self.result = result
        except (ValueError, OSError) as exc:
            self.error = f"Replay did not return a valid result: {exc}"
        self._cleanup()
        return True

    def cancel(self, reason="Analysis cancelled"):
        if self.closed:
            return
        # Kill only the process group created by this job, then reap it.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait()
        self.error = reason
        self._cleanup()

    def _cleanup(self):
        try:
            self.log = self.log_path.read_text(errors="replace")[-16000:]
        except OSError:
            pass
        self.closed = True
        self.directory.cleanup()
