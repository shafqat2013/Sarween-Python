#!/usr/bin/env python3
"""Production Mac receiver versus the Swift listener, with SYNTHETIC images.

Developer-only check. Requires a precompiled USBProtocolHarness executable.
This bypasses usbmux and uses localhost; it proves neither USB nor real capture.
"""
from pathlib import Path
import json
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from phone_camera import PhoneCameraCapture


class SyntheticLocalTransport:
    def connect(self, device_id, port, *, cancel_event=None):
        assert device_id == "SYNTHETIC-NOT-A-USB-DEVICE"
        return socket.create_connection(("127.0.0.1", port), timeout=.2)


def wait_frame(capture, predicate=lambda metadata: True, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        metadata = capture.last_frame_metadata
        if ok and predicate(metadata):
            return frame, metadata
    raise AssertionError(f"No matching frame: {capture.status}")


def run(harness):
    process = capture = None
    checks = []
    with tempfile.TemporaryDirectory(prefix="sarween-synthetic-protocol-") as directory:
        jpeg = Path(directory) / "SYNTHETIC.jpg"
        image = np.zeros((1920, 1080, 3), np.uint8)
        image[:, :540] = (30, 50, 220)
        image[:, 540:] = (220, 100, 20)
        cv2.putText(image, "SYNTHETIC TEST", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        assert cv2.imwrite(str(jpeg), image)
        def start():
            return subprocess.Popen([str(harness), str(jpeg)], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        def stop(child):
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=2)
        try:
            process = start()
            capture = PhoneCameraCapture("SYNTHETIC-NOT-A-USB-DEVICE", transport=SyntheticLocalTransport(),
                read_timeout=.3, reconnect_delays=(.05,), packet_timeout=2)
            frame, first = wait_frame(capture)
            assert frame.shape == (1920, 1080, 3)
            assert frame[100, 100, 2] > 180 and frame[100, 900, 0] > 180
            checks.append("Swift JPEG decoded with actual portrait pixels and expected synthetic colors")
            _, next_frame = wait_frame(capture)
            assert next_frame["sessionId"] == first["sessionId"] and next_frame["seq"] > first["seq"]
            assert next_frame["captureTimeNs"] > first["captureTimeNs"]
            checks.append("Production receiver sends valid ACKs and receives strictly advancing frames")
            # Intentional consumer stall verifies the single latest slot, not a video queue.
            time.sleep(.35)
            _, newest = wait_frame(capture)
            assert newest["seq"] > next_frame["seq"] + 1 and newest["droppedFrames"] > 0
            checks.append("Slow consumer replaces old frames in the latest slot")
            stop(process)
            process = None
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                ok, absent = capture.read()
                if not ok and not capture.status["fresh"]:
                    assert absent is None
                    break
            else:
                raise AssertionError("Disconnected source remained fresh")
            checks.append("Stopped Swift server produces an explicit missing frame, never frozen fresh pixels")
            process = start()
            _, restarted = wait_frame(capture, lambda metadata: metadata["sessionId"] != first["sessionId"])
            assert restarted["connectionGeneration"] > first["connectionGeneration"]
            checks.append("Swift restart reconnects with a new session UUID and receiver generation")
            started = time.monotonic()
            capture.release()
            assert time.monotonic() - started < 2 and not capture.isOpened()
            checks.append("Release cancels the production receiver promptly")
            return {"scope": "SYNTHETIC localhost, production Swift server and Python receiver; NOT actual USB/camera",
                    "passed": len(checks), "checks": checks}
        finally:
            if capture is not None:
                capture.release()
            if process is not None:
                stop(process)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: test_mac_receiver.py /path/to/compiled/USBProtocolHarness")
    print(json.dumps(run(Path(sys.argv[1]).resolve()), indent=2))
