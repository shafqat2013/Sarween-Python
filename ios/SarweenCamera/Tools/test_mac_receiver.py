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
        def start(path=jpeg, preset="standard", orientation="portrait"):
            return subprocess.Popen([str(harness), str(path), preset, orientation], stdout=subprocess.DEVNULL,
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
            stop(process)
            process = None
            formats = []
            source_ids = {}
            for preset, orientation, width, height, fps in (
                    ("standard", "portrait", 1080, 1920, 15),
                    ("smooth", "portrait", 1080, 1920, 30),
                    ("detail", "portrait", 2160, 3840, 15),
                    ("standard", "landscape", 1920, 1080, 15)):
                sample = np.zeros((height, width, 3), np.uint8)
                sample[:, :width // 2] = (30, 50, 220)
                sample[:, width // 2:] = (220, 100, 20)
                path = Path(directory) / f"SYNTHETIC-{preset}-{orientation}.jpg"
                assert cv2.imwrite(str(path), sample)
                process = start(path, preset, orientation)
                capture = PhoneCameraCapture("SYNTHETIC-NOT-A-USB-DEVICE", transport=SyntheticLocalTransport(),
                    read_timeout=.3, reconnect_delays=(.05,), packet_timeout=3)
                pixels, metadata = wait_frame(capture)
                assert pixels.shape == (height, width, 3)
                assert metadata["format"]["fps"] == fps
                assert metadata["format"]["capturePreset"] == preset
                assert pixels[100, 100, 2] > 180 and pixels[100, width - 100, 0] > 180
                _, later = wait_frame(capture)
                assert later["sessionId"] == metadata["sessionId"] and later["seq"] > metadata["seq"]
                source_ids[preset, orientation] = metadata["sourceId"]
                formats.append({"preset": preset, "orientation": orientation, "width": width,
                                "height": height, "nominalFPS": fps, "decoded": True})
                capture.release()
                stop(process)
                process = None
            assert source_ids["standard", "portrait"] == first["sourceId"]
            assert source_ids["smooth", "portrait"] == first["sourceId"]
            assert source_ids["detail", "portrait"] != first["sourceId"]
            assert source_ids["standard", "landscape"] != first["sourceId"]
            checks.extend([
                "Standard, Smooth, Detail and landscape synthetic JPEGs decode at their declared native dimensions",
                "Preset and nominal FPS metadata survive the production receiver handshake",
                "FPS-only Standard/Smooth preserve optical identity; resolution/orientation changes do not",
            ])
            return {"scope": "SYNTHETIC localhost, production Swift server and Python receiver; NOT actual USB/camera",
                    "passed": len(checks), "checks": checks, "syntheticFormats": formats,
                    "performanceCaveat": "Nominal producer FPS is configured test input, not measured iPhone or USB performance"}
        finally:
            if capture is not None:
                capture.release()
            if process is not None:
                stop(process)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: test_mac_receiver.py /path/to/compiled/USBProtocolHarness")
    print(json.dumps(run(Path(sys.argv[1]).resolve()), indent=2))
