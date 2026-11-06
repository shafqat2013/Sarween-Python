"""Independent wire/freshness contracts; never opens a camera or real socket."""
from collections import deque
import json
import plistlib
import socket
import struct
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from phone_camera import PhoneCameraCapture
from phone_camera_protocol import ProtocolError, read_packet, validate_frame
from phone_usb import PhoneUSBError, USBMuxClient


def packet(header, payload=b""):
    """Construct published v1 bytes independently of the production encoder."""
    raw = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack(">I", len(raw)) + raw + payload


def hello(session="stream-one", width=32, height=24):
    return packet({"type": "hello", "protocolVersion": 1, "sessionId": session,
                   "deviceName": "Test iPhone", "payloadBytes": 0,
                   "captureFormat": {"width": width, "height": height, "fps": 15,
                                     "pixelFormat": "jpeg", "lens": "wide", "zoom": 1.0,
                                     "stabilization": "off"}})


def frame(seq, stamp, session="stream-one", *, declared=(32, 24), actual=None):
    width, height = actual or declared
    image = np.full((height, width, 3), seq * 20, np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    payload = encoded.tobytes()
    header = {"type": "frame", "protocolVersion": 1, "sessionId": session,
              "payloadBytes": len(payload), "seq": seq, "captureTimeNs": stamp,
              "width": declared[0], "height": declared[1],
              "rotationDegrees": 0, "mirrored": False}
    return packet(header, payload)


class Wire:
    """Thread-safe in-memory duplex stream with real recv-count semantics."""
    def __init__(self, incoming=b"", *, fragment=None, intermittent_timeouts=False):
        self._condition = threading.Condition()
        self._incoming = bytearray(incoming)
        self.fragment = fragment
        self.intermittent_timeouts = intermittent_timeouts
        self.recv_calls = 0
        self.sent = []
        self.closed = False
        self.timeout = .02

    def push(self, incoming):
        with self._condition:
            self._incoming.extend(incoming)
            self._condition.notify_all()

    def settimeout(self, value):
        self.timeout = value

    def recv(self, count):
        with self._condition:
            self.recv_calls += 1
            if self.intermittent_timeouts and self.recv_calls % 5 == 0:
                raise socket.timeout()
            if not self._incoming and not self.closed:
                self._condition.wait(min(.02, self.timeout or .02))
            if self.closed:
                return b""
            if not self._incoming:
                raise socket.timeout()
            count = min(count, self.fragment or count, len(self._incoming))
            result = bytes(self._incoming[:count])
            del self._incoming[:count]
            return result

    def sendall(self, value):
        with self._condition:
            if self.closed:
                raise OSError("closed fake stream")
            self.sent.append(bytes(value))

    def shutdown(self, _how):
        self.close()

    def close(self):
        with self._condition:
            self.closed = True
            self._condition.notify_all()


class Transport:
    def __init__(self, *wires):
        self.wires = deque(wires)
        self.connections = []

    def list_devices(self, *, cancel_event=None):
        return [{"id": "selected-phone", "transport": "usb"},
                {"id": "network-phone", "transport": "network"}]

    def connect(self, device_id, port, *, cancel_event=None):
        self.connections.append((device_id, port))
        if not self.wires:
            raise PhoneUSBError("app_unavailable", "No further fake connections")
        return self.wires.popleft()


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(.002)
    raise AssertionError("Fake-transport condition did not arrive within deadline")


class PhoneCameraContractTests(unittest.TestCase):
    def capture(self, *wires, **options):
        transport = Transport(*wires)
        capture = PhoneCameraCapture("selected-phone", transport=transport,
            read_timeout=.03, packet_timeout=2, reconnect_delays=(.01,), **options)
        self.addCleanup(capture.release)
        return capture, transport

    def test_fragmented_timeout_interrupted_and_coalesced_packets_keep_boundaries(self):
        stop = {"type": "stop", "protocolVersion": 1, "sessionId": "stream-one", "payloadBytes": 0}
        wire = Wire(hello() + frame(1, 900) + packet(stop), fragment=3, intermittent_timeouts=True)
        deadline = time.monotonic() + 1
        first, payload = read_packet(wire, deadline=deadline)
        self.assertEqual((first["type"], payload), ("hello", b""))
        second, payload = read_packet(wire, deadline=deadline)
        self.assertEqual((second["type"], second["seq"]), ("frame", 1))
        self.assertTrue(payload.startswith(b"\xff\xd8"))
        third, payload = read_packet(wire, deadline=deadline)
        self.assertEqual((third, payload), (stop, b""))

    def test_equal_sensor_timestamp_with_new_sequence_is_not_new_capture_evidence(self):
        header = {"type": "frame", "protocolVersion": 1, "sessionId": "stream-one",
                  "payloadBytes": 100, "seq": 2, "captureTimeNs": 900,
                  "width": 32, "height": 24, "rotationDegrees": 0, "mirrored": False}
        with self.assertRaises(ProtocolError):
            validate_frame(header, session_id="stream-one", previous_seq=1, previous_capture_ns=900)

    def test_latest_frame_replaces_backlog_and_read_does_not_repeat_it(self):
        capture, _ = self.capture(Wire(hello() + frame(1, 900) + frame(2, 1000) + frame(3, 1100)))
        wait_for(lambda: capture.status["droppedFrames"] == 2)
        ok, image = capture.read()
        self.assertTrue(ok)
        self.assertEqual(image.shape, (24, 32, 3))
        self.assertEqual(capture.last_frame_metadata["seq"], 3)
        self.assertEqual(capture.read(), (False, None))

    def test_phone_epoch_is_preserved_and_status_packets_do_not_refresh_stale_pixels(self):
        clock = Clock()
        wire = Wire(hello() + frame(1, 9_000_000_000_000_000))
        capture, _ = self.capture(wire, clock=clock, stale_timeout=.25)
        wait_for(lambda: capture.status["state"] == "streaming")
        self.assertTrue(capture.read()[0])
        metadata = capture.last_frame_metadata
        self.assertEqual(metadata["captureTimeNs"], 9_000_000_000_000_000)
        self.assertEqual(metadata["hostReceivedMonotonic"], 100.0)
        clock.value = 102.0
        wire.push(packet({"type": "status", "protocolVersion": 1, "sessionId": "stream-one",
                          "payloadBytes": 0, "message": "Still connected"}))
        wait_for(lambda: capture.status["message"] == "Still connected")
        self.assertFalse(capture.status["fresh"])
        self.assertEqual(capture.read(), (False, None))

    def test_unread_stale_frame_is_discarded(self):
        clock = Clock()
        capture, _ = self.capture(Wire(hello() + frame(1, 900)), clock=clock, stale_timeout=.25)
        wait_for(lambda: capture.status["state"] == "streaming")
        clock.value = 102.0
        self.assertEqual(capture.read(), (False, None))
        self.assertIsNone(capture.last_frame_metadata)

    def test_jpeg_dimension_mismatch_is_rejected_before_decode_or_acknowledgment(self):
        wire = Wire(hello() + frame(1, 900, actual=(40, 24)))
        with patch("phone_camera.cv2.imdecode") as decode:
            capture, _ = self.capture(wire)
            wait_for(lambda: capture.status["code"] == "invalid_stream")
            self.assertFalse(capture.status["fresh"])
            self.assertEqual(capture.read(), (False, None))
            decode.assert_not_called()
            commands = [read_packet(Wire(sent))[0]["type"] for sent in wire.sent]
            self.assertNotIn("ack", commands)

    def test_reconnect_changes_generation_and_keeps_selected_physical_device(self):
        first = Wire(hello() + frame(7, 100_000))
        second = Wire(hello("stream-two") + frame(1, 10, "stream-two"))
        capture, transport = self.capture(first, second)
        wait_for(lambda: capture.status["state"] == "streaming")
        self.assertTrue(capture.read()[0])
        before = capture.last_frame_metadata
        first.close()
        wait_for(lambda: capture.status["sessionId"] == "stream-two" and capture.status["fresh"])
        self.assertTrue(capture.read()[0])
        after = capture.last_frame_metadata
        self.assertEqual((before["sessionId"], before["connectionGeneration"]), ("stream-one", 1))
        self.assertEqual((after["sessionId"], after["connectionGeneration"]), ("stream-two", 2))
        self.assertEqual((after["seq"], after["captureTimeNs"]), (1, 10))
        self.assertEqual({device for device, _ in transport.connections}, {"selected-phone"})
        self.assertEqual(before["deviceId"], after["deviceId"])

    def test_reconnect_reusing_old_session_cannot_publish_restarted_sequence(self):
        first = Wire(hello() + frame(7, 100_000))
        reused = Wire(hello() + frame(1, 10))
        capture, _ = self.capture(first, reused)
        wait_for(lambda: capture.status["state"] == "streaming")
        self.assertTrue(capture.read()[0])
        first.close()
        wait_for(lambda: capture.status["code"] == "invalid_stream")
        self.assertEqual(capture.read(), (False, None))
        self.assertFalse(capture.status["fresh"])
        self.assertEqual(capture.last_frame_metadata["seq"], 7)
        self.assertEqual(capture.status["connectionGeneration"], 1)
        self.assertEqual(reused.sent, [])

    def test_ambiguous_usb_selection_never_opens_the_first_available_phone(self):
        transport = Transport()
        transport.list_devices = lambda **_: [
            {"id": "first-phone", "transport": "usb"},
            {"id": "second-phone", "transport": "usb"},
        ]
        capture = PhoneCameraCapture(transport=transport, read_timeout=.03)
        self.addCleanup(capture.release)
        wait_for(lambda: capture.status["code"] == "selection_required")
        self.assertFalse(capture.isOpened())
        self.assertEqual(capture.read(), (False, None))
        self.assertEqual(transport.connections, [])

    def test_release_unblocks_idle_receive_and_leaves_no_capture_worker(self):
        wire = Wire(hello())
        capture, _ = self.capture(wire)
        wait_for(lambda: capture.status["connected"])
        started = time.monotonic()
        capture.release()
        self.assertLess(time.monotonic() - started, .5)
        self.assertFalse(capture._thread.is_alive())
        self.assertFalse(capture.isOpened())
        self.assertEqual(capture.read(), (False, None))
        self.assertTrue(wire.closed)
        capture.release()


class USBMuxContractTests(unittest.TestCase):
    def test_usb_device_selection_network_order_and_raw_stream_handoff(self):
        replies = deque([
            {"DeviceList": [
                {"DeviceID": 7, "Properties": {"SerialNumber": "wireless", "ConnectionType": "Network"}},
                {"DeviceID": 19, "Properties": {"SerialNumber": "selected-phone", "ConnectionType": "USB"}},
            ]},
            {"DeviceList": [{"DeviceID": 20, "Properties": {"SerialNumber": "selected-phone", "ConnectionType": "USB"}}]},
            {"MessageType": "Result", "Number": 0},
        ])
        created, requests = [], []

        class DaemonWire(Wire):
            def connect(self, path):
                self.path = path

            def sendall(self, raw):
                length, version, kind, tag = struct.unpack("<IIII", raw[:16])
                self.assert_header = (length == len(raw), version, kind)
                request = plistlib.loads(raw[16:])
                requests.append(request)
                response = plistlib.dumps(replies.popleft())
                tail = hello() if request["MessageType"] == "Connect" else b""
                self.push(struct.pack("<IIII", 16 + len(response), 1, 8, tag) + response + tail)

        def factory(*_args):
            wire = DaemonWire(fragment=5)
            created.append(wire)
            return wire

        client = USBMuxClient(socket_factory=factory)
        devices = client.list_devices()
        self.assertEqual([item["id"] for item in devices], ["selected-phone"])
        self.assertTrue(created[0].closed)
        connection = client.connect("selected-phone")
        self.addCleanup(connection.close)
        self.assertEqual(requests[-1]["DeviceID"], 20)
        self.assertEqual(requests[-1]["PortNumber"], socket.htons(52781))
        self.assertEqual(created[-1].assert_header, (True, 1, 8))
        self.assertFalse(connection.closed)
        self.assertEqual(read_packet(connection)[0]["type"], "hello")


if __name__ == "__main__":
    unittest.main()
