"""Packet boundaries and hostile/malformed phone data, with no devices."""
import json
import socket
import struct
import threading
import time
import unittest

import cv2
import numpy as np

from phone_camera_protocol import (
    MAX_HEADER_BYTES, MAX_JPEG_BYTES, CameraCancelled, ProtocolError,
    encode_packet, jpeg_dimensions, read_packet, validate_frame, validate_frame_optics, validate_hello,
)


class BytesSocket:
    def __init__(self, data, fragment=3):
        self.data = bytearray(data)
        self.fragment = fragment

    def recv(self, size):
        count = min(size, self.fragment, len(self.data))
        value = bytes(self.data[:count])
        del self.data[:count]
        return value


def hello():
    return {"type": "hello", "protocolVersion": 1, "sessionId": "a",
            "deviceName": "My iPhone", "payloadBytes": 0,
            "captureFormat": {"width": 32, "height": 24, "fps": 15, "pixelFormat": "jpeg"}}


def frame(size=10):
    return {"type": "frame", "protocolVersion": 1, "sessionId": "a", "seq": 1,
            "captureTimeNs": 123, "width": 32, "height": 24,
            "rotationDegrees": 0, "mirrored": False, "payloadBytes": size}


class CameraProtocolTests(unittest.TestCase):
    def test_fragmented_and_coalesced_packets_remain_distinct(self):
        sock = BytesSocket(encode_packet(hello()) + encode_packet(frame(5), b"abcde"), fragment=1)
        self.assertEqual(read_packet(sock), (hello(), b""))
        self.assertEqual(read_packet(sock), (frame(5), b"abcde"))

    def test_short_header_payload_and_eof_raise_without_partial_packet(self):
        packet = encode_packet(frame(5), b"abcde")
        for cut in (0, 2, 8, len(packet) - 1):
            with self.subTest(cut=cut), self.assertRaises(EOFError):
                read_packet(BytesSocket(packet[:cut]))

    def test_limits_and_json_are_validated_before_payload_allocation(self):
        cases = [struct.pack(">I", MAX_HEADER_BYTES + 1), struct.pack(">I", 0)]
        for raw in (b"[]", b"{", b'\xff', b'{"type":"hello","type":"frame"}',
                    b'{"type":[],"protocolVersion":1}', b'{"value":1e999}'):
            cases.append(struct.pack(">I", len(raw)) + raw)
        bad = frame(MAX_JPEG_BYTES + 1)
        raw = json.dumps(bad).encode()
        cases.append(struct.pack(">I", len(raw)) + raw)
        for case in cases:
            with self.subTest(case=case[:30]), self.assertRaises(ProtocolError):
                read_packet(BytesSocket(case))

    def test_no_boolean_counts_wrong_version_or_command_payload(self):
        for changed in ({"payloadBytes": True}, {"protocolVersion": True},
                        {"protocolVersion": 2}, {"payloadBytes": 1}, {"sessionId": ""}):
            with self.subTest(changed=changed), self.assertRaises(ProtocolError):
                encode_packet({**hello(), **changed})
        with self.assertRaises(ProtocolError):
            encode_packet(frame(2), b"a")

    def test_cancel_and_deadline_work_with_silent_socket(self):
        class Silent:
            def recv(self, _):
                raise socket.timeout()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(CameraCancelled):
            read_packet(Silent(), cancel_event=cancelled)
        with self.assertRaises(TimeoutError):
            read_packet(Silent(), deadline=time.monotonic() + .005)

    def test_hello_and_frame_schema_checks(self):
        self.assertEqual(validate_hello(hello())["width"], 32)
        self.assertEqual(validate_frame(frame(), session_id="a"), (1, 123))
        for changed in ({"seq": 0}, {"captureTimeNs": 1}, {"sessionId": "b"},
                        {"mirrored": True}, {"rotationDegrees": 90}, {"width": 9000}):
            with self.subTest(changed=changed), self.assertRaises(ProtocolError):
                validate_frame({**frame(), **changed}, session_id="a", previous_seq=0, previous_capture_ns=100)
        for fmt in ({"fps": float("nan")}, {"height": 0}, {"pixelFormat": "h264"}):
            value = hello()
            value["captureFormat"].update(fmt)
            with self.subTest(fmt=fmt), self.assertRaises(ProtocolError):
                validate_hello(value)
        with self.assertRaises(ProtocolError):
            validate_hello({**hello(), "sourceId": {"not": "a string"}})

    def test_jpeg_dimensions_are_checked_without_decoding(self):
        ok, jpeg = cv2.imencode(".jpg", np.zeros((24, 32, 3), np.uint8))
        self.assertTrue(ok)
        self.assertEqual(jpeg_dimensions(jpeg.tobytes()), (32, 24))
        for bad in (b"bad", b"\xff\xd8\xff\xd9", b"\xff\xd8\xff\xe0\xff\xff"):
            with self.subTest(bad=bad), self.assertRaises(ProtocolError):
                jpeg_dimensions(bad)

    def test_physical_lens_and_native_field_of_view_are_bounded(self):
        for lens in ("wide", "ultra-wide", "telephoto"):
            value = hello()
            value["captureFormat"].update(lens=lens, nativeFieldOfViewDegrees=119.5)
            self.assertEqual(validate_hello(value)["lens"], lens)
        for changed in ({"lens": "triple"}, {"lens": []}, {"nativeFieldOfViewDegrees": True},
                        {"nativeFieldOfViewDegrees": float("nan")}, {"nativeFieldOfViewDegrees": 180}):
            value = hello()
            value["captureFormat"].update(changed)
            with self.subTest(changed=changed), self.assertRaises(ProtocolError):
                validate_hello(value)

    def test_same_resolution_lens_change_cannot_bypass_new_handshake(self):
        fmt = {"lens": "wide", "zoom": 1.0, "nativeFieldOfViewDegrees": 64.0}
        validate_frame_optics({}, fmt)  # Original v1 frames lacked redundant optical fields.
        validate_frame_optics(dict(fmt), fmt)
        for changed in ({"lens": "ultra-wide"}, {"zoom": 2}, {"nativeFieldOfViewDegrees": 120.0}):
            with self.subTest(changed=changed), self.assertRaises(ProtocolError):
                validate_frame_optics(changed, fmt)


if __name__ == "__main__":
    unittest.main()
