"""Actual receiver carries source identity and bounded optical diagnostics."""
import unittest

import cv2
import numpy as np

from phone_camera import PhoneCameraCapture
from tests.test_phone_camera_contract import Wire, Transport, packet, hello, frame, wait_for


class PhoneCameraMetadataTests(unittest.TestCase):
    def test_receiver_rejects_lens_change_before_decode_or_ack(self):
        from unittest.mock import patch
        from phone_camera_protocol import read_packet
        header, jpeg = read_packet(Wire(frame(1, 20)))
        header["lens"] = "ultra-wide"
        wire = Wire(hello() + packet(header, jpeg))
        with patch("phone_camera.cv2.imdecode") as decode:
            capture = PhoneCameraCapture("selected-phone", transport=Transport(wire), read_timeout=.03)
            self.addCleanup(capture.release)
            wait_for(lambda: capture.status["code"] == "invalid_stream")
            self.assertEqual(capture.read(), (False, None))
            decode.assert_not_called()
            self.assertNotIn("ack", [read_packet(Wire(sent))[0]["type"] for sent in wire.sent])

    def test_hello_identity_and_frame_telemetry_survive_decode_without_opaque_header(self):
        hello = {"type": "hello", "protocolVersion": 1, "sessionId": "metadata-session",
            "payloadBytes": 0, "deviceName": "Fixture iPhone", "sourceId": "fixed-wide-gdc-off",
            "captureFormat": {"width": 32, "height": 24, "fps": 15, "pixelFormat": "jpeg",
                              "lens": "wide", "geometricDistortionCorrection": "off"}}
        ok, encoded = cv2.imencode(".jpg", np.zeros((24, 32, 3), np.uint8))
        self.assertTrue(ok)
        jpeg = encoded.tobytes()
        frame = {"type": "frame", "protocolVersion": 1, "sessionId": "metadata-session",
            "payloadBytes": len(jpeg), "seq": 1, "captureTimeNs": 20, "rotationDegrees": 0,
            "mirrored": False, "width": 32, "height": 24,
            "torchOn": True, "exposureLocked": True, "whiteBalanceLocked": True,
            "torchLevel": .75, "lensPosition": .4, "iso": 120.5,
            "exposureDurationNs": 30_000_000, "whiteBalanceGains": [1.5, 1., 1.8],
            "opaqueUnknownData": {"not": "a supported measurement"}}
        wire = Wire(packet(hello) + packet(frame, jpeg))
        capture = PhoneCameraCapture("selected-phone", transport=Transport(wire), read_timeout=.3)
        self.addCleanup(capture.release)
        self.assertTrue(capture.read()[0])
        metadata = capture.last_frame_metadata
        self.assertEqual(metadata["sourceId"], "fixed-wide-gdc-off")
        self.assertEqual(capture.status["sourceId"], metadata["sourceId"])
        for key in ("torchOn", "exposureLocked", "whiteBalanceLocked", "torchLevel", "lensPosition", "iso", "exposureDurationNs", "whiteBalanceGains"):
            self.assertEqual(metadata[key], frame[key])
        self.assertNotIn("opaqueUnknownData", metadata)
        metadata["whiteBalanceGains"][0] = 999
        self.assertEqual(capture.last_frame_metadata["whiteBalanceGains"], [1.5, 1., 1.8])


if __name__ == "__main__":
    unittest.main()
