"""Opt-in duplex guidance cannot refresh pixels, leak sessions or block UI."""
import json
import time
import unittest
import uuid

from phone_camera import PhoneCameraCapture
from phone_camera_protocol import ProtocolError, read_packet
from tests.test_phone_camera_contract import Wire, Transport, Clock, frame, packet, wait_for


def hello(session="stream-one", capable=True):
    return packet({"type": "hello", "protocolVersion": 1, "sessionId": session, "deviceName": "Fixture",
        "payloadBytes": 0, "hostFeedback": capable,
        "captureFormat": {"width": 32, "height": 24, "fps": 15, "pixelFormat": "jpeg"}})


def request(action="calibration.capture", *, request_id=None, session="stream-one"):
    return packet({"type": "request", "protocolVersion": 1, "sessionId": session, "payloadBytes": 0,
        "action": action, "requestId": request_id or str(uuid.uuid4())})


def commands(wire):
    return [read_packet(Wire(raw))[0] for raw in wire.sent]


class FeedbackTransportTests(unittest.TestCase):
    def capture(self, *wires, **kwargs):
        cap = PhoneCameraCapture("selected-phone", transport=Transport(*wires), read_timeout=.1,
            reconnect_delays=(.01,), **kwargs)
        self.addCleanup(cap.release)
        return cap


    def test_requests_are_bounded_deduplicated_and_never_make_old_pixels_fresh(self):
        clock = Clock()
        wire = Wire(hello() + frame(1, 900))
        cap = self.capture(wire, clock=clock)
        self.assertTrue(cap.read()[0])
        clock.value = 102.
        identifier = str(uuid.uuid4())
        wire.push(request(request_id=identifier) + request(request_id=identifier))
        wait_for(lambda: len(cap._requests) == 1)
        wire.push(b"".join(request() for _ in range(20)))
        wait_for(lambda: len(cap._request_ids) == 21)
        rows = cap.take_requests()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({row["requestId"] for row in rows}), 16)
        self.assertTrue(all(row["connectionGeneration"] == 1 for row in rows))
        self.assertEqual(cap.take_requests(), [])
        self.assertFalse(cap.status["fresh"])
        self.assertEqual(cap.last_frame_metadata["seq"], 1)
        self.assertEqual(cap.read(), (False, None))

    def test_feedback_latest_slot_and_write_time_freshness_guard(self):
        clock = Clock()
        wire = Wire(hello() + frame(1, 900))
        cap = self.capture(wire, clock=clock)
        self.assertTrue(cap.read()[0])
        cap._feedback_sent_at = time.monotonic()
        self.assertTrue(cap.send_feedback({"kind": "preview", "sourceSeq": 1,
                                           "cpl": {"state": "reference-needed", "message": "first"}}))
        self.assertTrue(cap.send_feedback({"kind": "preview", "sourceSeq": 1,
                                           "cpl": {"state": "reference-needed", "message": "latest"}}))
        self.assertEqual(cap._pending_feedback[1]["cpl"]["message"], "latest")
        self.assertFalse(cap.send_feedback({"kind": "preview", "sourceSeq": 999}))
        with self.assertRaises(ProtocolError):
            cap.send_feedback({"kind": "preview", "cpl": {"index": float("inf")}})
        clock.value = 102.
        cap.release()  # a queued old4/4 must not acquire a fresh phone expiry
        self.assertFalse(any(item["type"] == "feedback" for item in commands(wire)))



if __name__ == "__main__": unittest.main()
