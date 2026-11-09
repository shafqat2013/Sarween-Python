"""Independent control-channel checks using in-memory camera packets only."""
import unittest
import uuid

from phone_camera import PhoneCameraCapture
from phone_camera_protocol import read_packet
from tests.test_phone_camera_contract import Clock, Transport, Wire, frame, hello, packet, wait_for


def greeting(session="stream-one", *, controls=True):
    header, _ = read_packet(Wire(hello(session)))
    if controls:
        header["hostFeedback"] = True
    return packet(header)


def request(session="stream-one", *, identity=None, action="calibration.capture"):
    return packet({"type": "request", "protocolVersion": 1, "sessionId": session,
        "payloadBytes": 0, "requestId": identity or str(uuid.uuid4()), "action": action})


def commands(wire):
    return [read_packet(Wire(value))[0] for value in list(wire.sent)]


class PhoneControlBoundaryTests(unittest.TestCase):
    def capture(self, *wires, **options):
        cap = PhoneCameraCapture("selected-phone", transport=Transport(*wires),
            read_timeout=.03, packet_timeout=2, reconnect_delays=(.01,), **options)
        self.addCleanup(cap.release)
        return cap

    def test_legacy_phone_never_receives_new_feedback_commands(self):
        wire = Wire(greeting(controls=False) + frame(1, 20))
        cap = self.capture(wire)
        self.assertTrue(cap.read()[0])
        self.assertFalse(cap.send_feedback({"kind": "preview", "sourceSeq": 1}))
        start = next(item for item in commands(wire) if item["type"] == "start")
        self.assertNotIn("hostFeedback", start)
        self.assertNotIn("feedback", [item["type"] for item in commands(wire)])

    def test_request_is_not_fresh_camera_evidence(self):
        clock = Clock()
        wire = Wire(greeting() + frame(1, 20))
        cap = self.capture(wire, clock=clock, stale_timeout=.1)
        self.assertTrue(cap.read()[0])
        clock.value += 1
        identity = str(uuid.uuid4())
        wire.push(request(identity=identity, action="cpl.target"))
        wait_for(lambda: not wire._incoming)
        result = cap.take_requests()
        self.assertEqual([item["requestId"] for item in result], [identity])
        self.assertFalse(cap.status["fresh"])
        self.assertEqual(cap.read(), (False, None))
        self.assertEqual(cap.last_frame_metadata["seq"], 1)
        self.assertEqual(cap.take_requests(), [])

    def test_repeated_button_request_is_delivered_once(self):
        identity = str(uuid.uuid4())
        wire = Wire(greeting() + request(identity=identity) * 3 + frame(1, 20))
        cap = self.capture(wire)
        self.assertTrue(cap.read()[0])
        result = cap.take_requests()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["requestId"], identity)
        self.assertEqual(result[0]["connectionGeneration"], 1)

    def test_reconnect_discards_unconsumed_old_calibration_buttons(self):
        first = Wire(greeting() + request() + frame(1, 20))
        second = Wire(greeting("stream-two") + frame(1, 1, "stream-two"))
        cap = self.capture(first, second)
        self.assertTrue(cap.read()[0])
        first.close()
        wait_for(lambda: cap.status["connectionGeneration"] == 2 and cap.status["fresh"])
        self.assertTrue(cap.read()[0])
        self.assertEqual(cap.take_requests(), [])

    def test_wrong_session_button_cannot_reach_calibration(self):
        wire = Wire(greeting() + frame(1, 20))
        cap = self.capture(wire)
        self.assertTrue(cap.read()[0])
        wire.push(request("another-session"))
        wait_for(lambda: cap.status["code"] == "invalid_stream")
        self.assertEqual(cap.take_requests(), [])
        self.assertFalse(cap.status["fresh"])

    def test_feedback_for_an_old_consumed_frame_is_refused(self):
        wire = Wire(greeting() + frame(1, 20))
        cap = self.capture(wire)
        self.assertTrue(cap.read()[0])
        wire.push(frame(2, 30))
        self.assertTrue(cap.read()[0])
        self.assertFalse(cap.send_feedback({"kind": "preview", "sourceSeq": 1}))
        self.assertTrue(cap.send_feedback({"kind": "preview", "sourceSeq": 2}))


if __name__ == "__main__":
    unittest.main()
