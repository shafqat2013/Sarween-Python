"""Fresh same-image marker/CPL guidance with explicit comparable conditions."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from phone_camera_feedback import PhoneCameraFeedback
from phone_camera_protocol import ProtocolError, validate_feedback
from printed_markers import guide_aligned_layout


class Clock:
    def __init__(self): self.now = 100.
    def __call__(self): return self.now


class Capture:
    def __init__(self):
        self.status = {"sessionId": "one", "connectionGeneration": 1, "hostFeedback": True}
        self.sent, self.requests = [], []
    def send_feedback(self, value): self.sent.append(deepcopy(value)); return True
    def take_requests(self):
        value, self.requests = self.requests, []
        return value


def image(level=120, *, printed=False, missing=None):
    frame = np.full((600, 1000, 3), level, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    ids = (20, 21, 22, 23) if printed else (10, 11, 12, 13)
    for marker, (x, y) in zip(ids, ((28, 28), (892, 28), (892, 492), (28, 492))):
        if marker == missing: continue
        frame[y-20:y+100, x-20:x+100] = 255
        frame[y:y+80, x:x+80] = cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary, marker, 80), cv2.COLOR_GRAY2BGR)
    # Full-board default ArUco detection is ambiguous at the smaller fixture's
    # quiet-tile/background contours. Use clear 160px generated markers; no
    # production detector settings are changed to accommodate the fixture.
    return cv2.resize(frame, (2000, 1200), interpolation=cv2.INTER_NEAREST)


VIEW = {"sceneId": "scene", "viewportWidth": 2000, "viewportHeight": 1200,
        "registration": {"left": 56, "top": 56, "right": 1944, "bottom": 1144}}


class PhoneFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.cap = Clock(), Capture()
        self.helper = PhoneCameraFeedback(self.cap, clock=self.clock)
        self.seq = 0

    def metadata(self, **changes):
        self.seq += 1
        return {"sessionId": "one", "connectionGeneration": 1, "deviceId": "phone", "seq": self.seq,
            "hostReceivedMonotonic": self.clock(), "exposureLocked": True, "iso": 400.,
            "exposureDurationNs": 20_000_000, "whiteBalanceGains": [1.5, 1., 2.], "torchOn": False,
            "torchLevel": 0, "format": {"width": 2000, "height": 1200, "lens": "wide"}, **changes}

    def publish(self, level=120, *, frame=None, helper=None, meta=None, **changes):
        self.clock.now += .6
        return (helper or self.helper).publish(image(level) if frame is None else frame,
            self.metadata() if meta is None else meta, view_key=VIEW, scene_revision=1, **changes)

    def request(self, action, helper=None, **changes):
        self.cap.requests.append({"action": action, "requestId": "fixture", "sessionId": "one",
            "connectionGeneration": 1, "hostReceivedMonotonic": self.clock(), **changes})
        return (helper or self.helper).take_requests()

    def test_counts_are_actual_same_pixels_and_stale_or_blank_frames_cannot_keep_four(self):
        row = self.publish()
        self.assertEqual(row["markers"]["ids"], [10, 11, 12, 13])
        self.assertEqual(row["markers"]["count"], 4)
        self.assertNotIn("locked", row["markers"])
        self.assertIsNone(self.helper.publish(image(), self.metadata(), view_key=VIEW, scene_revision=1))
        row = self.publish(frame=np.full((1200, 2000, 3), 120, np.uint8), locked=True)
        self.assertEqual(row["markers"]["count"], 0)
        self.assertTrue(row["markers"]["locked"], "Registered lock is separate from current visibility")
        stale = self.helper.publish(None, None, locked=True)
        self.assertEqual(stale["markers"]["count"], 0)
        self.assertFalse(stale["markers"]["locked"])
        self.assertNotIn("sourceSeq", stale)

    def test_reference_target_index_is_explicit_relative_lcd_brightness(self):
        self.publish()
        self.assertIsNone(self.helper._reference)
        self.request("cpl.reference")
        row = self.publish(60)
        self.assertEqual(row["cpl"]["index"], 50.)
        self.assertIn("Not an angle or optical transmission", row["cpl"]["message"])
        self.request("cpl.target")
        row = self.publish(62)
        self.assertEqual(row["cpl"]["state"], "target-matched")
        self.assertEqual(row["cpl"]["target"], 50.)
        self.request("cpl.reset")
        self.assertEqual(self.publish()["cpl"]["state"], "reference-needed")

    def test_exposure_scene_torch_view_and_missing_markers_invalidate_index(self):
        self.publish(); self.request("cpl.reference")
        for change in ({"exposureLocked": False}, {"iso": 600.}, {"torchOn": True},
                       {"format": {"width": 2000, "height": 1200, "lens": "ultra-wide"}},
                       {"whiteBalanceGains": [2., 1., 2.]}):
            with self.subTest(change=change):
                self.clock.now += .6
                row = self.helper.publish(image(60), self.metadata(**change), view_key=VIEW, scene_revision=1)
                self.assertNotIn("index", row["cpl"])
        self.clock.now += .6
        row = self.helper.publish(image(60), self.metadata(), view_key=VIEW, scene_revision=2)
        self.assertNotIn("index", row["cpl"])
        self.clock.now += .6
        view = {**VIEW, "registration": {**VIEW["registration"], "left": 64}}
        row = self.helper.publish(image(60), self.metadata(), view_key=view, scene_revision=1)
        self.assertNotIn("index", row["cpl"])
        row = self.publish(frame=image(60, missing=11))
        self.assertEqual(row["markers"]["count"], 3)
        self.assertNotIn("index", row["cpl"])

    def test_projective_phone_jitter_keeps_the_registered_lcd_brightness_comparable(self):
        frame = image(120)
        self.publish(frame=frame)
        self.request("cpl.reference")
        source = np.float32([[0, 0], [1999, 0], [1999, 1199], [0, 1199]])
        destination = np.float32([[35, 25], [1950, 15], [1935, 1170], [65, 1180]])
        transform = cv2.getPerspectiveTransform(source, destination)
        moved = cv2.warpPerspective(frame, transform, (2000, 1200), borderValue=(210, 210, 210))
        row = self.publish(frame=moved)
        self.assertEqual(row["markers"]["count"], 4)
        self.assertAlmostEqual(row["cpl"]["index"], 100., delta=.5)
        # Dim the actual image, including the marker whites. Re-registering a
        # changed camera view must preserve the brightness drop, not reset100.
        dimmed = (frame.astype(np.float32) * .5).astype(np.uint8)
        moved_dimmed = cv2.warpPerspective(dimmed, transform, (2000, 1200), borderValue=(210, 210, 210))
        row = self.publish(frame=moved_dimmed)
        self.assertEqual(row["markers"]["count"], 4)
        self.assertAlmostEqual(row["cpl"]["index"], 50., delta=.5)

    def test_context_unavailable_unknown_exposure_and_expired_request_never_create_reference(self):
        self.clock.now += .6
        row = self.helper.publish(image(), self.metadata(), view_key=None, scene_revision=None)
        self.assertIn("Connect Foundry", row["cpl"]["message"])
        self.request("cpl.reference")
        self.assertIsNone(self.helper._reference)
        self.publish(); self.clock.now += 2
        self.request("cpl.reference")
        self.assertIsNone(self.helper._reference)
        self.publish(); self.request("cpl.reference", sessionId="old")
        self.assertIsNone(self.helper._reference)
        self.request("cpl.reference")
        self.assertIsNotNone(self.helper._reference)
        self.request("cpl.reset", connectionGeneration=999)
        self.assertIsNotNone(self.helper._reference)

    def test_printed_counter_and_fresh_registered_lcd_solve_use_active_paper_ids(self):
        helper = PhoneCameraFeedback(self.cap, marker_ids=(20, 21, 22, 23), clock=self.clock)
        registration = {"mode": "printed", "printedMarkerLayout": guide_aligned_layout(VIEW, 240, 16)}
        row = self.publish(frame=image(120, printed=True), helper=helper, registration=registration, locked=True)
        self.assertEqual(row["markers"]["ids"], [20, 21, 22, 23])
        self.assertEqual(row["cpl"]["state"], "reference-needed")
        self.request("cpl.reference", helper)
        row = self.publish(frame=image(60, printed=True), helper=helper, registration=registration, locked=True)
        self.assertEqual(row["cpl"]["index"], 50.)
        row = self.publish(frame=image(60, printed=True, missing=21), helper=helper, registration=registration, locked=True)
        self.assertEqual(row["markers"]["count"], 3)
        self.assertNotIn("index", row["cpl"])

    def test_calibration_never_misreads_charuco_ids_as_tv_lock_and_supports_no_pixels(self):
        def forbidden(_): raise AssertionError("TV detector used on calibration board")
        helper = PhoneCameraFeedback(self.cap, kind="calibration", detector=forbidden, clock=self.clock)
        state = {"active": True, "captured": 2, "required": 10, "title": "Move slowly", "message": "Sweep",
                 "reason": "pose", "mode": "sweep", "canFinish": False}
        row = helper.publish(image(), self.metadata(), calibration=state)
        self.assertEqual(row["markers"]["count"], 0)
        self.assertNotIn("index", row["cpl"])
        row = helper.publish(None, None, calibration={**state, "active": False})
        self.assertNotIn("sourceSeq", row)
        self.assertFalse(row["calibration"]["active"])

    def test_only_exact_source_seq_can_reuse_an_existing_same_frame_detection(self):
        corners, ids, rejected = self.helper._detect(image())
        def forbidden(_): raise AssertionError("Redundant ArUco detection")
        helper = PhoneCameraFeedback(self.cap, detector=forbidden, clock=self.clock)
        meta = self.metadata()
        row = helper.publish(image(), meta, view_key=VIEW, scene_revision=1,
            markers={"sourceSeq": meta["seq"], "corners": corners, "ids": ids})
        self.assertEqual(row["markers"]["count"], 4)
        self.clock.now += .6
        with self.assertRaisesRegex(AssertionError, "Redundant"):
            helper.publish(image(), self.metadata(), view_key=VIEW, scene_revision=1,
                markers={"sourceSeq": meta["seq"], "corners": corners, "ids": ids})


class FeedbackSchemaTests(unittest.TestCase):
    def test_bad_nested_values_and_unbounded_messages_are_rejected(self):
        valid = {"kind": "preview", "markers": {"ids": [10], "requiredIds": [10, 11, 12, 13], "count": 1, "required": 4}}
        validate_feedback(valid)
        bad = [{"kind": "preview", "execute": "anything"}, {"kind": "preview", "sourceSeq": True},
               {"kind": "preview", "cpl": {"index": float("nan")}},
               {"kind": "preview", "cpl": {"message": "x" * 301}},
               {**valid, "markers": {**valid["markers"], "count": 4}},
               {**valid, "markers": {**valid["markers"], "ids": [{}]}},
               {"kind": "calibration", "calibration": {"mode": "anything"}}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                validate_feedback(value)


class PreviewRegistrationTests(unittest.TestCase):
    def test_source_and_bundle_preview_use_active_paper_ids_layout_and_explicit_invalid_state(self):
        import foundryoutput as fo
        from tests.test_phone_setup import setup, BUNDLE
        layout = guide_aligned_layout(VIEW, 240, 16)
        for module in (setup, BUNDLE):
            for blocked in (False, True):
                controller = Mock()
                controller.configuration.return_value = {"mode": "screen" if blocked else "printed",
                    "layout": None if blocked else layout, "blocked": blocked, "pending": False,
                    "message": "Register printed markers" if blocked else "", "profileId": "temporary"}
                controller.view_context.return_value = {"ready": True, "view": VIEW}
                with self.subTest(module=module.__name__, blocked=blocked), \
                     patch.object(fo, "get_scene_visual_revision", return_value=7):
                    ids, registration, view, revision = module._phone_preview_feedback_context(controller)
                self.assertEqual(ids, (20, 21, 22, 23))
                self.assertEqual(registration["mode"], "printed")
                self.assertEqual(registration["blocked"], blocked)
                self.assertEqual(registration["printedMarkerLayout"], None if blocked else layout)
                self.assertEqual(view, VIEW)
                self.assertEqual(revision, 7)


if __name__ == "__main__": unittest.main()
