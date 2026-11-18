"""Artwork is a negative observation; physical tracking still owns identity.

Detector proposals are injected to exercise ambiguity which the existing ring
gate already rejects in the saved camera clips. Pixel matching is real here.
These tests establish integration, not live camera accuracy.
"""
import base64
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from avatar_catalog import CATALOG_METHOD, validate_avatar_catalog
from observer_engine import BoardGeometry, ObserverEngine
from tests.test_avatar_artwork import portrait, draw


def view():
    transform = {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0}
    return {"sceneId": "scene-a", "viewportWidth": 320, "viewportHeight": 220,
            "registration": {"left": 0, "top": 0, "right": 320, "bottom": 220},
            "canvasTransform": transform.copy(), "clientToCanvasTransform": transform.copy(),
            "gridSize": 32, "gridOriginX": 0, "gridOriginY": 0}


def wire_catalog():
    art = portrait()
    ok, encoded = cv2.imencode(".png", art)
    assert ok
    return validate_avatar_catalog({"type": "avatarCatalog", "schemaVersion": 1,
        "sceneId": "scene-a", "catalogId": "catalog-a", "revision": 1,
        "method": CATALOG_METHOD, "status": "available", "viewTransform": view(),
        "capturedAtEpochMs": 1, "unsupportedTokens": [],
        "templates": [{"templateId": "portrait", "width": 48, "height": 48,
            "logicalWidth": 32, "logicalHeight": 32, "tokenIds": ["token-a"],
            "image": "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")}]})


def proposal(point):
    return {"cx": point[0], "cy": point[1], "radius": 16, "confidence": .9,
            "uncertainty_px": 1., "ring_verified": True}


def board(*, exposed=False, physical=((80, 80), (240, 80))):
    image = np.full((220, 320, 3), 40, np.uint8)
    if exposed:
        draw(image, portrait(), center=(80, 80), size=32)
    for point in physical:
        cv2.circle(image, point, 16, (190, 40, 190), 4)
        cv2.rectangle(image, (point[0] - 5, point[1] - 10),
                      (point[0] + 5, point[1] + 8), (140, 140, 140), -1)
    return image


class ArtworkIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def setUp(self):
        self.detector = Mock(spec=["detect"])
        self.engine = ObserverEngine(BoardGeometry(view(), (320, 220)),
            {"first": (80, 80), "second": (240, 80)},
            detector_factory=lambda _: self.detector)
        self.catalog = wire_catalog()

    def observe(self, now, points, image=None, **kwargs):
        self.detector.detect.return_value = [proposal(point) for point in points]
        return self.engine.process(board() if image is None else image, None, now,
                                   avatar_catalog=kwargs.pop("avatar_catalog", self.catalog), **kwargs)

    def acquire(self):
        for timestamp in (0., .125, .25):
            result = self.observe(timestamp, [(80, 80), (240, 80)])
        self.assertTrue(result["acquired"])

    def test_catalog_is_compiled_once_and_stationary_frames_do_not_match(self):
        with patch("observer_engine.decode_avatar_catalog", wraps=__import__(
                "avatar_catalog").decode_avatar_catalog) as decode:
            self.acquire()
            self.observe(.375, [(80, 80), (240, 80)])
            self.assertEqual(decode.call_count, 1)
        with patch.object(self.engine.avatar_matcher, "filter_candidates",
                          side_effect=AssertionError("stationary frames need no comparison")):
            result = self.observe(.5, [(80, 80), (240, 80)])
        self.assertFalse(result["avatar_artwork"]["applied"])

    def test_ordinary_single_departure_arrival_needs_no_artwork_comparison(self):
        self.acquire()
        moved = board(physical=((160, 150), (240, 80)))
        with patch.object(self.engine.avatar_matcher, "filter_candidates",
                          side_effect=AssertionError("single mover needs no comparison")):
            for now in (.375, .5, .625, .75):
                result = self.observe(now, [(160, 150), (240, 80)], moved)
        self.assertEqual(len(self.engine.events), 1)
        self.assertEqual(self.engine.events[0]["mini"], "first")
        self.assertFalse(result["avatar_artwork"]["applied"])

    def test_exposed_old_avatar_is_rejected_so_physical_arrival_can_confirm(self):
        self.acquire()
        moved = board(exposed=True, physical=((160, 150), (240, 80)))
        decisions = [self.observe(now, [(80, 80), (240, 80), (160, 150)], moved)
                     for now in (.375, .5, .625, .75)]
        rejected = decisions[0]["avatar_artwork"]["rejected"]
        self.assertTrue(decisions[0]["avatar_artwork"]["applied"])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(self.engine.events), 1)
        self.assertEqual(self.engine.events[0]["mini"], "first")
        self.assertEqual(self.engine.events[0]["to_xy"], (160., 150.))

    def test_no_catalog_preserves_ambiguous_old_avatar_hold(self):
        self.acquire()
        moved = board(exposed=True, physical=((160, 150), (240, 80)))
        for now in (.375, .5, .625, .75):
            result = self.observe(now, [(80, 80), (240, 80), (160, 150)], moved,
                                  avatar_catalog=None)
        self.assertFalse(result["avatar_artwork"]["ready"])
        self.assertEqual(self.engine.events, [])

    def test_overlapping_physical_base_is_retained_during_ambiguity(self):
        self.acquire()
        image = board(exposed=True, physical=((80, 80), (240, 80), (160, 150)))
        result = self.observe(.375, [(80, 80), (240, 80), (160, 150)], image)
        self.assertTrue(result["avatar_artwork"]["applied"])
        self.assertEqual(result["avatar_artwork"]["rejected"], [])
        self.assertEqual(self.engine.events, [])

    def test_missing_multiple_minis_never_uses_artwork_to_assign_identity(self):
        self.acquire()
        image = board(exposed=True, physical=((160, 150), (270, 170)))
        for now in (.375, .5, .625, .75):
            result = self.observe(now, [(80, 80), (160, 150), (270, 170)], image)
        self.assertEqual(self.engine.events, [])
        self.assertEqual(set(result["state"]["missing"]), {"first", "second"})

    def test_changed_geometry_clears_compiled_authority_before_new_snapshot(self):
        self.acquire()
        self.assertTrue(self.engine.avatar_matcher.last_catalog_status["ready"])
        changed = view()
        changed["canvasTransform"]["tx"] = 1
        self.engine.geometry = BoardGeometry(changed, (320, 220))
        result = self.observe(.375, [(80, 80), (240, 80)])
        self.assertFalse(result["avatar_artwork"]["ready"])

    def test_wrong_scene_or_corrupt_catalog_clears_optional_authority(self):
        self.acquire()
        for offset, catalog in enumerate(({"sceneId": "scene-b"}, ["not-a-catalog"])):
            result = self.observe(.375 + offset * .125, [(80, 80), (240, 80)],
                                  avatar_catalog=catalog)
            self.assertFalse(result["avatar_artwork"]["ready"])
            self.assertFalse(result["needs_reset"])
            self.assertEqual(result["events"], [])

    def test_changed_payload_cannot_reuse_a_claimed_content_digest(self):
        self.acquire()
        changed = deepcopy(self.catalog)
        changed["status"] = "unavailable"
        changed["templates"] = []
        result = self.observe(.375, [(80, 80), (240, 80)], avatar_catalog=changed)
        self.assertFalse(result["avatar_artwork"]["ready"])
        self.assertFalse(result["needs_reset"])

    def test_optional_matching_error_passes_candidates_through_without_reset(self):
        self.acquire()
        with patch.object(self.engine.avatar_matcher, "filter_candidates", side_effect=ValueError):
            result = self.observe(.375, [(80, 80), (240, 80), (160, 150)],
                                  board(exposed=True, physical=((160, 150), (240, 80))))
        self.assertFalse(result["needs_reset"])
        self.assertFalse(result["avatar_artwork"]["ready"])
        self.assertEqual(self.engine.events, [])


if __name__ == "__main__":
    unittest.main()
