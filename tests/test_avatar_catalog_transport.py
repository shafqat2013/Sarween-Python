"""Bounded optional artwork assets cannot outlive their scene/connection."""
import asyncio
import base64
import copy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import avatar_catalog as ac
import foundryoutput as fo
from tests import test_vtt_bridge as bridge_fixture


def view(scene="scene-a", tx=0):
    return {"sceneId": scene, "viewportWidth": 1000, "viewportHeight": 800,
            "registration": {"left": 100, "top": 100, "right": 900, "bottom": 700},
            "canvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": tx, "ty": 0},
            "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": -tx, "ty": 0},
            "gridOriginX": 0, "gridOriginY": 0, "gridSize": 50}


def png(width=20, height=16):
    pixels = np.zeros((height, width, 4), np.uint8)
    pixels[2:-2, 3:-3] = (180, 20, 190, 255)
    encoded, image = cv2.imencode(".png", pixels)
    assert encoded
    return "data:image/png;base64," + base64.b64encode(image).decode("ascii")


def payload(**overrides):
    return {"type": "avatarCatalog", "schemaVersion": 1, "sceneId": "scene-a",
            "catalogId": "catalog-a", "revision": 1, "method": ac.CATALOG_METHOD,
            "status": "available", "viewTransform": view(), "capturedAtEpochMs": 1,
            "templates": [{"templateId": "avatar-1", "image": png(), "width": 20, "height": 16,
                           "logicalWidth": 50, "logicalHeight": 40, "tokenIds": ["magenta"]}],
            "unsupportedTokens": [], **overrides}


class AvatarCatalogValidationTests(unittest.TestCase):
    def test_validation_copies_and_hashes_without_decoding(self):
        source = payload()
        with patch.object(ac.cv2, "imdecode", side_effect=AssertionError("pixel decode too soon")):
            catalog = ac.validate_avatar_catalog(source, scene_id="scene-a")
        self.assertEqual(len(catalog["contentSha256"]), 64)
        self.assertEqual(ac.validate_avatar_catalog(catalog), catalog)
        source["templates"][0]["tokenIds"].append("changed")
        self.assertEqual(catalog["templates"][0]["tokenIds"], ["magenta"])
        tampered = copy.deepcopy(catalog)
        tampered["templates"][0]["logicalWidth"] = 51
        with self.assertRaisesRegex(ValueError, "digest"):
            ac.validate_avatar_catalog(tampered)

    def test_envelope_and_numeric_limits(self):
        bad = [payload(schemaVersion=True), payload(revision=False), payload(revision=0),
               payload(sceneId=""), payload(catalogId="x" * 257), payload(method="viewport-crop"),
               payload(status=[]), payload(capturedAtEpochMs=float("nan")),
               payload(templates=[payload()["templates"][0]] * 33),
               payload(extra="x" * ac.MAX_CATALOG_BYTES)]
        for field, value in (("width", True), ("width", 97), ("height", 0),
                             ("logicalWidth", 0), ("logicalWidth", 8193),
                             ("logicalHeight", float("inf")), ("logicalWidth", 10**1000),
                             ("tokenIds", ["x"] * 65), ("templateId", "bad\nidentifier")):
            candidate = payload()
            candidate["templates"][0][field] = value
            bad.append(candidate)
        for candidate in bad:
            with self.subTest(keys=list(candidate)), self.assertRaises(ValueError):
                ac.validate_avatar_catalog(candidate)
        with self.assertRaisesRegex(ValueError, "another scene"):
            ac.validate_avatar_catalog(payload(), scene_id="scene-b")

    def test_bad_png_headers_rejected_before_decode(self):
        invalid = ["data:image/jpeg;base64,AAAA", "data:image/png;base64,!!!!",
                   "data:image/png;base64," + "A" * ac.MAX_IMAGE_CHARACTERS,
                   png(97, 16), payload()["templates"][0]["image"][:-12]]
        for image in invalid:
            candidate = payload()
            candidate["templates"][0]["image"] = image
            with self.subTest(image=image[:35]), patch.object(ac.cv2, "imdecode") as decode:
                self.assertIsNone(ac.decode_avatar_catalog(candidate, view(), (800, 600)))
                decode.assert_not_called()

    def test_duplicate_ids_and_invalid_unsupported_reasons_rejected(self):
        candidate = payload()
        candidate["templates"] *= 2
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            ac.validate_avatar_catalog(candidate)
        for rows in ([{"tokenId": "x", "reason": "private/path.png"}], [None], "bad"):
            with self.assertRaises(ValueError):
                ac.validate_avatar_catalog(payload(unsupportedTokens=rows))

    def test_empty_reset_snapshots_allow_null_geometry(self):
        for status in ("unsupported", "unavailable"):
            candidate = payload(status=status, templates=[], viewTransform=None)
            self.assertEqual(ac.validate_avatar_catalog(candidate)["status"], status)
            self.assertIsNone(ac.decode_avatar_catalog(candidate, view(), (800, 600)))
            with self.assertRaises(ValueError):
                ac.validate_avatar_catalog(payload(status=status))

    def test_decodes_alpha_and_maps_expected_size_without_age_expiry(self):
        candidate = payload()
        decoded = ac.decode_avatar_catalog(candidate, view(), (1600, 900))
        self.assertIsNotNone(decoded)
        template = decoded["templates"][0]
        self.assertEqual(template["image_bgra"].shape, (16, 20, 4))
        self.assertEqual(template["image_bgra"].dtype, np.uint8)
        self.assertEqual(int(template["image_bgra"][0, 0, 3]), 0)
        self.assertEqual(int(template["image_bgra"][5, 5, 3]), 255)
        self.assertFalse(template["image_bgra"].flags.writeable)
        self.assertEqual((template["board_width"], template["board_height"]), (100, 60))
        self.assertEqual(decoded["fingerprint"], ac.decode_avatar_catalog(candidate, view(), (1600, 900))["fingerprint"])
        self.assertNotEqual(decoded["fingerprint"], ac.decode_avatar_catalog(candidate, view(), (800, 600))["fingerprint"])

    def test_mismatched_geometry_disables_optional_filter(self):
        for current in (None, view(scene="scene-b"), view(tx=1)):
            self.assertIsNone(ac.decode_avatar_catalog(payload(), current, (800, 600)))
        for warp in ((0, 600), (800.0, 600), (800, True), (800, 600, 4), None):
            self.assertIsNone(ac.decode_avatar_catalog(payload(), view(), warp))
        candidate = payload()
        candidate["viewTransform"]["registration"]["left"] = float("inf")
        self.assertIsNone(ac.decode_avatar_catalog(candidate, view(), (800, 600)))


class AvatarCatalogConnectionTests(unittest.IsolatedAsyncioTestCase):
    until = bridge_fixture.VTTBridgeTests.until

    async def asyncSetUp(self):
        await bridge_fixture.VTTBridgeTests.asyncSetUp(self)
        extra = {"_avatar_catalog_state": {}, "_connection_generation": 0,
                 "_scene_info_generation": None, "_connection_scene_info": None}
        for name, value in extra.items():
            item = patch.object(fo, name, value)
            item.start()
            self.patches.append(item)

    async def asyncTearDown(self):
        await bridge_fixture.VTTBridgeTests.asyncTearDown(self)

    async def connect(self, supported=True):
        socket = bridge_fixture.Socket()
        task = asyncio.create_task(fo.handler(socket))
        self.handlers.append(task)
        await socket.incoming.put({"type": "hello", "protocolVersion": 2, "avatarCatalog": supported})
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "scene-a", "width": 1000,
                                   "height": 800, "gridSize": 50, "gridType": 1})
        await self.until(lambda: fo._connection_ready)
        if supported:
            await self.until(lambda: any(message["type"] == "avatarCatalogGet" for message in socket.sent))
        return socket, task

    async def test_automatic_request_and_incoming_snapshot_are_copied(self):
        socket, _ = await self.connect()
        self.assertEqual([m for m in socket.sent if m["type"] == "avatarCatalogGet"],
                         [{"type": "avatarCatalogGet", "sceneId": "scene-a"}])
        await socket.incoming.put(payload())
        await self.until(lambda: fo.get_latest_avatar_catalog() is not None)
        snapshot = fo.get_latest_avatar_catalog()
        snapshot["templates"][0]["tokenIds"].append("changed")
        self.assertEqual(fo.get_latest_avatar_catalog()["templates"][0]["tokenIds"], ["magenta"])

    async def test_unsupported_old_module_leaves_relay_ready(self):
        socket, _ = await self.connect(supported=False)
        self.assertFalse(fo.record_avatar_catalog(payload()))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertTrue(fo._connection_ready)
        self.assertFalse(any(message["type"] == "avatarCatalogGet" for message in socket.sent))

    async def test_revisions_conflicts_and_empty_status_clear_artwork_only(self):
        await self.connect()
        self.assertTrue(fo.record_avatar_catalog(payload(revision=3)))
        self.assertFalse(fo.record_avatar_catalog(payload(revision=2)))
        self.assertFalse(fo.record_avatar_catalog(payload(revision=2, method="bad-old-message")))
        self.assertEqual(fo.get_latest_avatar_catalog()["revision"], 3)
        self.assertFalse(fo.record_avatar_catalog(payload(revision=3, capturedAtEpochMs=2)))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload(revision=3)))
        self.assertTrue(fo.record_avatar_catalog(payload(revision=4)))
        self.assertTrue(fo.record_avatar_catalog(payload(revision=5, status="unsupported", templates=[], viewTransform=None)))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload(revision=4)))
        self.assertTrue(fo._connection_ready)

    async def test_bad_current_snapshot_clears_but_other_scene_does_not_replace(self):
        await self.connect()
        self.assertTrue(fo.record_avatar_catalog(payload()))
        self.assertFalse(fo.record_avatar_catalog(payload(sceneId="scene-b")))
        self.assertIsNotNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload(method="unsafe-crop")))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload()))
        self.assertTrue(fo.record_avatar_catalog(payload(revision=2)))
        tampered = fo.get_latest_avatar_catalog()
        tampered["templates"][0]["logicalWidth"] = 51
        self.assertFalse(fo.record_avatar_catalog(tampered))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload(revision=2)))
        self.assertTrue(fo._connection_ready)

    async def test_scene_disconnect_reconnect_and_old_hello_clear(self):
        socket, task = await self.connect()
        self.assertTrue(fo.record_avatar_catalog(payload()))
        fo.set_scene_params(scene_id="scene-b")
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertFalse(fo.record_avatar_catalog(payload(sceneId="scene-b", viewTransform=view(scene="scene-b"))))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.assertIsNone(fo.get_latest_avatar_catalog())
        second, _ = await self.connect()
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertTrue(fo.record_avatar_catalog(payload(catalogId="new-connection")))
        await second.incoming.put({"type": "hello", "protocolVersion": 2})
        await self.until(lambda: not fo._connection_ready)
        self.assertIsNone(fo.get_latest_avatar_catalog())

    async def test_invalid_scene_info_and_validation_race_cannot_resurrect(self):
        socket, _ = await self.connect()
        original = fo.validate_avatar_catalog

        def change_scene_while_validating(source, **kwargs):
            catalog = original(source, **kwargs)
            fo._reset_avatar_catalog()
            return catalog

        with patch.object(fo, "validate_avatar_catalog", side_effect=change_scene_while_validating):
            self.assertFalse(fo.record_avatar_catalog(payload()))
        self.assertIsNone(fo.get_latest_avatar_catalog())
        self.assertTrue(fo.record_avatar_catalog(payload()))
        await socket.incoming.put({"type": "sceneInfo", "sceneId": "scene-a", "error": "not ready"})
        await self.until(lambda: fo._scene_info_generation is None)
        self.assertIsNone(fo.get_latest_avatar_catalog())


if __name__ == "__main__":
    unittest.main()
