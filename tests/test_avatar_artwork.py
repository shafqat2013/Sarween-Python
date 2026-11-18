"""Exposed artwork can be rejected; uncertain and physical overlaps are retained."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from avatar_artwork import AvatarArtworkMatcher, MAX_COMPARISONS


def portrait(seed=1, *, neutral=False):
    """Procedural textured art, independent of the saved purple-avatar clip."""
    rng = np.random.default_rng(seed)
    if neutral:
        values = rng.integers(20, 235, (48, 48), dtype=np.uint8)
        bgr = np.repeat(values[..., None], 3, axis=2)
    else:
        bgr = rng.integers(20, 235, (48, 48, 3), dtype=np.uint8)
    return np.dstack((bgr, np.full((48, 48), 255, np.uint8)))


def icon():
    image = np.zeros((64, 64, 4), np.uint8)
    ink = (190, 30, 190, 255)
    cv2.ellipse(image, (32, 22), (17, 20), 0, 0, 360, ink, -1)
    cv2.fillPoly(image, [np.array(((28, 37), (10, 61), (57, 57), (36, 36)))], ink)
    cv2.rectangle(image, (21, 15), (26, 22), (0, 0, 0, 0), -1)
    cv2.rectangle(image, (37, 17), (42, 24), (0, 0, 0, 0), -1)
    cv2.line(image, (25, 31), (41, 29), (0, 0, 0, 0), 3)
    return image


def board():
    return np.full((220, 320, 3), (40, 40, 40), np.uint8)


def draw(image, art, center=(80, 80), size=32):
    rgba = cv2.resize(art, (size, size), interpolation=cv2.INTER_AREA)
    x, y = round(center[0] - size / 2), round(center[1] - size / 2)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.
    region = image[y:y + size, x:x + size]
    region[:] = np.rint(rgba[..., :3] * alpha + region * (1 - alpha)).astype(np.uint8)
    return image


def candidate(center=(80, 80), radius=16):
    return {"cx": center[0], "cy": center[1], "radius": radius,
            "confidence": .9, "ring_verified": True, "compatible_minis": ["piece"]}


def catalog(art=None, **changes):
    return {"sceneId": "scene-a", "catalogId": "art-a", "revision": 1,
            "templates": [{"templateId": "portrait", "image_bgra": portrait() if art is None else art,
                           "board_width": 32., "board_height": 32.}], **changes}


class ArtworkMatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def matcher(self, art=None):
        matcher = AvatarArtworkMatcher()
        self.assertTrue(matcher.set_catalog(catalog(art)))
        return matcher

    def test_exact_exposed_patterns_and_bounded_scale_offset_are_rejected(self):
        for art in (portrait(), icon(), portrait(neutral=True)):
            for size in (29, 32, 35):
                with self.subTest(size=size, neutral=np.array_equal(art[..., 0], art[..., 1])):
                    matcher = self.matcher(art)
                    image = draw(board(), art, center=(82, 79), size=size)
                    values = [candidate(radius=size / 2)]
                    before = deepcopy(values)
                    kept, result = matcher.filter_candidates(image, values)
                    self.assertEqual(kept, [])
                    self.assertEqual(values, before)
                    self.assertEqual(result["rejected"][0]["reason"], "strong_exposed_artwork_match")

    def test_brightness_change_preserves_a_complete_pattern(self):
        matcher = self.matcher()
        image = draw(board(), portrait())
        image = np.rint(image.astype(float) * .6 + 10).astype(np.uint8)
        kept, result = matcher.filter_candidates(image, [candidate()])
        self.assertEqual(kept, [])
        self.assertGreater(result["rejected"][0]["score"], .98)

    def test_bounded_optical_blur_matches_without_relaxing_occlusion_checks(self):
        for art in (portrait(), icon()):
            matcher = self.matcher(art)
            clear = draw(board(), art)
            image = cv2.GaussianBlur(clear, (0, 0), .6)
            kept, result = matcher.filter_candidates(image, [candidate()])
            self.assertEqual(kept, [])
            self.assertTrue(result["rejected"])
            cv2.rectangle(clear, (73, 72), (87, 88), (150, 150, 150), -1)
            image = cv2.GaussianBlur(clear, (0, 0), .6)
            self.assertEqual(matcher.filter_candidates(image, [candidate()])[0], [candidate()])

    def test_neutral_template_cannot_explain_a_colored_lookalike(self):
        art = portrait(neutral=True)
        matcher = self.matcher(art)
        image = draw(board(), art)
        patch_pixels = image[64:96, 64:96].astype(np.int16)
        patch_pixels[..., 0] += 35
        patch_pixels[..., 2] += 35
        image[64:96, 64:96] = np.clip(patch_pixels, 0, 255).astype(np.uint8)
        self.assertEqual(matcher.filter_candidates(image, [candidate()])[0], [candidate()])

    def test_partial_or_central_occlusion_does_not_reject_the_physical_candidate(self):
        for rect in (((73, 72), (87, 88)), ((64, 64), (70, 95)), ((64, 64), (95, 70))):
            with self.subTest(rect=rect):
                matcher = self.matcher()
                image = draw(board(), portrait())
                cv2.rectangle(image, *rect, (150, 150, 150), -1)
                values = [candidate()]
                kept, result = matcher.filter_candidates(image, values)
                self.assertEqual(kept, values)
                self.assertFalse(result["rejected"])

    def test_avatar_visible_inside_a_larger_ring_cannot_explain_the_ring(self):
        matcher = self.matcher()
        image = draw(board(), portrait())
        cv2.circle(image, (80, 80), 24, (180, 30, 180), 4)
        values = [candidate(radius=24)]
        kept, result = matcher.filter_candidates(image, values)
        self.assertEqual(kept, values)
        self.assertFalse(result["rejected"])

    def test_other_art_and_same_color_rings_are_not_artwork_matches(self):
        matcher = self.matcher(icon())
        for color in ((190, 30, 190), (20, 180, 240)):
            image = board()
            cv2.circle(image, (80, 80), 16, color, 4)
            cv2.rectangle(image, (76, 68), (84, 85), (140, 140, 140), -1)
            self.assertEqual(matcher.filter_candidates(image, [candidate()])[0], [candidate()])
        image = draw(board(), portrait(20))
        self.assertEqual(matcher.filter_candidates(image, [candidate()])[0], [candidate()])

    def test_plain_solid_and_ring_only_catalogs_do_not_establish_authority(self):
        for hole in (None, 10):
            art = np.zeros((48, 48, 4), np.uint8)
            cv2.circle(art, (24, 24), 18, (190, 30, 190, 255), -1)
            if hole is not None:
                cv2.circle(art, (24, 24), hole, (0, 0, 0, 0), -1)
            matcher = AvatarArtworkMatcher()
            self.assertFalse(matcher.set_catalog(catalog(art)))
            self.assertEqual(matcher.filter_candidates(draw(board(), art), [candidate()])[0], [candidate()])
        plain = np.full((48, 48, 4), (190, 30, 190, 255), np.uint8)
        self.assertFalse(AvatarArtworkMatcher().set_catalog(catalog(plain)))

    def test_masked_or_clipped_artwork_remains_unknown(self):
        matcher = self.matcher()
        image = draw(board(), portrait())
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[80, 80] = 255
        kept, result = matcher.filter_candidates(image, [candidate()], mask)
        self.assertEqual(kept, [candidate()])
        self.assertFalse(result["rejected"])
        self.assertEqual(matcher.filter_candidates(image, [candidate(center=(2, 2))])[0], [candidate(center=(2, 2))])

    def test_unknown_catalog_and_missing_geometry_preserve_candidates(self):
        image, values = draw(board(), portrait()), [candidate()]
        matcher = AvatarArtworkMatcher()
        self.assertEqual(matcher.filter_candidates(image, values)[0], values)
        matcher.set_catalog(catalog())
        self.assertFalse(matcher.set_catalog(catalog(sceneId=None)))
        self.assertEqual(matcher.filter_candidates(image, values)[0], values)
        missing = catalog()
        del missing["templates"][0]["board_width"]
        self.assertFalse(matcher.set_catalog(missing))
        self.assertEqual(matcher.filter_candidates(image, values)[0], values)

    def test_catalog_changes_clear_old_patterns_and_copies_own_image_data(self):
        art = portrait()
        matcher = self.matcher(art)
        old_frame = draw(board(), art)
        art[:] = 0
        self.assertEqual(matcher.filter_candidates(old_frame, [candidate()])[0], [])
        matcher.set_catalog(catalog(portrait(20), catalogId="new-art", revision=2))
        self.assertEqual(matcher.filter_candidates(old_frame, [candidate()])[0], [candidate()])
        self.assertEqual(matcher.catalog_id, "new-art")

    def test_unchanged_catalog_reuses_compiled_features(self):
        matcher = AvatarArtworkMatcher()
        snapshot = catalog()
        with patch.object(matcher, "_compile", wraps=matcher._compile) as compile:
            self.assertTrue(matcher.set_catalog(snapshot))
            self.assertTrue(matcher.set_catalog(snapshot))
            self.assertEqual(compile.call_count, 1)

    def test_corrupted_image_cannot_reuse_authority_from_the_same_catalog_key(self):
        snapshot = catalog()
        matcher = self.matcher()
        snapshot["templates"][0]["image_bgra"] = None
        self.assertFalse(matcher.set_catalog(snapshot))
        self.assertEqual(matcher.filter_candidates(draw(board(), portrait()), [candidate()])[0], [candidate()])

    def test_changed_content_or_geometry_fingerprint_replaces_cached_artwork(self):
        for field in ("fingerprint", "contentSha256"):
            with self.subTest(field=field):
                matcher = AvatarArtworkMatcher()
                first = catalog(**{field: "first"})
                self.assertTrue(matcher.set_catalog(first))
                self.assertEqual(matcher.filter_candidates(draw(board(), portrait()), [candidate()])[0], [])
                # IDs, revision and expected dimensions are unchanged. The
                # decoded content/geometry fingerprint still revokes old art.
                second = catalog(portrait(20), **{field: "second"})
                self.assertTrue(matcher.set_catalog(second))
                self.assertEqual(matcher.filter_candidates(draw(board(), portrait()), [candidate()])[0], [candidate()])

    def test_comparison_budget_preserves_unexamined_candidates(self):
        snapshot = catalog()
        snapshot["templates"] = [{"templateId": f"art-{i}", "image_bgra": portrait(i + 10),
                                   "board_width": 32, "board_height": 32} for i in range(32)]
        matcher = self.matcher()
        matcher.set_catalog(snapshot)
        values = [candidate((x, 80)) for x in (40, 80, 120, 160, 200, 240)]
        kept, result = matcher.filter_candidates(board(), values)
        self.assertEqual(kept, values)
        self.assertLessEqual(result["comparisons"], MAX_COMPARISONS)
        self.assertTrue(result["budget_exhausted"])


if __name__ == "__main__":
    unittest.main()
