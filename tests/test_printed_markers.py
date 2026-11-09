"""Registration contracts using generated planar detections, not a webcam."""
from copy import deepcopy
import unittest

import cv2
import numpy as np

import printed_markers as pm


def view(width=1920, height=1080, margin=28):
    return {"sceneId": "scene", "viewportWidth": width, "viewportHeight": height,
            "registration": {"left": margin, "top": margin,
                             "right": width - margin, "bottom": height - margin}}


def layout():
    centers = [(-.04, -.035), (1.04, -.035), (1.04, 1.035), (-.04, 1.035)]
    points = {}
    for marker_id, center, angle in zip(pm.PRINTED_IDS.values(), centers, (0, 90, 30, 180)):
        radians = np.deg2rad(angle)
        rotation = np.array([[np.cos(radians), -np.sin(radians)],
                             [np.sin(radians), np.cos(radians)]])
        physical = np.array([[-28, -28], [28, -28], [28, 28], [-28, 28]]) @ rotation.T
        normalized = physical / (1920, 1080) + center
        points[str(marker_id)] = normalized.tolist()
    return pm.validate_layout({"schemaVersion": 1, "dictionary": pm.DICTIONARY,
        "markerIds": dict(pm.PRINTED_IDS), "markerCorners": points,
        "viewportAspectRatio": 1920 / 1080})


def camera_transform(shift=(0, 0)):
    target = np.array([[300, 180], [1500, 120], [1600, 850], [240, 900]], np.float32)
    target += shift
    return cv2.getPerspectiveTransform(np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32), target)


def detected(paper=None, *, include_screen=False, included=(20, 21, 22, 23), H=None):
    paper = paper or layout()
    H = camera_transform() if H is None else H
    by_id = {int(k): v for k, v in paper["markerCorners"].items() if int(k) in included}
    if include_screen:
        guides = pm.guide_aligned_layout(view(), 120, 8)["markerCorners"]
        by_id.update({k - 10: guides[str(k)] for k in (20, 21, 22, 23)})
    ids = np.array([[k] for k in by_id], np.int32)
    corners = [cv2.perspectiveTransform(np.array(v, np.float64).reshape(1, 4, 2), H).astype(np.float32)
               for v in by_id.values()]
    return corners, ids


class PrintedGeometryTests(unittest.TestCase):
    def test_covision_recovers_all_corners_with_bezel_offsets_and_rotated_prints(self):
        expected = layout()
        corners, ids = detected(expected, include_screen=True)
        result = pm.calibrate_layout(corners, ids, view())
        self.assertEqual(result["calibrationMethod"], "co-visible")
        for key in expected["markerCorners"]:
            np.testing.assert_allclose(result["markerCorners"][key], expected["markerCorners"][key], atol=2e-6)

    def test_guide_alignment_needs_paper_only_and_is_explicitly_declared(self):
        expected = pm.guide_aligned_layout(view(), 120, 8)
        self.assertEqual(expected["calibrationMethod"], "guide-aligned-user-declared")
        self.assertAlmostEqual(expected["markerCorners"]["20"][0][0], 28 / 1920)
        self.assertAlmostEqual(expected["markerCorners"]["20"][1][0], 108 / 1920)
        corners, ids = detected(expected)
        self.assertIsNotNone(pm.solve_homography(corners, ids, expected, view(), (1280, 720)))
        with self.assertRaisesRegex(ValueError, "screen markers"):
            pm.calibrate_layout(corners, ids, view())
        with self.assertRaisesRegex(ValueError, "do not match"):
            pm.guide_aligned_layout(view(), 160, 8)

    def test_independent_white_margin_keeps_guide_pattern_width_unchanged(self):
        for margin in (0, 6, 12):
            current = view(margin=8 + margin)
            expected = pm.guide_aligned_layout(current, inset=8, pattern_size=48, margin=margin)
            first = np.asarray(expected["markerCorners"]["20"]) * (1920, 1080)
            np.testing.assert_allclose(first[0], (8 + margin, 8 + margin))
            np.testing.assert_allclose(first[1] - first[0], (48, 0))
            corners, ids = detected(expected)
            self.assertIsNotNone(pm.solve_homography(corners, ids, expected, current, (1280, 720)))
        # A stale view must never silently reinterpret changed white padding.
        with self.assertRaisesRegex(ValueError, "do not match"):
            pm.guide_aligned_layout(view(margin=14), inset=8, pattern_size=48, margin=12)
        smaller = pm.guide_aligned_layout(view(margin=18), 60, 8)
        self.assertAlmostEqual((smaller["markerCorners"]["20"][1][0] - smaller["markerCorners"]["20"][0][0]) * 1920, 40)

    def test_perspective_fit_maps_actual_lcd_instead_of_paper_hull(self):
        saved = layout()
        corners, ids = detected(saved)
        H = pm.solve_homography(corners, ids, saved, view(), (1280, 720))
        normalized = np.array([[[.25, .35], [.5, .5], [.9, .8]]], np.float64)
        source = cv2.perspectiveTransform(normalized, camera_transform())
        projected = cv2.perspectiveTransform(source, H)
        expected = normalized.copy()
        expected[..., 0] = (expected[..., 0] * 1920 - 28) / (1920 - 56) * 1279
        expected[..., 1] = (expected[..., 1] * 1080 - 28) / (1080 - 56) * 719
        np.testing.assert_allclose(projected, expected, atol=.01)
        camera_mask, warp_mask = pm.roi_masks((1080, 1920, 3), H, (1280, 720))
        paper_center = np.mean(corners[0][0], axis=0).astype(int)
        self.assertEqual(camera_mask[paper_center[1], paper_center[0]], 0)
        self.assertEqual(warp_mask.shape, (720, 1280))
        self.assertTrue(np.all(warp_mask == 255))

    def test_proportional_viewport_resize_uses_current_virtual_registration(self):
        saved = layout()
        corners, ids = detected(saved)
        target_view = view(1280, 720, margin=20)
        H = pm.solve_homography(corners, ids, saved, target_view, (1280, 720))
        point = np.array([[[.2, .3]]], np.float64)
        source = cv2.perspectiveTransform(point, camera_transform())
        expected = [(.2 * 1280 - 20) / 1240 * 1279, (.3 * 720 - 20) / 680 * 719]
        np.testing.assert_allclose(cv2.perspectiveTransform(source, H)[0, 0], expected, atol=.01)
        with self.assertRaisesRegex(ValueError, "aspect changed"):
            pm.solve_homography(corners, ids, saved, view(1024, 768), (1280, 720))

    def test_three_good_cards_hold_but_two_or_inconsistent_cards_do_not(self):
        saved = layout()
        all_corners, all_ids = detected(saved)
        H = pm.solve_homography(all_corners, all_ids, saved, view(), (1280, 720))
        for count in (2, 3):
            corners, ids = detected(saved, included=tuple(range(20, 20 + count)))
            self.assertIsNone(pm.solve_homography(corners, ids, saved, view(), (1280, 720)))
            self.assertEqual(pm.agrees_with_homography(corners, ids, H, saved, view(), (1280, 720)), count == 3)
        moved = [p.copy() for p in all_corners]
        moved[-1] += (25, 0)
        self.assertIsNone(pm.solve_homography(moved, all_ids, saved, view(), (1280, 720)))
        self.assertFalse(pm.agrees_with_homography(moved, all_ids, H, saved, view(), (1280, 720)))

    def test_camera_motion_needs_four_cards_to_refresh(self):
        saved = layout()
        c, ids = detected(saved)
        H = pm.solve_homography(c, ids, saved, view(), (1280, 720))
        moved_H = camera_transform((30, -15))
        partial, partial_ids = detected(saved, included=(20, 21, 22), H=moved_H)
        self.assertFalse(pm.agrees_with_homography(partial, partial_ids, H, saved, view(), (1280, 720)))
        all_corners, all_ids = detected(saved, H=moved_H)
        refreshed = pm.solve_homography(all_corners, all_ids, saved, view(), (1280, 720))
        self.assertIsNotNone(refreshed)
        self.assertTrue(pm.agrees_with_homography(all_corners, all_ids, refreshed, saved, view(), (1280, 720)))

    def test_duplicate_ids_and_one_bad_corner_are_rejected(self):
        saved = layout()
        c, ids = detected(saved)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            pm.solve_homography([*c, c[0]], np.vstack([ids, ids[0]]), saved, view(), (1280, 720))
        c[0] = c[0].copy()
        c[0][0, 0] += (20, 0)
        self.assertIsNone(pm.solve_homography(c, ids, saved, view(), (1280, 720)))

    def test_layout_validation_is_detached_and_rejects_degenerate_or_bad_inputs(self):
        original = layout()
        canonical = pm.validate_layout(original)
        canonical["markerCorners"]["20"][0][0] = 99
        self.assertNotEqual(canonical, original)
        for change in ({"dictionary": "wrong"}, {"schemaVersion": True}, {"viewportAspectRatio": float("nan")},
                       {"calibrationMethod": "verified-by-guessing"}):
            bad = deepcopy(original)
            bad.update(change)
            with self.assertRaises(ValueError):
                pm.validate_layout(bad)
        bad = deepcopy(original)
        bad["markerCorners"]["20"] = [[0, 0]] * 4
        with self.assertRaises(ValueError):
            pm.validate_layout(bad)


if __name__ == "__main__":
    unittest.main()
