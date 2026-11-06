"""Static-scene change evidence, independent of expected moves and identities."""

import unittest

import cv2
import numpy as np

from poc_scene_reference import SceneReference


def image():
    return np.full((160, 220, 3), (115, 95, 45), dtype=np.uint8)


def candidate(x=150, y=80, radius=16):
    return {"cx": x, "cy": y, "radius": radius, "confidence": .8,
            "uncertainty_px": 2.5}


def physical_disk(frame, x=150, y=80):
    cv2.circle(frame, (x, y), 16, (25, 45, 210), -1)


class SceneReferenceTests(unittest.TestCase):
    def test_unchanged_map_circle_is_not_new_base_evidence(self):
        frame = image()
        physical_disk(frame)
        reference = SceneReference(25)
        reference.update(frame)
        self.assertEqual(reference.filter(frame, [candidate()], {"old": (40, 80)}), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "unchanged_from_settled_reference")
        self.assertEqual(reference.last_evidence[0]["changed_fraction"], 0)

    def test_new_base_is_supported_by_local_change(self):
        before = image()
        after = before.copy()
        physical_disk(after)
        reference = SceneReference(25)
        reference.update(before)
        row = candidate()
        self.assertEqual(reference.filter(after, [row], {"old": (40, 80)}), [row])
        self.assertEqual(reference.last_evidence[0]["reason"], "changed_from_settled_reference")

    def test_confirmed_anchor_passes_without_requiring_pixel_change(self):
        frame = image()
        row = candidate(40, 80)
        reference = SceneReference(25)
        self.assertEqual(reference.filter(frame, [row], {"stationary": (42, 80)}), [row])
        self.assertEqual(reference.last_evidence[0]["reason"], "confirmed_anchor")
        reference.update(frame)
        self.assertEqual(reference.filter(frame, [row], {"stationary": (42, 80)}), [row])

    def test_new_positions_wait_for_an_explicit_reference(self):
        reference = SceneReference(25)
        self.assertFalse(reference.has_reference)
        self.assertEqual(reference.filter(image(), [candidate()], {"old": (40, 80)}), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "reference_unavailable")
        self.assertEqual(reference.reference_updates, 0)

    def test_current_caption_is_unavailable_not_movement_evidence(self):
        before = image()
        after = before.copy()
        mask = np.zeros(before.shape[:2], np.uint8)
        cv2.rectangle(after, (142, 60), (159, 100), (255, 255, 255), -1)
        cv2.rectangle(mask, (142, 60), (159, 100), 255, -1)
        reference = SceneReference(25)
        reference.update(before)
        self.assertEqual(reference.filter(after, [candidate()], {"old": (40, 80)}, mask), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "insufficient_unmasked_support")

    def test_reference_caption_disappearing_is_not_movement_evidence(self):
        after = image()
        before = after.copy()
        mask = np.zeros(before.shape[:2], np.uint8)
        cv2.rectangle(before, (142, 60), (159, 100), (255, 255, 255), -1)
        cv2.rectangle(mask, (142, 60), (159, 100), 255, -1)
        reference = SceneReference(25)
        reference.update(before, mask)
        self.assertEqual(reference.filter(after, [candidate()], {"old": (40, 80)}), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "insufficient_unmasked_support")

    def test_both_caption_masks_contribute_to_available_fraction(self):
        before = image()
        after = before.copy()
        physical_disk(after)
        old_mask = np.zeros(before.shape[:2], np.uint8)
        new_mask = np.zeros_like(old_mask)
        old_mask[:, :150] = 255
        new_mask[:, 150:] = 255
        reference = SceneReference(25)
        reference.update(before, old_mask)
        self.assertEqual(reference.filter(after, [candidate()], {"old": (40, 80)}, new_mask), [])
        self.assertEqual(reference.last_evidence[0]["valid_fraction"], 0)

    def test_partial_mask_leaves_valid_changed_pixels_usable(self):
        before = image()
        after = before.copy()
        physical_disk(after)
        mask = np.zeros(before.shape[:2], np.uint8)
        mask[:, :144] = 255
        reference = SceneReference(25)
        reference.update(before)
        row = candidate()
        self.assertEqual(reference.filter(after, [row], {"old": (40, 80)}, mask), [row])
        self.assertGreater(reference.last_evidence[0]["valid_fraction"], .6)

    def test_tiny_change_does_not_validate_whole_candidate(self):
        before = image()
        after = before.copy()
        cv2.circle(after, (150, 80), 2, (0, 0, 255), -1)
        reference = SceneReference(25)
        reference.update(before)
        self.assertEqual(reference.filter(after, [candidate()], {"old": (40, 80)}), [])
        self.assertLess(reference.last_evidence[0]["changed_fraction"], .2)

    def test_small_camera_noise_is_below_change_threshold(self):
        before = image()
        after = (before.astype(np.int16) + 20).astype(np.uint8)
        reference = SceneReference(25)
        reference.update(before)
        self.assertEqual(reference.filter(after, [candidate()], {"old": (40, 80)}), [])

    def test_reference_update_allows_return_to_vacated_position(self):
        original = image()
        physical_disk(original, 40, 80)
        moved = image()
        physical_disk(moved, 150, 80)
        reference = SceneReference(25)
        reference.update(original)
        self.assertEqual(len(reference.filter(moved, [candidate()], {"mini": (40, 80)})), 1)
        reference.update(moved)
        returned = reference.filter(original, [candidate(40, 80)], {"mini": (150, 80)})
        self.assertEqual(len(returned), 1)
        self.assertEqual(reference.reference_updates, 2)

    def test_filter_never_learns_candidate_frames_or_mutates_inputs(self):
        before = image()
        after = before.copy()
        physical_disk(after)
        snapshot = after.copy()
        rows = [candidate()]
        row_copy = dict(rows[0])
        reference = SceneReference(25)
        reference.update(before)
        self.assertEqual(reference.filter(after, rows, {"old": (40, 80)}), rows)
        self.assertEqual(reference.filter(after, rows, {"old": (40, 80)}), rows)
        self.assertEqual(reference.reference_updates, 1)
        np.testing.assert_array_equal(after, snapshot)
        self.assertEqual(rows[0], row_copy)

    def test_update_copies_past_image_and_mask(self):
        before = image()
        after = before.copy()
        physical_disk(after)
        mask = np.zeros(before.shape[:2], np.uint8)
        reference = SceneReference(25)
        reference.update(before, mask)
        before[:] = after
        mask[:] = 255
        self.assertEqual(len(reference.filter(after, [candidate()], {"old": (40, 80)})), 1)

    def test_partial_frame_is_not_full_valid_disk_support(self):
        before = image()
        after = before.copy()
        physical_disk(after, 0, 80)
        reference = SceneReference(25)
        reference.update(before)
        self.assertEqual(reference.filter(after, [candidate(0, 80)], {"old": (40, 40)}), [])
        self.assertLess(reference.last_evidence[0]["valid_fraction"], .6)

    def test_geometry_and_malformed_measurements_are_rejected(self):
        for grid in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                SceneReference(grid)
        reference = SceneReference(25)
        with self.assertRaises(ValueError):
            reference.update(np.zeros((20, 20), np.uint8))
        with self.assertRaises(ValueError):
            reference.update(image(), np.zeros((3, 3), np.uint8))
        reference.update(image())
        with self.assertRaises(ValueError):
            reference.filter(np.zeros((200, 200, 3), np.uint8), [], {})
        for row in ({"cx": 150, "cy": 80}, candidate(radius=-1), candidate(x=float("inf"))):
            with self.assertRaises(ValueError):
                reference.filter(image(), [row], {})
        with self.assertRaises(ValueError):
            reference.filter(image(), [], {"bad": (0, float("nan"))})


if __name__ == "__main__":
    unittest.main()
