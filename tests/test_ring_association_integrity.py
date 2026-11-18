"""Physical appearance constrains presence without assigning identity by color."""

import unittest

import numpy as np

from poc_scene_reference import SceneReference
from single_mover_state import SingleMover


def ring(x, y, compatible, **values):
    return {"cx": x, "cy": y, "radius": 18, "confidence": .9,
            "uncertainty_px": 1.5, "compatible_minis": compatible, **values}


class RingAssociationIntegrityTests(unittest.TestCase):
    def make(self):
        return SingleMover({"yellow": (30, 30), "purple": (100, 30)}, 25)

    def test_wrong_appearance_at_old_position_cannot_hide_departure_or_block_arrival(self):
        mover = self.make()
        frame = [ring(30, 30, ["purple"], radius=13),
                 ring(100, 30, ["purple"]), ring(65, 85, ["yellow"])]
        self.assertEqual(mover.update(frame, 0), [])
        self.assertEqual(mover.status["missing"], ["yellow"])
        move, = mover.update(frame, .3)
        self.assertEqual((move["mini"], move["to_xy"]), ("yellow", (65, 85)))
        # The nonmatching avatar remains unexplained, so callers must not
        # silently absorb it into a settled reference after the real move.
        self.assertEqual(mover.status["unassigned"], [(30, 30)])
        self.assertEqual(mover.positions["purple"], (100, 30))

    def test_wrong_appearance_destination_never_establishes_a_move(self):
        mover = self.make()
        wrong = [ring(100, 30, ["purple"]), ring(65, 85, ["purple"])]
        for timestamp in (0, .2, .4):
            self.assertEqual(mover.update(wrong, timestamp), [])
            self.assertIsNone(mover.status["pending"])
        correct = [ring(100, 30, ["purple"]), ring(65, 85, ["yellow"])]
        self.assertEqual(mover.update(correct, .5), [])
        move, = mover.update(correct, .7)
        self.assertEqual(move["first_visible_time"], .5)

    def test_appearance_cannot_choose_identity_when_two_origins_are_unresolved(self):
        mover = self.make()
        for timestamp in (0, .3, .6):
            self.assertEqual(mover.update([ring(65, 85, ["yellow"])], timestamp), [])
        self.assertEqual(mover.status["unresolved_departures"], ["purple", "yellow"])
        self.assertEqual(mover.positions, {"yellow": (30, 30), "purple": (100, 30)})

    def test_anonymous_cache_cannot_reuse_wrong_appearance_settling_time(self):
        mover = self.make()
        self.assertEqual(mover.update([ring(65, 85, ["purple"])], 0), [])
        self.assertEqual(mover.update([ring(65, 85, ["purple"])], .2), [])
        correct = [ring(100, 30, ["purple"]), ring(65, 85, ["yellow"])]
        self.assertEqual(mover.update(correct, .3), [])
        move, = mover.update(correct, .5)
        self.assertEqual(move["first_visible_time"], .3)

    def test_duplicate_fits_preserve_supported_compatibility_and_one_occupancy(self):
        mover = self.make()
        frame = [ring(30, 30, ["purple"], confidence=.95),
                 ring(30.5, 30, ["yellow"], confidence=.8),
                 ring(100, 30, ["purple"])]
        self.assertEqual(mover.update(frame, 0), [])
        self.assertEqual(mover.status["reason"], "layout_confirmed")
        self.assertEqual(mover.status["missing"], [])
        self.assertEqual(mover.status["unassigned"], [])

    def test_empty_compatibility_is_not_physical_evidence_and_malformed_is_rejected(self):
        mover = self.make()
        self.assertEqual(mover.update([ring(30, 30, []), ring(100, 30, ["purple"])], 0), [])
        self.assertEqual(mover.status["missing"], ["yellow"])
        before = dict(mover.status)
        for compatible in (None, "yellow", [""], [1], {"yellow": True}):
            with self.subTest(compatible=compatible), self.assertRaises(ValueError):
                mover.update([ring(30, 30, compatible)], .1)
        self.assertEqual(mover.status, before)

    def test_unexplained_extra_ring_is_uncertain_even_when_every_origin_is_visible(self):
        mover = self.make()
        frame = [ring(30, 30, ["yellow"]), ring(100, 30, ["purple"]),
                 ring(65, 85, ["yellow"])]
        for timestamp in (0, .3):
            self.assertEqual(mover.update(frame, timestamp), [])
        self.assertEqual(mover.status["phase"], "uncertain")
        self.assertEqual(mover.status["reason"], "extra_candidates_with_full_layout")
        self.assertEqual(mover.status["unassigned"], [(65, 85)])

    def test_reference_anchor_shortcut_requires_matching_physical_appearance(self):
        image = np.full((150, 180, 3), 30, np.uint8)
        reference = SceneReference(25)
        reference.update(image)
        avatar = ring(30, 30, ["purple"])
        self.assertEqual(reference.filter(image, [avatar], {"yellow": (30, 30)}), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "unchanged_from_settled_reference")
        valid = ring(30, 30, ["yellow"])
        self.assertEqual(reference.filter(image, [valid], {"yellow": (30, 30)}), [valid])
        self.assertEqual(reference.last_evidence[0]["reason"], "confirmed_anchor")
        empty = ring(30, 30, [])
        self.assertEqual(reference.filter(image, [empty], {"yellow": (30, 30)}), [])
        self.assertEqual(reference.last_evidence[0]["reason"], "incompatible_ring_appearance")


if __name__ == "__main__":
    unittest.main()
