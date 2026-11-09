"""Temporal reflection safety with real image detection at two physical scales.

These synthetic static-map sequences exercise association, not a claim that
real glare is solved. All geometry is relative to grid spacing; no recording
coordinates, injected candidates or per-recording thresholds are used.
"""
from copy import deepcopy
import unittest

import cv2
import numpy as np

from hybrid_ring_detector import HybridRingDetector
from observer_engine import BoardGeometry, ObserverEngine, CONTINUOUS_DISPLAY_POLICY


class ReflectionScene:
    def __init__(self, grid):
        self.grid = int(grid)
        self.width, self.height = 10 * self.grid, 9 * self.grid
        self.positions = {f"mini{index}": self.point(x, y) for index, (x, y) in enumerate(
            ((2, 2), (5, 2), (8, 2), (2, 7), (5, 7), (8, 7)), 1)}
        self.map_circle = self.point(8, 4)

    def point(self, x, y):
        return (float(x * self.grid), float(y * self.grid))

    def ring(self, image, center):
        center = tuple(round(value) for value in center)
        cv2.circle(image, center, round(.75 * self.grid), (35, 75, 180), -1)
        cv2.circle(image, center, round(.60 * self.grid), (160, 160, 160), -1)
        cv2.line(image, center, (round(center[0] - 1.075 * self.grid), center[1]),
                 (160, 160, 160), max(1, round(.20 * self.grid)))

    def image(self, positions=None, *, artifacts=False, reflection=None, exposure=1.0,
              hide_origin=False):
        image = np.full((self.height, self.width, 3), (150, 90, 40), np.uint8)
        if artifacts:
            # A static printed/map circle is deliberately visually identical to
            # a base. Only temporal scene evidence can reject it as a new mini.
            self.ring(image, self.map_circle)
            center = tuple(round(value) for value in self.point(5, 4.5))
            cv2.circle(image, center, round(.85 * self.grid), (90, 145, 210), -1)
            cv2.circle(image, center, round(.70 * self.grid), (255, 255, 255), -1)
        for center in (self.positions if positions is None else positions).values():
            self.ring(image, center)
        image = np.clip(image.astype(np.float32) * exposure, 0, 255)
        if reflection is not None:
            yy, xx = np.mgrid[:self.height, :self.width]
            cx, cy = self.point(*reflection)
            light = 220 * np.exp(-.5 * (((xx - cx) / (.65 * self.grid)) ** 2
                                      + ((yy - cy) / (.90 * self.grid)) ** 2))
            image = np.minimum(255, image + light[..., None])
        image = image.astype(np.uint8)
        if hide_origin:
            cv2.circle(image, tuple(round(value) for value in self.positions["mini1"]),
                       round(1.15 * self.grid), (255, 255, 255), -1)
        return image

    def origin_mask(self):
        mask = np.zeros((self.height, self.width), np.uint8)
        cv2.circle(mask, tuple(round(value) for value in self.positions["mini1"]),
                   round(1.15 * self.grid), 255, -1)
        return mask

    def engine(self):
        view = {"sceneId": "reflection-fixture",
                "registration": {"left": 0, "top": 0, "right": self.width, "bottom": self.height},
                "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
                "gridSize": self.grid, "gridOriginX": 0, "gridOriginY": 0}
        return ObserverEngine(BoardGeometry(view, (self.width, self.height)), self.positions,
            detector_factory=HybridRingDetector, display_transition_policy=CONTINUOUS_DISPLAY_POLICY)


class TemporalReflectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def acquire(self, fixture, engine, *, artifacts=False):
        for now in (0, .125, .25):
            row = engine.process(fixture.image(artifacts=artifacts), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"], row["state"])
        self.assertEqual(row["acquisition"]["visible_count"], 6, row["acquisition"])
        self.assertFalse(row["held"], row["state"])
        self.assertEqual(row["events"], [])
        return row

    def assert_positions_near(self, actual, expected, grid):
        self.assertEqual(set(actual), set(expected))
        for mini, point in expected.items():
            self.assertLess(np.linalg.norm(np.asarray(actual[mini]) - point), .15 * grid, mini)

    def test_stationary_six_with_soft_moving_reflection_and_exposure_emit_no_moves(self):
        for grid in (40, 60):
            with self.subTest(grid=grid):
                fixture = ReflectionScene(grid)
                engine = fixture.engine()
                self.acquire(fixture, engine)
                for index, (point, exposure) in enumerate(zip(
                        ((2, 2), (3, 2.2), (4, 3.5), (5, 4.5), (6, 3.5), (8, 2)),
                        (.96, 1.04, 1.08, 1.0, .92, 1.04))):
                    pixels = fixture.image(reflection=point, exposure=exposure)
                    row = engine.process(pixels, None, .375 + index * .125, scene_visual_revision=0)
                    self.assertEqual(row["events"], [], row["state"])
                for now in (1.25, 1.375, 1.5, 1.625):
                    row = engine.process(fixture.image(), None, now, scene_visual_revision=0)
                self.assertEqual(engine.events, [])
                self.assert_positions_near(engine.mover.positions, fixture.positions, grid)
                self.assertEqual(row["acquisition"]["visible_count"], 6)
                self.assertFalse(row["held"], row["state"])

    def test_glare_hiding_one_origin_cannot_jump_to_existing_map_circle_or_white_highlight(self):
        for grid in (40, 60):
            with self.subTest(grid=grid):
                fixture = ReflectionScene(grid)
                engine = fixture.engine()
                initial = self.acquire(fixture, engine, artifacts=True)
                self.assertGreaterEqual(initial["raw_candidate_count"], 7,
                    "The map-circle distractor must reach the real detector before association rejects it")
                for now in (.375, .5, .625, .75, .875):
                    row = engine.process(fixture.image(artifacts=True, hide_origin=True), None, now,
                                         scene_visual_revision=0)
                    self.assertEqual(row["events"], [], row["state"])
                    self.assertEqual(engine.mover.positions, fixture.positions)
                self.assertIn("mini1", row["held"], row["state"])
                for now in (1., 1.125, 1.25, 1.375):
                    row = engine.process(fixture.image(artifacts=True), None, now, scene_visual_revision=0)
                self.assertEqual(engine.events, [])
                self.assert_positions_near(engine.mover.positions, fixture.positions, grid)
                self.assertEqual(row["acquisition"]["visible_count"], 6)
                self.assertFalse(row["held"], row["state"])

    def test_real_relocation_during_glare_mask_waits_for_clear_then_confirms_only_that_identity(self):
        for grid in (40, 60):
            with self.subTest(grid=grid):
                fixture = ReflectionScene(grid)
                engine = fixture.engine()
                self.acquire(fixture, engine)
                moved = deepcopy(fixture.positions)
                moved["mini1"] = fixture.point(2, 4)
                for now in (.375, .5, .625, .75):
                    pixels = fixture.image(moved, hide_origin=True)
                    row = engine.process(pixels, fixture.origin_mask(), now, scene_visual_revision=0)
                    self.assertEqual(row["events"], [], row["state"])
                    self.assertEqual(engine.mover.positions, fixture.positions)
                    self.assertIn("mini1", row["unavailable"])
                first_clear = .875
                for now in (first_clear, 1., 1.125, 1.25, 1.375):
                    row = engine.process(fixture.image(moved), None, now, scene_visual_revision=0)
                self.assertEqual([event["mini"] for event in engine.events], ["mini1"])
                event, = engine.events
                self.assertGreaterEqual(event["first_visible_time"], first_clear)
                self.assertGreaterEqual(event["confirmed_time"], first_clear + engine.mover.settle_seconds)
                self.assert_positions_near(engine.mover.positions, moved, grid)
                self.assertEqual(row["acquisition"]["visible_count"], 6)
                self.assertFalse(row["held"], row["state"])


if __name__ == "__main__":
    unittest.main()
