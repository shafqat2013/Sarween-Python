"""Synthetic camera evidence for expected own redraws; no live/fog claim."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from observer_engine import BoardGeometry, ObserverEngine, CONTINUOUS_DISPLAY_POLICY
from observer_journal import ObserverJournal, replay_journal
from observer_session import LiveObserverController
from tests.test_observer_session import FakeWindow


POLICY = "recover-expected-display-changes-v1"
RECENT_POLICY = "recover-expected-display-changes-v2"
POSITIVE_POLICY = "recover-expected-display-changes-v3"
POSITIONS = {f"mini{i + 1}": point for i, point in enumerate(
    ((80, 80), (200, 80), (320, 80), (80, 280), (200, 280), (320, 280)))}
VIEW = {"sceneId": "test-scene", "registration": {"left": 0, "top": 0, "right": 400, "bottom": 360},
        "clientToCanvasTransform": {"a": 1, "b": 0, "c": 0, "d": 1, "tx": 0, "ty": 0},
        "gridSize": 40, "gridOriginX": 0, "gridOriginY": 0}


def image(*, redraw=False, moved=False, patch_x=180):
    pixels = np.full((360, 400, 3), (150, 90, 40), np.uint8)
    if redraw:
        pixels[160:200, patch_x:patch_x + 60] = (210, 210, 210)
    for mini, point in POSITIONS.items():
        center = (80, 160) if moved and mini == "mini1" else point
        cv2.circle(pixels, center, 25, (35, 75, 180), -1)
        cv2.line(pixels, center, (center[0] - 30, center[1]), (35, 75, 180), 7)
    return pixels


def command(identity="move1"):
    return {"command_id": identity, "scene_id": "test-scene", "token_id": "token1"}


def notice(revision=1, commands=("move1",), **overrides):
    return {"revision": revision, "reason": "sarweenTokenMove", "reasons": ["sarweenTokenMove"],
            "sceneId": "test-scene", "tokenIds": ["token1"], "commandIds": list(commands), **overrides}


class ExpectedDisplayRecoveryTests(unittest.TestCase):
    settle_samples = 3

    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(0)

    def engine(self, policy=POLICY):
        return ObserverEngine(BoardGeometry(VIEW, (400, 360)), POSITIONS,
                              display_transition_policy=policy)

    def acquire(self, engine):
        for now in (0, .125, .25):
            row = engine.process(image(), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"])
        self.assertEqual(row["acquisition"]["visible_count"], 6)

    def arm(self, engine):
        return engine.process(image(), None, .375, scene_visual_revision=0,
                              expected_display_commands=[command()])

    def test_ack_and_notification_cannot_release_unchanged_old_camera_pixels(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, .75, 1, 1.25, 2):
            row = engine.process(image(), None, now, scene_visual_revision=1,
                scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertFalse(row["needs_reset"])
            self.assertEqual(row["state"]["reason"], "waiting_for_visible_foundry_redraw")
            self.assertEqual(row["events"], [])
        self.assertFalse(row["display_transition"]["camera_change_observed"])

    def test_visible_settled_redraw_reseeds_all_six_without_rebinding_or_emitting(self):
        engine = self.engine()
        self.acquire(engine)
        old_positions, old_cells = deepcopy(engine.mover.positions), deepcopy(engine.confirmed_cells)
        old_reference = engine.reference.reference_updates
        self.arm(engine)
        for now in (.5 + index * .25 for index in range(self.settle_samples)):
            row = engine.process(image(redraw=True), None, now, scene_visual_revision=1,
                scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
        self.assertFalse(row["held"])
        self.assertFalse(row["needs_reset"])
        self.assertEqual(engine.mover.positions, old_positions)
        self.assertEqual(engine.confirmed_cells, old_cells)
        self.assertEqual(engine.reference.reference_updates, old_reference + 1)
        self.assertEqual(row["events"], [])
        recovered_at = row["capture_seconds"]
        for now in (recovered_at + index * .125 for index in range(1, 6)):
            row = engine.process(image(redraw=True, moved=True), None, now, scene_visual_revision=1)
        self.assertEqual(len(engine.events), 1)
        self.assertEqual(engine.events[0]["mini"], "mini1")

    def test_camera_redraw_without_acknowledgement_does_not_release(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, .75, 1, 1.25):
            row = engine.process(image(redraw=True), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice())
        self.assertEqual(row["state"]["reason"], "waiting_for_foundry_move_confirmation")
        self.assertEqual(row["events"], [])

    def test_unknown_manual_mixed_scene_or_missing_revision_is_a_permanent_hold(self):
        for revision, change in ((1, notice(commands=("foreign",))),
                (1, notice(reason="tokenUpdate")), (1, notice(reasons=["sarweenTokenMove", "pan"])),
                (1, notice(sceneId="other")), (1, notice(tokenIds=["other"])),
                (2, notice(revision=2)), (1, notice(reasons=[{}])), (None, None)):
            with self.subTest(revision=revision, change=change):
                engine = self.engine()
                self.acquire(engine)
                self.arm(engine)
                row = engine.process(image(redraw=True), None, .5, scene_visual_revision=revision,
                    scene_visual_change=change, confirmed_command_ids=["move1"])
                self.assertTrue(row["needs_reset"])
                self.assertEqual(row["events"], [])

    def test_contiguous_history_is_checked_including_earlier_unknown_change(self):
        for earlier in (notice(), notice(reason="pan")):
            engine = self.engine()
            self.acquire(engine)
            self.arm(engine)
            latest = notice(revision=2)
            latest["changes"] = [earlier, deepcopy(latest)]
            row = engine.process(image(redraw=True), None, .5, scene_visual_revision=2,
                                 scene_visual_change=latest, confirmed_command_ids=["move1"])
            self.assertEqual(row["needs_reset"], earlier["reason"] != "sarweenTokenMove")

    def test_one_unseen_or_moved_base_prevents_reseed_even_with_redraw(self):
        for masked in (False, True):
            engine = self.engine()
            self.acquire(engine)
            self.arm(engine)
            mask = np.zeros((360, 400), np.uint8)
            if masked:
                mask[40:120, 40:120] = 255
            updates = engine.reference.reference_updates
            for now in (.5, .75, 1, 1.25):
                row = engine.process(image(redraw=True, moved=not masked), mask, now,
                    scene_visual_revision=1, scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertEqual(row["state"]["reason"], "reacquiring_bases_after_token_move")
            self.assertEqual(engine.reference.reference_updates, updates)

    def test_lock_or_mapping_loss_remains_strict(self):
        for kwargs in ({"locked": False}, {"geometry_key": {"different": True}}):
            engine = self.engine()
            self.acquire(engine)
            self.arm(engine)
            row = engine.process(image(redraw=True), None, .5, scene_visual_revision=1,
                scene_visual_change=notice(), confirmed_command_ids=["move1"], **kwargs)
            self.assertTrue(row["needs_reset"])

    def test_camera_gap_and_continuously_changing_board_do_not_count_as_settled(self):
        for frames in (((.5, 180), (1.25, 180), (2, 180)),
                       ((.5, 160), (.75, 180), (1, 200), (1.25, 160))):
            engine = self.engine()
            self.acquire(engine)
            self.arm(engine)
            for now, patch in frames:
                row = engine.process(image(redraw=True, patch_x=patch), None, now, scene_visual_revision=1,
                    scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertNotEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_noop_ack_resumes_but_late_correlated_redraw_still_has_known_provenance(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        row = engine.process(image(), None, .5, scene_visual_revision=0,
            confirmed_command_ids=["move1"], unchanged_command_ids=["move1"])
        self.assertFalse(row["held"])
        self.assertIsNone(engine._display_recovery)
        for now in (.75 + index * .25 for index in range(self.settle_samples)):
            row = engine.process(image(redraw=True), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice())
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_successful_retry_confirms_lost_ack_and_late_first_attempt_redraw(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        row = engine.process(image(), None, .5, scene_visual_revision=0,
            expected_display_commands=[command("retry")],
            superseded_display_commands=[{"command_id": "move1", "replacement_id": "retry"}],
            confirmed_command_ids=["retry"], unchanged_command_ids=["retry"])
        self.assertFalse(row["held"])
        for now in (.75 + index * .25 for index in range(self.settle_samples)):
            row = engine.process(image(redraw=True), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice())
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
        duplicate = engine.process(image(redraw=True), None, row["capture_seconds"] + .25, scene_visual_revision=2,
                                   scene_visual_change=notice(revision=2))
        self.assertFalse(duplicate["held"])
        self.assertFalse(duplicate["needs_reset"])

    def test_recorded_expected_redraw_inputs_replay_every_decision_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = self.engine()
            journal = ObserverJournal(Path(directory) / "recording", engine.initialization())
            for now in (0, .125, .25, .375, .5, .75, 1):
                metadata = {"scene_visual_revision": int(now >= .5)}
                if now == .375:
                    metadata["expected_display_commands"] = [command()]
                if now >= .5:
                    metadata.update(scene_visual_change=notice(), confirmed_command_ids=["move1"])
                pixels = image(redraw=now >= .5)
                row = engine.process(pixels, None, now, **metadata)
                journal.append(pixels, None, now, row, metadata=metadata)
            journal.close()
            replay = replay_journal(journal.folder)
            self.assertTrue(replay["verification_passed"])
            self.assertTrue(replay["identical_decisions"])

    def test_all_six_coalesced_initial_commands_recover_at_regular_point_six_second_cadence(self):
        engine = self.engine()
        for now in (0, .6, 1.2):
            row = engine.process(image(), None, now, scene_visual_revision=0)
        self.assertTrue(row["acquired"])
        commands = [{"command_id": f"align{i}", "scene_id": "test-scene", "token_id": f"token{i}"}
                    for i in range(6)]
        ids = [item["command_id"] for item in commands]
        change = notice(commands=ids, tokenIds=[f"token{i}" for i in range(6)])
        for index in range(3, 10):
            kwargs = {"expected_display_commands": commands} if index == 3 else {}
            row = engine.process(image(redraw=True), None, index * .6, scene_visual_revision=1,
                scene_visual_change=change, confirmed_command_ids=ids, **kwargs)
            if row["state"]["reason"] == "expected_display_reacquired":
                break
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
        self.assertEqual(engine.mover.positions, POSITIONS)
        self.assertEqual(engine.events, [])


class RecentDisplayRecoveryTests(ExpectedDisplayRecoveryTests):
    settle_samples = 5

    def engine(self):
        return super().engine(policy=RECENT_POLICY)

    @staticmethod
    def partial_redraw():
        pixels = image()
        pixels[175:180, 190:209, 0] += 21
        return pixels

    @staticmethod
    def complete_redraw():
        pixels = image()
        pixels[174:181, 189:210, 0] += 80
        pixels[174, 189, 0] = 150
        return pixels

    def test_partial_first_redraw_cannot_poison_later_three_complete_observations(self):
        baseline = cv2.GaussianBlur(image(), (5, 5), 0)
        partial, complete = self.partial_redraw(), self.complete_redraw()
        for pixels, expected_count in ((partial, 15), (complete, 193)):
            changed = np.max(cv2.absdiff(cv2.GaussianBlur(pixels, (5, 5), 0), baseline), axis=2) > 20
            self.assertEqual(np.count_nonzero(changed), expected_count)
        for policy in (POLICY, RECENT_POLICY):
            with self.subTest(policy=policy):
                engine = super().engine(policy=policy)
                self.acquire(engine)
                updates = engine.reference.reference_updates
                positions, cells = deepcopy(engine.mover.positions), deepcopy(engine.confirmed_cells)
                self.arm(engine)
                for now, pixels in ((.5, partial), (.75, complete), (1, complete), (1.25, complete),
                                    (1.5, complete), (1.75, complete)):
                    row = engine.process(pixels, None, now, scene_visual_revision=1,
                                         scene_visual_change=notice(), confirmed_command_ids=["move1"])
                    self.assertEqual(row["events"], [])
                    if now < 1.75 or policy == POLICY:
                        self.assertTrue(row["held"])
                    if now < 1.25 or policy == POLICY:
                        self.assertFalse(row["display_transition"]["camera_change_observed"])
                self.assertEqual(engine.mover.positions, positions)
                self.assertEqual(engine.confirmed_cells, cells)
                self.assertEqual(engine.events, [])
                if policy == RECENT_POLICY:
                    self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
                    self.assertEqual(engine.reference.reference_updates, updates + 1)
                else:
                    self.assertEqual(row["state"]["reason"], "waiting_for_visible_foundry_redraw")
                    self.assertEqual(engine.reference.reference_updates, updates)

    def test_intermittent_or_disjoint_small_flashes_cannot_supply_persistent_change(self):
        for disjoint in (False, True):
            with self.subTest(disjoint=disjoint):
                engine = self.engine()
                self.acquire(engine)
                self.arm(engine)
                for index in range(12):
                    pixels = image()
                    if disjoint or index % 2 == 0:
                        x = 160 + (index % 3) * 60 if disjoint else 190
                        pixels[175:182, x:x + 11, 0] += 80
                    row = engine.process(pixels, None, .5 + index * .25, scene_visual_revision=1,
                                         scene_visual_change=notice(), confirmed_command_ids=["move1"])
                    self.assertTrue(row["held"])
                    self.assertFalse(row["needs_reset"])
                    self.assertEqual(row["events"], [])
                    self.assertLessEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)
                self.assertFalse(row["display_transition"]["camera_change_observed"])
                # These are small enough to stay below the broad motion gate;
                # the recent common-pixel requirement is what prevents release.
                self.assertEqual(row["display_transition"]["stable_observations"], 12)

    def test_sixty_fps_still_requires_half_a_second_without_a_three_frame_span_deadlock(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for index in range(35):
            now = .5 + index / 60
            row = engine.process(self.complete_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertEqual(row["events"], [])
            if not row["held"]:
                break
            self.assertLessEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)
        self.assertGreaterEqual(now, 1. + 2 / 60)
        self.assertLess(now, 1.08)
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_old_quiet_time_cannot_settle_a_later_redraw_at_sixty_fps(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for index in range(37):
            row = engine.process(image(), None, .5 + index / 60, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertTrue(row["held"])
        self.assertIsNone(engine._display_recovery["positive_evidence_since"])
        first_positive = 1.12 + 2 / 60
        for index in range(35):
            now = 1.12 + index / 60
            row = engine.process(self.complete_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertEqual(row["events"], [])
            if now - first_positive < .5:
                self.assertTrue(row["held"])
            if not row["held"]:
                break
        self.assertGreaterEqual(now - first_positive, .5)
        self.assertLess(now - first_positive, .55)
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_loss_of_common_positive_pixels_restarts_the_redraw_settle_clock(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, .75, 1., 1.25):
            row = engine.process(self.complete_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertEqual(engine._display_recovery["positive_evidence_since"], 1.)
        row = engine.process(image(), None, 1.5, scene_visual_revision=1,
                             scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertTrue(row["held"])
        self.assertIsNone(engine._display_recovery["positive_evidence_since"])
        for now in (1.75, 2., 2.25, 2.5, 2.75):
            row = engine.process(self.complete_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            if now < 2.75:
                self.assertTrue(row["held"])
            self.assertEqual(row["events"], [])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_new_notice_or_capture_gap_discards_the_earlier_positive_window_and_clock(self):
        for trigger in ("new_notice", "capture_gap", "masked_base"):
            with self.subTest(trigger=trigger):
                engine = self.engine()
                self.acquire(engine)
                self.arm(engine)
                for now in (.5, .75, 1., 1.25):
                    row = engine.process(self.complete_redraw(), None, now, scene_visual_revision=1,
                                         scene_visual_change=notice(), confirmed_command_ids=["move1"])
                self.assertEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)
                self.assertEqual(engine._display_recovery["positive_evidence_since"], 1.)
                kwargs = {}
                revision, change, confirmed, timestamp, mask = 1, notice(), ["move1"], 1.5, None
                if trigger == "new_notice":
                    revision, change, confirmed = 2, notice(revision=2, commands=("move2",)), ["move1", "move2"]
                    kwargs["expected_display_commands"] = [command("move2")]
                elif trigger == "capture_gap":
                    timestamp = 2.
                else:
                    mask = np.zeros((360, 400), np.uint8)
                    mask[40:120, 40:120] = 255
                row = engine.process(self.complete_redraw(), mask, timestamp, scene_visual_revision=revision,
                                     scene_visual_change=change, confirmed_command_ids=confirmed, **kwargs)
                self.assertTrue(row["held"])
                self.assertEqual(row["events"], [])
                self.assertIsNone(engine._display_recovery["positive_evidence_since"])
                self.assertEqual(len(engine._display_recovery["recent_pixel_changes"]),
                                 0 if trigger == "masked_base" else 1)
                if trigger == "masked_base":
                    timestamp += .25
                    row = engine.process(self.complete_redraw(), None, timestamp, scene_visual_revision=revision,
                                         scene_visual_change=change, confirmed_command_ids=confirmed)
                    self.assertTrue(row["held"])
                for offset in (.25, .5, .75, 1.):
                    row = engine.process(self.complete_redraw(), None, timestamp + offset,
                                         scene_visual_revision=revision, scene_visual_change=change,
                                         confirmed_command_ids=confirmed)
                    self.assertEqual(row["events"], [])
                    if offset < 1.:
                        self.assertTrue(row["held"])
                self.assertEqual(row["state"]["reason"], "expected_display_reacquired")


class PositiveDisplayRecoveryTests(ExpectedDisplayRecoveryTests):
    def engine(self):
        return super().engine(policy=POSITIVE_POLICY)

    @staticmethod
    def small_redraw(x=190):
        pixels = image()
        pixels[175:182, x:x + 11, 0] += 80
        return pixels

    def test_partial_71_then_332_and_319_common_147_releases_on_third_clean_frame(self):
        positions = {f"mini{i + 1}": point for i, point in enumerate(
            ((80, 80), (480, 80), (880, 80), (80, 480), (480, 480), (880, 480)))}
        view = {**deepcopy(VIEW), "registration": {"left": 0, "top": 0, "right": 1024, "bottom": 576},
                "gridSize": 22}

        def pixels(stage=0):
            result = np.full((576, 1024, 3), (150, 90, 40), np.uint8)
            for center in positions.values():
                cv2.circle(result, center, 15, (35, 75, 180), -1)
                cv2.line(result, center, (center[0] - 20, center[1]), (35, 75, 180), 4)
            if stage == 1:
                result[230:235, 300:327, 0] += 22  # 71 changed pixels after blur.
            elif stage in (2, 3):
                result[230:241, 300:325, 0] += 21  # The 147 pixels shared by both full draws.
                if stage == 2:
                    result[230:239, 400:429, 0] += 22  # 185 independent changed pixels.
                else:
                    result[230:240, 500:524, 0] += 22  # 172 different changed pixels.
            return result

        baseline = cv2.GaussianBlur(pixels(), (5, 5), 0)
        changed = [np.max(cv2.absdiff(cv2.GaussianBlur(pixels(stage), (5, 5), 0), baseline), axis=2) > 20
                   for stage in (1, 2, 3)]
        self.assertEqual([np.count_nonzero(mask) for mask in changed], [71, 332, 319])
        self.assertEqual(np.count_nonzero(changed[1] & changed[2]), 147)
        for policy in (POLICY, RECENT_POLICY, POSITIVE_POLICY):
            with self.subTest(policy=policy):
                engine = ObserverEngine(BoardGeometry(view, (1024, 576)), positions,
                                        display_transition_policy=policy)
                for now in (0, .6, 1.2):
                    row = engine.process(pixels(), None, now, scene_visual_revision=0)
                self.assertTrue(row["acquired"])
                engine.process(pixels(), None, 1.8, scene_visual_revision=0,
                               expected_display_commands=[command()])
                for stage, now in enumerate((2.4, 3., 3.6), 1):
                    row = engine.process(pixels(stage), None, now, scene_visual_revision=1,
                                         scene_visual_change=notice(), confirmed_command_ids=["move1"])
                    self.assertEqual(row["events"], [])
                    if stage < 3 or policy != POSITIVE_POLICY:
                        self.assertTrue(row["held"])
                if policy == POSITIVE_POLICY:
                    self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
                    self.assertEqual(engine.mover.positions, positions)
                    self.assertEqual(engine.events, [])
                else:
                    self.assertEqual(row["state"]["reason"], "waiting_for_visible_foundry_redraw")

    def test_two_positive_observations_over_half_a_second_still_need_three_clean_layouts(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, 1.):
            row = engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertEqual(row["display_transition"]["stable_observations"], 2)
        self.assertTrue(row["held"])
        row = engine.process(self.small_redraw(), None, 1.25, scene_visual_revision=1,
                             scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")
        self.assertEqual(row["events"], [])

    def test_sixty_fps_late_positive_pixels_need_their_own_half_second(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for index in range(37):
            row = engine.process(image(), None, .5 + index / 60, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
        for index in range(35):
            now = 1.12 + index / 60
            row = engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertEqual(row["events"], [])
            if now - 1.12 < .5:
                self.assertTrue(row["held"])
            if not row["held"]:
                break
            self.assertLessEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)
        self.assertGreaterEqual(now - 1.12, .5)
        self.assertLess(now - 1.12, .55)
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_negative_fresh_frame_resets_positive_clock(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, .75):
            engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                           scene_visual_change=notice(), confirmed_command_ids=["move1"])
        row = engine.process(image(), None, 1., scene_visual_revision=1,
                             scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertTrue(row["held"])
        self.assertIsNone(engine._display_recovery["positive_evidence_since"])
        self.assertFalse(engine._display_recovery["recent_pixel_changes"])
        for now in (1.25, 1.5, 1.75):
            row = engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            if now < 1.75:
                self.assertTrue(row["held"])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_common_support_loss_reseeds_from_current_positive_instead_of_poisoning(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        engine.process(self.small_redraw(160), None, .5, scene_visual_revision=1,
                       scene_visual_change=notice(), confirmed_command_ids=["move1"])
        row = engine.process(self.small_redraw(240), None, .75, scene_visual_revision=1,
                             scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertTrue(row["held"])
        self.assertEqual(engine._display_recovery["positive_evidence_since"], .75)
        self.assertEqual(len(engine._display_recovery["recent_pixel_changes"]), 1)
        for now in (1., 1.25):
            row = engine.process(self.small_redraw(240), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_intermittent_and_disjoint_noise_cannot_run_out_a_positive_clock(self):
        for disjoint in (False, True):
            with self.subTest(disjoint=disjoint):
                engine = self.engine()
                self.acquire(engine)
                self.arm(engine)
                for index in range(12):
                    pixels = (self.small_redraw(160 + (index % 3) * 60) if disjoint else
                              self.small_redraw() if index % 2 == 0 else image())
                    row = engine.process(pixels, None, .5 + index * .25, scene_visual_revision=1,
                                         scene_visual_change=notice(), confirmed_command_ids=["move1"])
                    self.assertTrue(row["held"])
                    self.assertFalse(row["needs_reset"])
                    self.assertEqual(row["events"], [])
                    self.assertLessEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)

    def test_late_ack_cannot_reuse_positive_evidence_gathered_while_unconfirmed(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for now in (.5, .75, 1.):
            row = engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice())
            self.assertEqual(row["state"]["reason"], "waiting_for_foundry_move_confirmation")
        for now in (1.25, 1.5, 1.75):
            row = engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            if now < 1.75:
                self.assertTrue(row["held"])
        self.assertEqual(row["state"]["reason"], "expected_display_reacquired")

    def test_existing_notice_gap_motion_and_base_guards_reset_the_v3_positive_track(self):
        for trigger in ("new_notice", "capture_gap", "quiet_motion", "masked_base"):
            with self.subTest(trigger=trigger):
                engine = self.engine()
                self.acquire(engine)
                self.arm(engine)
                for now in (.5, .75):
                    engine.process(self.small_redraw(), None, now, scene_visual_revision=1,
                                   scene_visual_change=notice(), confirmed_command_ids=["move1"])
                self.assertEqual(engine._display_recovery["positive_evidence_since"], .5)
                kwargs = {}
                revision, change, confirmed, timestamp, mask = 1, notice(), ["move1"], 1., None
                pixels = self.small_redraw()
                if trigger == "new_notice":
                    revision, change, confirmed = 2, notice(revision=2, commands=("move2",)), ["move1", "move2"]
                    kwargs["expected_display_commands"] = [command("move2")]
                elif trigger == "capture_gap":
                    timestamp = 1.5
                elif trigger == "quiet_motion":
                    pixels = image(redraw=True)
                else:
                    mask = np.zeros((360, 400), np.uint8)
                    mask[40:120, 40:120] = 255
                row = engine.process(pixels, mask, timestamp, scene_visual_revision=revision,
                                     scene_visual_change=change, confirmed_command_ids=confirmed, **kwargs)
                self.assertTrue(row["held"])
                self.assertEqual(row["events"], [])
                self.assertEqual(engine._display_recovery["positive_evidence_since"],
                                 None if trigger == "masked_base" else timestamp)
                self.assertEqual(len(engine._display_recovery["recent_pixel_changes"]),
                                 0 if trigger == "masked_base" else 1)

    def test_high_fps_drifting_changes_cannot_borrow_an_earlier_positive_clock(self):
        engine = self.engine()
        self.acquire(engine)
        self.arm(engine)
        for index in range(45):
            now = .5 + index / 60
            row = engine.process(self.small_redraw(160 + index), None, now, scene_visual_revision=1,
                                 scene_visual_change=notice(), confirmed_command_ids=["move1"])
            self.assertTrue(row["held"])
            self.assertEqual(row["events"], [])
            self.assertLess(now - engine._display_recovery["positive_evidence_since"], .5)
            self.assertLessEqual(len(engine._display_recovery["recent_pixel_changes"]), 3)


class LiveControllerRecoveryTests(unittest.TestCase):
    def test_live_flags_queued_provenance_and_default_observer_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = LiveObserverController(None, Path(directory), observation_only=False,
                                                window_factory=FakeWindow)
            self.addCleanup(controller.close)
            self.assertIsNone(controller.binding_generation)
            controller.update(image(), None, 0, view=VIEW, locked=True,
                              metadata={"scene_visual_revision": 0})
            board, context = controller.current
            controller.initialize(POSITIONS, board, context)
            for now in (.125, .25):
                controller.update(image(), None, now, view=VIEW, locked=True,
                                  metadata={"scene_visual_revision": 0})
            self.assertTrue(controller.binding_generation)
            self.assertEqual(controller.engine.display_transition_policy, CONTINUOUS_DISPLAY_POLICY)
            self.assertTrue(controller.snapshot()["foundry_token_output"])
            self.assertTrue(controller.journal.manifest["foundry_token_output"])
            self.assertTrue(controller.expect_display_change("move1", "test-scene", "token1"))
            self.assertFalse(controller.expect_display_change("move1", "test-scene", "token1"))
            self.assertTrue(controller.display_waiting)
            self.assertFalse(controller.engine._expected_display_commands)
            for now in (.5, .75, 1, 1.25, 1.5):
                result = controller.update(image(redraw=True), None, now, view=VIEW, locked=True,
                    metadata={"scene_visual_revision": 1, "scene_visual_change": notice(),
                              "confirmed_command_ids": ["move1"]})
            self.assertFalse(result["held"])
            self.assertEqual(result["state"]["reason"], "layout_confirmed")
            self.assertFalse(result["display_transition"]["physical_tracking_waits_for_render"])
            for now in (1.625, 1.75, 1.875, 2, 2.125):
                controller.update(image(redraw=True, moved=True), None, now, view=VIEW, locked=True,
                                  metadata={"scene_visual_revision": 1})
            self.assertEqual(controller.movement_journal.status()["acceptedMoveCount"], 1)
            self.assertTrue(controller.movement_journal.recent_events()[0]["foundry_token_output"])
            controller._stop_recording()
            inputs = [json.loads(line) for line in (controller.journal.folder / "inputs.jsonl").read_text().splitlines()]
            self.assertEqual(inputs[3]["metadata"]["expected_display_commands"], [command()])
            self.assertTrue(replay_journal(controller.journal.folder)["verification_passed"])
            self.assertFalse(controller.display_waiting)
            held = controller.update(image(redraw=True, moved=True), None, 2.25, view=VIEW, locked=False,
                                     metadata={"scene_visual_revision": 1})
            self.assertTrue(held["needs_reset"])
            self.assertIsNone(controller.binding_generation)


if __name__ == "__main__":
    unittest.main()
