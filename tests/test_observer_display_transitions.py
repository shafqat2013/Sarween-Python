"""Known display mutations are holds, not claims of physical movement or settled fog."""
import json
from pathlib import Path
import tempfile
import unittest

from observer_journal import ObserverJournal, replay_journal
from tests.test_observer_journal import make_engine, image

POLICY = 'hold-known-display-changes-v1'


class DisplayTransitionTests(unittest.TestCase):
    def engine(self):
        return make_engine(display_transition_policy=POLICY)

    def acquire(self, engine, revision=0):
        for now in (0, .125, .25):
            engine.process(image(), None, now, scene_visual_revision=revision)
        self.assertTrue(engine.last_record['acquired'])

    def test_known_notification_holds_even_if_camera_still_sees_old_unchanged_pixels(self):
        engine = self.engine()
        self.acquire(engine)
        held = engine.process(image(), None, .5, scene_visual_revision=1)
        self.assertEqual(held['state']['reason'], 'display_changed_rebind_required')
        self.assertEqual(held['positions']['piece'], (100, 100))
        self.assertEqual(held['held'], ['piece'])
        self.assertTrue(held['display_transition']['requires_explicit_rebind'])
        # Neither an old quiet image nor later new pixels prove display timing.
        for now, pixels in ((1.0, image()), (2.0, image((260, 100))), (5.0, image())):
            row = engine.process(pixels, None, now, scene_visual_revision=1)
            self.assertTrue(row['needs_reset'])
            self.assertEqual(row['events'], [])
        self.assertEqual(engine.events, [])

    def test_pending_move_cannot_commit_after_known_display_mutation(self):
        engine = self.engine()
        self.acquire(engine, revision=7)
        engine.process(image((260, 100)), None, .375, scene_visual_revision=7)
        held = engine.process(image((260, 100)), None, .75, scene_visual_revision=8)
        self.assertTrue(held['needs_reset'])
        self.assertEqual(held['events'], [])
        self.assertEqual(held['positions']['piece'], (100, 100))

    def test_explicit_rebind_establishes_new_revision_and_fresh_acquisition(self):
        engine = self.engine()
        self.acquire(engine)
        engine.process(image(), None, .5, scene_visual_revision=1)
        engine.reset({'piece': (260, 100)}, {'piece': 'G3'})
        for now in (1.0, 1.125, 1.25):
            row = engine.process(image((260, 100)), None, now, scene_visual_revision=1)
        self.assertFalse(row['needs_reset'])
        self.assertTrue(row['acquired'])
        self.assertEqual(row['display_transition']['baseline_revision'], 1)
        self.assertEqual(row['events'], [])
        self.assertEqual(row['current_cells']['piece']['cell'], 'G3')

    def test_revision_loss_and_counter_restart_are_explicit_holds(self):
        for revision, reason in ((None, 'display_revision_unavailable'), (0, 'display_changed_rebind_required')):
            with self.subTest(revision=revision):
                engine = self.engine()
                self.acquire(engine, revision=9)
                held = engine.process(image(), None, .5, scene_visual_revision=revision)
                self.assertEqual(held['state']['reason'], reason)
                self.assertEqual(held['events'], [])

    def test_unavailable_metadata_is_explicit_and_legacy_policy_remains_unchanged(self):
        engine = self.engine()
        self.acquire(engine, revision=None)
        self.assertFalse(engine.needs_reset)
        self.assertFalse(engine.last_record['display_transition']['revision_available'])
        legacy = make_engine()
        self.acquire(legacy, revision=3)
        row = legacy.process(image(), None, .5, scene_visual_revision=4)
        self.assertFalse(row['needs_reset'])
        self.assertNotIn('display_transition', row)

    def test_bad_revision_is_rejected_before_advancing_clock(self):
        engine = self.engine()
        self.acquire(engine)
        for value in (True, -1, 1.5, '1', 2**64):
            with self.assertRaisesRegex(ValueError, 'Scene visual revision'):
                engine.process(image(), None, .5, scene_visual_revision=value)
        self.assertFalse(engine.process(image(), None, .5, scene_visual_revision=0)['needs_reset'])

    def test_replay_restores_policy_and_exact_recorded_revision_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'capture'
            engine = self.engine()
            journal = ObserverJournal(folder, engine.initialization())
            for index, revision in enumerate((4, 4, 4, 5, 5)):
                now = index / 8
                pixels = image()
                record = engine.process(pixels, None, now, scene_visual_revision=revision)
                journal.append(pixels, None, now, record, metadata={'scene_visual_revision': revision})
            journal.close()
            report = replay_journal(folder)
            self.assertTrue(report['verification_passed'])
            self.assertTrue(report['identical_decisions'])
            self.assertEqual(report['events'], [])
            decisions = [json.loads(line) for line in (folder / 'decisions.jsonl').read_text().splitlines()]
            self.assertEqual(decisions[-1]['state']['reason'], 'display_changed_rebind_required')
            # Same pixels without their revision evidence are not the same run.
            lines = [json.loads(line) for line in (folder / 'inputs.jsonl').read_text().splitlines()]
            for entry in lines:entry['metadata'].pop('scene_visual_revision')
            (folder / 'inputs.jsonl').write_text(''.join(json.dumps(entry)+'\n' for entry in lines))
            changed = replay_journal(folder)
            self.assertFalse(changed['identical_decisions'])


if __name__ == '__main__':
    unittest.main()
