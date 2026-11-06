"""Export readers can detect a JPEG/status pair captured between atomic writes."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import observer_session as session
from tests.test_observer_session import FakeWindow, image, view


class SnapshotPairTests(unittest.TestCase):
    def test_status_identifies_exact_preview_bytes_and_capture_context(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = session.LiveObserverController(None, Path(directory), window_factory=FakeWindow)
            try:
                controller.update(image(), None, 12.5, view=view(), locked=True, frame_index=19)
                status = json.loads((controller.folder/'latest.json').read_text())
                digest = hashlib.sha256((controller.folder/'latest.jpg').read_bytes()).hexdigest()
                self.assertEqual(status['preview']['sha256'], digest)
                self.assertEqual(status['preview']['frame_index'], 19)
                self.assertEqual(status['preview']['capture_timestamp'], 12.5)
                self.assertFalse(status['foundry_token_output'])
            finally:
                controller.close()

    def test_reader_between_writes_detects_mismatch_then_final_pair_matches(self):
        with tempfile.TemporaryDirectory() as directory:
            controller = session.LiveObserverController(None, Path(directory), window_factory=FakeWindow)
            try:
                controller.update(image(), None, 0, view=view(), locked=True, frame_index=0)
                controller.update(image((200, 100)), None, 1, view=view(), locked=True, frame_index=8)
                pixels, context = controller.current
                observed = []
                original = session._atomic
                def write(path, data):
                    original(path, data)
                    if path.name == 'latest.jpg':
                        old = json.loads((controller.folder/'latest.json').read_text())
                        observed.append(old['preview']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest())
                with patch.object(session, '_atomic', side_effect=write):
                    controller._publish(pixels, context, None, 'test', force=True)
                self.assertEqual(observed, [False])
                final = json.loads((controller.folder/'latest.json').read_text())
                self.assertEqual(final['preview']['sha256'], hashlib.sha256((controller.folder/'latest.jpg').read_bytes()).hexdigest())
                self.assertEqual(final['frame_index'], 8)
            finally:
                controller.close()


if __name__ == '__main__':
    unittest.main()
