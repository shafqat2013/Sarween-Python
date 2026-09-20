import unittest

import cv_core as core


class _EndOfVideoCapture:
    def __init__(self):
        self.rewind_attempts = 0

    def read(self):
        return False, None

    def set(self, *_args):
        self.rewind_attempts += 1
        return True


class VideoEofTest(unittest.TestCase):
    def test_file_session_stops_at_eof_by_default(self):
        session = core.CVCoreSession.__new__(core.CVCoreSession)
        session.frame_idx = 0
        session.before_frame_callback = None
        session.cap = _EndOfVideoCapture()
        session._source_is_file = True
        session.marker_mode = "legacy"
        session.loop_video = False

        self.assertEqual(list(session.frames()), [])
        self.assertEqual(session.cap.rewind_attempts, 0)


if __name__ == "__main__":
    unittest.main()
