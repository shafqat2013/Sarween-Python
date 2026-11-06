"""Exercise delayed relay startup without networking, cameras or native windows."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
import v3_tracking


class RelayStartupTests(unittest.TestCase):
    def test_cold_relay_taking_more_than_three_seconds_reaches_live_session(self):
        clock = SimpleNamespace(now=0.0)

        class Service:
            access = Mock()
            error = ""
            start = Mock()

            @property
            def state(self):
                return "Listening" if clock.now >= 4.0 else "Starting"

        def sleep(seconds):
            clock.now += seconds

        with patch.object(main.s, "initialize"), \
             patch.object(main.s, "selected_mode", main.s.MODE_FOUNDRY), \
             patch.object(main.c, "calibrate") as calibrate, \
             patch.object(v3_tracking, "begin_session", return_value=None) as begin, \
             patch.object(v3_tracking, "close_pending_recording"), \
             patch("time.monotonic", side_effect=lambda: clock.now), \
             patch("time.sleep", side_effect=sleep):
            main.run_live_session(Service())
        calibrate.assert_called_once()
        begin.assert_called_once()
        self.assertGreaterEqual(clock.now, 4.0)

    def test_actual_bind_failure_is_reported_without_opening_camera(self):
        service = SimpleNamespace(access=Mock(), start=Mock(), state="Failed",
                                  error="Address already in use")
        with patch.object(main.s, "initialize"), \
             patch.object(main.s, "selected_mode", main.s.MODE_FOUNDRY), \
             patch.object(main.c, "calibrate") as calibrate:
            with self.assertRaisesRegex(RuntimeError, "Address already in use"):
                main.run_live_session(service)
        calibrate.assert_not_called()
