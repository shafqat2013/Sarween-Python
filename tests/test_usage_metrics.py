import json
from pathlib import Path
import tempfile
import unittest

from usage_metrics import UsageMetrics


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "usage.json"
        self.wall = 1790985600.0
        self.mono = 1.0
        self.metrics = UsageMetrics(self.path, wall=lambda: self.wall, monotonic=lambda: self.mono)
        self.user = "11111111-1111-4111-8111-111111111111"
        self.metrics.bind(self.user)

    def advance(self, seconds=1, allowed=True):
        self.wall += seconds
        self.mono += seconds
        self.metrics.tick(allowed)

    def test_time_modes_suspend_and_locked_time(self):
        self.metrics.state("tracking", True)
        self.metrics.state("tracking", True)
        self.advance(2)
        self.metrics.state("tracking", False)
        self.metrics.state("foundry", True)
        self.advance(3)
        self.advance(3600)  # A suspended Mac is not an hour of engagement.
        self.advance(3, allowed=False)
        row = self.metrics.snapshot()[0]
        self.assertEqual(row["app_seconds"], 5)
        self.assertEqual(row["tracking_seconds"], 2)
        self.assertEqual(row["foundry_seconds"], 3)
        self.assertEqual(row["tracking_sessions"], 1)
        self.assertEqual(row["foundry_connections"], 1)

    def test_offline_persistence_restart_and_acknowledgement(self):
        self.metrics.event("demo_opens")
        self.advance(2)
        self.metrics.persist()
        restarted = UsageMetrics(self.path, wall=lambda: self.wall, monotonic=lambda: self.mono)
        restarted.bind(self.user)
        self.assertEqual(restarted.snapshot()[0]["demo_opens"], 1)
        old = restarted.snapshot()
        restarted.tick(True)
        self.assertEqual(len(restarted.snapshot()), 2)
        restarted.acknowledge(old)
        self.assertEqual(len(restarted.snapshot()), 1)
        self.assertNotEqual(restarted.run_id, old[0]["run_id"])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_account_change_and_logout_never_reassign_reports(self):
        self.advance()
        self.metrics.bind("22222222-2222-4222-8222-222222222222")
        self.assertEqual(self.metrics.snapshot(), [])
        self.advance()
        self.metrics.bind(None)
        self.metrics.event("demo_opens")
        self.advance()
        self.assertEqual(self.metrics.snapshot(), [])
        self.metrics.persist()
        self.assertEqual(json.loads(self.path.read_text())["reports"], [])

    def test_out_of_order_ack_cannot_discard_new_data(self):
        self.advance()
        sent = self.metrics.snapshot()
        self.advance()
        self.metrics.acknowledge(sent)
        self.assertEqual(self.metrics.snapshot()[0]["app_seconds"], 2)

    def test_daily_split_expiry_and_no_freeform_content(self):
        self.advance()
        self.wall += 86400
        self.advance()
        self.assertEqual(len(self.metrics.snapshot()), 2)
        self.metrics.event("secret video filename.mp4")
        self.metrics.persist()
        self.assertNotIn("secret", self.path.read_text())
        self.wall += 31 * 86400
        self.assertEqual(self.metrics.snapshot(), [])

    def test_corrupt_cache_is_ignored(self):
        self.path.write_text('{"credential": "not-a-valid-report"}')
        fresh = UsageMetrics(self.path)
        self.assertIsNone(fresh.user_id)
        self.assertEqual(fresh.snapshot(), [])


if __name__ == "__main__":
    unittest.main()
