"""Current-lens profile setup stays asynchronous, cancellable and read-only until fresh."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from camera_sources import (PHONE_SOURCE, PhoneMetadataProbe, create_phone_profile,
                            matching_phone_profiles, phone_camera_index)
from hardware_profiles import HardwareProfiles
from tests.test_phone_setup import environment, setup, BUNDLE, PHONE, SOURCE, WIDE_METADATA
from tests.test_hardware_selection import Widget, DISPLAYS
from tests.test_session_startup import Scheduler


def metadata(lens="ultra-wide"):
    return {**deepcopy(WIDE_METADATA), "format": {**WIDE_METADATA["format"], "lens": lens,
            "cameraPosition": "back", "zoom": 1, "nativeFieldOfViewDegrees": 106.75 if lens == "ultra-wide" else 70.29}}


class PhoneMetadataProbeTests(unittest.TestCase):
    def probe(self, *, timeout=10, guard=lambda: None):
        self.scheduler = Scheduler()
        self.capture = Mock(status={"fresh": False})
        self.complete, self.failed = Mock(), Mock()
        factory = Mock(return_value=self.capture)
        probe = PhoneMetadataProbe(PHONE["index"], SOURCE, schedule=self.scheduler.after,
            cancel_schedule=self.scheduler.after_cancel, complete=self.complete, failed=self.failed,
            capture_factory=factory, guard=guard, timeout=timeout, clock=lambda: self.scheduler.now)
        self.addCleanup(probe.cancel)
        return probe

    def test_wait_is_nonblocking_then_fresh_actual_lens_is_delivered_once(self):
        probe = self.probe()
        self.complete.assert_not_called()
        self.capture.read.assert_not_called()
        self.scheduler.next()
        self.complete.assert_not_called()
        self.capture.status = metadata()
        self.scheduler.next()
        self.complete.assert_called_once()
        self.assertEqual(self.complete.call_args.args[0]["format"]["lens"], "ultra-wide")
        self.assertFalse(probe.active)
        self.assertFalse(self.scheduler.pending)
        self.capture.release.assert_called_once()
        self.capture.read.assert_not_called()

    def test_cancel_or_timeout_always_releases_without_a_profile_callback(self):
        probe = self.probe()
        probe.cancel()
        self.assertFalse(self.scheduler.pending)
        self.capture.release.assert_called_once()
        self.complete.assert_not_called()
        probe = self.probe(timeout=.2)
        self.scheduler.next()
        self.scheduler.next()
        self.scheduler.next()
        self.failed.assert_called_once()
        self.assertIn("No fresh iPhone feed", self.failed.call_args.args[0])
        self.capture.release.assert_called_once()
        self.complete.assert_not_called()

    def test_unknown_lens_and_revoked_permission_fail_closed(self):
        probe = self.probe()
        self.capture.status = {"fresh": True, "format": {"width": 1080, "height": 1920}}
        self.scheduler.next()
        self.failed.assert_called_once()
        self.complete.assert_not_called()
        self.capture.release.assert_called_once()
        guard = Mock()
        probe = self.probe(guard=guard)
        guard.side_effect = PermissionError("Access revoked")
        self.scheduler.next()
        self.assertEqual(self.failed.call_args.args[0], "Access revoked")
        self.capture.release.assert_called_once()


class PhoneLensProfileSetupTests(unittest.TestCase):
    def panel(self, module, store):
        panel = module.HardwareSelectionPanel(Widget(), store=store)
        panel.set_inventory(DISPLAYS, [PHONE])
        scheduler = Scheduler()
        panel.after, panel.after_cancel = scheduler.after, scheduler.after_cancel
        return panel, scheduler

    def test_ultra_wide_creation_does_not_relabel_or_reuse_existing_wide_calibration(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                wide = create_phone_profile(store, SOURCE, metadata=metadata("wide"))
                store.save_camera_calibration(wide["id"], [[1000, 0, 540], [0, 1000, 960], [0, 0, 1]],
                    [0, 0, 0, 0], (1080, 1920), rms=.2)
                before = store.get_profile("camera", wide["id"])
                panel, scheduler = self.panel(module, store)
                capture = Mock(status=metadata())
                with patch.object(module, "open_camera_capture", return_value=capture):
                    self.assertTrue(panel.create_iphone_profile())
                    self.assertEqual(store.get_profile("camera", wide["id"]), before)
                    scheduler.next()
                created = panel.profile_picker.selected_profile()
                self.assertEqual(created["lens"], "Back ultra-wide")
                self.assertIn("Ultra Wide", created["name"])
                self.assertIsNone(created["calibration"])
                self.assertFalse(created["bindingVerified"])
                self.assertEqual(store.get_profile("camera", wide["id"]), before)
                self.assertFalse(panel.profile_confirmed.get())
                capture.release.assert_called_once()
                capture.read.assert_not_called()

    def test_wide_profile_is_selected_unchanged_and_destroy_or_cancel_stops_pending_probe(self):
        for module in (setup, BUNDLE):
            with self.subTest(module=module.__name__), environment(module) as (store, _):
                wide = create_phone_profile(store, SOURCE, metadata=metadata("wide"))
                before = store.get_profile("camera", wide["id"])
                panel, scheduler = self.panel(module, store)
                capture = Mock(status=metadata("wide"))
                with patch.object(module, "open_camera_capture", return_value=capture):
                    panel.create_iphone_profile()
                    scheduler.next()
                self.assertEqual(panel.profile_picker.selected_id(), wide["id"])
                self.assertEqual(store.get_profile("camera", wide["id"]), before)
                capture = Mock(status={"fresh": False})
                with patch.object(module, "open_camera_capture", return_value=capture):
                    panel.create_iphone_profile()
                    self.assertEqual(panel.create_phone_profile_button.options["text"], "Cancel connection")
                    self.assertFalse(panel.create_iphone_profile())
                capture.release.assert_called_once()
                self.assertFalse(scheduler.pending)
                capture = Mock(status={"fresh": False})
                with patch.object(module, "open_camera_capture", return_value=capture):
                    panel.create_iphone_profile()
                    panel.bindings["<Destroy>"](SimpleNamespace(widget=panel))
                capture.release.assert_called_once()
                self.assertFalse(scheduler.pending)

    def test_profile_creation_requires_observed_lens_and_filters_resolution_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HardwareProfiles(Path(directory) / "profiles.json")
            with self.assertRaisesRegex(ValueError, "physical lens"):
                create_phone_profile(store, SOURCE, metadata={"deviceId": SOURCE["deviceId"], "format": {}})
            ultra = create_phone_profile(store, SOURCE, metadata=metadata())
            wide = create_phone_profile(store, SOURCE, metadata=metadata("wide"))
            self.assertEqual([p["id"] for p in matching_phone_profiles(store, SOURCE, metadata())], [ultra["id"]])
            self.assertEqual([p["id"] for p in matching_phone_profiles(store, SOURCE, metadata("wide"))], [wide["id"]])


if __name__ == "__main__":
    unittest.main()
