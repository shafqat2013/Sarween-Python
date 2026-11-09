"""Guided session sequencing; no real cameras, Foundry windows or sockets."""
from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

from tests import test_session_startup as models
from tests.test_session_startup import Value


class GuidedSessionTests(unittest.TestCase):
    setUp = models.SessionStartupTests.setUp
    model = models.SessionStartupTests.model
    finish_inventory = models.SessionStartupTests.finish_inventory

    def guided(self):
        ui = self.model()
        ui.step = Value()
        ui._guided_ready = False
        ui._placement_warning = None
        ui._scroll_to_top = Mock()
        ui.selector.pack = Mock()
        ui.selector.pack_forget = Mock()
        ui.foundry_setup = Mock()
        ui.foundry_setup.begin.return_value = True
        ui.foundry_setup.status.return_value = {
            "phase": "foundry", "ready": False, "message": "Joining the world…", "error": None, "warning": ""}
        return ui

    def test_hardware_next_only_opens_configuration_without_connecting(self):
        ui = self.guided()
        ui.continue_setup()
        self.assertEqual(ui.phase, "foundry_setup")
        self.assertEqual(ui.selection_data, ui.selector.draft)
        ui.foundry_setup.set_display.assert_called_once_with(ui.displays[0])
        ui.foundry_setup.begin.assert_not_called()
        self.assertEqual(ui.service.start_count, 0)
        self.camera.assert_not_called()
        ui.on_ready.assert_not_called()

    def test_final_start_waits_for_foundry_settings_before_camera_handoff(self):
        ui = self.guided()
        ui.continue_setup()
        ui.continue_setup()
        self.assertEqual(ui.phase, "foundry")
        ui.foundry_setup.begin.assert_called_once()
        for _ in range(5):
            self.scheduler.next()
        ui.on_ready.assert_not_called()
        self.camera.assert_not_called()
        ui.service.phase = "ready"
        ui.foundry_setup.status.return_value.update(phase="ready", ready=True, message="Foundry and TV ready")
        self.scheduler.until(lambda: ui.on_ready.called)
        ui.on_ready.assert_called_once_with(ui)

    def test_settings_failure_keeps_form_editable_and_retries_same_choices(self):
        ui = self.guided()
        ui.continue_setup()
        draft = deepcopy(ui.selection_data)
        ui.foundry_setup.status.return_value.update(phase="failed", error="Unknown Foundry user")
        ui.continue_setup()
        self.assertEqual(ui.phase, "failed")
        self.assertIn("Unknown Foundry user", ui.message.get())
        ui.foundry_setup.set_enabled.assert_called_with(True)
        self.assertEqual(ui.selection_data, draft)
        ui.foundry_setup.status.return_value.update(phase="foundry", error=None)
        ui.continue_setup()
        self.assertEqual(ui.phase, "foundry")
        self.assertEqual(ui.foundry_setup.begin.call_count, 2)

    def test_manual_tv_placement_pauses_before_settings_or_camera(self):
        ui = self.guided()
        ui.continue_setup()
        ui.foundry_setup.status.return_value.update(
            phase="placement_attention", warning="Place Foundry on the TV manually.")
        ui.continue_setup()
        self.assertEqual(ui.phase, "placement")
        self.assertIn("manually", ui.message.get())
        self.assertFalse(self.scheduler.pending)
        self.assertFalse(ui.continue_button.disabled)
        ui.on_ready.assert_not_called()
        ui.foundry_setup.confirm_placement.assert_not_called()
        ui.foundry_setup.status.return_value.update(phase="settings", warning="", message="Applying TV settings…")
        ui.continue_setup()
        ui.foundry_setup.confirm_placement.assert_called_once_with()
        self.assertEqual(ui.phase, "foundry")
        self.assertTrue(self.scheduler.pending)
        ui.on_ready.assert_not_called()

    def test_back_cancels_startup_preserves_choices_and_never_opens_camera(self):
        ui = self.guided()
        ui.continue_setup()
        ui.continue_setup()
        draft = deepcopy(ui.selection_data)
        ui.edit_hardware()
        self.assertEqual(ui.phase, "hardware")
        self.assertEqual(ui.selector.draft, draft)
        ui.foundry_setup.cancel.assert_called_once()
        self.assertFalse(self.scheduler.pending)
        self.camera.assert_not_called()

    def test_cancel_stops_only_setup_owned_relay_and_closes_guided_controller(self):
        for running in (False, True):
            ui = self.guided()
            ui.service.running = running
            ui.continue_setup()
            ui.continue_setup()
            # The guided coordinator owns starting the relay, not the fake.
            ui.service.running = True
            ui.cancel()
            ui.foundry_setup.close.assert_called_once()
            self.assertEqual(ui.service.stop_count, 0 if running else 1)
            ui.on_ready.assert_not_called()

    def test_airplay_name_restores_current_index_instead_of_old_number(self):
        ui = self.guided()
        ui.inventory_loader.return_value = ([
            {"index": 0, "name": "Mac", "width": 1440, "height": 900},
            {"index": 3, "name": "TV", "width": 1920, "height": 1080}], [])
        with patch("setup.load_last_selection", return_value={"display_index": 0, "display_name": "TV"}):
            self.finish_inventory(ui)
        self.assertEqual(ui.selector.inventories[-1][2]["display_index"], 3)

    def test_missing_saved_tv_requires_choice_instead_of_selecting_mac(self):
        ui = self.guided()
        ui.inventory_loader.return_value = ([{"index": 0, "name": "Mac", "width": 1440, "height": 900}], [])
        with patch("setup.load_last_selection", return_value={"display_index": 0, "display_name": "TV"}):
            self.finish_inventory(ui)
        self.assertEqual(ui.selector.inventories[-1][2]["display_index"], -1)
