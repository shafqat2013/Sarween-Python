"""Desktop VTT controls exercised without creating Tk or using native apps."""

import unittest
from unittest.mock import Mock

from vtt_controls import BOOL_SETTINGS, NUMBER_SETTINGS, VTTControls, capture_participants, parse_settings


class Variable:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


ACTIONS = ["settings.set", "ui.setHidden", "capture.start", "capture.confirm", "capture.stop",
           "capture.new", "capture.resume", "movement.select", "movement.undo", "movement.reset",
           "assignment.save", "assignment.cancel", "lifecycle.reload"]


def ready_status(**overrides):
    return {
        "connected": True, "supported": True, "pending": False, "blocked": False,
        "message": "Ready", "capabilities": ACTIONS,
        "settings": {**{name: default for name, (_label, default) in BOOL_SETTINGS.items()},
                     **{name: values[-1] for name, values in NUMBER_SETTINGS.items()}},
        "scene": {"id": "scene", "name": "Test scene"},
        "tokens": [{"tokenId": "red-token", "tokenName": "Red"},
                   {"tokenId": "blue-token", "tokenName": "Blue"}],
        "minis": [{"miniId": "red10", "route": "A", "tokenName": "Red", "tokenId": "red-token"},
                  {"miniId": "blue", "route": "B", "tokenName": "Blue", "tokenId": "blue-token"}],
        "capture": {"state": "idle", "recording": False, "waiting": False},
        **overrides,
    }


class VTTControlTests(unittest.TestCase):
    def test_compact_status_is_one_line_without_changing_full_message(self):
        controls = self.controls()
        controls.footer_message = Variable()
        controls.connection.set("Foundry: connected")
        original = "First line\n" + "Detailed explanation " * 40
        controls.message.set(original)
        controls._sync_status_summary()
        self.assertNotIn("\n", controls.footer_message.get())
        self.assertLessEqual(len(controls.footer_message.get()), 160)
        self.assertIn("Foundry: connected", controls.footer_message.get())
        self.assertEqual(controls.message.get(), original)

    def controls(self):
        controls = VTTControls.__new__(VTTControls)
        controls._require_access = Mock()
        controls._service = None
        controls._live_session = True
        controls._read_status = Mock(return_value=ready_status())
        controls._request = Mock(return_value=True)
        controls._enabled = True
        controls._last_refresh = float("-inf")
        controls._last_status = {}
        controls._syncing = False
        controls._dirty = set()
        controls._submitted = {}
        controls._remote_settings = {}
        controls._notice = ""
        controls._actions = []
        controls._setting_widgets = []
        controls._capture_setup_widgets = []
        controls._tokens = {}
        controls._token_labels = {}
        controls._mini_labels = {}
        controls._library_rows = []
        controls._capture_initialized = set()
        controls._choice_scene_id = None
        controls._choice_token_ids = {}
        controls._assignment_request = None
        controls._display_busy = False
        controls._capture_rows = [{"miniId": None, "enabled": Variable(False), "name": Variable(),
                                   "token": Variable(), "combo": Mock(), "check": Mock()} for _ in range(6)]
        controls.variables = {name: Variable(value) for name, value in ready_status()["settings"].items()}
        for name in ("connection", "relay", "scene", "message", "movement_message", "movement_choice",
                     "capture_message", "assignment_message", "assignment_mini", "assignment_token", "notes"):
            setattr(controls, name, Variable())
        controls.scenario = Variable("Fog on")
        for name in ("movement_combo", "assignment_token_combo", "assignment_mini_combo", "connect_button"):
            setattr(controls, name, Mock())
        controls.grid_controls = Mock()
        controls.display_profiles = Mock()
        controls._display = Mock()
        controls._refresh_display = Mock()
        controls.refresh(ready_status())
        return controls

    def test_disconnected_pending_unsupported_and_grid_resize_block_commands(self):
        for state in ({"connected": False}, {"supported": False}, {"pending": True}, {"blocked": True}):
            with self.subTest(state=state):
                controls = self.controls()
                controls._read_status.return_value = ready_status(**state)
                self.assertFalse(controls.command("movement.reset"))
                controls._request.assert_not_called()

    def test_capabilities_and_auth_are_required_before_sending(self):
        controls = self.controls()
        controls._read_status.return_value = ready_status(capabilities=[])
        self.assertFalse(controls.command("lifecycle.reload"))
        controls._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(controls.command("lifecycle.reload"))
        self.assertEqual(controls.message.get(), "Sign in first")
        controls._request.assert_not_called()

    def test_capture_only_enables_placed_during_an_active_placement(self):
        controls = self.controls()
        for phase, waiting, expected in (("placement", False, True), ("placement", True, False),
                                          ("Baseline", False, False)):
            controls.refresh(ready_status(capture={"recording": True, "phase": phase, "waiting": waiting}))
            self.assertEqual(controls._allowed("capture.confirm"), expected)
            self.assertTrue(controls._allowed("capture.stop"))
            self.assertFalse(controls._allowed("settings.set"))
            self.assertFalse(controls._allowed("lifecycle.reload"))
            self.assertFalse(controls._allowed("capture.start"))
            self.assertFalse(controls._allowed("capture.resume"))

    def test_capture_setup_sends_unique_mini_token_pairs_and_scenario(self):
        controls = self.controls()
        controls.notes.set("Evening lighting")
        controls.scenario.set("Fog off")
        self.assertTrue(controls.start_capture())
        controls._request.assert_called_once_with("capture.start", {
            "participants": [{"miniId": "red10", "tokenId": "red-token"},
                             {"miniId": "blue", "tokenId": "blue-token"}],
            "scenario": "fog-off", "notes": "Evening lighting"})
        controls._capture_rows[1]["token"].set(controls._capture_rows[0]["token"].get())
        self.assertFalse(controls.start_capture())
        self.assertIn("different", controls.message.get())

    def test_six_capture_rows_bind_to_tokens_without_color_profiles(self):
        controls = self.controls()
        names = ["Red", "Blue", "Yellow", "Green", "White", "Orange"]
        ids = ["red10", "blue", "yellow", "green", "white", "orange"]
        state = ready_status(
            tokens=[{"tokenId": f"token-{index}", "tokenName": name} for index, name in enumerate(names)],
            minis=[{"miniId": mini_id, "route": chr(65 + index), "tokenName": names[index],
                    "tokenId": f"token-{index}"} for index, mini_id in enumerate(ids)])
        controls.refresh(state)
        controls._read_status.return_value = state
        self.assertTrue(controls.start_capture())
        payload = controls._request.call_args.args[1]
        self.assertEqual(payload["participants"], [
            {"miniId": mini_id, "tokenId": f"token-{index}"} for index, mini_id in enumerate(ids)])
        self.assertEqual(controls._capture_rows[-1]["name"].get(), "F · Orange")

    def test_capture_roster_accepts_six_and_rejects_seven(self):
        rows = [(f"mini-{index}", True, f"token-{index}") for index in range(7)]
        tokens = {row[2] for row in rows}
        self.assertEqual(len(capture_participants(rows[:6], tokens)), 6)
        with self.assertRaisesRegex(ValueError, "one to six"):
            capture_participants(rows, tokens)

    def test_settings_preserve_drafts_then_clear_after_acknowledgement(self):
        controls = self.controls()
        controls.variables["viewportMarkerPatternSize"].set("160")
        controls._edited("viewportMarkerPatternSize")
        remote = {**ready_status()["settings"], "viewportMarkerInset": 16}
        controls.refresh(ready_status(settings=remote))
        self.assertEqual(controls.variables["viewportMarkerPatternSize"].get(), "160")
        self.assertEqual(controls.variables["viewportMarkerInset"].get(), 16.0)
        controls._read_status.side_effect = [ready_status(settings=remote), ready_status(pending=True, settings=remote)]
        self.assertTrue(controls.apply_settings(("viewportMarkerPatternSize", "viewportMarkerInset")))
        controls._request.assert_called_once_with("settings.set", {"settings": {
            "viewportMarkerPatternSize": 160.0, "viewportMarkerInset": 16.0}})
        controls.refresh(ready_status(settings={**remote, "viewportMarkerPatternSize": 160}))
        self.assertEqual(controls._dirty, set())

    def test_failed_setting_keeps_draft_and_disconnection_message_remains_visible(self):
        controls = self.controls()
        controls.variables["viewportMarkerPatternSize"].set("160")
        controls._edited("viewportMarkerPatternSize")
        controls._request.return_value = False
        self.assertFalse(controls.apply_settings(("viewportMarkerPatternSize",)))
        self.assertEqual(controls.variables["viewportMarkerPatternSize"].get(), "160")
        controls._notice = "Old validation message"
        controls.refresh(ready_status(connected=False, message="Connect Foundry to use VTT controls."))
        self.assertEqual(controls.message.get(), "Connect Foundry to use VTT controls.")

    def test_margin_field_reports_total_without_resizing_the_pattern(self):
        controls = self.controls()
        controls.marker_total = Variable()
        controls.variables["viewportMarkerPatternSize"].set("48")
        for margin, total in ((0, 48), (6, 60), (12, 72)):
            controls.variables["viewportMarkerMargin"].set(str(margin))
            controls._update_marker_total()
            self.assertIn(f"Total width: {total} pixels", controls.marker_total.get())
            self.assertEqual(controls.variables["viewportMarkerPatternSize"].get(), "48")

    def test_small_width_and_zero_margin_are_allowed_and_invalid_margin_rejected(self):
        self.assertEqual(parse_settings({"viewportMarkerPatternSize": "24", "viewportMarkerMargin": "0"},
                         ("viewportMarkerPatternSize", "viewportMarkerMargin")),
                         {"viewportMarkerPatternSize": 24, "viewportMarkerMargin": 0})
        for value in (-1, 13, float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_settings({"viewportMarkerMargin": value}, ("viewportMarkerMargin",))

    def test_legacy_module_size_is_converted_for_new_controls(self):
        controls = self.controls()
        controls._sync_settings({"viewportMarkerSize": 72}, False)
        self.assertEqual(controls.variables["viewportMarkerPatternSize"].get(), 48)
        self.assertEqual(controls.variables["viewportMarkerMargin"].get(), 12)

    def test_assignment_request_supports_new_mini_and_duplicate_token_names(self):
        controls = self.controls()
        state = ready_status(assignment={"miniId": "new-mini", "sceneId": "scene"},
            tokens=[{"tokenId": "one", "tokenName": "Goblin"}, {"tokenId": "two", "tokenName": "Goblin"}])
        controls._read_status.return_value = state
        controls.refresh(state)
        self.assertEqual(controls._mini_labels[controls.assignment_mini.get()], "new-mini")
        self.assertEqual(len(controls._token_labels), 2)
        controls.assignment_token.set("Goblin (two)")
        self.assertTrue(controls.save_assignment())
        controls._request.assert_called_once_with("assignment.save", {"miniId": "new-mini", "tokenId": "two"})

    def test_default_capture_choices_do_not_overwrite_manual_selection(self):
        controls = self.controls()
        red = controls._capture_rows[0]
        red["enabled"].set(False)
        red["token"].set("Blue (blue-token)")
        controls._capture_edited(red)
        controls.refresh(ready_status())
        self.assertFalse(red["enabled"].get())
        self.assertEqual(red["token"].get(), "Blue (blue-token)")

    def test_capture_scene_change_uses_new_bindings_and_token_rename_preserves_id(self):
        controls = self.controls()
        red = controls._capture_rows[0]
        red["token"].set("Blue (blue-token)")
        controls._capture_edited(red)
        renamed = ready_status(tokens=[{"tokenId": "red-token", "tokenName": "Red"},
                                      {"tokenId": "blue-token", "tokenName": "Azure"}])
        controls.refresh(renamed)
        self.assertEqual(red["token"].get(), "Azure (blue-token)")
        new_scene = ready_status(scene={"id": "next-scene", "name": "Next scene"},
            tokens=[{"tokenId": "new-red", "tokenName": "Red"}],
            minis=[{"miniId": "red10", "route": "A", "tokenName": "Red", "tokenId": "new-red"}])
        controls.refresh(new_scene)
        self.assertEqual(red["token"].get(), "Red (new-red)")
        self.assertTrue(red["enabled"].get())
        self.assertEqual(controls._capture_rows[1]["miniId"], None)

    def test_capture_rebinds_when_old_token_disappears_without_changing_scene(self):
        controls = self.controls()
        state = ready_status(tokens=[{"tokenId": "replacement", "tokenName": "Red"}],
            minis=[{"miniId": "red10", "route": "A", "tokenName": "Red", "tokenId": "replacement"}])
        controls.refresh(state)
        self.assertEqual(controls._capture_rows[0]["token"].get(), "Red (replacement)")

    def test_auth_gates_native_display_automation_and_relay_start(self):
        controls = self.controls()
        controls.set_enabled(False)
        controls._display.set_enabled.assert_called_with(False)
        controls._service = Mock(running=False, state="Stopped")
        controls._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(controls.toggle_connection())
        controls._service.start.assert_not_called()
        controls._service.running = True
        self.assertTrue(controls.toggle_connection())
        controls._service.stop.assert_called_once()

    def test_capture_temporarily_pauses_native_placement_without_changing_preference(self):
        controls = self.controls()
        controls._display.status.return_value = {"supported": True, "permission": True,
            "displays": [{"id": "tv", "name": "TV", "primary": False}],
            "selectedDisplay": "tv", "auto": True, "busy": False, "message": "Ready"}
        controls.display_choice = Variable()
        controls.display_auto = Variable()
        controls.display_message = Variable()
        controls.display_combo = Mock()
        controls.display_place = Mock()
        controls.display_auto_check = Mock()
        controls._refresh_display = VTTControls._refresh_display.__get__(controls)
        for capture in ({"recording": True}, {"recording": False, "phase": "Starting"}):
            controls.refresh(ready_status(capture=capture))
            controls.set_enabled(True)
            controls._display.set_enabled.assert_called_with(False)
            self.assertFalse(controls._native_allowed())
            controls.grid_controls.set_enabled.assert_called_with(False)
            controls._display.configure.assert_not_called()
        controls.refresh(ready_status())
        controls.set_enabled(True)
        controls._display.set_enabled.assert_called_with(True)
        self.assertTrue(controls._native_allowed())

    def test_capture_cannot_start_while_foundry_window_is_moving(self):
        controls = self.controls()
        controls._display_busy = True
        self.assertFalse(controls.start_capture())
        controls._request.assert_not_called()
        controls._display_busy = False
        self.assertTrue(controls.start_capture())

    def test_home_capture_requires_a_running_live_session(self):
        controls = self.controls()
        controls._live_session = False
        controls.refresh(ready_status())
        self.assertFalse(controls.start_capture())
        controls._request.assert_not_called()
        self.assertIn("Start a live session before recording.", controls.capture_message.get())

    def test_saving_capture_keeps_module_and_native_actions_paused(self):
        controls = self.controls()
        for phase in ("Saving", "Restoring"):
            controls.refresh(ready_status(capture={"recording": False, "phase": phase}))
            for action in ("capture.start", "capture.stop", "capture.new", "capture.resume",
                           "movement.select", "settings.set", "lifecycle.reload"):
                with self.subTest(action=action, phase=phase):
                    self.assertFalse(controls._allowed(action))
            self.assertFalse(controls._native_allowed())
        controls.refresh(ready_status(capture={"active": True, "recording": False, "phase": "Finishing"}))
        self.assertFalse(controls._allowed("capture.start"))
        self.assertFalse(controls._allowed("capture.resume"))
        self.assertFalse(controls._native_allowed())

    def test_display_callback_saves_the_user_choice_before_status_refresh(self):
        controls = self.controls()
        controls._display_names = {"Mac": "one", "TV": "two"}
        controls.display_choice = Variable("TV")
        controls.display_auto = Variable(True)
        controls.display_message = Variable()

        def refresh(_state):
            # The normal status poll restores the controller's saved values.
            controls.display_choice.set("Mac")
            controls.display_auto.set(False)

        controls.refresh = Mock(side_effect=refresh)
        controls.configure_display()
        controls._display.configure.assert_called_once_with(display_id="two", auto=True)

    def test_validation_rejects_invalid_settings_and_capture_rosters(self):
        for settings in ({"viewportMarkerSize": "bad"}, {"viewportMarkerSize": float("nan")},
                         {"viewportMarkerInset": 81}, {"defaultMovementSpeed": -1},
                         {"animateTokenMovement": "true"}, {"unknown": 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                parse_settings(settings, settings)
        for rows in ([], [("red10", True, "missing")],
                     [("red10", True, "one"), ("blue", True, "one")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                capture_participants(rows, {"one"})


if __name__ == "__main__":
    unittest.main()
