"""Independent manual actions use confirmed state and existing safety gates."""
from unittest import TestCase
from unittest.mock import Mock

from tests import test_vtt_controls as fixture
from vtt_controls import VTTControls


def ready(**changes):
    return fixture.ready_status(capabilities=[*fixture.ACTIONS, "lifecycle.setPaused"],
                                paused=False, **changes)


class QuickControlsTests(TestCase):
    def controls(self):
        controls = fixture.VTTControlTests().controls()
        for name in ("pause_message", "fullscreen_message", "display_choice", "display_auto", "display_message"):
            setattr(controls, name, fixture.Variable())
        for name in ("pause_button", "unpause_button", "fullscreen_enter", "fullscreen_exit",
                     "display_combo", "display_place", "display_auto_check"):
            setattr(controls, name, Mock())
        controls._actions.extend([(controls.pause_button, "lifecycle.setPaused"),
                                  (controls.unpause_button, "lifecycle.setPaused")])
        controls._refresh_display = VTTControls._refresh_display.__get__(controls)
        controls._display.status.return_value = {"supported": True, "permission": True,
            "fullscreenSupported": True, "fullscreen": None, "fullscreenMessage": "",
            "displays": [{"id": "mac", "name": "Mac", "primary": True}],
            "selectedDisplay": "saved-tv", "busy": False, "auto": False,
            "message": "The saved display is disconnected."}
        controls._read_status.return_value = ready()
        controls.refresh(ready())
        return controls

    def state(self, widget):
        return widget.configure.call_args.kwargs["state"]

    def test_pause_buttons_follow_confirmed_state_and_external_changes(self):
        controls = self.controls()
        for paused, pause_state, unpause_state in ((True, "disabled", "normal"),
                                                  (False, "normal", "disabled")):
            status = ready()
            status["paused"] = paused
            controls.refresh(status)
            self.assertEqual(self.state(controls.pause_button), pause_state)
            self.assertEqual(self.state(controls.unpause_button), unpause_state)
            self.assertEqual(controls.pause_message.get(), "Game: paused" if paused else "Game: unpaused")
        controls.refresh(ready(connected=False))
        self.assertEqual(controls.pause_message.get(), "Game: status unavailable")
        self.assertEqual(self.state(controls.pause_button), "disabled")
        self.assertEqual(self.state(controls.unpause_button), "disabled")

    def test_absolute_pause_does_not_optimistically_change_displayed_state(self):
        controls = self.controls()
        self.assertTrue(controls.set_paused(True))
        controls._request.assert_called_once_with("lifecycle.setPaused", {"paused": True})
        self.assertEqual(controls.pause_message.get(), "Game: unpaused")
        self.assertTrue(controls.set_paused(False))
        controls._request.assert_called_with("lifecycle.setPaused", {"paused": False})
        controls._request.reset_mock()
        self.assertFalse(controls.set_paused(1))
        controls._request.assert_not_called()

    def test_pending_capture_and_revoked_access_block_pause(self):
        for status in (ready(pending=True), ready(capture={"phase": "Saving"}),
                       ready(blocked=True), ready(connected=False), ready(outputPaused=True)):
            controls = self.controls()
            controls._read_status.return_value = status
            self.assertFalse(controls.set_paused(True))
            controls._request.assert_not_called()
        controls = self.controls()
        controls.set_enabled(False)
        self.assertFalse(controls.set_paused(False))
        controls._request.assert_not_called()

    def test_fullscreen_does_not_require_relay_or_connected_saved_tv(self):
        controls = self.controls()
        controls._read_status.return_value = ready(connected=False)
        controls._display.set_fullscreen.return_value = True
        self.assertTrue(controls.set_fullscreen(True))
        controls._display.set_fullscreen.assert_called_once_with(True)
        controls._display.place.assert_not_called()
        controls._display.configure.assert_not_called()
        controls._request.assert_not_called()
        self.assertEqual(self.state(controls.fullscreen_enter), "normal")
        self.assertEqual(self.state(controls.fullscreen_exit), "normal")

    def test_fullscreen_buttons_remain_absolute_when_last_check_may_be_stale(self):
        controls = self.controls()
        controls._display.status.return_value.update(fullscreen=True,
            fullscreenMessage="Foundry is full screen.")
        controls.refresh(ready())
        self.assertEqual(self.state(controls.fullscreen_exit), "normal")
        self.assertEqual(self.state(controls.fullscreen_enter), "normal")
        self.assertIn("Last confirmed window: full screen", controls.fullscreen_message.get())
        self.assertTrue(controls.set_fullscreen(False))
        controls._display.set_fullscreen.assert_called_once_with(False)

    def test_fullscreen_errors_are_visible_and_permission_or_busy_disables_both(self):
        controls = self.controls()
        for change in ({"busy": True}, {"permission": False}, {"fullscreenSupported": False}):
            previous = dict(controls._display.status.return_value)
            controls._display.status.return_value.update(change)
            controls.refresh(ready())
            self.assertEqual(self.state(controls.fullscreen_enter), "disabled")
            self.assertEqual(self.state(controls.fullscreen_exit), "disabled")
            controls._display.status.return_value = previous
        controls._display.set_fullscreen.side_effect = RuntimeError("Window control unavailable")
        self.assertFalse(controls.set_fullscreen(True))
        self.assertEqual(controls.fullscreen_message.get(), "Window control unavailable")

    def test_capture_and_auth_block_native_actions_without_prompting(self):
        for status in (ready(capture={"recording": True}), ready(capture={"phase": "Restoring"})):
            controls = self.controls()
            controls._read_status.return_value = status
            self.assertFalse(controls.set_fullscreen(True))
            controls._display.set_fullscreen.assert_not_called()
        controls = self.controls()
        controls.set_enabled(False)
        self.assertEqual(self.state(controls.fullscreen_enter), "disabled")
        self.assertFalse(controls.set_fullscreen(False))
        controls._display.set_fullscreen.assert_not_called()
        controls.set_enabled(True)
        controls._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(controls.set_fullscreen(True))
        self.assertEqual(controls.fullscreen_message.get(), "Sign in first")

    def test_four_buttons_share_a_compact_group_and_send_absolute_actions(self):
        # Construction uses the headless widget harness, not native Tk/apps.
        from contextlib import ExitStack
        from unittest.mock import patch
        from tests import test_control_panel_layout as layout
        import vtt_controls
        controls = self.controls()
        with ExitStack() as patches:
            for name in ("Frame", "LabelFrame", "Label", "Button"):
                patches.enter_context(patch.object(vtt_controls.ttk, name, layout.Widget))
            patches.enter_context(patch.object(vtt_controls.tk, "StringVar", layout.Variable))
            parent = layout.Widget()
            controls._build_quick_controls(parent)
        self.assertEqual(len(parent.children), 1)
        section = parent.children[0]
        self.assertEqual(section.options["text"], "Quick controls")
        self.assertEqual(controls.pause_button.layout["row"], controls.unpause_button.layout["row"])
        self.assertEqual(controls.fullscreen_enter.layout["row"], controls.fullscreen_exit.layout["row"])
        controls.set_paused = Mock()
        controls.set_fullscreen = Mock()
        controls.pause_button.invoke()
        controls.unpause_button.invoke()
        self.assertEqual([call.args for call in controls.set_paused.call_args_list], [(True,), (False,)])
        controls.fullscreen_enter.invoke()
        controls.fullscreen_exit.invoke()
        self.assertEqual([call.args for call in controls.set_fullscreen.call_args_list], [(True,), (False,)])
