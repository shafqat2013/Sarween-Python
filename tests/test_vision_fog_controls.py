"""Vision controls preserve drafts and require explicit, current-scene requests."""
from copy import deepcopy
import unittest
from unittest.mock import Mock

from vision_fog_controls import SCENE_FIELDS, VisionFogControls, sight_range


class Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def ready(**changes):
    return {"connected": True, "supported": True, "pending": False, "blocked": False,
            "sceneId": "scene-a", "scene": {"name": "Forest"}, "world": {"id": "testing"},
            "capabilities": ["vision.scene.set", "vision.token.set", "vision.token.select"],
            "vision": {"tokenVision": False, "fogExploration": True, "globalIllumination": True,
                       "selectedTokenId": "red", "distanceUnits": "ft"},
            "tokens": [{"tokenId": "red", "tokenName": "Red", "vision": {"enabled": True, "range": 30}},
                       {"tokenId": "blue", "tokenName": "Blue", "vision": {"enabled": False, "range": 0}}],
            **changes}


class VisionFogTests(unittest.TestCase):
    def controls(self):
        ui = VisionFogControls.__new__(VisionFogControls)
        ui._read_status, ui._request = Mock(return_value=ready()), Mock(return_value="request-1")
        ui._require_access, ui._can_change = Mock(), Mock(return_value=True)
        ui._enabled = True
        ui._status, ui._tokens, ui._labels = {}, {}, {}
        ui._context = ui._token_id = ui._pending = None
        ui._dirty, ui._submitted, ui._remote = set(), {}, {}
        ui._syncing = False
        ui._notice = ""
        ui.variables = {name: Value(False) for name in SCENE_FIELDS}
        ui.variables.update(enabled=Value(False), range=Value("0"))
        for name in ("token", "scene_name", "tv_view", "range_label", "message"):
            setattr(ui, name, Value())
        ui._scene_widgets = [Mock() for _ in SCENE_FIELDS]
        for name in ("apply_scene_button", "token_combo", "vision_check", "range_entry",
                     "apply_token_button", "show_view_button"):
            setattr(ui, name, Mock())
        ui.refresh(ready())
        return ui

    def edit(self, ui, name, value):
        ui.variables[name].set(value)
        ui._edited(name)

    def choose(self, ui, token_id):
        ui.token.set(next(label for label, key in ui._labels.items() if key == token_id))
        ui._choose_token()

    def test_range_rejects_invalid_values_without_an_arbitrary_resolution_limit(self):
        for value in (True, None, "", "invalid", -1, "nan", "inf", "-inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                sight_range(value)
        for value in (0, "0", "30.5", 1000000):
            self.assertEqual(sight_range(value), float(value))

    def test_remote_settings_and_current_tv_view_are_read_without_sending(self):
        ui = self.controls()
        self.assertEqual(ui.scene_name.get(), "Scene: Forest")
        self.assertEqual(ui.variables["tokenVision"].get(), False)
        self.assertEqual(ui.variables["range"].get(), "30")
        self.assertEqual(ui.range_label.get(), "Sight range (ft)")
        self.assertEqual(ui.tv_view.get(), "TV view: Red")
        for _ in range(5):
            ui.refresh(ready())
        ui._request.assert_not_called()

    def test_polling_preserves_unsaved_edits_but_updates_other_remote_fields(self):
        ui = self.controls()
        self.edit(ui, "tokenVision", True)
        self.edit(ui, "range", "45")
        state = ready()
        state["vision"]["fogExploration"] = False
        state["tokens"][0]["vision"]["range"] = 60
        ui.refresh(state)
        self.assertTrue(ui.variables["tokenVision"].get())
        self.assertFalse(ui.variables["fogExploration"].get())
        self.assertEqual(ui.variables["range"].get(), "45")
        ui._request.assert_not_called()

    def test_scene_apply_is_explicit_and_failed_ack_keeps_edit_for_retry(self):
        ui = self.controls()
        self.edit(ui, "tokenVision", True)
        self.edit(ui, "globalIllumination", False)
        self.assertTrue(ui.apply_scene())
        ui._request.assert_called_once_with("vision.scene.set", {
            "tokenVision": True, "fogExploration": True, "globalIllumination": False})
        self.assertEqual(ui.apply_scene_button.configure.call_args.kwargs["state"], "disabled")
        ui.refresh(ready(lastResult={"requestId": "request-1", "success": False, "message": "Permission denied"}))
        self.assertEqual(ui.message.get(), "Permission denied")
        self.assertTrue(ui.variables["tokenVision"].get())
        self.assertFalse(ui.variables["globalIllumination"].get())
        self.assertIn("tokenVision", ui._dirty)
        self.assertTrue(ui.apply_scene())
        state = ready(lastResult={"requestId": "request-1", "success": True, "message": "Scene applied"})
        state["vision"].update(tokenVision=True, globalIllumination=False)
        ui.refresh(state)
        self.assertEqual(ui.message.get(), "Scene applied")
        self.assertFalse(ui._dirty)

    def test_token_apply_and_view_selection_need_no_mini_mapping(self):
        ui = self.controls()
        self.choose(ui, "blue")
        self.edit(ui, "enabled", True)
        self.edit(ui, "range", "25.5")
        self.assertTrue(ui.apply_token())
        ui._request.assert_called_once_with("vision.token.set", {"tokenId": "blue", "enabled": True, "range": 25.5})
        state = ready(lastResult={"requestId": "request-1", "success": True, "message": "Token applied"})
        state["tokens"][1]["vision"].update(enabled=True, range=25.5)
        ui.refresh(state)
        self.assertTrue(ui.show_view())
        self.assertEqual(ui._request.call_args.args, ("vision.token.select", {"tokenId": "blue"}))

    def test_switching_token_discards_its_draft_without_touching_scene_draft(self):
        ui = self.controls()
        self.edit(ui, "tokenVision", True)
        self.edit(ui, "range", "45")
        self.choose(ui, "blue")
        self.assertEqual(ui.variables["range"].get(), "0")
        self.assertFalse(ui.variables["enabled"].get())
        self.assertEqual(ui._dirty, {"tokenVision"})
        ui._request.assert_not_called()

    def test_token_renames_preserve_id_but_deletion_does_not_retarget_a_stale_click(self):
        ui = self.controls()
        state = ready()
        state["tokens"][0]["tokenName"] = "Scarlet"
        ui.refresh(state)
        self.assertEqual(ui.token.get(), "Scarlet (red)")
        self.edit(ui, "range", "45")
        state["tokens"] = state["tokens"][1:]
        ui._read_status.return_value = state
        self.assertFalse(ui.apply_token())
        ui._request.assert_not_called()
        self.assertEqual(ui._token_id, "blue")
        self.assertEqual(ui.variables["range"].get(), "0")
        self.assertIn("changed", ui.message.get())

    def test_scene_switch_at_click_discards_drafts_and_never_sends_to_new_scene(self):
        ui = self.controls()
        self.edit(ui, "tokenVision", True)
        ui._read_status.return_value = ready(sceneId="scene-b")
        self.assertFalse(ui.apply_scene())
        ui._request.assert_not_called()
        self.assertFalse(ui.variables["tokenVision"].get())
        self.assertEqual(ui._dirty, set())
        self.assertIn("changed", ui.message.get())

    def test_live_observer_capture_connection_pending_and_old_module_guards(self):
        for changes in ({"connected": False}, {"supported": False}, {"pending": True}, {"blocked": True},
                        {"capabilities": []}, {"sceneId": None}, {"outputPaused": True},
                        {"capture": {"recording": True}}, {"capture": {"phase": "Saving"}}):
            with self.subTest(changes=changes):
                ui = self.controls()
                ui._read_status.return_value = ready(**changes)
                self.assertFalse(ui.apply_scene())
                self.assertFalse(ui.apply_token())
                self.assertFalse(ui.show_view())
                ui._request.assert_not_called()
        for enabled, can_change in ((False, True), (True, False)):
            ui = self.controls()
            ui._enabled = enabled
            ui._can_change.return_value = can_change
            self.assertFalse(ui.apply_scene())
            self.assertFalse(ui.apply_token())
            self.assertFalse(ui.show_view())
            ui._request.assert_not_called()

    def test_auth_failure_and_bad_range_never_send(self):
        ui = self.controls()
        self.edit(ui, "range", "nan")
        self.assertFalse(ui.apply_token())
        self.assertIn("sight range", ui.message.get())
        ui._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(ui.apply_scene())
        self.assertEqual(ui.message.get(), "Sign in first")
        ui._request.assert_not_called()

    def test_missing_remote_fields_cannot_reuse_last_known_settings(self):
        ui = self.controls()
        state = ready(vision={})
        state["tokens"][0]["vision"] = {}
        ui.refresh(state)
        self.assertEqual(ui.apply_scene_button.configure.call_args.kwargs["state"], "disabled")
        self.assertEqual(ui.apply_token_button.configure.call_args.kwargs["state"], "disabled")
        ui._read_status.return_value = state
        self.assertFalse(ui.apply_scene())
        self.assertFalse(ui.apply_token())
        ui._request.assert_not_called()

    def test_uncertain_timeout_does_not_apply_or_retry_automatically(self):
        ui = self.controls()
        self.edit(ui, "tokenVision", True)
        self.assertTrue(ui.apply_scene())
        ui.refresh(ready(pending=True, lastResult={"requestId": "old-request", "success": True}))
        self.assertEqual(ui._pending, "request-1")
        ui.refresh(ready(lastResult={"requestId": "request-1", "success": False, "message": "Action timed out"}))
        self.assertIsNone(ui._pending)
        self.assertIn("timed out", ui.message.get())
        self.assertTrue(ui.variables["tokenVision"].get())
        ui._request.assert_called_once()


if __name__ == "__main__":
    unittest.main()
