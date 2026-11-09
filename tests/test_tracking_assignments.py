"""Player binding confirmation uses the existing scene-scoped VTT commands."""
import unittest
from unittest.mock import Mock, patch

from tests import test_vtt_controls as models
import vtt_controls


class TrackingAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.controls = models.VTTControlTests().controls()
        ui = self.controls
        ui._tracking_context = None
        ui._tracking_rows = []
        ui._tracking_read_only = False
        ui._tracking_confirmed = {}
        ui._tracking_acknowledged = {}
        ui._tracking_queue = []
        ui._tracking_pending = None
        ui._legacy_assignment_section = Mock()
        ui._tracking_assignment_section = Mock()
        ui._tracking_assignment_message = models.Variable()
        ui._tracking_confirm_button = Mock()
        def build(rows):
            bindings = ui._last_status.get("miniBindings") or {}
            ui._tracking_rows = []
            for row in rows:
                token_id = row.get("tokenId") or bindings.get(row["miniId"])
                label = next((label for label, identity in ui._token_labels.items() if identity == token_id), "")
                ui._tracking_rows.append({**row, "selected_id": token_id,
                    "token": models.Variable(label), "combo": Mock()})
        ui._build_tracking_rows = Mock(side_effect=build)
        self.state = models.ready_status(sceneId="scene", miniBindings={"mini1": "red-token", "mini2": "blue-token"})
        ui.refresh(self.state)
        ui._read_status.return_value = self.state
        ui._request.side_effect = ["assign-1", "assign-2", "assign-3", "assign-4"]
        ui.set_tracking_minis(["mini1", "mini2"], 1, "scene")

    def acknowledge(self, request, mini, token, **changes):
        self.state = {**self.state, "pending": False,
            "miniBindings": {**self.state["miniBindings"], mini: token},
            "lastResult": {"requestId": request, "sceneId": "scene", "success": True, **changes}}
        self.controls._read_status.return_value = self.state
        self.controls.refresh(self.state)

    def confirm_all(self):
        self.assertTrue(self.controls.confirm_tracking_assignments())
        self.acknowledge("assign-1", "mini1", "red-token")
        self.acknowledge("assign-2", "mini2", "blue-token")

    def test_saved_hints_are_not_confirmed_until_every_ack_matches(self):
        ui = self.controls
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {})
        self.assertTrue(ui.confirm_tracking_assignments())
        ui._request.assert_called_once_with("assignment.save", {"miniId": "mini1", "tokenId": "red-token"})
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {})
        self.acknowledge("unrelated", "mini1", "red-token")
        self.assertEqual(ui._request.call_count, 1)
        self.acknowledge("assign-1", "mini1", "red-token")
        self.assertEqual(ui._request.call_count, 2)
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {})
        self.acknowledge("assign-2", "mini2", "blue-token")
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {"mini1": "red-token", "mini2": "blue-token"})

    def test_new_starting_layout_requires_fresh_confirmation(self):
        self.confirm_all()
        self.controls.set_tracking_minis(["mini1", "mini2"], 2, "scene")
        self.assertEqual(self.controls.get_confirmed_tracking_assignments(), {})
        self.assertEqual(self.controls._tracking_acknowledged, {})

    def test_editing_one_row_invalidates_the_old_confirmation(self):
        self.confirm_all()
        ui = self.controls
        row = ui._tracking_rows[0]
        row["token"].set(next(label for label, identity in ui._token_labels.items() if identity == "blue-token"))
        ui._tracking_assignment_edited(row)
        ui.refresh(self.state)
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {})
        self.assertFalse(ui.confirm_tracking_assignments())
        self.assertIn("different", ui._tracking_assignment_message.get())

    def test_disconnect_scene_change_and_external_remap_clear_confirmation(self):
        for update in ({"connected": False}, {"sceneId": "other-scene"},
                       {"miniBindings": {"mini1": "blue-token", "mini2": "red-token"}}):
            with self.subTest(update=update):
                self.setUp()
                self.confirm_all()
                self.controls.refresh({**self.state, **update})
                self.assertEqual(self.controls.get_confirmed_tracking_assignments(), {})

    def test_failed_wrong_scene_or_wrong_binding_ack_never_confirms(self):
        for changes in ({"success": False, "message": "Denied"}, {"sceneId": "other-scene"}):
            with self.subTest(changes=changes):
                self.setUp()
                self.controls.confirm_tracking_assignments()
                self.acknowledge("assign-1", "mini1", "red-token", **changes)
                self.assertEqual(self.controls.get_confirmed_tracking_assignments(), {})
                self.assertEqual(self.controls._request.call_count, 1)
        self.setUp()
        self.controls.confirm_tracking_assignments()
        self.acknowledge("assign-1", "mini1", "blue-token")
        self.assertEqual(self.controls.get_confirmed_tracking_assignments(), {})

    def test_read_only_context_and_old_module_cannot_confirm(self):
        ui = self.controls
        ui.set_tracking_minis(["mini1", "mini2"], 1, "scene", read_only=True)
        self.assertFalse(ui.confirm_tracking_assignments())
        ui._request.assert_not_called()
        ui.set_tracking_minis(["mini1", "mini2"], 2, "scene")
        ui._read_status.return_value = models.ready_status(sceneId="scene")
        self.assertFalse(ui.confirm_tracking_assignments())
        self.assertIn("Reload Foundry", ui._tracking_assignment_message.get())

    def test_legacy_capture_roster_is_not_replaced_by_player_ids(self):
        self.assertEqual([row["miniId"] for row in self.controls._capture_rows[:2]], ["red10", "blue"])
        self.assertEqual([row["miniId"] for row in self.controls._tracking_rows], ["mini1", "mini2"])

    def test_assignment_form_builds_six_distinct_click_order_rows(self):
        ui = self.controls
        ui._tracking_assignment_section.winfo_children.return_value = []
        rows = [{"miniId": f"mini{index}", "name": f"Mini {index}"} for index in range(1, 7)]
        variables = lambda *_args, **_kwargs: models.Variable(_args[1])
        with patch.object(vtt_controls.tk, "StringVar", side_effect=variables), \
                patch.object(vtt_controls.ttk, "Label") as labels, \
                patch.object(vtt_controls.ttk, "Combobox") as combos, \
                patch.object(vtt_controls.ttk, "Button") as buttons, \
                patch.object(vtt_controls, "_wrapped_label"):
            vtt_controls.VTTControls._build_tracking_rows(ui, rows)
        self.assertEqual([call.kwargs["text"] for call in labels.call_args_list],
                         [f"Mini {index}" for index in range(1, 7)])
        self.assertEqual([row["miniId"] for row in ui._tracking_rows],
                         [f"mini{index}" for index in range(1, 7)])
        self.assertEqual(combos.call_count, 6)
        self.assertEqual(buttons.call_args.kwargs["text"], "Confirm assignments")
        self.assertEqual(buttons.call_args.kwargs["command"], ui.confirm_tracking_assignments)

    def test_access_revocation_stops_the_batch_before_another_assignment(self):
        ui = self.controls
        self.assertTrue(ui.confirm_tracking_assignments())
        ui._require_access.side_effect = PermissionError("Access expired")
        self.acknowledge("assign-1", "mini1", "red-token")
        self.assertEqual(ui._request.call_count, 1)
        self.assertEqual(ui.get_confirmed_tracking_assignments(), {})
        self.assertIn("Access expired", ui._tracking_assignment_message.get())
