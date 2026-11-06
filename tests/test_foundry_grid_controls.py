"""Headless checks for desktop grid editing and acknowledgement handling."""

import unittest
from unittest.mock import Mock

from foundry_grid_controls import FoundryGridControls, GRID_FIELDS, parse_grid_settings


class Variable:
    def __init__(self, value=""):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


def ready_status(**overrides):
    return {
        "connected": True, "supported": True, "pending": False, "message": "Ready",
        "settings": {"targetGridInches": 1.0, "displayDiagonalInches": 55.0,
                     "physicalGridCalibration": 1.0},
        **overrides,
    }


class GridControlTests(unittest.TestCase):
    def make_controls(self):
        # Exercise the widget controller without starting Tk or a network relay.
        controls = FoundryGridControls.__new__(FoundryGridControls)
        controls._require_access = Mock()
        controls._read_status = Mock(return_value=ready_status())
        controls._request = Mock(return_value=True)
        controls._access_enabled = True
        controls._editing_enabled = None
        controls._last_refresh = float("-inf")
        controls._last_status = {}
        controls._remote_settings = {}
        controls._dirty = set()
        controls._submitted = None
        controls._syncing = False
        controls._notice = ""
        controls.variables = {name: Variable(f"{default:.{places}f}")
                              for name, _label, _lo, _hi, _step, default, places in GRID_FIELDS}
        controls._fields = [Mock(), Mock(), Mock()]
        controls.apply_button = Mock()
        controls.one_inch_button = Mock()
        controls.message = Variable()
        controls.refresh(ready_status())
        return controls

    def edit(self, controls, name, value):
        controls.variables[name].set(value)
        controls._field_edited(name)

    def test_remote_refresh_preserves_dirty_fields_and_updates_other_settings(self):
        controls = self.make_controls()
        self.edit(controls, "targetGridInches", "1.25")
        remote = ready_status(settings={"targetGridInches": 0.75, "displayDiagonalInches": 65.0,
                                       "physicalGridCalibration": 1.02})
        controls.refresh(remote)
        self.assertEqual(controls.variables["targetGridInches"].get(), "1.25")
        self.assertEqual(controls.variables["displayDiagonalInches"].get(), "65.0")
        self.assertEqual(controls.variables["physicalGridCalibration"].get(), "1.020")

    def test_apply_checks_access_and_retains_draft_until_acknowledged(self):
        controls = self.make_controls()
        self.edit(controls, "targetGridInches", "1.25")
        pending = ready_status(pending=True, message="Applying grid size…")
        controls._read_status.side_effect = [ready_status(), pending]
        self.assertTrue(controls.apply())
        controls._require_access.assert_called_once()
        controls._request.assert_called_once_with({"targetGridInches": 1.25,
            "displayDiagonalInches": 55.0, "physicalGridCalibration": 1.0}, apply=True)
        self.assertEqual(controls.variables["targetGridInches"].get(), "1.25")
        self.assertFalse(controls._editing_enabled)
        self.assertIn("Applying", controls.message.get())
        acknowledged = ready_status(settings={"targetGridInches": 1.25,
            "displayDiagonalInches": 55.0, "physicalGridCalibration": 1.0})
        controls.refresh(acknowledged)
        self.assertEqual(controls._dirty, set())
        self.assertTrue(controls._editing_enabled)

    def test_failed_apply_keeps_edits_for_retry_and_shows_bridge_error(self):
        controls = self.make_controls()
        self.edit(controls, "targetGridInches", "1.25")
        controls._read_status.side_effect = [ready_status(), ready_status(pending=True)]
        self.assertTrue(controls.apply())
        controls.refresh(ready_status(message="Foundry did not acknowledge the grid size."))
        self.assertEqual(controls.variables["targetGridInches"].get(), "1.25")
        self.assertIn("targetGridInches", controls._dirty)
        self.assertIn("did not acknowledge", controls.message.get())
        self.assertTrue(controls._editing_enabled)

    def test_two_panels_sync_without_overwriting_each_others_drafts(self):
        first, second = self.make_controls(), self.make_controls()
        state = ready_status()
        requests = []

        def request(settings, *, apply):
            requests.append(settings)
            state.update(pending=True, message="Applying TV grid size…")
            return True

        for controls in (first, second):
            controls._read_status.side_effect = lambda: state
            controls._request.side_effect = request
        self.edit(first, "targetGridInches", "1.25")
        self.edit(second, "displayDiagonalInches", "65")
        self.assertTrue(first.apply())
        second.refresh(state)
        self.assertFalse(second._editing_enabled)
        state.update(pending=False, settings=requests[-1], message="Applied")
        first.refresh(state)
        second.refresh(state)
        self.assertEqual(first._dirty, set())
        self.assertEqual(second.variables["targetGridInches"].get(), "1.25")
        self.assertEqual(second.variables["displayDiagonalInches"].get(), "65")
        self.assertTrue(second.apply())
        self.assertEqual(requests[-1], {"targetGridInches": 1.25,
            "displayDiagonalInches": 65.0, "physicalGridCalibration": 1.0})
        state.update(pending=False, settings=requests[-1], message="Applied")
        first.refresh(state)
        second.refresh(state)
        self.assertEqual(first.variables["displayDiagonalInches"].get(), "65.0")
        self.assertEqual(second._dirty, set())

    def test_connection_messages_take_priority_over_old_validation_notice(self):
        controls = self.make_controls()
        self.edit(controls, "targetGridInches", "invalid")
        self.assertFalse(controls.apply())
        controls.refresh(ready_status(supported=False,
            message="Reload Foundry with the updated Sarween module."))
        self.assertEqual(controls.message.get(), "Reload Foundry with the updated Sarween module.")

    def test_disconnected_unsupported_pending_or_capture_cannot_send(self):
        for overrides in ({"connected": False}, {"supported": False}, {"pending": True},
                          {"blocked": True, "message": "Finish guided capture first."}):
            with self.subTest(overrides=overrides):
                controls = self.make_controls()
                controls._read_status.return_value = ready_status(**overrides)
                self.assertFalse(controls.apply())
                controls._request.assert_not_called()
                self.assertFalse(controls._editing_enabled)

    def test_access_denial_and_invalid_input_never_send(self):
        controls = self.make_controls()
        controls._require_access.side_effect = PermissionError("Sign in first")
        self.assertFalse(controls.apply())
        controls._request.assert_not_called()
        self.assertEqual(controls.message.get(), "Sign in first")
        controls._require_access.side_effect = None
        self.edit(controls, "targetGridInches", "not a number")
        self.assertFalse(controls.apply())
        controls._request.assert_not_called()
        self.assertIn("Enter a number", controls.message.get())
        controls.set_enabled(False)
        self.assertFalse(controls._editing_enabled)

    def test_one_inch_preset_preserves_tv_calibration(self):
        controls = self.make_controls()
        self.edit(controls, "displayDiagonalInches", "65")
        self.edit(controls, "physicalGridCalibration", "1.08")
        self.edit(controls, "targetGridInches", "1.25")
        # Real Tk traces this change; mark it dirty to emulate the preset's trace.
        controls._dirty.add("targetGridInches")
        self.assertTrue(controls.use_one_inch())
        controls._request.assert_called_once_with({"targetGridInches": 1.0,
            "displayDiagonalInches": 65.0, "physicalGridCalibration": 1.08}, apply=True)

    def test_invalid_remote_numbers_do_not_replace_the_editor(self):
        controls = self.make_controls()
        controls.refresh(ready_status(settings={"targetGridInches": float("nan"),
            "displayDiagonalInches": True, "physicalGridCalibration": -5}))
        self.assertEqual(controls.variables["targetGridInches"].get(), "1.00")
        self.assertEqual(controls.variables["displayDiagonalInches"].get(), "55.0")
        self.assertEqual(controls.variables["physicalGridCalibration"].get(), "1.000")

    def test_numeric_limits_reject_nonfinite_or_out_of_range_requests(self):
        initial = ready_status()["settings"]
        for key, value in (("targetGridInches", 0.1), ("targetGridInches", float("nan")),
                           ("displayDiagonalInches", 0), ("physicalGridCalibration", float("inf")),
                           ("physicalGridCalibration", 2)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                parse_grid_settings({**initial, key: value})


if __name__ == "__main__":
    unittest.main()
