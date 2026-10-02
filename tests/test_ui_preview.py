import io
import os
import unittest
from unittest.mock import patch

from ui_preview import PreviewPanel, SCENARIOS, main, sample_roster


class PreviewDataTests(unittest.TestCase):
    def test_roster_is_sample_data_and_fresh_each_time(self):
        rows = sample_roster()
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row["id"] for row in rows}), 5)
        self.assertTrue(all("sample" in row["token"] for row in rows))
        self.assertEqual(rows[-1]["scanStatus"], "Needs scan")
        rows[0]["name"] = "changed"
        self.assertEqual(sample_roster()[0]["name"], "Red")

    def test_invalid_timeout_never_opens_window(self):
        with patch("ui_preview.PreviewPanel") as panel, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(["--close-after", "nan"])
            panel.assert_not_called()


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class PreviewUITests(unittest.TestCase):
    def test_actual_panels_use_only_sample_state_and_disable_hardware(self):
        from tkinter import ttk
        from ui_preview import descendants
        with patch("control_panel._save_config_patch", side_effect=AssertionError("preview must not save")):
            panel = PreviewPanel()
            try:
                panel.root.update()
                self.assertIn("sample", panel.var_mode.get())
                self.assertEqual(len(panel._pos_rows), 5)
                for name in SCENARIOS:
                    panel.set_scenario(name)
                    self.assertEqual(panel._scenario.get(), name)
                    self.assertIn("sample", panel.var_delivery.get())
                panel.set_scenario("Missing markers")
                self.assertIn("2/4", panel.var_markers.get())
                panel.set_scenario("Disconnected")
                self.assertIn("disconnected", panel.var_foundry.get())
                panel.set_scenario("Move failed")
                panel._retry_moves_btn.invoke()
                self.assertEqual(panel._scenario.get(), "Move pending")
                self.assertFalse(panel.pop_actions()["retry_moves"])
                panel._show_mini_library()
                panel.root.update()
                self.assertEqual(len(panel._library_tree.get_children()), 5)
                for root in (panel.root, panel._library_window):
                    for widget in descendants(root):
                        if isinstance(widget, ttk.Button) and widget.cget("text") in (
                                "Calibrate", "Auto", "Scan selected", "Full brightness scan", "Calibrate Minis"):
                            self.assertIn("disabled", widget.state())
                with self.assertRaises(ValueError):
                    panel.set_scenario("Not a state")
                panel._act_exit()
                self.assertTrue(panel.pop_actions()["exit"])
            finally:
                panel.root.destroy()


if __name__ == "__main__":
    unittest.main()
