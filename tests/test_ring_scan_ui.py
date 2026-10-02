"""Opt-in native UI checks; never open windows during normal test discovery."""

import os
import unittest


@unittest.skipUnless(os.environ.get("SARWEEN_GUI_TESTS") == "1", "Native UI checks are opt-in")
class RingScanUITest(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        self.root.destroy()

    def test_picker_requires_selection_and_explicit_confirmation(self):
        import cv2
        import numpy as np
        from tkinter import ttk
        from ring_scan_dialog import show_ring_scan

        image = np.full((720, 1280, 3), 240, dtype=np.uint8)
        cv2.circle(image, (500, 300), 24, (0, 0, 255), -1)
        cv2.circle(image, (500, 300), 16, (60, 60, 130), -1)
        samples = []
        window = show_ring_scan(self.root, "red10", image, samples.append)
        self.root.update()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        use = next(widget for widget in descendants(window)
                   if isinstance(widget, ttk.Button) and widget.cget("text") == "Use ring sample")
        self.assertIn("disabled", use.state())
        view = window.winfo_children()[0]
        view.event_generate("<Button-1>", x=round(521 * int(view.cget("width")) / 1280),
                            y=round(300 * int(view.cget("height")) / 720))
        self.root.update()
        self.assertNotIn("disabled", use.state())
        self.assertEqual(samples, [])
        self.assertLess(window.winfo_width(), self.root.winfo_screenwidth())
        self.assertLess(window.winfo_height(), self.root.winfo_screenheight() - 30)
        use.invoke()
        self.assertEqual(samples[0]["name"], "red10")
        self.assertGreater(samples[0]["sample_lab"][1], 60)
        self.assertFalse(samples[0]["replace_colors"])

    def test_control_panel_delivery_status_survives_fps_refresh_and_retries(self):
        from control_panel import ControlPanel
        panel = ControlPanel(mode="foundry", tk_root=self.root)
        panel.set_status(foundry_connected=True)
        before = panel.var_foundry.get()
        panel.set_status(fps=15)
        self.assertEqual(panel.var_foundry.get(), before)
        panel.set_delivery_status({"message": "Move failed: red10", "retryAvailable": True})
        panel._retry_moves_btn.invoke()
        self.assertTrue(panel.pop_actions()["retry_moves"])
        self.assertFalse(panel.pop_actions()["retry_moves"])
        self.root.update()
        self.assertLessEqual(panel.root.winfo_height(), self.root.winfo_screenheight() - 60)


if __name__ == "__main__":
    unittest.main()
