"""Control-panel construction and responsive behavior without a Tcl/Tk window."""

from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import control_panel as panel_module
import map_view
import vtt_controls


class Variable:
    def __init__(self, *args, value=None, **kwargs):
        self.value = value if value is not None else (args[1] if len(args) > 1 else "")

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class Widget:
    def __init__(self, parent=None, **options):
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)
        self.options = options
        self.layout = {}
        self.bindings = {}
        self.class_bindings = {}
        self.columns, self.rows = {}, {}
        self.tab_items = []
        self.tags = (str(id(self)), type(self).__name__, "all")

    def pack(self, **options):
        self.layout = {"manager": "pack", **options}

    def grid(self, **options):
        self.layout = {"manager": "grid", **options}

    def configure(self, **options):
        self.options.update(options)

    def columnconfigure(self, index, **options):
        self.columns[index] = options

    def rowconfigure(self, index, **options):
        self.rows[index] = options

    def bind(self, event, callback, **kwargs):
        self.bindings[event] = callback

    def bind_class(self, tag, event, callback):
        self.class_bindings[tag, event] = callback

    def bindtags(self, tags=None):
        if tags is not None:
            self.tags = tags
        return self.tags

    def winfo_children(self):
        return self.children

    def add(self, page, **options):
        self.tab_items.append((page, options))

    def title(self, value):
        self.title_value = value

    def geometry(self, value):
        self.geometry_value = value

    def minsize(self, *values):
        self.minimum = values

    def resizable(self, *values):
        self.resizable_value = values

    def protocol(self, name, callback):
        self.protocol_value = (name, callback)

    def winfo_screenwidth(self):
        return 1440

    def winfo_screenheight(self):
        return 900

    def after(self, _delay, callback):
        callback()

    def set(self, *values):
        self.scroll_values = values

    def invoke(self):
        return self.options["command"]()


class Canvas(Widget):
    def create_window(self, *args, **options):
        self.window_options = options
        return 1

    def itemconfigure(self, _window, **options):
        self.window_options.update(options)

    def bbox(self, _tag):
        return (0, 0, 600, 900)

    def yview(self, *values):
        self.view_values = values

    def yview_scroll(self, *values):
        self.view_scroll = values


class Treeview(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entries = {}
        self.column_options, self.headings = {}, {}

    def column(self, name, **options):
        self.column_options[name] = options

    def heading(self, name, **options):
        self.headings[name] = options

    def get_children(self):
        return list(self.entries)

    def exists(self, name):
        return name in self.entries

    def insert(self, _parent, _index, *, iid, **options):
        self.entries[iid] = options

    def item(self, iid, **options):
        self.entries[iid].update(options)

    def delete(self, iid):
        del self.entries[iid]

    def yview(self, *values):
        self.view_values = values


class VTT(Widget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_controls = object()


def descendants(widget):
    return [widget, *(child for item in widget.children for child in descendants(item))]


class ControlPanelLayoutTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.multiple(panel_module.tk,
            Tk=Widget, Toplevel=Widget, Canvas=Canvas,
            StringVar=Variable, BooleanVar=Variable, IntVar=Variable))
        self.patches.enter_context(patch.multiple(panel_module.ttk,
            Frame=Widget, Label=Widget, Notebook=Widget, Scrollbar=Widget,
            Entry=Widget, Button=Widget, Checkbutton=Widget, Separator=Widget,
            Scale=Widget, Treeview=Treeview))
        self.patches.enter_context(patch.object(map_view, "MapView", Widget))
        self.patches.enter_context(patch.object(vtt_controls, "VTTControls", VTT))
        self.panel = panel_module.ControlPanel(mode="foundry")

    def widget_with(self, **options):
        return next(widget for widget in descendants(self.panel.root)
                    if all(widget.options.get(key) is value if key == "textvariable"
                           else widget.options.get(key) == value for key, value in options.items()))

    def test_initial_and_minimum_sizes_fit_logical_screen_bounds(self):
        bounds = panel_module.control_panel_window_bounds
        self.assertEqual(bounds(1440, 900), (800, 760, 680, 520))
        self.assertEqual(bounds(3840, 2160), (800, 760, 680, 520))
        for width, height in ((1280, 800), (800, 600), (640, 480)):
            requested_w, requested_h, min_w, min_h = bounds(width, height)
            self.assertLessEqual(requested_w, width - 64)
            self.assertLessEqual(requested_h, height - 100)
            self.assertLessEqual(min_w, requested_w)
            self.assertLessEqual(min_h, requested_h)
        self.assertEqual(self.panel.root.geometry_value, "800x760")
        self.assertEqual(self.panel.root.minimum, (680, 520))

    def test_notebook_fills_root_without_a_canvas_ancestor(self):
        notebook = self.panel.tabs
        outer = notebook.parent
        self.assertIs(outer.parent, self.panel.root)
        self.assertEqual(outer.layout, {"manager": "pack", "fill": "both", "expand": True})
        self.assertEqual(outer.rows[1]["weight"], 1)
        self.assertEqual(outer.columns[0]["weight"], 1)
        self.assertEqual(notebook.layout["sticky"], "nsew")
        self.assertEqual([options["text"] for _, options in notebook.tab_items],
                         ["Session", "Diagnostics", "VTT Controls", "Map"])
        self.assertFalse(isinstance(outer, Canvas))

    def test_hints_and_access_status_remain_outside_notebook_pages(self):
        self.panel.set_hint("Ring sample saved: red10")
        self.panel.set_access_status("Developer mode — login bypassed")
        hint = self.widget_with(textvariable=self.panel.var_hint)
        self.assertIs(hint.parent, self.panel._footer)
        self.assertIs(self.panel._access_label.parent, self.panel._footer)
        self.assertIs(self.panel._footer.parent, self.panel.tabs.parent)
        self.assertEqual(self.panel._footer.layout["row"], 2)
        self.assertEqual(self.panel.var_hint.get(), "Ring sample saved: red10")
        hint.bindings["<Configure>"](SimpleNamespace(width=370))
        self.assertEqual(hint.options["wraplength"], 370)

    def test_status_is_paired_and_camera_message_wraps_without_truncation(self):
        for left, right in ((self.panel.var_mode, self.panel.var_fps),
                            (self.panel.var_lock, self.panel.var_foundry),
                            (self.panel.var_markers, self.panel.var_missing)):
            one, two = self.widget_with(textvariable=left), self.widget_with(textvariable=right)
            self.assertEqual(one.layout["row"], two.layout["row"])
            self.assertEqual((one.layout["column"], two.layout["column"]), (0, 1))
        camera = self.widget_with(textvariable=self.panel.var_camera_calibration)
        self.assertEqual(camera.layout["columnspan"], 2)
        metadata = {"profileName": "iPhone via OBS — Wide 1x with a long profile name",
                    "applied": False, "reason": "Camera geometry changed; choose the matching profile",
                    "frameSize": [1920, 1080]}
        self.panel.set_camera_calibration_status(metadata)
        camera.bindings["<Configure>"](SimpleNamespace(width=530))
        self.assertEqual(camera.options["wraplength"], 530)
        camera.bindings["<Configure>"](SimpleNamespace(width=1))
        self.assertEqual(camera.options["wraplength"], 530)
        self.assertEqual(self.panel.var_camera_calibration.get(),
                         panel_module.format_camera_calibration_status(metadata))

    def test_self_hosted_mode_preserves_its_four_tabs(self):
        panel = panel_module.ControlPanel(mode="self_hosted")
        self.assertIsNone(panel.vtt_controls)
        self.assertEqual([options["text"] for _, options in panel.tabs.tab_items],
                         ["Session", "Scans", "Diagnostics", "Map"])

    def test_session_and_scan_actions_preserve_commands(self):
        self.panel = panel_module.ControlPanel(mode="self_hosted")
        for name in ("Recapture BG", "Stop session", "Dump State", "Calibrate Minis"):
            self.widget_with(text=name).invoke()
        self.widget_with(textvariable=self.panel._rec_btn_var).invoke()
        self.panel._band_name_var.set("red10")
        self.widget_with(text="Calibrate").invoke()
        actions = self.panel.pop_actions()
        for name in ("recapture_bg", "exit", "dump_state", "calibrate_minis", "toggle_recording"):
            self.assertTrue(actions[name])
        self.assertEqual(actions["calibrate_band"], {"name": "red10"})
        self.widget_with(text="Auto").invoke()
        self.assertEqual(self.panel.pop_actions()["calibrate_band"], {"name": None, "auto": True})
        self.assertEqual(self.widget_with(text="Mini Library").options["command"],
                         self.panel._show_mini_library)

    def test_foundry_session_offers_rebinding_instead_of_legacy_scan_workflow(self):
        self.widget_with(text="Choose positions again").invoke()
        self.assertTrue(self.panel.pop_actions()["recapture_bg"])
        texts = [widget.options.get("text") for widget in descendants(self.panel.root)]
        self.assertNotIn("Mini Library", texts)
        self.assertNotIn("Recapture BG", texts)
        self.panel.set_delivery_status({"message": "Ready", "retryAvailable": True})
        self.panel._retry_moves_btn.invoke()
        self.assertTrue(self.panel.pop_actions()["retry_moves"])

    def test_only_long_diagnostics_page_scrolls_and_resize_updates_content_width(self):
        canvases = [widget for widget in descendants(self.panel.root) if isinstance(widget, Canvas)]
        self.assertEqual(len(canvases), 1)
        canvas = canvases[0]
        diagnostics = next(widget for widget, options in self.panel.tabs.tab_items
                           if options["text"] == "Diagnostics")
        self.assertIs(canvas.parent, diagnostics)
        self.assertNotIn("<MouseWheel>", self.panel.root.bindings)
        canvas.bindings["<Configure>"](SimpleNamespace(width=712))
        self.assertEqual(canvas.window_options["width"], 712)
        canvas.bindings["<Configure>"](SimpleNamespace(width=0))
        self.assertEqual(canvas.window_options["width"], 1)
        content = canvas.children[0]
        callback = next(callback for (tag, event), callback in content.class_bindings.items()
                        if event == "<MouseWheel>")
        self.assertEqual(callback(SimpleNamespace(delta=-1)), "break")
        self.assertEqual(canvas.view_scroll, (1, "units"))
        self.assertFalse(any(tag.startswith("ControlPanelScroll") for tag in self.panel.tabs.tags))

    def test_position_table_updates_in_place_and_removes_outdated_minis(self):
        self.panel.update_positions({"red10": "L8", "blue20": None})
        table = self.panel._pos_tree
        self.assertEqual(table.entries["red10"]["values"], ("red10", "L8"))
        self.assertEqual(table.entries["blue20"]["values"], ("blue20", "—"))
        self.panel.update_positions({"red10": "Q11"})
        self.assertEqual(list(table.entries), ["red10"])
        self.assertEqual(table.entries["red10"]["values"], ("red10", "Q11"))
        self.assertTrue(table.column_options["mini"]["stretch"])
        self.assertEqual(self.panel._pos_table.layout["fill"], "both")


if __name__ == "__main__":
    unittest.main()
