"""Launcher creation is file-only; these tests never open an application."""

import os
from pathlib import Path
import plistlib
import shlex
import sys
import tempfile
import unittest

from scripts.build_video_audit_app import BUNDLE_ID, build_app


class AuditLauncherTests(unittest.TestCase):
    def test_app_opens_standalone_entry_with_literal_quoted_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "Source with spaces ' $literal"
            source.mkdir()
            (source / "audit_videos.py").write_text("raise SystemExit(0)\n")
            destination = Path(folder) / "Video Review.app"
            self.assertEqual(build_app(destination, source=source), destination)
            info = plistlib.loads((destination / "Contents/Info.plist").read_bytes())
            self.assertEqual(info["CFBundleIdentifier"], BUNDLE_ID)
            entry = destination / "Contents/MacOS" / info["CFBundleExecutable"]
            self.assertTrue(os.access(entry, os.X_OK))
            lines = entry.read_text().splitlines()
            self.assertEqual(shlex.split(next(line for line in lines if line.startswith("cd "))),
                             ["cd", str(source.resolve())])
            command = shlex.split(next(line for line in lines if line.startswith("exec ")))
            self.assertEqual(command[:3], ["exec", str(Path(sys.executable).resolve()),
                                          str(source.resolve() / "audit_videos.py")])
            self.assertNotIn("main.py", entry.read_text())

    def test_existing_unrelated_app_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            (source / "audit_videos.py").write_text("")
            destination = Path(folder) / "Video Review.app"
            info_file = destination / "Contents/Info.plist"
            info_file.parent.mkdir(parents=True)
            original = plistlib.dumps({"CFBundleIdentifier": "some.other.application"})
            info_file.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "another application"):
                build_app(destination, source=source)
            self.assertEqual(info_file.read_bytes(), original)

    def test_missing_interpreter_and_symbolic_link_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source / "audit_videos.py").write_text("")
            with self.assertRaises(ValueError):
                build_app(source / "Review.app", source=source, python=source / "missing-python")
            target = source / "target"
            target.mkdir()
            app = source / "Review.app"
            app.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                build_app(app, source=source)
            self.assertFalse((target / "Contents").exists())


if __name__ == "__main__":
    unittest.main()
