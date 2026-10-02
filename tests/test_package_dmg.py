from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_dmg import package


class DmgTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.app = self.root / "Sarween.app"
        (self.app / "Contents/MacOS").mkdir(parents=True)
        (self.app / "Contents/MacOS/Sarween").write_bytes(b"fake executable")
        (self.app / "Contents/Info.plist").write_bytes(plistlib.dumps({
            "CFBundleExecutable": "Sarween", "CFBundleShortVersionString": "0.1.0"}))
        self.output = self.root / "installer.dmg"

    def test_packages_app_and_applications_link_without_touching_source(self):
        def run(command, **kwargs):
            self.assertEqual(kwargs["timeout_seconds"], 15)
            if command[1] == "create":
                volume = Path(command[command.index("-srcfolder") + 1])
                self.assertEqual((volume / "Applications").readlink(), Path("/Applications"))
                self.assertEqual((volume / "Sarween.app/Contents/MacOS/Sarween").read_bytes(), b"fake executable")
                Path(command[-1]).write_bytes(b"test dmg")
        with patch("scripts.package_dmg.run_bounded", side_effect=run) as runner:
            result = package(self.app, self.output, timeout_seconds=15)
        self.assertEqual(runner.call_count, 2)
        self.assertEqual(self.output.read_bytes(), b"test dmg")
        self.assertEqual(result["version"], "0.1.0")
        self.assertEqual(len(result["sha256"]), 64)
        self.assertEqual(list(self.root.glob("sarween-dmg-*")), [])
        self.assertEqual((self.app / "Contents/MacOS/Sarween").read_bytes(), b"fake executable")

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"previous installer")
        with patch("scripts.package_dmg.run_bounded") as run:
            with self.assertRaisesRegex(ValueError, "already exists"):
                package(self.app, self.output)
        run.assert_not_called()
        self.assertEqual(self.output.read_bytes(), b"previous installer")

    def test_failed_verification_leaves_no_installer_or_staging(self):
        def run(command, **_kwargs):
            if command[1] == "create":
                Path(command[-1]).write_bytes(b"bad dmg")
            else:
                raise RuntimeError("verify failed")
        with patch("scripts.package_dmg.run_bounded", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "verify failed"):
                package(self.app, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob("sarween-dmg-*")), [])

    def test_invalid_app_or_timeout_stops_before_copying(self):
        with patch("scripts.package_dmg.shutil.copytree") as copy:
            for timeout in (0, -1, float("nan")):
                with self.assertRaises(ValueError):
                    package(self.app, self.output, timeout_seconds=timeout)
            (self.app / "Contents/MacOS/Sarween").unlink()
            with self.assertRaisesRegex(ValueError, "executable"):
                package(self.app, self.output)
        copy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
