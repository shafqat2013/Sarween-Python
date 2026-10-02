import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import build_bundle as builder


FAKE_BUILDER = '''import json, os, pathlib, sys, time
source = pathlib.Path.cwd()
assert source.name == "source"
assert (source / "main.py").read_text() == "# uncommitted entry point\\n"
assert (source / "setup.py").read_text() == "# bundle-specific setup\\n"
assert (source / "tracking_engine.py").read_text() == "# new untracked source\\n"
dist = pathlib.Path(sys.argv[sys.argv.index("--distpath") + 1])
trace = pathlib.Path(os.environ["SARWEEN_TEST_BUILD_TRACE"])
trace.write_text(json.dumps({"cwd": str(source), "cache": os.environ["PYINSTALLER_CONFIG_DIR"]}))
if os.environ.get("SARWEEN_TEST_BUILD_MODE") == "timeout":
    time.sleep(60)
app = dist / "Sarween.app"
app.mkdir(parents=True)
(app / "built.txt").write_text("test build")
if os.environ.get("SARWEEN_TEST_BUILD_MODE") == "fail":
    sys.exit(17)
'''


class BundleBuildTest(unittest.TestCase):
    def test_unconfigured_alpha_cannot_produce_a_distributable_build(self):
        self.write("auth_config.json", "{}")
        with patch.object(builder, "run_pyinstaller") as compile_app:
            with self.assertRaisesRegex(ValueError, "not configured"):
                builder.build(self.root, self.root / "dist")
            compile_app.assert_not_called()

    def test_server_secrets_cannot_be_added_as_bundle_data(self):
        for name in (".env", ".env.production", "alpha.private.pem", "signing.key", "supabase/secret.txt"):
            with self.subTest(name=name):
                self.write(name, "private-test-value")
                with self.assertRaisesRegex(ValueError, "must not be bundled"):
                    builder.local_path(self.root, name)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "project with spaces"
        self.root.mkdir()
        self.write("main.py", "# uncommitted entry point\n")
        self.write("setup.py", "# user's current development setup\n")
        self.write("tracking_engine.py", "# new untracked source\n")
        self.write("app_bundle_overrides/setup.py", "# bundle-specific setup\n")
        self.write("app_bundle_overrides/sarween.spec", 'added_files = [("profile.json", "."), ("maps", "maps")]\n')
        self.write("sarween.spec", "# original spec must survive\n")
        self.write("profile.json", "{}")
        self.write("maps/demo.png", "fake image data")
        self.write(".git/index", "preserve staged git changes")
        self.write("build/old.txt", "previous build")
        self.write("dist/Sarween.app/old.txt", "previous app")
        self.write("sarween_rec_private.mp4", "do not stage videos")
        self.write("auto-wall-main/unrelated.py", "do not stage downloaded source")
        self.write("PyInstaller.py", FAKE_BUILDER)
        self.trace = Path(self.temporary.name) / "trace.json"
        env = patch.dict(os.environ, {"SARWEEN_TEST_BUILD_TRACE": str(self.trace), "SARWEEN_TEST_BUILD_MODE": "success"})
        env.start()
        self.addCleanup(env.stop)

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)

    def original_hashes(self):
        return {path.relative_to(self.root): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.root.rglob("*") if path.is_file()}

    def assert_inputs_unchanged(self, original):
        for relative, digest in original.items():
            self.assertEqual(hashlib.sha256((self.root / relative).read_bytes()).hexdigest(), digest, str(relative))

    def test_stage_only_preserves_dirty_source_and_records_overrides(self):
        original = self.original_hashes()
        with patch.object(builder, "run_pyinstaller") as run:
            output = builder.build(self.root, self.root / "dist", stage_only=True)
        run.assert_not_called()
        self.assert_inputs_unchanged(original)
        self.assertEqual((output / "source/setup.py").read_text(), "# bundle-specific setup\n")
        self.assertTrue((output / "source/tracking_engine.py").is_file())
        self.assertFalse((output / "source/.git").exists())
        self.assertFalse((output / "source/dist").exists())
        self.assertFalse((output / "source/sarween_rec_private.mp4").exists())
        self.assertFalse((output / "source/auto-wall-main").exists())
        manifest = json.loads((output / "build-inputs.json").read_text())
        setup = next(row for row in manifest["files"] if row["path"] == "setup.py")
        self.assertEqual(setup["source"], "app_bundle_overrides/setup.py")
        self.assertEqual(setup["sha256"], hashlib.sha256((output / "source/setup.py").read_bytes()).hexdigest())

    def test_success_builds_in_temporary_directory_and_never_replaces_old_output(self):
        original = self.original_hashes()
        outputs = [builder.build(self.root, self.root / "dist", timeout_seconds=5) for _ in range(2)]
        self.assertNotEqual(*outputs)
        for output in outputs:
            self.assertEqual((output / "Sarween.app/built.txt").read_text(), "test build")
            self.assertFalse(json.loads((output / "build-inputs.json").read_text())["stageOnly"])
        self.assert_inputs_unchanged(original)
        trace = json.loads(self.trace.read_text())
        self.assertFalse(Path(trace["cwd"]).exists())
        self.assertFalse(Path(trace["cache"]).exists())

    def test_failure_removes_staging_but_preserves_checkout_and_previous_app(self):
        original = self.original_hashes()
        with patch.dict(os.environ, {"SARWEEN_TEST_BUILD_MODE": "fail"}), \
             patch.object(builder.os, "killpg", wraps=os.killpg) as kill:
            with self.assertRaisesRegex(RuntimeError, "code 17"):
                builder.build(self.root, self.root / "dist", timeout_seconds=5)
        self.assertEqual([call.args[1] for call in kill.call_args_list], [signal.SIGTERM, signal.SIGKILL])
        self.assert_inputs_unchanged(original)
        self.assertFalse(Path(json.loads(self.trace.read_text())["cwd"]).exists())
        self.assertEqual(sorted(path.name for path in (self.root / "dist").iterdir()), ["Sarween.app"])

    def test_timeout_stops_and_reaps_builder_and_removes_staging(self):
        original = self.original_hashes()
        real_popen = subprocess.Popen
        children = []
        def start(*args, **kwargs):
            child = real_popen(*args, **kwargs)
            children.append(child)
            return child
        with patch.dict(os.environ, {"SARWEEN_TEST_BUILD_MODE": "timeout"}), \
             patch.object(builder.subprocess, "Popen", side_effect=start), \
             patch.object(builder.os, "killpg", wraps=os.killpg) as kill:
            with self.assertRaises(subprocess.TimeoutExpired):
                builder.build(self.root, self.root / "dist", timeout_seconds=.5)
        self.assertIsNotNone(children[0].returncode)
        self.assertIn(unittest.mock.call(children[0].pid, signal.SIGTERM), kill.call_args_list)
        self.assertIn(unittest.mock.call(children[0].pid, signal.SIGKILL), kill.call_args_list)
        self.assert_inputs_unchanged(original)
        self.assertFalse(Path(json.loads(self.trace.read_text())["cwd"]).exists())

    def test_missing_data_fails_before_building(self):
        (self.root / "profile.json").unlink()
        with patch.object(builder, "run_pyinstaller") as run:
            with self.assertRaisesRegex(ValueError, "Missing bundle input: profile.json"):
                builder.build(self.root, self.root / "dist")
        run.assert_not_called()

    def test_interrupt_stops_process_group_and_waits_for_builder(self):
        with patch.object(builder.subprocess, "Popen") as popen, patch.object(builder.os, "killpg") as kill:
            child = popen.return_value
            child.pid = 12345
            child.wait.side_effect = [KeyboardInterrupt, 0, 0]
            with self.assertRaises(KeyboardInterrupt):
                builder.run_pyinstaller(self.root, self.root, 5)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            self.assertEqual(kill.call_args_list, [unittest.mock.call(12345, signal.SIGTERM),
                                                  unittest.mock.call(12345, signal.SIGKILL)])
            self.assertEqual(child.wait.call_count, 3)

    def test_data_paths_cannot_escape_project_or_include_build_output(self):
        for source in ("../outside", "/tmp/outside", "dist", ".git", "deprecated", "Deprecated"):
            with self.subTest(source=source):
                self.write("app_bundle_overrides/sarween.spec", f"added_files = [({source!r}, '.')]")
                with self.assertRaises(ValueError):
                    builder.collect_inputs(self.root)

    def test_current_bundle_excludes_retired_engines(self):
        import ast
        retired = {"band_tracking", "alt_band_tracking", "blob_tracking", "hsv_visualizer", "resource_path_snippet"}
        plan = builder.collect_inputs(builder.ROOT)
        self.assertFalse({Path(name).stem for name in plan} & retired)
        self.assertFalse(any(path.relative_to(builder.ROOT).parts[0].lower() == "deprecated"
                             for path in plan.values()))
        for relative in ("sarween.spec", "app_bundle_overrides/sarween.spec"):
            tree = ast.parse((builder.ROOT / relative).read_text())
            hidden = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Name) and target.id == "hidden" for target in node.targets))
            self.assertFalse(set(ast.literal_eval(hidden)) & retired)

    def test_symlinked_data_is_rejected(self):
        (self.root / "maps/link.png").symlink_to(self.root / "profile.json")
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            builder.collect_inputs(self.root)

    def test_personal_data_is_rejected_even_if_added_to_spec(self):
        for name in ("combo_profiles.json", "hardware_config.json", "mini_library.json.bak",
                     "camera_matrix.npy", "mini_captures", "sarween_rec_private.mp4"):
            with self.subTest(name=name):
                self.write("app_bundle_overrides/sarween.spec", f"added_files = [({name!r}, '.')]")
                with self.assertRaisesRegex(ValueError, "Personal user data"):
                    builder.collect_inputs(self.root)

    def test_real_bundle_specs_exclude_personal_state(self):
        import ast
        import app_paths
        self.assertTrue(set(app_paths.STATE_FILES) <= builder.PERSONAL_FILES)
        for relative in ("sarween.spec", "app_bundle_overrides/sarween.spec"):
            tree = ast.parse((builder.ROOT / relative).read_text())
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "added_files" for t in node.targets))
            for source, _ in ast.literal_eval(assignment.value):
                self.assertNotIn(Path(source).name, builder.PERSONAL_FILES)

    def test_check_is_read_only_and_timeout_is_validated(self):
        original = self.original_hashes()
        with patch.object(builder, "ROOT", self.root), patch.object(builder, "build") as build:
            self.assertEqual(builder.main(["--check"]), 0)
            self.assertEqual(builder.main(["--timeout-seconds", "0"]), 1)
            self.assertEqual(builder.main(["--timeout-seconds", "nan"]), 1)
        build.assert_not_called()
        self.assertEqual(self.original_hashes(), original)

    def test_compatibility_scripts_work_from_unrelated_directory(self):
        for name in ("build_app.sh", "build_bundle.sh", "app_bundle_overrides/build_app.sh", "app_bundle_overrides/build_bundle.sh"):
            self.write(name, (builder.ROOT / name).read_text())
        self.write("scripts/build_bundle.py", (builder.ROOT / "scripts/build_bundle.py").read_text())
        for name in ("build_app.sh", "build_bundle.sh", "app_bundle_overrides/build_app.sh", "app_bundle_overrides/build_bundle.sh"):
            with self.subTest(script=name):
                run = subprocess.run(["bash", str(self.root / name), "--check"], cwd=self.temporary.name,
                                     env={**os.environ, "PYTHON_BIN": sys.executable}, text=True,
                                     capture_output=True, timeout=5)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertIn("Check only", run.stdout)


if __name__ == "__main__":
    unittest.main()
