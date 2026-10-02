"""Build from a disposable source snapshot without modifying the checkout."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PERSONAL_FILES = {"hardware_config.json", "combo_profiles.json", "band_profiles.json",
                  "mini_library.json", "mini_token_map.json", "camera_matrix.npy",
                  "dist_coeffs.npy", "mini_database.csv", "migration-v1.json"}


def local_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Bundle input must be a relative project path: {relative}")
    if path.parts[0].lower() in {".git", "build", "dist", "auto-wall-main", "deprecated"}:
        raise ValueError(f"Not a bundle input directory: {relative}")
    if (path.name.startswith(".env") or path.name.endswith((".private.pem", ".key"))
            or path.parts[0] == "supabase"):
        raise ValueError(f"Server credentials/backend files must not be bundled: {relative}")
    if (path.name.removesuffix(".bak") in PERSONAL_FILES or "mini_captures" in path.parts
            or path.name.startswith("sarween_rec_")):
        raise ValueError(f"Personal user data must not be bundled: {relative}")
    candidate = root / path
    for parent in (candidate, *candidate.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError(f"Symlinked bundle inputs are not supported: {relative}")
    if not candidate.exists():
        raise ValueError(f"Missing bundle input: {relative}")
    return candidate


def collect_inputs(root):
    """Use the spec's literal data list; do not import or execute app code."""
    root = root.resolve()
    plan = {path.name: local_path(root, path.name) for path in sorted(root.iterdir())
            if path.is_file() and path.suffix in {".py", ".js", ".mjs"}}
    overrides = root / "app_bundle_overrides"
    if overrides.is_dir():
        for path in sorted(overrides.glob("*.py")):
            plan[path.name] = local_path(root, path.relative_to(root))
    spec = overrides / "sarween.spec" if (overrides / "sarween.spec").exists() else root / "sarween.spec"
    plan["sarween.spec"] = local_path(root, spec.relative_to(root))
    tree = ast.parse(spec.read_text(encoding="utf-8"), filename=str(spec))
    assignments = [node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "added_files" for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("Bundle spec must define one literal added_files list")
    data = ast.literal_eval(assignments[0])
    for source, destination in data:
        if not isinstance(source, str) or not isinstance(destination, str):
            raise ValueError("Bundle data entries must contain string paths")
        if Path(destination).is_absolute() or ".." in Path(destination).parts:
            raise ValueError(f"Unsafe bundle data destination: {destination}")
        source_path = local_path(root, source)
        files = sorted(source_path.rglob("*")) if source_path.is_dir() else [source_path]
        for path in files:
            relative = path.relative_to(root).as_posix()
            checked = local_path(root, relative)
            if checked.is_file():
                plan.setdefault(relative, checked)
    if "main.py" not in plan:
        raise ValueError("Missing bundle entry point: main.py")
    if "auth_config.json" in plan:
        # An empty placeholder can be staged/reviewed; populated configuration must
        # remain public-only even in a source staging directory.
        value = json.loads(plan["auth_config.json"].read_text())
        if value != {}:
            sys.path.insert(0, str(ROOT))
            from auth_config import AuthConfig
            AuthConfig.load(plan["auth_config.json"])
    return dict(sorted(plan.items()))


def stage_inputs(root, plan, directory):
    directory.mkdir()
    entries = []
    for relative, source in plan.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({"path": relative, "source": source.relative_to(root).as_posix(),
                        "bytes": target.stat().st_size, "sha256": digest.hexdigest()})
    return {"schemaVersion": 1, "python": sys.version, "executable": sys.executable, "files": entries}


def run_pyinstaller(source, workspace, timeout_seconds):
    env = os.environ.copy()
    env["PYINSTALLER_CONFIG_DIR"] = str(workspace / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
               "--distpath", str(workspace / "dist"), "--workpath", str(workspace / "work"),
               "sarween.spec"]
    run_bounded(command, cwd=source, env=env, timeout_seconds=timeout_seconds)


def run_bounded(command, *, timeout_seconds, cwd=None, env=None):
    child = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
    try:
        code = child.wait(timeout=timeout_seconds)
        if code:
            raise RuntimeError(f"{Path(command[0]).name} exited with code {code}")
    except BaseException:
        # Build tools may spawn children; stop the entire owned process group.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise


def build(root, output_root, *, stage_only=False, timeout_seconds=600):
    root = root.resolve()
    plan = collect_inputs(root)
    if not stage_only and (root / "auth_config.json").exists():
        # Staging remains available for review; a distributable build must have
        # complete public-only auth configuration. Never ship a login bypass.
        sys.path.insert(0, str(root))
        from auth_config import AuthConfig
        config = AuthConfig.load(root / "auth_config.json")
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if not isinstance(load_pem_public_key(config.verification_key.encode()), Ed25519PublicKey):
            raise ValueError("A valid Ed25519 PUBLIC verification key is required")
    with tempfile.TemporaryDirectory(prefix="sarween-build-") as directory:
        workspace = Path(directory)
        source = workspace / "source"
        manifest = stage_inputs(root, plan, source)
        if not stage_only:
            run_pyinstaller(source, workspace, timeout_seconds)
            if not (workspace / "dist" / "Sarween.app").is_dir():
                raise RuntimeError("Build did not produce Sarween.app")
        output_root.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="Sarween-", dir=output_root))
        if stage_only:
            shutil.move(str(source), str(output / "source"))
        else:
            for artifact in (workspace / "dist").iterdir():
                shutil.move(str(artifact), str(output / artifact.name))
            warnings = workspace / "work" / "sarween" / "warn-sarween.txt"
            if warnings.is_file():
                shutil.copy2(warnings, output / "build-warnings.txt")
        manifest["stageOnly"] = stage_only
        (output / "build-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", action="store_true", help="Validate and summarize inputs without copying or building.")
    actions.add_argument("--stage-only", action="store_true", help="Save the staged source snapshot without invoking PyInstaller.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dist")
    parser.add_argument("--timeout-seconds", type=float, default=600, help="PyInstaller deadline (default: 600 seconds).")
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
            raise ValueError("Build timeout must be finite and positive")
        if args.check:
            root = ROOT.resolve()
            plan = collect_inputs(root)
            size = sum(path.stat().st_size for path in plan.values())
            print(f"Bundle inputs: {len(plan)} files, {size / 1024**2:.1f} MiB")
            for target, source in plan.items():
                if source.relative_to(root).as_posix() != target:
                    print(f"  Override: {target} <- {source.relative_to(root)}")
            print(f"Python: {sys.executable}")
            print("Check only: no source changes, staging, build, or output replacement.")
            return 0
        output = build(ROOT, args.output_dir.expanduser().resolve(), stage_only=args.stage_only,
                       timeout_seconds=args.timeout_seconds)
        print(f"{'Staged source' if args.stage_only else 'Built app'}: {output / ('source' if args.stage_only else 'Sarween.app')}")
        print(f"Input manifest: {output / 'build-inputs.json'}")
        return 0
    except subprocess.TimeoutExpired:
        print("Build timed out; builder process group stopped and staging removed.", file=sys.stderr)
    except KeyboardInterrupt:
        print("Build cancelled; builder process group stopped and staging removed.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
