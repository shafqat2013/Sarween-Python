"""Create a personal drag-to-Applications DMG without replacing prior builds."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import plistlib
import shutil
import sys
import tempfile

try:
    from scripts.build_bundle import run_bounded
except ModuleNotFoundError:
    from build_bundle import run_bounded


def package(app, output, *, timeout_seconds=180):
    app = app.resolve()
    output = output.absolute()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new path: {output}")
    if output.suffix.lower() != ".dmg":
        raise ValueError("Output must end in .dmg")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("Timeout must be finite and positive")
    info_path = app / "Contents/Info.plist"
    if app.suffix != ".app" or not info_path.is_file():
        raise ValueError("Input must be a built .app bundle")
    with info_path.open("rb") as stream:
        info = plistlib.load(stream)
    executable = info.get("CFBundleExecutable", "")
    if not executable or Path(executable).name != executable or not (app / "Contents/MacOS" / executable).is_file():
        raise ValueError("App executable is missing or invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sarween-dmg-", dir=output.parent) as directory:
        temporary = Path(directory)
        volume = temporary / "volume"
        volume.mkdir()
        shutil.copytree(app, volume / "Sarween.app", symlinks=True)
        (volume / "Applications").symlink_to("/Applications", target_is_directory=True)
        image = temporary / "Sarween.dmg"
        run_bounded(["hdiutil", "create", "-volname", "Sarween", "-srcfolder", str(volume),
                     "-format", "UDZO", "-imagekey", "zlib-level=6", str(image)],
                    timeout_seconds=timeout_seconds)
        run_bounded(["hdiutil", "verify", str(image)], timeout_seconds=timeout_seconds)
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        # No-clobber publication, including a competing build finishing first.
        output.hardlink_to(image)
    return {"app": str(app), "dmg": str(output), "sha256": digest.hexdigest(),
            "version": info.get("CFBundleShortVersionString"),
            "bytes": output.stat().st_size, "notarized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    args = parser.parse_args(argv)
    result = package(args.app, args.output, timeout_seconds=args.timeout_seconds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
