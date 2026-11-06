"""Write a lightweight Mac app that opens the standalone review desk.

This creates a launcher only. It does not open any application or video.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import plistlib
import shlex
import sys


BUNDLE_ID = "com.sarween.video-review"
ROOT = Path(__file__).resolve().parents[1]


def build_app(destination, *, source=ROOT, python=sys.executable):
    source = Path(source).resolve()
    python = Path(python).resolve()
    destination = Path(destination).expanduser().absolute()
    if destination.suffix != ".app":
        raise ValueError("Choose an .app destination")
    if destination.is_symlink():
        raise ValueError("Refusing to replace a symbolic link")
    if not (source / "audit_videos.py").is_file() or not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("The standalone review script and a working Python interpreter are required")
    if destination.exists():
        try:
            info = plistlib.loads((destination / "Contents/Info.plist").read_bytes())
        except (OSError, ValueError):
            raise ValueError("An unrelated item already exists at the app destination") from None
        if info.get("CFBundleIdentifier") != BUNDLE_ID:
            raise ValueError("Refusing to replace another application")
    contents = destination / "Contents"
    executable = contents / "MacOS/audit-launch"
    executable.parent.mkdir(parents=True, exist_ok=True)
    info = {"CFBundleIdentifier": BUNDLE_ID, "CFBundleName": "Sarween Video Review",
            "CFBundleDisplayName": "Sarween Video Review", "CFBundleExecutable": "audit-launch",
            "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1", "NSHighResolutionCapable": True}
    (contents / "Info.plist").write_bytes(plistlib.dumps(info))
    (contents / "PkgInfo").write_bytes(b"APPL????")
    # Quote paths as shell syntax, never interpolate arbitrary path text as code.
    script = ("#!/bin/zsh\nset -e\n"
              'review_log_dir="$HOME/Library/Application Support/Sarween"\n'
              'mkdir -p "$review_log_dir"\n'
              f"cd {shlex.quote(str(source))}\n"
              f"exec {shlex.quote(str(python))} {shlex.quote(str(source / 'audit_videos.py'))}"
              ' >> "$review_log_dir/video-audit-launch.log" 2>&1\n')
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o755)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", type=Path,
                        default=ROOT / "Sarween Video Review.app")
    args = parser.parse_args(argv)
    print(build_app(args.destination))


if __name__ == "__main__":
    main()
