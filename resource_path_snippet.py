# ── Add this to setup.py (or a shared utils.py) ───────────────────────────────
#
# PyInstaller bundles data files next to the executable inside the .app,
# but os.path.dirname(__file__) won't work correctly when frozen.
# Use resource_path() everywhere you open a file by name.
#
# Example:
#   profiles = json.loads(open(resource_path("band_profiles.json")).read())
#   db_path  = resource_path("mini_database.csv")

import os
import sys


def resource_path(filename: str) -> str:
    """
    Return the absolute path to a bundled data file.

    Works in two contexts:
      - Normal dev run:   resolves relative to this file's directory
      - PyInstaller .app: resolves relative to the executable inside the bundle
    """
    if getattr(sys, "frozen", False):
        # Inside a PyInstaller bundle — data files sit next to the executable
        base = os.path.dirname(sys.executable)
    else:
        # Normal dev environment — data files sit next to the source
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, filename)
