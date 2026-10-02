"""Compatibility imports for older path examples.

Use resource_path("maps/dnd1.jpg") for read-only bundled assets.
Use data_path("combo_profiles.json") for writable user data.
New code should import these helpers directly from app_paths.
"""

from app_paths import data_path, resource_path
