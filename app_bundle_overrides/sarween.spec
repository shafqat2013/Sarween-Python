# sarween.spec
# PyInstaller spec file for Sarween — macOS .app bundle
#
# Usage:
#   pyinstaller sarween.spec
#
# Output:
#   dist/Sarween.app

import sys
from pathlib import Path

block_cipher = None

# ── Collect all data files that need to ship with the app ─────────────────────
# These are files Sarween reads at runtime (not imported as Python modules).
# Format: (source_path, dest_folder_inside_app)
added_files = [
    ("band_profiles.json",      "."),   # HSV band definitions
    ("hardware_config.json",    "."),   # camera index etc.
    ("combo_profiles.json",     "."),   # live ring-color tracker profiles
    ("mini_library.json",       "."),   # player mini metadata and scan portfolio
    ("mini_token_map.json",     "."),   # Foundry token mapping
    ("mini_database.csv",       "."),   # mini capture DB (may be empty)
    ("module.js",               "."),   # Foundry module JS
    ("module.json",             "."),   # Foundry module manifest
    ("tk_camera_preview.py",    "."),   # Tkinter camera preview window
    ("maps",                    "maps"),# map image files
]

# ── Hidden imports ─────────────────────────────────────────────────────────────
# PyInstaller's static analysis misses some imports (dynamic imports, plugins).
hidden = [
    # OpenCV internals
    "cv2",
    # band/blob engines are imported dynamically in main.py
    "band_tracking",
    "blob_tracking",
    "alt_band_tracking",
    # tk_camera_preview new dependency
    "tk_camera_preview",
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    # Standard lib async/threading (usually fine, but explicit is safer)
    "asyncio",
    "threading",
    # tkinter (used by control_panel / setup dialogs)
    "tkinter",
    "tkinter.ttk",
    "tkinter.messagebox",
    "tkinter.filedialog",
    # numpy / scipy internals sometimes missed
    "numpy.core._methods",
    "numpy.lib.format",
]

a = Analysis(
    ["main.py"],                        # entry point
    pathex=["."],                       # add project root to sys.path
    binaries=[],
    datas=added_files,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",                   # not used, keeps bundle smaller
        "IPython",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sarween",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,                          # UPX can corrupt OpenCV dylibs on macOS
    console=False,                      # no terminal window
    # icon="assets/sarween.icns",       # uncomment when you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Sarween",
)

app = BUNDLE(
    coll,
    name="Sarween.app",
    # icon="assets/sarween.icns",       # uncomment when you have an icon
    bundle_identifier="com.sarween.app",
    info_plist={
        # ── Required for camera access on macOS ────────────────────────────
        "NSCameraUsageDescription":
            "Sarween needs camera access to track miniature positions on the map.",
        # ── Required for network (Foundry WebSocket) ───────────────────────
        "NSLocalNetworkUsageDescription":
            "Sarween connects to Foundry VTT on your local network.",
        # ── App metadata ───────────────────────────────────────────────────
        "CFBundleName":                 "Sarween",
        "CFBundleDisplayName":          "Sarween",
        "CFBundleVersion":              "0.1.0",
        "CFBundleShortVersionString":   "0.1.0",
        "LSMinimumSystemVersion":       "12.0",     # Monterey+
        "NSHighResolutionCapable":      True,
        "NSRequiresAquaSystemAppearance": False,    # allow dark mode
    },
)
