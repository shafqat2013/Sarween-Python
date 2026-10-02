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
    ("auth_config.json",        "."),   # validated public configuration only
    # Personal profiles, settings, calibration and captures never ship in the app.
    ("module.js",               "."),   # Foundry module JS
    ("capture_logic.mjs",       "."),   # guided dataset capture rules
    ("movement_logic.mjs",      "."),   # movement budget state machine
    ("module.json",             "."),   # Foundry module manifest
    ("tk_camera_preview.py",    "."),   # Tkinter camera preview window
    ("maps/dnd1.jpg",           "maps"),# demo map only; custom maps stay external
    # Only the approved privacy-edited real example ships. Synthetic clips stay in tests.
    ("demo/tabletop.mp4", "demo"),
    ("demo/tabletop.tracking.json", "demo"),
    ("demo/tabletop.profiles.json", "demo"),
    ("demo/tabletop.provenance.json", "demo"),
]

# ── Hidden imports ─────────────────────────────────────────────────────────────
# PyInstaller's static analysis misses some imports (dynamic imports, plugins).
hidden = [
    "keyring.backends.macOS",
    "jwt.algorithms",
    # OpenCV internals
    "cv2",
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
    # The NumPy hook collects version-specific private modules.
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
        "CFBundleVersion":              "0.2.0",
        "CFBundleShortVersionString":   "0.2.0",
        "LSMinimumSystemVersion":       "12.0",     # Monterey+
        "NSHighResolutionCapable":      True,
        "NSRequiresAquaSystemAppearance": False,    # allow dark mode
    },
)
