#!/bin/bash
# build_app.sh — Build Sarween.app with PyInstaller
#
# Run this from the root of your Sarween project directory.
# Produces:  dist/Sarween.app
#
# ─────────────────────────────────────────────────────────────────────────────
# FIRST TIME SETUP (do once per machine)
# ─────────────────────────────────────────────────────────────────────────────
#
#   1. Install PyInstaller into your existing Python env:
#        pip install pyinstaller
#
#   2. Confirm you're using the right Python (the one with cv2, numpy, etc.):
#        which python3
#        python3 -c "import cv2; print(cv2.__version__)"
#
#   3. Make sure this script is executable:
#        chmod +x build_app.sh
#
# ─────────────────────────────────────────────────────────────────────────────
# BUILDING
# ─────────────────────────────────────────────────────────────────────────────

set -e  # exit on any error

ARCH=$(uname -m)   # arm64 or x86_64
echo "──────────────────────────────────────────"
echo " Sarween .app builder"
echo " Architecture: $ARCH"
echo "──────────────────────────────────────────"

# Clean previous build artifacts
echo "Cleaning previous build..."
rm -rf build/ dist/

# Run PyInstaller
echo "Running PyInstaller..."
pyinstaller sarween.spec

# Confirm output
if [ -d "dist/Sarween.app" ]; then
    echo ""
    echo "✅ Build succeeded!"
    echo "   Output: $(pwd)/dist/Sarween.app"
    echo "   Arch:   $ARCH"
    echo ""
    echo "To test it:"
    echo "   open dist/Sarween.app"
    echo ""
    echo "To distribute it:"
    echo "   Zip the .app:  ditto -c -k --keepParent dist/Sarween.app dist/Sarween-$ARCH.zip"
    echo "   Upload Sarween-$ARCH.zip to your website."
else
    echo ""
    echo "❌ Build failed — dist/Sarween.app not found."
    echo "   Check the output above for errors."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# COMMON ERRORS & FIXES
# ─────────────────────────────────────────────────────────────────────────────
#
# ERROR: "ModuleNotFoundError: No module named 'cv2'"
#   PyInstaller is using a different Python than your project.
#   Fix: Use the full path:  /path/to/your/python -m PyInstaller sarween.spec
#
# ERROR: "No such file: band_profiles.json"  (at runtime, not build time)
#   The app can't find its data files next to the executable.
#   Fix: In your Python code, always resolve paths relative to sys.executable
#   or use the helper below. Add to main.py or setup.py:
#
#       import sys, os
#       def resource_path(filename):
#           """Works both in dev and inside .app bundle."""
#           if getattr(sys, 'frozen', False):
#               # Running inside PyInstaller bundle
#               base = os.path.dirname(sys.executable)
#           else:
#               base = os.path.dirname(os.path.abspath(__file__))
#           return os.path.join(base, filename)
#
# ERROR: Camera permission denied (no popup ever appears)
#   macOS sandboxing blocked it silently.
#   Fix: The NSCameraUsageDescription in sarween.spec Info.plist is set.
#   If still failing, try running once from Terminal to see the OS prompt:
#       ./dist/Sarween.app/Contents/MacOS/Sarween
#
# ERROR: "dyld: Library not loaded" for a .dylib
#   A compiled dependency wasn't bundled.
#   Fix: Find the dylib with `otool -L dist/Sarween.app/Contents/MacOS/Sarween`
#   and add it manually to `binaries` in sarween.spec.
#
# ERROR: App opens and immediately closes (no error visible)
#   console=False swallows output. Temporarily set console=True in sarween.spec
#   and rebuild, then run from Terminal to see the traceback.
#
# ─────────────────────────────────────────────────────────────────────────────
# REPEATING ON INTEL MAC (at home)
# ─────────────────────────────────────────────────────────────────────────────
#
#   1. Clone/copy your project to the Intel Mac (same files, same structure).
#   2. Set up the same Python env:
#        python3 -m venv venv
#        source venv/bin/activate
#        pip install opencv-python numpy pyinstaller websockets
#        pip install <any other deps>
#   3. Run this script:  ./build_app.sh
#   4. Output will be dist/Sarween.app (x86_64 build).
#   5. Zip it:  ditto -c -k --keepParent dist/Sarween.app dist/Sarween-x86_64.zip
#
# On your website, link both zips:
#   - Sarween-arm64.zip  → Apple Silicon (M1/M2/M3)
#   - Sarween-x86_64.zip → Intel Mac
