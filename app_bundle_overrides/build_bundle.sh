# build_bundle.sh (in project root)
cp app_bundle_overrides/*.py .
cp app_bundle_overrides/sarween.spec .
pyinstaller sarween.spec
git checkout band_tracking.py alt_band_tracking.py calibration.py