#!/bin/bash
set -e
cp app_bundle_overrides/*.py .
cp app_bundle_overrides/sarween.spec .
cp app_bundle_overrides/build_app.sh .
bash build_app.sh
git checkout band_tracking.py alt_band_tracking.py calibration.py setup.py
