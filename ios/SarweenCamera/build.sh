#!/bin/bash
set -euo pipefail

sarween_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sarween_platform="${1:-device}"
case "$sarween_platform" in
  device) sarween_destination='generic/platform=iOS'; sarween_sdk=iphoneos ;;
  simulator) sarween_destination='generic/platform=iOS Simulator'; sarween_sdk=iphonesimulator ;;
  *) echo 'Usage: build.sh [device|simulator]' >&2; exit 2 ;;
esac
sarween_derived="${SARWEEN_IOS_DERIVED_DATA:-/tmp/sarween-camera-${sarween_platform}-build}"
exec xcodebuild -project "$sarween_project_dir/SarweenCamera.xcodeproj" \
  -scheme SarweenCamera -configuration Debug -sdk "$sarween_sdk" \
  -destination "$sarween_destination" -derivedDataPath "$sarween_derived" \
  CODE_SIGNING_ALLOWED=NO CLANG_MODULE_CACHE_PATH="$sarween_derived/ModuleCache.noindex" \
  SWIFT_MODULE_CACHE_PATH="$sarween_derived/SwiftModuleCache" build
