#!/bin/bash
# Compatibility entry point; all builds use isolated staging.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/build_bundle.sh" "$@"
