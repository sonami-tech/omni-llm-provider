#!/usr/bin/env bash
# Thin wrapper around the shared capture framework for Claude general capture.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
ARGS=(capture run --provider claude --mode general --live-capture)
if [ "$#" -gt 0 ]; then
  ARGS+=(--models "$@")
fi
exec python3 -m tools.capture "${ARGS[@]}"