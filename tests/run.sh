#!/usr/bin/env bash
#
# Run the claude-code-provider live integration test suite.
#
# Usage:
#   ./tests/run.sh                     # Run all live integration tests
#   ./tests/run.sh -k "TestAuth"       # Run only auth tests
#   ./tests/run.sh -k "test_health"    # Run a single test
#   ./tests/run.sh -x                  # Stop on first failure
#   ./tests/run.sh --co                # List all tests without running
#
# The server is built and started automatically. Requires valid local Claude
# OAuth credentials because completion tests call Anthropic through CCP.

set -euo pipefail
cd "$(dirname "$0")/.."

exec uv run \
	--with httpx --with openai --with pytest --with pytest-asyncio --with xxhash \
	pytest tests/test_integration.py \
	-v --tb=short \
	"$@"
