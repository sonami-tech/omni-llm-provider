#!/usr/bin/env bash
#
# Run the claude-code-provider integration test suite.
#
# Usage:
#   ./tests/run.sh                     # Run all tests
#   ./tests/run.sh -k "TestAuth"       # Run only auth tests
#   ./tests/run.sh -k "test_health"    # Run a single test
#   ./tests/run.sh -x                  # Stop on first failure
#   ./tests/run.sh --co                # List all tests without running
#
# The server is built and started automatically. No manual setup needed.

set -euo pipefail
cd "$(dirname "$0")/.."

exec uv run \
	--with httpx --with openai --with pytest --with pytest-asyncio \
	pytest tests/test_integration.py \
	-v --tb=short \
	"$@"
