#!/usr/bin/env bash
# Capture a real Claude Code -> api.anthropic.com flow for fingerprint re-baselining.
#
# Starts mitmdump as a reverse-proxy recorder in front of the real API, drives the
# installed `claude` CLI through it for the default model + each model id passed as
# an argument, then extracts each captured POST /v1/messages and the model-list GET.
#
# Usage:
#   tools/fingerprint/capture_baseline.sh claude-haiku-4-5 claude-sonnet-4-6 claude-opus-4-8
#
# Output: writes the raw flow to /tmp/cc-baseline.flow and prints the extracted
# header+body report to stdout (also saved to /tmp/cc-baseline-extract.md).
#
# Requires: mitmdump (via PATH or `uv tool run --from mitmproxy`), a working `claude`
# login. Uses CCP's own extract_flow.py for the report.
set -euo pipefail

# Pick a free port unless one is explicitly forced. Hardcoding a port risks a
# collision (e.g. a browser-use mitmproxy already on :8080), which makes mitmdump
# fail to bind and silently produce a 0-byte flow.
if [ -n "${CCP_CAPTURE_PORT:-}" ]; then
  PORT="$CCP_CAPTURE_PORT"
else
  PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
fi
FLOW="/tmp/cc-baseline.flow"
EXTRACT="/tmp/cc-baseline-extract.md"
PROMPT="Say OK"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve mitmdump: prefer PATH, else uv.
if command -v mitmdump >/dev/null 2>&1; then
  MITM=(mitmdump)
else
  MITM=(uv tool run --from mitmproxy mitmdump)
fi

echo "[capture] starting mitmdump reverse-proxy on :$PORT -> https://api.anthropic.com" >&2
rm -f "$FLOW"
"${MITM[@]}" -w "$FLOW" --mode "reverse:https://api.anthropic.com" --listen-port "$PORT" \
  --set keep_host_header=false >/tmp/cc-mitm.log 2>&1 &
MITM_PID=$!
# Ensure we always tear the proxy down.
cleanup() { kill "$MITM_PID" 2>/dev/null || true; wait "$MITM_PID" 2>/dev/null || true; }
trap cleanup EXIT

# Wait for the listener to accept connections (bounded). Fail LOUD if mitmdump
# died (e.g. bind error) instead of silently producing an empty flow.
bound=""
for _ in $(seq 1 30); do
  if ! kill -0 "$MITM_PID" 2>/dev/null; then
    echo "[capture] FATAL: mitmdump exited during startup. Log:" >&2
    cat /tmp/cc-mitm.log >&2 || true
    exit 1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&- 3<&-; bound=1; break; fi
  sleep 0.3
done
if [ -z "$bound" ]; then
  echo "[capture] FATAL: mitmdump did not accept connections on :$PORT within timeout. Log:" >&2
  cat /tmp/cc-mitm.log >&2 || true
  exit 1
fi
echo "[capture] mitmdump listening on :$PORT" >&2

export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export CLAUDE_CODE_ENTRYPOINT="sdk-cli"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

drive() { # $1 = optional model
  if [ -n "${1:-}" ]; then
    echo "[capture] claude --model $1" >&2
    claude --print --no-session-persistence --model "$1" -- "$PROMPT" >/dev/null 2>&1 || \
      echo "[capture] WARN: claude call for model '$1' returned nonzero (continuing)" >&2
  else
    echo "[capture] claude (default model, no --model)" >&2
    claude --print --no-session-persistence -- "$PROMPT" >/dev/null 2>&1 || \
      echo "[capture] WARN: default claude call returned nonzero (continuing)" >&2
  fi
}

drive ""              # default-model resolution
for m in "$@"; do drive "$m"; done

# Give mitmdump a moment to flush the flow to disk.
sleep 1
cleanup
trap - EXIT

echo "[capture] extracting $FLOW -> $EXTRACT" >&2
if command -v mitmdump >/dev/null 2>&1 && python3 -c "import mitmproxy" 2>/dev/null; then
  python3 "$SCRIPT_DIR/extract_flow.py" "$FLOW" --body-bytes 8000 | tee "$EXTRACT"
else
  uv tool run --from mitmproxy python3 "$SCRIPT_DIR/extract_flow.py" "$FLOW" --body-bytes 8000 | tee "$EXTRACT"
fi
echo "[capture] done. flow=$FLOW extract=$EXTRACT" >&2
