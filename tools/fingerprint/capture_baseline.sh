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
# Output: writes the raw flow + extracted report into a private, owner-only dir on
# a RAM-backed tmpfs (printed at the end). The raw .flow contains the LIVE OAuth
# bearer token, so: the script REFUSES to run without a tmpfs (never writes the
# token to persistent disk), creates the dir with umask 077, and shreds the flow on
# exit unless you set KEEP_FLOW=1 (retention is still tmpfs-only, gone on reboot).
# The extract has auth/x-api-key headers redacted by extract_flow.py.
#
# Token-safety is best-effort against TRAPPABLE termination (normal exit, INT, TERM,
# HUP, QUIT). An untrappable SIGKILL, a hard crash, or power loss cannot run the
# purge — the tmpfs-only guarantee is the backstop there: the token lives in RAM and
# does not survive a reboot, and shred/rm on the same tmpfs is a clean unlink.
#
# Requires: a RAM-backed tmpfs (XDG_RUNTIME_DIR, /dev/shm, or /run/user/UID),
# mitmdump (via PATH or `uv tool run --from mitmproxy`), and a working `claude`
# login. Uses CCP's own extract_flow.py for the report.
set -euo pipefail

# The raw flow holds a live credential. Make every file we create owner-only
# (the tmpfs workdir below is also per-user 0700).
umask 077

# Pick a free port unless one is explicitly forced. Hardcoding a port risks a
# collision (e.g. a browser-use mitmproxy already on :8080), which makes mitmdump
# fail to bind and silently produce a 0-byte flow.
if [ -n "${CCP_CAPTURE_PORT:-}" ]; then
  PORT="$CCP_CAPTURE_PORT"
else
  PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
fi
# The raw flow holds a live credential, so it must live ONLY on a RAM-backed
# (tmpfs/ramfs) filesystem — shred/rm are not reliable erasure on SSDs, CoW,
# journaled, or snapshotted disks, and an untrappable SIGKILL / power loss would
# strand the token on persistent storage. We therefore FAIL CLOSED: pick the first
# candidate that the kernel reports as tmpfs/ramfs, and refuse to run if none is.
# XDG_RUNTIME_DIR (per-user, 0700, tmpfs) is ideal; /dev/shm is the common fallback.
is_ramfs() { # $1 = dir; true only if its backing fs is tmpfs or ramfs
  local t
  t="$(stat -f -c %T "$1" 2>/dev/null)" || return 1
  [ "$t" = "tmpfs" ] || [ "$t" = "ramfs" ]
}
WORKBASE=""
for cand in "${XDG_RUNTIME_DIR:-}" /dev/shm /run/user/"$(id -u)"; do
  [ -n "$cand" ] && [ -d "$cand" ] && [ -w "$cand" ] && is_ramfs "$cand" || continue
  WORKBASE="$cand"; break
done
if [ -z "$WORKBASE" ]; then
  echo "[capture] FATAL: no RAM-backed tmpfs dir (XDG_RUNTIME_DIR, /dev/shm, /run/user/UID)" >&2
  echo "[capture]        is available/writable. Refusing to write a live OAuth token to" >&2
  echo "[capture]        persistent disk. Mount a tmpfs and set XDG_RUNTIME_DIR to it." >&2
  exit 1
fi
echo "[capture] workdir base: $WORKBASE (tmpfs, RAM-backed)" >&2
WORKDIR="$(mktemp -d "$WORKBASE/cc-baseline.XXXXXX")"
FLOW="$WORKDIR/cc-baseline.flow"
EXTRACT="$WORKDIR/cc-baseline-extract.md"
MITMLOG="$WORKDIR/cc-mitm.log"
PROMPT="Say OK"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve mitmdump: prefer PATH, else uv.
if command -v mitmdump >/dev/null 2>&1; then
  MITM=(mitmdump)
else
  MITM=(uv tool run --from mitmproxy mitmdump)
fi

# Teardown correctness depends on pgrep to find the real mitmdump when it runs as a
# descendant of the `uv tool run` wrapper. Without it we could only kill the wrapper
# and ORPHAN the worker holding the live-token .flow open. Fail closed, loudly.
if ! command -v pgrep >/dev/null 2>&1; then
  echo "[capture] FATAL: pgrep not found. It is required to reliably stop the mitmdump" >&2
  echo "[capture]        worker before purging the token-bearing flow. Install procps." >&2
  exit 1
fi

echo "[capture] starting mitmdump reverse-proxy on :$PORT -> https://api.anthropic.com" >&2
rm -f "$FLOW"
# Launch the proxy as a plain background job and record its pid. When mitmdump is
# on PATH, $! IS mitmdump. When we fall back to `uv tool run ... mitmdump`, $! is
# the uv wrapper and the real mitmdump is a descendant. We deliberately do NOT use
# setsid here: `setsid foo & ; pid=$!` does not reliably name the worker — setsid
# forks, and $!/its pgid can name the short-lived launcher, so a pgid kill misses
# the real process. Instead teardown walks the descendant tree (below).
"${MITM[@]}" -w "$FLOW" --mode "reverse:https://api.anthropic.com" \
  --listen-host 127.0.0.1 --listen-port "$PORT" \
  --set keep_host_header=false >"$MITMLOG" 2>&1 &
MITM_PID=$!

# Recursively signal a pid and ALL its descendants, children BEFORE the parent, so
# the real mitmdump is signaled while its parent (the uv wrapper, if any) is still
# alive — killing the parent first would reparent and orphan the worker, leaving it
# holding the .flow open. Handles both the on-PATH ($! == mitmdump, no children) and
# the uv-wrapper ($! -> mitmdump child) cases with one primitive.
kill_tree() {
  local pid="$1" sig="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child" "$sig"
  done
  kill -"$sig" "$pid" 2>/dev/null || true
}
# Echo every still-live pid in the process tree rooted at $1 (root + all
# descendants). Used to GATE the purge on the real worker being gone, not just the
# (possibly uv-wrapper) pid we backgrounded — the worker is the process that holds
# the token-bearing .flow fd open, and may outlive its parent by a moment.
tree_pids() {
  local pid="$1" child
  kill -0 "$pid" 2>/dev/null && echo "$pid"
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    tree_pids "$child"
  done
}
# Ensure we always tear the proxy down so the real mitmdump dies and releases the
# .flow BEFORE purge_flow runs. Idempotent (guarded by MITM_STOPPED) so the explicit
# mid-script call and the EXIT trap can't double-signal a possibly-reused pid.
# Shutdown is BOUNDED: TERM the tree, poll up to ~3s until NO pid in the tree
# survives (so the fd-holding worker is provably gone, not just the wrapper), then
# KILL the tree as a last resort — a mitmdump that ignores/stalls on TERM must not
# wedge the trap before the purge, and the worker must not outlive the wait.
MITM_STOPPED=""
cleanup() {
  [ -n "$MITM_STOPPED" ] && return
  MITM_STOPPED=1
  kill_tree "$MITM_PID" TERM
  local i
  for i in $(seq 1 15); do
    [ -z "$(tree_pids "$MITM_PID")" ] && { wait "$MITM_PID" 2>/dev/null || true; MITM_PID=""; return; }
    sleep 0.2
  done
  kill_tree "$MITM_PID" KILL
  # Final bounded wait for the whole tree to actually disappear after KILL.
  for i in $(seq 1 10); do
    [ -z "$(tree_pids "$MITM_PID")" ] && break
    sleep 0.1
  done
  wait "$MITM_PID" 2>/dev/null || true
  MITM_PID=""
}
# On ANY exit, purge the token-bearing raw flow (unless KEEP_FLOW=1) so a live
# bearer token is never left on disk. The redacted extract is kept either way.
purge_flow() {
  if [ "${KEEP_FLOW:-0}" = "1" ]; then
    echo "[capture] KEEP_FLOW=1 set: leaving token-bearing flow at $FLOW" >&2
    return
  fi
  if [ -f "$FLOW" ]; then
    command -v shred >/dev/null 2>&1 && shred -u "$FLOW" 2>/dev/null || rm -f "$FLOW"
  fi
}
on_exit() { cleanup; purge_flow; }
# EXIT covers normal end + `exit` builtins. The signal traps cover an interactive
# Ctrl-C (INT), a quit (QUIT), or a kill (TERM/HUP) during the foreground
# claude/extract calls — the common manual aborts — which would otherwise skip the
# purge. Each one tears down, purges, clears the EXIT trap (so it doesn't
# double-run), then re-raises the signal so the exit status honestly reflects the
# interruption. SIGKILL is untrappable by design; the tmpfs-only base (above) is the
# backstop for that and for crashes/power loss.
on_signal() { local sig="$1"; trap - EXIT; on_exit; trap - "$sig"; kill -s "$sig" $$; }
trap on_exit EXIT
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP'  HUP
trap 'on_signal QUIT' QUIT

# Wait for the listener to accept connections (bounded). Fail LOUD if mitmdump
# died (e.g. bind error) instead of silently producing an empty flow.
#
# On the pre-picked-port TOCTOU: mitmdump binds 127.0.0.1 only, and the liveness
# check below runs BEFORE the readiness probe each iteration. Two processes cannot
# both bind 127.0.0.1:$PORT — so if anything raced the port, mitmdump's bind fails,
# it exits, and `kill -0` trips the FATAL path before any token is sent. Winning the
# race to pre-bind a loopback ephemeral port already requires local code execution
# on this dev box; the residual exposure beyond that is not reachable here.
bound=""
for _ in $(seq 1 30); do
  if ! kill -0 "$MITM_PID" 2>/dev/null; then
    echo "[capture] FATAL: mitmdump exited during startup. Log:" >&2
    cat "$MITMLOG" >&2 || true
    exit 1
  fi
  if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then exec 3>&- 3<&-; bound=1; break; fi
  sleep 0.3
done
if [ -z "$bound" ]; then
  echo "[capture] FATAL: mitmdump did not accept connections on :$PORT within timeout. Log:" >&2
  cat "$MITMLOG" >&2 || true
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

# Give mitmdump a moment to flush the flow, then stop just the proxy so the .flow
# is closed before we read it. Keep the EXIT trap armed so purge_flow still shreds
# the token-bearing flow afterwards.
sleep 1
cleanup

echo "[capture] extracting $FLOW -> $EXTRACT" >&2
if command -v mitmdump >/dev/null 2>&1 && python3 -c "import mitmproxy" 2>/dev/null; then
  python3 "$SCRIPT_DIR/extract_flow.py" "$FLOW" --body-bytes 8000 | tee "$EXTRACT"
else
  uv tool run --from mitmproxy python3 "$SCRIPT_DIR/extract_flow.py" "$FLOW" --body-bytes 8000 | tee "$EXTRACT"
fi
echo "[capture] done. workdir=$WORKDIR extract=$EXTRACT (redacted)" >&2
if [ "${KEEP_FLOW:-0}" = "1" ]; then
  echo "[capture] WARNING: KEEP_FLOW=1 — $FLOW still holds a LIVE OAuth token. Delete it when done." >&2
fi
