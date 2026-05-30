# Claude Code Provider

OpenAI-compatible API proxy for Claude Max accounts. Written in Rust (2024 edition).

CCP exposes the OpenAI Chat Completions API and talks directly to Anthropic's Messages API using the OAuth token from `~/.claude/.credentials.json`. It mimics Claude Code's wire fingerprint enough for Anthropic's subscription OAuth gate, but it does not invoke the Claude Code CLI per request.

## Build

```sh
cargo build --release
```

## Test

### Unit tests (Rust)

```sh
cargo test
```

### Integration tests (Python/pytest)

```sh
./tests/run.sh
```

This automatically builds the binary, starts CCP server instances on random ports, runs the live integration tests, and tears down. It requires valid local Claude OAuth credentials because completion tests call Anthropic through CCP. Uses `uv run` with inline deps (httpx, openai, pytest, pytest-asyncio).

Useful flags:

```sh
./tests/run.sh -k "TestAuth"           # Run one test class
./tests/run.sh -k "test_health"        # Run one test
./tests/run.sh -x                      # Stop on first failure
./tests/run.sh --co                    # List tests without running
```

## Run

```sh
cargo run -- --no-auth --port 18321
```

## Architecture

- `src/main.rs` - Server setup, router, startup credential validation.
- `src/config.rs` - CLI args and env vars (all prefixed `CCP_`).
- `src/routes/completions.rs` - Main handler entrypoint for streaming and non-streaming.
- `src/routes/completions_v2.rs` - Direct Anthropic Messages request handling.
- `src/upstream/` - Credentials loading, Claude Code fingerprint headers, HTTP client, SSE parser.
- `src/translate/` - OpenAI request/response translation to and from Anthropic Messages API shapes.
- `src/models.rs` - Model definitions, alias resolution.
- `src/auth.rs` - API key middleware.
- `src/stats.rs` - Persistent stats (redb), per-model and per-key tracking.
- `src/replacements.rs` - Text replacement rules (TOML).

## Key design decisions

- CCP reads Claude OAuth credentials from `~/.claude/.credentials.json` fresh for each request and retries once after a 401 with a fresh read.
- Requests include Claude Code-compatible headers, beta flags, and session IDs so subscription OAuth calls are accepted by Anthropic.
- Claude Code fingerprint profiles are selected with `--fingerprint-profile` / `CCP_FINGERPRINT_PROFILE`; `latest` resolves to the newest known-good pinned profile.
- The canonical Claude Code system identifier is prepended by default to satisfy the OAuth gate. `--no-preamble` is for upstream debugging only.
- Tools are passed natively to Anthropic Messages API. PascalCase masking for tool names is often required via text replacement to satisfy OAuth gate fingerprinting.
- Text replacement applies outbound to prompts and tool surfaces, then inbound to assistant text and tool call names/arguments.

## Fingerprint exactness (the core invariant)

For every Claude Code version CCP supports, it MUST reproduce that version's wire fingerprint **byte-for-byte** — the version string, `anthropic-beta` flags, stainless versions, the `x-anthropic-billing-header` cch checksum, the model catalog, and wire defaults. This exactness is the entire point of the application: an inexact fingerprint is eventually rejected by Anthropic's subscription OAuth gate. "Close" is a failure, not a partial success.

When the installed Claude Code CLI is newer than the newest pinned profile (or after any CC update), re-baseline before relying on it. Exactness is won by a **live capture of real Claude Code traffic** (the authoritative source for beta flags, stainless versions, and wire defaults) plus the drift checker (version + cch); the live test suite proves Anthropic *accepts* the profile, not that it is byte-identical. The full procedure is in [`tools/fingerprint/REBASELINE.md`](tools/fingerprint/REBASELINE.md). Start by running `uv run tools/fingerprint/check_claude_code_drift.py` to detect drift — but a green checker is **not** sufficient on its own (it sees only version + cch); a full re-baseline always requires the live capture in Step 3.
