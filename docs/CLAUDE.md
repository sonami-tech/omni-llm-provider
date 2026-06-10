# omni-llm-provider

OpenAI-compatible proxy (Rust 2024, cargo workspace) that aggregates Anthropic
Claude and xAI Grok behind one OpenAI Chat Completions surface. Claude is reached
with a byte-exact "Claude Code" wire fingerprint so Anthropic's subscription
OAuth gate accepts the requests; Grok uses the standard OpenAI-compatible xAI
endpoint.

## Layout

This is a virtual workspace (no root `[package]`). Members:

- `crates/omni-core` — canonical types + the `LlmProvider` trait (`send`,
  `send_stream`). The cross-backend contract.
- `crates/omni-common` — shared infrastructure: `http` (OpenAI request/response
  types, `to_canonical`/`from_canonical`, SSE stream framing), `auth`,
  `stats` (persistent redb), `replacements` (TOML), `error`, `session`.
- `crates/provider-claude` — Claude-specific logic, all isolated here:
  `fingerprint` (profiles, cch, billing header, wire defaults, model catalog),
  `credentials`, `translate` (Canonical <-> Anthropic Messages + identity
  injection), `upstream` (HTTP client, retry/refresh, bounded SSE).
- `crates/provider-grok` — xAI Grok provider (OpenAI-compatible upstream).
- `crates/bin/{omni,omni-claude,omni-grok}` — the three binaries.

Nothing Claude-specific (cch, betas, preamble, profiles, billing suffix) is
allowed in `omni-*`; that isolation is what protects the fingerprint invariant.

## Build / test / run

```sh
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --workspace        # hermetic: live tests skip when creds are absent

cargo run -p omni -- --providers claude,grok --port 18323
cargo run -p omni-claude -- --no-auth --port 18401
cargo run -p omni-grok -- --no-auth --port 18402
```

Tests are hermetic: any test that would hit a real upstream guards on credential
presence and skips (printing why) when absent, so `cargo test` is green offline
and never burns subscription quota. CI runs the four gates above on push/PR.

## Credentials (read fresh per request, never cached)

- Claude OAuth: `~/.claude/.credentials.json` (override `$CLAUDE_CREDENTIALS_PATH`).
  On a 401 the client re-reads once and retries (picks up a CLI refresh).
- Grok: file-only (no env-var key), mirroring Claude. Sources in precedence order:
  `$XAI_CREDENTIALS_PATH` -> `~/.xai/.credentials.json` (static key `{"apiKey":"xai-..."}`)
  -> `~/.grok/auth.json` (the Grok CLI's own OIDC login, auto-detected just like Claude
  reads `~/.claude`). The OIDC JWT is read read-only; on expiry omni warns and the user
  re-runs the Grok CLI login (omni never refreshes or rewrites that file).

## Routing (omni aggregator)

Model prefix selects the backend: `claude:<model>` / `grok:<model>`
(case-insensitive). With a single provider enabled, a bare model name routes to
it; with multiple enabled, a bare name is rejected to force an explicit prefix.
The single-backend binaries (`omni-claude`, `omni-grok`) skip routing entirely.

## HTTP surface

`POST /v1/chat/completions` (non-stream JSON or `stream:true` SSE),
`GET /v1/models` + `/models`, `GET /stats`, `GET /health`, `GET /`. The OpenAI
Responses API (`/v1/responses`) is not implemented.

## Fingerprint exactness (the core invariant)

For every Claude Code version it supports, the Claude path MUST reproduce that
version's wire fingerprint **byte-for-byte**: the version string, `anthropic-beta`
flags, stainless versions, the `x-anthropic-billing-header` cch checksum, the
billing suffix, the system preamble, the model catalog, and the wire defaults
(`max_tokens` / `temperature` / `output_config.effort` per model). An inexact
fingerprint is eventually rejected by Anthropic's subscription OAuth gate;
"close" is a failure, not a partial success.

All of this lives in `crates/provider-claude/src/fingerprint.rs`, with the
captured values held as dated constants (currently profiles `cc-2.1.142` through
`cc-2.1.165`) and pinned by unit tests in that file and in `translate.rs` (e.g.
`wire_defaults_applied_for_default_request_matches_capture`). When the installed
Claude Code CLI is newer than the newest pinned profile, capture real traffic for
the new version and add a profile (constants + per-model betas + wire overrides),
then update the pinning tests to the new captured bytes. The live test
(`provider-claude` `claude_send_exercises_full_fingerprint_path`, creds-gated)
proves Anthropic *accepts* the profile; the offline unit pins prove the bytes are
exact. Both are required for a rebaseline.
