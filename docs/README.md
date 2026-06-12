# omni-llm-provider (monorepo)

OpenAI-compatible proxy that aggregates Anthropic Claude (via the Claude Code
wire fingerprint for the subscription OAuth gate) and xAI Grok behind one
OpenAI Chat Completions surface.

## Binaries

All three are full proxies (non-stream JSON + streaming SSE, auth, persistent
stats):

- `omni` — aggregator. Routes to either backend by model prefix
  (`claude:...` / `grok:...`) or, when only one provider is enabled, by bare
  model name. Enable providers with `--providers claude,grok` or
  `OMNI_PROVIDERS`.
- `omni-claude` — single backend locked to Claude.
- `omni-grok` — single backend locked to Grok.

## Crates

- `crates/omni-core` — canonical types (`CanonicalRequest`/`Response`/
  `StreamEvent`) and the `LlmProvider` trait (`send` + `send_stream`). The
  cross-backend contract.
- `crates/omni-common` — shared cross-cutting infrastructure: the
  OpenAI-compatible HTTP layer (`http`: request/response types,
  canonical conversion, SSE stream framing), auth middleware, persistent stats
  (redb), TOML replacements, error envelope, session derivation.
- `crates/provider-claude` — all Claude-specific logic (fingerprint profiles,
  cch, credentials, Anthropic Messages translation, identity injection, wire
  defaults, model catalog). Isolated to protect the fingerprint invariant; no
  Claude specifics leak into `omni-*`.
- `crates/provider-grok` — xAI Grok provider (OpenAI-compatible upstream).
- `crates/bin/{omni,omni-claude,omni-grok}` — the three binaries.

## HTTP surface (all binaries)

- `POST /v1/chat/completions` — text + sampling; non-stream JSON, or
  `stream:true` for OpenAI SSE chunks terminated by `data: [DONE]`.
- `GET /v1/models`, `GET /models` — model catalog.
- `GET /stats` — persistent counters (requests, tokens-by-model, errors).
- `GET /health`, `GET /`.

The OpenAI Responses API (`/v1/responses`) is not implemented; the OAI surface
is Chat Completions only.

## Build, run, test

```bash
cargo build --workspace
cargo run -p omni -- --providers grok --no-auth --port 18321   # one backend, no creds gate on startup
cargo test --workspace                                          # hermetic: live tests skip without creds
```

Credentials are read fresh per request (never cached). Both backends are file-only
(no env-var key): omni piggybacks on the CLIs' own logins. Claude reads
`~/.claude/.credentials.json` (the Claude CLI's file; override `$CLAUDE_CREDENTIALS_PATH`).
Grok resolves in precedence order: `$XAI_CREDENTIALS_PATH` ->
`~/.xai/.credentials.json` (a static `{"apiKey":"xai-..."}` key) -> `~/.grok/auth.json`
(the Grok CLI's OIDC login, auto-detected). So a logged-in `grok` CLI Just Works, exactly
as a logged-in Claude CLI does. Tests that would call a real upstream skip cleanly when the
corresponding credentials are absent, so the suite is green offline.

## More

- Architecture rationale and prior art: `DESIGN.md`, `INVESTIGATION.md`.
- Claude fingerprint invariant and rebaseline: `CLAUDE.md`.
- Grok wire details: `grok-gate.md`, `GROK_AND_MULTI_PROVIDER_ACCESS_GUIDE.md`.
- The original CCP source (ported from) lives in `reference-src-claude/`.
