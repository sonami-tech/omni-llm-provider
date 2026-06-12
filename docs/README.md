# omni-llm-provider

OpenAI-compatible Rust proxy that serves Anthropic Claude and xAI Grok from one
`omni` server binary. Provider-specific protocol, credential, and fingerprint
logic remains isolated in provider crates.

## Binary

- `omni` — the only server binary. Routes by model prefix (`claude:...` /
  `grok:...`) or, when only one provider is enabled, by bare model name.

## Crates

- `crates/omni-core` — canonical types and the `LlmProvider` trait.
- `crates/omni-common` — OpenAI-compatible HTTP conversion and SSE framing,
  Responses conversion, auth middleware, persistent stats, replacements, error
  envelope, session derivation.
- `crates/provider-claude` — Claude-specific fingerprint profiles, cch,
  credentials, Anthropic Messages translation, identity injection, wire defaults,
  and model catalog.
- `crates/provider-grok` — xAI Grok provider and OpenAI-compatible xAI wire
  mapping.
- `crates/bin/omni` — server setup, routing, auth, stats, and model catalog
  aggregation.

## HTTP Surface

- `POST /v1/chat/completions` — non-stream JSON or `stream:true` OpenAI SSE
  chunks terminated by `data: [DONE]`.
- `POST /v1/responses` — supported OpenAI Responses subset, non-stream JSON or
  Responses SSE events.
- `GET /v1/models`, `GET /models` — provider-owned model catalogs with prefixed
  IDs.
- `GET /stats` — persistent request, token, and error counters.
- `GET /health`, `GET /`.

## Build, Run, Test

```bash
cargo build --workspace
cargo run -p omni -- --providers claude,grok --no-auth --port 18321
cargo test --workspace
```

Useful server flags:

- `--providers claude,grok` / `OMNI_PROVIDERS`
- `--port 18321` / `OMNI_PORT`
- `--bind 127.0.0.1` / `OMNI_BIND`
- `--public` / `OMNI_PUBLIC` for `0.0.0.0`
- `--stats-db <path>` / `OMNI_STATS_DB`
- `--no-auth` / `OMNI_NO_AUTH`

`OMNI_API_KEYS` enables bearer-token auth when set to a comma-separated key list.

Credentials are read fresh per request, never cached. Claude reads
`~/.claude/.credentials.json` or `$CLAUDE_CREDENTIALS_PATH`. Grok resolves
`$XAI_CREDENTIALS_PATH`, then `~/.xai/.credentials.json`, then
`~/.grok/auth.json`.
