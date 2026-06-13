# Design: Single Server Binary, Isolated Providers

## Decision

The workspace ships one server binary: `omni`.

Provider implementations remain separate crates:

- `provider-claude` owns the Claude Code fingerprint invariant, credentials,
  Anthropic Messages translation, streaming parser, and Claude model catalog.
- `provider-grok` owns the xAI wire mapping, credential resolution, streaming
  parser, and Grok model catalog.
- `provider-codex` owns Codex config discovery, custom-provider auth
  resolution, and Responses wire mapping.
- `omni-common` owns shared OpenAI-compatible HTTP conversion, Responses
  conversion, SSE framing, auth, stats, conversation logging, session
  derivation, replacements, and error envelopes.
- `omni-core` owns canonical types and the `LlmProvider` trait.
- `crates/bin/omni` owns server startup, routing, auth wiring, stats wiring,
  optional conversation-log wiring, and model catalog aggregation.

## Why One Binary

- Users run one local endpoint for Claude, Grok, and Codex.
- Auth, stats, HTTP routes, and model-list behavior have one implementation.
- Provider crates still protect provider invariants; no Claude cch or
  fingerprint logic moves into `omni`.
- Model routing uses provider-owned catalogs. Bare canonical ids and documented
  aliases route when they uniquely match an enabled provider. `claude:`,
  `grok:`, and `codex:` prefixes remain as an explicit provider escape hatch.

## HTTP Surface

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages` for Claude-native Anthropic Messages inbound
- `POST /v1/messages/count_tokens` for Claude-native Anthropic token counting
- `GET /v1/models`, `GET /models`
- `GET /stats`
- `GET /health`, `GET /`

OpenAI-compatible inbound surfaces route through `LlmProvider` and can target
any enabled provider. Anthropic inbound is different: it is provider-native and
routes only to Claude. The Claude provider owns the closed Anthropic allowlist,
model resolution, identity injection, cch finalization, raw JSON response path,
raw SSE forwarding, and `count_tokens` body shaping.

Codex currently supports the non-streaming OpenAI inbound path. Codex
`stream:true` requests fail loudly until `provider-codex` implements native
Responses SSE parsing.

`LlmProvider` remains canonical-only. Native Anthropic methods are not on the
shared trait because Grok and Codex cannot preserve Anthropic wire fidelity.

## Build

```bash
cargo build -p omni
cargo run -p omni -- --providers claude,grok,codex --port 18321
```

## Non-Goals

- Do not merge provider internals into `omni`.
- Do not route unknown or ambiguous bare model names heuristically when more
  than one provider is enabled.
- Do not emulate Anthropic inbound for non-Claude providers.
- Do not add provider-specific server binaries unless there is a concrete
  compatibility requirement.

## Codex/OpenAI Backend

The Codex backend is implemented in `provider-codex` and implements
`LlmProvider` for OpenAI-compatible inbound surfaces. It reads Codex
configuration from `$CODEX_HOME` or `~/.codex`, including provider overrides
such as `base_url`, `wire_api`, `env_key`, `http_headers`,
`env_http_headers`, query parameters, and command-backed auth. Secret auth
material is resolved per request and is not logged. Anthropic inbound remains
Claude-only.

## Custom Upstream Auth

When a provider is pointed at a custom upstream endpoint, that custom
configuration owns auth for that provider and default credentials must not leak:

- Claude: `ANTHROPIC_BASE_URL` activates custom gateway mode. Omni uses
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, reads them per request, and does not read local Claude OAuth credentials
  for that gateway.
- Grok: `GROK_MODELS_BASE_URL` activates custom endpoint mode. Omni uses
  `XAI_API_KEY` per request if present, otherwise no Authorization header, and
  does not read the default xAI/Grok credential files.
- Codex: Codex config controls custom-provider auth.
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, and
  `env_key` do not fall back to OpenAI auth unless
  `requires_openai_auth = true`.

## Runtime State

Stats default to `omni-stats.redb` under the OS temp directory. Production or
multi-instance runs should set `--stats-db` / `OMNI_STATS_DB` to a durable,
instance-specific path.

Conversation logging is disabled by default. It can write to stderr, a rotating
single file, or per-session files via `--log-conversations`, `--log-file`, or
`--log-dir`. Session ids prefer `x-session-id`, then request `user`, then API-key
id, then an anonymous fallback.
