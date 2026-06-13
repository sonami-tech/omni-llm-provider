# Design: Single Server Binary, Isolated Providers

## Decision

The workspace ships one server binary: `omni`.

Provider implementations remain separate crates:

- `provider-claude` owns the Claude Code fingerprint invariant, credentials,
  Anthropic Messages translation, streaming parser, and Claude model catalog.
- `provider-grok` owns the xAI wire mapping, credential resolution, streaming
  parser, and Grok model catalog.
- `omni-common` owns shared OpenAI-compatible HTTP conversion, Responses
  conversion, SSE framing, auth, stats, conversation logging, session
  derivation, replacements, and error envelopes.
- `omni-core` owns canonical types and the `LlmProvider` trait.
- `crates/bin/omni` owns server startup, routing, auth wiring, stats wiring,
  optional conversation-log wiring, and model catalog aggregation.

## Why One Binary

- Users run one local endpoint for Claude and Grok.
- Auth, stats, HTTP routes, and model-list behavior have one implementation.
- Provider crates still protect provider invariants; no Claude cch or
  fingerprint logic moves into `omni`.
- Model routing uses provider-owned catalogs. Bare canonical ids and documented
  aliases route when they uniquely match an enabled provider. `claude:` and
  `grok:` prefixes remain as an explicit provider escape hatch.

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

`LlmProvider` remains canonical-only. Native Anthropic methods are not on the
shared trait because Grok and the planned Codex/OpenAI backend cannot preserve
Anthropic wire fidelity.

## Build

```bash
cargo build -p omni
cargo run -p omni -- --providers claude,grok --port 18321
```

## Non-Goals

- Do not merge provider internals into `omni`.
- Do not route unknown or ambiguous bare model names heuristically when more
  than one provider is enabled.
- Do not emulate Anthropic inbound for non-Claude providers.
- Do not add provider-specific server binaries unless there is a concrete
  compatibility requirement.

## Planned Codex/OpenAI Backend

The planned Codex backend should be a new provider crate implementing
`LlmProvider` for OpenAI-compatible inbound surfaces. It should read Codex
configuration from `$CODEX_HOME` or `~/.codex`, including provider overrides
such as `base_url` and `wire_api`, and read secret auth material fresh per
request without logging or embedding token values. Anthropic inbound remains
Claude-only after that backend is added.

## Runtime State

Stats default to `omni-stats.redb` under the OS temp directory. Production or
multi-instance runs should set `--stats-db` / `OMNI_STATS_DB` to a durable,
instance-specific path.

Conversation logging is disabled by default. It can write to stderr, a rotating
single file, or per-session files via `--log-conversations`, `--log-file`, or
`--log-dir`. Session ids prefer `x-session-id`, then request `user`, then API-key
id, then an anonymous fallback.
