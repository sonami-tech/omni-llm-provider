# Design: Single Server Binary, Isolated Providers

## Decision

The workspace ships one server binary: `omni`.

Provider implementations remain separate crates:

- `provider-claude` owns the Claude Code fingerprint invariant, credentials,
  Anthropic Messages translation, streaming parser, and Claude model catalog.
- `provider-grok` owns the xAI wire mapping, credential resolution, streaming
  parser, and Grok model catalog.
- `omni-common` owns shared OpenAI-compatible HTTP conversion, Responses
  conversion, SSE framing, auth, stats, replacements, and error envelopes.
- `omni-core` owns canonical types and the `LlmProvider` trait.
- `crates/bin/omni` owns server startup, routing, auth wiring, stats wiring, and
  model catalog aggregation.

## Why One Binary

- Users run one local endpoint for Claude and Grok.
- Auth, stats, HTTP routes, and model-list behavior have one implementation.
- Provider crates still protect provider invariants; no Claude cch or
  fingerprint logic moves into `omni`.
- Multi-provider ambiguity is explicit: with multiple providers enabled, model
  IDs require `claude:` or `grok:` prefixes. With one provider enabled, bare
  model names route to that provider.

## HTTP Surface

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/models`, `GET /models`
- `GET /stats`
- `GET /health`, `GET /`

## Build

```bash
cargo build -p omni
cargo run -p omni -- --providers claude,grok --port 18321
```

## Non-Goals

- Do not merge provider internals into `omni`.
- Do not route bare model names heuristically when more than one provider is
  enabled.
- Do not add provider-specific server binaries unless there is a concrete
  compatibility requirement.

## Runtime State

Stats default to `omni-stats.redb` under the OS temp directory. Production or
multi-instance runs should set `--stats-db` / `OMNI_STATS_DB` to a durable,
instance-specific path.
