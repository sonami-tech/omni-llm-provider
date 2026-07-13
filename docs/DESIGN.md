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
- When no provider list is configured, startup enables all locally detected
  providers. `--providers` / `OMNI_PROVIDERS` remains an explicit override.

## HTTP Surface

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages` dual-mode Anthropic Messages inbound (Claude native, or
  Grok/Codex via canonical translation)
- `POST /v1/messages/count_tokens` Claude-native only (non-Claude → 400)
- `GET /v1/models`, `GET /models`
- `GET /stats`
- `GET /health`, `GET /`

OpenAI-compatible inbound surfaces route through `LlmProvider` and can target
any enabled provider.

Anthropic inbound is **dual-mode** (same shared model resolver as chat):

| Resolved provider | Path |
|---|---|
| **claude** | Native passthrough: fingerprint, cch, raw JSON/SSE. Original body is not run through Anthropic→canonical. |
| **grok** / **codex** | Translated: Anthropic → Canonical → `LlmProvider` → Anthropic JSON/SSE (`omni-common::anthropic`). Best-effort protocol fidelity; lossy fields documented in `docs/anthropic-compat.md`. |

Claude native path stays in `provider-claude`. Mappers live in `omni-common`;
dispatch in `bin/omni`. Providers remain canonical-only on the trait.

Codex supports OpenAI inbound non-streaming and streaming paths by posting to
the Codex Responses API and translating native Responses SSE events into
canonical stream events.

`LlmProvider` remains canonical-only. Native Anthropic methods stay Claude-only
on the Claude provider (not on the shared trait).

## Build

```bash
cargo build -p omni
cargo run -p omni -- --version
cargo run -p omni -- --port 18321
```

## Non-Goals

- Do not merge provider internals into `omni`.
- Do not route unknown or ambiguous bare model names heuristically when more
  than one provider is enabled.
- Do not claim perfect Anthropic wire fidelity on Grok/Codex translated path
  (see `docs/anthropic-compat.md`).
- Do not emit thinking blocks on the translated Anthropic path (v1).
- Do not add a separate `openai` provider id; OpenAI-compat backends use Codex
  (and existing custom endpoints).
- Do not add provider-specific server binaries unless there is a concrete
  compatibility requirement.

## Codex/OpenAI Backend

The Codex backend is implemented in `provider-codex` and implements
`LlmProvider` for OpenAI-compatible inbound surfaces. It reads Codex
configuration from `$CODEX_HOME` or `~/.codex`, including provider overrides
such as `base_url`, `wire_api`, `env_key`, `http_headers`,
`env_http_headers`, query parameters, and command-backed auth. Secret auth
material is resolved per request and is not logged. Anthropic inbound may also
target Codex via the dual-mode translated path (best-effort).

## Custom Upstream Auth

When a provider is pointed at a custom upstream endpoint, that custom
configuration owns auth for that provider and default credentials must not leak:

- Claude forced override: `OMNI_CLAUDE_BASE_URL` wins over
  `ANTHROPIC_BASE_URL` and uses only `OMNI_CLAUDE_AUTH_TOKEN`,
  `OMNI_CLAUDE_API_KEY`, and `OMNI_CLAUDE_CUSTOM_HEADERS`.
- Claude: `ANTHROPIC_BASE_URL` activates custom gateway mode. Omni uses
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, reads them per request, and does not read local Claude OAuth credentials
  for that gateway.
- Grok forced override: `OMNI_GROK_BASE_URL` wins over `GROK_MODELS_BASE_URL`
  and uses only `OMNI_GROK_AUTH_TOKEN`, `OMNI_GROK_API_KEY`, and
  `OMNI_GROK_CUSTOM_HEADERS`.
- Grok: `GROK_MODELS_BASE_URL` activates custom endpoint mode. Omni uses
  `XAI_API_KEY` per request if present, otherwise no Authorization header, and
  does not read the default xAI/Grok credential files.
- Codex forced override: `OMNI_CODEX_BASE_URL` is resolved inside
  `provider-codex`; it feeds detection, catalog, aliases, and request config,
  and uses only `OMNI_CODEX_AUTH_TOKEN`, `OMNI_CODEX_API_KEY`,
  `OMNI_CODEX_CUSTOM_HEADERS`, `OMNI_CODEX_MODEL`, and `OMNI_CODEX_WIRE_API`.
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
