# Key Decisions and Findings

## Architecture

- Decision: one server binary (`omni`) with separate provider crates.
- Rationale:
  - Claude fingerprint logic is isolated in `provider-claude`.
  - Grok wire logic is isolated in `provider-grok`.
  - Codex config and Responses wire logic is isolated in `provider-codex`.
  - Shared HTTP conversion, Responses conversion, auth, stats, replacements,
    session derivation, conversation logging, and error envelopes live in
    `omni-common`.
  - `omni` only routes, frames responses, exposes catalogs, records stats, and
    wires optional conversation logging.

## Routing

- Prefix routing selects the backend: `claude:<model>`, `grok:<model>`, or
  `codex:<model>`.
- With exactly one provider enabled, bare model names are accepted.
- With multiple providers enabled, bare model names are accepted only when the
  model id or alias uniquely matches one provider catalog.
- Anthropic inbound (`/v1/messages` and `/v1/messages/count_tokens`) is
  Claude-only. Non-Claude prefixes or models return an Anthropic-shaped request
  error instead of falling back to another provider.

## Provider Boundaries

- Claude: cch, betas, preamble, profiles, model aliases, credentials, and
  Anthropic wire defaults stay in `provider-claude`. Native Anthropic inbound
  reconciliation, raw JSON passthrough, raw SSE forwarding, and count-token body
  shaping also stay there.
- Grok: xAI request/response mapping, streaming parsing, credential resolution,
  and model catalog stay in `provider-grok`.
- Codex: Codex config discovery, auth parsing, provider override handling, and
  OpenAI-compatible Responses wire mapping stay in `provider-codex`.
- Server concerns: auth, stats, bind/public flags, route registration, and model
  routing stay in `omni`.

## Current Surfaces

- OpenAI Chat Completions: `/v1/chat/completions`
- OpenAI Responses subset: `/v1/responses`
- Anthropic Messages, Claude only: `/v1/messages`
- Anthropic token count, Claude only: `/v1/messages/count_tokens`
- Models: `/v1/models`, `/models`
- Stats: `/stats`
- Health/root: `/health`, `/`

## Credentials

Credentials are read fresh per request.

- Claude: `$CLAUDE_CREDENTIALS_PATH` or `~/.claude/.credentials.json`
- Grok: `$XAI_CREDENTIALS_PATH`, a usable `~/.xai/.credentials.json`, or
  `~/.grok/auth.json`
- Codex: `CODEX_API_KEY`, `OPENAI_API_KEY`, `CODEX_ACCESS_TOKEN`, or
  `$CODEX_HOME` / `~/.codex` config and auth state.

Omni refreshes Claude/Codex/Grok OAuth primary-login tokens in-place by default
(atomic write-back of rotated refresh tokens). Disable with
`--no-oauth-refresh`, `OMNI_NO_OAUTH_REFRESH=1`, or `OMNI_OAUTH_REFRESH=0` (or
`false`/`off`/`no`) to keep CLI-delegated re-read only. Static API keys are
never refreshed. Write-back preserves file mode, requires a rotated
`refresh_token` in the grant, and refuses to clobber if the on-disk RT changed
mid-refresh (concurrent CLI/Omni). Full flock parity with vendor CLI lockfiles
is not claimed.

Custom upstream endpoint configuration owns provider auth and must not fall
back to default credentials:

- Omni forced overrides are highest precedence:
  - Claude: `OMNI_CLAUDE_BASE_URL` uses only `OMNI_CLAUDE_AUTH_TOKEN`,
    `OMNI_CLAUDE_API_KEY`, and `OMNI_CLAUDE_CUSTOM_HEADERS`.
  - Grok: `OMNI_GROK_BASE_URL` uses only `OMNI_GROK_AUTH_TOKEN`,
    `OMNI_GROK_API_KEY`, and `OMNI_GROK_CUSTOM_HEADERS`.
  - Codex: `OMNI_CODEX_BASE_URL` is owned by `provider-codex` and uses only
    `OMNI_CODEX_AUTH_TOKEN`, `OMNI_CODEX_API_KEY`,
    `OMNI_CODEX_CUSTOM_HEADERS`, `OMNI_CODEX_MODEL`, and
    `OMNI_CODEX_WIRE_API`.
- Claude: `ANTHROPIC_BASE_URL` enables custom gateway mode using
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, resolved per request.
- Grok: `GROK_MODELS_BASE_URL` enables custom endpoint mode using
  `XAI_API_KEY` per request only; if it is absent, no Authorization header is
  sent.
- Codex: Codex custom provider config uses
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, or
  `env_key`; it uses OpenAI auth only when `requires_openai_auth = true`.

Codex OpenAI inbound supports non-streaming and `stream:true` paths. Streaming
uses native Responses SSE parsing in `provider-codex`, not buffered
pseudo-streaming.

## Observability

- Every request runs inside an `info_span!("request", request_id, session_id,
  provider)` opened by a middleware layer in `crates/bin/omni`. `request_id` is
  generated once in the layer and shared via a request extension so the span id,
  the response id, and the conversation-log `request` all derive from one value.
  `session_id` and `provider` are recorded late by the handlers.
- SSE streams outlive the handler, so the span is attached to them with a
  per-poll adapter (`SpannedStream`), NOT by holding a `Span::enter` guard across
  the stream's awaits. Holding a guard across `.await` leaves the span entered on
  the worker thread while the task is suspended, so a different concurrent request
  resuming on that thread would log under the wrong `request_id`. A concurrency
  test asserts no such cross-request bleed.
- `OMNI_LOG_COLOR` (`auto|always|never`, plus `NO_COLOR` and stderr TTY
  detection) gates colorized log fields (`crates/bin/omni/src/log_color.rs`):
  `request_id`/`session_id` get stable hashed hues, each provider a fixed color.
  The formatter sanitizes ANSI escapes in every value (matching upstream
  `tracing-subscriber`), so a provider echoing raw upstream bytes cannot inject
  terminal control sequences into the operator's log.

## Tests

- Default tests are hermetic and must not call live providers.
- Live provider tests require `OMNI_LIVE_TESTS=1` plus usable credentials.
- Subprocess HTTP tests use shared Rust helpers in `omni-common::test_support`
  instead of shelling out to `curl`.
