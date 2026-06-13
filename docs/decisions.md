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
- Grok: `$XAI_CREDENTIALS_PATH`, `~/.xai/.credentials.json`, or
  `~/.grok/auth.json`
- Codex: `$CODEX_HOME` or `~/.codex` config and auth state.

Custom upstream endpoint configuration owns provider auth and must not fall
back to default credentials:

- Claude: `ANTHROPIC_BASE_URL` enables custom gateway mode using
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, resolved per request.
- Grok: `GROK_MODELS_BASE_URL` enables custom endpoint mode using
  `XAI_API_KEY` per request only; if it is absent, no Authorization header is
  sent.
- Codex: Codex custom provider config uses
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, or
  `env_key`; it uses OpenAI auth only when `requires_openai_auth = true`.

Codex OpenAI inbound support is non-streaming for now. Codex `stream:true`
requests fail loudly until native Responses SSE is implemented.

## Tests

- Default tests are hermetic and must not call live providers.
- Live provider tests require `OMNI_LIVE_TESTS=1` plus usable credentials.
- Subprocess HTTP tests use shared Rust helpers in `omni-common::test_support`
  instead of shelling out to `curl`.
