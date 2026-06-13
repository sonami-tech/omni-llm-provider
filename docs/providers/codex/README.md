# Codex Provider

Codex-specific behavior lives in `crates/provider-codex`.

## Source Of Truth

- Provider implementation and Responses mapping:
  `crates/provider-codex/src/lib.rs`
- Codex configuration source: `$CODEX_HOME/config.toml` or
  `~/.codex/config.toml`
- Codex auth source: `$CODEX_HOME/auth.json`, `~/.codex/auth.json`, configured
  env vars, or configured auth commands
- Omni routing and model catalog aggregation: `crates/bin/omni/src/main.rs`

## Invariant

Codex is an OpenAI-compatible backend for Omni's OpenAI inbound surfaces:

- `/v1/chat/completions` non-streaming
- `/v1/responses` non-streaming

Codex `stream:true` requests fail loudly until native Responses SSE parsing is
implemented in `provider-codex`. They do not use buffered pseudo-streaming.

Anthropic inbound stays Claude-only. Codex does not attempt Anthropic wire
fidelity.

## Config And Auth

The provider reads Codex config fresh per request. The current model comes from
`model`, and custom provider selection comes from `model_provider` plus
`[model_providers.<name>]`.

Supported custom-provider fields:

- `base_url`
- `wire_api = "responses"`
- `requires_openai_auth`
- `env_key`
- `experimental_bearer_token`
- `[model_providers.<name>.auth] command = "..."`
- `http_headers`
- `env_http_headers`
- `query_params`

Auth precedence for a custom provider is:

1. `[model_providers.<name>.auth]` command stdout token
2. `experimental_bearer_token`
3. OpenAI/Codex auth, only when `requires_openai_auth = true`
4. `env_key`
5. no Authorization header

That precedence is intentional: custom provider auth overrides ambient OpenAI
auth unless the config explicitly asks for OpenAI auth. Tests pin both the
override and no-auth cases so `auth.json`, `OPENAI_API_KEY`, or `CODEX_API_KEY`
cannot leak to arbitrary custom provider URLs by default.

The reserved built-in provider id is `openai`. Use `openai_base_url` to point
the built-in OpenAI provider at another base URL, or use a non-reserved
`model_provider` name for custom providers.

Default tests are hermetic and use wiremock. Live Codex calls should remain
explicitly opt-in because they may spend quota and depend on account state.
