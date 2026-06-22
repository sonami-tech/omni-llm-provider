# Codex Provider

Codex-specific behavior lives in `crates/provider-codex`.

## Source Of Truth

- Provider implementation and Responses mapping:
  `crates/provider-codex/src/lib.rs`
- Codex configuration source: `$CODEX_HOME/config.toml` or
  `~/.codex/config.toml`
- Codex auth source: `CODEX_API_KEY`, `OPENAI_API_KEY`,
  `CODEX_ACCESS_TOKEN`, `$CODEX_HOME/auth.json`, `~/.codex/auth.json`,
  configured env vars, or configured auth commands
- Omni routing and model catalog aggregation: `crates/bin/omni/src/main.rs`

## Invariant

Codex is an OpenAI-compatible backend for Omni's OpenAI inbound surfaces:

- `/v1/chat/completions` non-streaming and `stream:true`
- `/v1/responses` non-streaming and `stream:true`

Codex streaming uses native Responses SSE parsing in `provider-codex`; it does
not use buffered pseudo-streaming.

Anthropic inbound stays Claude-only. Codex does not attempt Anthropic wire
fidelity.

## Config And Auth

The provider reads Codex config fresh per request. The current model comes from
`model`, and custom provider selection comes from `model_provider` plus
`[model_providers.<name>]`.

Default auto-detection enables Codex when `OMNI_CODEX_BASE_URL`, a non-empty
`CODEX_API_KEY`, `OPENAI_API_KEY`, `CODEX_ACCESS_TOKEN`, or Codex config/auth
files are present.

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

## Provider Extras

Codex accepts these provider extras on OpenAI-compatible inbound surfaces and
forwards them to the upstream Responses-compatible body:

- `store`
- `previous_response_id`
- `metadata`
- `parallel_tool_calls`
- `service_tier`

Unsupported extras fail loudly.

## Capture

Use the shared capture framework in `tools/capture/` when Codex wire behavior,
auth refresh, or custom `base_url` routing changes:

```sh
# General capture (requires OMNI_CAPTURE_LIVE=1 or --live-capture)
python3 -m tools.capture capture run --provider codex --mode general --live-capture

# Refresh capture forces stale auth and also needs OMNI_CAPTURE_REFRESH=1
python3 -m tools.capture capture run --provider codex --mode refresh \
  --live-capture --refresh-capture

# Dry-run prints the planned mitmdump and codex commands without network I/O
python3 -m tools.capture capture run --provider codex --mode general --dry-run
```

Refresh validation proves traffic to the selected API `base_url` (from
`openai_base_url` or `[model_providers.<id>].base_url`). Separate auth-host proof
awaits a stable observed auth endpoint; do not invent auth hosts.

Dry-run uses placeholder credential paths only. It does not copy real credentials
or create a tmpfs workdir.

The shared CLI copies `config.toml` and `auth.json` into an isolated
`CODEX_HOME`, runs `codex exec -c 'mcp_servers={}' -` with the prompt on stdin,
and records traffic through a local mitmproxy. Live runs remove the tmpfs workdir
(including staged credential copies) by default. `KEEP_FLOW=1` retains the
workdir and raw flow on tmpfs and prints warnings. Extract with:

```sh
python3 -m tools.capture extract flow <capture.flow> --provider codex
```

Raw `.flow` files contain live bearer tokens. Keep them on tmpfs only and never
commit them.

Refresh capture requires `auth.json`; API-key-only Codex setups cannot prove the
OAuth refresh path.
