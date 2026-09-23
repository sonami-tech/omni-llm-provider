# Codex Provider

Codex-specific behavior lives in `crates/provider-codex`.

Single pin: Codex CLI **0.156.1**. `--codex-version` / `OMNI_CODEX_VERSION` and
match-system flags are removed (issue #12). Rebaseline overwrites this pin;
older wire needs an older Omni release.

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
- `text`

Each is copied through unchanged, with one exception: when a non-null
`response_format` is also present, Omni merges the translated format into the
`text` object it forwards. See the merge rules below.

Codex also accepts Chat Completions `response_format`, but never forwards that
key. Responses has no `response_format`, so Omni translates it into
`text.format` and drops the original (issue #32). Translation rules:

- `{"type": "text"}` and `{"type": "json_object"}` map 1:1 into `text.format`.
- `{"type": "json_schema", ...}` unwraps the nested `json_schema` object into a
  flat `text.format` carrying `type`, `name`, and `schema`, plus `description`
  and `strict` when present. Null `description` / `strict` are omitted.
- `response_format: null` counts as absent. Omni does not invent `text`.

These `response_format` shapes are a 400:

- A non-null non-object, or a missing or non-string `type`.
- Any `type` other than `text`, `json_object`, or `json_schema`.
- An unknown field under `response_format`, or under
  `response_format.json_schema`.
- For `json_schema`: a missing or non-object `json_schema` wrapper, a missing
  or non-string `name`, a missing or non-object `schema`, a non-null
  non-string `description`, or a non-null non-boolean `strict`.

The merge into `text` keeps client intent:

- No `text` extra: Omni creates `{"format": <mapped>}`.
- `text` without `format`: Omni adds `format` and keeps siblings such as
  `verbosity`.
- `text.format` already equal to the mapped value: accepted unchanged.
- `text.format` set to anything else: 400, never a silent overwrite.
- `text` present but not an object, `text: null` included: 400. A null `text`
  is not treated as absent the way `response_format: null` is. These rules
  apply only when a non-null `response_format` is also present. On its own,
  `text` is forwarded exactly as the client sent it.

Both Codex transports build the request from the same body builder, so the REST
path and the ChatGPT WebSocket path send the same `text` and `text.format`.

Unsupported extras fail loudly.

## Capture

Rebaseline overwrites the single pin, including the model catalog. The catalog
source is `codex debug models --bundled` (`visibility=list` slugs). A custom
Responses `base_url` is not a reason to skip. If that command fails or lists
no models, stop. Do not keep the previous pin's catalog. The 2026-09-23
capture returned a successful Responses POST on the configured custom endpoint.
`visibility=list` includes `gpt-6-astra`, `gpt-6-sol`, `gpt-6-luna`,
`gpt-5.6-sol`, `gpt-5.6-terra`, `gpt-5.6-luna`, and `gpt-5.5`. The user-agent
comes from that successful exec, not the bundled catalog.

```sh
python3 -m tools.capture catalog --provider codex
```

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
and records traffic through a local mitmproxy. If the selected custom provider
uses `OPENAI_API_KEY` or `CODEX_API_KEY` as `env_key`, explicitly set
`OMNI_CAPTURE_CODEX_ENV_KEY_HOST` to the selected non-reserved provider's
HTTPS `base_url` hostname before a live capture; otherwise the runner refuses
to forward an ambient key. The reserved `openai` provider, active Codex
profiles, and URLs with userinfo or non-ASCII/escaped host syntax cannot use
this exception. Do not set the approval variable from an untrusted config. Live runs remove the tmpfs workdir
(including staged credential copies) by default. `KEEP_FLOW=1` retains the
workdir and raw flow on tmpfs and prints warnings. Extract with:

```sh
python3 -m tools.capture extract flow <capture.flow> --provider codex
```

Raw `.flow` files contain live bearer tokens. Keep them on tmpfs only and never
commit them.

Refresh capture requires `auth.json`; API-key-only Codex setups cannot prove the
OAuth refresh path.
