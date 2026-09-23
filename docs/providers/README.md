# Provider Maintenance

Omni has one server binary, but provider maintenance stays provider-specific.

## Breaking change (issue #12)

Each provider ships **one live pin**. Multi-version selection is gone:
`--claude-version`, `--grok-version`, `--codex-version`,
`--match-system`, `--match-system-exact`, and env counterparts
`OMNI_CLAUDE_VERSION`, `OMNI_GROK_VERSION`, `OMNI_CODEX_VERSION`,
`OMNI_MATCH_SYSTEM`, `OMNI_MATCH_SYSTEM_EXACT` are removed. Rebaseline
**overwrites** the pin. Use an older Omni release for older wire.

A rebaseline must obtain a **fresh catalog listing** from this pin's source.
If that listing cannot be read, stop and surface the error. Do not keep the
previous pin's model list.

Catalog sources:

- Claude: model ids on captured `POST /v1/messages`
- Grok: `GET /v1/models` on `cli-chat-proxy.grok.com` from a **clean-HOME**
  capture. Do not use the operator `grok models` UI or
  `~/.grok/config.toml` `[models] default`. Those can be local overrides
  (custom models, a non-Grok default). The pin default is the no-`--model`
  chat body plus the proxy `default_model` / `/v1/models` list.
- Codex: `codex debug models --bundled` (not ChatGPT `/codex/models`; a custom
  Responses `base_url` is not a reason to skip). Do not treat
  `~/.codex/config.toml` `model =` as the catalog.

```sh
python3 -m tools.capture catalog --provider claude --flow-file <capture.flow>
python3 -m tools.capture catalog --provider grok --flow-file <capture.flow>
python3 -m tools.capture catalog --provider codex
```

General live capture runs this check and fails closed if the listing is empty.

- Claude: `docs/providers/claude/README.md`
- Grok: `docs/providers/grok/README.md`
- Codex: `docs/providers/codex/README.md`
- Shared capture and refresh-capture tooling: `tools/capture/`

Default tests are hermetic. Any test or tool that calls a live provider, spends
quota, or captures credentials must be explicitly opted into and run by an
operator.

## Capture Policy

Use the shared Python capture framework for provider wire baselines and OAuth
refresh-capture work:

```sh
python3 -m tools.capture capture run --provider claude --mode general --dry-run
python3 -m tools.capture capture run --provider grok --mode general --dry-run
python3 -m tools.capture capture run --provider codex --mode general --dry-run
```

Live capture requires `--live-capture` or `OMNI_CAPTURE_LIVE=1`. Refresh capture
also requires `--refresh-capture` or `OMNI_CAPTURE_REFRESH=1`.

## Live Test Policy

Normal verification:

```sh
cargo test --workspace
```

Live provider tests require both credentials and:

```sh
OMNI_LIVE_TESTS=1 cargo test --workspace
```

Do not set `OMNI_LIVE_TESTS=1` in CI or shared shell profiles. Live tests may
spend quota and fail on provider rate limits, account state, or model access.

## Provider Extras

OpenAI-compatible inbound surfaces preserve top-level extension fields as
provider extras, except gateway metadata such as `user`. Official cache fields
are not extras: Chat Completions, Responses, and Anthropic Messages parse them
into internal cache intent. Outbound backends receive that backend's official
cache fields only (Grok Chat and Responses send body `prompt_cache_key` and
never `x-grok-conv-id`; Claude consumes routing identity and emits
`cache_control`; Codex REST and ChatGPT WS share one cache payload). Anthropic
inbound `prompt_cache_key` is 400. See
[`docs/cache-translation.md`](../cache-translation.md).

The selected provider validates remaining extras against its allowlist before
dispatch. Unsupported extras fail loudly with a request error.

Current allowlists:

- Grok: `service_tier`, `search_parameters`, `response_format`,
  `parallel_tool_calls`, `seed`, `stop`, `n`, `tools`
- Codex: `store`, `previous_response_id`, `metadata`,
  `parallel_tool_calls`, `service_tier`, `text`; chat `response_format: null`
  acts as absent. Non-null chat `response_format` is validated and translated
  into Responses `text.format`, not forwarded as `response_format`. It creates
  `text` when absent, or merges into object `text` while keeping siblings;
  equal existing `format` is accepted. Conflicting `text.format`, non-object
  `text` (including null) during a merge, and invalid non-null
  `response_format` are 400. Without non-null `response_format`, `text` is
  forwarded unchanged. Unsupported extras fail loudly.
  See [Codex provider details](codex/README.md).
- Claude OpenAI-compatible path: no provider extras passthrough
- Claude native: closed Anthropic request allowlist only
