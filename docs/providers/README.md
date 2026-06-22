# Provider Maintenance

Omni has one server binary, but provider maintenance stays provider-specific.

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
provider extras, except gateway metadata such as `user`. The selected provider
validates extras against its allowlist before dispatch. Unsupported extras fail
loudly with a request error.

Current allowlists:

- Grok: `service_tier`, `search_parameters`, `response_format`,
  `parallel_tool_calls`, `seed`, `stop`, `n`, `tools`
- Codex: `store`, `previous_response_id`, `metadata`,
  `parallel_tool_calls`, `service_tier`, `response_format`, `text`
- Claude OpenAI-compatible path: no provider extras passthrough
- Claude native: closed Anthropic request allowlist only
