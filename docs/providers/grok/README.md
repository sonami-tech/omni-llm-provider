# Grok Provider

Grok-specific behavior lives in `crates/provider-grok`.

## Source Of Truth

- xAI mapping and streaming parser: `crates/provider-grok/src/lib.rs`
- Credential loader: `crates/provider-grok/src/credentials.rs`
- Gate notes: `docs/grok-gate.md`
- Capture procedure: `docs/providers/grok/CAPTURE.md`
- Shared capture framework: `tools/capture/`
- Compatibility capture wrappers: `tools/providers/grok/capture/`

## Invariant

Grok does not currently require Claude-style billing cch or injected identity
preambles. The maintained contract is:

- CLI-parity wire to `cli-chat-proxy.grok.com` (`POST /v1/responses`),
- fresh credential resolution per request (prefers `~/.grok/auth.json` OIDC),
- grok-shell fingerprint headers for the pinned CLI version,
- correct non-stream and stream decoding via the shared Responses parser,
- model catalog kept current (latest pin: grok-shell 0.2.101; `grok-4.5` only).

Default path credentials: `$XAI_CREDENTIALS_PATH`, then `~/.grok/auth.json`, then
`~/.xai/.credentials.json`. Custom endpoint mode is different by design.
`OMNI_GROK_BASE_URL` is Omni's forced override and uses only
`OMNI_GROK_AUTH_TOKEN`, `OMNI_GROK_API_KEY`, and `OMNI_GROK_CUSTOM_HEADERS`.
Legacy `GROK_MODELS_BASE_URL` remains supported and uses `XAI_API_KEY` per
request only, or no Authorization header if it is unset. Default CLI credentials
must not be sent to a custom endpoint.

Default tests use wiremock. Live calls require `OMNI_LIVE_TESTS=1`.

Capture and refresh-capture work uses `python3 -m tools.capture`; see
`docs/providers/grok/CAPTURE.md`.

## Provider Extras

Grok accepts these provider extras on OpenAI-compatible inbound surfaces:

- `service_tier`
- `search_parameters`
- `response_format`
- `parallel_tool_calls`
- `seed`
- `stop`
- `n`
- `tools`

Unsupported extras fail loudly. `previous_response_id` is not forwarded on the
custom chat-completions override path.
