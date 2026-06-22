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

Grok does not currently require Claude-style byte-exact release profiles,
billing cch, or injected identity preambles. The maintained contract is:

- fresh credential resolution per request,
- correct xAI/OpenAI-compatible request body,
- expected auth and content headers,
- correct non-stream and stream decoding,
- model catalog kept current.

Default Grok mode resolves credentials from `$XAI_CREDENTIALS_PATH`, then a
usable `~/.xai/.credentials.json`, then `~/.grok/auth.json`. Custom endpoint mode is
different by design. `OMNI_GROK_BASE_URL` is Omni's forced override and uses
only `OMNI_GROK_AUTH_TOKEN`, `OMNI_GROK_API_KEY`, and
`OMNI_GROK_CUSTOM_HEADERS`. Legacy `GROK_MODELS_BASE_URL` remains supported and
uses `XAI_API_KEY` per request only, or no Authorization header if it is unset.
The default xAI credential files must not be sent to a custom endpoint.

Default tests use wiremock. Live xAI calls require `OMNI_LIVE_TESTS=1`.

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

Unsupported extras fail loudly. `previous_response_id` is not forwarded to Grok
chat completions.
