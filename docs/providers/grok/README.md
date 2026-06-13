# Grok Provider

Grok-specific behavior lives in `crates/provider-grok`.

## Source Of Truth

- xAI mapping and streaming parser: `crates/provider-grok/src/lib.rs`
- Credential loader: `crates/omni-common/src/credentials.rs`
- Gate notes: `docs/grok-gate.md`
- Capture procedure: `docs/providers/grok/CAPTURE.md`
- Capture tooling: `tools/providers/grok/capture/`

## Invariant

Grok does not currently require Claude-style byte-exact release profiles,
billing cch, or injected identity preambles. The maintained contract is:

- fresh credential resolution per request,
- correct xAI/OpenAI-compatible request body,
- expected auth and content headers,
- correct non-stream and stream decoding,
- model catalog kept current.

Default Grok mode resolves credentials from `$XAI_CREDENTIALS_PATH`,
`~/.xai/.credentials.json`, then `~/.grok/auth.json`. Custom endpoint mode is
different by design: `GROK_MODELS_BASE_URL` switches the provider to use
`XAI_API_KEY` per request only, or no Authorization header if it is unset. The
default xAI credential files must not be sent to a custom endpoint.

Default tests use wiremock. Live xAI calls require `OMNI_LIVE_TESTS=1`.
