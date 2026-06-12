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

Default tests use wiremock. Live xAI calls require `OMNI_LIVE_TESTS=1`.
