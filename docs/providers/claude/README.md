# Claude Provider

Claude-specific behavior lives in `crates/provider-claude`.

## Source Of Truth

- Profiles, cch, beta flags, preamble, wire defaults:
  `crates/provider-claude/src/fingerprint.rs`
- Model catalog: `crates/provider-claude/src/models.rs`
- Credentials: `crates/provider-claude/src/credentials.rs`
- Upstream HTTP and streaming: `crates/provider-claude/src/upstream.rs`
- Rebaseline procedure: `docs/providers/claude/REBASELINE.md`
- Fingerprint tooling: `tools/providers/claude/fingerprint/`

## Invariant

For every supported Claude Code version, Omni must reproduce that version's
wire fingerprint exactly enough for the Claude OAuth subscription path:
version string, `anthropic-beta`, stainless versions, billing header cch,
billing suffix, system preamble, model catalog, and wire defaults.

Offline tests pin the captured bytes. Live Anthropic calls are opt-in via
`OMNI_LIVE_TESTS=1`.

## Provider Extras

Claude's OpenAI-compatible path has no provider extras passthrough today.
Unsupported provider extras fail loudly before a fingerprint-sensitive wire
request is built.

Claude native `/v1/messages` uses a closed request allowlist. Fields such as
`betas`, `metadata`, `service_tier`, `mcp_servers`, `container`, and
`output_config` remain intentionally unsupported unless a fingerprint rebaseline
proves they belong on the Claude Code path.
