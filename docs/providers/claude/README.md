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
