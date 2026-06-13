# Claude Provider Notes

`omni` is the only server binary, but Claude-specific behavior stays isolated in
`crates/provider-claude`.

## Layout

- `crates/provider-claude/src/fingerprint.rs` - profiles, cch billing header,
  per-model betas, system preamble, model catalog, and wire defaults.
- `crates/provider-claude/src/credentials.rs` - fresh reads from
  `~/.claude/.credentials.json` or `$CLAUDE_CREDENTIALS_PATH`.
- `crates/provider-claude/src/translate.rs` - canonical request/response
  conversion and Claude Code identity injection.
- `crates/provider-claude/src/anthropic_passthrough.rs` - Claude-native
  Anthropic inbound preparation, closed client allowlist, raw response/SSE
  replacement helpers, and count-token body shaping.
- `crates/provider-claude/src/upstream.rs` - Anthropic HTTP client and SSE
  handling.
- `crates/bin/omni` - server routing, auth, stats, `/v1/models`, `/stats`, and
  Claude-only Anthropic inbound route registration.

Nothing Claude-specific, including cch, betas, preamble, profiles, billing
suffixes, or Claude Code header values, belongs in `omni`.

## Run

```bash
cargo run -p omni -- --providers claude --no-auth --port 18321
cargo run -p omni -- --providers claude,grok --no-auth --port 18321
```

With both providers enabled, canonical ids such as `claude-sonnet-4-6` and
aliases such as `sonnet` are accepted when unique. Use prefixed model IDs such
as `claude:sonnet` only when you need to force a provider.

## Fingerprint Invariant

For each supported Claude Code version, the Claude path must reproduce that
version's wire fingerprint byte-for-byte: version string, `anthropic-beta`
flags, stainless versions, `x-anthropic-billing-header` cch checksum, billing
suffix, system preamble, model catalog, and wire defaults. An inexact fingerprint
is a failure, not a partial success.

The offline unit tests pin the captured bytes. Live tests are credential-gated
and prove Anthropic accepts the current profile when the account has capacity.

Omni's `/v1/messages` and `/v1/messages/count_tokens` routes are native
Anthropic inbound routes for Claude only. They do not use canonical OpenAI
framing, but they still run through the same Claude provider fingerprint,
credential, retry, identity, and cch machinery before reaching Anthropic.

Rebaseline procedure and tooling are documented at
`docs/providers/claude/REBASELINE.md` and
`tools/providers/claude/fingerprint/`.
