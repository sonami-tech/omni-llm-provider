# Claude Provider Notes

`omni` is the only server binary, but Claude-specific behavior stays isolated in
`crates/provider-claude`.

## Layout

- `crates/provider-claude/src/fingerprint.rs` - active fingerprint pin, cch
  billing header, per-model betas, system preamble, model catalog, and wire
  defaults.
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

Nothing Claude-specific, including cch, betas, preamble, fingerprint pin,
billing suffixes, or Claude Code header values, belongs in `omni`.

## Run

```bash
cargo run -p omni -- --no-auth --port 18321
cargo run -p omni -- --providers claude --no-auth --port 18321
cargo run -p omni -- --providers claude,grok,codex --no-auth --port 18321
```

With both providers enabled, canonical ids such as `claude-sonnet-5` and
aliases such as `sonnet` are accepted when unique. Use prefixed model IDs such
as `claude:sonnet` only when you need to force a provider.

## Fingerprint Invariant

Omni ships one Claude Code pin. That pin must reproduce the captured wire
fingerprint byte-for-byte: version string, `anthropic-beta` flags, stainless
versions, no-cch `x-anthropic-billing-header`, `cc_version` suffix,
system preamble, model catalog, and wire defaults. An inexact
fingerprint is a failure, not a partial success.

Rebaseline overwrites this single pin (issue #12). Multi-version flags are
removed; use an older Omni release for older wire.

The offline unit tests pin the captured bytes. Live tests are credential-gated
and prove Anthropic accepts the current pin when the account has capacity.

Omni's `/v1/messages` and `/v1/messages/count_tokens` routes are native
Anthropic inbound routes for Claude only. They do not use canonical OpenAI
framing, but they still run through the same Claude provider fingerprint,
credential, retry, identity, and cch machinery before reaching Anthropic.

`ANTHROPIC_BASE_URL` switches Claude to a custom Anthropic-compatible gateway.
In that mode, gateway auth is taken from `ANTHROPIC_AUTH_TOKEN`,
`ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS` only; local Claude OAuth
credentials are not read or sent to the custom host.

Rebaseline procedure and tooling are documented at
`docs/providers/claude/REBASELINE.md` and
`tools/providers/claude/fingerprint/`.
