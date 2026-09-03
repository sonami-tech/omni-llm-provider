# Claude Provider

Claude-specific behavior lives in `crates/provider-claude`.

## Source Of Truth

- Active fingerprint pin, billing header, beta flags, preamble, wire defaults:
  `crates/provider-claude/src/fingerprint.rs`
- Historical cch algorithm (not live): `docs/providers/claude/CCH_ALGORITHM.md`
- Model catalog: `crates/provider-claude/src/models.rs`
- Credentials: `crates/provider-claude/src/credentials.rs`
- Upstream HTTP and streaming: `crates/provider-claude/src/upstream.rs`
- Rebaseline procedure: `docs/providers/claude/REBASELINE.md`
- Shared capture framework: `tools/capture/`
- Claude fingerprint tooling and compatibility wrappers:
  `tools/providers/claude/fingerprint/`

## Invariant

Omni ships one Claude Code pin. That pin must reproduce the captured wire
fingerprint exactly enough for the Claude OAuth subscription path: version
string, `anthropic-beta`, stainless versions, no-cch billing header,
billing suffix, system preamble, model catalog, and wire defaults.

Rebaseline overwrites this single pin. Historical multi-version selection and
flags (`--claude-version`, `OMNI_CLAUDE_VERSION`, match-system) are removed
(issue #12). Use an older Omni release for older wire.

Offline tests pin the captured bytes. Live Anthropic calls are opt-in via
`OMNI_LIVE_TESTS=1`.

## Reasoning effort

OpenAI chat/Responses effort is free-string at the edge (issues #20 / #24). On
the **OpenAI inbound → Claude** path, **effort-capable** models pass free-string
**non-`none`** effort through to `output_config.effort` (issue #26). Only
`minimal`→`low` is remapped (`max` kept as-is). There is no closed local Claude
effort allowlist; unknown names may 400 from upstream by design. Fail loud only
when the thinking-budget ladder cannot map the value (e.g. Haiku + `xhigh`).
Precedence remains **client effort > pin default > absent**. This does not
change native Anthropic `/v1/messages` (closed allowlist still drops client
`output_config`). Operator summary: `docs/README.md`.

## max_tokens and thinking (issue #19)

Omni does **not** raise client `max_tokens` when a thinking budget is larger.
Client `max_tokens` is sent as given. When the client omits `max_tokens`, only
fingerprint wire defaults apply. If the pair is invalid for Anthropic
(`max_tokens <= thinking.budget_tokens`), upstream rejects it; the gateway does
not auto-fix. That can also occur when a wire default is below a client (or
effort-mapped) thinking budget, for example Haiku's 32k default with effort
`max` (budget 32768). See `docs/decisions.md`.

## Provider Extras

Claude's OpenAI-compatible path has no provider extras passthrough today.
Unsupported provider extras fail loudly before a fingerprint-sensitive wire
request is built.

Claude native `/v1/messages` uses a closed request allowlist. Fingerprint and
billing fields (`betas`, `metadata`, `service_tier`, `mcp_servers`, `container`)
remain intentionally unsupported on that door.

`output_config` / effort is **client intent** (issue #22): when the client sets
it, Door-2 passes it through. Capture/pin defaults apply only when the client
left `output_config` unset. Precedence: **client > pin default > absent**
(same idea as OpenAI-compat chat / issues #20 / #24 / #27).
