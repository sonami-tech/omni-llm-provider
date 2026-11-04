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
when the thinking-budget ladder cannot map the value (e.g. `xhigh` on a model
without `output_config.effort`; the active pin has none).
Precedence remains **client effort > pin default > absent**. Native Anthropic
`/v1/messages` also forwards client `output_config` as given and fills the pin
default only when the client omits it (issue #22; see Provider Extras below).
Operator summary: `docs/README.md`.

## max_tokens and thinking (issue #19)

Omni does **not** raise client `max_tokens` when a thinking budget is larger.
Client `max_tokens` is sent as given. When the client omits `max_tokens`, only
fingerprint wire defaults apply. If the pair is invalid for Anthropic
(`max_tokens <= thinking.budget_tokens`), upstream rejects it; the gateway does
not auto-fix. That can also occur when a wire default is below a client (or
effort-mapped) thinking budget. See `docs/decisions.md`.

## Parallel tool control

Chat and Responses `parallel_tool_calls` is canonical intent, not an extra.
Both booleans invert into `tool_choice.disable_parallel_tool_use`. A forced
`any` or named `tool` is then sent as `auto` with that flag (see Forced tool
choice). With tools and no choice, an explicit flag uses `auto`. `none` stays
`none` without a disabling flag. Without tools the choice is omitted; forced
choices still fail. Native Messages keeps its existing nested control and
validation, and its forced choices are sent as `auto` the same way.

Disabling parallel calls permits at most one call. Forced choices are sent as
`auto`, so a call is not guaranteed. Mock serialization does not establish
live model acceptance.

## Forced tool choice

Current Claude models reject `tool_choice` `any` and `tool` with an upstream
400, including on token counting. Anthropic's list of affected models is the
source of truth:
<https://platform.claude.com/docs/en/api/errors#forced-tool-use-not-supported>.

Omni maps OpenAI `required` to `any` and a named function to `tool`. At the
end of request preparation, `relax_forced_tool_choice` in `translate.rs` sends
every forced choice as `{"type":"auto"}`. The same rule runs on Chat
Completions, Responses, native `/v1/messages`, and
`/v1/messages/count_tokens`, streaming or not, for every model and endpoint.
`disable_parallel_tool_use` keeps its value. No other field changes and no
text is added. `auto`, `none`, an omitted choice, and a forced choice with no
tools are unchanged.

When the upstream request is fully built, just before it is sent, the
provider records the relaxed choice. The edge then sets
`x-omni-dropped: tool_choice` (merged with any other value) and logs one
warning with route, model, and choice kind, on success and on upstream or
transport errors after the send attempt.
A required call becomes optional: the model may answer without a tool. See
the Tool choice section in `docs/compatibility-matrix.md`.

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
