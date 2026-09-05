# Key Decisions and Findings

## Gateway philosophy (issue #27)

- Decision: Omni is a **many-to-many gateway**. One endpoint serves multiple
  providers and multiple inbound interfaces (OpenAI chat, Responses, Anthropic
  Messages, and related paths). Clients and providers connect without inventing
  a third omni-only request dialect (canonical types are internal only).
- Decision: **Pass through or translate.** Prefer pass-through when the wire
  allows it; otherwise translate into the provider's native form. Fields that
  change the end result must not be silently stripped. Silent omit of
  meaningful client intent is a bug.
- Decision: **Minimal surface / defaults.** Defaults exist only for
  compatibility when the client omits a value, and only in the smallest form
  that keeps the request valid. Do not invent maintenance-heavy defaults that
  go stale.
- Decision: **Client intent first.** Precedence is **explicit client value →
  minimal compat/capture default when absent → absent**. Fingerprint or
  capture fidelity must not win over meaningful client intent (Door-2
  honor-client cases included; see issue #22). Claude effort is the concrete
  instance: client effort > fingerprint pin default > absent
  (issues #20 / #24).
- Decision: **Fail loud when unmappable.** If a backend cannot honor
  meaningful client intent (a value that changes the end result), return a
  clear error rather than silent drop. Aligned with open-edge reasoning effort
  and adapter map-or-fail (issues #20 / #24). Documented protocol-lossy gaps
  on translated paths stay listed in `docs/anthropic-compat.md`; those are
  explicit exceptions, not a license to strip other intent.
- Rationale: operators want one local endpoint for many clients and providers.
  A third dialect or silent field stripping forces client-side workarounds and
  hides bugs. Minimal compat defaults keep requests valid without freezing a
  stale "omni default profile." Fail-loud keeps errors actionable when a
  backend cannot honor intent.
- Non-goals for this decision: implementing behavior changes (philosophy
  applies now; remaining product work stays on #19, #22, and related
  issues; #18 / #26 are documented contracts below; #25 shipped in the
  shared unsupported-effort error path); new config knobs; replacing the
  compatibility matrix or roadmap with philosophy text.
- Source of truth: `docs/README.md` (operator summary), `docs/DESIGN.md`
  (principles + dual-mode Anthropic notes), this entry (citation anchor).

## Claude max_tokens and thinking budget (issue #19)

- Decision: **Passthrough.** Omni must not auto-bump client `max_tokens` when a
  Claude thinking budget would prefer a larger limit. Send the client
  `max_tokens` as given. If the client omitted `max_tokens`, only fingerprint
  wire defaults fill the capture value; thinking budget does not rewrite it.
- Rationale: client intent first (#27). Auto-bump hid undersized client caps and
  diverged from the values the caller set. Upstream may reject
  `max_tokens <= thinking.budget_tokens`; that stays a client/request concern,
  not a gateway mutation.
- Behavior change: both Claude doors (OpenAI→Anthropic translate and native
  `/v1/messages`) previously raised `max_tokens` to `budget+1024` (capped) when
  thinking was enabled and `max_tokens` was at or below the budget. That bump is
  removed in `finalize_claude_wire_request`.
- Non-goals: rejecting low `max_tokens` at the gateway; changing wire-default
  fill when `max_tokens` is omitted; changing thinking/effort mapping itself.
- Source of truth: `crates/provider-claude/src/translate.rs`
  (`finalize_claude_wire_request`), tests in that file and
  `anthropic_passthrough.rs`.

## Architecture

- Decision: one server binary (`omni`) with separate provider crates and a
  **capability-based thin edge**.
- Rationale:
  - Claude fingerprint logic is isolated in `provider-claude`.
  - Grok wire logic is isolated in `provider-grok`.
  - Codex config and Responses wire logic is isolated in `provider-codex`.
  - Shared HTTP conversion, Responses conversion, auth, stats, replacements,
    session derivation, conversation logging, and error envelopes live in
    `omni-common`.
  - `omni-core` owns canonical types, `LlmProvider` (canonical-only), optional
    `AnthropicNativeSurface`, and `BootstrappedProvider`.
  - `omni` only routes, frames responses, exposes catalogs, records stats, and
    wires optional conversation logging. Detect/init, model catalog export, and
    `provider_extras` allowlists live in provider crates behind bootstrap
    factories registered by the edge.

## Thin edge imports

- Edge **may** depend on provider crates for: registration (`bootstrap`,
  `detected`, `PROVIDER_ID`), and test-only constructors.
- Edge **must not** import: `FingerprintProfile` resolution, `anthropic_passthrough`,
  or concrete-only Claude fields on app state. Native Anthropic Messages /
  count_tokens go through `AnthropicNativeSurface` on the uniform entry.
- Fourth provider = bootstrap + `LlmProvider` (+ optional native surface) +
  registration only; no new wire/fingerprint code in `main.rs`.

## Routing

- Prefix routing selects the backend: `claude:<model>`, `grok:<model>`, or
  `codex:<model>`.
- With exactly one provider enabled, bare model names are accepted.
- With multiple providers enabled, bare model names are accepted only when the
  model id or alias uniquely matches one provider catalog.
- Anthropic inbound (`/v1/messages`) is **dual-mode**: same model resolver as
  chat. Claude → native passthrough; Grok/Codex → Anthropic↔canonical
  translation in `omni-common`. See `docs/anthropic-compat.md`.
- Anthropic `count_tokens` remains Claude-only; non-Claude models return 400.

## Provider Boundaries

- Claude: stale-cch handling, betas, preamble, profiles, model aliases,
  credentials, and Anthropic wire defaults stay in `provider-claude`. Native
  Anthropic inbound reconciliation, raw JSON passthrough, raw SSE forwarding,
  and count-token body shaping also stay there.
- Grok: xAI request/response mapping, streaming parsing, credential resolution,
  and model catalog stay in `provider-grok`.
- Codex: Codex config discovery, auth parsing, provider override handling, and
  OpenAI-compatible Responses wire mapping stay in `provider-codex`.
- Server concerns: auth, stats, bind/public flags, route registration, and model
  routing stay in `omni`.

## Prompt cache translation

- Decision: internal cache intent plus block/tool marks. Clients keep official
  inbound fields. Providers receive official outbound fields. See
  `docs/cache-translation.md`.
- Decision: compatibility wins for TTL and cache mode. Do not 400 the request
  when those cannot be copied exactly.
- Decision: TTL clamp-down. Pick the longest official backend value that is
  ≤ the requested duration. Do not round up. Do not reset to the backend
  default when a longer-but-still-≤ value exists. If nothing is ≤ the
  request, omit TTL and use the backend native/default cache. Grok has no
  TTL; drop it.
- Decision: inbound Chat accepts `x-grok-conv-id` and maps it to routing
  identity. Body `prompt_cache_key` wins if both are present. Outbound never
  emits the header; Grok Chat and Responses both send body `prompt_cache_key`.
- Decision: Claude slot cap. At most 4 Anthropic cache slots including
  automatic. Keep the last 4 marks; drop extras; do not 400.
- Decision: OpenAI `mode: explicit` with no breakpoints, routed to Claude:
  do not inject gateway auto-cache.
- Decision: invalid values on the inbound dialect itself are 400. Clamp-down
  is only for legal inbound values the chosen backend cannot copy.
- Source of truth: `docs/cache-translation.md` (shipped spec), provider cache
  docs (field names and enums).

## Current Surfaces

- OpenAI Chat Completions: `/v1/chat/completions`
- OpenAI Responses subset: `/v1/responses`
- Anthropic Messages, dual-mode (Claude native / Grok+Codex translated): `/v1/messages`
- Anthropic token count, Claude only (else 400): `/v1/messages/count_tokens`
- Models: `/v1/models`, `/models`
- Stats: `/stats`
- Health/root: `/health`, `/`

## Credentials

Credentials are read fresh per request.

- Claude: `$CLAUDE_CREDENTIALS_PATH` or `~/.claude/.credentials.json`
- Grok: `$XAI_CREDENTIALS_PATH`, a usable `~/.xai/.credentials.json`, or
  `~/.grok/auth.json`
- Codex: `CODEX_API_KEY`, `OPENAI_API_KEY`, `CODEX_ACCESS_TOKEN`, or
  `$CODEX_HOME` / `~/.codex` config and auth state.

Omni refreshes Claude/Codex/Grok OAuth primary-login tokens in-place by default
(atomic write-back of rotated refresh tokens). Disable with
`--no-oauth-refresh`, `OMNI_NO_OAUTH_REFRESH=1`, or `OMNI_OAUTH_REFRESH=0` (or
`false`/`off`/`no`) to keep CLI-delegated re-read only. Static API keys are
never refreshed.

Recovery shape (transparent to callers): on each credential load, re-read disk;
if the access token is expired or within **15 minutes** of expiry (or after an
upstream 401 force path), run up to **3** turns of refresh-under-lock then
full re-read. Refresh is serialized with an in-process single-flight mutex per
credential path plus a sibling `*.lock` flock (same naming as Grok's
`auth.json.lock`). Spent-RT / `invalid_grant` and CAS peer-rotation are treated
as success when disk already holds a fresh AT (peer or CLI won). After exhausted
turns with a still-stale AT, fail closed with a specific error (no soft-warn
continuing with a dead token). Write-back preserves file mode and requires a
rotated `refresh_token` in the grant.

On a default-path **upstream 401**, force-refresh once, then **replay the
inference request once** if refresh produced a live token (issue #31). Do not
retry 5xx or mid-stream model errors. Custom endpoints do not use CLI files
and do not take this 401 replay. See the issue #31 entry below.

Custom upstream endpoint configuration owns provider auth and must not fall
back to default credentials:

- Omni forced overrides are highest precedence:
  - Claude: `OMNI_CLAUDE_BASE_URL` uses only `OMNI_CLAUDE_AUTH_TOKEN`,
    `OMNI_CLAUDE_API_KEY`, and `OMNI_CLAUDE_CUSTOM_HEADERS`.
  - Grok: `OMNI_GROK_BASE_URL` uses only `OMNI_GROK_AUTH_TOKEN`,
    `OMNI_GROK_API_KEY`, and `OMNI_GROK_CUSTOM_HEADERS`.
  - Codex: `OMNI_CODEX_BASE_URL` is owned by `provider-codex` and uses only
    `OMNI_CODEX_AUTH_TOKEN`, `OMNI_CODEX_API_KEY`,
    `OMNI_CODEX_CUSTOM_HEADERS`, `OMNI_CODEX_MODEL`, and
    `OMNI_CODEX_WIRE_API`.
- Claude: `ANTHROPIC_BASE_URL` enables custom gateway mode using
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, resolved per request.
- Grok: `GROK_MODELS_BASE_URL` enables custom endpoint mode using
  `XAI_API_KEY` per request only; if it is absent, no Authorization header is
  sent.
- Codex: Codex custom provider config uses
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, or
  `env_key`; it uses OpenAI auth only when `requires_openai_auth = true`.

Codex OpenAI inbound supports non-streaming and `stream:true` paths. Streaming
uses native Responses SSE parsing in `provider-codex`, not buffered
pseudo-streaming.

## Auth recovery: proxy rotates keys, 401 replays once (issue #31)

- Decision: Omni **owns OAuth rotation** for the default CLI login files
  (Claude `~/.claude/.credentials.json`, Grok `~/.grok/auth.json`, Codex
  `~/.codex/auth.json`). The proxy runs 24/7; the vendor CLIs often do not.
  Key rotation is part of keeping that connection up. Static API keys are
  never refreshed.
- Decision: **same contract on all three default paths.** Read fresh every
  request. Refresh before send when the token is expired or inside the
  15-minute window. Fail closed if recovery cannot produce a live token
  (Grok must not warn-and-send a dead token).
- Decision: on default-path **upstream 401**, force-refresh once (the file
  clock can still look valid). If that works, **send the same inference
  request once more** (non-stream and stream-open only; no mid-stream replay).
  If refresh fails, return the 401. This is credential maintenance, not
  model retry. The client should not have to know the vendor login files.
- Decision: **do not** retry HTTP 5xx, stream drops, or upstream "Internal
  error during token generation." Those are model/upstream failures. Return
  them; the client retries if it wants.
- Decision: custom / override endpoints keep isolated auth and must not
  401-replay via the default CLI files.
- Rationale: Omni replicates dedicated vendor APIs and holds their login
  files. Refresh tokens rotate and revoke. Relying on the CLIs fails when
  they are off. A 401 after a successful rotation is our token, not a flaky
  model, so completing that one call keeps the connection stable. 5xx retry
  would hide upstream faults and stack with client retry (passthrough, #27).
- Non-goals: changing OAuth token URLs, lock/flock, or the 3-turn recovery
  loop; adding 5xx retry; custom-endpoint auth.
- Implementation: issue #31 is implemented on Claude, Grok, and Codex
  default CLI-file paths. Fail closed after recovery (Grok does not
  warn-and-send a dead token). 5xx / stream-drop / mid-stream model errors
  are client retries. Custom/override endpoints do not 401-replay via CLI
  files.
- Source of truth: this entry (citation). Operator summary: `docs/README.md`.
  Load/lock details: Credentials section above and
  `omni_common::oauth_refresh`.

## Observability

- Every request runs inside an `info_span!("request", request_id, session_id,
  provider)` opened by a middleware layer in `crates/bin/omni`. `request_id` is
  generated once in the layer and shared via a request extension so the span id,
  the response id, and the conversation-log `request` all derive from one value.
  `session_id` and `provider` are recorded late by the handlers.
- SSE streams outlive the handler, so the span is attached to them with a
  per-poll adapter (`SpannedStream`), NOT by holding a `Span::enter` guard across
  the stream's awaits. Holding a guard across `.await` leaves the span entered on
  the worker thread while the task is suspended, so a different concurrent request
  resuming on that thread would log under the wrong `request_id`. A concurrency
  test asserts no such cross-request bleed.
- `OMNI_LOG_COLOR` (`auto|always|never`, plus `NO_COLOR` and stderr TTY
  detection) gates colorized log fields (`crates/bin/omni/src/log_color.rs`):
  `request_id`/`session_id` get stable hashed hues, each provider a fixed color.
  The formatter sanitizes ANSI escapes in every value (matching upstream
  `tracing-subscriber`), so a provider echoing raw upstream bytes cannot inject
  terminal control sequences into the operator's log.

## Tests

- Default tests are hermetic and must not call live providers.
- Live provider tests require `OMNI_LIVE_TESTS=1` plus usable credentials.
- Subprocess HTTP tests use shared Rust helpers in `omni-common::test_support`
  instead of shelling out to `curl`.
- Residual live HTTP checks against a **running** omni process live in
  `tools/live_http_suite` (issue #15). Opt-in only
  (`python3 -m tools.live_http_suite`); not spawned by cargo, not run in CI.
  Separate from `OMNI_LIVE_TESTS`.

## Reasoning effort open edge and fail-loud adapters (issue #20)

- Decision: treat effort as a free string at the OpenAI chat and Responses
  edges (chat: top-level `reasoning_effort` or nested `reasoning.effort`;
  Responses: nested `reasoning.effort`). Validate **lexical hygiene only**
  (non-empty, max 32 chars, charset `A-Z a-z 0-9 _ -`). Drop any global
  valid-values allowlist so real levels such as `xhigh` reach
  `CanonicalReasoning`. Empty strings fail at the edge; JSON `null` is absent.
- Decision: model-catalog effort lists are **discovery-only**. They never gate
  or reject a request that passes lexical hygiene.
- Decision: Grok, Codex, and Claude **fail loud** on unmappable explicit effort
  via `ProviderError::unsupported_reasoning_effort` (structured BadRequest).
  Claude's `prepare_anthropic_request` returns `ProviderError` end to end
  (issue #30). Known aliases remain (`minimal`→`low` on Grok; `max`→`high` on
  both; Codex keeps first-class `xhigh`). Silent omit of client effort is a bug.
- Decision: Claude precedence is **client effort > fingerprint pin default >
  absent**. Client effort sets `output_config.effort` before pin wire defaults.
  Pin defaults apply only when effort is absent. Explicit `none` suppresses pin
  fill. Effort-capable models prefer `output_config` over effort-derived
  thinking budgets; models without that surface (e.g. Haiku) use thinking
  budgets and fail loud when unmappable.
- Rationale: clients and providers add new effort names faster than a gateway
  allowlist can track. Closing the edge name set either blocks real traffic or
  forces silent drops. Open edge + per-adapter map-or-fail preserves client
  intent and keeps errors actionable.
- Source of truth: `omni_common::validate_reasoning_effort_lexical`,
  `ProviderError::unsupported_reasoning_effort` (Grok/Codex/Claude), Claude
  mapper in `provider-claude` (issues #20/#25/#30). User summary: `docs/README.md`.
  Design: `docs/DESIGN.md`.

## Grok omit effort when client omits (issue #18)

- Decision: when a Grok client **omits** effort, Omni keeps omitting it on the
  upstream wire. No invented default, no force-floor, and no new disable that
  is not on the wire (this issue’s scope is documentation of that contract).
- Decision: the provider/model default then applies. On grok-4.5 that default
  is often `high`, and CLI capture notes that reasoning cannot be disabled
  there (`docs/providers/grok/CAPTURE.md`).
- Decision: explicit client effort still maps or fails loud per issue #20
  (`low|medium|high`, plus `xhigh` on `grok-4.6`; aliases; `"none"` omits).
  Clients that need a specific level must set it.
- Rationale: inventing a floor or a fake disable would fight the model default
  and invent wire that does not exist. Leave absent as absent; document the
  consequence.
- Non-goals: changing Grok mapping; product knobs for “minimum effort” or
  force-disable; inventing an upstream disable flag.
- Source of truth: `grok_reasoning_effort` / body builders in
  `crates/provider-grok` (omit when absent or `"none"`). User summary:
  `docs/README.md`. Design: `docs/DESIGN.md`. Capture: `docs/providers/grok/CAPTURE.md`.

## Claude free-string effort pass-through (issue #26)

- Decision: on the **OpenAI chat/Responses → Claude** path, for **effort-capable**
  models (pin exposes `output_config.effort` or the effort beta), free-string
  **non-`none`** client effort passes through to `output_config.effort`. Only
  the documented alias `minimal`→`low` is remapped; other strings (including
  `xhigh`, `max`, and future names) are not closed by a local Claude effort
  allowlist. Explicit `"none"` still suppresses pin fill and does not set
  `output_config.effort` (same precedence as issue #20).
- Decision: this pass-through is the Door-1 / OpenAI-inbound mapper contract.
  Native Anthropic `/v1/messages` keeps its closed request allowlist (client
  `output_config` remains unsupported there unless a rebaseline says otherwise).
- Decision: unknown or model-unsupported names may **400 from upstream by
  design**. Omni does not maintain a stale closed list of Anthropic effort
  tokens to reject earlier.
- Decision: fail loud locally only when the adapter **cannot express** the
  value (for example models with no effort surface where the thinking-budget
  ladder cannot map the value). Same structured
  `unsupported reasoning_effort` shape as issue #20 / #25.
- Rationale: matches open-edge + map-or-fail (#20). A local Claude allowlist
  goes stale as Anthropic adds levels; pass-through keeps client intent and
  lets the provider own validity.
- Non-goals: early reject lists of “known unsupported” Claude tokens; changing
  pin precedence (still client > pin default > absent); changing native
  Anthropic allowlist behavior.
- Source of truth: `claude_output_effort_value` /
  `apply_client_effort_to_output_config` in `crates/provider-claude/src/translate.rs`.
  User summary: `docs/README.md`. Design: `docs/DESIGN.md`.
