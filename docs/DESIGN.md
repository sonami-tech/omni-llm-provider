# Design: Single Server Binary, Isolated Providers

## Gateway principles

Product framing for how the gateway treats client requests and provider wires
(issue #27). User-facing summary: `docs/README.md`. ADR-style anchor:
`docs/decisions.md`.

1. **Many-to-many.** One `omni` endpoint serves multiple providers and multiple
   inbound interfaces. Clients and providers connect without a third omni-only
   request dialect (canonical types are internal only).
2. **Pass through or translate.** Prefer pass-through when the upstream wire
   can carry the field. Otherwise translate into the provider's native form.
   Fields that change the end result must not be silently stripped; silent omit
   of meaningful client intent is a bug.
3. **Minimal surface / defaults.** Fill defaults only for compatibility when
   the client omits a value, and only in the smallest form that keeps the
   request valid. Do not invent maintenance-heavy defaults that go stale and
   break compatibility.
4. **Client intent first.** Precedence is **explicit client value → minimal
   compat/capture default when absent → absent**. Fingerprint or capture
   fidelity must not override meaningful client intent on this gateway
   (Door-2 honor-client cases included; see issue #22). Claude effort is the
   worked example: client effort > fingerprint pin default > absent
   (issues #20 / #24).
5. **Fail loud when unmappable.** When a backend cannot honor meaningful client
   intent (a value that changes the end result), fail with a clear structured
   error rather than silent drop (same contract as open-edge
   `reasoning_effort` and adapter map-or-fail). Documented protocol-lossy
   exceptions on translated paths remain listed in `docs/anthropic-compat.md`
   (for example `top_k`, `cache_control`); those are not silent intent bugs
   when explicitly documented, and they are not a license to strip other
   intent.

**Dual-mode Anthropic** stays consistent with these rules: Claude uses native
passthrough (fingerprint and wire defaults apply only where the client did not
supply the value). Grok/Codex use translated Anthropic→canonical→provider
paths; translation is best-effort protocol fidelity (`docs/anthropic-compat.md`)
but must not strip meaningful client intent in favor of capture or fingerprint
habits. Lossy protocol gaps are documented; silent intent drops are not.

## Decision

The workspace ships one server binary: `omni`.

Provider implementations remain separate crates:

- `provider-claude` owns the Claude Code fingerprint invariant, credentials,
  Anthropic Messages translation, streaming parser, and Claude model catalog.
- `provider-grok` owns the xAI wire mapping, credential resolution, streaming
  parser, and Grok model catalog.
- `provider-codex` owns Codex config discovery, custom-provider auth
  resolution, and Responses wire mapping.
- `omni-common` owns shared OpenAI-compatible HTTP conversion, Responses
  conversion, SSE framing, auth, stats, conversation logging, session
  derivation, replacements, error envelopes, and the OAuth refresh gate
  used by all three providers (issue #31).
- `omni-core` owns canonical types, the `LlmProvider` trait (canonical-only),
  optional `AnthropicNativeSurface`, and `BootstrappedProvider`.
- `crates/bin/omni` owns server startup, routing, auth wiring, stats wiring,
  optional conversation-log wiring, and model catalog aggregation. It is a
  **capability-based thin edge**: provider detect/init, extras allowlists, and
  version/model catalogs are produced by provider-crate bootstrap factories.
  The edge must not resolve fingerprint profiles or call Anthropic passthrough
  helpers; Claude native Messages/count_tokens dispatch through
  `entry.anthropic_native()`.

## Why One Binary

- Users run one local endpoint for Claude, Grok, and Codex.
- Auth, stats, HTTP routes, and model-list behavior have one implementation.
- Provider crates still protect provider invariants; no Claude stale-cch
  handling or fingerprint logic moves into `omni`.
- Model routing uses provider-owned catalogs. Bare canonical ids and documented
  aliases route when they uniquely match an enabled provider. `claude:`,
  `grok:`, and `codex:` prefixes remain as an explicit provider escape hatch.
- When no provider list is configured, startup enables all locally detected
  providers. `--providers` / `OMNI_PROVIDERS` remains an explicit override.

## HTTP Surface

- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/messages` dual-mode Anthropic Messages inbound (Claude native, or
  Grok/Codex via canonical translation)
- `POST /v1/messages/count_tokens` Claude-native only (non-Claude → 400)
- `GET /v1/models`, `GET /models`
- `GET /stats` (plain text by default with `Refresh: 5` for browsers; `?format=json` or `/stats/json` for JSON)
- `GET /health`, `GET /`

OpenAI-compatible inbound surfaces route through `LlmProvider` and can target
any enabled provider.

Anthropic inbound is **dual-mode** (same shared model resolver as chat):

| Resolved provider | Path |
|---|---|
| **claude** | Native passthrough: fingerprint, billing header, raw JSON/SSE. Original body is not run through Anthropic→canonical. |
| **grok** / **codex** | Translated: Anthropic → Canonical → `LlmProvider` → Anthropic JSON/SSE (`omni-common::anthropic`). Best-effort protocol fidelity; lossy fields documented in `docs/anthropic-compat.md`. |

Claude native path stays in `provider-claude`. Mappers live in `omni-common`;
dispatch in `bin/omni`. Providers remain canonical-only on the trait.

Codex supports OpenAI inbound non-streaming and streaming paths by posting to
the Codex Responses API and translating native Responses SSE events into
canonical stream events.

`LlmProvider` remains canonical-only. Optional native Anthropic work uses a
separate object-safe capability (`AnthropicNativeSurface`) exposed on the
uniform provider entry (Claude only today). Grok/Codex do not implement it;
their Anthropic inbound stays on the translated path.

## Build

```bash
cargo build -p omni
cargo run -p omni -- --version
cargo run -p omni -- --port 18321
```

## Non-Goals

- Do not merge provider internals into `omni`.
- Do not route unknown or ambiguous bare model names heuristically when more
  than one provider is enabled.
- Do not claim perfect Anthropic wire fidelity on Grok/Codex translated path
  (see `docs/anthropic-compat.md`).
- Do not emit thinking blocks on the translated Anthropic path (v1).
- Do not add a separate `openai` provider id; OpenAI-compat backends use Codex
  (and existing custom endpoints).
- Do not add provider-specific server binaries unless there is a concrete
  compatibility requirement.

## Reasoning effort (issue #20)

OpenAI chat and Responses lift free-string effort into
`CanonicalReasoning.effort`. Wire shapes: chat top-level `reasoning_effort`
(or nested `reasoning.effort`; top-level wins when both are present);
Responses nested `reasoning.effort`. Edge validation is **lexical only**
(non-empty, length ≤ 32, safe charset `A-Z a-z 0-9 _ -`). Empty strings fail at
the edge; JSON `null` means absent. Explicit `"none"` is preserved in canonical
so adapters can disable (Codex) or omit (Claude/Grok).

**Catalog never gates.** Model-catalog effort advertisements are discovery
hints for UIs. They do not close the request name set and must not reject
values that lexical hygiene accepts.

**Adapters map or fail loud.** Providers must not silently drop an explicit
client effort. Unmappable values return HTTP 400 BadRequest with a stable
`unsupported reasoning_effort` message shape (provider, path, requested value,
optional model, optional supported list). Grok and Codex use
`ProviderError::unsupported_reasoning_effort` (Claude via
`prepare_anthropic_request`, same structured BadRequest).

- **Grok:** wire `low|medium|high`, plus `xhigh` on `grok-4.6`. Aliases:
  `minimal`→`low`, `max`→`high`. Explicit `"none"` omits the field. `xhigh` on
  `grok-4.5` and other unknowns fail loud.
  When the client **omits** effort entirely, Omni also omits the upstream field
  (issue #18). No invented default, force-floor, or disable. The
  provider/model default applies (often `high` on grok-4.5; capture notes:
  `docs/providers/grok/CAPTURE.md`).
- **Codex:** wire `none|minimal|low|medium|high|xhigh`. Alias: `max`→`high`.
  Explicit `"none"` still emits so Codex can disable reasoning. Unknowns
  (e.g. `ultra`) fail loud.
- **Claude:** precedence is **client effort > pin default > absent**. Client
  non-none effort sets `output_config.effort` before fingerprint wire defaults
  (so pin cannot overwrite). Pin defaults apply only when client effort is
  absent. Explicit `"none"` suppresses pin fill. Models with
  an effort surface prefer `output_config` over effort-derived thinking
  budgets; models without it (e.g. Haiku) use the thinking-budget ladder and
  fail loud when the ladder cannot express the value.
  On the OpenAI chat/Responses path, effort-capable models pass free-string
  **non-`none`** effort through to `output_config.effort` (issue #26); only
  `minimal`→`low` is remapped (`max` and others stay as-is). There is no closed
  local Claude effort allowlist. Upstream may 400 unknown names by design.
  Fail loud only when the thinking-budget ladder (or effort surface) cannot
  express the value.

Shared edge helper: `omni_common::validate_reasoning_effort_lexical`. Shared
constructor: `ProviderError::unsupported_reasoning_effort` (Grok/Codex/Claude).
User-facing summary: `docs/README.md`. Decision record: `docs/decisions.md`.

## Codex/OpenAI Backend

The Codex backend is implemented in `provider-codex` and implements
`LlmProvider` for OpenAI-compatible inbound surfaces. It reads Codex
configuration from `$CODEX_HOME` or `~/.codex`, including provider overrides
such as `base_url`, `wire_api`, `env_key`, `http_headers`,
`env_http_headers`, query parameters, and command-backed auth. Secret auth
material is resolved per request and is not logged. Anthropic inbound may also
target Codex via the dual-mode translated path (best-effort).

## Custom Upstream Auth

When a provider is pointed at a custom upstream endpoint, that custom
configuration owns auth for that provider and default credentials must not leak
(no CLI-file 401 replay on these paths; issue #31):

- Claude forced override: `OMNI_CLAUDE_BASE_URL` wins over
  `ANTHROPIC_BASE_URL` and uses only `OMNI_CLAUDE_AUTH_TOKEN`,
  `OMNI_CLAUDE_API_KEY`, and `OMNI_CLAUDE_CUSTOM_HEADERS`.
- Claude: `ANTHROPIC_BASE_URL` activates custom gateway mode. Omni uses
  `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_CUSTOM_HEADERS`
  only, reads them per request, and does not read local Claude OAuth credentials
  for that gateway.
- Grok forced override: `OMNI_GROK_BASE_URL` wins over `GROK_MODELS_BASE_URL`
  and uses only `OMNI_GROK_AUTH_TOKEN`, `OMNI_GROK_API_KEY`, and
  `OMNI_GROK_CUSTOM_HEADERS`.
- Grok: `GROK_MODELS_BASE_URL` activates custom endpoint mode. Omni uses
  `XAI_API_KEY` per request if present, otherwise no Authorization header, and
  does not read the default xAI/Grok credential files.
- Codex forced override: `OMNI_CODEX_BASE_URL` is resolved inside
  `provider-codex`; it feeds detection, catalog, aliases, and request config,
  and uses only `OMNI_CODEX_AUTH_TOKEN`, `OMNI_CODEX_API_KEY`,
  `OMNI_CODEX_CUSTOM_HEADERS`, `OMNI_CODEX_MODEL`, and `OMNI_CODEX_WIRE_API`.
- Codex: Codex config controls custom-provider auth.
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, and
  `env_key` do not fall back to OpenAI auth unless
  `requires_openai_auth = true`.

## Runtime State

Stats default to `omni-stats.redb` in the process current working directory.
Multi-instance runs should set `--stats-db` / `OMNI_STATS_DB` to a distinct
path per instance.

Conversation logging is disabled by default. It can write to stderr, a rotating
single file, or per-session files via `--log-conversations`, `--log-file`, or
`--log-dir`. Session ids prefer `x-session-id`, then request `user`, then API-key
id, then an anonymous fallback.
