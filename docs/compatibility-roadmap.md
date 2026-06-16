# Compatibility Roadmap

This document tracks go-forward compatibility work across sessions. It covers
the active compatibility plan only.

Last updated: 2026-06-16.

## Current Status

| Phase | Status | Owner Notes |
|---|---|---|
| 0. Baseline | Done | Compatibility gaps are documented in `docs/compatibility-gaps.md`; this roadmap now tracks the active go-forward phases. |
| 1. Extras Contract | Done | Unsupported provider extras now fail loudly, supported extras forward by provider allowlist, and docs list allowlists. |
| 2. Multimodal Request Core | Done | Image URL and base64 image inputs are canonicalized and mapped to Claude, Grok, and Codex. |
| 3. Responses API V1 Expansion | Done | Codex forwards state, metadata, service tier, response format, and text format; unsupported provider extras still fail loudly. |
| 4. Rich Output Preservation | Done | Usage details, provider metadata, annotations, response metadata, and non-stream Claude reasoning blocks are additive canonical fields. |
| 5. Compatibility Matrix | Done | `docs/compatibility-matrix.md` tracks supported request, Responses, and rich-output behavior. |

## Phase 1: Extras Contract

Goal: make provider-specific extras behavior explicit and testable.

Scope:

- Define what happens when unsupported extra request fields are received.
- Prefer clear rejection or explicit diagnostics over silent drops.
- Document provider allowlists and intentionally unsupported fields.
- Keep Claude native passthrough conservative to preserve its fingerprint path.

Likely files:

- `crates/omni-common/src/http.rs`
- `crates/provider-grok/src/lib.rs`
- `crates/provider-codex/src/lib.rs`
- `crates/provider-claude/src/translate.rs`
- `docs/compatibility-gaps.md`
- `docs/providers/README.md`

Done when:

- Unsupported extras have predictable behavior.
- Grok and Codex allowlists are documented.
- Claude-native exclusions remain intentional and documented.
- Focused tests cover accepted and unsupported extras.
- Standard formatting, linting, and relevant tests pass.

Open decisions:

- Settled: unsupported provider extras fail with a request error for the
  selected provider. Omni does not add warning envelopes.
- Settled: `user` remains gateway/session metadata and is not provider
  passthrough.

Result: OpenAI-compatible Chat and Responses extension fields are preserved into
canonical provider extras, then validated against the selected provider's
allowlist before dispatch. Claude has no OpenAI-compatible provider extras
passthrough today. Claude native passthrough remains conservative to preserve
the Claude Code fingerprint path.

## Phase 2: Multimodal Request Core

Goal: accept common multimodal OpenAI-compatible request shapes without losing
ordered content semantics.

Scope:

- Add canonical content blocks for images.
- Support OpenAI Chat content arrays for text and images.
- Support Responses `input_image` alongside `input_text`.
- Map supported image inputs to Claude, Grok, and Codex where provider APIs
  support them.
- Fail loudly for unsupported media types.

Done when:

- Existing text-only clients remain backward compatible.
- Image URL and base64 inputs work through supported providers.
- Unsupported audio, file, and unknown media parts produce clear errors.
- Tests cover request parsing, canonical conversion, provider mapping, and
  unsupported cases.

Result: `CanonicalBlock::Image` carries either a URL or base64 media type/data.
OpenAI Chat content arrays and Responses `input_image` parts preserve ordering
with adjacent text. Claude receives native image blocks, Grok receives
OpenAI-compatible `image_url` content parts, and Codex receives Responses
`input_image` parts. Audio and files remain unsupported and fail loudly.

## Phase 3: Responses API V1 Expansion

Goal: make `/v1/responses` usable by more modern OpenAI-compatible clients.

Scope:

- Support `previous_response_id`.
- Support metadata and service tier where providers can forward them.
- Support structured output controls.
- Expand common output item handling without committing to every provider tool.

Done when:

- Common Responses clients can use stateful continuation and structured output
  request fields.
- Unsupported Responses tools and item types fail clearly.
- Responses streaming and non-streaming tests cover new fields.

Result: Codex forwards `previous_response_id`, `metadata`, `service_tier`,
`response_format`, `text`, and `parallel_tool_calls`. Grok continues to forward
its chat-compatible extras such as `service_tier`, `response_format`, and
`parallel_tool_calls`; Responses-native state fields remain unsupported for
Grok. Claude OpenAI-compatible provider extras remain unsupported.

Deferred: non-function hosted tools and additional provider-specific output item
types.

## Phase 4: Rich Output Preservation

Goal: preserve useful provider metadata without polluting the core text/tool
contract.

Scope:

- Add additive usage details where providers expose them.
- Preserve annotations, citations, reasoning-token counts, service tier, system
  fingerprint, search metadata, and response ids.
- Expose provider-specific details through a namespaced extension shape.

Done when:

- Existing response JSON remains backward compatible.
- Rich fields are available to clients that opt into reading them.
- Tests cover provider fields that were previously parsed or available but lost.

Result: text, refusal, tool calls, finish reason, and basic usage remain stable.
Optional canonical fields now preserve usage details, response metadata,
annotations, and non-stream reasoning blocks. Chat and Responses envelopes emit
extension fields only when present. Claude streaming thinking deltas are
canonical events, but public Chat/Responses SSE does not synthesize
provider-specific reasoning events yet.

## Phase 5: Compatibility Matrix

Goal: make support status easy to verify and maintain.

Scope:

- Add or update fixtures for key OpenAI Chat, OpenAI Responses, Anthropic, xAI,
  and Codex shapes.
- Document provider behavior by inbound API surface.
- Identify optional live-provider checks, but keep them opt-in.

Done when:

- Docs state what works per provider and API shape.
- Compatibility fixtures cover the main supported and unsupported cases.
- Normal test commands remain quota-free.

Result: `docs/compatibility-matrix.md` states supported behavior by API surface
and provider. Focused hermetic tests cover image inputs, Responses extras,
rich-output details, and unsupported media errors.

## Recommended Next Phase

No remaining compatibility phase from this roadmap is open.

Recommended next work: run live-provider smoke checks manually with
`OMNI_LIVE_TESTS=1` when quota and credentials are explicitly approved.

## Source Documents

- Gap analysis: `docs/compatibility-gaps.md`
- Compatibility matrix: `docs/compatibility-matrix.md`
- Project overview: `docs/README.md`
- Provider notes: `docs/providers/README.md`
