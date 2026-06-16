# Compatibility Gaps

This document tracks current functionality gaps that affect general client
compatibility. It is intentionally concise; provider-specific invariants remain
in `docs/providers/`.

Last reviewed: 2026-06-15.

Implementation phases and status are tracked in
`docs/compatibility-roadmap.md`.

## Priority Order

1. Multimodal request support.
   - Impact: high. Many modern OpenAI and Responses clients send content arrays,
     images, files, audio, or typed input parts.
   - Current state: OpenAI chat input is text-only, and Responses rejects
     non-text parts such as `input_image`.
   - Likely scope: add canonical media blocks, then map OpenAI Chat,
     Responses, Claude, Grok, and Codex request shapes.
   - Effort: high.

2. Broader Responses API support.
   - Impact: high. Newer OpenAI-compatible clients increasingly use
     `/v1/responses`.
   - Current state: Omni supports text input, function tools, reasoning effort,
     basic tool loops, non-streaming responses, and Responses SSE. Other item
     and tool types are rejected.
   - Likely scope: expand supported input/output item types, structured output,
     metadata, stateful continuation, and provider-specific equivalents.
   - Effort: medium to high.

3. Richer provider output preservation.
   - Impact: medium. Search, agent, and observability clients benefit from
     citations, annotations, reasoning token counts, service tier, response ids,
     and provider metadata.
   - Current state: canonical responses preserve text, refusal, function calls,
     finish reason, basic usage, cache usage, and response id. Several provider
     details are parsed or available but not surfaced.
   - Likely scope: add additive canonical metadata fields and map provider
     outputs into OpenAI Chat and Responses envelopes.
   - Effort: medium.

## Implementation Readiness

1. Multimodal request support.
   - Needs a canonical content decision before implementation.
   - Recommended first scope: image URL and base64 blocks, ordered with text and
     tool blocks. Defer audio and files until the image path is proven.

2. Broader Responses API support.
   - Needs a bounded v1 scope before implementation.
   - Recommended first scope: `previous_response_id`, metadata/service tier,
     structured output, and common output items. Defer non-function tools until
     provider mappings are explicit.

3. Richer provider output preservation.
   - Needs a response schema decision before implementation.
   - Recommended shape: additive usage-details and provider-namespaced
     extensions for annotations, citations, reasoning tokens, service tier,
     system fingerprint, and search/source metadata.

## Easiest First

1. Preserve richer provider outputs.
2. Broaden the Responses subset.
3. Add multimodal support.

## Resolved

1. Explicit provider extras behavior.
   - Resolved: unsupported provider extras fail loudly instead of being silently
     dropped.
   - Current behavior: OpenAI-compatible Chat and Responses top-level extension
     fields are preserved as canonical provider extras, except gateway metadata
     such as `user`. The selected provider validates extras against its
     allowlist before dispatch.
   - Provider allowlists are documented in `docs/providers/`.

## Notes

Claude native `/v1/messages` intentionally has a closed allowlist to preserve
the Claude fingerprint path. Do not treat dropped Anthropic fields as the first
general-compatibility priority unless there is a concrete Claude-native client
failure to support.

Official API references checked for this review:

- OpenAI Responses API:
  <https://developers.openai.com/api/reference/resources/responses/methods/create/>
- OpenAI Chat Completions API:
  <https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/>
- Anthropic Messages API: <https://docs.anthropic.com/en/api/messages>
- xAI Chat and Responses API:
  <https://docs.x.ai/developers/rest-api-reference/inference/chat>

Two sealed Grok second opinions on 2026-06-15 agreed on strategic priority:
multimodal first, broader Responses second, richer outputs third, provider
extras fourth. Provider extras was completed first because it was the easiest
low-risk compatibility improvement.
