# Compatibility Gaps

This document tracks current functionality gaps that affect general client
compatibility. It is intentionally concise; provider-specific invariants remain
in `docs/providers/`.

Last reviewed: 2026-06-16.

Implementation phases and status are tracked in
`docs/compatibility-roadmap.md`.

## Priority Order

No high-priority compatibility gap from the June roadmap remains open.

Implemented support now covers:

1. Multimodal request support for image URL and base64 image inputs.
2. Broader Responses passthrough for Codex state, metadata, service tier, and
   structured-output controls.
3. Richer provider output preservation for usage details, annotations, provider
   metadata, service tier, system fingerprint, and non-stream Claude reasoning
   blocks.

## Implementation Readiness

1. Audio and file input support.
   - Needs explicit canonical media shape and provider mapping decisions.
   - Defer until concrete client need exists.

2. Non-function hosted tools.
   - Needs provider-specific mappings and compatibility tests.
   - Defer until provider semantics are explicit.

3. Live-provider compatibility smoke checks.
   - Requires explicit operator approval because they may spend quota.

## Easiest First

1. Add optional live-provider smoke checks.
2. Add hosted-tool mappings for one provider at a time.
3. Add audio/file input support once provider behavior is chosen.

## Resolved

1. Explicit provider extras behavior.
   - Resolved: unsupported provider extras fail loudly instead of being silently
     dropped.
   - Current behavior: OpenAI-compatible Chat and Responses top-level extension
     fields are preserved as canonical provider extras, except gateway metadata
     such as `user`. The selected provider validates extras against its
     allowlist before dispatch.
   - Provider allowlists are documented in `docs/providers/`.

2. Image multimodal request support.
   - Resolved: image URL and base64 image inputs are accepted for OpenAI Chat
     content arrays and Responses `input_image` parts, then mapped to Claude,
     Grok, and Codex.

3. Broader Responses API support.
   - Resolved: Codex forwards Responses state, metadata, service tier, and
     structured-output extras. Unsupported provider extras still fail loudly.

4. Rich provider output preservation.
   - Resolved: optional canonical extension fields preserve usage details,
     provider metadata, annotations, and non-stream reasoning blocks.

5. Compatibility matrix.
   - Resolved: `docs/compatibility-matrix.md` tracks current support status.

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
