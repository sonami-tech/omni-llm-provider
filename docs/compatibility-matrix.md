# Compatibility Matrix

Last updated: 2026-07-12.

Normal tests are hermetic and quota-free. Live provider checks remain opt-in via
`OMNI_LIVE_TESTS=1`.

## Request Inputs

| Feature | Chat Completions | Responses | Claude | Grok | Codex |
|---|---:|---:|---:|---:|---:|
| Text messages | Yes | Yes | Yes | Yes | Yes |
| Function tools | Yes | Yes | Yes | Yes | Yes |
| Tool result loops | Yes | Yes | Yes | Yes | Yes |
| Image URL input | Yes | Yes | Yes | Yes | Yes |
| Base64 image input | Yes | Yes | Yes | Yes | Yes |
| Audio input | No | No | No | No | No |
| File input | No | No | No | No | No |

Unsupported typed media parts fail loudly with a request error.

## Anthropic inbound (`POST /v1/messages`)

| Feature | Claude (native) | Grok (translated) | Codex (translated) |
|---|---:|---:|---:|
| Text + multi-block | Yes | Yes | Yes |
| Function tools + tool loops | Yes | Yes | Yes |
| Images (url/base64) | Yes | Yes | Yes |
| Streaming SSE | Yes (raw) | Yes (framed) | Yes (framed) |
| Thinking wire emit | Yes | No | No |
| Hosted/computer tools | Passthrough | No (400) | No (400) |
| `count_tokens` | Yes | No (400) | No (400) |
| Fingerprint / cch | Yes | No | No |

Details and lossy fields: `docs/anthropic-compat.md`.

## Responses Fields

| Feature | Claude | Grok | Codex |
|---|---:|---:|---:|
| `previous_response_id` | No | No | Yes |
| `metadata` passthrough | No | No | Yes |
| `service_tier` passthrough | No | Yes | Yes |
| `response_format` passthrough | No | Yes | Yes |
| `text.format` passthrough | No | No | Yes |
| `parallel_tool_calls` passthrough | No | Yes | Yes |

Gateway metadata such as `user` is not provider passthrough.

## Rich Outputs

| Field | Chat output | Responses output | Source providers |
|---|---:|---:|---|
| Native response id | Synthetic chat id only | Yes | Claude, Grok, Codex |
| `system_fingerprint` | Yes, when present | Yes, when present | Grok, Codex |
| `service_tier` | Yes, when present | Yes, when present | Grok, Codex |
| Usage cache details | Yes, when present | Yes, when present | Claude, Grok, Codex |
| Reasoning token counts | Yes, when present | Yes, when present | Grok, Codex |
| Annotations/citations | Provider metadata | Output annotations | Codex |
| Claude thinking blocks | Non-stream canonical only | Non-stream canonical only | Claude |

Claude streaming thinking deltas are preserved as canonical stream events for
internal consumers. Public Chat/Responses SSE does not currently synthesize
provider-specific reasoning events from those deltas.

## Source Of Truth

- Core request and response contract: `crates/omni-core/src/canonical.rs`
- Chat conversion and framing: `crates/omni-common/src/http.rs`
- Responses conversion and framing: `crates/omni-common/src/responses.rs`
- Anthropic dual-mode mappers/framer: `crates/omni-common/src/anthropic.rs`
- Anthropic translated-path notes: `docs/anthropic-compat.md`
- Provider allowlists: `docs/providers/README.md`
