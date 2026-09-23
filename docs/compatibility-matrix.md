# Compatibility Matrix

Last updated: 2026-09-04.

Normal tests are hermetic and quota-free. Live provider checks remain opt-in via
`OMNI_LIVE_TESTS=1`.

## Request Inputs

| Feature | Chat Completions | Responses | Claude | Grok | Codex |
|---|---:|---:|---:|---:|---:|
| Text messages | Yes | Yes | Yes | Yes | Yes |
| Official cache fields | Yes (`prompt_cache_key`, `x-grok-conv-id`, options, breakpoints) | Yes (`prompt_cache_key`, options, breakpoints) | Consume routing identity; emit `cache_control` | Yes (body `prompt_cache_key`; never `x-grok-conv-id`) | Yes (REST and ChatGPT WS share one cache payload) |
| Function tools | Yes | Yes | Yes | Yes | Yes |
| OpenAI `allowed_tools` function subset | Yes (filtered) | Yes (filtered) | Translated | Translated | Translated |
| Tool result loops | Yes | Yes | Yes | Yes | Yes |
| Image URL input | Yes | Yes | Yes | Yes | Yes |
| Base64 image input | Yes | Yes | Yes | Yes | Yes |
| Audio input | No | No | No | No | No |
| File input | PDF `file_id` / `file_data` | `input_file` URL / ID / data | PDF URL / data; no OpenAI ID or detail | Responses URL / ID only; Chat rejects files | Responses URL / ID / data |

Unsupported typed media parts and file variants that a selected backend cannot honor fail loudly with a request error. File IDs belong to the receiving provider; Omni does not upload or transfer files between providers. Grok file support uses its Responses route on agentic-capable models.

## Anthropic inbound (`POST /v1/messages`)

| Feature | Claude (native) | Grok (translated) | Codex (translated) |
|---|---:|---:|---:|
| Text + multi-block | Yes | Yes | Yes |
| Function tools + tool loops | Yes | Yes | Yes |
| `tool_choice` (`auto` / `any` / `tool` / `none`) | Yes | Yes | Yes |
| Images (url/base64) | Yes | Yes | Yes |
| Streaming SSE | Yes (raw) | Yes (framed) | Yes (framed) |
| Thinking wire emit | Yes | No | No |
| Hosted/computer tools | Passthrough | No (400) | No (400) |
| `count_tokens` | Yes | No (400) | No (400) |
| Claude Code fingerprint | Yes | No | No |
| Official `cache_control` | Native passthrough | Translated | Translated |
| Body `prompt_cache_key` | 400 | 400 | 400 |

Details and lossy fields: `docs/anthropic-compat.md`.
Shipped cache translation: `docs/cache-translation.md`.

## Responses Fields

| Feature | Claude | Grok | Codex |
|---|---:|---:|---:|
| `store` passthrough | No | No | Yes |
| `previous_response_id` | No | No | Yes |
| `metadata` passthrough | No | No | Yes |
| `service_tier` passthrough | No | Yes | Yes |
| `response_format` passthrough | No | Yes | Translated |
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
