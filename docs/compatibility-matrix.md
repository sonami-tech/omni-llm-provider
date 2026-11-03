# Compatibility Matrix

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
| Images in tool results | Yes (`image_url` parts on `role:"tool"`) | Yes (`function_call_output.output` parts) | Yes (native blocks) | Yes (Responses parts; Chat: see below) | Yes (Responses parts) |
| Audio input | No | No | No | No | No |
| File input | PDF `file_id` / `file_data` | `input_file` URL / ID / data | PDF URL / data; no OpenAI ID or detail | Responses URL / ID only; Chat rejects files | Responses URL / ID / data |

Unsupported typed media parts and file variants that a selected backend cannot honor fail loudly with a request error. File IDs belong to the receiving provider; Omni does not upload or transfer files between providers. Grok file support uses its Responses route on agentic-capable models.

Tool results hold text and images only; files in tool results fail with a
request error. A text-only tool result reaches each backend as the same string
as before. The Chat spec allows only text in tool messages, so the Grok Chat
path (custom OpenAI-compatible endpoints) puts a placeholder such as
`[image 1 from tool call <id>: see next user message]` in the tool message, and
adds one user message with the labeled images after the run of tool messages.
That path has offline tests only.

Chat, Responses, and Anthropic routes share one request body limit,
`MAX_REQUEST_BODY_BYTES` in `crates/bin/omni/src/main.rs`.

Codex caller output caps (`max_tokens` / `max_completion_tokens` on Chat,
`max_output_tokens` on Responses, `max_tokens` on `/v1/messages`): the REST
path (API key or custom gateway) forwards the cap as `max_output_tokens`.
The ChatGPT WebSocket path does not enforce the cap and ignores `max_tokens`.
That path logs one warning (model, route, requested value) and returns
`x-omni-dropped: max_output_tokens` on streaming and non-streaming responses,
including when the upstream request fails after the drop.
The proxy does not truncate output. The split follows the transport actually
used, not the model name. `/v1/messages` still requires `max_tokens`.

## Anthropic inbound (`POST /v1/messages`)

| Feature | Claude (native) | Grok (translated) | Codex (translated) |
|---|---:|---:|---:|
| Text + multi-block | Yes | Yes | Yes |
| Function tools + tool loops | Yes | Yes | Yes |
| `tool_choice` (`auto` / `any` / `tool` / `none`) | Yes | Yes | Yes |
| Images (url/base64) | Yes | Yes | Yes |
| Images in `tool_result` | Yes | Yes | Yes |
| Streaming SSE | Yes (raw) | Yes (framed) | Yes (framed) |
| Thinking wire emit | Yes | No | No |
| Hosted/computer tools | Passthrough | No (400) | No (400) |
| `count_tokens` | Yes | No (400) | No (400) |
| Claude Code fingerprint | Yes | No | No |
| Official `cache_control` | Native passthrough | Translated | Translated |
| Body `prompt_cache_key` | 400 | 400 | 400 |
| `max_tokens` | Client value when set | Required on the translated path | Required. REST forwards it. ChatGPT WebSocket ignores it and does not enforce it |

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
| `parallel_tool_calls` canonical control | Inverse nested disabling flag | Yes (Chat and CLI Responses) | Yes (REST and WebSocket) |

Chat and Responses accept boolean `parallel_tool_calls`; missing or null means
absent, and other types are a 400. It is canonical intent, not a provider extra.
Claude preserves auto/required/named choices and uses `auto` when tools exist
but choice is omitted and a flag needs a carrier. Claude `none` stays `none`;
with no tools no choice is sent, and forced choices still fail validation.
Grok and Codex preserve explicit booleans, including no-tools requests, and
omit absent controls in streaming and non-streaming builders. These are tested
serialization contracts, not proof of real CLI/model acceptance. Live explicit
true/false acceptance and the incident target Claude model's forced-choice and
final thinking compatibility remain unverified.

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
