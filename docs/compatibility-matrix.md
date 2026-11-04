# Compatibility Matrix

Normal tests are hermetic and quota-free. Live provider checks remain opt-in via
`OMNI_LIVE_TESTS=1`.

## Request Inputs

| Feature | Chat Completions | Responses | Claude | Grok | Codex |
|---|---:|---:|---:|---:|---:|
| Text messages | Yes | Yes | Yes | Yes | Yes |
| Official cache fields | Yes (`prompt_cache_key`, `x-grok-conv-id`, options, breakpoints) | Yes (`prompt_cache_key`, options, breakpoints) | Consume routing identity; emit `cache_control` | Yes (body `prompt_cache_key`; never `x-grok-conv-id`) | Yes (REST and ChatGPT WS share one cache payload) |
| Function tools | Yes | Yes | Yes | Yes | Yes |
| Tool choice `auto` / `none` | Yes | Yes | Yes | Yes | Yes |
| Forced tool choice (`required` / named) | Yes | Yes | Sent as `auto` ([Tool choice](#tool-choice)) | Yes | Yes |
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
| `tool_choice` `auto` / `none` | Yes | Yes | Yes |
| `tool_choice` `any` / `tool` | Sent as `auto` ([Tool choice](#tool-choice)) | Yes | Yes |
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
Claude keeps `auto`, sends required and named choices as `auto` with the
same flag (see Tool choice), and uses `auto` when tools exist but choice is
omitted and a flag needs a carrier. Claude `none` stays `none`;
with no tools no choice is sent, and forced choices still fail validation.
Grok and Codex preserve explicit booleans, including no-tools requests, and
omit absent controls in streaming and non-streaming builders. These are tested
serialization contracts, not proof of real CLI/model acceptance. A live
`auto` request with `parallel_tool_calls: false` was accepted by every model
in the tool choice check below; that shows acceptance, not enforcement.

Gateway metadata such as `user` is not provider passthrough.

## Tool choice

The tool choice protocol is unchanged: OpenAI has `auto`, `required`, `none`,
and a named function; Anthropic has `auto`, `any`, `tool`, and `none`. Omni
maps `required` to `any` and a named function to `tool`.

Current Claude models reject the two forced choices, `any` and `tool`, with an
upstream 400: `tool_choice: type "tool" and "any" are not supported for this
model.` The restriction also applies to token counting. Anthropic's list of
affected models is the source of truth:
<https://platform.claude.com/docs/en/api/errors#forced-tool-use-not-supported>.

The forward-facing baseline for every provider is `auto` or `none`. On every
Claude request, Omni sends a forced choice (`any`, or `tool` with a name) as
`{"type":"auto"}`. This applies to Chat Completions, Responses, and native
`/v1/messages`, streaming or not, and to `/v1/messages/count_tokens`. The rule
is fixed: it does not depend on the model ID or the endpoint.

- `disable_parallel_tool_use` keeps its value on the `auto` choice, or stays
  absent when absent.
- Tools, schemas, system, messages, cache markers, and all other fields stay
  unchanged. Omni adds no text.
- `auto`, `none`, and an omitted choice stay unchanged. A forced choice with
  no tools keeps its existing error.
- The response carries `x-omni-dropped: tool_choice`, merged with any other
  `x-omni-dropped` value. This includes upstream and transport errors after
  Omni tried to send the request, and streaming responses (the header comes
  before the body). Requests that fail before Omni sends them (validation,
  preparation, credentials, request build) do not get it. Omni logs one warning
  with route, model, and choice kind. The log has no tool names or prompts.
- Omni adds no compatibility retry or probe and does not remember model
  support.

A required tool call becomes optional on Claude: the model may answer without
a tool or pick another tool. Changing `tool_choice` can still affect message
caching under Anthropic's normal cache rules.

Grok and Codex forward forced choices unchanged; the current models accept
them.

Live check through Omni on 2026-11-03, before the Claude change, non-streaming,
one request per cell:

| Model | `auto` | `none` | `required` | Named tool | `auto` + parallel off |
|---|---:|---:|---:|---:|---:|
| `claude-opus-5-5` | 200 | 200 | 400 | 400 | 200 |
| `claude-sonnet-5-5` | 200 | 200 | 400 | 400 | 200 |
| `claude-fable-5-1` | 200 | 200 | 400 | 400 | 200 |
| `gpt-6-astra` | 200 | 200 | 200 | 200 | 200 |
| `gpt-6.1-sol` | 200 | 200 | 200 | 200 | 200 |
| `gpt-6-luna` | 200 | 200 | 200 | 200 | 200 |
| `grok-4.7` | 200 | 200 | 200 | 200 | 200 |

`claude-opus-5-5` also returned 400 for a named tool on `/v1/responses` and
native `/v1/messages`, so the rejection does not come from Omni's translation.
Streaming and tool-result turns were not part of this check.

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
