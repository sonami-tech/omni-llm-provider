# Anthropic inbound compatibility (dual-mode)

`POST /v1/messages` is dual-mode. Model routing uses the **same** resolver as
OpenAI chat (`resolve_provider_and_model`).

| Resolved provider | Behavior |
|---|---|
| **claude** | Native Anthropic passthrough (fingerprint, cch). No Anthropic→canonical. |
| **grok** / **codex** | Translated through canonical. Best-effort Anthropic shape. |
| other / missing | Anthropic-shaped 400 |

Full contracts: `docs/anthropic-frontend-multi-backend-plan.md`.

## count_tokens

| Provider | Behavior |
|---|---|
| claude | Native `/v1/messages/count_tokens` |
| grok / codex | **400** — token counting unsupported for this backend |

## Tool ID invariant

On the translated path, Anthropic `tool_use.id` is the **exact** backend
`tool_call_id` (no sanitize, no synthesize). Next-turn `tool_result.tool_use_id`
is passed through unchanged.

## `is_error` encoding (Grok/Codex wire)

Canonical `ToolResult.is_error` maps to OpenAI-style tool content:

| `is_error` | content | wire string |
|---|---|---|
| false | any | content as-is |
| true | empty | `"error"` |
| true | non-empty | `"ERROR: " + content` |

## No thinking on translated wire

Historical `thinking` / `redacted_thinking` blocks are dropped. Response
thinking is never emitted on Grok/Codex Anthropic SSE/JSON.

## Documented lossy fields (translated path)

| Item | Behavior |
|---|---|
| User text interleaved with `tool_result` in one user message | All tool results first, then trailing text/images (OAI adjacency) |
| `top_k` | Dropped |
| `cache_control` | Dropped |
| `stop_sequences` | Grok → extras `stop`; Codex dropped. Response always `stop_sequence: null` |
| Mid-conversation `role: "system"` | 400 |
| Trailing assistant prefill | 400 |
| `document` / hosted tools / computer use | 400 |
| Unknown top-level keys | Ignored |
| Adjacent same-role after thinking drop | 400 (fail loud) |

## Errors (translated path)

| Failure | HTTP | Anthropic error type |
|---|---|---|
| Gateway key missing/invalid | 401 | authentication_error |
| Client bad request / unsupported | 400 | invalid_request_error |
| Upstream 4xx | 400 | invalid_request_error |
| Upstream 429 | 429 | rate_limit_error |
| Upstream auth / missing provider credentials | **502** | api_error (never 401) |
| Upstream 5xx / transport / protocol | 502 | api_error |

After SSE `message_start`, failures are SSE `error` frames only (no happy
`message_stop`).

## Client tips

Point Anthropic clients at Omni with explicit `grok:…` or `codex:…` model ids
when multiple providers are enabled. Claude Code preamble/cch are not applied
on Grok/Codex.
