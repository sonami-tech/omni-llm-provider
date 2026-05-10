# claude CLI baseline wire fingerprint (captured 2026-05-10)

Captured via mitmproxy 11.x against `claude --print --model claude-haiku-4-5 "..."` (SDK CLI mode, claude version 2.1.138, SDK package 0.93.0).

Source flow: `tools/fingerprint/scenarios/01-plain-text.flow`. Replay with:

```sh
uv tool run --from mitmproxy python3 tools/fingerprint/extract_flow.py \
  tools/fingerprint/scenarios/01-plain-text.flow
```

## Endpoint

`POST https://api.anthropic.com/v1/messages?beta=true`

The `?beta=true` query parameter is sent on every Messages request.

## Headers (in send order)

| Header | Value | Notes |
|---|---|---|
| `Accept` | `application/json` | always |
| `Authorization` | `Bearer <oauth>` | OAuth subscription token from credentials.json |
| `Content-Type` | `application/json` | always |
| `User-Agent` | `claude-cli/2.1.138 (external, sdk-cli)` | versioned with claude CLI release |
| `X-Claude-Code-Session-Id` | UUIDv4 | per-session, stable across requests in same session |
| `X-Stainless-Arch` | `x64` | Anthropic Node SDK telemetry |
| `X-Stainless-Lang` | `js` | |
| `X-Stainless-OS` | `Linux` | |
| `X-Stainless-Package-Version` | `0.93.0` | Anthropic SDK version bundled with claude CLI |
| `X-Stainless-Retry-Count` | `0` | increments on retries |
| `X-Stainless-Runtime` | `node` | |
| `X-Stainless-Runtime-Version` | `v24.3.0` | Node bundled by claude CLI |
| `X-Stainless-Timeout` | `600` | request timeout in seconds |
| `anthropic-beta` | (see below) | comma-separated list, varies by request kind |
| `anthropic-dangerous-direct-browser-access` | `true` | always |
| `anthropic-version` | `2023-06-01` | always |
| `x-app` | `cli` | always |
| `x-client-request-id` | UUIDv4 | per-request |

## anthropic-beta lists observed

**Default (main user reply, with tools, with thinking):**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,advisor-tool-2026-03-01,extended-cache-ttl-2025-04-11
```

**Title-generation / structured-output request:**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advisor-tool-2026-03-01,structured-outputs-2025-12-15
```

(Differences: `claude-code-20250219` and `extended-cache-ttl-2025-04-11` present on user replies; `structured-outputs-2025-12-15` present when sending an `output_config.format` JSON schema.)

CCP v2 will use the **Default** list for all requests as a starting point; revisit if Anthropic rejects on any specific consumer flow.

## Body structural notes

- `system` is an **array of text blocks** (not a flat string).
- The FIRST `system` block claude sends is a billing/telemetry header:
  ```
  x-anthropic-billing-header: cc_version=2.1.138.40b; cc_entrypoint=sdk-cli; cch=cd8d6;
  ```
  This is structured *as if* it were instruction text but is just claude's identity marker. CCP v2 should emit a similar marker as the first system block to ensure billing routes correctly to Max subscription.
- `metadata.user_id` is a **JSON-encoded string** containing `{device_id, account_uuid, session_id}`. Anthropic accepts this opaquely.
- `tools: []` empty array sent even when no tools used (so the field is always present).
- `temperature: 1`, `max_tokens: 32000` are claude defaults for haiku-4-5.
- `stream: true` is the default; non-streaming is rare from the CLI.

## Step 0 verification — confirmed working

The Step 0 minimal request (Haiku 4-5, OAuth Bearer, anthropic-beta with claude-code+oauth, custom non-CC system prompt) returned 200 with these headers — proving CCP v2 doesn't need byte-exact fingerprint match to function. Phase 5 (fingerprint hardening) is when we tighten to exact match.

## TODO during Phase 5

- Capture remaining 7 scenarios (with-tools, multi-turn, streaming, image, token-refresh, prompt-caching, errors).
- Diff CCP v2 captures against this baseline.
- Decide policy on `X-Stainless-*` (do we match exactly, or use our own values?).
- Consider whether `X-Claude-Code-Session-Id` should be derived from CCP request_id or be a stable per-process UUID.
- Verify HTTP/2 vs HTTP/1.1 — capture above is 1.1 because mitmproxy downgrades; re-check with TLS-passthrough or by capturing CCP's outbound.
