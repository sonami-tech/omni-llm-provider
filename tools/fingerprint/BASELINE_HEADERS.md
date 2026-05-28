# claude CLI baseline wire fingerprint

Active baseline: 2026-05-28 against local Claude Code 2.1.154. CCP keeps coherent compatibility profiles for 2.1.142 and newer only:

| Profile | Claude Code | SDK package | Runtime | Entrypoint | Source |
|---|---|---|---|---|---|
| `cc-2.1.154-sdk-cli` | `2.1.154` | `0.94.0` | `v24.3.0` | `sdk-cli` | local `ANTHROPIC_BASE_URL` fake-server probe, 2026-05-28 |
| `cc-2.1.150-sdk-cli` | `2.1.150` | `0.94.0` | `v24.3.0` | `sdk-cli` | local MITM reverse-proxy probe, 2026-05-25 |
| `cc-2.1.142-sdk-cli` | `2.1.142` | `0.94.0` | `v24.3.0` | `sdk-cli` | local debug probe, 2026-05-15 |

`latest` resolves to `cc-2.1.154-sdk-cli`, the newest known-good pinned profile. Do not make the default an automatic max-version calculation; only move `latest` after a profile has been re-baselined and live-smoked.

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
| `User-Agent` | `claude-cli/<profile version> (external, sdk-cli)` | versioned with pinned Claude Code compatibility profile |
| `X-Claude-Code-Session-Id` | UUIDv4 | per-session, stable across requests in same session |
| `X-Stainless-Arch` | `x64` | Anthropic Node SDK telemetry |
| `X-Stainless-Lang` | `js` | |
| `X-Stainless-OS` | `Linux` | |
| `X-Stainless-Package-Version` | `<profile SDK package>` | Anthropic SDK version bundled with pinned Claude Code profile |
| `X-Stainless-Retry-Count` | `0` | increments on retries |
| `X-Stainless-Runtime` | `node` | |
| `X-Stainless-Runtime-Version` | `<profile runtime>` | Node bundled by claude CLI |
| `X-Stainless-Timeout` | `600` | request timeout in seconds |
| `anthropic-beta` | (see below) | comma-separated list, varies by request kind |
| `anthropic-dangerous-direct-browser-access` | `true` | always |
| `anthropic-version` | `2023-06-01` | always |
| `x-app` | `cli` | always |
| `x-client-request-id` | UUIDv4 | per-request |

## anthropic-beta lists observed

**Claude Code 2.1.154 default / Opus 4.8 reply:**
```
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

**Claude Code 2.1.154 Sonnet / legacy Opus reply:**
```
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

**Claude Code 2.1.154 Haiku reply:**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11
```

**Legacy 2.1.150 default (main user reply, with tools, with thinking):**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,advisor-tool-2026-03-01,extended-cache-ttl-2025-04-11
```

**Title-generation / structured-output request:**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,advisor-tool-2026-03-01,structured-outputs-2025-12-15
```

(Differences: `claude-code-20250219` and `extended-cache-ttl-2025-04-11` present on user replies; `structured-outputs-2025-12-15` present when sending an `output_config.format` JSON schema.)

CCP v2.1.154 selects the observed reply list by outbound model. Older profiles
retain their original default list.

## Body structural notes

- `system` is an **array of text blocks** (not a flat string).
- The FIRST `system` block claude sends is a billing/telemetry header:
  ```
  x-anthropic-billing-header: cc_version=2.1.154.cea; cc_entrypoint=sdk-cli; cch=00000;
  ```
  This is structured *as if* it were instruction text but is just claude's identity marker. CCP v2 emits this marker as the first system block, followed by the canonical Claude Code preamble block, before user-provided system content.
- Claude Code 2.1.142, 2.1.150, and 2.1.154's visible attribution builder and debug log emit `cch=00000`, but the final HTTP body rewrites that sentinel to a deterministic five-hex checksum. CCP mirrors the recovered final-body algorithm for these pinned profiles: standard `xxHash64` over the exact serialized body bytes while `cch=00000` is still present, seed `0x4d659218e32a3268`, then `hash & 0xfffff` formatted as five lowercase hex digits. See `tools/fingerprint/CCH_ALGORITHM.md`.
- The `cc_version` suffix is dynamic per request:
  ```
  suffix = sha256("59cf53e54c78" + chars + claude_version).hex()[0..3]
  chars = first_user_text[4] + first_user_text[7] + first_user_text[20]
  ```
  Missing positions are the literal ASCII `0`. Indices are JavaScript UTF-16 string indices, not Rust scalar-value indices. The source text is the first text content in the first user message after CCP outbound replacements. If the first user message has no text block, use empty text and do not fall through to later user messages.
- `metadata.user_id` is a **JSON-encoded string** containing `{device_id, account_uuid, session_id}`. Anthropic accepts this opaquely.
- Claude Code `--tools ""` still sends a non-empty default SDK tool surface;
  CCP only sends consumer-provided tools.
- `temperature: 1`, `max_tokens: 32000` are Claude Code 2.1.154 defaults for
  Sonnet and Haiku. Opus 4.8 omits temperature and uses `max_tokens: 64000`.
- `stream: true` is the default; non-streaming is rare from the CLI.

## Step 0 verification — confirmed working

The Step 0 minimal request (Haiku 4-5, OAuth Bearer, anthropic-beta with claude-code+oauth, custom non-CC system prompt) returned 200 with these headers — proving CCP v2 doesn't need byte-exact fingerprint match to function. CCP now implements the recovered `cch` body checksum for active profiles.

## Remaining Baseline Work

- Capture remaining 7 scenarios (with-tools, multi-turn, streaming, image, token-refresh, prompt-caching, errors).
- Diff CCP v2 captures against this baseline.
- Watch documented differences that have not blocked OAuth use so far, including omitted `tools: []`, omitted default `temperature: 1`, and CCP-specific body field ordering.
- Run `tools/fingerprint/check_claude_code_drift.py` after Claude Code updates to detect installed-version or `cch` algorithm drift before moving `latest`.
