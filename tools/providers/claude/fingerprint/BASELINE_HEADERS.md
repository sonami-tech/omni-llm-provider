# claude CLI baseline wire fingerprint

Active baseline: 2026-06-05 against local Claude Code 2.1.165 (re-baselined per REBASELINE.md + capture_baseline.sh). Omni keeps coherent compatibility profiles for 2.1.142 and newer only:

| Profile | Claude Code | SDK package | Runtime | Entrypoint | Source |
|---|---|---|---|---|---|
| `cc-2.1.165-sdk-cli` | `2.1.165` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy reverse-proxy capture (haiku/sonnet/opus + default), clean CWD, 2026-06-05 |
| `cc-2.1.162-sdk-cli` | `2.1.162` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy reverse-proxy capture (haiku/sonnet/opus + default), clean CWD, 2026-06-04 |
| `cc-2.1.161-sdk-cli` | `2.1.161` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy reverse-proxy capture (haiku/sonnet/opus + default), 2026-06-03 |
| `cc-2.1.158-sdk-cli` | `2.1.158` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy reverse-proxy capture (haiku/sonnet/opus + default), 2026-05-30 |
| `cc-2.1.154-sdk-cli` | `2.1.154` | `0.94.0` | `v24.3.0` | `sdk-cli` | local `ANTHROPIC_BASE_URL` fake-server probe, 2026-05-28 |
| `cc-2.1.150-sdk-cli` | `2.1.150` | `0.94.0` | `v24.3.0` | `sdk-cli` | local MITM reverse-proxy probe, 2026-05-25 |
| `cc-2.1.142-sdk-cli` | `2.1.142` | `0.94.0` | `v24.3.0` | `sdk-cli` | local debug probe, 2026-05-15 |

`latest` resolves to `cc-2.1.165-sdk-cli`, the newest known-good pinned profile. Do not make the default an automatic max-version calculation; only move `latest` after a profile has been re-baselined and live-smoked.

2026-06-12 note: local Claude Code 2.1.175 was captured against the live API and
accepted default, Fable, Opus, Sonnet, and Haiku requests. Its headers remain on
SDK package `0.94.0`, runtime `v24.3.0`, and Anthropic version `2023-06-01`.
However, the final transport `cch` changed and no longer matches the supported
xxHash64 body algorithm below. Because Claude fingerprint exactness is a hard
gate, Omni does not promote a 2.1.175 profile or move `latest` until that
checksum path is recovered and tested.

2.1.165 is a pure version bump from 2.1.162: the 2026-06-05 capture re-confirmed that the per-model beta lists, stainless versions (`0.94.0` / `v24.3.0`), wire defaults (opus 64k/no-temp/effort=high; sonnet & haiku 32k/temp=1; haiku no effort - all three confirmed on full untruncated bodies), default-model resolution (opus → `claude-opus-4-8`), the model catalog, and the cch algorithm (xxh64 seed `0x4d659218e32a3268`, validated `ok`, synthetic-probe cch `b5d33`) are all unchanged. Only the version string and the version-derived `cc_version` suffix moved (real on-wire billing header read `cc_version=2.1.165.492` for the "Say OK" probe). The model catalog is carried forward from 2.1.158/2.1.161/2.1.162 (the capture sets `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, which suppresses the startup `/v1/models` GET, so the catalog is not freshly GET-enumerable; all three pinned ids were confirmed accepted in real bodies and 2.1.165 carries no model rename).

2.1.162 was likewise a pure version bump from 2.1.161 (2026-06-04 capture; "Say OK" suffix `2.1.162.b87`). 2.1.161 was a pure version bump from 2.1.158 (2026-06-03 capture; "Say OK" suffix `2.1.161.d2b`).

Raw mitmproxy `.flow` files are not committed because they contain live bearer
tokens and account identifiers. To inspect a fresh local capture, keep the
`.flow` on tmpfs and run:

```sh
uv tool run --from mitmproxy python3 tools/providers/claude/fingerprint/extract_flow.py \
  /path/to/local/tmpfs/capture.flow
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

**Claude Code 2.1.165** (captured 2026-06-05, clean CWD), **2.1.162** (captured
2026-06-04, clean CWD), **and 2.1.161** (captured 2026-06-03): the DEFAULT/opus,
Sonnet, and Haiku beta lists are **byte-identical** to the 2.1.158 lists below -
verified per-model from each live capture, not assumed. The 2.1.158 strings
therefore stand as the 2.1.161/2.1.162/2.1.165 reference; no separate
2.1.161/2.1.162/2.1.165 listing is reproduced to avoid drift between copies of the
same value.

**Claude Code 2.1.158 DEFAULT (and opus resolution; captured with context-1m):**
```
claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

**Claude Code 2.1.158 Sonnet reply:**
```
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

**Claude Code 2.1.158 Haiku reply:**
```
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11
```

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

Omni v2.1.154 selects the observed reply list by outbound model. Older profiles
retain their original default list.

## Body structural notes

- `system` is an **array of text blocks** (not a flat string).
- The FIRST `system` block claude sends is a billing/telemetry header:
  ```
  x-anthropic-billing-header: cc_version=2.1.154.cea; cc_entrypoint=sdk-cli; cch=00000;
  ```
  This is structured *as if* it were instruction text but is just claude's identity marker. Omni v2 emits this marker as the first system block, followed by the canonical Claude Code preamble block, before user-provided system content.
- Claude Code 2.1.142, 2.1.150, 2.1.154, 2.1.158, 2.1.161, 2.1.162, and 2.1.165's visible attribution builder and debug log emit `cch=00000`, but the final HTTP body rewrites that sentinel to a deterministic five-hex checksum. Omni mirrors the recovered final-body algorithm for these pinned profiles: standard `xxHash64` over the exact serialized body bytes while `cch=00000` is still present, seed `0x4d659218e32a3268`, then `hash & 0xfffff` formatted as five lowercase hex digits. See `tools/providers/claude/fingerprint/CCH_ALGORITHM.md`.
- The `cc_version` suffix is dynamic per request (2.1.165 "Say OK" suffix = 492; 2.1.162 was b87; 2.1.161 was d2b; 2.1.158 was 175):
  ```
  suffix = sha256("59cf53e54c78" + chars + claude_version).hex()[0..3]
  chars = first_user_text[4] + first_user_text[7] + first_user_text[20]
  ```
  Missing positions are the literal ASCII `0`. Indices are JavaScript UTF-16 string indices, not Rust scalar-value indices. The source text is the first text content in the first user message after Omni outbound replacements. If the first user message has no text block, use empty text and do not fall through to later user messages.
- `metadata.user_id` is a **JSON-encoded string** containing `{device_id, account_uuid, session_id}`. Anthropic accepts this opaquely.
- Claude Code `--tools ""` still sends a non-empty default SDK tool surface;
  Omni only sends consumer-provided tools.
- `temperature: 1`, `max_tokens: 32000` are Claude Code 2.1.154/2.1.158/2.1.161/2.1.162/2.1.165 defaults for
  Sonnet and Haiku (confirmed in 2026-05-30, 2026-06-03, and 2026-06-04 captures). Opus 4.8 omits temperature and uses `max_tokens: 64000`. On the wire, effort=high is carried as `output_config: {"effort":"high"}` (present on opus + sonnet, absent on haiku).
- `stream: true` is the default; non-streaming is rare from the CLI.
- 2.1.165 default_model (no --model) resolves to opus (observed 2026-06-05: body "model":"claude-opus-4-8"); same as 2.1.158/2.1.161/2.1.162.
- 2.1.165 catalog identical to 2.1.158/2.1.161/2.1.162/2.1.154 (no model-list GET in capture under `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`; bodies confirmed opus-4-8/sonnet-4-6/haiku-4-5 acceptance). The 165 profile reuses `CATALOG_CC_2_1_158` rather than duplicating an identical catalog.
- context-1m-2025-08-07 flag appears in 2.1.165/2.1.162/2.1.161/2.1.158 DEFAULT/opus calls (included in the DEFAULT beta).

## Step 0 verification - confirmed working

The Step 0 minimal request (Haiku 4-5, OAuth Bearer, anthropic-beta with claude-code+oauth, custom non-CC system prompt) returned 200 with these headers - proving Omni v2 doesn't need byte-exact fingerprint match to function. Omni now implements the recovered `cch` body checksum for active profiles (including 2.1.158).

## Remaining Baseline Work

- Capture remaining 7 scenarios (with-tools, multi-turn, streaming, image, token-refresh, prompt-caching, errors).
- Diff Omni v2 captures against this baseline.
- Watch documented differences that have not blocked OAuth use so far, including omitted `tools: []`, omitted default `temperature: 1`, and Omni-specific body field ordering.
- Run `tools/providers/claude/fingerprint/check_claude_code_drift.py` after Claude Code updates to detect installed-version or `cch` algorithm drift before moving `latest`. (Current active: 2.1.165).
