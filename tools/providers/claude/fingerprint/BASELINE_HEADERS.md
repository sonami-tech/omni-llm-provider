# Claude CLI Baseline Wire Fingerprint

Active baseline: Claude Code 2.1.197, captured 2026-07-01. `latest` resolves to
`cc-2.1.197-sdk-cli`.

Raw mitmproxy `.flow` files are not committed because they contain live bearer
tokens and account identifiers. Keep raw captures on tmpfs and inspect them with
`tools/providers/claude/fingerprint/extract_flow.py`.

## Profiles

| Profile | Claude Code | SDK package | Runtime | Entrypoint | Source |
|---|---|---|---|---|---|
| `cc-2.1.197-sdk-cli` | `2.1.197` | `0.94.0` | `v26.3.0` | `sdk-cli` | live shared-capture mitmproxy, Opus/Sonnet/Haiku/default, 2026-07-01 |
| `cc-2.1.186-sdk-cli` | `2.1.186` | `0.94.0` | `v24.3.0` | `sdk-cli` | live shared-capture mitmproxy, default/Opus/Sonnet/Haiku, 2026-06-22 |
| `cc-2.1.175-sdk-cli` | `2.1.175` | `0.94.0` | `v24.3.0` | `sdk-cli` | fake-server and live capture, Fable/Opus/Sonnet/Haiku/default, 2026-06-12 |
| `cc-2.1.165-sdk-cli` | `2.1.165` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy capture, 2026-06-05 |
| `cc-2.1.162-sdk-cli` | `2.1.162` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy capture, 2026-06-04 |
| `cc-2.1.161-sdk-cli` | `2.1.161` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy capture, 2026-06-03 |
| `cc-2.1.158-sdk-cli` | `2.1.158` | `0.94.0` | `v24.3.0` | `sdk-cli` | mitmproxy capture, 2026-05-30 |
| `cc-2.1.154-sdk-cli` | `2.1.154` | `0.94.0` | `v24.3.0` | `sdk-cli` | fake-server probe, 2026-05-28 |
| `cc-2.1.150-sdk-cli` | `2.1.150` | `0.94.0` | `v24.3.0` | `sdk-cli` | reverse-proxy probe, 2026-05-25 |
| `cc-2.1.142-sdk-cli` | `2.1.142` | `0.94.0` | `v24.3.0` | `sdk-cli` | local debug probe, 2026-05-15 |

Do not make `latest` an automatic max-version calculation. Move it only after a
profile is re-baselined, covered by vectors, and live-smoked.

## Endpoint And Headers

`POST https://api.anthropic.com/v1/messages?beta=true`

Header names and dynamic values are pinned in
`crates/provider-claude/src/fingerprint.rs`. Static send-order members are:

- `Accept: application/json`
- `Authorization: Bearer <oauth>`
- `Content-Type: application/json`
- `User-Agent: claude-cli/<profile version> (external, sdk-cli)`
- `X-Claude-Code-Session-Id: <uuid>`
- `X-Stainless-Arch: x64`
- `X-Stainless-Lang: js`
- `X-Stainless-OS: Linux`
- `X-Stainless-Package-Version: <profile SDK package>`
- `X-Stainless-Retry-Count: <retry count>`
- `X-Stainless-Runtime: node`
- `X-Stainless-Runtime-Version: <profile runtime>`
- `X-Stainless-Timeout: 600`
- `anthropic-beta: <per-model list>`
- `anthropic-dangerous-direct-browser-access: true`
- `anthropic-version: 2023-06-01`
- `x-app: cli`
- `x-client-request-id: <uuid>`

## 2.1.175 Model Surface

| Input | Wire model | Beta list | max_tokens | temperature | output_config.effort |
|---|---|---|---:|---:|---|
| no `--model` | `claude-opus-4-8` | default | 64000 | omitted | `xhigh` |
| `opus` | `claude-opus-4-8` | opus | 64000 | omitted | `xhigh` |
| `fable` | `claude-fable-5` | fable | 64000 | omitted | `xhigh` |
| `sonnet` | `claude-sonnet-4-6` | sonnet | 32000 | `1` | `high` |
| `haiku` | `claude-haiku-4-5-20251001` | haiku | 32000 | `1` | omitted |
| `claude-haiku-4-5` | `claude-haiku-4-5` | haiku | 32000 | `1` | omitted |

Default beta:

```text
claude-code-20250219,oauth-2025-04-20,context-1m-2025-08-07,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

Explicit Opus beta:

```text
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

Fable beta:

```text
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,effort-2025-11-24,fallback-credit-2026-06-01,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

Sonnet beta:

```text
claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,effort-2025-11-24,afk-mode-2026-01-31,extended-cache-ttl-2025-04-11
```

Haiku beta:

```text
oauth-2025-04-20,interleaved-thinking-2025-05-14,context-management-2025-06-27,prompt-caching-scope-2026-01-05,claude-code-20250219,extended-cache-ttl-2025-04-11
```

## Body Notes

- `system` is an array of text blocks.
- The first `system` block is the billing marker:

  ```text
  x-anthropic-billing-header: cc_version=2.1.175.174; cc_entrypoint=sdk-cli; cch=00000;
  ```

- The `cc_version` suffix remains:

  ```text
  suffix = sha256("59cf53e54c78" + chars + claude_version).hex()[0..3]
  chars = first_user_text[4] + first_user_text[7] + first_user_text[20]
  ```

- `metadata.user_id` is a JSON-encoded string containing device, account, and
  session identifiers.
- Claude Code `--tools ""` still sends a default SDK tool surface. Omni only
  sends consumer-provided tools.
- `stream: true` is the CLI default.
- cch behavior is version-specific. See
  `tools/providers/claude/fingerprint/CCH_ALGORITHM.md`.

## Maintenance

Run this after Claude Code updates:

```sh
uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py
```

Regenerate vectors only after the pinned profile has been updated:

```sh
uv run --script tools/providers/claude/fingerprint/check_claude_code_drift.py \
  --emit-vectors tools/providers/claude/fingerprint/vectors
```
