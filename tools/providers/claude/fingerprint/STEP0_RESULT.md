# Phase 0 Step 0 - Result: PASS

Date: 2026-05-10
Model used for verification: `claude-haiku-4-5-20251001`

## What was verified

OAuth subscription token from `~/.claude/.credentials.json` accepted by `api.anthropic.com/v1/messages` with the `claude-code-20250219` beta header AND a non-Claude-Code system prompt.

This was the blocking gate for the entire Omni v2 plan. It confirms:

1. OAuth tokens (`sk-ant-oat01-…`) authenticate via `Authorization: Bearer …`.
2. `anthropic-beta: oauth-2025-04-20` is the beta required for Bearer-token acceptance.
3. `anthropic-beta: claude-code-20250219` does NOT require Claude Code's own system prompt - arbitrary system prompts work fine alongside it.
4. `interleaved-thinking-2025-05-14` is accepted in the same list.
5. Bypassing the claude CLI subprocess is feasible.

## Minimal working request

```bash
TOKEN=$(jq -r '.claudeAiOauth.accessToken' ~/.claude/.credentials.json)
curl -sS -X POST https://api.anthropic.com/v1/messages \
  -H "Authorization: Bearer $TOKEN" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: claude-code-20250219,oauth-2025-04-20,interleaved-thinking-2025-05-14" \
  -H "Content-Type: application/json" \
  -H "User-Agent: claude-cli/2.1.138 (external, cli)" \
  -d '{
    "model": "claude-haiku-4-5-20251001",
    "max_tokens": 32,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Say OK"}]
  }'
```

Result: HTTP 200, `{"content":[{"type":"text","text":"OK"}], …}`.

## Observations on Opus/Sonnet 4.6

`claude-opus-4-6`, `claude-opus-4-7`, `claude-sonnet-4-6` returned `429 rate_limit_error` on the same headers - auth+format accepted, real subscription rate limit hit. This is exactly the pass-through behavior locked into the plan (decision #2).

## Model ID notes

- Alias `claude-haiku-4-5` resolves server-side to `claude-haiku-4-5-20251001`.
- Aliases `claude-opus-4-6`, `claude-opus-4-7`, `claude-sonnet-4-6` are valid (return 429, not 404).
- Dated forms guessed from the env (e.g., `claude-sonnet-4-6-20251201`) returned 404 - use aliases or capture real dated forms from claude itself.

## Cleared to proceed

Phase 1 is unblocked.

## Scope cut: 8-scenario capture deferred to Phase 5

The plan called for capturing 8 baseline `claude` scenarios in Phase 0. Skipped for now because:

1. mitmproxy is not installed and the `claude` CLI is an ELF binary (likely `sea`/`pkg` bundled Node, which doesn't reliably honor `NODE_OPTIONS` - undici monkeypatch backup is unreliable).
2. Step 0 verified the headers/auth approach works without precise fingerprint matching.
3. Captures are only directly used at Phase 5 (diff harness vs. Omni captures). Doing them then keeps the install + capture work co-located with the diff work.

When Phase 5 arrives:
- `pip install mitmproxy` (or `pipx install mitmproxy`).
- Run `mitmdump --set save_stream_file=…` while running claude with `HTTPS_PROXY=http://127.0.0.1:8080` and `NODE_EXTRA_CA_CERTS=~/.mitmproxy/mitmproxy-ca.pem`.
- If the ELF bundle ignores those env vars, fall back to a transparent proxy via iptables on a single user-namespaced session.

Headers used in Step 0 are the working baseline; Phase 5 will tighten matches.
