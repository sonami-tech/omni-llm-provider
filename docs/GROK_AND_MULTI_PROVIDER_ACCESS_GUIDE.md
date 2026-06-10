# Grok, xAI, and Multi-Provider Access Guide

**Purpose:** This document captures the full investigation and practical knowledge for connecting to different LLM providers (primarily xAI/Grok, with context on Anthropic/Claude and OpenAI compatibility) using normalized internal representations. It is written so that another LLM or developer can pick up exactly where this investigation left off, reproduce the MITM capture, understand the header "fingerprint", implement adapters for chat_completions vs responses vs messages formats, and integrate into an omni-provider like the one in this monorepo.

**Date of investigation:** June 2026 (based on timestamps and binary versions)
**Key projects referenced:**
- Grok Build CLI (`~/.grok/bin/grok`, v0.2.33)
- claude-code-provider (OpenAI Chat Completions surface + Anthropic Messages passthrough)
- redclaw/redclaw (normalized `rc:` model + Format::Anthropic + Format::OpenAi adapters)
- omni-llm-provider (this monorepo: frontends for `anthropic-messages` and `openai-chat`; providers for claude and grok)
- xAI public API (`api.x.ai/v1`) and internal CLI proxy (`cli-chat-proxy.grok.com/v1`)

---

## 1. The Three Core Formats

There are three primary wire formats relevant to agentic coding tools and omni providers:

1. **OpenAI Chat Completions** (`chat_completions`)
   - Endpoint: `POST /v1/chat/completions`
   - Request: `{ "model": "...", "messages": [...], "tools": [...], "tool_choice": ..., "max_tokens": ..., "stream": bool, "reasoning_effort": "low|medium|high" }`
   - Messages: array of `{ "role": "system|user|assistant|tool", "content": "...", "tool_calls": [...] }`
   - Tool calls: `tool_calls[].function.arguments` is a **JSON string**.
   - Response: `choices[0].message.content` + `tool_calls`.
   - Streaming: standard `data: {"choices": [{"delta": ...}]}`.
   - Used by: redclaw's `Format::OpenAi` adapter, claude-code-provider's primary OpenAI surface, most generic OpenAI SDK clients.
   - Strengths: Maximum compatibility. Weaknesses: Stateless, weaker native support for deep reasoning traces and agent state.

2. **OpenAI Responses** (`responses`)
   - Endpoint: `POST /v1/responses`
   - Request: `{ "model": "...", "input": [...], "tools": [...], "reasoning": {"summary": "concise"}, "stream": true, "store": false, "previous_response_id"?: "..." }`
   - `input` is more flexible than flat messages (supports richer structures for agentic flows).
   - Strong native support for reasoning, multi-turn state, computer-use tools, partial image generation, typed streaming events (`response.output_text.delta`, `response.reasoning_text.delta`, `response.function_call_arguments.delta`, etc.).
   - Used by: Grok Build CLI internally (primary for `grok-build` and Composer models), xAI public API (preferred for agentic models like `grok-build-0.1`).
   - In Grok CLI config: `api_backend = "responses"`.
   - Strengths: Best for long-horizon agents, rich reasoning, state. Weaknesses: Newer; not all clients/gateways support it yet.

3. **Anthropic Messages** (`messages`)
   - Endpoint: `POST /v1/messages` (or `/v1/messages/count_tokens`)
   - Request: `{ "model": "...", "system": "...", "messages": [...], "tools": [...], "thinking": {"type": "enabled", "budget_tokens": N}, "max_tokens": ..., "stream": bool }`
   - Messages contain `content` as an **array of blocks**: `{ "type": "text", "text": "..." }`, `{ "type": "tool_use", "id": "...", "name": "...", "input": {object} }`, `{ "type": "tool_result", "tool_use_id": "...", "content": "..." }`.
   - Tools use `input_schema` (JSON Schema object).
   - Strong native support for extended thinking (with signatures for replay), computer use, cache control.
   - Used by: redclaw's `Format::Anthropic`, claude-code-provider's native surface and all upstream calls.
   - In Grok CLI config: `api_backend = "messages"`.
   - Strengths: Excellent for Claude's thinking model and exact wire fingerprinting. Weaknesses: Different from OpenAI shapes; requires careful block handling.

**Key insight:** All three are *semantically* about the same thing (conversation turns + tool calls/results + optional reasoning). They differ in:
- Message representation (flat roles vs content blocks vs flexible input).
- Tool argument serialization (stringified JSON vs native objects).
- Reasoning/thinking (effort string vs budget + signed blocks vs native reasoning objects).
- Streaming events.
- State and advanced agent features (computer use, partials, multi-agent).

They are **translatable** with a canonical internal representation (see redclaw's `RcRequest`/`RcMessage`/`RcToolCall` etc., and claude-code-provider's translate layers). Translation is "good enough" for text + basic tools, but lossy for exact reasoning signatures, certain tool types, and streaming fidelity. High-fidelity agents (Grok Build, Claude Code) require preserving provider-specific details.

---

## 2. Models Available (Grok / xAI Focus)

### Internal CLI models (from `~/.grok/models_cache.json` and `grok models`)
These are what the Grok Build TUI actually uses:
- `grok-build` (default for advanced coding; 512k context in cache; "Best for advanced coding tasks"; uses `grok-build-plan` agent type; supports backend search).
- `grok-composer-2.5-fast` ("Composer 2.5" — "Cursor's latest coding model"; 200k context; `agent_type: "cursor"`).

Run `grok models` locally for the live list (it prefetches from the proxy).

### Public xAI API models (from `api.x.ai/v1` and docs)
These are the documented, stable slugs for external use:
- `grok-4.3` (flagship; 1M context; strong agentic tool calling).
- `grok-build-0.1` (the public coding/agentic model that powers Grok Build; 256k context; "fast coding model trained specifically for agentic coding workflows"; early access at time of investigation; pricing ~$1/M input, $2/M output).
- `grok-4.20-0309-reasoning`
- `grok-4.20-0309-non-reasoning`
- `grok-4.20-multi-agent-0309`

Aliases and older names (e.g., `grok-code-fast-1`) often redirect to `grok-build-0.1`.

**Public API base:** `https://api.x.ai/v1` (standard OpenAI SDK with `XAI_API_KEY` or session token).

**Internal proxy base (what Grok Build CLI uses):** `https://cli-chat-proxy.grok.com/v1`

The proxy requires special "fingerprint" headers (see below) and the session JWT. Public API uses standard Bearer + `XAI_API_KEY`.

---

## 3. How to Connect — Full MITM Investigation and Header Discovery

### Background and Why MITM Was Needed
The Grok Build CLI (`~/.grok/bin/grok`) talks to `cli-chat-proxy.grok.com/v1` using the user's `grok login` session (OIDC JWT stored in `~/.grok/auth.json`).

Early attempts to call the proxy directly (using the JWT as Bearer + obvious headers like `X-XAI-Token-Auth: xai-grok-cli` and `x-grok-model-override`) failed with:
```
{"error":"Your Grok CLI version (none) is outdated. Please update to version 0.1.202 or later via `grok update`..."}
```

The server enforces a strict client "fingerprint" (version + identifier + exact User-Agent + other `x-grok-*` headers). The official Rust binary sends a precise set that plain `curl`/Python requests did not match. We needed to observe the *live* traffic from the running binary.

### MITM Capture Process (Using mitmproxy)
We used the available `mitmproxy` (found at `/home/username/.browser-use-env/bin/mitmdump`).

**Exact command pattern used (adapted for the environment):**
```bash
MITM_LOG=/tmp/mitm_grok_headers.log
> "$MITM_LOG"

/home/username/.browser-use-env/bin/mitmdump \
  --listen-port 8080 \
  --set flow_detail=3 \
  --set console_eventlog_verbosity=debug \
  > "$MITM_LOG" 2>&1 &

MITM_PID=$!
sleep 4

export HTTPS_PROXY=http://127.0.0.1:8080
export HTTP_PROXY=http://127.0.0.1:8080
export NO_PROXY=localhost,127.0.0.1

# Provide mitm CA so the Rust/rustls binary trusts the MITM cert
MITM_CA=$(python3 -c "
import os, glob
certs = glob.glob(os.path.expanduser('~/.mitmproxy/mitmproxy-ca-cert.pem'))
print(certs[0] if certs else '')
")
[ -n "$MITM_CA" ] && export SSL_CERT_FILE="$MITM_CA"

# Trigger a real inference call (minimal prompt to avoid heavy tool use)
timeout 35 ~/.grok/bin/grok \
  -p "Reply with the single word: HELLO" \
  --output-format json \
  --always-approve \
  --max-turns 1 \
  --no-memory \
  2>&1 | cat

kill $MITM_PID 2>/dev/null || true
sleep 2
cat "$MITM_LOG"
```

**What the capture revealed (key flows):**
- The binary first does GETs to `/v1/settings` (to fetch config like default model, tips, etc.).
- Real inference goes to `POST /v1/responses` (Responses API shape, **not** `/chat/completions`).
- It sends the full fingerprint on every call.
- Body for inference used Responses format: `"input": [ { "type": "message", "role": "...", "content": "..." } ]`, `"model": "grok-build"`, `"reasoning": {...}`, `"stream": true`, internal tools (e.g., for session title), etc.
- Multiple `x-grok-*` headers carried context (some empty on simple calls).

### Discovered Headers (Exact from Capture)
These are the **full set** that made the official binary succeed. Replicate them *exactly* (casing, values, presence of empty headers where shown).

**Core auth + fingerprint (present on settings and inference):**
- `authorization: Bearer <full JWT from ~/.grok/auth.json>`
- `x-xai-token-auth: xai-grok-cli`
- `x-grok-client-version: 0.2.33`
- `x-grok-client-identifier: grok-shell`
- `user-agent: grok-shell/0.2.33 (linux; x86_64)`
- `accept-encoding: gzip, br, deflate`

**Inference-specific (on the /v1/responses POST):**
- `x-authenticateresponse: authenticate-response`
- `x-grok-user-id: 8fdddb7e-cd88-4492-901a-b7c328a1a6ab` (from auth.json)
- `x-grok-model-override: grok-build` (or `grok-composer-2.5-fast`, etc.)
- `x-grok-conv-id: ` (empty string in trace)
- `x-grok-req-id: ` (empty)
- `x-grok-session-id: ` (empty)
- `x-grok-agent-id: ` (empty)
- `content-type: application/json`
- `accept: text/event-stream`
- `content-length: <exact>`

**Settings call extras (sometimes present):**
- `x-userid: <same as user-id>`
- `x-email: <from auth.json>`

**How to extract the token in code (Python example):**
```python
import json, os
auth = json.load(open(os.path.expanduser("~/.grok/auth.json")))
token = auth["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"]["key"]
user_id = auth["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"].get("user_id")
```

**JWT note:** These are OIDC tokens from `auth.x.ai`. They have `refresh_token` in the file; the CLI refreshes automatically. For external use, treat as short-lived and refresh or fall back to `XAI_API_KEY` on the public API.

### Successful Connection Examples (After Matching Fingerprint)

**Using raw urllib (Python, no extra deps):**
(See the exact script from the investigation that produced 200s for all seven models on `/responses` and most on `/chat/completions`. It used the headers above + `x-grok-model-override`.)

**Using official OpenAI SDK (recommended for external apps):**
```python
from openai import OpenAI
import json, os

auth = json.load(open(os.path.expanduser("~/.grok/auth.json")))
token = auth["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"]["key"]
user_id = auth["https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828"].get("user_id")

extra = {
    "X-XAI-Token-Auth": "xai-grok-cli",
    "x-grok-client-version": "0.2.33",
    "x-grok-client-identifier": "grok-shell",
    "User-Agent": "grok-shell/0.2.33 (linux; x86_64)",
    "x-grok-user-id": user_id,
    "x-authenticateresponse": "authenticate-response",
    "x-grok-model-override": "grok-build",  # per-call
}

client = OpenAI(
    base_url="https://cli-chat-proxy.grok.com/v1",
    api_key=token,
)

resp = client.chat.completions.create(
    model="grok-build",
    messages=[{"role": "user", "content": "Reply with the single word: SUCCESS"}],
    max_tokens=5,
    extra_headers=extra,
)
print(resp.choices[0].message.content)
```

**Responses shape (what the binary actually sent):**
```python
resp = client.responses.create(
    model="grok-build",
    input=[{"role": "user", "content": prompt}],
    max_output_tokens=5,
    extra_headers=extra,  # same fingerprint
    stream=True,
)
```

**Results from direct tests (with exact headers):**
- All seven models succeeded on `/responses`.
- Most succeeded on `/chat/completions`.
- `grok-4.20-multi-agent-0309` failed on chat.completions with "Multi Agent requests are not allowed on chat completions".
- `grok-composer-2.5-fast` was reliable on responses but occasionally flaky on chat.completions.
- The proxy returns compressed responses; real SDKs handle decompression.

**For public API (no fingerprint needed):**
Use `https://api.x.ai/v1` + `XAI_API_KEY` (or the same JWT) + public slugs (`grok-build-0.1`, `grok-4.3`, etc.). Standard OpenAI SDK works without the `x-grok-*` headers.

### Redclaw/redclaw Approach (for comparison)
- Uses `Format::OpenAi` → hard `POST {base}/chat/completions` + Chat Completions body.
- Normalized `Rc*` types for tool args (parses stringified OpenAI args into structured JSON).
- No Responses support (only chat_completions + anthropic messages).
- Excellent for generic OpenAI-compatible backends (Ollama, vLLM, OpenRouter, etc.).

### claude-code-provider Approach
- Exposes OpenAI Chat Completions (`/v1/chat/completions`) + native Anthropic Messages (`/v1/messages`).
- Heavy bidirectional translation + fingerprinting for Claude Code wire (preamble injection, cch checksums, betas, signed thinking blocks).
- No Responses surface in the main code (focus is Claude fidelity).

### Omni-llm-provider Integration Notes
Current structure (as built):
- OpenAI Chat Completions surface lives in `crates/omni-common/src/http.rs`
  (request/response types, canonical conversion, SSE framing) and is served by
  all three binaries.
- Back-end specific logic: `crates/provider-grok/` and `crates/provider-claude/`.
- The OpenAI Responses shape (`/v1/responses`) and a native Anthropic Messages
  inbound surface are NOT yet implemented; they would be added as new modules in
  `omni-common::http` (see the Responses follow-up task).

**Recommended canonical internal model (inspired by redclaw):**
Use something like `RcRequest` (system + messages + tools + max_tokens) + rich `RcMessage` (text + tool_calls + tool_results) + separate thinking/reasoning metadata. Then have adapters:
- `to_openai_chat_completions(...)`
- `to_openai_responses(...)`
- `to_anthropic_messages(...)`

Outbound: apply replacements, then format-specific mutation (Grok fingerprint headers, Claude preamble/cch, etc.).
Inbound: normalize back to canonical, then apply response replacements.

### Gotchas & Lessons Learned
1. **Version/fingerprint is server-enforced and brittle.** Always capture from the *official* binary for that exact version. "0.2.33" worked; "none" or wrong UA failed.
2. **JWT handling:** Read from `~/.grok/auth.json`. The CLI refreshes it; external code must handle expiry (or use `XAI_API_KEY` on public API).
3. **Endpoint choice matters:** Internal models often require `/responses` + specific body for full agentic behavior.
4. **Compression:** Many responses come gzipped/br; let the HTTP client handle it.
5. **Model names:** Internal names (`grok-build`) vs public (`grok-build-0.1`). Use `x-grok-model-override` on the proxy.
6. **Streaming vs non-stream:** Proxy often prefers streaming internally; non-stream works for simple tests.
7. **For other LLMs:** Re-run the exact mitm command with the target binary (grok, claude, etc.) to re-capture when versions change. Keep `SSLKEYLOGFILE` + pcap as backup when strace can't see plaintext (TLS).
8. **Translation cost:** Expect to maintain per-format tool arg serialization, reasoning block mapping, and error shaping. Test with real agent loops (not just "say hello").

### Reproducible Next Steps for Another LLM
1. `ls ~/.grok/auth.json` and extract the JWT + user_id.
2. Install mitmproxy if needed; run the capture command above with a fresh prompt.
3. Parse the log for the POST to `/v1/responses` (or `/chat/completions`).
4. Copy the exact header set into your client (OpenAI SDK `default_headers` or `extra_headers`).
5. Test `client.chat.completions.create` and `client.responses.create` (if SDK supports) with `x-grok-model-override`.
6. For omni-llm-provider: implement a `grok` provider crate that injects the fingerprint and chooses `/responses` for agentic models.
7. Add tests that assert the exact headers are sent (like the contract tests in redclaw-llm).
8. When the binary updates, re-capture and bump the version string in code/config.

This report should let another system (or LLM) reproduce the entire connection process, understand why plain OpenAI calls fail, and implement robust multi-format support without repeating the discovery work.

**References to prior artifacts in this repo:**
- `docs/INVESTIGATION.md` (monorepo/translation analysis)
- `docs/reference-architecture.md`
- `research/claude-code-programmatic-usage.md` and related (fingerprint work)
- Captured mitmproxy logs from the investigation session (search session JSONL for "cli-chat-proxy" or the exact JWT prefix)
- `~/.grok/models_cache.json` (live internal model list)
- `~/.grok/auth.json` (live token source)

Update this document with new captures when versions change.
