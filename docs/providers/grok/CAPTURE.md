# Grok Capture Procedure

Use this when xAI changes request requirements, headers, model access, streaming
shape, or credential behavior.

## Safety

- Captures may contain live bearer tokens.
- Never commit raw captures or unredacted headers.
- Do not call xAI or run a MITM capture without explicit operator approval.

## Shared Capture CLI

For provider wire drift, prefer the shared capture framework in `tools/capture/`:

```sh
# General capture (requires OMNI_CAPTURE_LIVE=1 or --live-capture)
python3 -m tools.capture capture run --provider grok --mode general --live-capture

# Refresh capture forces stale credentials and also needs OMNI_CAPTURE_REFRESH=1
python3 -m tools.capture capture run --provider grok --mode refresh \
  --live-capture --refresh-capture

# Dry-run prints the planned mitmdump and grok commands without network I/O
python3 -m tools.capture capture run --provider grok --mode general --dry-run
```

Dry-run uses placeholder credential paths only. It does not copy real credentials
or create a tmpfs workdir.

The shared CLI stages credentials into a clean tmpfs HOME, drives `grok --single`
through a local mitmproxy, and writes a redacted Markdown extract. Live runs
remove the tmpfs workdir (including staged credential copies) by default.
`KEEP_FLOW=1` retains the workdir and raw flow on tmpfs and prints warnings.
Use `tools.capture extract flow` for mitmproxy `.flow` files. Refresh capture
requires OIDC credentials in `~/.grok/auth.json`; static xAI key files cannot be
force-expired.

`tools/providers/grok/capture/extract_grok_http.py` remains a compatibility
wrapper around `tools.capture extract jsonl` for sanitized JSONL exports.

## Procedure

1. Start a local Omni server with only Grok enabled:

   ```sh
   OMNI_PROVIDERS=grok cargo run -p omni -- --no-auth --port 18322
   ```

   Prefer a random or otherwise unused loopback port. Do not reuse a running
   operator instance.

2. Send a minimal non-stream request and a stream request through Omni, only
   after approval:

   ```sh
   curl -sS http://127.0.0.1:18322/v1/chat/completions \
     -H 'content-type: application/json' \
     -d '{"model":"grok","messages":[{"role":"user","content":"Say OK"}],"max_tokens":8}'
   ```

3. If wire details are needed, capture at one boundary:

   - Preferred for Omni behavior: point `GrokProvider::new_for_test` at
     wiremock and assert headers/body in Rust.
   - Preferred for provider drift: use a short-lived local proxy and redact
     `Authorization` before storing any report.

4. Extract and review with:

   ```sh
   python3 -m tools.capture extract jsonl <capture-jsonl> --provider grok
   ```

   Or the compatibility wrapper:

   ```sh
   python3 tools/providers/grok/capture/extract_grok_http.py <capture-jsonl>
   ```

   The expected JSONL input is one object per request with optional `method`,
   `url`, `headers`, `body`, `status`, and `response_headers` fields. The tool is
   intentionally simple so sanitized exports from mitmproxy or browser tooling can
   be normalized without preserving raw secrets.

5. Update code/tests:

   - `crates/provider-grok/src/lib.rs` for request/response mapping.
   - `crates/provider-grok/src/credentials.rs` for credential file changes.
   - Wiremock tests for auth, body shape, errors, and streaming frames.
   - `docs/grok-gate.md` if gate behavior changes.

6. Verify:

   ```sh
   cargo test -p provider-grok
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   ```

7. Optional live smoke, only with approval:

   ```sh
   OMNI_LIVE_TESTS=1 cargo test -p provider-grok test_send_real_if_key_present
   ```

## Done Criteria

- Default tests pass without xAI credentials or network.
- Any new xAI wire requirement is pinned in a hermetic test.
- Docs link to source files instead of copying volatile model lists.

## Current Chat Model Findings

Re-baselined against grok-shell **0.2.93** and xAI on **2026-07-11**.

### Conservative (cli-chat-proxy.grok.com `/v1/models`)

- `grok-4.5` (CLI default; reasoning efforts `low`/`medium`/`high`, default `high`)
- `grok-composer-2.5-fast`

Wire notes from live MITM of `grok --single`:
- Host: `cli-chat-proxy.grok.com`, path `POST /v1/responses`
- UA / version: `grok-shell/0.2.93 (linux; x86_64)`, `x-grok-client-version: 0.2.93`
- Fingerprint headers unchanged in shape from 0.2.77: `x-xai-token-auth`,
  `x-authenticateresponse`, `x-grok-client-identifier`, `x-grok-model-override`,
  `accept: text/event-stream`
- Main chat body: `model: "grok-4.5"`, `reasoning: { "effort": "high", "summary": "concise" }`,
  `include: ["reasoning.encrypted_content"]`, `store: false`, `stream: true`
- Session-title side call still uses model `grok-build` (not advertised in `/v1/models`)

### Extended (api.x.ai chat/completions, verified 200)

- `grok-4.5` (default; alias `grok`)
- `grok-4.3`
- `grok-build-0.1`
- `grok-4.20-0309-reasoning`
- `grok-4.20-0309-non-reasoning`
- work-but-unlisted: `grok-3`, `grok-4`, `grok-build` (alias `build`),
  `grok-composer-2.5-fast` (alias `composer`)

`grok-4.20-multi-agent-0309` is advertised on api.x.ai but is multi-agent-only and
is not listed by Omni's chat provider. Omni accepts `grok` as shorthand for
`grok-4.5` and `composer` as shorthand for `grok-composer-2.5-fast`, but
`/v1/models` emits only canonical upstream ids.

### Thinking / reasoning_effort

- Chat completions: top-level `"reasoning_effort": "low"|"medium"|"high"`
- Responses: nested `"reasoning": { "effort": "..." }`
- On `grok-4.5`, default is `high` and reasoning cannot be disabled (xAI docs +
  CLI model catalog). Omni maps `CanonicalReasoning.effort` when the client sets it;
  it does not invent a default when the client omits reasoning.
