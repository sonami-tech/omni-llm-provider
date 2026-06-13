# Grok Capture Procedure

Use this when xAI changes request requirements, headers, model access, streaming
shape, or credential behavior.

## Safety

- Captures may contain live bearer tokens.
- Never commit raw captures or unredacted headers.
- Do not call xAI or run a MITM capture without explicit operator approval.

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

Verified against xAI chat completions on 2026-06-12:

- `grok-4.3`
- `grok-build`
- `grok-build-0.1`
- `grok-composer-2.5-fast`
- `grok-4.20-0309-reasoning`
- `grok-4.20-0309-non-reasoning`

`grok-4.20-multi-agent-0309` rejected chat completions with a multi-agent-only
error and is not listed by Omni's chat provider. Omni accepts `grok` as shorthand
for `grok-4.3` and `composer` as shorthand for `grok-composer-2.5-fast`, but
`/v1/models` emits only canonical upstream ids.
