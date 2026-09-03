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

5. Obtain this pin's catalog from captured `GET /v1/models` on
   `cli-chat-proxy.grok.com` (`python3 -m tools.capture catalog --provider grok
   --flow-file ...`). If that listing is missing or empty, stop. Do not keep
   the previous pin's catalog.

   The shared capture already uses a clean tmpfs HOME. Do **not** take the
   catalog or default from the operator machine's `grok models` UI or from
   `~/.grok/config.toml` (`[models] default`, custom `[model.*]` entries).
   Those are local overrides. Confirm the pin default from the no-`--model`
   `POST /v1/responses` body and from proxy `default_model`.

6. Update code/tests:

   - `crates/provider-grok/src/lib.rs` for request/response mapping.
   - `crates/provider-grok/src/credentials.rs` for credential file changes.
   - Wiremock tests for auth, body shape, errors, and streaming frames.
   - `docs/grok-gate.md` if gate behavior changes.

7. Verify:

   ```sh
   cargo test -p provider-grok
   cargo test --workspace
   cargo clippy --workspace --all-targets -- -D warnings
   ```

8. Optional live smoke, only with approval:

   ```sh
   OMNI_LIVE_TESTS=1 cargo test -p provider-grok test_send_real_if_key_present
   ```

## Done Criteria

- Default tests pass without xAI credentials or network.
- Any new xAI wire requirement is pinned in a hermetic test.
- Docs link to source files instead of copying volatile model lists.

## Current Wire + Model Findings

Re-baselined against grok-shell **1.0.13** on **2026-09-03**.

### Default path (cli-chat-proxy.grok.com)

Catalog (`/v1/models`):

- `grok-4.6` (default; alias `grok`; reasoning efforts `low`/`medium`/`high`/`xhigh`; top-level and captured chat default `high`)
- `grok-4.5` (still advertised; no inbound alias)

An operator `grok models` listing may show custom models and a non-Grok
default (for example `gpt-luna` from `[models] default` in
`~/.grok/config.toml`). That is not the provider catalog. This pin uses only
`cli-chat-proxy.grok.com /v1/models` from the clean-HOME capture.

Wire notes from live MITM of `grok --single`:
- Host: `cli-chat-proxy.grok.com`, path `POST /v1/responses`
- UA / version: `grok-shell/1.0.13 (linux; x86_64)`, `x-grok-client-version: 1.0.13`
- Fingerprint headers: `x-xai-token-auth`, `x-authenticateresponse`,
  `x-grok-client-identifier`, `x-grok-client-mode: headless`,
  `x-grok-model-override`, `accept: text/event-stream`
- Main chat body: `model: "grok-4.6"`, `reasoning: { "effort": "high", "summary": "concise" }`,
  `include: ["reasoning.encrypted_content", "no_inline_citations"]`, `store: false`, `stream: true`,
  plus CLI-only `prompt_cache_key` (Omni sends cache fields only from client intent)
- Session-title side call uses the selected chat model (not `grok-build`)
- CLI also sends session/compaction headers (`x-compaction-at`,
  `x-compactions-remaining`, `x-grok-doom-loop-check`,
  `x-grok-exact-repetition-check`); Omni still omits those on single-shot
  requests

`/v1/models` emits only canonical upstream ids. Omni accepts aliases inbound only.

### Custom endpoint override

`OMNI_GROK_BASE_URL` (and legacy `GROK_MODELS_BASE_URL`) switches to an
OpenAI-compatible `/chat/completions` gateway with custom auth only. That is an
operator override, not a second catalog mode.

### Thinking / reasoning_effort

- Responses (default): nested `"reasoning": { "effort": "..." }`
- Custom chat: top-level `"reasoning_effort": "low"|"medium"|"high"`
- On `grok-4.5`, CLI default is `high` and reasoning cannot be disabled.
- **Omit contract (issue #18):** client omits effort (or sends `"none"`) → Omni
  omits the upstream field → provider/model default applies (often `high`). No
  force-floor and no invented disable. Explicit set values map or fail loud
  (issue #20).
