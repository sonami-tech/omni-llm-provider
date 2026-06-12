# Grok Gate (xAI Access Controls, Headers, and Credential Handling)

## Overview
Unlike the Claude Code Provider's heavy "core invariant" (byte-exact wire fingerprint, `x-anthropic-billing-header` cch checksum, specific betas, preamble injection, per-profile wire defaults, etc. that must be reproduced to pass Anthropic's subscription OAuth gate), the Grok/xAI "gate" is lightweight and standard.

xAI does **not** require:
- Any Claude-Code-style fingerprint emulation
- cch checksums
- Mandatory system preambles or identity blocks
- Per-release profiles or stainless version pinning
- Special TLS client hellos

The primary "gate" is simply a valid API key. Additional controls exist for:
- Priority/enterprise access (`service_tier`)
- Built-in server-side tools (web_search, x_search, code_execution, etc.)
- Rate limiting and abuse detection (standard for any public API)
- Client identification in proxies or enterprise setups (optional headers)

## Required / Recommended Headers
From official docs and observed SDK/proxy behavior (2026):

**Always required for basic access:**
- `Authorization: Bearer $XAI_API_KEY`
- `Content-Type: application/json`

**Optional but useful for "passing gates" in certain environments:**
- `X-Title`: Human-readable client name (e.g. "omni", "my-app"). Seen in proxy examples (TypingMind, etc.) for usage attribution and to avoid generic blocks.
- `HTTP-Referer`: Origin URL (e.g. "https://example.com"). Helps with tracking and some rate-limit policies in shared/proxy scenarios.

These are **not** enforced on the raw wire for direct calls (the API happily accepts plain `curl` with only Bearer), but they are the practical "headers needed for Grok in order to pass their gates" when using proxies, IDEs, or enterprise routing layers.

In the body (not header):
- `service_tier`: "default" | "priority" - affects scheduling priority and billing for paid/enterprise users.
- For built-in tools on the Responses surface or via special tool objects: specific JSON shapes that trigger server-side execution (returns citations + detailed `usage.server_side_tool_usage_details`).

## Credential Handling - Same Fresh-Read Technique as Claude
Grok uses the same fresh credential read contract as the Claude provider:

- **Never cache** the key in memory for the lifetime of the process.
- **Always re-read fresh per request** (via `load_resolved_async`, layered on `load_fresh` / `load_fresh_async`).
- This lets any background refresh (the Grok CLI refreshing its login, rotating a console key, etc.) be picked up without restarting the server.
- **File-only, no env-var key.** There is no `XAI_API_KEY`-as-key path. Just as the Claude provider reads the Claude CLI's own `~/.claude/.credentials.json`, the Grok provider reads the Grok CLI's own login file, so an existing `grok` login Just Works.
- **Source precedence (highest first):**
  1. `$XAI_CREDENTIALS_PATH` (if set) → use exactly that file. A failure here is loud (we do not silently fall through a deliberately-pointed path).
  2. `~/.xai/.credentials.json` → a simple static key you created on purpose. Explicit beats ambient, so this wins over the CLI login.
  3. `~/.grok/auth.json` → the Grok CLI's own OIDC login file (auto-detected).
- **Two on-disk shapes** (either may sit behind `$XAI_CREDENTIALS_PATH`):
  - Static key (`~/.xai/.credentials.json`):
    ```json
    { "apiKey": "xai-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" }
    ```
    (a top-level `"xaiApiKey"` alias is also accepted.)
  - Grok CLI OIDC (`~/.grok/auth.json`), keyed by `https://auth.x.ai::<client-id>`:
    ```json
    { "https://auth.x.ai::<id>": { "key": "<JWT>", "auth_mode": "oidc",
        "refresh_token": "...", "expires_at": "2026-06-10T22:20:22.000000Z" } }
    ```
    The `key` JWT is a Bearer that authenticates `api.x.ai/v1` directly.
- **OIDC tokens are read READ-ONLY.** We never write `~/.grok/auth.json` or consume its single-use `refresh_token`. The parsed `expires_at` drives a non-fatal expiry warning; on a genuinely dead token the upstream 401s and the user re-runs the Grok CLI login. Static keys carry no expiry and never warn.

The loader lives in `omni-common::credentials::GrokCredentials` and keeps the same "fresh per request" contract, plus a real expiry check for OIDC tokens.

In `provider-grok` the key is obtained inside the `send` / `send_stream` path (fresh resolution through the source chain) and used only for that request's `Authorization: Bearer ...` header. The `GrokProvider` struct no longer holds a long-lived key (except in test helpers).

This is the "same technique for omni and for grok" as requested: read the CLI's own login file, fresh, every request.

## Relationship to the Rest of Omni
- The "grok gate" logic (headers + fresh creds) lives only inside the Grok provider (or the thin common credentials loader it uses). It does not leak into `omni-common` policy or the Claude path.
- Replacements (from omni-common) are still applied at the prompt/response boundaries around the gate, exactly as for Claude.
- The Omni server can enable "grok" (and/or "claude") via `--providers` / `OMNI_PROVIDERS` and route by model prefix. No special Grok knowledge is required at the server layer.

## References
- Official xAI docs (quickstart, /v1/chat/completions, /v1/responses, tools).
- Prior art in the investigation (how other proxies handle optional X-Title/Referer for xAI).
- The actual loader implementation in `crates/omni-common/src/credentials.rs`.
- `provider-grok` usage of the loader (fresh read inside send).
- `docs/providers/grok/CAPTURE.md` for the Grok capture/update workflow.

If xAI ever publishes an official credentials file format or additional mandatory headers for their "gate", this document and the common loader can be updated in one place while keeping the same "look for the file, read fresh" contract.
