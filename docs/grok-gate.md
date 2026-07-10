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
- **Default mode is file-only, no env-var key.** There is no `XAI_API_KEY` key
  path for the normal xAI endpoint. Just as the Claude provider reads the
  Claude CLI's own `~/.claude/.credentials.json`, the Grok provider reads the
  Grok CLI's own login file, so an existing `grok` login Just Works.
- **Source precedence (highest first):**
  1. `$XAI_CREDENTIALS_PATH` (if set) → use exactly that file. A failure here is loud (we do not silently fall through a deliberately-pointed path).
  2. `~/.xai/.credentials.json` → a simple static key you created on purpose. A usable static key wins over the CLI login; if the file is present but has no usable key, Omni falls through to the CLI login.
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
- **OIDC refresh (on by default).** Omni may refresh a near-expired OIDC access token via `POST https://auth.x.ai/oauth2/token` (form-urlencoded, public client) and **atomically write back** the rotated `refresh_token` to the same file. RTs rotate with a short grace window then revoke — the rotated token must be persisted. Disable with `--no-oauth-refresh`, `OMNI_NO_OAUTH_REFRESH=1`, or `OMNI_OAUTH_REFRESH=0` (or `false`/`off`/`no`) to only re-read (CLI may still refresh in the background). Static API keys (`~/.xai/.credentials.json`) are never refreshed. The parsed `expires_at` still drives a non-fatal expiry warning when refresh is off or fails.

The loader lives in `provider-grok::credentials::GrokCredentials` and keeps the same "fresh per request" contract, plus a real expiry check for OIDC tokens.

In `provider-grok` the key is obtained inside the `send` / `send_stream` path (fresh resolution through the source chain) and used only for that request's `Authorization: Bearer ...` header. The `GrokProvider` struct no longer holds a long-lived key (except in test helpers).

This is the "same technique for omni and for grok" as requested: read the CLI's own login file, fresh, every request.

## Custom Endpoint Override

`OMNI_GROK_BASE_URL` is the forced Omni override for Grok and wins over
`GROK_MODELS_BASE_URL`. In that mode, custom provider auth owns the request:

- `OMNI_GROK_AUTH_TOKEN` sends `Authorization: Bearer ...`.
- If that is empty, `OMNI_GROK_API_KEY` sends `Authorization: Bearer ...`.
- `OMNI_GROK_CUSTOM_HEADERS` accepts one `Name: value` header per line.
- No default xAI credential file or `XAI_API_KEY` is read for this endpoint.

The legacy `GROK_MODELS_BASE_URL` path remains supported:

- `XAI_API_KEY` sends `Authorization: Bearer ...`.
- If `XAI_API_KEY` is absent, Omni sends no Authorization header.
- `$XAI_CREDENTIALS_PATH`, `~/.xai/.credentials.json`, and `~/.grok/auth.json`
  are not read for the custom endpoint.

This exception mirrors Grok CLI custom-model behavior, where explicit model
configuration overrides the ambient xAI login and prevents signed-in xAI tokens
from leaking to arbitrary hosts.

## Relationship to the Rest of Omni
- The "grok gate" logic (headers + fresh creds) lives inside the Grok provider. It does not leak into `omni-common` policy or the Claude path.
- Replacements (from omni-common) are still applied at the prompt/response boundaries around the gate, exactly as for Claude.
- The Omni server can enable "grok" (and/or "claude") via `--providers` /
  `OMNI_PROVIDERS` and route by canonical model id, documented alias, or
  optional provider prefix. No Grok wire-gate knowledge is required at the
  server layer.

## References
- Official xAI docs (quickstart, /v1/chat/completions, /v1/responses, tools).
- Prior art in the investigation (how other proxies handle optional X-Title/Referer for xAI).
- The actual loader implementation in `crates/provider-grok/src/credentials.rs`.
- `provider-grok` usage of the loader (fresh read inside send).
- `docs/providers/grok/CAPTURE.md` for the Grok capture/update workflow.

If xAI ever publishes an official credentials file format or additional mandatory headers for their "gate", this document and the provider-owned loader can be updated in one place while keeping the same "look for the file, read fresh" contract.
