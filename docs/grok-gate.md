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
- `X-Title`: Human-readable client name (e.g. "omni-grok", "my-app"). Seen in proxy examples (TypingMind, etc.) for usage attribution and to avoid generic blocks.
- `HTTP-Referer`: Origin URL (e.g. "https://example.com"). Helps with tracking and some rate-limit policies in shared/proxy scenarios.

These are **not** enforced on the raw wire for direct calls (the API happily accepts plain `curl` with only Bearer), but they are the practical "headers needed for Grok in order to pass their gates" when using proxies, IDEs, or enterprise routing layers.

In the body (not header):
- `service_tier`: "default" | "priority" — affects scheduling priority and billing for paid/enterprise users.
- For built-in tools on the Responses surface or via special tool objects: specific JSON shapes that trigger server-side execution (returns citations + detailed `usage.server_side_tool_usage_details`).

## Credential Handling — Same Technique as Claude Code Provider
We deliberately copied the exact pattern from the original Claude Code Provider (see `reference-src-claude/src/upstream/credentials.rs` and the "Locked design" comments):

- **Never cache** the key in memory for the lifetime of the process.
- **Always re-read fresh per request** (via `load_fresh` / `load_fresh_async`).
- This lets any background refresh (user running a helper that writes the file, rotating a key in the console, etc.) be picked up without restarting the server.
- **Default path logic with env override**:
  - `$XAI_CREDENTIALS_PATH` (if set) → use exactly that file.
  - Otherwise `~/.xai/.credentials.json` (or a reasonable fallback).
- Simple on-disk JSON shape (documented so users/tools can create it):
  ```json
  {
    "apiKey": "xai-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  }
  ```
  We also accept a top-level `"xaiApiKey"` for compatibility with third-party tools that store xAI material.

The loader lives in `omni-common::credentials::GrokCredentials` (modeled 1:1 on CCP's `Credentials` — same method names, same "fresh per request" contract, same expiry-check hook for future OAuth tokens).

In `provider-grok` the key is obtained inside the `send` path (fresh load, with env fallback for compatibility) and used only for that request's `Authorization: Bearer ...` header. The `GrokProvider` struct no longer holds a long-lived key (except in test helpers).

This is the "same technique for omni and for grok" as requested.

## Relationship to the Rest of Omni
- The "grok gate" logic (headers + fresh creds) lives only inside the Grok provider (or the thin common credentials loader it uses). It does not leak into `omni-common` policy or the Claude path.
- Replacements (from omni-common) are still applied at the prompt/response boundaries around the gate, exactly as for Claude.
- The light Omni wrapper can enable "grok" (and/or "claude") via `--providers` / `OMNI_PROVIDERS` and route by model prefix. No special Grok knowledge is required at the aggregator layer.

## References
- Official xAI docs (quickstart, /v1/chat/completions, /v1/responses, tools).
- Prior art in the investigation (how other proxies handle optional X-Title/Referer for xAI).
- The actual loader implementation in `crates/omni-common/src/credentials.rs`.
- `provider-grok` usage of the loader (fresh read inside send).

If xAI ever publishes an official credentials file format or additional mandatory headers for their "gate", this document + the common loader can be updated in one place while keeping the same "look for the file, read fresh" contract that the original Claude Code Provider used so successfully.
