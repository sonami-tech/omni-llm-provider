# Design: Monorepo with Separate Binaries + Shared Components

> **Status update (as built).** All three binaries are now full proxies, not a
> deferred idea: `omni` (the aggregator that routes to either backend by model
> prefix), `omni-claude`, and `omni-grok`. Each serves OpenAI Chat Completions
> (non-stream JSON + streaming SSE), `/v1/models`, `/stats`, and `/health` over
> the shared `omni-common::http` layer, with `LlmProvider::send_stream` wired
> end to end. The "native `/v1/messages`" and OpenAI Responses surfaces are not
> implemented. The original decision and rationale below are kept as a record.

## Decision (original step)
We started by building a monorepo that produces **separate binaries**:
- `omni-claude` — the high-fidelity Claude Code / Anthropic Max provider (preserves the "core invariant" of byte-exact Claude Code wire fingerprint for the subscription OAuth gate).
- `omni-grok` — the Grok / xAI provider (standard OpenAI-compatible, much lighter).

Shared components live in library crates so we avoid duplication without compromising isolation.

The "Omni wrapper/aggregator" (single binary that can speak to multiple providers) was deferred at first; it has since been built as the `omni` binary and is the primary entry point.

## Crate Layout
- `crates/omni-common` — Truly shared, provider-agnostic pieces: replacements engine, redb stats, auth middleware, conversation logging, session derivation, base error types, time utilities, etc.
- `crates/omni-core` — Canonical internal types (messages, tools, usage, reasoning config, etc.) + core traits (`LlmProvider`) + any routing/registry primitives. This enables future "connect anything to anything".
- `crates/provider-claude` — **All** Claude-specific logic that must stay isolated:
  - Fingerprint profiles, cch checksum, billing headers, system preamble injection.
  - `~/.claude/.credentials.json` handling + 401 refresh.
  - Translation to/from Anthropic Messages.
  - Wire defaults, identity blocks, native `/v1/messages` surface, re-baselining support.
  - This crate + the `omni-claude` binary own the invariant completely.
- `crates/provider-grok` — xAI-specific (model names, built-in tools, Responses vs chat, any xAI extensions). Thin adapter over the OpenAI-compatible API.
- `crates/bin/omni-claude` — Produces the `omni-claude` executable. Depends on the shared libs + provider-claude. Contains the server setup, routing (OAI + native surfaces), config, etc.
- `crates/bin/omni-grok` — Produces the `omni-grok` executable. Depends on shared + provider-grok.

## Why separate binaries (not one omni binary yet)
- Protects the Claude fingerprint invariant (no Grok code path can accidentally affect cch serialization, header construction, or identity blocks).
- Matches the original project's philosophy (narrow, exact, surgical).
- Each binary can have its own defaults, validation (Claude creds at startup only for omni-claude), and release cadence.
- Still gets massive reuse via the library crates.
- Users who only need one side run only that binary.
- Easy to add `omni-codex` etc. later as another `bin/` + `provider-*/` pair.

## Building
```bash
cargo build -p omni-claude   # produces target/debug/omni-claude
cargo build -p omni-grok     # produces target/debug/omni-grok
```

## Future
- Full port of the original claude-code-provider logic into `provider-claude` (using `reference-src-claude/` as source).
- Real implementation in `provider-grok` (direct to `api.x.ai/v1`).
- Richer use of `omni-core` canonical types so frontends and providers compose cleanly.
- Later: an optional aggregator binary or a thin router that lets one process expose multiple providers.

See INVESTIGATION.md for the full research (cliproxyapi, LiteLLM, original CCP data flow, etc.) that led to this structure.
