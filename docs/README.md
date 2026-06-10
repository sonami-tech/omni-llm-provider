# omni-llm-provider (monorepo)

Monorepo for pluggable LLM providers and frontends.

**Current focus (initial step):** Separate binaries with shared components.

- `omni-claude` — High-fidelity binary for Claude Max / Claude Code (preserves the exact wire fingerprint invariant for Anthropic's subscription OAuth gate).
- `omni-grok` — Binary for xAI Grok (standard OpenAI-compatible backend).

## Structure

- `crates/omni-common` — Shared cross-cutting concerns (auth, stats/redb, replacements/TOML, logging, session derivation, base error shapes, etc.).
- `crates/omni-core` — Canonical types + core traits (LlmProvider, etc.) for "connect anything to anything".
- `crates/provider-claude` — All Claude-specific logic (fingerprint, credentials, Anthropic Messages translation, identity injection, cch, profiles). Isolated to protect the invariant.
- `crates/provider-grok` — Grok/xAI specific logic (light adapter).
- `crates/bin/omni-claude` — Produces the `omni-claude` binary.
- `crates/bin/omni-grok` — Produces the `omni-grok` binary.

## Building the binaries

```bash
cargo build -p omni-claude
cargo build -p omni-grok
```

This produces separate executables while maximizing code reuse in the shared crates.

The "Omni wrapper/aggregator" idea is noted for later (a single binary that can activate multiple providers).

See DESIGN.md and INVESTIGATION.md for architecture rationale and prior art (cliproxyapi, LiteLLM, original claude-code-provider).

Reference copy of the original CCP source lives in `reference-src-claude/` for porting the Claude logic.
