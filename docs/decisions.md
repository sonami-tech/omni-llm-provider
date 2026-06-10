# Key Decisions and Findings for Omni LLM Provider

## Architecture: Separate Binaries + Shared Components (Monorepo)
- Decision: Multiple focused binaries (omni-claude, omni-grok) + optional light wrapper/aggregator (omni), sharing omni-common and omni-core.
- Rationale (from investigation + subagent work):
  - Protects Claude's strict "core invariant" (byte-exact fingerprint, cch checksum in billing header, specific betas, preamble, per-profile wire defaults, etc.). All of this lives *only* in provider-claude (and its isolated modules: fingerprint.rs, etc.). No leakage into common or grok path.
  - Grok is "light" (standard OpenAI-compatible, no fingerprint gate). Real HTTP to api.x.ai/v1, full mapper to/from Canonical* (tools, reasoning_effort, provider_extras for search_parameters/service_tier, etc.).
  - Shared: replacements (prompt/response scopes + hooks in both providers), stats (redb + TokenUsage + Active guards, per-model/per-key), auth (bearer middleware), session derivation, error shapes, canonical types/traits (LlmProvider), logging.
  - Wrapper (light omni): routes by model prefix (grok:foo, claude:bar) or --providers/OMNI_PROVIDERS config. Unified OAI surfaces. Delegates to providers. Thin (no heavy logic).
- Why not single binary with runtime backends? Risk to Claude invariant (shared dispatch/config could drift cch/headers). Separate binaries keep claude pure (matches original CCP philosophy + CLAUDE.md rules: surgical, match conventions, fail loud on invariants). Monorepo still gives code reuse.
- "Omni wrapper" interesting for users who want one process, but start with focused binaries + optional aggregator.

## Grok-Specific (Main Focus)
- No special "fingerprint gates" like Claude Code (no cch, no x-anthropic-billing-header, no mandatory preamble, no per-version profiles, no stainless versions).
- Required headers: Authorization: Bearer $XAI_API_KEY, Content-Type: application/json.
- Optional/recommended for proxies/clients (to pass tracking/rate-limit "gates" or identify in logs): X-Title, HTTP-Referer (seen in some SDK/proxy examples like TypingMind).
- service_tier (body): "default" | "priority" (enterprise/priority access).
- Built-in tools (web_search, x_search, code_execution, etc.): special tool objects or search_parameters. They execute server-side and return citations + detailed usage (num_sources_used, server_side_tool_usage_details).
- reasoning_effort: top-level in chat.completions ("low"|"medium"|"high" etc.); nested in Responses.
- Primary: /v1/chat/completions (OAI compat). /v1/responses for stateful/agentic (previous_response_id, compact, etc.).
- Findings from research (official docs + SDKs 2026): Extremely straightforward compared to Claude. OpenAI SDK works with base_url swap. No re-baselining ceremony. Focus of implementation: real reqwest calls + canonical mapping + Replacements hooks + tools/reasoning.
- See grok-requirements.md for full.

## Claude Incorporation
- Full port of critical invariant logic from reference-src-claude (fingerprint profiles + cch patch + headers + betas + preamble + identity injection + wire defaults + models catalogs + credentials fresh-read + 401 retry + translate to Anth Messages).
- All isolated in provider-claude. Tests include original pins (cch snapshots, per-model betas, suffix vectors, etc.).
- Uses shared: Replacements (before identity so suffix/cch see post-repl text), Canonical* for requests/responses, LlmProvider trait.
- Bin omni-claude can wire it (preserves original behavior for Claude Max users).

## Tests (Fully Encompassing Both Backends)
- omni-common: replacements (all scopes, hooks, parity, duplicates), stats (records, active guards, snapshots, per-key/model), auth, error, session.
- omni-core: canonical serde roundtrips, trait parity (mocks for grok + claude).
- provider-grok: mappers (tools, reasoning, extras, repl hooks), send (mocked bad-port + real-if-XAI_API_KEY present).
- provider-claude: ported fingerprint/models/translate/upstream + new (ctors, shared repl, full send path exercising cch/headers/identity).
- Wrapper (in bin mains): routing by name/prefix/config to *both*, unified surfaces (OAI + health), shared usage, error cases (unknown provider).
- Mocks: bad ports for grok upstream, no-creds/auth-err or live for claude (depends on ~/.claude creds in env). Real-if-key for grok.
- cargo test --workspace --lib: passes (providers have their suites; wrapper routing tests cover both backends; shared exercised uniformly).
- No python integration yet (future; would use original CCP-style with real creds for claude, xai key for grok).

## Other Decisions
- Canonical in omni-core: Text-focused for now (extendable to blocks for vision/tool/result/reasoning per research on cliproxy/LiteLLM/xAI Responses).
- Replacements: prompt (outbound system/msgs/tools/schemas) before provider mutation; response (inbound text/tool names/args) after. Hooks in both providers.
- Future: full streaming in wrapper, Responses support in grok, more providers (codex), python SDK, real multi-account like cliproxyapi research.
- Expendable area: everything here can be iterated without affecting original claude-code-provider.

See INVESTIGATION.md (prior art: cliproxyapi multi-OAuth + translators + executors owning quirks; LiteLLM unified + per-provider transforms + shared gateway), DESIGN.md, grok-requirements.md, and source for details.
