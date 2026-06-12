# Omni LLM Provider — Monorepo/Shared Components Investigation

**Staging area:** `~/omni-llm-provider`  
**Date of this investigation pass:** 2026-06 (based on context)  
**Goal of this document:** Deeply analyze the existing `claude-code-provider` (CCP) to test the hypothesis that "it's really just the back-end provider that differs" and evaluate a monorepo + shared components + multiple front-end formats + multiple back-end providers architecture ("connect anything to anything").

This document is written *as if we are seriously pursuing the monorepo/shared approach* for further exploration (per user request), while remaining ruthlessly honest about boundaries, risks to the Claude "core invariant", and complexity.

> Historical note: this investigation predates the single-binary layout. Current
> builds ship one server binary, `omni`, with provider implementation still kept
> in separate crates.

---

## 1. Executive Summary of Findings (Current Pass)

**User hypothesis under test:** The application is simple in data flow. The only real difference between providers (Claude Code vs Grok vs future Codex) is the "back-end". Front-end formats (OpenAI compat, native Anthropic, etc.), output/logging, replacement engine, stats, auth, and "three front-end code methods" are heavily overlapping and can be shared. A monorepo with proper boundaries makes sense for an "omni" system.

**Findings from deep dive into CCP:**

- **There is substantial overlap in the outer shell.** Auth (bearer keys), stats (redb per-model tracking + active requests), replacements (TOML engine applied outbound/inbound), conversation logging, basic server scaffolding (axum, CORS, body limits, health, /stats), config skeleton, session ID derivation, and the *idea* of an OpenAI-compatible surface are reusable.

- **"Just the back-end" understates the Claude reality.** The Claude "back-end" is not a simple API call. It is a high-fidelity wire emulator whose job is to produce *byte-exact* traffic for Anthropic's subscription OAuth gate. This touches:
  - Per-request fresh read + 401 retry of a specific `~/.claude/.credentials.json` shape.
  - Heavy, stateful, bidirectional translation (OAI Chat <-> Anthropic Messages) because the shapes differ significantly (system as special blocks that carry the billing/preamble identity, content blocks vs parts, tool_use vs tool_calls + input JSON, separate thinking blocks, output_config.effort, cache_control, usage fields with cache tokens, stop reasons).
  - **Outbound mutation for the gate**: `prepend_claude_code_identity` (injects `x-anthropic-billing-header` block + canonical preamble as the *first* system blocks, stripping stale ones), `apply_profile_wire_defaults` (overrides max_tokens/temp/effort per model/profile), tool name PascalCasing via replacements (gate fingerprints tool surfaces).
  - **Byte-sensitive finalization**: `finalize_body_json` + cch logic in `fingerprint.rs` that searches the *serialized* JSON for the placeholder inside the system block and patches a 5-hex checksum (xxh64 over body bytes). Serialization order matters.
  - Per-Claude-Code-version "profiles" (7+ pinned releases) with betas, stainless versions, wire defaults, model catalogs, cch algorithm. Re-baselining process is first-class (live capture is authoritative; tools/fingerprint/ + vectors + drift checker).
  - The native `/v1/messages` + `/v1/messages/count_tokens` surface (bypasses OAI<->Anth translate on both ends but *still* runs the full fingerprint/identity/replacement path so the gate is satisfied and real Claude Code / Anthropic SDKs can point at CCP).

- **The "front-end" is not uniform.** 
  - OpenAI Chat Completions surface (primary for most users).
  - Native Anthropic Messages surface (added with multiple adversarial review gates precisely to keep the *single* fingerprint path intact while allowing raw passthrough responses/SSE).
  - `/v1/models` has dual shapes (OAI vs Anthropic) based on headers.
  - Error shaping, model resolution, and stats keying are influenced by the active "profile" and provider.

- **Data flow is linear but the middle is provider+format specific.** See the project's own `reference-architecture.md` (copied here). The "OAuth Gate Layer" and the two translation steps are where most complexity and Claude-specific invariants live. Replacements are applied at the boundaries (outbound before gate, inbound after reverse translate).

- **Prior art in research/**: `research/cliproxyapi/` (Go) is a production example of a much more ambitious "multi" system (many CLI OAuth providers: Claude Code, Codex, Gemini, Qwen, etc.; multiple auth mechanisms; translator pipelines; model registry; runtime executors; watcher for config/hot reload; separate SDK). It shows that "front-end format" <-> internal <-> "back-end provider (with its auth + quirks)" is a viable but non-trivial pattern. It has per-provider translators and careful handling of differences.

**Conclusion on monorepo viability (this pass):** 

A monorepo with two (or N) binaries + narrow shared components **is viable** and is a better middle ground than a single binary with runtime `if provider`. It allows the Claude binary to keep its exactness guarantees local and uncontested.

However, the overlap is **not** "mostly everything except the backend call." The Claude backend *is* the complex translation + mutation + auth + profile system. A clean "omni" design requires an explicit **canonical internal representation** + adapters on both sides (front-end formats and back-end providers). The shared surface is real but smaller and more "infrastructure + cross-cutting concerns" than "the whole request pipeline."

If we enforce **very strict boundaries** from day one (no shared AppState, no shared request pipeline, no leaking of "fingerprint profile" or "Anthropic Messages types" into common), it can work. The risk is gradual erosion of those boundaries over time (the natural DRY instinct in a monorepo).

The user's vision ("back-end with multiple providers + front-end supporting variety of formats so you can connect anything to anything") is achievable but requires treating translation/normalization as a first-class concern, not an afterthought.

---

## 2. Precise Data Flow in Current CCP (Deep Map)

From code + `reference-architecture.md` + `anthropic-compat-design.md` + source:

### Common / Outer Layers (high reuse potential)
- **Auth** (`auth.rs`): Simple Bearer key middleware. Attaches `ApiKeyId`. Same for all surfaces.
- **Stats** (`stats.rs`): redb-backed. `record_request(model, key)`, `record_response` (with token usage + TTFT + duration), `record_error`. Active request guards. Snapshot for /stats. Keyed by canonical model name.
- **Replacements** (`replacements.rs`): TOML-driven. `apply_prompt` (outbound: system, messages, tool names/descriptions/schemas, tool results) and `apply_response` (inbound: assistant text, tool names, tool args). Special streaming state machines that buffer for correctness (especially tool args and text under rules). Applied in both OAI and native paths.
- **Conversation logging** (`conversation_log.rs`): Optional full raw I/O or per-session files. Used at inbound, pre-upstream, post-response.
- **Config** (parts of `config.rs`): Port, host, api-keys(-file), replace-rules, log-*, data-dir, no-auth, verbose. Some flags are Claude-specific (`no_preamble`, `fingerprint_profile`).
- **Server basics** (`main.rs`): Axum router, CORS (permissive), body limit, graceful shutdown, LAN IP detection, tracing setup.
- **Error shaping** (`error.rs`): `AppError` variants map to OpenAI error envelope for the OAI surface. Native surface has separate Anthropic error renderer.
- **Session** (`session.rs`): Stable ID derivation from header or `user` field or key.

### Request Surfaces ("Front-ends" — some shared, some not)
1. **OpenAI Chat Completions** (`routes/completions.rs` + `completions_v2.rs`):
   - Parse `ChatCompletionRequest`.
   - Resolve model via active `fingerprint_profile`.
   - Build `MessagesRequest` (heavy OAI->Anth in `translate/build.rs` + `messages.rs` + `tool_translate.rs`).
   - Apply outbound replacements.
   - **Claude Gate layer**: `prepend_claude_code_identity`, `apply_profile_wire_defaults`.
   - Finalize for cch.
   - Upstream send (stream or not).
   - Reverse: `build_oai_response` or `OaiStreamConverter`.
   - Inbound replacements.
   - Stats on entry + exit.

2. **Native Anthropic Messages** (`routes/messages.rs`):
   - Parse `ClientMessagesRequest` (closed allowlist — important for fingerprint safety).
   - `reconcile_client_request` (still does model resolve + replacements + identity strip+inject + wire defaults).
   - Same upstream path (raw for SSE passthrough).
   - Returns raw Anthropic JSON/SSE (with inbound replacements applied to leaves).
   - Special `count_tokens` handling (counts client body as-sent, strips sampling fields because Anth rejects them on that endpoint).
   - This surface was deliberately engineered (multiple gates with Grok/Codex adversarial review) to keep the *one true fingerprint path* while allowing faithful passthrough.

3. **Models + other** (`routes/models.rs`): Dual shape based on `anthropic-version` header. Built from the profile's catalog.

### The "Provider" / Middle / Gate (heavily Claude-unique)
- `upstream/fingerprint.rs`: The heart of the "core invariant". `FingerprintProfile`, all the pinned CC versions, beta strings (with model overrides), `WireDefaults`, billing scheme, `finalize_body_bytes` (cch patch), `build_headers`, `billing_header_text`, `CLAUDE_CODE_SYSTEM_PREAMBLE`, `is_claude_code_billing_header`.
- `upstream/credentials.rs`: Claude-specific file format + always-fresh + expiry check + 401 refresh.
- `upstream/client.rs`: Hardcoded Anthropic URLs. Retry logic tied to 401 credential refresh. `send_messages_json/stream/raw`, `count_tokens`.
- `translate/`: The bidirectional adapter. `anthropic.rs` wire types (system as Text/Blocks, ContentBlock variants including Thinking/ToolUse/ToolResult/Image, Tool, ToolChoice, etc.). Build logic that handles reasoning_effort -> thinking, parallel_tool_calls -> disable_parallel, etc. Reverse conversion that handles cache tokens, signatures on thinking, etc.
- `routes/completions_v2.rs`: `prepend_claude_code_identity`, `apply_profile_wire_defaults`, the `StreamReplState` that interacts with replacements during Anth->OAI streaming.
- Model catalogs and resolution live in `models.rs` but are driven by the active profile.
- Replacements interact with the gate (tool names often need masking for the OAuth gate fingerprint).

### Cross-cutting that must be placed carefully in a shared design
- Replacements are applied at specific *semantic* points (prompt surface vs response surface). In a multi-frontend world this needs to be tied to "before leaving the frontend" or "in canonical form" or "per-provider".
- Stats recording must happen for the *actual* work done (on the provider side) but surfaced in a way that makes sense for the frontend the client used.
- Logging wants raw inbound + the wire that actually hit the provider + the final client response.

---

## 3. Proposed Architecture for the "Omni" Monorepo Vision

**High-level goal (user-stated):** Back-end = pluggable providers (ClaudeCode with its gate, Grok/xAI, future Codex, possibly direct OpenAI, etc.). Front-end = pluggable formats (OpenAI Chat Completions, Anthropic Messages native/compat, perhaps Gemini, Responses, etc.). Cross-cutting shared: auth, stats, replacements, logging, config, observability. "Connect anything to anything."

**Recommended crate boundaries (strict to protect invariants):**

```
omni-llm-provider/                  # monorepo root (workspace)
├── Cargo.toml                      # [workspace] members = ["crates/*"]
├── crates/
│   ├── omni-common/                # Highest reuse. Pure infrastructure.
│   │   ├── auth/
│   │   ├── stats/                  # redb layer + guards + snapshot types
│   │   ├── replacements/           # the engine + rule application (prompt/response scopes)
│   │   ├── conversation_log/
│   │   ├── config/                 # base flags + env var loading (provider-specific flags live in their crates)
│   │   ├── error/                  # core error types + OpenAI-shaped renderer (Anthropic renderer can live with the Anth frontend)
│   │   ├── session/
│   │   └── types/                  # small shared things (ApiKeyId, etc.)
│   │
│   ├── omni-core/                  # The "lingua franca" — critical for "anything to anything"
│   │   ├── canonical.rs            # Internal representation:
│   │   │   - CanonicalMessage (role + content blocks that are richer than OAI but not tied to any one provider)
│   │   │   - CanonicalTool, CanonicalToolChoice
│   │   │   - ThinkingConfig / ReasoningEffort
│   │   │   - Sampling params, etc.
│   │   │   This is where we normalize.
│   │   ├── pipeline.rs             # RequestContext, maybe a simple Pipeline that chains frontend -> canonical -> provider
│   │   └── traits.rs               # 
│   │       pub trait LlmProvider: Send + Sync {
│   │           async fn send(&self, req: CanonicalRequest, ctx: &RequestContext) -> Result<CanonicalResponse>;
│   │           // streaming variant
│   │       }
│   │       pub trait FrontendFormat { ... handle incoming, produce canonical or call provider directly for passthrough cases }
│   │
│   ├── frontends/
│   │   ├── openai-chat/            # The current OAI /v1/chat/completions surface + translation to canonical
│   │   ├── anthropic-messages/     # Native /v1/messages + count_tokens. For Claude provider this can short-circuit some work but still must run gate layer.
│   │   └── (future: gemini, responses, etc.)
│   │
│   ├── providers/
│   │   ├── claude-code/            # Almost the entire current CCP "guts" moved here:
│   │   │   - fingerprint.rs (the 1000 LOC invariant)
│   │   │   - credentials.rs (Claude-specific)
│   │   │   - client.rs (Anthropic URLs + headers + cch finalize)
│   │   │   - translate/ (OAI<->Anth or canonical<->Anth Messages)
│   │   │   - identity.rs (prepend/strip)
│   │   │   - wire_defaults.rs + profile handling
│   │   │   - Its own main.rs / binary name "claude-code-provider" or "omni-claude"
│   │   │   - Keeps the re-baselining tools close (or reference from root)
│   │   │   - AppState can be local to this provider crate
│   │   │
│   │   ├── grok/                     # Much smaller. xAI is already OpenAI-compatible.
│   │   │   - Simple key auth (env/file)
│   │   │   - Thin or zero translation (or light model name mapping)
│   │   │   - Direct upstream to api.x.ai
│   │   │   - Its own binary
│   │   │
│   │   └── codex/ (future)         # Will likely have its own OAuth / CLI quirks, similar in spirit to claude-code but different details
│   │
│   └── binaries/                   # Or just put mains inside the provider crates
│       └── omni-server/            # Optional thin aggregator binary that can load multiple providers + frontends at runtime (if we ever want one process serving many)
│
├── docs/
├── research/                       # cliproxyapi, other proxies, etc. (already copied some)
├── tools/                          # fingerprint rebaseline stays under claude provider or top-level with clear ownership
└── tests/                          # Integration tests will be per-provider mostly
```

**Key design principles for boundaries (to make monorepo safe):**

- **Claude invariant is sacred and local.** The `fingerprint_profile`, cch patching, `finalize_body_json`, specific preamble, per-version betas, etc. live *only* inside `providers/claude-code`. Nothing in `omni-common` or `omni-core` or other providers may depend on or even name these concepts.
- **Canonical types in omni-core are the contract.** Frontends produce `Canonical*`. Providers consume `Canonical*` and produce `Canonical*` (or raw if passthrough). This isolates format differences.
- **Replacements scope is explicit.** Rules declare `scope = "prompt" | "response" | "tool"`. Application points are owned by the frontend/provider adapters, not a global "apply everywhere".
- **Stats and logging are cross-cutting but record at the right abstraction level** (canonical model id + provider id + frontend id).
- **No shared "AppState" that mixes concerns.** Each provider binary (or loaded provider) can have its own state that *uses* common pieces (Arc<Replacements>, Arc<Stats>, etc.).
- **Native passthrough surfaces are provider + format paired.** The Anthropic native surface only really makes sense paired with the Claude provider. It can live in the intersection of `frontends/anthropic-messages` and `providers/claude-code`.
- **Separate binaries by default.** `claude-code-provider` binary continues to exist and behave exactly as today (for fingerprint safety and user familiarity). Grok gets its own small binary. A future "omni" aggregator binary can be added later if users want one process with multiple providers.

**Data flow in the new world (example):**

OpenAI client -> `frontends/openai-chat` (parses, produces CanonicalRequest + applies outbound replacements scoped to prompt) -> `omni-core` pipeline -> `providers/grok` (or claude) adapter (for claude: more outbound replacements + identity + wire defaults + cch finalize) -> actual upstream call -> reverse adapter -> inbound replacements -> back to frontend -> emit OAI-shaped (or raw for native paths).

For Claude provider + Anthropic frontend: the frontend can pass the client body almost verbatim into the provider's "reconcile + gate" path (exactly as the current native surface does).

---

## 4. Next Steps for This Investigation (as if building)

1. **This doc + reference copy** (done in this pass).
2. Set up the workspace `Cargo.toml` + stub `omni-common` by extracting the 5-6 reusable modules (auth, stats, replacements, etc.) with minimal changes.
3. Define the `Canonical*` types in `omni-core` (start small: roles + text content + basic tool defs + sampling).
4. Port the current CCP's OpenAI surface + Claude provider as the first "full" vertical slice inside the monorepo (so we don't regress the existing behavior).
5. Add a minimal Grok provider (pass-through or very light) to prove the boundary.
6. Write a small "connect anything" example (e.g. Anthropic SDK client talking to Grok provider, or OAI client talking to Claude).
7. Document the enforcement mechanisms (e.g. a test that the claude provider crate does not re-export fingerprint types; clippy lints or just code review culture).

---

## 5. Open Questions & Risks to Track

- How much of the current "translate" layer moves into canonical adapters vs stays provider-specific?
- Streaming state machines for replacements + TTFT measurement are subtle — where do the buffers live?
- Model registry / alias resolution: per-provider? Or a global one with provider-specific overrides?
- Auth: some providers may want per-model or per-account routing (see cliproxyapi's multi-account round-robin).
- The re-baselining process and fingerprint vectors/tests must stay 100% owned by the Claude provider.
- Performance: extra translation layers for the "simple" paths (Grok) must not add unacceptable latency.
- Versioning of the canonical types (they will evolve as we add more providers/frontends).
- Whether a single process serving multiple providers (via the aggregator binary) is ever desirable vs separate processes + a thin router in front.

---

**This document will be iterated as we build the skeleton.** The goal of going down the monorepo path in this staging area is to discover the real boundaries by attempting to draw them, rather than arguing from afar.

Next concrete actions in this workspace: set up the Cargo workspace, extract the common pieces, define initial canonical types, and bring the Claude vertical up inside the new structure while keeping it isolated.
