# Repo Audit — omni-llm-provider

**Date:** 2026-06-09 · **Commit audited:** `6ed9ad0` (initial commit) · **Method:** full read of all 7 workspace crates + both stub binaries + all docs; fresh runs of `cargo build`, `cargo clippy --workspace --all-targets`, `cargo fmt --all --check`, `cargo test --workspace` (2 independent full-suite runs). `reference-src-claude/` treated as vendored port source (light review). Findings are labeled **(Fact)** — verified in code/output — or **(Judgment)**.

---

## 1. Executive Summary

**Overall health grade: C.** The domain core is genuinely strong — the Claude fingerprint port (`fingerprint.rs`, `upstream.rs`) is production-grade work with exemplary capture provenance, and the Grok provider is complete for non-streaming — but the repo's *shell* contradicts its *claims* at almost every seam. The two binaries the README advertises are empty stubs; the default test suite fails; clippy has a hard error; `cargo fmt --check` fails; the stats module is a no-op facade; and the project's own #1 invariant ("byte-for-byte wire fidelity including wire defaults") is only partially wired into the new send path.

**Top 3 risks:**
1. **Fingerprint fidelity gap in the live send path.** The canonical→Anthropic adapter never applies the profile's wire defaults: `output_config` is always `None`, `temperature` is only client-passthrough, and default `max_tokens` comes from the catalog (128k/64k) instead of the captured wire values (64k/32k). Real Claude Code 2.1.165 bodies carry all three. The gate accepts it *today*, but this is exactly the "close is failure" drift the invariant doc warns about.
2. **The suite is red and nothing enforces green.** `cargo test --workspace` fails reproducibly (port-collision in omni-grok bin tests), clippy fails on a denied lint, fmt fails, and there is no CI.
3. **Claims-vs-reality gap misleads the next reader.** Stub binaries presented as deliverables, a stats facade with empty method bodies, docs describing a directory layout that doesn't exist, and a unit test that silently requires live Anthropic credentials.

**Top 3 opportunities:**
1. Wire the profile wire-defaults into `translate.rs` (small, high-value, restores the stated invariant).
2. One day of hygiene (free ports in tests, clippy/fmt clean, tiny CI) flips the repo to verifiably green forever.
3. The provider logic already works — wiring it into the two stub binaries is mostly assembly, not new engineering.

---

## 2. Repo Map

**Purpose:** Multi-backend, OpenAI-compatible LLM proxy (Rust 2024 workspace). Lets OpenAI-format clients reach Anthropic Claude Max (via byte-exact "Claude Code" wire fingerprinting for the subscription OAuth gate) and xAI Grok, over a shared canonical type system.

**Stack:** axum 0.8 · tokio · reqwest/rustls (+http2 for Claude) · serde · clap · thiserror · ring (SHA-256 for billing suffix) · redb (declared, unused) · toml · chrono.

**Maturity:** Prototype / active port. One functional binary (`omni`); provider libraries largely complete; shell unfinished.

**Architecture (as actually wired today):**

```
OpenAI-format client
   │
   ├── omni         WORKING  /v1/chat/completions (non-stream), /v1/models, /health, auth
   ├── omni-claude  STUB     /health + / only (main.rs:13 TODO)
   └── omni-grok    STUB     /health + / only (main.rs:13 TODO)
         │
         ▼ LlmProvider trait (omni-core/src/traits.rs:5 — send() only, no streaming)
   provider-claude ─► api.anthropic.com/v1/messages   (fingerprint, cch, identity, 401-retry)
   provider-grok   ─► api.x.ai/v1/chat/completions    (OpenAI-compat, fresh-creds)
         │
         ▼
   omni-core    CanonicalRequest/Response (flat Text content only today)
   omni-common  auth ✓, GrokCredentials ✓, Replacements ✓ │ stats ✗(facade), conversation_log ✗(unwired), session ✗(unwired)
```

**Key directories:**
- `crates/omni-core` — canonical types + `LlmProvider` trait; small, clean, the right backbone.
- `crates/omni-common` — auth middleware, Grok creds, replacements engine (all real); stats (no-op facade), conversation_log (real but unwired), session/time_util (unwired).
- `crates/provider-claude` — fingerprint profiles ×7, cch (hand-rolled xxh64), translate, upstream client with full SSE machinery, fresh-read creds. The invariant crate.
- `crates/provider-grok` — complete non-streaming xAI provider with typed wire structs.
- `crates/bin/omni` — the only functional binary; 1110 lines, ~55% of which is its test module.
- `crates/bin/omni-claude`, `crates/bin/omni-grok` — stubs.
- `crates/providers/{claude-code,grok}` — **stray placeholder source files with no Cargo.toml** — not crates, not in the workspace.
- `reference-src-claude/` — vendored original CCP source (port reference).
- `docs/` — 12 files; quality ranges from excellent (INVESTIGATION.md, GROK guide) to actively wrong (CLAUDE.md).
- `examples/` — one orphaned file (workspace root is a virtual manifest with no `[package]`, so it never compiles).
- `design/`, `research/` — empty directories.

**Surprises this pass:** (1) the wire-defaults machinery being dead in the send path despite being the invariant's centerpiece; (2) `stats.rs` having literally empty method bodies behind a real API; (3) a unit test that performs an unconditional live Anthropic API call; (4) `free_port()` helpers already written in both stub bins but never used by the tests that needed them.

---

## 3. Audit Report

### Architecture & design

> **[High]** _Fingerprint wire defaults are not applied in the live send path_ — `wire_defaults_for_model()` and `outbound_model()` have zero callers outside `fingerprint.rs`; `output_config` is only ever set to `None` (`crates/provider-claude/src/translate.rs:344`); default `max_tokens` comes from the catalog (`translate.rs:295,348-350` → 128k opus / 64k sonnet, `models.rs:101,108`) instead of the captured wire defaults (64k/32k, `fingerprint.rs:663-699`); `temperature` is client-passthrough only (`translate.rs:310-314`) where real CC sends `1.0`. The profiles' own capture notes state real bodies carry `output_config:{"effort":"high"}` (`fingerprint.rs:657-662`), and `docs/INVESTIGATION.md` lists `apply_profile_wire_defaults` as gate-relevant outbound mutation. Why it matters: a default-shaped request through provider-claude differs from real Claude Code traffic in at least three body fields — precisely the drift class the invariant declares a failure (`fingerprint.rs:4-11`). The gate accepts it today; the design gives no margin for "today". _(Fact for the dead code paths; Judgment that the gate-rejection risk is material.)_

> **[High]** _Advertised binaries are stubs_ — `omni-claude` and `omni-grok` serve only `/health` and `/` with `// TODO: In full implementation...` where provider wiring belongs (`crates/bin/omni-claude/src/main.rs:13-15`, `crates/bin/omni-grok/src/main.rs:13-14`), hardcoded ports 18321/18322, no CLI/config. `docs/README.md` presents them as the two deliverables; `docs/DESIGN.md` calls the (actually-working) `omni` aggregator "deferred". The docs invert reality. Why it matters: a user who builds what the README points at gets an empty server. _(Fact)_

> **[Medium]** _Streaming is fully ported but unreachable_ — `upstream.rs` contains complete typed + raw SSE streaming (`EventStream`, `RawEventStream`, `send_messages_stream*`, `upstream.rs:151-612,934-1056`) with a bounded 16 MiB reassembly buffer, yet `LlmProvider` has no streaming method (`omni-core/src/traits.rs:5-9`) and `omni` 400s on `stream:true` (`crates/bin/omni/src/main.rs:455-461`). Why it matters: ~450 lines of working, tested machinery deliver zero user value; OpenAI clients default to streaming. _(Fact)_

> **[Medium]** _Stray duplicate "provider" sources_ — `crates/providers/claude-code/src/lib.rs` and `crates/providers/grok/src/lib.rs` are placeholder structs with **no Cargo.toml**, not workspace members, duplicating the real providers' names. Why it matters: grep/navigation hits two `GrokProvider`s; implies structure that doesn't exist. _(Fact)_

> **[Low]** _Orphaned example_ — the workspace root is a virtual manifest (no `[package]` in `Cargo.toml`), so `examples/shared_replacements_demo.rs` is never compiled by any cargo invocation, and its claim "full redb in lib" is false (see stats finding). _(Fact)_

### Code quality

> **[High]** _stats.rs is a no-op facade_ — `record_request`/`record_response`/`record_error` have empty bodies (`crates/omni-common/src/stats.rs:89-91`, comment "stubbed for compile in this pass"); the redb `Database` is opened but never written; clippy confirms `db` never read, `TokenStats` never constructed, and 9 table/limit constants unused (`stats.rs:14-23`). `docs/decisions.md` lists "stats (redb + TokenUsage + Active guards, per-model/per-key)" among the shared, tested components. Why it matters: any caller (and any reader of the docs) believes stats are recorded; nothing is. A facade is worse than an absence because it passes review. _(Fact)_

> **[Medium]** _Clippy: 1 denied error + ~44 warnings_ — the hard error is `assert!(out.status.success() || true)` (`crates/bin/omni/src/main.rs:1047`, `#[deny(clippy::overly_complex_bool_expr)]`), which also asserts nothing. Warnings include 4× `MutexGuard` held across `await` (provider-grok credential tests, e.g. `provider-grok/src/lib.rs:962-981`), 10× spawned child never `wait()`ed (zombie processes in bin tests), 2× unused `warn` import (both stub bins), unused-import clusters in `stats.rs`, and `if thinking_active { None } else { None }` identical branches (`provider-claude/src/translate.rs:320-321`). Why it matters: `cargo clippy --workspace --all-targets` exits non-zero, so a lint gate cannot be turned on without fixes; several warnings (zombies, mutex-across-await) are real defect classes, not style. _(Fact)_

> **[Medium]** _Formatting is not normalized_ — `cargo fmt --all --check` fails (`crates/bin/omni/src/main.rs` import order), and indentation is mixed across the tree: tabs in `omni-common/src/auth.rs`, `stats.rs`, `conversation_log.rs`, `time_util.rs`, `provider-claude/src/credentials.rs`; 4-space elsewhere. No `rustfmt.toml`. Why it matters: every future PR carries noise diffs until one normalization commit lands. _(Fact)_

> **[Low]** _Hand-rolled calendar math beside chrono_ — `time_util.rs` reimplements days→date conversion while `chrono` is already a dependency of the same crate (`omni-common/Cargo.toml`). _(Fact)_ Leap-second/edge risk is negligible but the duplication is pointless. _(Judgment)_

> **[Low]** _Pointless hash in session fallback_ — the anonymous fallback hashes the string literal `"default"`, producing the same constant every time (`omni-common/src/session.rs:25-28`); a constant string would be honest. _(Fact)_

### Security

Healthy for a localhost developer proxy. Credentials are read fresh per request and never cached (`provider-claude/src/credentials.rs:3-4`, `omni-common/src/credentials.rs:63-75`); API keys are logged redacted (`auth.rs:47-56`); the committed tree contains no secrets and `.gitignore` blocks credential files; binds are loopback-only (`bin/omni/src/main.rs:251`).

> **[Low]** _Non-constant-time key comparison_ — `valid_keys.contains(k)` on a `HashSet` (`omni-common/src/auth.rs:33`). Why it matters: timing side-channels are theoretical here (localhost, self-issued keys); fix only if the proxy is ever network-exposed. _(Judgment)_

### Testing

> **[High]** _`cargo test --workspace` fails reproducibly_ — `focused_omni_grok_subprocess_binary_health` panicked with an empty health body in both full-suite runs this session; two tests spawn the binary on the same hardcoded port 18322 (`crates/bin/omni-grok/src/main.rs:152,168`) and the stub `main` panics on a taken port via `.unwrap()` on bind (`main.rs:23`). The fix is half-written: a `free_port()` helper exists in both stub bins but is never used by these tests (clippy: "function `free_port` is never used"). The same fixed-port pattern exists in omni-claude (`main.rs:101`). Why it matters: the default verification command is red; nobody can tell a real regression from this known flake. _(Fact)_

> **[High]** _Unit test performs an unconditional live Anthropic call_ — `claude_send_exercises_full_fingerprint_path` calls `.send().await.expect("claude provider send must succeed with creds")` with no credentials guard (`crates/provider-claude/src/lib.rs:266-276`). Why it matters: `cargo test -p provider-claude` fails on any machine without `~/.claude/.credentials.json`, requires network, and spends real Claude Max quota on every test run. Grok's equivalent test guards and skips (`provider-grok/src/lib.rs:680-686`); Claude's does not. _(Fact)_

> **[Medium]** _Assert-light tests inflate the count_ — examples: router construction "verified" by Debug-string matching (`bin/omni/src/main.rs:705`), `assert!(true, "streaming...")` (`provider-grok/src/lib.rs:1172`), the `|| true` assert (`bin/omni/src/main.rs:1047`), three near-identical "all variants map to OAI shape" tests in `error.rs` (`omni-common/src/error.rs:83,114,147`). Why it matters: the suite's headline number (~160 tests passing before the failure) overstates real behavioral coverage; these cannot fail when logic changes. _(Fact for the cited tests; Judgment on the aggregate effect.)_

> **[Medium]** _~600 lines of tests inside the `omni` binary's `main.rs`_ — including a test-only router builder that duplicates `main`'s construction and can drift from it (`bin/omni/src/main.rs:677-695`). _(Fact; the drift risk is Judgment.)_

**Testing strengths:** the fingerprint pins are real and valuable — full header-set baseline per profile, per-model beta full-string locks against captured traffic (`fingerprint.rs:1283-1343`), cch placeholder/identity-ordering tests (`translate.rs:721-750`), SSE parser unit tests (`upstream.rs:618-705`), and honest replacements/credentials/error suites.

### Performance

Healthy for the maturity; no N+1, no blocking-in-async (credential reads use `tokio::fs`), upstream client is well-tuned (http2, keepalive, pooling — `upstream.rs:744-753`), SSE buffer is bounded (`upstream.rs:378`). One nit: `choices.drain(..).next()` allocates to take the first element (`provider-grok/src/lib.rs:294`). _(Fact; not a hotspot — Judgment.)_

### Dependencies

> **[Medium]** _redb shipped for a facade_ — `redb = "2.4"` is a workspace dependency consumed only by the no-op stats module. Why it matters: real build-time and supply-chain surface for zero runtime value; decide implement-or-delete. _(Fact)_

> **[Low]** _Version declarations scattered_ — `async-trait` (3 crates), `dirs = "6"` (2 crates), `ring`, `bytes`, `futures-util` are declared per-crate instead of `[workspace.dependencies]`; `workspace.package` defines `license`/`authors` but member crates inherit only `version`/`edition`, so built crates carry no license metadata. _(Fact)_

Lockfile is committed (correct for binaries); no abandoned or duplicate-major dependencies observed.

### DevEx & operations

> **[High]** _No CI, no enforced gates_ — no `.github/`, no `rustfmt.toml`/`clippy.toml`, no hooks. Why it matters: every red signal in this audit (failing test, clippy error, fmt drift, dead code) postdates nothing — there was never a floor to fall below. A 20-line workflow ends the class. _(Fact)_

> **[Low]** _Stub bins are unconfigurable_ — fixed `127.0.0.1:18321/18322`, no `--port`/auth/env handling (`bin/omni-claude/src/main.rs:21`). Matters mostly because the tests collide (see Testing/High). _(Fact)_

### Documentation

> **[High]** _docs/CLAUDE.md documents a different repository_ — its Build/Test/Architecture sections reference `src/main.rs`, `src/routes/completions.rs`, `src/config.rs` (`CCP_*` env vars), `./tests/run.sh`, and `tools/fingerprint/REBASELINE.md` — none of which exist in this tree (the layout belongs to the original CCP, now under `reference-src-claude/`). Why it matters: this is the conventional first-read for any agent or contributor; every operational instruction in it fails here. _(Fact)_

> **[Medium]** _README/DESIGN invert the actual state_ — `docs/README.md` presents `omni-claude`/`omni-grok` as the working focus and the aggregator as "noted for later"; in code the aggregator is the only thing that works and the two binaries are stubs. `docs/live-testing-results.md` reports "all green" for `--lib` scope and per-crate counts that no longer match (e.g. "omni-common 33+" vs 91 today), with no date/commit stamp. Why it matters: planning from these docs produces wrong priorities. _(Fact)_

> **[Low]** _Empty `design/` and `research/` directories._ _(Fact)_

**Documentation strengths:** `INVESTIGATION.md` is an unusually honest architecture analysis; `GROK_AND_MULTI_PROVIDER_ACCESS_GUIDE.md` and `grok-gate.md`/`grok-requirements.md` are excellent provider references; the in-code capture-provenance comments in `fingerprint.rs` are exemplary (every constant says when and how it was captured).

### Strengths (preserve these)

1. **The fingerprint port is the crown jewel.** Capture-dated constants, per-profile pins, hand-rolled xxh64 matching Claude Code's cch, placeholder-patch finalization, JS UTF-16 suffix semantics handled correctly (`fingerprint.rs:1070-1143`) — this is careful, verifiable work.
2. **upstream.rs is production-grade:** typed retry classification with a locked 429-passthrough rule, 401-refresh-once, bounded SSE reassembly, raw-frame passthrough mode for native-surface fidelity.
3. **Isolation discipline held:** no Claude-specific constants leak into `omni-core`/`omni-common` (grep-verified); the crate boundary the design demands actually exists.
4. **provider-grok is a complete, honest non-streaming implementation** with typed wire structs and correct error mapping.
5. **Security posture:** fresh-per-request creds, no caching, redacted key logging, loopback binds, no secrets in tree.
6. **conversation_log.rs is real quality** (bounded queue, dedicated writer thread, rotation, bounded shutdown flush) — it just isn't wired to anything yet.

### Highest-priority ugly parts

1. The **wire-defaults fidelity gap** — the one finding that touches the project's reason to exist.
2. **Red default suite + clippy error + no CI** — currently impossible to certify any change.
3. **The claims layer** (stub bins, stats facade, stale CLAUDE.md/README) — every reader is being misled the same direction: "more done than it is."

---

## 4. Improvement Strategy

**Theme 1 — Close the invariant, don't just document it.** The fingerprint machinery defines wire defaults, outbound-model preservation, and streaming fidelity; the new send path uses none of them. *Target state:* `prepare_anthropic_request` consumes `wire_defaults_for_model()` + `outbound_model()` and a vector-style test pins a default-shaped body against captured 2.1.165 values — or the invariant docs explicitly scope the canonical path down. *Principle:* an invariant that isn't enforced by code or test is a hope.

**Theme 2 — Make green mean green.** Failing default suite, a denied lint, fmt drift, and a unit test that needs live credentials make the basic verification loop untrustworthy. *Target state:* `cargo test --workspace` passes hermetically (no net, no creds — live tests opt-in via env/`#[ignore]`); clippy `-D warnings` and `fmt --check` pass; a small CI enforces all three. *Principle:* the machine owns the floor.

**Theme 3 — Say what is, delete what isn't.** Stub bins sold as products, a stats facade, stray non-crates, an orphaned example, docs for another repo. *Target state:* every advertised artifact either works or is labeled WIP in both code and README; facades and strays are deleted or implemented. *Principle:* the repo should never require oral tradition to interpret.

**Theme 4 — Tests that can fail.** Count-driven test culture (prose-heavy, assert-light, tautological) dilutes the genuinely excellent pins. *Target state:* tautologies deleted; bin tests live in `tests/`; every remaining test fails when its behavior changes. *Principle:* a test's value is the bug it can catch.

**Explicit trade-offs — recommend NOT fixing now:**
- **Don't split `fingerprint.rs` (2153 lines).** It is cohesive, invariant-critical, and its size is mostly captured data tables. Splitting risks the gate for aesthetics.
- **Don't delete `reference-src-claude/`** while the port still consults it (wire-defaults work will need it). Revisit after M2.
- **Don't build streaming through the trait yet.** Sequence it after the bins are wired; it's the largest item and currently has no consumer.
- **Don't add constant-time auth, rate limits, or body caps** for a loopback dev proxy.
- **Keep the per-version re-typed override tables** in fingerprint.rs — the duplication is documented as deliberate (explicit wire surface per profile) and it's data, not logic.

**Definition of done (measurable):**
- `cargo test --workspace` green on a machine with **no credentials and no network**, three consecutive runs (live tests behind `OMNI_LIVE_TESTS=1` or `#[ignore]`).
- `cargo clippy --workspace --all-targets -- -D warnings` exits 0.
- `cargo fmt --all --check` exits 0, after a one-time normalization commit.
- CI runs all three on push; red blocks merge.
- A pinned test asserts a default-shaped provider-claude body carries the captured 2.1.165 `max_tokens`/`temperature`/`output_config` (or docs re-scope the invariant explicitly).
- Every path/command in `docs/CLAUDE.md` and `docs/README.md` resolves against this tree; stub status stated where applicable.
- Zero empty-body "implemented" APIs (stats either records or is gone).

---

## 5. Task Plan

### Quick wins (high impact, S effort — do immediately)
- **QW1:** Use the existing-but-unused `free_port()` in the stub bins' subprocess tests → suite goes green.
- **QW2:** Guard the live Claude send test behind a creds/env check (mirror Grok's skip) → suite goes hermetic.
- **QW3:** Fix the clippy hard error (`|| true`) + unused imports → lint gate becomes possible.
- **QW4:** Add `.github/workflows/ci.yml` (test + clippy `-D warnings` + fmt `--check`).
- **QW5:** Delete `crates/providers/` strays and the orphaned example (or attach the example to `omni-common`).

### Milestones

| ID | Title | Milestone | Files/areas | Acceptance criteria | Effort | Change-risk | Depends on |
|----|-------|-----------|-------------|---------------------|--------|-------------|------------|
| T1 | Free ports in stub-bin subprocess tests (QW1) | M0 | `bin/omni-grok/src/main.rs`, `bin/omni-claude/src/main.rs` | `cargo test --workspace` green ×3 runs, default threads | S | Low | — |
| T2 | Hermetic default suite (QW2) | M0 | `provider-claude/src/lib.rs:266`, audit other creds-dependent tests | Full suite passes with `HOME` pointing at empty dir, no network | S | Low | — |
| T3 | Clippy clean under `-D warnings` (QW3) | M0 | `bin/omni/src/main.rs:1047,1076,1088`, `provider-grok/src/lib.rs:1172`, stub bins, `stats.rs` imports, `translate.rs:320,814` | `cargo clippy --workspace --all-targets -- -D warnings` exits 0 | M | Low | — |
| T4 | One-time `cargo fmt` normalization | M0 | whole tree (tabs→fmt default) | `cargo fmt --all --check` exits 0; single dedicated commit | S | Low | T3 |
| T5 | CI gate (QW4) | M0 | `.github/workflows/ci.yml` | CI red on any of test/clippy/fmt; green on clean checkout | S | Low | T1–T4 |
| T6 | Apply profile wire defaults in canonical path | M1 | `provider-claude/src/translate.rs`, `lib.rs`; pin test using captured values | Default-shaped body for opus/sonnet/haiku matches captured `max_tokens`/`temperature`/`output_config`; existing pins stay green; live smoke passes | M | **Med** | T1,T2,T5 |
| T7 | Reconcile docs with the tree | M1 | `docs/CLAUDE.md`, `docs/README.md`, `docs/live-testing-results.md` (stamp or archive), `docs/DESIGN.md` status note | Every referenced path/command resolves; bin stub status stated; test claims match `cargo test` output | M | Low | — |
| T8 | Delete strays (QW5) | M1 | `crates/providers/`, `examples/`, empty `design/`+`research/` | `git grep -l ClaudeCodeProvider` only hits real code; build unaffected | S | Low | — |
| T9 | Wire `omni-grok` binary to GrokProvider | M2 | `bin/omni-grok/src/main.rs` (+ shared OAI structs, see sketch) | Serves `/v1/chat/completions` via GrokProvider with `--port`/auth; live curl with key returns completion | M | Med | T1,T5 |
| T10 | Wire `omni-claude` binary to ClaudeProvider | M2 | `bin/omni-claude/src/main.rs` | Serves completions through full fingerprint path; live curl with `~/.claude` creds works | L | Med | T6,T9 |
| T11 | Extract `omni` bin tests to `tests/`; de-duplicate router construction | M2 | `bin/omni/src/main.rs` → `bin/omni/tests/` | `main.rs` ≤ ~450 lines; no `mk_app_with` duplicate; suite green | M | Med | T1 |
| T12 | Stats: implement or delete the facade | M2 | `omni-common/src/stats.rs`, `redb` dep, `examples` claim | Either `record_*` write redb and a snapshot endpoint exists, or module+dep removed | M | Med | owner decision (OQ2) |
| T13 | Sharpen assert-light tests | M3 | `bin/omni`, `provider-grok`, `omni-common/error.rs` | Tautological asserts removed/replaced; duplicate error tests merged | M | Low | T11 |
| T14 | Consolidate workspace dependencies + license inheritance | M3 | root + member `Cargo.toml`s | Shared deps in `[workspace.dependencies]`; `license.workspace = true` in members | S | Low | — |
| T15 | Streaming through the trait (SSE end-to-end) | M3 | `omni-core/traits.rs`, both providers, bins | `stream:true` returns OpenAI-style SSE for Grok (and Claude if T10 done) | XL | Med | T9, T10 |

### Top-3 implementation sketches

**T6 — Apply profile wire defaults (the invariant fix).**
*Approach:* thread the profile into the body-build the way the reference does. In `build_messages_request_from_canonical` (or a wrapper in `prepare_anthropic_request`), after model resolution: take `profile.wire_defaults_for_model(model_def.canonical)`; when the client didn't set them, fill `max_tokens` (replacing the catalog-derived default at `translate.rs:295`), `temperature`, and map `output_effort` → `OutputConfig { effort }`; use `profile.outbound_model(&canon.model, model_def)` for the wire model string so explicit pins forward verbatim (`preserve_explicit_model`). *Key steps:* (1) consult `reference-src-claude/translate/build.rs` / `routes/completions_v2.rs` for the original precedence (client-set beats default; thinking forces temp 1.0 — keep `translate.rs:310-314` ordering *after* defaults); (2) write a pin test: default canonical request for each of opus/sonnet/haiku asserts the serialized body's `max_tokens`/`temperature`/`output_config` equal the captured constants in `fingerprint.rs:663-747`; (3) run the existing live smoke. *Gotchas:* serde field order is load-bearing for cch — `output_config` already exists on `MessagesRequest` (`translate.rs:69`) so no struct change needed; do NOT reorder fields. Thinking-active requests override temperature to 1.0 last. Haiku gets no `output_config` (capture says none).

**T1+T2 — Hermetic green suite.**
*Approach:* in `bin/omni-grok` and `bin/omni-claude` tests, replace `const PORT` with the already-present `free_port()`; add `--port` (clap or plain `std::env::args` parse) to the stub mains so the tests can pass it — or simpler, have tests skip-bind: pick `free_port()`, pass via new `--port`. For T2: wrap the live Claude test body in the same guard Grok uses (`has_claude_creds()` exists in `bin/omni/src/main.rs:636`; replicate or move to a tiny shared test-util) and `eprintln!("skipping …")` + return when absent; alternatively mark `#[ignore]` with a doc comment naming the opt-in env. *Gotchas:* the stub mains `.unwrap()` on bind — with free ports per test the collision disappears, but also `.wait()` the children after `.kill()` to clear the zombie-process clippy lints in the same pass.

**T9 — Wire `omni-grok` to its provider.**
*Approach:* the `omni` aggregator already contains the whole pattern (CLI, `AppState`, auth layer, `to_canonical`/`from_canonical`, handler). Extract the OAI request/response structs + the two conversion functions into a small shared module (new `omni-common::oai` or a `bin-support` crate) so `omni` and `omni-grok` share one copy — this also shrinks `omni/main.rs` toward T11. Then `omni-grok/main.rs` = clap CLI (`--port`, `--no-auth`, `OMNI_API_KEYS`) + single `GrokProvider::new(None)` + the shared handler with a one-provider map. *Gotchas:* don't create a third copy of the OAI structs (there are already two shapes: `omni`'s and the test `mk_app_with` duplication); keep `omni`'s routing semantics (bare model allowed when single provider). Update the bin's subprocess tests to exercise a real completion path against a bad-port base URL for a deterministic upstream error.

---

## 6. Open Questions (need the owner)

1. **Is the wire-defaults omission deliberate?** If the gate demonstrably tolerates client-shaped `max_tokens`/`temperature`/no-`output_config` (because real CC also varies them per user config), T6 could be descoped to a documented decision instead of code. A fresh mitmproxy capture comparing CCP-original vs omni bodies would settle it. Until then the invariant doc and the code disagree.
2. **Stats & conversation_log: implement or delete?** Both are scaffolding for a "full" deployment story. If the `omni` aggregator is the product, wiring conversation_log + a real stats snapshot endpoint is a natural M2; if the per-provider bins are the product, decide there. (T12 blocks on this.)
3. **Which binary story wins?** Three binaries (claude/grok/aggregator) or aggregator-only? T9/T10 assume the README's story; if you'd rather consolidate on `omni`, T9/T10 become "delete the stubs and update docs" (cheaper).
4. **`reference-src-claude/` lifecycle** — keep permanently as the port oracle, or remove after T6/T10 complete and the pins are trusted?
5. **Exposure target** — will this ever listen on non-loopback? If yes, the Low security items (constant-time compare, body limits, health-exempt auth) get promoted.
6. **Streaming priority** — is non-streaming acceptable for your clients (e.g. specific tools), or is T15 actually the next most valuable feature after wiring?

---

*Audit produced with no code modifications. Previous session's commit `6ed9ad0` (initial import + `.gitignore`) is the audited baseline.*
