# Consolidation & Simplification Report — omni-llm-provider

**Date:** 2026-07-02 · **Method:** full read of the six workspace crates (production + test code), cross-provider duplication mapping, and direct verification of every dead-code / duplication claim below. Findings are labeled **(Fact)** — verified in code — or **(Judgment)** — an opinion on what to do about it.

**Scope:** This report is *analysis only* for the consolidation findings. No consolidation code was changed. It identifies where logic is duplicated across the three provider crates, where small hygiene cleanups exist, and ranks the work by value against risk. It deliberately does **not** recommend restructuring the crate layering — that is sound (see [What not to touch](#7-what-not-to-touch)).

**Also tracked here:** a separate, approved **observability feature track** — restoring correlation-ID logging and adding colorized log output — is folded into the plan in [§9](#9-observability-track-correlation-ids--colorized-logs-separate-feature-work). That work *adds* functionality (it is not consolidation) and has its own detailed handoff; §9 records it and its one coordination point with the consolidation batch.

---

## 1. Executive Summary

**The architecture is fundamentally sound.** The high-risk machinery is already consolidated: the OpenAI-Responses SSE parser, the response→canonical mapper, the `ErrorRedactor` trait, and the entire version/catalog abstraction all live in the shared crates (`omni-core`, `omni-common`). This is not a codebase that needs restructuring.

**What it has is request-side boilerplate duplicated across the three providers, plus a few small hygiene items.** None of it is load-bearing complexity; most is copy-paste that the source comments themselves acknowledge (e.g. "Mirrors grok's `redact`", "modeled exactly after the Claude Code Provider technique").

**One reframing:** the files that look enormous are mostly tests.

| File | Total | Production | Test | Test % |
|---|---|---|---|---|
| `crates/bin/omni/src/main.rs` | 6215 | ~1996 | ~4219 | 68% |
| `crates/provider-grok/src/lib.rs` | 5253 | ~2016 | ~3237 | 62% |
| `crates/provider-codex/src/lib.rs` | 4192 | ~1695 | ~2497 | 60% |
| `crates/provider-claude/src/fingerprint.rs` | 2701 | ~1584 | ~1117 | 41% |
| `crates/omni-common/src/responses.rs` | 2689 | ~1409 | ~1280 | 48% |

That is a healthy test-to-code ratio for Rust. "Split the monoliths" is really about the ~1700–2000 production lines in each; the tests would move alongside whatever production module they exercise, not shrink.

**Approved first batch (2026-07-02 owner walkthrough — high value, low risk):** [#1](#31-secret-redaction--clearest-win) (secret redaction), [#2](#32-envheader-plumbing-helpers) (env/header helpers), [#6](#6-duplication-within-omni-common-edge-protocol-pair) (shared format leaf-helpers), and the [hygiene items](#5-hygiene-items-cheap-low-risk) (empty `crates/core/`, dead `iso_now`, loose `executables/`). A few hundred lines removed with near-zero behavioral risk, because the shared trait and the tests that pin behavior already exist. **Deferred:** credential-loading merge and the Responses body-builder merge (higher risk / lower reward). **Implemented 2026-07-02** on branch `consolidation-batch-1` (5 commits; full `cargo test --workspace` = 581 passed, 0 failed). See [§8](#8-ranked-action-plan) for the full decision record and as-built notes.

**Plus a separate approved feature track ([§9](#9-observability-track-correlation-ids--colorized-logs-separate-feature-work)):** restore correlation-ID logging (Phase 1) and add colorized log output (Phase 2). This *adds* observability rather than removing duplication; it touches `main.rs` (one coordination point with consolidation item #2) and is best landed **after** the consolidation batch.

---

## 2. Architecture as it stands (for context)

```
OpenAI / Anthropic-format client
   │
   ▼
crates/bin/omni        HTTP edge: axum router, 8 routes, auth middleware,
                       route handlers, stats, conversation logging, CLI/config
   │
   ├── omni-common     Shared infra: OpenAI Chat + Responses wire types,
   │                   canonical↔wire conversion, SSE framing (both directions),
   │                   the shared Responses upstream parser + ErrorRedactor trait,
   │                   AppError/classify_upstream, auth layer, stats, replacements
   │
   ├── omni-core       Pure types + traits, no I/O: CanonicalRequest/Response/Stream,
   │                   LlmProvider trait, ProviderError, CatalogModel/ProviderVersion/
   │                   resolve_version (the version+catalog abstraction)
   │
   └── provider-claude / provider-codex / provider-grok
                       Per-provider: credentials, HTTP/WS transport, request-body
                       builders, wire fingerprint (headers/signing), model catalog data
```

Dependency direction is clean and acyclic: `bin/omni` → providers → `omni-common` → `omni-core`.

**Already-consolidated (no action — noted so the report is honest about what is already good):**
- `omni-common`: `ErrorRedactor` trait, `ResponsesSseBuffer`, `ResponsesStreamParser`, `response_to_canonical`, OpenAI/Responses request↔canonical conversion, outbound SSE framing, `AppError`/`classify_upstream`, `auth_layer`.
- `omni-core`: canonical request/response/stream types, `CatalogModel`/`ProviderVersion`/`resolve_version`, `LlmProvider` trait + `ProviderError`.

Codex and Grok-conservative both consume the shared Responses parser and mapper directly — the wire parsing is genuinely shared, not copied.

---

## 3. Cross-provider duplication (the real consolidation work)

Ranked by value ÷ risk. Every line reference below was read directly.

### 3.1 Secret redaction — clearest win

**(Fact)** The free function `fn redact(input: &str) -> String` is copy-pasted three times:

| Provider | Location | Markers |
|---|---|---|
| Claude | `provider-claude/src/lib.rs:341` | `["sk-", "eyJ"]` |
| Grok | `provider-grok/src/lib.rs:900` | `["sk-", "xai-", "eyJ"]` |
| Codex | `provider-codex/src/lib.rs:1681` | `["sk-", "xai-", "eyJ"]` |

Grok and Codex are **byte-for-byte identical**; Claude differs only by omitting the `"xai-"` marker. All three use the same delimiter set (`whitespace | '"' | '\'' | ','`) and the same `while let Some(pos) = out.find(marker)` loop.

On top of that, each provider wraps `redact` in a near-identical struct:

| Provider | Struct | Location |
|---|---|---|
| Claude | `ClaudeErrorRedactor` | `provider-claude/src/lib.rs:361-390` |
| Grok | `GrokErrorRedactor` | `provider-grok/src/lib.rs:925-956` |
| Codex | `CodexErrorRedactor` | `provider-codex/src/lib.rs:1610-1656` |

All three are `#[derive(Clone, Debug, Default)] struct … { secrets: Vec<String> }` with the same "sort longest-first, dedup, then replace each exact secret after calling `redact()`" body (~6 identical lines each). Claude and Grok both have a `for_credentials(&…)` constructor with ~10 identical lines. Codex substitutes `for_secrets(…)` + `from_request(&Url, &HeaderMap)` (it harvests secrets from request headers/query) but the redaction body is identical. The source comments explicitly acknowledge the copies.

**(Judgment)** **Consolidate into a single `SecretRedactor` in `omni-common`** taking a configurable marker list plus the exact-secret vector. This is the safest consolidation in the report: the three types already implement (or mirror) the `ErrorRedactor` trait that *already lives in `omni-common`*, and the parser tests pin the behavior.

- **Effort:** S · **Risk:** Low · **Removes:** ~130 lines · **Decision (2026-07-02): APPROVED — first batch.**

### 3.2 Env/header plumbing helpers

**(Fact)** Three tiny, pure, dependency-free functions are duplicated:

- **`fn env_nonempty(name: &str) -> Option<String>`** — defined **4 times**, byte-identical: `provider-claude/src/lib.rs:323`, `provider-codex/src/lib.rs:1090`, `provider-grok/src/lib.rs:958`, **and** `crates/bin/omni/src/main.rs:533`.
- **`parse_custom_headers`** (parse `Name: value` lines from an env string) — near-identical ~15-line copies: `provider-claude/src/upstream.rs:1272`, `provider-grok/src/lib.rs:880`, `provider-codex/src/lib.rs:1104`.
- **`headers_from_env`** (read env var → `parse_custom_headers`) — three ~7-line copies: `provider-claude/src/upstream.rs:1289`, `provider-grok/src/lib.rs:869`, `provider-codex/src/lib.rs:1097`.

**(Judgment)** **Move all three into `omni-common` as free functions.** Pure and side-effect-free (modulo reading env), no wire-fidelity implications.

- **Effort:** S · **Risk:** Low · **Removes:** ~60 lines · **Decision (2026-07-02): APPROVED — first batch.**

### 3.3 Provider builder scaffolding

**(Fact)** The catalog/version *logic* is already shared in `omni-core::version` (`ProviderVersion`, `CatalogMode`, `resolve_version`, `ProviderVersion::resolve_model`). Only the thin wrapper methods repeat between Grok and Codex:

| Method | Codex | Grok | Relationship |
|---|---|---|---|
| `with_mode(mut self, mode) -> Self` | `lib.rs:109-112` | `lib.rs:263-266` | identical |
| `with_version(&str) -> Result<Self,_>` | `lib.rs:117-130` | `lib.rs:271-284` | near-identical (~14 lines) |
| `active_catalog()` | `lib.rs:133-135` | `lib.rs:287-289` | identical |
| `fn versions()` trait impl | `lib.rs:367-369` | `lib.rs:1770-1772` | identical |
| `mode` + `version` field pair (+ doc comments) | `lib.rs:89-92` | `lib.rs:170-172` | identical |

Claude is intentionally excluded here — it ties its model catalog to the wire-fingerprint profile (`FingerprintProfile`), a parallel system with the same exact-then-alias resolution semantics but a different struct (`ModelDef` in `provider-claude/src/models.rs:9`, `resolve_model_in_catalog` at `models.rs:160`).

**(Judgment)** **Provide the shared behavior as default trait methods (or a small macro).** Low value — the wrappers are shallow — but trivially safe.

- **Effort:** S · **Risk:** Low · **Removes:** ~30 lines · **Priority:** low · **Decision (2026-07-02): NOT in first batch** — deferred to keep the initial commit focused.

### 3.4 Credentials loading pattern

**(Fact)** All three providers share the pattern "resolve a path (env override → home dir) → read fresh per request (never cached) → parse JSON → extract token → optional expiry check → typed error." No token *refresh* happens anywhere (deliberate — the upstream CLI owns refresh; Claude re-reads on 401, Grok/Codex surface expiry).

- The `default_path()` idiom (`env_var_os(OVERRIDE)` → `dirs::home_dir().join(...)`) is triplicated: `provider-claude/src/credentials.rs:51`, `provider-grok/src/credentials.rs:95`, Codex `codex_home()` at `provider-codex/src/lib.rs:1156` (~8 lines each).
- `load_fresh` / `load_fresh_async` / `from_bytes` + `check_expired` are near-identical between Claude (`credentials.rs:65-104`) and Grok (`credentials.rs:196-265`) — ~32 lines, same doc-comment rationale. Grok's file header says it is "modeled exactly after the Claude Code Provider technique."
- The credential error enums mirror each other: Claude `UpstreamError::{CredentialsRead, CredentialsParse, CredentialsMissingToken, TokenExpired}` vs Grok `GrokCredentialsError::{Read, Parse, MissingToken, Expired, NoSource}` — same variant set, renamed.

**(Judgment)** The on-disk JSON shapes genuinely differ (Claude OAuth-only, Grok static-key *and* OIDC, Codex TOML config + subprocess), so full unification is not clean. **Extract only the read-fresh/parse/expiry skeleton** (Claude + Grok) into a shared helper; leave Codex's multi-source loader separate.

- **Effort:** M · **Risk:** Medium (touches auth) · **Removes:** ~40 lines · **Priority:** medium · **Decision (2026-07-02): DEFERRED** — touches authentication, low reward; revisit after the safe wins land.

### 3.5 Responses request-body builders — highest-value, highest-risk

**(Fact)** `to_grok_responses_request` (`provider-grok/src/lib.rs:1227`) and `codex_responses_body` (`provider-codex/src/lib.rs:1411`) are same-shape POST-body builders: identical `tools` array shaping (`{type:"function", name, description, parameters}`), identical `tool_choice` match arms (Auto/Required/None/Specific), identical `max_output_tokens`/`temperature`/`top_p`/`reasoning:{effort}` blocks. Their block converters `append_grok_responses_items` (`grok lib.rs:1305`) and `append_message_items` (`codex lib.rs:1484`) are near-identical (`input_text` / `input_image` via `source.as_image_url()` / `function_call` / `function_call_output`). ~160 near-duplicate lines each. They differ only in: Codex hoists system/developer roles → `instructions`, and Codex applies a `provider_extras` allow-list. Grok's comments reference the Codex versions directly.

**(Judgment)** This is the **largest single block of duplicated logic** but also the **most delicate** — each body must match a live-captured wire fingerprint byte-for-byte, and the two intentionally diverge (the instructions-hoist). **Do this last, if at all.** A shared builder would need to parameterize the hoist and the allow-list. The wire-parity tests are what would make it safe; do not attempt without them green.

- **Effort:** L · **Risk:** High (wire fidelity) · **Removes:** ~160 lines · **Priority:** defer · **Decision (2026-07-02): DEFERRED** — highest risk; wait until wire-parity tests are confirmed green.

### 3.6 HTTP client construction & retry

**(Fact)** Each provider builds its own `reqwest::Client`; there is no shared builder. The `Client::builder()…timeout(…).build().map_err(…)` idiom repeats three times (`provider-claude/src/upstream.rs:816`, `provider-codex/src/lib.rs:96`, `provider-grok/src/lib.rs:210`), but the tuning genuinely differs — Claude adds HTTP/2 prior-knowledge + rustls + keepalive + pool settings that are **load-bearing for its fingerprint**; Codex uses 600s timeout; Grok 300s. Retry logic exists **only** in Claude, where `send_messages_json` and `count_tokens` (`upstream.rs:855` and `:934`) contain a near-identical 401-refresh + exponential-backoff loop that duplicates *each other* (~55 lines).

**(Judgment)** A cross-provider client builder would need an options struct and would save little; the divergence is real, not accidental. **Skip the cross-provider builder.** The *intra-Claude* retry-loop duplication (see [#4](#42-provider-claudesrcupstreamrs)) is the part worth fixing.

- **Effort:** M · **Risk:** Medium · **Priority:** skip (cross-provider); low (intra-Claude retry).

---

## 4. Within-file duplication

Relevant primarily *if/when* the large files are split. Each is internal to one file.

### 4.1 `crates/bin/omni/src/main.rs`

- **(Fact)** `chat_completions_handler` (`:1089`) and `responses_handler` (`:1892`) are near-duplicate handler bodies — same sequence of request-id generation, session-id resolution, inbound logging, `resolve_provider_and_model`, provider lookup, `*_to_canonical`, `validate_provider_extras`, stats recording, and stream-vs-non-stream branch. They differ only in request/response types and the SSE framer called. **Largest intra-file duplication in the binary.**
- **(Fact)** The provider-error `.map_err` closure is copy-pasted 4× (`:1136`, `:1167`, `:1940`, `:1976`), each an identical `|e| { if let Some(stats) = &state.stats { stats.record_error(…) } map_provider_err(e) }`.
- **(Fact)** `short_request_id = request_id.chars().take(8).collect::<String>()` preceded by `Uuid::new_v4()` is repeated 4× (`:1096`, `:1416`, `:1553`, `:1899`).
- **(Fact)** Three parallel `*_session_id` helpers with the same header→body→api-key fallback shape: `chat_session_id` (`:1336`), `responses_session_id` (`:1349`), `anthropic_session_id` (`:1668`).

**(Judgment)** If split: extract the handler skeleton (a generic over the protocol's to/from-canonical + SSE framer), the error closure, and a single session-id resolver. Cleanest module extraction targets: model→provider routing (`:928-1054`, already pure), Anthropic SSE framing (`:1686-1891`, self-contained), CLI (`:107-211`).

### 4.2 `crates/provider-claude/src/upstream.rs`

- **(Fact)** Two retry loops (`send_messages_json:855`, `count_tokens:934`) duplicate each other ~55 lines.
- **(Fact)** `EventStream` (`:402-469`) and `RawEventStream` (`:539-627`) are near-identical `poll_next` implementations differing only in `parse_event_block` vs `parse_raw_frame`.

### 4.3 `crates/provider-codex/src/lib.rs`

- **(Fact)** `send` (`:371`) and `send_stream` (`:427`) share a large identical preamble (config load, conservative-WS eligibility branch, `WireApi::Responses` guard — lines `:373-384` vs `:429-440` are line-for-line the same).
- **(Fact)** `openai_auth_token` (`:1011`) and `openai_auth_token_fallback` (`:980`) are ~90% identical (differ only in `Result` vs `Option` and env-vs-file precedence). The env-key array `["CODEX_API_KEY", "OPENAI_API_KEY", "CODEX_ACCESS_TOKEN"]` is hard-coded in 3 places (`:651`, `:1001`, `:1012`).
- **(Fact)** Two conservative-header builders set the same headers twice in different styles: `conservative_codex_headers` (`:1214`, via `header::HeaderMap`) and `conservative_ws_request` (`:1251`, via raw `http::HeaderValue`), with two parallel insert helpers `insert_header` (`:1319`) and `http_header_value` (`:1307`).

---

## 5. Hygiene items (cheap, low-risk)

- **(Fact)** **`crates/core/` is an empty leftover directory** (no files). The real crate is `crates/omni-core`. → Delete it.
- **(Fact)** **`omni-common::time_util::iso_now` is dead code** — zero callers workspace-wide (verified). The module hand-rolls a leap-year/days-to-date calendar while `chrono` is *already* a workspace dependency of `omni-common`. Its only live function, `time_of_day_now`, has a single caller (`conversation_log.rs:170`), and `stats.rs:222` already uses `chrono` for timestamps. → Delete `iso_now`; fold `time_of_day_now` into `conversation_log.rs` or replace with a chrono call; drop the module.
- **(Fact)** **`executables/` (162 MB) — DONE (2026-07-02).** Contained two built `omni` binaries (`debug/omni` 155 MB, `release/omni` 14 MB) loose in the working tree. Verified they were **never committed** (`git log -- executables/` empty; largest historical blob is 634 KB), so **no history rewrite was needed** — history was already clean. Deleted from disk and added `/executables` to `.gitignore` next to `/target`.
- **(Fact)** In `omni-common/src/responses_upstream.rs`, `response_payload` (`:957`), `response_status` (`:948`), and `response_incomplete_reason` (`:971`) are `pub` but have no external callers — used only within the file. → Make private.
- **(Fact/Judgment)** **`reference-src-claude/` (512 KB, 33 files tracked in git)** is the vendored port source, referenced only in doc-comment provenance notes and the historical audit — not compiled. Harmless, but if the provenance is captured elsewhere it could be dropped from the tree. → Optional; leave unless the repo wants a smaller checkout.

---

## 6. Duplication within `omni-common` (edge protocol pair)

**(Fact)** `http.rs` (Chat Completions, client edge) and `responses.rs` (Responses API, client edge) carry parallel, partly copy-pasted mapping helpers in the `from_canonical` direction:

- `usage_detail_json` — defined **twice, byte-identical** (`http.rs:552`, `responses.rs:575`).
- The audio-token-split idiom is triplicated (`http.rs:520-525`, `responses.rs:545-550`, `responses.rs:763-769`).
- `provider_metadata_json` (`http.rs:562`) vs `responses_provider_metadata_json` (`responses.rs:585`) are near-identical.
- `chat_usage_from_canonical` (`http.rs:519`) and `responses_usage_from_canonical` (`responses.rs:544`) are the same mapping with `prompt/completion` vs `input/output` field names.
- The `tool_choice` mapping (`http.rs:285-301` vs `responses.rs:356-370`), the non-function-tool rejection loop (`http.rs:261-281` vs `responses.rs:333-352`), and the has-image content collapse (`http.rs:405-450` vs `responses.rs:404-440`) are effectively copy-pasted.
- Within `responses.rs`, the non-streaming (`responses_from_canonical:449`) and streaming (`responses_stream_envelope:716`) builders independently rebuild the same `output` items / `usage` object / `msg_`+`fc_` id-prefix conventions.

`responses_upstream.rs` is **not** a duplicate of these — it parses the same wire vocabulary in the *opposite* direction (upstream reply → canonical) and is shared by Codex + Grok. It earns its separate existence.

**(Judgment)** A small shared `canonical_mapping` helper module (usage-detail, audio-split, provider-metadata, tool-choice, image-collapse) would remove the `http.rs`↔`responses.rs` redundancy. **Medium value** — the wire *types* legitimately differ so the message-walking can't fully merge, but the leaf helpers can. **Effort:** M · **Risk:** Medium (both protocols are pinned by large test suites — safe to attempt with them green). **Decision (2026-07-02): APPROVED — first batch** (merge the leaf helpers only, not the overall message-walking).

---

## 7. What not to touch

These are correctly separated and legitimately provider- or layer-specific. Leave them:

- **The core/common/provider layering** and the acyclic dependency direction.
- **The version/catalog abstraction** in `omni-core::version` — already the right shape; providers own only their catalog *data*.
- **The shared Responses upstream parser** (`ResponsesSseBuffer`, `ResponsesStreamParser`, `response_to_canonical`) and the `ErrorRedactor` trait.
- **Per-provider wire fingerprints** — Claude's billing "cch" checksum + Stainless headers (`fingerprint.rs`), Grok's `x-grok-*` conservative headers (`lib.rs:595`), Codex's WebSocket transport and `chatgpt-account-id` headers (`lib.rs:207`, `:1214`). Each mimics a different CLI's byte-exact wire; the apparent "duplication" is coincidental structure, not shared logic.
- **The per-protocol SSE payload parsers** — Anthropic message events (Claude), OpenAI chat chunks (Grok extended), Responses events (shared). Only the byte→event framing is conceptually similar; the payload parsing must differ by protocol. (The framing itself is a *possible* consolidation but low value and touches three hot paths — not recommended.)

---

## 8. Ranked action plan

**Decisions recorded 2026-07-02** (owner walkthrough). The first batch was **implemented 2026-07-02** on branch `consolidation-batch-1` (commits below); the observability track (§9) has not started.

### Approved — first batch

GitHub issues filed 2026-07-02 on `sonami-tech/omni-llm-provider`. Implemented on branch `consolidation-batch-1`, landed **before** the observability track (§9); one commit per issue (the env/header item split into two so the `main.rs` `env_nonempty` removal is isolated); per-step gate = build + clippy + fmt --check + tests **scoped to the crates touched**; final **full `cargo test --workspace` = 581 passed, 0 failed, 0 warnings**. `Closes #N` trailers auto-close the issues on merge to `master`.

| # | Item | Issue | Effort | Risk | ~Lines removed | Status |
|---|---|---|---|---|---|---|
| 1 | Unify secret redaction → shared `redact_prefixed_secrets` ([§3.1](#31-secret-redaction--clearest-win)) | [#5](https://github.com/sonami-tech/omni-llm-provider/issues/5) | S | Low | ~130 | **DONE** (`3a62c1c`) |
| 2 | Move `env_nonempty` / `parse_custom_headers` / `headers_from_env` to `omni-common` ([§3.2](#32-envheader-plumbing-helpers)) | [#8](https://github.com/sonami-tech/omni-llm-provider/issues/8) | S | Low | ~60 | **DONE** (`29faa9b` env_nonempty + `f434311` headers) |
| 6 | Shared `canonical_mapping` **leaf helpers** for `http.rs`↔`responses.rs` ([§6](#6-duplication-within-omni-common-edge-protocol-pair)) | [#7](https://github.com/sonami-tech/omni-llm-provider/issues/7) | M | Medium | ~75 | **DONE** (`c86dd5c`) — scoped to the two identical output helpers (`usage_detail_json`, `provider_metadata_json`); tool-choice + image-collapse left per-format (see note) |
| H1 | Delete empty leftover `crates/core/` directory ([§5](#5-hygiene-items-cheap-low-risk)) | [#6](https://github.com/sonami-tech/omni-llm-provider/issues/6) | S | Low | — | **DONE 2026-07-02** (untracked; closed manually) |
| H2 | Delete dead `time_util::iso_now`; fold/replace `time_of_day_now` ([§5](#5-hygiene-items-cheap-low-risk)) | [#9](https://github.com/sonami-tech/omni-llm-provider/issues/9) | S | Low | ~90 | **DONE** (`58f90d1`) |
| H3 | Delete loose `executables/` + gitignore it | — | S | Low | 162 MB | **DONE 2026-07-02** (no history rewrite needed) |

### Deferred — revisit after the first batch lands

| # | Item | Effort | Risk | ~Lines | Why deferred |
|---|---|---|---|---|---|
| 3 | Shared provider builder methods ([§3.3](#33-provider-builder-scaffolding)) | S | Low | ~30 | Low value; kept out to keep the first commit focused |
| 4 | Claude+Grok credential read-fresh/expiry skeleton ([§3.4](#34-credentials-loading-pattern)) | M | Medium | ~40 | Touches authentication, low reward |
| 5 | Shared Responses body builder ([§3.5](#35-responses-request-body-builders--highest-value-highest-risk)) | L | **High** | ~160 | Byte-exact wire fidelity; needs green wire-parity tests first |
| — | Intra-file dedup (handler skeleton, retry loops, auth-token fns) ([§4](#4-within-file-duplication)) | M | Medium | ~200 | Bundle with any future file split |

**First commit batch = #1 + #2 + #6 + H1 + H2** (H3 already done). Near-zero behavioral risk: the shared `ErrorRedactor` trait and the test suites that pin both wire protocols already exist. Removes a few hundred lines and establishes the `omni-common` home for the shared request-side helpers that the deferred items would later build on.

**Implementation notes (2026-07-02, as built):**

- **#2 — custom-header parsing unified on the *validating* variant.** Grok and Claude validated header syntax at parse time; Codex only at insertion. The shared `omni_common::env::parse_custom_headers` validates, which is behavior-preserving for Codex because its `insert_header` already ran the same `HeaderName`/`HeaderValue` checks — a malformed header now fails one step earlier with an equivalent error (no test pinned Codex's parse-time message). The shared functions return a `String` error; each caller maps it to its own type (`ProviderError::Auth` for Grok/Codex, `UpstreamError::Decode` for Claude). Claude keeps its `validate_custom_header` helper, which independently validates config-supplied headers.
- **#6 — scoped tighter than the candidate list.** Only the two genuinely-identical *output-direction* leaves were shared: `usage_detail_json` (byte-identical) and `provider_metadata_json` (near-identical; the Chat/Responses divergence on `annotations` is passed as an explicit `include_annotations` flag so it stays visible at the call site). The `tool_choice` and image-content helpers were **left per-format**: they sit in the `to_canonical` direction and operate on the distinct `ChatToolChoice`/`ChatMessageContent` vs `ResponsesToolChoice`/`ResponsesInputContent` input enums, so sharing them would couple two wire parsers — the same reason the message-walking routines were excluded. The audio-split idiom was left inline (a 5-line pattern, not a leaf function).

---

## 9. Observability track: correlation IDs + colorized logs (separate feature work)

This is **new functionality, not consolidation** — it *adds* observability that existed in the legacy `claude-code-provider` (v1.1.16–v1.1.18) and was lost in the move to the `omni` workspace. It is folded into this plan for coordination, but it has its own source-of-truth handoff: `scratchpad/correlation-logging-handoff.md` (verified against `master` on 2026-07-02). **Not started.**

### What it delivers (two independent phases)

- **Phase 1 — Correlation IDs on operational logs.** Today the tracing subscriber is bare (`crates/bin/omni/src/main.rs:234-238`) and there is **no** `info_span!`/`.instrument()` anywhere in the live crates (verified — count 0), so an `info!`/`warn!`/`error!` emitted from inside a provider mid-stream carries **no** `request_id`/`session_id`. Phase 1 adds a single `middleware::from_fn` span layer (mirroring the existing `omni_auth_layer` wiring at `main.rs:633`) that opens an `info_span!("request", request_id, session_id = Empty)` per request; each of the three inference handlers records `session_id` after it derives it; and the streaming path is `.instrument()`-ed at the one attachment point, `wrap_stream_for_stats` (`main.rs:1231`). Fixes a real observability regression; ships even if Phase 2 slips.
- **Phase 2 — Colorized stderr stream (the coloring functionality).** Ports the original color module (`reference-src-claude/log_color.rs`, 279 lines) into the **bin crate** (`crates/bin/omni/src/log_color.rs`) — presentation for the one binary that owns stderr; explicitly *not* `omni-common`. Stable per-value hues so one session/request reads as a vertical color stripe when tailing interleaved multi-provider traffic. **No new dependency** — `nu-ansi-term` 0.50.3 is already in `Cargo.lock` (transitive via `tracing-subscriber`).

### Decisions already made in the handoff (do not re-litigate)

- **Span creation in middleware, not per-handler** — the legacy per-handler `info_span!`+`.instrument()` leaked at every new `.await`/spawn boundary; one tower layer avoids repeating that across three handlers.
- **`session_id` recorded late** — open the span with `session_id = tracing::field::Empty` (middleware runs before body parse) and `Span::current().record(...)` it in each handler. Do **not** parse the body in middleware.
- **ID ownership:** generate `request_id` in the layer, read it in the handlers (so the span ID and the conversation-log ID are guaranteed identical).
- **What gets colored changed from v1** (axes shifted since v1 was single-backend):
  - `request_id`, `session_id` → hashed palette (FNV-1a % palette length), as before.
  - `provider` (claude|grok|codex) → **fixed** assigned colors, NOT hashed (closed set of 3; more useful at a glance than the old session stripe).
  - `finish_reason` → keep the cue: bold amber `tool_calls`, green `stop`.
  - `pid` → **dropped** (no subprocess model anymore; dead key).
- **Env var rename** `CCP_LOG_COLOR` → `OMNI_LOG_COLOR` (`auto|always|never`); keep `NO_COLOR` support (takes precedence). Auto-disable on non-TTY (Docker/CI/redirect).
- **Rejected** (don't reintroduce): JSON logs + external colorizer (tspin/lnav — no off-the-shelf tool does stable per-value hues); OpenTelemetry (far beyond a local dev proxy).

### Coordination with the consolidation batch — one interaction point

**(Fact)** The only overlap between the two tracks is the file `crates/bin/omni/src/main.rs`:
- Consolidation item **#2** removes `main.rs`'s own copy of `env_nonempty` (`main.rs:533`) as part of moving it to `omni-common`.
- The observability track edits `main.rs`'s subscriber setup, adds the span middleware, and touches the three handlers + `wrap_stream_for_stats`.

There is **no symbol conflict** — the logging work does not use `env_nonempty`, `iso_now`, or the `http.rs`/`responses.rs` leaf helpers (verified). It is purely a same-file merge concern.

**(Judgment) Sequencing: land the consolidation first batch (§8) before the observability track.** The consolidation batch is smaller, lower-risk, and log-neutral; doing it first means the observability track edits a `main.rs` that already has the `env_nonempty` removal settled, avoiding a merge conflict on that hunk. If the tracks must run in parallel, keep the `env_nonempty` removal (#2) and the subscriber/middleware edits in separate commits so they rebase cleanly.

### Acceptance (from the handoff)

- Phase 1: a `warn!`/`info!` from inside a provider during **both** a streaming and non-streaming request shows `request_id=` and `session_id=`; the `request_id` on an operational line matches the `request=` in the corresponding conversation-log entry; no wire/parity regression (spans are log-only).
- Phase 2: same `session_id` always paints the same hue across lines and across process restarts; each provider has a distinct fixed color; piping stderr to a file or `| cat` yields zero escape codes while `OMNI_LOG_COLOR=always` forces them on; original module unit tests ported (palette-in-range, same-value-same-color, `finish_reason` styles, unknown-fields-uncolored, `NO_COLOR`-disables) plus one new test for the fixed provider colors.

### Suggested commit shape (from the handoff)

- Phase 1: one commit — "Attach request/session span to every log line via middleware".
- Phase 2: one commit — "Colorize correlation IDs and provider in the stderr log".
- Neither modifies any wire-facing test fixture.
