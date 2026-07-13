# Plan: Anthropic Frontend → Grok / Codex Backends in Omni

**Status:** Implemented (2026-07-12)  
**Date:** 2026-07-12  
**Adversarial review:** Opus + GPT-5.6-sol + Grok-4.5 @ high (multi-round)  
**Compat note:** `docs/anthropic-compat.md`

---

## Compact summary (read this first after context compact)

| Item | Decision |
|---|---|
| **Goal** | Anthropic clients (`/v1/messages`) can use **Grok** or **Codex** backends |
| **Claude** | Unchanged **native** passthrough (fingerprint/cch); original body **bytes** |
| **Grok/Codex** | Anthropic → **Canonical** → `LlmProvider` → Anthropic JSON/SSE |
| **“OpenAI” v1** | **Codex provider** only (no new `openai` provider) |
| **Code** | `omni-common/src/anthropic.rs`; dual-mode dispatch in `bin/omni`; `is_error` wire in grok/codex |
| **PRs** | PR1–4 landed in implementation train |
| **Status** | Implemented; hermetic mapper + dual-mode route tests green |

**Pipeline:** auth → body bytes → peek model → shared resolver (same as chat) → branch claude native **or** translate.

**Hard rules:** tool ID exact round-trip; tool results before user text; no thinking on translated wire; stream success only on clean completion; gateway 401 vs upstream 502; non-Claude `count_tokens` = 400.

Full contracts below (§4–§11). Do not re-litigate settled defaults in §13 unless implementation is blocked.

---

**Goal (full):** Make `POST /v1/messages` a first-class Omni frontend that can route to **Grok** and **Codex** (OpenAI-compat via Codex provider / custom endpoints), not only Claude.  
**Scope note:** “OpenAI backend” in product language means the existing **`codex` provider** (and Grok custom OpenAI-compat endpoints). No fourth `openai` provider in v1.

**Non-goal for v1:** Perfect Anthropic wire fidelity on non-Claude backends. Claude-native fingerprint/cch path must remain intact and isolated.

---

## 1. Problem statement

Today Omni already does “anything OpenAI → any provider” via:

```
OpenAI Chat/Responses → to_canonical → LlmProvider (claude|grok|codex) → from_canonical
```

Anthropic inbound is deliberately Claude-only (`docs/DESIGN.md`, `docs/decisions.md`):

- `/v1/messages` and `/v1/messages/count_tokens` hard-require Claude.
- Path uses Claude **native passthrough** (fingerprint/cch).
- Explicit non-goal: “Do not emulate Anthropic inbound for non-Claude providers.”

**New product goal:** Anthropic-protocol clients (Claude Code, Anthropic SDKs) can point at Omni and select Grok or Codex as backend via model id / prefix, with best-effort protocol translation through **canonical** (not via OpenAI wire as an intermediate hop).

---

## 2. Success criteria (v1)

1. **Anthropic → Grok:** multi-turn text + tools + images work through Anthropic-shaped request/response (stream and non-stream).
2. **Anthropic → Codex:** same via `codex:` models.
3. **Claude path unchanged:** native fingerprint passthrough; golden/hermetic Claude Anthropic tests unchanged in behavior; raw Claude body not run through Anthropic→canonical.
4. **Fail loud:** no silent wrong tools/images; documented drops only for listed lossy fields.
5. **Docs flipped** to dual-mode Anthropic inbound.
6. **Hermetic tests** for mappers, framer, routing, tool-loop round-trip; live opt-in.

---

## 3. Key design: dual-mode Anthropic handler

| Resolved provider | Path |
|---|---|
| **claude** | **Native passthrough** (today). Zero Anthropic→canonical. |
| **grok** or **codex** | **Translated:** Anthropic → Canonical → `LlmProvider` → Canonical → Anthropic response/SSE |
| missing / ambiguous | Anthropic-shaped 400 |

Rejected: always-translate Claude; CLIProxyAPI sidecar; Anthropic methods on `LlmProvider` for all providers.

---

## 4. Architecture and hard invariants

### 4.1 Pipeline order (mandatory)

```
auth (gateway)
  → read raw request body **bytes** (retain for Claude arm)
  → peek model string only for routing (no body re-encode; do not add Claude-path
     rejections Anthropic would not apply)
  → shared model resolver — **model identity only** (SAME function as chat)
  → BRANCH on provider id
       ├─ claude: forward **original body bytes** into prepare_anthropic_messages /
       │          native send only (no translate filters, no duplicate-key gate)
       └─ grok|codex: strict parse (reject duplicate top-level keys) →
                 anthropic_to_canonical → LlmProvider → anthropic framer
```

**Invariants:**

1. **Branch-before-Claude-prepare:** Claude arm never calls `anthropic_to_canonical`. Translated arm never calls `prepare_anthropic_messages` / cch / fingerprint.
2. **Claude body byte-immutability:** Claude arm receives the **original request body bytes**. Model extraction must not re-encode that body (key order/whitespace/number formatting can break cch).
3. **Shared resolver = model identity only:** one function for chat + Anthropic model string resolution. Anthropic-only content rules (prefill, document, mid-system) live in the translated arm after branch, never in the shared resolver.
4. **Stats:** `stats_model_key(provider, canonical_model)` on both arms; never attribute Grok/Codex usage to `claude`.

### 4.2 Code placement

| Piece | Crate |
|---|---|
| `anthropic_to_canonical(body, provider_id)` | `omni-common` (provider_id selects extras allowlist: stop_sequences, parallel_tool_calls) |
| `canonical_to_anthropic` + SSE framer | `omni-common` |
| Dual-mode dispatch | `crates/bin/omni` |
| Providers | unchanged; canonical-only |

### 4.3 Auth

Gateway auth on `/v1/messages` unchanged. Upstream credentials per provider.  
**Error split:**

| Failure | Anthropic HTTP + error type |
|---|---|
| Gateway key missing/invalid | **401** `authentication_error` |
| Client bad request / unsupported / invalid model on Omni | **400** `invalid_request_error` |
| Upstream **4xx** (bad params, unknown model, payload) | **400** `invalid_request_error` |
| Upstream **429** | **429** `rate_limit_error` |
| Upstream auth / missing Omni-held provider credentials | **502** `api_error` (**never 401**) |
| Upstream **5xx** / transport / protocol | **502** `api_error` |

**Status mapper:** this table only on translated Anthropic — **do not** call the chat status mapper. Body always Anthropic.

**Stream open order (mandatory):** (1) resolve + `anthropic_to_canonical` (2) open upstream / first successful acceptance of the stream (3) **only then** HTTP 200 + `message_start` (4) pre-start failures use the HTTP table above (5) post-`message_start` failures → SSE `error` only.

---

## 5. Request translation: Anthropic → Canonical

### 5.0 Content shapes (common Anthropic)

| Shape | Rule |
|---|---|
| Message `content` string | Treat as single text block |
| Message `content` array | Walk blocks |
| `tool_result.content` string | Pass as tool result text |
| `tool_result.content` array of text | Join with `\n` |
| `tool_result.content` with image/other | **400** |
| Block-by-role matrix | user: text, image, tool_result only; assistant: **text, tool_use only** (no images; thinking dropped before matrix); illegal role+type → **400** |
| Unpaired `tool_result` (no prior outstanding tool_use id) | **400** |
| Assistant `tool_use` in history | non-empty unique `id`, non-empty `name`, `input` object (or empty object) — else **400** |

### 5.1 Closed decisions (no longer open)

| Topic | v1 rule |
|---|---|
| Top-level `system` string | One system message with that text |
| Top-level `system` array of text blocks | Join with `\n` into one system message |
| Mid-conversation `role: "system"` | **400** on translated path |
| **Validation order** | (1) parse → (2) prefill check (last message assistant → 400) → (3) drop thinking → (4) strip empty messages → (5) **re-check:** conversation non-empty and **last message is user** (else 400; catches empty trailing user stripping into prefill) → (6) tool-pairing / id checks. |
| Historical `thinking` / `redacted_thinking` | **Drop blocks**; remove empty messages after; if adjacent same-role messages result (e.g. user→user), **400** fail-loud (do not silently merge) |
| User text interleaved with tool_results in one user message | **Documented lossy:** emit all tool_results first, then trailing user text (OAI adjacency). |
| Anthropic image → canonical | `source.type=base64` → Base64{media_type,data}; `source.type=url` → Url{url}; other source types → **400**; same size/MIME policy as OpenAI chat |
| Trailing assistant **prefill** | **400** (checked before thinking strip) |
| Unknown **content block** types | **400** |
| Unknown **top-level** keys | **Ignore** (forward-compatible) |
| `temperature`, `top_p` | Map to canonical (same as OpenAI chat path) |
| `top_k` | **Drop** (documented lossy; no canonical field) |
| `metadata` | Session only (`user_id` → session); not upstream passthrough |
| `stream` | Handler control only; never a canonical/provider extra |
| `cache_control` | Drop (documented) |
| `document` blocks | **400** |
| `stop_sequences` | Grok: map to extras `stop` if allowlisted; **Codex: drop + log**. **Response:** always `stop_sequence: null`; stop-sequence finishes collapse to `end_turn` (documented lossy — matched sequence not available from all backends). |
| `max_tokens` | **Required** on translated path (missing → 400). Pass through value; no silent clamp (provider may still 400 on oversize). |
| `model`, `messages` | **Required** (missing/empty messages → 400) |
| `thinking` request config | Map to `CanonicalReasoning` |
| Tools without plain `name` + `input_schema` (hosted/server tools, computer use, etc.) | **400** unsupported tool kind |
| `tool_choice.disable_parallel_tool_use` | Map to provider extras if allowlisted (`parallel_tool_calls: false` style for Grok/Codex when supported); else drop + log |
| Response thinking blocks | **Do not emit** on translated path (v1) |

### 5.2 Tool history rewrite (critical)

Anthropic embeds `tool_result` **inside user messages**. Canonical/OpenAI expect **separate tool-result messages**.

**Deterministic rewrite:**

1. Walk messages in order.
2. For an assistant message: map **text** only (+ tool_use); **assistant images → 400**. Map each `tool_use` with non-empty unique id, non-empty name, object input.
3. **After any assistant turn that contains ≥1 tool_use:** the **immediately next** message must be a **user** message that resolves **every** outstanding id from that assistant turn **exactly once**.  
   - Intervening turns / missing / extra / unknown / duplicate / deferred results → **400**  
   - Unpaired tool_result anywhere → **400**
4. For that resolving user message (lossy reorder, documented):
   - Emit **all** tool-result turns **first**, then **one** user message for non-tool text/image.  
   - Each `tool_result`: exact id; string or text-array content; image → **400**.  
   - **`is_error` encoding (normative, end-to-end):**  
     1. Anthropic → `CanonicalBlock::ToolResult { is_error }`  
     2. Grok/Codex request wire: OpenAI `role:tool` message content string =  
        - `is_error && content.empty` → `"error"`  
        - `is_error && !content.empty` → `"ERROR: " + content`  
        - `!is_error` → content as-is  
     3. Hermetic tests assert the **provider-bound JSON** contains that content string (not merely that the canonical struct has the bit).

### 5.3 Tool ID round-trip invariant (critical)

| Direction | Rule |
|---|---|
| Upstream tool call from Grok/Codex | Emit Anthropic `tool_use.id` = **exactly** the backend `tool_call_id` (**no sanitize, no synthesize, no rewrite**). |
| Next request `tool_result.tool_use_id` | Pass through **unchanged**. |
| Incomplete backend tool call (missing id, missing name, empty name, duplicate ids, conflicting index metadata) | **Upstream protocol error** (502 / SSE `error`) — never emit a partial tool_use and never invent an id. |
| Test | Full loop id match; incomplete-call fixtures fail loud. |

### 5.4 Tools and tool_choice

| Anthropic | Canonical |
|---|---|
| tools input_schema | parameters; ensure object `properties` default `{}` if missing type object |
| tool_choice auto/any/tool/none | Auto / Required / Specific{name} / None |
| Specific name | Must survive to Grok/Codex forced-function fields — **e2e hermetic test required** |

### 5.5 Images

Reuse OpenAI chat image ingress policy (URL + base64). Same size/MIME rules. **No new server-side fetch** beyond what chat already does. Unsupported → Anthropic 400.

---

## 6. Response translation: Canonical → Anthropic

### 6.1 Non-stream message envelope (normative fields)

```json
{
  "id": "<canonical or msg_…>",
  "type": "message",
  "role": "assistant",
  "model": "<stripped model id>",
  "content": [ /* blocks */ ],
  "stop_reason": "end_turn|tool_use|max_tokens|…",
  "stop_sequence": null,
  "usage": { "input_tokens": 0, "output_tokens": 0 }
}
```

- `content[]`:
  - **No `thinking` blocks**
  - text if non-empty
  - each tool_call → `tool_use` with exact backend id/name
  - **Tool arguments policy (both stream and non-stream):**
    - empty or missing args string → **`input: {}`** (valid zero-arg tools)
    - non-empty string must parse as a **JSON object**; otherwise **upstream protocol error** (502 / stream `error`) — never coerce garbage to `{}`
- Tool-only turn: only tool_use blocks (valid).
- **Finish classification (normative; apply in this priority order):**

| Priority | Condition | Outcome |
|---|---|---|
| 1 | content_filter, refusal, error, cancel, incomplete tools, invalid tool args | **error** (dominates; do not execute tools from filtered/cancelled turns) |
| 2 | finish `length` | success `max_tokens` |
| 3 | (finish `tool_calls`/`function_call` **and** ≥1 complete tool_use) **or** (≥1 complete tool_use + ordinary stop) | success `tool_use`; `tool_calls` finish with **zero** complete tools → **error** |
| 4 | ordinary stop/end, no tools | success `end_turn` |
| 5 | stream closed without any Finish | **error** only if canonical layer did not synthesize Finish |

**Canonical invariant:** synthesize Finish(`stop`) only on **explicit successful stream completion** (provider `[DONE]` / clean `send_stream` end after a normal response). **Bare transport EOF / truncated body without `[DONE]` must not synthesize Finish** — those are SSE `error` (false-success ban).

- **Empty content:** no text/tools + success finish class → one empty text block + mapped stop_reason; no text/tools + error class → error envelope.
- `usage`: input/output; cache when non-zero.

### 6.2 Streaming SSE state machine (required spec for PR2)

State:

- `next_index`, `open_blocks`, `tools` (id, name, started, args, anth_index)
- `saw_tool_use`, `terminal_sent`, `usage_acc`
- `pending_finish: Option<FinishReason>` — Finish does **not** immediately terminalize

Rules:

1. First event: `message_start` with complete message object including **`id`** (canonical or `msg_…`), `type`, `role`, `model`, empty content, null stops, zero usage.
2. Never emit thinking blocks.
3. **Block lifecycle:** for each content block index: `content_block_start` → zero or more deltas → **`content_block_stop`**. When switching kinds (text→tool), stop the open text block before starting a tool block (and vice versa if needed). Tool `content_block_start` payload includes `type: tool_use`, `id`, `name`, `input: {}`.
4. `TextDelta` → ensure text block open; `text_delta`; monotonic `index`.
5. `ToolCallDelta`:
   - Buffer until **non-empty id and non-empty name** both known before start.
   - **No id synthesis.** Incomplete identity at terminalization → error path.
   - Duplicate ids / conflicting index metadata → error path.
   - Start tool_use; set `saw_tool_use`; forward provisional `partial_json`.
   - On block close: empty/missing args → `{}`; non-empty must be JSON object or error path.
6. **Wire order freeze (v1):** emit **all text first**, then tools. Buffer tool deltas until text phase ends (no further TextDelta / on Finish). Then emit tools **sequentially**. Never open a text block after any tool_use has started. **Late TextDelta after tools started:** drop + debug log (documented lossy). Never interleave partial_json across two open Anthropic tool blocks.
7. `Usage`: **replace** with latest full snapshot only — never sum snapshots.
8. On `Finish`: set `pending_finish` + classify. Error class → rule 10 immediately. After first Finish: **ignore further content deltas**; accept only Usage + stream end (or hard error).
9. **Successful terminalization** only when `pending_finish` is success class **and** the provider adapter reports **clean completion** (explicit `[DONE]` / normal stream end — not bare TCP EOF). Abnormal abort always → rule 10. On success close: re-validate tools — incomplete/invalid → **override** to rule 10. Else emit exactly one:
   ```json
   {"type":"message_delta","delta":{"stop_reason":"<…>","stop_sequence":null},"usage":{"input_tokens":N,"output_tokens":M}}
   ```
   then `message_stop`. Usage = last snapshot (never sum).
10. **Error path:** incomplete tools at close, invalid args, error-class Finish, cancel, truncated EOF without `[DONE]`, stream error after start: best-effort `content_block_stop`; SSE `error` frame; never `message_stop`; `record_error`.

### 6.3 Error envelope

Always Anthropic error JSON. **Status from §4.3 only**. After `message_start`, failures are SSE `error` only.

---

## 7. count_tokens

| Provider | Behavior |
|---|---|
| Claude | Existing native |
| Grok / Codex | **400** Anthropic error: token counting unsupported for this backend |

**Ship in the same dual-mode enablement PR as `/v1/messages` routing** (not deferred).

---

## 8. Claude Code notes

- No Claude Code preamble/cch on Grok/Codex path.
- Never flatten multi-block content to a single string; tool_result/text reorder for OAI adjacency is the only documented exception (§5.1).
- Tools + text first; no thinking wire emission on translated path.
- Point clients at `grok:…` / `codex:…` models explicitly when multi-provider enabled.

---

## 9. Docs

1. Flip DESIGN non-goal → dual-mode.
2. Update decisions.md routing.
3. Compatibility matrix: Anthropic-inbound × provider.
4. Short `docs/anthropic-compat.md`: lossy table, count_tokens, tool ID invariant, no thinking on translated path.

---

## 10. Testing

### Hermetic required

1. anthropic_to_canonical: text, multi-block, image, tools, tool_choice Specific, thinking config, system join.
2. Tool history: multi tool_result fan-out, is_error, id preservation.
3. Full tool loop: response tool_use id → next request tool_result → canonical tool_call_id match.
4. canonical_to_anthropic non-stream: text, tool-only, invalid args → error, stop_reason consistency.
5. SSE state machine: parallel tools, text+tools, error after message_start, EOF without Finish, invalid final args.
6. Routing: grok/codex success; claude never calls translate; shared resolver parity with chat; count_tokens 400 for non-Claude.
7. Regression: existing Claude Anthropic tests.

### Live opt-in

Claude Code or curl → Omni → Grok credentials.

---

## 11. PR plan

### PR1 — Request mapper + history rewrite + tests
- `anthropic_to_canonical` with closed decisions in §5.1–5.3.
- Tool id/is_error/fan-out fixtures.
- No routing yet.

### PR2 — Response mapper + SSE state machine + tests
- Non-stream + full framer state machine from §6.2.
- Invalid args / mid-stream error fixtures.
- No routing yet.

### PR3 — Dual-mode enablement (messages + count_tokens)
- Shared resolver; branch-before-prepare; Claude immutability tests.
- Wire Grok + Codex.
- count_tokens 400 for non-Claude.
- Invert old reject-grok tests.
- Stats/logging/error split (401 vs 502).

### PR4 — Docs + matrix
- DESIGN/decisions/matrix/anthropic-compat.md.
- Optional live smoke note.

Feature flag optional: if PR3 must land incrementally, gate translated path behind `OMNI_ANTHROPIC_TRANSLATE=1`; default off until PR2 tests green. Prefer ship enabled when PR1–3 complete in one release train.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Claude fingerprint regression | Branch-before-prepare; immutability tests; no shared translate |
| Tool loops | Explicit ID invariant + loop tests |
| Stream hang / false success | Error terminal path; no happy message_stop on failure |
| Wrong tools | No `{}` on bad JSON; fail loud |
| Scope creep | Codex not generic OpenAI provider; no thinking wire |

---

## 13. Settled product defaults

| Question | Decision |
|---|---|
| OpenAI backend | **Codex provider** (+ existing custom endpoints); no new `openai` provider |
| count_tokens non-Claude | 400, co-ship with dual-mode |
| Historical thinking | Drop; strip empty messages |
| Emit thinking on translate | **No** (v1) |
| Mid system role | 400 |
| Prefill | 400 |
| max_tokens over ceiling | Provider error, no clamp |
| Unknown top-level fields | Ignore |
| Response model field | Stripped id |

---

## 14. Out of scope (v1)

- Computer-use / non-function hosted tools  
- Prompt caching semantics  
- Document/file/audio  
- Thinking signatures / emit thinking on translate  
- Separate OpenAI provider id  
- Changing Claude native fingerprint  
- CLIProxyAPI dependency  

---

## 15. Key decisions

1. Dual-mode Anthropic inbound (native Claude vs translated Grok/Codex).  
2. Canonical bridge; providers stay canonical-only.  
3. Hard pipeline invariants: shared resolver, branch-before-prepare, Claude body isolation.  
4. Tool ID round-trip + history fan-out + is_error are first-class.  
5. Fail-loud tools/images; no silent `{}` args; no false stream success.  
6. No thinking blocks on translated wire.  
7. count_tokens co-ships with enablement.  
8. Codex is the OpenAI-compat backend for v1.

---

## 16. Adversarial review log

**Roster:** Claude Opus @ high, GPT-5.6-sol @ high, Grok-4.5 @ high.

### Round 1 → fixed

Tool ID round-trip; tool_result fan-out + is_error; no silent bad-JSON tools; SSE machine; stop_reason from emitted blocks; system/prefill pins; Specific tool_choice tests; empty content; Claude isolation; no thinking emit; stream error terminalization; auth 401/502 split; count_tokens co-ship; empty-after-drop; shared resolver.

### Round 1 rejected

max_tokens clamp; closed top-level allowlist; OpenAI-vs-Codex scope ambiguity.

### Round 2 → fixed

Tool-result adjacency; ID accounting; temperature/top_p/top_k/metadata/stream maps; message envelope; hosted tools 400; byte-exact Claude body; empty args `{}`; prefill-before-strip; partial_json provisional.

### Round 3 → fixed

Immediate next-user resolves all tool ids; no id synthesis; pending_finish; finish classification; stop_sequence lossy.

### Rounds 4–5 → fixed

Duplicate keys; finish priority + `tool_calls` row; stream close = canonical end not TCP EOF; Finish synthesis; sequential parallel-tool emit; role×block + assistant images 400; unpaired tool_result 400; history tool_use validation; thinking-drop adjacency 400; text/tool_result reorder documented lossy; status §4.3 only; message_delta payload; usage last-wins; mid-stream SSE error frame.

### Round 6 → fixed (after Sol/Grok; Opus was clean)

Full status table (4xx/429/502); stream open-before-message_start; post-normalize last-message-must-be-user; is_error encoding; tool_calls finish requires ≥1 tool; text-then-tools wire order.

### Deferred

Feature flag optional; live smoke; future thinking emit / true stop_sequence echo.

---

## 17. Implementation readiness

Still medium effort, but **not** “mappers only”: history rewrite + SSE state machine are the bulk. PR1–2 must close those contracts before PR3 wires traffic.
