# omni-llm-provider

OpenAI-compatible Rust proxy that serves Anthropic Claude, xAI Grok, and Codex
from one `omni` server binary. Provider-specific protocol, credential, and
fingerprint logic remains isolated in provider crates.

## Gateway philosophy

Omni is a **many-to-many gateway**: any supported client shape to any supported
provider, with a minimal gateway surface. Clients should not need a third
request dialect beyond the inbound APIs Omni already speaks (canonical types
are internal only).

1. **Many-to-many.** One endpoint, multiple providers and inbound interfaces
   (OpenAI chat, Responses, Anthropic Messages, and related paths).
2. **Pass through or translate.** Fields that change the end result must not be
   silently stripped. Prefer pass-through when the wire allows it; otherwise
   translate into the provider's native form. Silent omit of meaningful client
   intent is a bug.
3. **Minimal surface.** Defaults exist only for compatibility when the client
   omits a value, and only in the smallest form that keeps the request valid.
   Avoid maintenance-heavy defaults that go stale.
4. **Client intent first.** Precedence: explicit client value → minimal
   compat/capture default when absent → absent. Fingerprint or capture fidelity
   must not win over meaningful client intent (including Door-2 honor-client
   cases; see #22).
5. **Fail loud when unmappable.** If a backend cannot honor meaningful client
   intent (a value that changes the end result), return a clear error rather
   than silent drop (aligned with the reasoning-effort contract; see below and
   issues #20 / #24). Documented protocol-lossy gaps on translated paths stay
   listed in [`anthropic-compat.md`](anthropic-compat.md); those are explicit
   exceptions, not a license to strip intent.

Design narrative: [`DESIGN.md`](DESIGN.md). Decision record:
[`decisions.md`](decisions.md). Prompt-cache translation (shipped):
[`cache-translation.md`](cache-translation.md).

## Binary

- `omni` - the only server binary. Routes by canonical upstream model id
  (`claude-sonnet-5`, `grok-4.6`, or the configured Codex model),
  documented shorthand alias (`sonnet`, `opus`, `haiku`, `fable`, `grok`,
  `gpt`), or optional provider prefix (`claude:...`,
  `grok:...`, `codex:...`) when a caller needs to force a provider.

## Crates

- `crates/omni-core` - canonical types and the `LlmProvider` trait.
- `crates/omni-common` - OpenAI-compatible HTTP conversion and SSE framing,
  Responses conversion, auth middleware, persistent stats, replacements, error
  envelope, session derivation.
- `crates/provider-claude` - Claude-specific fingerprint pin, no-cch billing
  header, credentials, Anthropic Messages translation, identity injection,
  wire defaults, and model catalog.
- `crates/provider-grok` - xAI Grok provider and OpenAI-compatible xAI wire
  mapping.
- `crates/provider-codex` - Codex configuration-backed provider, Codex/OpenAI
  auth resolution, and Responses wire mapping.
- `crates/bin/omni` - server setup, routing, auth, stats, and model catalog
  aggregation.

## HTTP Surface

- `POST /v1/chat/completions` - non-stream JSON or `stream:true` OpenAI SSE
  chunks terminated by `data: [DONE]`.
- `POST /v1/responses` - supported OpenAI Responses subset, non-stream JSON or
  Responses SSE events.
- `POST /v1/messages` - dual-mode Anthropic Messages inbound. Claude models use
  the native fingerprint path; Grok/Codex models use the translated path (see
  `docs/anthropic-compat.md`).
- `POST /v1/messages/count_tokens` - native Anthropic token counting for Claude
  models only.
- `GET /v1/models`, `GET /models` - provider-owned canonical model catalogs.
  Shorthand aliases are accepted on requests but are not emitted as model ids.
- `GET /stats` - persistent request, token, and error counters (human text
  auto-refreshes every 5s in browsers via the `Refresh` header).
- `GET /health`, `GET /`.

File input uses official Chat `file` and Responses `input_file` parts. Responses
file cache breakpoints stay on the file block. Codex emits them on the file;
Claude maps them to `cache_control`; Grok uses native prefix caching. Codex
accepts file URL, data, and provider-local ID; Claude accepts PDF URL or PDF data;
Grok Responses
accepts URL or provider-local ID on agentic-capable models. Grok Chat rejects
files. Unsupported file variants return a request error instead of losing the
attachment. Omni does not upload files or translate IDs between providers.

Current client compatibility gaps and priority notes are tracked in
[`compatibility-gaps.md`](compatibility-gaps.md).
The go-forward implementation tracker is
[`compatibility-roadmap.md`](compatibility-roadmap.md).
Current per-provider support status is tracked in
[`compatibility-matrix.md`](compatibility-matrix.md).
Cross-provider consolidation and simplification opportunities, plus the approved
observability track (correlation-ID logging + colorized logs), are catalogued in
[`consolidation-2026-07-02.md`](consolidation-2026-07-02.md).

## Build, Run, Test

```bash
cargo build --workspace
cargo run -p omni -- --version
cargo run -p omni -- --no-auth --port 18321
cargo test --workspace
```

Useful server flags:

- `--version` prints the Omni binary version and exits.
- `--providers claude,grok,codex` / `OMNI_PROVIDERS` overrides auto-detection
- `--port 18321` / `OMNI_PORT`
- `--bind 127.0.0.1` / `OMNI_BIND`
- `--public` / `OMNI_PUBLIC` for `0.0.0.0`
- `--stats-db <path>` / `OMNI_STATS_DB`
- `--log-conversations` / `OMNI_LOG_CONVERSATIONS`
- `--log-file <path>` / `OMNI_LOG_FILE`
- `--log-dir <path>` / `OMNI_LOG_DIR`
- `--log-max-bytes <n>` / `OMNI_LOG_MAX_BYTES`
- `--log-backups <n>` / `OMNI_LOG_BACKUPS`
- `--no-auth` / `OMNI_NO_AUTH`
- `--strict-cloud-fidelity` / `OMNI_STRICT_CLOUD_FIDELITY` (default off)
- `--anthropic-auth-scheme api-key|oauth` / `OMNI_ANTHROPIC_AUTH_SCHEME`
  (default `api-key`; only enforced when strict cloud fidelity is on)

If `--stats-db` is omitted, Omni writes stats to `omni-stats.redb` in the
process current working directory. Use `--stats-db` when running more than one
server instance (each needs its own file).

`OMNI_API_KEYS` enables gateway auth when set to a comma-separated key list.
Clients send the key as `Authorization: Bearer <key>`; on the native Anthropic
paths (`/v1/messages`, `/v1/messages/count_tokens`) the key is also accepted via
`x-api-key: <key>`, so stock Anthropic SDKs work unchanged.
Sending both a non-empty `x-api-key` and a non-empty `Authorization: Bearer` on
those Anthropic paths is always rejected as ambiguous credentials (HTTP 400).
With `--strict-cloud-fidelity`, Anthropic paths also require the configured
single-header scheme, and `/v1/chat/completions` enforces OpenAI token-cap field
shape for known model families (`o1`/`o3`/`o4` vs `gpt-*`).
On startup, Omni logs its banner and current package version before serving
requests.

When `--providers` / `OMNI_PROVIDERS` is omitted or empty, Omni enables all
locally detected providers. Detection checks Claude credentials,
`OMNI_CLAUDE_BASE_URL`, or `ANTHROPIC_BASE_URL`; Grok credentials,
`OMNI_GROK_BASE_URL`, or `GROK_MODELS_BASE_URL`; and Codex config/auth under
`$CODEX_HOME` / `~/.codex`, `CODEX_API_KEY`, `OPENAI_API_KEY`,
`CODEX_ACCESS_TOKEN`, or `OMNI_CODEX_BASE_URL`.

Current shorthand aliases are resolved from provider-owned catalogs at startup:

- `sonnet` -> `claude-sonnet-5`
- `opus` -> `claude-opus-5`
- `haiku` -> `claude-haiku-4-5-20251001`
- `fable` -> `claude-fable-5-1`
- `grok` -> `grok-4.6`
- `gpt` -> the current Codex model from `$CODEX_HOME/config.toml` or
  `~/.codex/config.toml`, falling back to the provider default

Credentials are read fresh per request, never cached. Omni refreshes
Claude/Codex/Grok OAuth tokens in-place by default (atomic write-back of
rotated refresh tokens); disable with `--no-oauth-refresh` or
`OMNI_OAUTH_REFRESH=0`. After a default-path upstream 401, Omni force-refreshes
once and replays that inference request once if the new token is live
(issue #31). It does not retry 5xx or mid-stream model errors. Recovery that
cannot produce a live token fails closed. Claude reads
`~/.claude/.credentials.json` or `$CLAUDE_CREDENTIALS_PATH`. Grok resolves
`$XAI_CREDENTIALS_PATH`, then a usable `~/.xai/.credentials.json`, then
`~/.grok/auth.json`. Codex reads `CODEX_API_KEY`, `OPENAI_API_KEY`,
`CODEX_ACCESS_TOKEN`, or `$CODEX_HOME` / `~/.codex` config and auth state per
request.

Custom upstream endpoint overrides are explicit and isolated from default
credentials:

- Claude forced override: `OMNI_CLAUDE_BASE_URL` switches Claude to a custom
  Anthropic-compatible gateway and wins over `ANTHROPIC_BASE_URL`.
  `OMNI_CLAUDE_AUTH_TOKEN` sends `Authorization: Bearer ...`; otherwise
  `OMNI_CLAUDE_API_KEY` sends `x-api-key`. `OMNI_CLAUDE_CUSTOM_HEADERS`
  accepts one `Name: value` header per line. In this mode Omni does not read or
  send the local Claude OAuth token or any `ANTHROPIC_*` auth variables.
- Claude: `ANTHROPIC_BASE_URL` switches Claude to a custom Anthropic-compatible
  gateway. `ANTHROPIC_AUTH_TOKEN` sends `Authorization: Bearer ...`; otherwise
  `ANTHROPIC_API_KEY` sends `x-api-key`. `ANTHROPIC_CUSTOM_HEADERS` accepts one
  `Name: value` header per line. In this mode Omni does not read or send the
  local Claude OAuth token.
- Grok forced override: `OMNI_GROK_BASE_URL` switches Grok to a custom
  OpenAI-compatible endpoint and wins over `GROK_MODELS_BASE_URL`.
  `OMNI_GROK_AUTH_TOKEN` sends `Authorization: Bearer ...`; otherwise
  `OMNI_GROK_API_KEY` sends `Authorization: Bearer ...`.
  `OMNI_GROK_CUSTOM_HEADERS` accepts one `Name: value` header per line. In this
  mode Omni does not read or send `XAI_API_KEY`, `$XAI_CREDENTIALS_PATH`,
  `~/.xai`, or `~/.grok` credentials.
- Grok: `GROK_MODELS_BASE_URL` switches Grok to a custom OpenAI-compatible
  endpoint. `XAI_API_KEY` sends `Authorization: Bearer ...`; if it is unset,
  no Authorization header is sent. In this mode Omni does not read or send
  `$XAI_CREDENTIALS_PATH`, `~/.xai`, or `~/.grok` credentials.
- Codex forced override: `OMNI_CODEX_BASE_URL` switches Codex to a custom
  Responses-compatible endpoint and wins over Codex config base URLs.
  `OMNI_CODEX_MODEL` controls the catalog and the `gpt` alias, falling
  back to the Codex config model or provider default. `OMNI_CODEX_AUTH_TOKEN`
  sends `Authorization: Bearer ...`; otherwise `OMNI_CODEX_API_KEY` does.
  `OMNI_CODEX_CUSTOM_HEADERS` accepts one `Name: value` header per line.
  `OMNI_CODEX_WIRE_API` currently supports `responses`. In this mode Omni does
  not read or send Codex/OpenAI native auth.
- Codex: Codex custom providers come from Codex config. A custom provider's
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, or
  `env_key` owns auth for that provider and does not fall back to OpenAI auth
  unless `requires_openai_auth = true`.

Inbound compatibility:

| Inbound API surface | Claude backend | Grok backend | Codex backend |
|---|---:|---:|---:|
| OpenAI `/v1/chat/completions` | Yes | Yes | Yes |
| OpenAI `/v1/responses` | Yes | Yes | Yes |
| Anthropic `/v1/messages` | Yes (native) | Yes (translated) | Yes (translated) |
| Anthropic `/v1/messages/count_tokens` | Yes | No | No |

Anthropic inbound is dual-mode: Claude is native; Grok/Codex are translated.
`count_tokens` remains Claude-only. Full notes: `docs/anthropic-compat.md`.

### Reasoning effort (issue #20)

Chat (`reasoning_effort` top-level, or nested `reasoning.effort`) and Responses
(`reasoning.effort`) accept free-string effort with **lexical hygiene only**
(non-empty, max 32 chars, charset `A-Z a-z 0-9 _ -`). There is no global
valid-values allowlist at the edge, so levels such as `xhigh` reach canonical.
Empty strings fail at the edge; JSON `null` means absent.

Catalog-advertised effort lists are **discovery-only** and never reject a
request. Adapters map known aliases or **fail loud** with a structured
`unsupported reasoning_effort` error (HTTP 400) when the upstream wire cannot
express the value (no silent omit of explicit client effort).

Adapter mapping (summary):

| Backend | Maps / keeps | `"none"` | Fails loud (examples) |
|---|---|---|---|
| **Grok** | `low`/`medium`/`high`; `xhigh` on `grok-4.6`; `minimal`→`low`, `max`→`high` | omits field | `xhigh` on `grok-4.5`, `ultra`, other unknowns |
| **Codex** | `none`/`minimal`/`low`/`medium`/`high`/`xhigh`; `max`→`high` | emits `none` | `ultra`, other unknowns |
| **Claude** | client effort wins; pin default only when absent | suppresses pin | unmappable without `output_config.effort` (e.g. Haiku + `xhigh`) |

Claude precedence: **client effort > fingerprint pin default > absent**.
Effort-capable models prefer `output_config.effort` over effort-derived thinking
budgets.

**Grok omit (issue #18):** When the client omits effort, Omni omits the upstream
field too. The provider/model default then applies (often `high` on grok-4.5).
No force-floor and no invented disable in this scope; clients that need a
specific level must set it explicitly.

**Claude free-string (issue #26):** On the OpenAI chat/Responses → Claude path,
effort-capable models pass free-string client effort through to
`output_config.effort` (only `minimal`→`low` is remapped; `max` and other names
are kept as-is, unlike Grok/Codex `max`→`high`). Explicit `"none"` still
suppresses pin fill and does not set `output_config.effort`. There is no closed
local Claude effort allowlist. Unknown names may 400 from upstream by design.
Fail loud only when the adapter cannot express the value (for example Haiku +
`xhigh`, where the thinking-budget ladder has no mapping).

Design notes: `docs/DESIGN.md`. Decision record: `docs/decisions.md`.

Provider maintenance docs live under `docs/providers/`. Live provider tests are
explicitly opt-in:

```bash
OMNI_LIVE_TESTS=1 cargo test --workspace
```

Do not enable `OMNI_LIVE_TESTS` in normal CI or shell profiles; live tests may
spend provider quota and depend on account state.

### Live HTTP suite (running process, issue #15)

Opt-in Python suite that targets a **running** omni instance. It does **not**
spawn omni, does **not** run under `cargo test`, and is **not** part of CI.
Default base URL is `http://127.0.0.1:18321` (override with `--base-url` or
`OMNI_BASE_URL`).

Start omni yourself first (example):

```bash
cargo run -p omni -- --no-auth --port 18321
```

Then run the suite from the repo root:

```bash
python3 -m tools.live_http_suite
python3 -m tools.live_http_suite --base-url http://127.0.0.1:19001
python3 -m tools.live_http_suite --dual-mode-off   # skip dual-mode Anthropic edge
```

Hermetic unit tests for oracles and retry policy (no network):

```bash
python3 -m unittest tools.live_http_suite.tests.test_hermetic -v
```

Model pins (override via env): `OMNI_TEST_CLAUDE_MODEL` (default
`claude-haiku-4-5-20251001`), `OMNI_TEST_DUAL_MODE_MODEL` (default `grok-4.5`),
`OMNI_TEST_RESPONSES_MODEL` (default Claude haiku pin). The suite fails loud if
a required pin is missing from `GET /v1/models`. Dual-mode skip is only via
`OMNI_TEST_DUAL_MODE_OFF=1` or `--dual-mode-off`, checked before resolving the
dual-mode pin (never inferred from 4xx).

Other knobs: `OMNI_TEST_HTTP_TIMEOUT_S` (per-attempt wall-clock seconds, default
60; retries reset the deadline), `--only NAME…`, `--list`. The suite sends no
gateway credentials; start omni with `--no-auth` (or an empty key set). Zero-pass
and all-skip runs exit non-zero. Prefer multi-provider omni so unknown-model
tests are rejected at the gateway (single-provider mode can pass unknown ids
through to upstream).
