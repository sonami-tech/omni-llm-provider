# omni-llm-provider

OpenAI-compatible Rust proxy that serves Anthropic Claude, xAI Grok, and Codex
from one `omni` server binary. Provider-specific protocol, credential, and
fingerprint logic remains isolated in provider crates.

## Binary

- `omni` - the only server binary. Routes by canonical upstream model id
  (`claude-sonnet-4-6`, `grok-4.3`, or the configured Codex model),
  documented shorthand alias (`sonnet`, `opus`, `haiku`, `fable`, `grok`,
  `composer`, `codex`), or optional provider prefix (`claude:...`,
  `grok:...`, `codex:...`) when a caller needs to force a provider.

## Crates

- `crates/omni-core` - canonical types and the `LlmProvider` trait.
- `crates/omni-common` - OpenAI-compatible HTTP conversion and SSE framing,
  Responses conversion, auth middleware, persistent stats, replacements, error
  envelope, session derivation.
- `crates/provider-claude` - Claude-specific fingerprint profiles, cch,
  credentials, Anthropic Messages translation, identity injection, wire defaults,
  and model catalog.
- `crates/provider-grok` - xAI Grok provider and OpenAI-compatible xAI wire
  mapping.
- `crates/provider-codex` - Codex configuration-backed provider, Codex/OpenAI
  auth resolution, and Responses wire mapping.
- `crates/bin/omni` - server setup, routing, auth, stats, and model catalog
  aggregation.

## HTTP Surface

- `POST /v1/chat/completions` - non-stream JSON or `stream:true` OpenAI SSE
  chunks terminated by `data: [DONE]`. Codex supports non-streaming on this
  route; Codex `stream:true` is rejected until native Responses SSE is added.
- `POST /v1/responses` - supported OpenAI Responses subset, non-stream JSON or
  Responses SSE events. Codex supports non-streaming on this route; Codex
  `stream:true` is rejected until native Responses SSE is added.
- `POST /v1/messages` - native Anthropic Messages inbound for Claude models
  only. This bypasses canonical OpenAI framing and forwards Anthropic JSON/SSE
  through the Claude provider's fingerprint path.
- `POST /v1/messages/count_tokens` - native Anthropic token counting for Claude
  models only.
- `GET /v1/models`, `GET /models` - provider-owned canonical model catalogs.
  Shorthand aliases are accepted on requests but are not emitted as model ids.
- `GET /stats` - persistent request, token, and error counters.
- `GET /health`, `GET /`.

## Build, Run, Test

```bash
cargo build --workspace
cargo run -p omni -- --version
cargo run -p omni -- --providers claude,grok,codex --no-auth --port 18321
cargo test --workspace
```

Useful server flags:

- `--version` prints the Omni binary version and exits.
- `--providers claude,grok,codex` / `OMNI_PROVIDERS`
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

If `--stats-db` is omitted, Omni writes stats to a fixed temp-file path
(`omni-stats.redb` under the OS temp directory). Use `--stats-db` for durable
stats or when running more than one server instance.

`OMNI_API_KEYS` enables bearer-token auth when set to a comma-separated key list.
On startup, Omni logs its banner and current package version before serving
requests.

Current shorthand aliases are resolved from provider-owned catalogs at startup:

- `sonnet` -> `claude-sonnet-4-6`
- `opus` -> `claude-opus-4-8`
- `haiku` -> `claude-haiku-4-5-20251001`
- `fable` -> `claude-fable-5`
- `grok` -> `grok-4.3`
- `composer` -> `grok-composer-2.5-fast`
- `codex` -> the current Codex model from `$CODEX_HOME/config.toml` or
  `~/.codex/config.toml`, falling back to the provider default

Credentials are read fresh per request, never cached. Claude reads
`~/.claude/.credentials.json` or `$CLAUDE_CREDENTIALS_PATH`. Grok resolves
`$XAI_CREDENTIALS_PATH`, then `~/.xai/.credentials.json`, then
`~/.grok/auth.json`. Codex reads `$CODEX_HOME` or `~/.codex` config and auth
state per request.

Custom upstream endpoint overrides are explicit and isolated from default
credentials:

- Claude: `ANTHROPIC_BASE_URL` switches Claude to a custom Anthropic-compatible
  gateway. `ANTHROPIC_AUTH_TOKEN` sends `Authorization: Bearer ...`; otherwise
  `ANTHROPIC_API_KEY` sends `x-api-key`. `ANTHROPIC_CUSTOM_HEADERS` accepts one
  `Name: value` header per line. In this mode Omni does not read or send the
  local Claude OAuth token.
- Grok: `GROK_MODELS_BASE_URL` switches Grok to a custom OpenAI-compatible
  endpoint. `XAI_API_KEY` sends `Authorization: Bearer ...`; if it is unset,
  no Authorization header is sent. In this mode Omni does not read or send
  `$XAI_CREDENTIALS_PATH`, `~/.xai`, or `~/.grok` credentials.
- Codex: Codex custom providers come from Codex config. A custom provider's
  `[model_providers.<name>.auth] command`, `experimental_bearer_token`, or
  `env_key` owns auth for that provider and does not fall back to OpenAI auth
  unless `requires_openai_auth = true`.

Inbound compatibility:

| Inbound API surface | Claude backend | Grok backend | Codex backend |
|---|---:|---:|---:|
| OpenAI `/v1/chat/completions` | Yes | Yes | Non-stream |
| OpenAI `/v1/responses` | Yes | Yes | Non-stream |
| Anthropic `/v1/messages` | Yes | No | No |
| Anthropic `/v1/messages/count_tokens` | Yes | No | No |

Anthropic inbound is intentionally provider-native. It routes only to Claude;
use OpenAI-compatible inbound surfaces for Grok and Codex.

Provider maintenance docs live under `docs/providers/`. Live provider tests are
explicitly opt-in:

```bash
OMNI_LIVE_TESTS=1 cargo test --workspace
```

Do not enable `OMNI_LIVE_TESTS` in normal CI or shell profiles; live tests may
spend provider quota and depend on account state.
