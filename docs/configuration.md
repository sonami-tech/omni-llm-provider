# Configuration Reference

All options can be set via CLI flags or environment variables. CLI flags take precedence over environment variables.

## Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p, --port` | `CCP_PORT` | `18321` | Listen port |
| `-H, --host` | `CCP_HOST` | `127.0.0.1` | Listen address |
| `--data-dir` | `CCP_DATA_DIR` | Platform default | Data directory for stats DB |
| `--api-keys` | `CCP_API_KEYS` | Auto-generated | Comma-separated API keys (min 8 chars each) |
| `--api-keys-file` | `CCP_API_KEYS_FILE` | None | File with one API key per line (`#` comments allowed) |
| `--no-auth` | `CCP_NO_AUTH` | Off | Disable authentication entirely |
| `--replace-rules` | `CCP_REPLACE_RULES` | None | TOML file with text replacement rules |
| `--log-conversations` | `CCP_LOG_CONVERSATIONS` | Off | Log full prompts and responses to stderr |
| `--log-file` | `CCP_LOG_FILE` | None | Write conversation logs to file (implies `--log-conversations`) |
| `--log-dir` | `CCP_LOG_DIR` | None | Write one conversation log file per resolved session id |
| `--log-max-bytes` | `CCP_LOG_MAX_BYTES` | `67108864` | Rotate `--log-file` after this many bytes; `0` disables rotation |
| `--log-backups` | `CCP_LOG_BACKUPS` | `5` | Number of rotated conversation log files to keep |
| `--no-preamble` | `CCP_NO_PREAMBLE` | Off | Skip the canonical Claude Code system identifier preamble; only useful for debugging upstream |
| `-v, --verbose` | | Off | Debug logging |

The `RUST_LOG` environment variable overrides `-v` when set, allowing fine-grained log filtering (for example, `RUST_LOG=claude_code_provider=debug`).

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

Date-suffixed model names are also resolved via substring matching. Unrecognized names fall back to Sonnet.

## Reasoning Effort

| Value | Behavior |
|-------|----------|
| `none` (or absent) | Default behavior, no extended thinking |
| `low` | Light reasoning |
| `medium` | Moderate reasoning |
| `high` | Deep reasoning |
| `max` | Maximum reasoning |

Pass `reasoning_effort` in the request body. CCP maps it to Anthropic Messages API `thinking` settings. Explicit `thinking` in the request body takes precedence.

When thinking is enabled, CCP coerces incompatible Anthropic parameters as needed: `temperature` becomes `1.0`, and `top_p`, `top_k`, and stop sequences are omitted.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `GET /v1/models` | List available models |
| `GET /health` | Server health and active request count |
| `GET /stats` | HTML stats dashboard |
| `GET /stats/json` | Stats as JSON |

All `/v1/*` endpoints also work without the prefix (`/chat/completions`, `/models`), so both `http://host:18321` and `http://host:18321/v1` work as a base URL.

## Authentication

There are three local API authentication modes, in priority order:

1. **No auth** (`--no-auth`) - all requests pass through. Use only on trusted networks.
2. **Explicit keys** (`--api-keys` or `--api-keys-file`) - requests must include `Authorization: Bearer <key>`.
3. **Auto-generated** (default) - a random UUID key is generated on startup and printed to the log.

Keys must be at least 8 characters. Both `--api-keys` and `--api-keys-file` can be used together; keys are deduplicated.

The `/health` and `/stats` endpoints are always accessible without local API authentication.

API key IDs (first 4 + last 4 characters) appear in request logs and the stats dashboard for tracking without exposing full keys.

## Claude OAuth Credentials

CCP talks directly to `api.anthropic.com` using the OAuth subscription token from `~/.claude/.credentials.json`, written by `claude login`. Override the credential path with `CLAUDE_CREDENTIALS_PATH` if the file is mounted somewhere else.

Credentials are read fresh for each request. If Anthropic returns 401, CCP re-reads the file once and retries, which allows refreshed tokens to be picked up without restarting the server.

## OAuth Gate Preamble

For Opus and Sonnet subscription calls, Anthropic expects the first system block to exactly identify the caller as Claude Code. CCP prepends that canonical system identifier by default, then appends any user-provided system/developer content.

Use `--no-preamble` or `CCP_NO_PREAMBLE=true` only for upstream debugging or when the caller supplies an equivalent first system block.

## Stats Persistence

Request statistics are stored in a `redb` database at `<data-dir>/stats.redb`. Stats survive server restarts. Delete this file to reset statistics.

## Text Replacement

Rules are defined in a TOML file passed via `--replace-rules` or `CCP_REPLACE_RULES`. Each rule has a `scope` (`prompt`, `response`, or `both`), a `search` string, and a `replace` string. Example:

```toml
[[rule]]
scope = "prompt"
search = "Acme"
replace = "SomeStringNobodyElseWouldChoose"

[[rule]]
scope = "response"
search = "SomeStringNobodyElseWouldChoose"
replace = "Acme"
```

Additional details:

- Rules are loaded once at startup. Restart the server to pick up changes.
- `Both`-scoped rules are expanded into separate prompt and response rules internally.
- Prompt replacements apply to outbound system text, message text, tool names, tool descriptions, tool schemas, and tool-result content.
- Response replacements apply to assistant text and tool-call names/arguments.
- For streaming responses, replacement is applied per text delta. If a search string spans a delta boundary, it will not be matched.

## Conversation Logging

Three output targets:

- `--log-conversations` logs to stderr, interleaved with server logs.
- `--log-file /path/to/file` logs to a dedicated file (also implies `--log-conversations`).
- `--log-dir /path/to/dir` logs each resolved session id to its own file
  named `<session-id>.log` after filename sanitization. Requests without a
  stable session id use `request-<request-id>.log`.

File logging rotates by default when the active log exceeds 64 MiB. CCP keeps five
backups named by appending numeric suffixes to the configured path, for example
`conversations.log.1` through `conversations.log.5`. Tune this with
`--log-max-bytes` and `--log-backups`; set `--log-max-bytes 0` to disable
rotation. Directory logging does not rotate because each session gets a separate
file.

Log format:

```
[HH:MM:SS] session=<session-id> request=<request-id> >>> Inbound OAI body
<raw request body>
------------------------------

[HH:MM:SS] session=<session-id> request=<request-id> >>> Anthropic body
<translated Anthropic request body>
------------------------------

[HH:MM:SS] session=<session-id> request=<request-id> <<< OpenAI response
<translated response>
--------------------------------
```

## Tool/Function Calling

v2 passes tools natively to the Anthropic Messages API as `tools[]`, maps OpenAI `tool_choice` to Anthropic `tool_choice`, converts Anthropic `tool_use` blocks back to OpenAI `tool_calls`, and converts OpenAI `tool` role messages to Anthropic `tool_result` blocks.

Anthropic's OAuth gate appears to fingerprint the tool surface. Tool name PascalCase masking is recommended via text replacement to satisfy that fingerprint, for example outbound `memory_search` -> `MemorySearch` and inbound `MemorySearch` -> `memory_search`.

Supported `tool_choice` values:

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Model decides whether to call tools or respond with text |
| `"none"` | Anthropic `tool_choice: {"type":"none"}` |
| `"required"` | Anthropic `tool_choice: {"type":"any"}` |
| `{"type":"function","function":{"name":"X"}}` | Directs the model to call function X |

## Known Limitations

- **Text only** - OpenAI `image_url` and audio content parts are not yet translated to Anthropic media blocks.
- **Subscription-bound** - CCP uses a Claude Max OAuth token from Claude Code credentials. Per-call billing API keys are not supported.
- **Tool surface fingerprinting** - non-Claude-Code-like tool names can be rejected by the OAuth gate unless masked with text replacement.
- **Streaming replacement boundaries** - replacements are per streamed text delta, not across arbitrary chunk boundaries.

## Request Limits

- **Request body**: 10 MB maximum.

## CORS

CORS is enabled permissively, allowing browser-based clients to call the API directly from any origin.

## Response Headers

All completions responses include an `x-request-id` header for debugging and log correlation.

## Graceful Shutdown

The server handles `SIGINT` (Ctrl+C) and `SIGTERM` for graceful shutdown. In-flight requests complete before the server exits. Docker sends `SIGTERM` on `docker stop`.

## Developer Messages

The `developer` message role (used by newer OpenAI SDKs) is treated identically to `system`.
