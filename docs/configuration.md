# Configuration Reference

All options can be set via CLI flags or environment variables. CLI flags take precedence over environment variables.

## Options

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p, --port` | `CCP_PORT` | `18321` | Listen port |
| `-H, --host` | `CCP_HOST` | `127.0.0.1` | Listen address |
| `-c, --max-concurrent` | `CCP_MAX_CONCURRENT` | `5` | Max simultaneous subprocesses |
| `-t, --timeout` | `CCP_TIMEOUT` | `600` | Subprocess inactivity timeout (seconds) |
| `-q, --queue-timeout` | `CCP_QUEUE_TIMEOUT` | `60` | Max time a request waits for an available slot (seconds) |
| `--claude-path` | `CCP_CLAUDE_PATH` | `claude` | Path to Claude CLI binary |
| `--data-dir` | `CCP_DATA_DIR` | Platform default | Data directory for config isolation and stats DB |
| `--working-dir` | `CCP_WORKING_DIR` | See below | Subprocess working directory |
| `--no-isolate` | `CCP_NO_ISOLATE` | Off | Use host Claude config directly (skip config isolation) |
| `--api-keys` | `CCP_API_KEYS` | Auto-generated | Comma-separated API keys (min 8 chars each) |
| `--api-keys-file` | `CCP_API_KEYS_FILE` | None | File with one API key per line (`#` comments allowed) |
| `--no-auth` | `CCP_NO_AUTH` | Off | Disable authentication entirely |
| `--no-tool-passthrough` | `CCP_NO_TOOL_PASSTHROUGH` | Off | Disable tool/function call passthrough |
| `--replace-rules` | `CCP_REPLACE_RULES` | None | TOML file with text replacement rules |
| `--log-conversations` | `CCP_LOG_CONVERSATIONS` | Off | Log full prompts and responses to stderr |
| `--log-file` | `CCP_LOG_FILE` | None | Write conversation logs to file (implies `--log-conversations`) |
| `-v, --verbose` | | Off | Debug logging |

The `RUST_LOG` environment variable overrides `-v` when set, allowing fine-grained log filtering (e.g., `RUST_LOG=claude_code_provider=debug`).

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

Date-suffixed model names (e.g., `claude-sonnet-4-6-20260101`) are also resolved via substring matching. Unrecognized names fall back to Sonnet.

## Reasoning Effort

| Value | Behavior |
|-------|----------|
| `none` (or absent) | Default behavior, no extended thinking |
| `low` | Light reasoning |
| `medium` | Moderate reasoning |
| `high` | Deep reasoning |
| `max` | Maximum reasoning |

Pass `reasoning_effort` in the request body. Maps directly to the CLI `--effort` flag. When absent or `none`, the flag is omitted and Claude uses its default behavior.

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

Three modes, in priority order:

1. **No auth** (`--no-auth`) - all requests pass through. Use only on trusted networks.
2. **Explicit keys** (`--api-keys` or `--api-keys-file`) - requests must include `Authorization: Bearer <key>`.
3. **Auto-generated** (default) - a random UUID key is generated on startup and printed to the log.

Keys must be at least 8 characters. Both `--api-keys` and `--api-keys-file` can be used together; keys are deduplicated.

The `/health` and `/stats` endpoints are always accessible without authentication.

API key IDs (first 4 + last 4 characters) appear in request logs and the stats dashboard for tracking without exposing full keys.

## Working Directory

The `--working-dir` flag controls where Claude CLI subprocesses run. The default depends on isolation mode:

- **Isolation on** (default): defaults to the isolated config directory (`<data-dir>/claude-config`).
- **Isolation off** (`--no-isolate`): defaults to the data directory.

## Config Isolation

By default, subprocesses use a separate config directory so the proxy never touches your existing Claude Code settings. The isolated directory:

- Is created at `<data-dir>/claude-config`.
- Is cleaned on each startup (stale `.claude.json` files can enable unexpected tools).
- Contains a symlink to your `~/.claude/.credentials.json` for authentication.

Set `--no-isolate` to use your host Claude configuration directly. The Docker image sets this by default since containers have no host config to isolate from.

## Concurrency and Queuing

The `--max-concurrent` flag limits how many Claude CLI subprocesses run simultaneously. When all slots are busy, incoming requests queue for up to `--queue-timeout` seconds. If the timeout expires, the server returns HTTP 503 with a `Retry-After: 30` header.

## Inactivity Timeout

Each subprocess has an inactivity timeout (`--timeout`). If the subprocess produces no output for this duration, it is killed and the request returns HTTP 504.

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
- For streaming responses, replacement is applied per-chunk. If a search string spans a chunk boundary, it won't be matched. This is rare in practice since the Claude CLI emits complete words per chunk.

## Conversation Logging

Two output targets:

- `--log-conversations` logs to stderr, interleaved with server logs.
- `--log-file /path/to/file` logs to a dedicated file (also implies `--log-conversations`).

Log format:

```
[HH:MM:SS] <request-id> >>> Prompt
<prompt text>
------------------------------

[HH:MM:SS] <request-id> <<< Response
<response text>
--------------------------------
```

System prompts are logged separately with the label `System`.

## Tool/Function Calling

Tool passthrough is enabled by default. When a request includes `tools`, the proxy:

1. Converts tool definitions into prompt instructions prepended to the user message.
2. Keeps `--tools ""` so Claude Code never executes tools internally.
3. Parses the model's text response for structured JSON tool calls.
4. Returns OpenAI-compatible `tool_calls` in the response.

Supported `tool_choice` values:

| Value | Behavior |
|-------|----------|
| `"auto"` (default) | Model decides whether to call tools or respond with text |
| `"none"` | Tool injection skipped entirely, normal text response |
| `"required"` | Soft nudge for the model to use a tool |
| `{"type":"function","function":{"name":"X"}}` | Directs the model to call function X |

Use `--no-tool-passthrough` to disable globally. When disabled, `tools` in requests are silently ignored.

**Streaming with tools:** When tools are present, the response is buffered before streaming to detect whether it contains tool calls or plain text. The SSE keepalive (15s interval) maintains the connection during buffering.

**Multi-turn:** Assistant messages with `tool_calls` and `tool` role messages with `tool_call_id` are formatted into the prompt so the model can read tool results and continue the conversation.

## Environment Variable Sanitization

Subprocesses have the following environment variables removed to prevent interference:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_AUTH_TOKEN`

## Known Limitations

- **Text only** — image and audio content parts in messages are silently ignored; only text is extracted.
- **Ignored parameters** — `max_tokens`, `temperature`, `top_p`, `stop`, and other sampling parameters are accepted for client compatibility but not forwarded to the CLI (no corresponding flags exist).
- **Subprocess per request** — each request spawns a new `claude` subprocess, adding startup latency compared to a direct API connection.
- **Buffered streaming with tools** — when `tools` are present, the response is buffered before streaming to detect whether it contains tool calls or plain text.

## Request Limits

- **Request body**: 10 MB maximum.
- **Prompt argument**: 128 KB maximum per argument (Linux `MAX_ARG_STRLEN` limit). Prompts or system prompts exceeding this return HTTP 400.

## CORS

CORS is enabled permissively, allowing browser-based clients to call the API directly from any origin.

## Response Headers

All completions responses include an `x-request-id` header for debugging and log correlation.

## Graceful Shutdown

The server handles `SIGINT` (Ctrl+C) and `SIGTERM` for graceful shutdown. In-flight requests complete before the server exits. Docker sends `SIGTERM` on `docker stop`.

## Subprocess Invocation

Each request spawns a `claude` subprocess with these flags:

- `-p` (print mode, non-interactive)
- `--verbose --output-format stream-json --include-partial-messages` (NDJSON streaming output)
- `--tools ""` (Claude Code's built-in tools disabled; tool calling is handled via prompt injection)
- `--model <model>` (resolved from the request)
- `--no-session-persistence` (no conversation memory between requests)
- `--append-system-prompt <text>` (if a system/developer message is present)
- `--effort <level>` (if reasoning_effort is specified)

The `developer` message role (used by newer OpenAI SDKs) is treated identically to `system`.
