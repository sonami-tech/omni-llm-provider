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
| `--replace-rules` | `CCP_REPLACE_RULES` | None | TOML file with text replacement rules |
| `--log-conversations` | `CCP_LOG_CONVERSATIONS` | Off | Log full prompts and responses to stderr |
| `--log-file` | `CCP_LOG_FILE` | None | Write conversation logs to file (implies `--log-conversations`) |
| `-v, --verbose` | | Off | Debug logging |

The `RUST_LOG` environment variable overrides `-v` when set, allowing fine-grained log filtering (e.g., `RUST_LOG=claude_code_provider=debug`).

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

See the [README](../README.md#text-replacement) for the TOML rule format. Additional details:

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

## Environment Variable Sanitization

Subprocesses have the following environment variables removed to prevent interference:

- `ANTHROPIC_API_KEY`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_AUTH_TOKEN`

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
- `--tools ""` (tools disabled)
- `--model <model>` (resolved from the request)
- `--no-session-persistence` (no conversation memory between requests)
- `--append-system-prompt <text>` (if a system/developer message is present)
- `--effort <level>` (if reasoning_effort is specified)

The `developer` message role (used by newer OpenAI SDKs) is treated identically to `system`.
