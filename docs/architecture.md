# Architecture

## Request Flow

```
OpenAI SDK Client
       │
       ▼
┌──────────────────┐
│  Auth Middleware  │  Validates API key, attaches key ID to request
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Completions     │  Parses OpenAI request, builds prompt,
│  Handler         │  applies text replacements
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Subprocess      │  Acquires semaphore slot (or queues),
│  Manager         │  spawns `claude` process
└──────┬───────────┘
       ▼
┌──────────────────┐
│  claude CLI      │  Runs with -p --output-format stream-json
│  Subprocess      │  Emits NDJSON on stdout
└──────┬───────────┘
       ▼
┌──────────────────┐
│  NDJSON Parser   │  Extracts model, content deltas, result,
│                  │  usage from streaming NDJSON
└──────┬───────────┘
       ▼
┌──────────────────┐
│  Response        │  Translates to OpenAI format (JSON or SSE),
│  Builder         │  applies response text replacements
└──────────────────┘
```

## Module Structure

```
src/
├── main.rs              # Server setup, startup validation, routing
├── config.rs            # CLI args and env var parsing (clap)
├── auth.rs              # API key middleware
├── error.rs             # Unified error types → OpenAI error format
├── models.rs            # Model definitions, aliases, resolution
├── replacements.rs      # TOML text replacement engine
├── conversation_log.rs  # Prompt/response logging
├── stats.rs             # Persistent stats (redb) + active request tracking
├── routes/
│   ├── completions.rs   # POST /v1/chat/completions handler
│   ├── models.rs        # GET /v1/models handler
│   ├── health.rs        # GET /health handler
│   └── stats.rs         # GET /stats and /stats/json handlers
├── subprocess/
│   ├── mod.rs           # SubprocessEvent type
│   ├── manager.rs       # Semaphore-based concurrency control
│   ├── process.rs       # Subprocess spawning, I/O, timeout
│   └── ndjson.rs        # NDJSON line parser for Claude CLI output
└── translate/
    ├── request.rs       # OpenAI messages → CLI args + prompt
    ├── response.rs      # CLI result → OpenAI response JSON
    ├── stream.rs        # SSE chunk builders for streaming
    └── tools.rs         # Tool call prompt injection and response parsing
```

## Design Decisions

### Subprocess Per Request

Each request spawns a new `claude` subprocess rather than maintaining persistent connections. This is intentional:

- The Claude CLI's `-p` (print) mode is designed for single-shot use.
- `--no-session-persistence` ensures no state leaks between requests.
- The trade-off is subprocess startup latency, mitigated by the CLI's relatively fast cold start.

### Config Isolation

By default, subprocesses run with `CLAUDE_CONFIG_DIR` pointing to a separate directory. This prevents the proxy from interfering with your personal Claude Code settings (plugins, hooks, `.claude.json`). The isolated directory is cleaned on each startup, and credentials are symlinked from `~/.claude/.credentials.json`.

### Concurrency Control

A `tokio::sync::Semaphore` limits concurrent subprocesses. This prevents resource exhaustion on the host while allowing requests to queue rather than fail immediately.

### Text Replacement

Replacement rules are loaded once at startup and stored in an `Arc<Replacements>`. Prompt replacements happen before the prompt is passed to the CLI. Response replacements happen after content is received (per-chunk for streaming, on full content for non-streaming).

### Tool Call Passthrough

The Claude Code CLI executes tools internally in its agent loop — there is no way to make it forward `tool_use` blocks to the caller. The proxy works around this with prompt injection:

1. When a request includes `tools`, tool definitions are converted to text and prepended to the user message.
2. `--tools ""` remains set so Claude Code never executes tools internally.
3. The model responds with JSON tool call arrays as plain text.
4. The proxy parses the text response, detects tool call JSON (stripping markdown code fences if present), and converts it to OpenAI `tool_calls` format.
5. For multi-turn, the client's `tool` role messages are formatted as `<tool_result>` XML tags in the prompt.

Key design constraints discovered through testing:
- The CLI's built-in agentic system prompt is replaced wholesale via `--system-prompt` (not `--append-system-prompt`), with a minimal CCP-owned preamble that yields formatting authority to the user message. Appending was insufficient: the CLI default still dominated and triggered tool-call retry loops and `error_max_turns` failures in single-shot mode. Client-supplied system prompts are appended after the preamble so they still take effect.
- Tool dispatch instructions live in the **user message** (prepended via `build_tool_prompt_prefix`). The system prompt deliberately stays out of formatting rules so the model treats the user-message instructions as authoritative.
- Haiku wraps output in markdown code fences; Sonnet outputs clean JSON. The parser handles both.
- When tools are present in streaming mode, the response is buffered to determine if it contains tool calls before emitting SSE chunks.

### Error Format

All errors use the OpenAI error format (`{"error": {"message": "...", "type": "...", "code": null}}`) so clients handle them correctly. HTTP status codes map to:

- 400 → `invalid_request_error`
- 401 → `authentication_error`
- 404 → `invalid_request_error`
- 500 → `server_error`
- 503 → `server_error` (with `Retry-After: 30` header)
- 504 → `server_error` (inactivity timeout)

### NDJSON Parsing

The Claude CLI outputs NDJSON with various message types. The parser extracts:

- `message_start` → model name
- Content block `text_delta` events → streamed content
- `result` → final message with usage stats, cost, session ID

Assistant role messages from `--include-partial-messages` are ignored since their content is already captured via deltas.

### Stats Persistence

Statistics are stored in a `redb` embedded database rather than in-memory. This survives restarts and provides accurate historical data for the dashboard without external dependencies.
