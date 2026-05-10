# Architecture

## Request Flow

```
OpenAI SDK Client
       |
       v
+-------------------+
| Auth Middleware   |  Validates local API key and attaches key ID
+---------+---------+
          |
          v
+-------------------+
| Completions       |  Parses OpenAI request, resolves model,
| Handler           |  records stats and request context
+---------+---------+
          |
          v
+-------------------+
| OAI -> Anthropic  |  Reshapes messages, tools, tool_choice,
| Translation       |  sampling params, thinking and system field
+---------+---------+
          |
          v
+-------------------+
| OAuth Gate Layer  |  Prepends Claude Code system identifier,
|                   |  applies outbound text replacements
+---------+---------+
          |
          v
+-------------------+
| Upstream Client   |  POSTs to api.anthropic.com/v1/messages
|                   |  with Claude Code fingerprint headers
+---------+---------+
          |
          v
+-------------------+
| Anthropic -> OAI  |  Converts JSON or SSE events back into
| Translation       |  OpenAI chat completion shapes
+---------+---------+
          |
          v
+-------------------+
| Response Builder  |  Applies response replacements and emits
|                   |  JSON or OpenAI-compatible SSE
+-------------------+
```

## Module Structure

```
src/
├── main.rs              # Server setup, startup credential validation, routing
├── config.rs            # CLI args and CCP_* env var parsing
├── auth.rs              # API key middleware
├── error.rs             # Unified error types -> OpenAI error format
├── models.rs            # Model definitions, aliases, resolution
├── replacements.rs      # TOML text replacement engine
├── conversation_log.rs  # Request/response logging
├── session.rs           # Stable session ID derivation
├── stats.rs             # Persistent stats (redb) + active request tracking
├── routes/
│   ├── completions.rs   # POST /v1/chat/completions entrypoint
│   ├── completions_v2.rs# Direct Anthropic Messages request handling
│   ├── models.rs        # GET /v1/models handler
│   ├── health.rs        # GET /health handler
│   └── stats.rs         # GET /stats and /stats/json handlers
├── translate/
│   ├── anthropic.rs     # Native Anthropic Messages wire types
│   ├── build.rs         # OpenAI request -> Anthropic Messages body
│   ├── messages.rs      # Message/content/tool-result reshaping
│   ├── tool_translate.rs# OpenAI tools -> Anthropic tools
│   ├── from_anthropic.rs# Anthropic JSON response -> OpenAI response
│   └── to_oai_stream.rs # Anthropic SSE events -> OpenAI SSE chunks
└── upstream/
    ├── client.rs        # reqwest client, retry policy, streaming
    ├── credentials.rs   # Reads ~/.claude/.credentials.json per request
    ├── fingerprint.rs   # Claude Code-compatible request headers
    └── stream.rs        # Anthropic SSE parser
```

## Design Decisions

### Direct Messages API

v2 does not invoke the Claude Code CLI per request. It translates OpenAI Chat Completions requests into Anthropic Messages API calls and sends them directly to `api.anthropic.com` with the OAuth token from `~/.claude/.credentials.json`.

The credentials file is re-read for each request, and a 401 response triggers one fresh read and retry. CCP still depends on Claude Code for login and token refresh, but the CLI is not in the request path.

### OAuth Gate Compatibility

Anthropic's subscription OAuth gate expects Claude Code-shaped traffic. CCP mimics the important wire-level pieces:

- Claude Code-compatible headers and beta flags.
- A stable `x-claude-code-session-id` per logical session.
- The canonical Claude Code system identifier as the first system block.

`--no-preamble` skips the system identifier and is intended only for debugging upstream behavior or for callers that provide an equivalent first system block themselves.

### Text Replacement

Replacement rules are loaded once at startup and stored in an `Arc<Replacements>`. Prompt replacements are applied to outbound system text, message text, tool names, descriptions, schemas, and tool-result content before the Anthropic request is sent. Response replacements are applied to assistant text and tool-call names/arguments before the OpenAI response is returned.

For streaming responses, replacements are applied per text delta and to streamed tool names. A search string split across separate deltas will not match.

### Native Tool Calling

v2 passes tools through as Anthropic Messages API `tools[]` and maps tool choices to native Anthropic `tool_choice` values. Assistant `tool_use` blocks are converted back to OpenAI `tool_calls`, and OpenAI `tool` messages are converted to Anthropic `tool_result` blocks.

The OAuth gate appears to fingerprint tool surfaces. Tool names that look unlike Claude Code tools can be rejected upstream, so PascalCase masking via text replacement is recommended: replace names such as `memory_search` with `MemorySearch` outbound and reverse them inbound.

### Streaming Translation

Anthropic SSE events are parsed into stateful OpenAI `chat.completion.chunk` messages. The converter emits the assistant role once, streams text deltas, streams tool-call argument fragments, carries thinking deltas in extension fields, and maps Anthropic stop reasons to OpenAI finish reasons.

### Error Format

All client-visible errors use the OpenAI error format (`{"error": {"message": "...", "type": "...", "code": null}}`) so SDK clients handle them consistently. Anthropic errors are mapped by status:

- 400 -> `invalid_request_error`
- 401/403 -> `authentication_error`
- 429 -> `rate_limit_error`
- 500/503/504 -> `server_error`

### Stats Persistence

Statistics are stored in a `redb` embedded database rather than in-memory. This survives restarts and provides historical data for the dashboard without external dependencies.
