# Claude Code Provider — Technical Specification

## Overview

Claude Code Provider is a Rust-based HTTP proxy that exposes an OpenAI-compatible API (`/v1/chat/completions`) backed by Claude Code CLI subprocess invocations. It routes requests through the user's Claude Max subscription via the official CLI, avoiding per-token API costs while staying within Anthropic's Terms of Service.

The proxy accepts standard OpenAI Chat Completions requests, translates them into `claude -p` CLI invocations, and streams the NDJSON output back as OpenAI-format SSE events.

## Goals

- Provide an OpenAI-compatible API endpoint that any standard client can use.
- Route all inference through `claude -p` (the official CLI's print mode) to stay ToS-compliant.
- Support concurrent requests with bounded parallelism.
- Support all available Claude models with thinking/effort level control.
- Provide a web-accessible statistics dashboard for operational visibility.

## Non-Goals (Explicitly Out of Scope)

- Anthropic Messages API format (`/v1/messages`).
- OAuth token extraction or management (Claude Code handles its own auth).
- MCP tool use or function calling passthrough.
- Multi-account load balancing.
- Session persistence or multi-turn conversation state (clients manage their own history).
- Web UI for chat (this is an API proxy, not a frontend).

---

## Architecture

```
┌─────────────┐     HTTP      ┌──────────────────┐   subprocess   ┌────────────┐
│   Client     │ ─────────────▶│  Claude Code      │ ──────────────▶│ claude -p  │
│  (any app)   │◀──── SSE ─────│  Provider (Rust)  │◀── NDJSON ────│   CLI      │
└─────────────┘               └──────────────────┘               └────────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  Stats Page  │
                               │  (HTML)      │
                               └──────────────┘
```

### Components

1. **HTTP Server (Axum)** — Accepts OpenAI-format requests, serves stats page.
2. **Request Translator** — Converts OpenAI Chat Completions request into CLI arguments and prompt text.
3. **Subprocess Manager** — Spawns and manages `claude -p` processes with bounded concurrency.
4. **Response Translator** — Parses CLI NDJSON output and converts to OpenAI SSE streaming or JSON response.
5. **Stats Collector** — Tracks request metrics for the dashboard.

---

## API Endpoints

### `POST /v1/chat/completions`

Primary endpoint. Accepts OpenAI Chat Completions format.

**Request:**

```json
{
	"model": "claude-sonnet-4-6",
	"messages": [
		{"role": "system", "content": "You are a helpful assistant."},
		{"role": "user", "content": "Hello"},
		{"role": "assistant", "content": "Hi there!"},
		{"role": "user", "content": "What is 2+2?"}
	],
	"max_tokens": 4096,
	"temperature": 0.7,
	"stream": true,
	"reasoning_effort": "high"
}
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (see Model Selection) |
| `messages` | array | Yes | Conversation messages with `role` and `content` |
| `stream` | bool | No | Enable SSE streaming (default: `false`) |
| `max_tokens` | int | No | Accepted for compatibility but **not forwarded** to the CLI (no flag exists). Claude uses its own default. |
| `max_completion_tokens` | int | No | OpenAI's newer equivalent of `max_tokens`. Accepted for compatibility but **not forwarded**. |
| `temperature` | float | No | Accepted for compatibility but **not forwarded** to the CLI (no flag exists). |
| `reasoning_effort` | string | No | Thinking level: `none`, `low`, `medium`, `high`, `max` |
| `stop` | string/array | No | Stop sequences (not forwarded to CLI — see Limitations) |

Any other OpenAI parameters (`n`, `logprobs`, `top_logprobs`, `user`, `seed`, `response_format`, `frequency_penalty`, `presence_penalty`, `top_p`, `top_k`, `stream_options`, `tools`, `tool_choice`, etc.) are silently accepted and ignored. The `ChatCompletionRequest` struct uses serde's default behavior (no `deny_unknown_fields`), so unknown fields pass through without error. This is critical for compatibility — real-world clients send many parameters the proxy doesn't support, and rejecting them would break integration.

**Streaming Response (SSE):**

```
data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712345678,"model":"claude-sonnet-4-6","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":"2 + 2"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712345678,"model":"claude-sonnet-4-6","system_fingerprint":null,"choices":[{"index":0,"delta":{"content":" = 4."},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712345678,"model":"claude-sonnet-4-6","system_fingerprint":null,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712345678,"model":"claude-sonnet-4-6","system_fingerprint":null,"choices":[],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}

data: [DONE]
```

**Non-Streaming Response:**

```json
{
	"id": "chatcmpl-a1b2c3d4",
	"object": "chat.completion",
	"created": 1712345678,
	"model": "claude-sonnet-4-6",
	"system_fingerprint": null,
	"choices": [
		{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": "2 + 2 = 4."
			},
			"finish_reason": "stop"
		}
	],
	"usage": {
		"prompt_tokens": 25,
		"completion_tokens": 8,
		"total_tokens": 33
	}
}
```

The `system_fingerprint` field is always `null` (no meaningful value from the CLI) but is included because some OpenAI clients expect it to be present.

### `GET /v1/models`

Returns available models.

**Response:**

```json
{
	"object": "list",
	"data": [
		{
			"id": "claude-opus-4-6",
			"object": "model",
			"created": 0,
			"owned_by": "anthropic",
			"context_window": 1000000,
			"max_tokens": 128000
		},
		{
			"id": "claude-sonnet-4-6",
			"object": "model",
			"created": 0,
			"owned_by": "anthropic",
			"context_window": 1000000,
			"max_tokens": 64000
		},
		{
			"id": "claude-haiku-4-5",
			"object": "model",
			"created": 0,
			"owned_by": "anthropic",
			"context_window": 200000,
			"max_tokens": 64000
		}
	]
}
```

The model list is static and configured at build time. Short aliases (`opus`, `sonnet`, `haiku`) are also accepted on the completions endpoint but are not listed here — only the canonical names appear. The `context_window` and `max_tokens` fields are non-standard OpenAI extensions but are included for client introspection (the reference implementation includes them, and clients like Open WebUI use them).

### `GET /stats`

Web-accessible HTML dashboard showing operational statistics.

**Displayed Metrics:**

- Server uptime.
- Total requests served (lifetime).
- Active requests (currently in-flight).
- Requests per model (breakdown).
- Average time-to-first-token (TTFT) per model.
- Average total request duration per model.
- Error count and recent errors (last 10).
- Token usage summary (input/output totals per model, from CLI result data).

The page auto-refreshes via a meta tag or lightweight JS polling. No frameworks — plain HTML with inline CSS.

### `GET /stats/json`

Returns the same statistics as `/stats` in JSON format for programmatic consumption. The `recent_errors` array contains the last 10 errors (newest first), matching the HTML dashboard — the in-memory buffer stores up to 50 but only the 10 most recent are included in both views.

**Response:**

```json
{
	"uptime_seconds": 3600,
	"total_requests": 150,
	"active_requests": 2,
	"errors": 3,
	"total_input_tokens": 45000,
	"total_output_tokens": 12000,
	"total_cache_read_input_tokens": 30000,
	"total_cache_creation_input_tokens": 15000,
	"models": {
		"claude-sonnet-4-6": {
			"requests": 120,
			"avg_ttft_ms": 450.5,
			"avg_duration_ms": 2100.3,
			"input_tokens": 35000,
			"output_tokens": 9000,
			"cache_read_input_tokens": 25000,
			"cache_creation_input_tokens": 10000
		}
	},
	"recent_errors": [
		{"timestamp": "2026-04-03T12:34:56Z", "model": "claude-opus-4-6", "message": "Inactivity timeout"}
	]
}
```

### `GET /health`

Returns `200 OK` with a JSON body indicating server status. Useful for load balancers or monitoring.

```json
{
	"status": "ok",
	"uptime_seconds": 3600,
	"active_requests": 2,
	"max_concurrent": 5
}
```

---

## Model Selection

### Accepted Model Identifiers

The proxy accepts both canonical model names and short aliases in the `model` field:

| Canonical Name | Short Aliases |
|----------------|---------------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

Any unrecognized model name falls back to `sonnet` with a warning logged.

### CLI Model Mapping

The proxy maps the resolved model to the CLI `--model` flag:

| Resolved Model | CLI `--model` value |
|----------------|---------------------|
| `claude-opus-4-6` | `opus` |
| `claude-sonnet-4-6` | `sonnet` |
| `claude-haiku-4-5` | `haiku` |

### Thinking / Effort Levels

When `reasoning_effort` is present in the request, it maps directly to `--effort`:

| `reasoning_effort` | CLI `--effort` | Notes |
|--------------------|----------------|-------|
| `none` | (flag omitted) | No thinking — default behavior. `none` is NOT a valid CLI `--effort` value; the flag is simply omitted. |
| `low` | `low` | Light reasoning |
| `medium` | `medium` | Moderate reasoning |
| `high` | `high` | Deep reasoning |
| `max` | `max` | Maximum reasoning |

When `reasoning_effort` is absent, the `--effort` flag is omitted entirely, letting Claude Code use its default behavior. The CLI accepts exactly four values for `--effort`: `low`, `medium`, `high`, `max`.

---

## Subprocess Management

### CLI Invocation

Each request spawns a `claude` subprocess:

```bash
claude -p \
	--verbose \
	--output-format stream-json \
	--include-partial-messages \
	--tools "" \
	--model <model> \
	--append-system-prompt "<system>" \ # only if system messages are present
	--effort <level> \           # only if reasoning_effort != none and is present
	--no-session-persistence \
	"<prompt>"
```

The subprocess is spawned with `CLAUDE_CONFIG_DIR` set to the proxy's isolated config directory (see Process Isolation below). This ensures zero plugins, MCP servers, hooks, and CLAUDE.md files are loaded regardless of the host environment.

**Flag rationale:**

| Flag | Purpose |
|------|---------|
| `-p` / `--print` | Non-interactive mode — output to stdout, exit when done |
| `--verbose` | **Mandatory** when using `--output-format stream-json` with `-p`. The CLI exits with an error without it: "When using --print, --output-format=stream-json requires --verbose" |
| `--output-format stream-json` | NDJSON output for streaming events |
| `--include-partial-messages` | Emit streaming content deltas (without this, only the final result is emitted) |
| `--tools ""` | **Disable all built-in tools.** The empty string removes Bash, Read, Write, Edit, Glob, Grep, WebSearch, WebFetch, and all other built-in tools from the CLI. **Important:** `--tools ""` only removes *built-in* tools; plugin and MCP server tools remain active if installed. Combined with `CLAUDE_CONFIG_DIR` process isolation (which eliminates plugins and MCP servers), this results in zero tools available — the model generates text only and never attempts tool calls. This saves ~4,000–8,000 tokens per request (tool definitions are not included in the system prompt) and guarantees single-turn responses |
| `--model` | Select the Claude model (accepts short names: `opus`, `sonnet`, `haiku`) |
| `--append-system-prompt` | Append the client's system prompt to Claude Code's default system prompt. Uses `--append-system-prompt` (not `--system-prompt`) because the CLI has its own default system context that should be preserved |
| `--effort` | Set thinking/reasoning level. Maps 1:1 from `reasoning_effort` request field |
| `--no-session-persistence` | Don't save session to disk (each request is independent) |

**Argument passing for `--tools ""`:** In Rust, pass this as two separate `Command` args: `.arg("--tools").arg("")`. The empty string must be an explicit empty argument, not omitted. Shell-style quoting (`--tools ""`) is a shell concept — `Command` passes args directly to the kernel without shell interpretation, so `.arg("")` correctly passes an empty string.

**Why not `--permission-mode plan`?** Earlier versions of this spec used `--permission-mode plan` to prevent tool execution. However, plan mode still loads all tool definitions into the system prompt (~12,500 tokens) and the model may still attempt tool calls that get denied, causing multi-turn overhead. The `--tools ""` approach is strictly superior: it eliminates tools entirely from the system prompt, saving ~40% in token costs and guaranteeing single-turn text-only responses.

**Why not `--bare`?** The `--bare` flag disables the OAuth/keychain authentication that Claude Max subscriptions rely on. The help text explicitly states: "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings (OAuth and keychain are never read)." This causes "Not logged in" failures for Claude Max users. Process isolation via `CLAUDE_CONFIG_DIR` achieves the same deterministic behavior (no hooks, plugins, CLAUDE.md) while preserving OAuth auth.

**Why not `--permission-mode bypassPermissions`?** The `claude-max-api-proxy-rs` reference uses this flag, but it is unnecessary and risky with our approach. With `--tools ""`, there are no tools to execute, so permissions are irrelevant. Using `bypassPermissions` would only matter if tools were somehow loaded (a configuration error), and in that case, bypassing permissions would make the error worse, not better.

**Flags NOT available** (accepted in requests for client compatibility but not forwarded):

| Request Parameter | Why Not Forwarded |
|-------------------|-------------------|
| `max_tokens` / `max_completion_tokens` | No `--max-tokens` CLI flag exists |
| `temperature` | No `--temperature` CLI flag exists |
| `top_p` | No `--top-p` CLI flag exists |
| `stop` | No `--stop` CLI flag exists |
| `frequency_penalty` | No CLI equivalent |
| `presence_penalty` | No CLI equivalent |

**Note:** The CLI has `--max-budget-usd` and `--fallback-model` flags (for print mode). `--max-turns` was introduced in a later version (v2.1.91+; not present in v2.1.81). None are exposed through the proxy API. The proxy always runs single-turn with `--tools ""`, so turn limits are irrelevant. `--max-budget-usd` could be added as a proxy-level safety limit in a future version (live testing confirms it produces `"subtype":"error_max_budget_usd"` in the result, which the proxy already handles). `--fallback-model <model>` provides automatic model fallback when the primary model is overloaded — could be exposed as a proxy config option.

**Other CLI flags NOT used by the proxy** (present in CLI v2.1.81 but intentionally excluded):

| CLI Flag | Why Not Used |
|----------|-------------|
| `--json-schema` | **Caution:** Live testing revealed that `--json-schema` **overrides `--tools ""`** by injecting a `StructuredOutput` tool, causing multi-turn behavior (2-3 turns, ~3x cost). The result text (`result.result`) becomes an empty string with output only in a `structured_output` field. This breaks the zero-tools single-turn guarantee. Do not use with this proxy. |
| `--allowedTools` / `--disallowedTools` | Unnecessary — `--tools ""` already disables all built-in tools. These flags are for selective tool filtering, which is irrelevant when all tools are removed. |
| `--permission-mode auto` | Live testing confirms this is silently ignored when `--tools ""` is active (init event shows `"permissionMode":"default"` regardless). No behavioral difference. |
| `--agent` / `--agents` | Multi-agent routing — out of scope for a text-only proxy. |
| `--strict-mcp-config` | Could supplement `CLAUDE_CONFIG_DIR` isolation, but the config directory approach is more thorough (eliminates all non-credential config, not just MCP). |
| `--betas` | Beta API headers — only relevant for API key users, not OAuth/Claude Max. |
| `--add-dir` | Adds directories to tool access scope — irrelevant with zero tools. |
| `--brief` | Enables `SendUserMessage` tool for agent-to-user communication — irrelevant for API proxy. |
| `--replay-user-messages` | Only useful with `--input-format stream-json` bidirectional streaming (a future consideration). |

### Prompt Construction

The proxy separates the OpenAI message array into a system prompt and a conversation prompt:

**System prompt** — Extracted from all `role: "system"` and `role: "developer"` messages, concatenated with newlines, and passed via the `--append-system-prompt` CLI flag. The `developer` role is OpenAI's newer equivalent of `system` (introduced for the o1/o3 model family) and should be treated identically. This appends to (not replaces) Claude Code's default system prompt. If no system messages are present, the flag is omitted entirely.

**Why `--append-system-prompt` instead of `--system-prompt`?** The CLI has its own default system prompt. Using `--system-prompt` would replace it entirely. `--append-system-prompt` preserves the defaults and adds the client's context. The reference implementation uses an alternative approach (wrapping system messages in `<system>` XML tags inline in the prompt), but `--append-system-prompt` is preferred because it uses the CLI's native mechanism.

**Conversation prompt** — The remaining `user` and `assistant` messages are converted into a single text string passed as the positional prompt argument. Messages are processed in their original array order (preserving conversational sequence). System messages are skipped during this pass (already extracted above). Each non-system message is formatted as:

1. **User messages** — Included as-is (bare text).
2. **Assistant messages** — Wrapped in `<previous_response>` tags.
3. **Developer messages** (`role: "developer"`) — Treated as system messages (concatenated into the system prompt). This is OpenAI's newer name for system instructions.
4. **Other roles** (e.g., `tool`, `function`) — Treated as user messages (included as bare text). These are uncommon in practice since the proxy doesn't support tool use, but clients may send them.
4. **Content field handling** — The OpenAI format allows `content` to be either a plain string or an array of content blocks. When it is an array, extract all `text` type blocks and concatenate them. Non-text blocks (images, etc.) are ignored with a warning logged.

Messages are joined with double newlines (`\n\n`) between them.

```
Hello

<previous_response>
Hi there!
</previous_response>

What is 2+2?
```

If there are no user messages after filtering out system messages, the request is rejected with 400.

Only text content is supported in the MVP. Image/multimodal content blocks are skipped.

### Process Isolation

The proxy runs each `claude` subprocess with `CLAUDE_CONFIG_DIR` set to an isolated configuration directory. This is the primary mechanism for ensuring zero plugins, MCP servers, hooks, and CLAUDE.md files are loaded.

**Why isolation is needed:** The `--tools ""` flag only disables built-in tools (Bash, Read, Edit, etc.). Without isolation, installed plugins (e.g., rust-analyzer-lsp, playwright) and MCP servers from the user's `~/.claude/` configuration still load and provide additional tools that the model can attempt to call. `CLAUDE_CONFIG_DIR` redirects all configuration reads to a clean directory, giving the CLI no plugins to discover.

**Live-tested (CLI v2.1.81):** Running `claude -p --tools "" --model haiku` **without** `CLAUDE_CONFIG_DIR` on a system with installed plugins produced a `system/init` event with 22 MCP tools (playwright browser_*, LSP) and 3 loaded plugins (rust-analyzer-lsp, playwright, codex) — despite `--tools ""`. The model had access to tools the proxy never intended to provide. With `CLAUDE_CONFIG_DIR` isolation, the same command produces `"tools":[]` and `"plugins":[]` as expected.

**Setup (performed once at startup):**

1. Create an isolated config directory at `<data-dir>/claude-config/` (e.g., `~/.local/share/claude-code-provider/claude-config/` on Linux, `~/Library/Application Support/claude-code-provider/claude-config/` on macOS). The data directory is resolved via `dirs::data_dir()` which returns the platform-appropriate location.
2. Symlink the credentials file: `<config-dir>/.credentials.json` → `~/.claude/.credentials.json`. This gives the CLI access to the user's OAuth tokens for Claude Max authentication without exposing any other configuration.
3. Pass `CLAUDE_CONFIG_DIR=<config-dir>` as an environment variable to every subprocess via `.env("CLAUDE_CONFIG_DIR", ...)`.

**What isolation eliminates:**

| Concern | Without Isolation | With Isolation |
|---------|-------------------|----------------|
| Built-in tools | Removed by `--tools ""` | Removed by `--tools ""` |
| Plugin tools (LSP, Playwright, etc.) | **Still loaded** | Eliminated |
| MCP server tools (local config) | Loaded from user config | Eliminated |
| MCP server tools (account-level) | Appear with `needs-auth` status | Appear with `needs-auth` status (harmless — no tools loaded) |
| Hooks | Loaded from user config | Eliminated |
| CLAUDE.md (user config) | Loaded from `~/.claude/` config | Eliminated |
| CLAUDE.md (working dir) | **Still loaded** from CWD and parents | Eliminated by setting CWD (see below) |
| System prompt tokens | ~8,500 (tools "" only) | ~4,500 (fully clean) |

**Docker deployment:** In a Docker container with a fresh Claude Code installation, there are no plugins installed at all. The `CLAUDE_CONFIG_DIR` approach is still recommended for defense-in-depth, but the primary isolation comes from the clean container. Users run `claude login` once inside the container to establish credentials, and the proxy uses that pristine environment.

**Credential symlink note:** The symlink target (`~/.claude/.credentials.json`) must exist before the first subprocess is spawned. On startup, the proxy should verify this file exists and exit with a clear error message if it does not (e.g., "Claude Code credentials not found. Run `claude login` first.").

**macOS credentials caveat:** On macOS, Claude Code may store credentials in the system Keychain rather than in `~/.claude/.credentials.json`. If the credentials file does not exist on macOS, the CLI may still authenticate via the Keychain. In this case, skip the symlink setup and rely on the CLI's own credential discovery. The startup `claude auth status` check will detect whether authentication is functional regardless of the credential storage mechanism. If `auth status` returns `loggedIn: true` but the credentials file doesn't exist, log a warning and proceed without the symlink — the CLI subprocess will inherit the parent environment's Keychain access.

**Config directory cleanup:** The CLI writes files into the config directory at runtime (`.claude.json` for cached feature flags, `mcp-needs-auth-cache.json`, `backups/`, `plugins/`, `projects/` subdirectories). Additionally, the CLI creates empty `memory/` subdirectories under `projects/<cwd-encoded>/memory/` even with `--no-session-persistence` — these are empty but accumulate over time. **Important:** On startup, delete ALL files and subdirectories in the config directory (including the `.credentials.json` symlink if it exists), then recreate the `.credentials.json` symlink unconditionally. This is simpler and more robust than trying to preserve the symlink during cleanup. Cached feature flags in `.claude.json` from a previous run can enable tools like `RemoteTrigger` that break the zero-tools guarantee. A fresh config directory on each startup ensures clean isolation.

**CLAUDE.md in working directory (live-tested):** `CLAUDE_CONFIG_DIR` isolation does NOT prevent CLAUDE.md files in the subprocess's working directory (or its parent directories) from being loaded. The CLI discovers CLAUDE.md via filesystem traversal from the CWD, independently of the config directory. In live testing, a CLAUDE.md placed in the CWD was loaded and injected into the system prompt, adding ~1,000 tokens and altering model behavior. **The fix:** Set the subprocess working directory to the isolated config directory itself (via `Command::current_dir(config.isolated_config_dir())`), which contains no CLAUDE.md. The `--working-dir` proxy config flag defaults to the isolated config dir, not the current directory. If users need a specific CWD (unlikely for a text-only proxy), they can override with `--working-dir`.

**Live-tested impact of stale `.claude.json`:** In testing (CLI v2.1.81), a second run using a stale `.claude.json` changed the CLI's behavior: different `slash_commands` appeared in the `system/init` message, and the cache tier shifted from `ephemeral_5m` to `ephemeral_1h` for `cache_creation_input_tokens`. While tools remained empty (the zero-tools guarantee held), the stale feature flags altered system prompt caching behavior. Cleaning on every startup is essential for deterministic behavior.

### Concurrency Control

A `tokio::sync::Semaphore` bounds the number of concurrent `claude` subprocesses:

- Default limit: **5** concurrent processes.
- Configurable via `--max-concurrent` CLI flag or `CCP_MAX_CONCURRENT` environment variable.
- When the limit is reached, new requests wait for a permit via `semaphore.acquire()`. Tokio's semaphore is approximately FIFO (waiters are woken in order) but does not strictly guarantee it.
- If the wait exceeds the queue timeout (default: 60 seconds), the request is rejected with HTTP 503 and a `Retry-After: 30` header. Use `tokio::time::timeout(Duration::from_secs(queue_timeout), semaphore.acquire())` for this.
- **Permit lifetime:** The `OwnedSemaphorePermit` must be moved into the subprocess reader task and held until the subprocess exits. This ensures the concurrency slot is occupied for the full duration of the request. Use `Arc<Semaphore>` and `semaphore.clone().acquire_owned()` to get an owned permit that can be moved across task boundaries. The permit is dropped automatically when the reader task completes (subprocess exits or is killed).
- **Spawn-then-delegate pattern:** The handler acquires the semaphore permit and spawns the subprocess synchronously (via `spawn_managed()`) BEFORE returning the response. If spawn or semaphore acquisition fails, the handler returns an HTTP error. If it succeeds, the reader task runs in the background, and the handler either collects events (non-streaming) or returns an SSE stream (streaming). The `spawn_managed` function signature:
```rust
pub async fn spawn_managed(
	config: Config,
	semaphore: Arc<Semaphore>,
	queue_timeout: Duration,
	cli_args: Vec<String>,
	tx: mpsc::Sender<SubprocessEvent>,
) -> Result<(), AppError>
```
It acquires the semaphore permit with timeout, then `tokio::spawn`s the reader task (which holds the permit until the subprocess exits). Returns `Ok(())` on successful spawn, `Err(AppError::ServiceUnavailable)` on queue timeout, `Err(AppError::ServerError)` on semaphore closed. The spawned reader task calls `run_subprocess(&config, cli_args, tx)` and holds the permit via `let _permit = permit;`.
- **Two distinct lifetimes:** The semaphore permit (concurrency slot) and the `ActiveRequestGuard` (active request counter) have different lifetimes. The permit lives in the reader task (dropped when subprocess exits). The `ActiveRequestGuard` lives in the handler (non-streaming) or the converter task (streaming) — it spans the time from request start to response completion. For streaming, the handler returns immediately after starting the SSE stream, so the guard must be in the converter task to track the full request-to-response-completion lifecycle.

### Process Lifecycle

1. **Spawn** — `tokio::process::Command` with stdout/stderr piped, stdin set to `Stdio::null()`, `kill_on_drop(true)`, `.env("CLAUDE_CONFIG_DIR", config.isolated_config_dir())`, `.current_dir(config.resolved_working_dir())`. The working directory defaults to the isolated config dir to prevent CLAUDE.md discovery from the host filesystem (see Process Isolation). There is no `--working-dir` or `--cwd` CLI flag on the `claude` binary; the CWD is set via `Command::current_dir()`. **Important:** stdin must be explicitly `Stdio::null()`, not `Stdio::piped()`. When stdin is a pipe, the CLI waits 3 seconds for stdin data before proceeding, adding unnecessary latency and logging a stderr warning. With `Stdio::null()`, the CLI immediately knows there is no piped input. All arguments are passed as separate `arg()` calls (not shell-concatenated), preventing shell injection. **`kill_on_drop(true)` is critical** — it ensures the subprocess is killed if the owning task is dropped (e.g., due to panic or cancellation), preventing orphan processes. **Spawn failure:** If `Command::spawn()` returns `Err` (e.g., binary not found at runtime, permission denied, OS resource exhaustion), return HTTP 500 immediately with a descriptive error. Do not retry — spawn failures are not transient. **Environment variable stripping:** The subprocess inherits the proxy's environment. To prevent loopback (where the CLI sends requests back to the proxy instead of using Claude Max OAuth), explicitly strip API-related environment variables: `.env_remove("ANTHROPIC_API_KEY").env_remove("ANTHROPIC_BASE_URL").env_remove("ANTHROPIC_AUTH_TOKEN")`. This is critical when users set `ANTHROPIC_BASE_URL` in their shell profile to route other tools through the proxy — without stripping, every subprocess would inherit that variable and loop requests back to itself, causing hangs. The meridian reference implementation does this unconditionally (not conditionally), and it is the correct approach. The cost is zero (removing absent vars is a no-op) and the protection against loopback is significant.
2. **Read** — Stdout and stderr read line-by-line via `BufReader::lines()` in a `tokio::select!` loop. Stdout lines are parsed as NDJSON. Stderr lines are logged at DEBUG level and accumulated in a `VecDeque<String>` buffer (last 50 lines, capped to prevent unbounded growth). This buffer is used for error messages when the process exits abnormally without a `result` message. Both stdout and stderr lines reset the inactivity timer. **Line buffer sizing:** NDJSON lines from the CLI can be very large (multi-KB content deltas, tool use results with file contents, base64-encoded data). The line reader must not impose a fixed buffer size limit. Rust's `BufReader::lines()` allocates dynamically per line, so this is handled correctly. Implementations in other languages (e.g., Go's `bufio.Scanner` with its 64KB default) must explicitly increase the buffer limit — the cliproxyapi reference sets a 50MB scanner buffer for this reason.

**Read loop skeleton:**
```rust
let stdout = BufReader::new(child.stdout.take().unwrap());
let stderr = BufReader::new(child.stderr.take().unwrap());
let mut stdout_lines = stdout.lines();
let mut stderr_lines = stderr.lines();
let inactivity = tokio::time::sleep(Duration::from_secs(config.timeout));
tokio::pin!(inactivity);
let progress = tokio::time::sleep(Duration::from_secs(30));
tokio::pin!(progress);
let mut stderr_buf: VecDeque<String> = VecDeque::new();

loop {
    tokio::select! {
        line = stdout_lines.next_line() => {
            match line {
                Ok(Some(line)) => {
                    inactivity.as_mut().reset(Instant::now() + Duration::from_secs(config.timeout));
                    // Parse NDJSON, emit SubprocessEvents via tx
                }
                Ok(None) => break, // stdout closed — process exiting
                Err(e) => { /* log error, break */ }
            }
        }
        line = stderr_lines.next_line() => {
            match line {
                Ok(Some(line)) => {
                    inactivity.as_mut().reset(Instant::now() + Duration::from_secs(config.timeout));
                    debug!(%line, "stderr");
                    if stderr_buf.len() >= 50 { stderr_buf.pop_front(); }
                    stderr_buf.push_back(line);
                }
                Ok(None) => {} // stderr closed — ignore, wait for stdout
                Err(_) => {}
            }
        }
        () = &mut inactivity => {
            warn!("Inactivity timeout");
            let _ = tx.send(SubprocessEvent::Error("Inactivity timeout".into())).await;
            let _ = child.kill().await;
            return; // Error sent before kill so consumer receives it
        }
        () = &mut progress => {
            info!(elapsed = ?start.elapsed(), lines, chunks, "Still running");
            progress.as_mut().reset(Instant::now() + Duration::from_secs(30));
        }
    }
}
// After loop: child.wait(), handle exit code, emit Error if no Result was sent
```

3. **Timeout** — Configurable inactivity timeout (default: 600 seconds / 10 minutes, set via `--timeout`). No stdout or stderr output for this duration. Subprocess is killed if exceeded.
4. **Progress** — For long-running requests, log a progress line every 30 seconds showing elapsed time, line count, and chunk count.
5. **Client disconnect** — Detected when the mpsc channel sender fails (receiver dropped). Subprocess is killed immediately via `child.kill()`.
6. **Completion** — On subprocess exit, capture exit code via `child.wait()`. Track a `got_result` boolean in the reader task — set to `true` when a `SubprocessEvent::Result` is sent. After the read loop, if `!got_result`, construct a `SubprocessEvent::Error` from the exit code and stderr buffer. If `got_result` is `true`, the exit code is redundant (even if non-zero, as in the invalid model case where `is_error: true` produces exit code 1). This suppression prevents duplicate error events from reaching the handler. Exit code `-1` if the wait itself fails. Signal-killed processes (e.g., from inactivity timeout) produce non-zero exit codes (137 for SIGKILL); this is expected and should not be logged as an unexpected error if the kill was initiated by the proxy.
7. **Cleanup** — On server shutdown, Axum drops in-flight request futures, which drops subprocess tasks, which triggers `kill_on_drop(true)` on each child process (sends SIGKILL).

---

## NDJSON Parsing

The Claude CLI outputs newline-delimited JSON objects on stdout. Each line is independently parseable. All lines share a common `"type"` field that identifies the message kind.

### Message Types

Every NDJSON line is a single tagged enum (`ClaudeCliMessage`) dispatched on `"type"`:

| `type` | Description | Action |
|--------|-------------|--------|
| `"system"` | Session init, API retries, hooks, compaction | See below |
| `"assistant"` | Complete assistant message with model info and content | Extract model name |
| `"stream_event"` | Streaming partial content (when `--include-partial-messages` is on) | Unwrap and forward text deltas |
| `"result"` | Final result with usage, duration, exit status | Map to OpenAI response |
| `"user"` | User-side message (e.g., tool_result in multi-turn scenarios) | **Ignored** — should not appear with `--tools ""` but handle gracefully |
| `"rate_limit_event"` | Rate limit status from Anthropic | **Ignored** — falls through to `Unknown` catch-all via `#[serde(other)]` |

Unknown `"type"` values must be silently ignored to ensure forward compatibility.

### Verified Event Sequence (Single-Turn)

The following sequence was observed in live testing (CLI v2.1.81, `--tools ""` with process isolation). This is the expected order for a single-turn response:

```
system  (subtype: init)                  # session metadata, model, tools
stream_event → message_start            # API message start, contains model
stream_event → content_block_start      # text block beginning
stream_event → content_block_delta      # text_delta × N (response tokens)
assistant                               # partial message snapshot (accumulated text so far)
stream_event → content_block_stop       # text block finished
stream_event → message_delta            # stop_reason: "end_turn"
stream_event → message_stop             # message complete
rate_limit_event                        # rate limit status from Anthropic
result                                  # final result with usage, duration, text
```

The `assistant` message (partial snapshot) appears mid-stream between content deltas and stop events. It contains accumulated text but is NOT the authoritative source for streaming — use `text_delta` events for that.

**Note:** Without `CLAUDE_CONFIG_DIR` isolation, `system` messages with subtypes `hook_started` and `hook_response` may appear **before** `system/init`. The parser must not assume `init` is the first NDJSON line. With isolation (the production configuration), hooks are not configured, so `system/init` is the first event as shown above.

### Multi-Turn Behavior (Defensive Handling)

With `--tools ""` and process isolation, the CLI should always complete in a single turn since no tools are available to propose. However, the parser should still handle multi-turn sequences defensively in case edge cases arise (e.g., incomplete isolation, future CLI behavior changes). In a multi-turn sequence:

1. Multiple `message_start` / `message_stop` pairs appear in the stream.
2. `user` type messages appear between turns (containing `tool_result` with denial info).
3. A `rate_limit_event` may appear between turns.
4. The `message_delta` for a tool-use turn has `stop_reason: "tool_use"` (not `"end_turn"`).

**The proxy must forward `text_delta` events from ALL turns**, not just the last one. The `result.result` field contains only the text from the **last turn**. For non-streaming mode, accumulate `text_delta` events from all turns. Fall back to `result.result` only if no deltas were collected.

### System Messages

Emitted at startup and during retries.

```json
{
	"type": "system",
	"subtype": "init",
	"cwd": "/path",
	"session_id": "...",
	"tools": [],
	"mcp_servers": [],
	"model": "claude-sonnet-4-6",
	"permissionMode": "default",
	"slash_commands": ["..."],
	"apiKeySource": "none",
	"claude_code_version": "2.1.81",
	"output_style": "default",
	"agents": ["general-purpose", "Explore", "Plan", "..."],
	"skills": ["..."],
	"plugins": [],
	"uuid": "...",
	"fast_mode_state": "off"
}
```

System message subtypes:
- `init` — Emitted once at startup. Contains session metadata. All fields present in observed output (CLI v2.1.81): `cwd` (string), `session_id` (UUID string), `tools` (array — empty with `--tools ""`), `mcp_servers` (array of objects), `model` (string), `permissionMode` (string), `slash_commands` (array of strings), `apiKeySource` (string, e.g., `"none"`), `claude_code_version` (string), `output_style` (string), `agents` (array of strings), `skills` (array of strings), `plugins` (array — empty with isolation), `uuid` (UUID string), `fast_mode_state` (string, e.g., `"off"`). Ignored for response translation except for extracting the `model` field as a fallback. With `CLAUDE_CONFIG_DIR` isolation, `tools` and `plugins` will be empty arrays. `mcp_servers` may contain account-level entries (e.g., `"claude.ai Gmail"`, `"claude.ai Google Calendar"`) from the OAuth credentials with `"status": "needs-auth"` — these are harmless (no tools are loaded from unauthenticated MCP servers).
- `api_retry` — Transient API failure, CLI is retrying. Log for visibility.
- `compact_boundary` — Context compaction occurred. Ignored.
- `hook_started` — A configured hook is executing. Ignored. (Not emitted with `CLAUDE_CONFIG_DIR` isolation since no hooks are configured.)
- `hook_response` — Hook execution completed. Ignored. (Not emitted with isolation.)

Unknown subtypes must be silently ignored.

API retry example (useful for logging):
```json
{
	"type": "system",
	"subtype": "api_retry",
	"attempt": 1,
	"max_retries": 5,
	"retry_delay_ms": 2000,
	"error_status": 529,
	"error": "server_error"
}
```

**Important:** `api_retry` events are informational only — the CLI handles retries internally. The proxy must NOT emit error events to the client when it sees `api_retry`. The retry will either succeed (and the response continues normally) or exhaust retries (and the CLI will exit with an error, which the proxy handles via the `result` message or non-zero exit code). Log `api_retry` events at WARN level for visibility but do not interrupt the response stream.

### Rate Limit Events

Emitted between turns (or after the last turn, before the result). Contains rate limit status from Anthropic. **Ignored** by the proxy but useful for logging.

```json
{
	"type": "rate_limit_event",
	"rate_limit_info": {
		"status": "allowed",
		"resetsAt": 1775275200,
		"rateLimitType": "five_hour",
		"overageStatus": "allowed",
		"overageResetsAt": 1775268000,
		"isUsingOverage": false
	},
	"uuid": "...",
	"session_id": "..."
}
```

The `rateLimitType` field varies (e.g., `"five_hour"`, `"seven_day_sonnet"`) depending on the active rate limit tier. All fields within `rate_limit_info` use camelCase.

### Assistant Messages

Contains model info and possibly inline content. Top-level keys: `type`, `message`, `parent_tool_use_id`, `session_id`, `uuid`. On error responses (e.g., invalid model), an additional top-level `error` field appears (e.g., `"error": "invalid_request"`).

```json
{"type": "assistant", "message": {"model": "claude-sonnet-4-6", "content": [{"type": "text", "text": "hello world"}], "id": "msg_...", "role": "assistant", "stop_reason": null, "stop_sequence": null, "stop_details": null, "usage": {...}, "context_management": null}, "parent_tool_use_id": null, "session_id": "...", "uuid": "..."}
```

The `message.model` field provides the actual model used (may differ from the requested model due to routing). The `message.content` array contains the accumulated text so far. The `message.stop_reason` may be `null` when emitted mid-stream (before the response is complete). Additional fields present on the message object: `stop_sequence` (always `null` in normal responses), `stop_details` (always `null` in normal responses), `context_management` (always `null` in observed testing). On synthetic error responses (e.g., invalid model), the message may also include `"container": null` and `"model": "<synthetic>"`. All of these are ignored by the proxy — only `model` and `content` are extracted.

The assistant message can appear **mid-stream** (interleaved with `stream_event` lines), and may appear multiple times in multi-turn sequences (e.g., once per turn). For the proxy, extract the model name from the `assistant` message (the implementation updates on every `Model` event; with `--tools ""` single-turn these all report the same model) but **do not re-emit content** from the assistant message — the `stream_event` deltas are the authoritative source for streaming text.

### Stream Events (Partial Messages)

When `--include-partial-messages` is used, the CLI emits streaming token events wrapped in a `stream_event` envelope. The raw Claude API event is nested inside the `event` field:

```json
{
	"type": "stream_event",
	"event": {
		"type": "content_block_delta",
		"index": 0,
		"delta": {"type": "text_delta", "text": "Hello"}
	},
	"parent_tool_use_id": null,
	"uuid": "...",
	"session_id": "..."
}
```

**Unwrapping:** Parse the top-level `stream_event`, then extract and dispatch on the inner `event` object. The inner `event.type` determines the streaming event kind.

The inner `event.delta.type` field determines what kind of content is being streamed. Only `text_delta` is forwarded to the client. All other delta types are skipped:

| Inner `event.delta.type` | Action | When It Appears |
|--------------------------|--------|-----------------|
| `text_delta` | **Forward** as SSE `delta.content` | Always (main response text) |
| `thinking_delta` | **Skip** | When `--effort` is active |
| `input_json_delta` | **Skip** | Should not appear with `--tools ""` but handle gracefully |

**Tool-related deltas:** With `--tools ""` and process isolation, the model has no tools and should not emit `tool_use` blocks or `input_json_delta` deltas. The parser skips all non-`text_delta` deltas regardless, so any unexpected tool-related events are harmless.

**Inner event types** (all except `content_block_delta` with `text_delta` are ignored):
- `message_start` — Message streaming started. Contains `message` object with `model`, `id`, `type`, `role`, empty `content`, `stop_reason` (null), `stop_sequence` (null), `stop_details` (null), `usage` (with `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`, `cache_creation` nested object, `output_tokens`, `service_tier`, `inference_geo`). Useful for extracting the model name. The usage here represents input-side token counts.
- `content_block_start` — New content block beginning. Contains `content_block.type` field (`"text"` or `"tool_use"`). For `tool_use`, also includes `name` and `id`.
- `content_block_delta` — Content delta. Check `delta.type` (see table above).
- `content_block_stop` — Content block finished.
- `message_delta` — Message-level update. Contains `delta.stop_reason` (`"end_turn"` or `"tool_use"`), `delta.stop_sequence` (usually `null`), `delta.stop_details` (usually `null`), `usage` (with `input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`), and `context_management` (with `applied_edits` array — always empty in observed testing; ignored by the proxy).
- `message_stop` — Message streaming completed. No payload.

### Result Messages

Final output with usage and exit status.

```json
{
	"type": "result",
	"subtype": "success",
	"is_error": false,
	"result": "The complete response text.",
	"stop_reason": "end_turn",
	"session_id": "...",
	"duration_ms": 1776,
	"duration_api_ms": 1729,
	"num_turns": 1,
	"total_cost_usd": 0.017064,
	"usage": {
		"input_tokens": 3,
		"output_tokens": 5,
		"cache_creation_input_tokens": 4528,
		"cache_read_input_tokens": 0,
		"server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0},
		"service_tier": "standard",
		"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 4528},
		"inference_geo": "",
		"iterations": [],
		"speed": "standard"
	},
	"modelUsage": {
		"claude-sonnet-4-6": {
			"inputTokens": 3,
			"outputTokens": 5,
			"cacheReadInputTokens": 0,
			"cacheCreationInputTokens": 4528,
			"webSearchRequests": 0,
			"costUSD": 0.017064,
			"contextWindow": 200000,
			"maxOutputTokens": 32000
		}
	},
	"permission_denials": [],
	"fast_mode_state": "off",
	"uuid": "..."
}
```

**Field naming conventions:**
- Top-level result fields use snake_case (`duration_ms`, `num_turns`, `total_cost_usd`, `is_error`, `stop_reason`, `session_id`, `permission_denials`, `fast_mode_state`, `uuid`), **except** `modelUsage` which is camelCase.
- The flat `usage` object uses snake_case (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`). It also contains nested objects (`server_tool_use` with `web_search_requests`/`web_fetch_requests`, `cache_creation` with `ephemeral_5m_input_tokens`/`ephemeral_1h_input_tokens`) and scalar fields (`service_tier`, `inference_geo`, `speed`), plus `iterations` (array).
- The `modelUsage` entries use **camelCase** (`inputTokens`, `outputTokens`, `cacheReadInputTokens`, `cacheCreationInputTokens`, `costUSD`). They also include `webSearchRequests`, `contextWindow`, and `maxOutputTokens`. **Implementation note:** Use `#[serde(rename_all = "camelCase")]` on the per-model usage struct. **Note:** The `contextWindow` and `maxOutputTokens` values in CLI output reflect subscription-tier limits (e.g., 200k context, 32k output for Sonnet under standard Max), which may be lower than the model's maximum capabilities (1M context, 64k output). These CLI-reported values are NOT used in the `/v1/models` response — our model list uses official Anthropic specs.
- **Note:** There is no `exitCode` field in the result NDJSON. The process exit code is obtained from `child.wait()`, not from the result message.

The result contains **two usage formats**: `modelUsage` (per-model, camelCase, includes cost and context window) and `usage` (flat aggregate, snake_case). Prefer `modelUsage` when present; fall back to `usage` if `modelUsage` is absent or empty.

**`modelUsage` key format:** The dictionary keys are model name strings. When the CLI uses the short model name (e.g., `--model sonnet`), the key is `"claude-sonnet-4-6"`. When the `[1m]` context suffix is used (e.g., `--model opus[1m]`), the key includes the suffix: `"claude-opus-4-6[1m]"`. Since our proxy always passes short names without the `[1m]` suffix, the keys will be standard canonical names in practice. However, the substring-based model normalization used for statistics handles the `[1m]` suffix correctly if it ever appears (e.g., due to subscription-level routing).

The `is_error` field is important: the `subtype` can be `"success"` while `is_error` is `true` (e.g., authentication failures, invalid model). Always check `is_error` in addition to `subtype`. When `is_error` is `true`, `modelUsage` is typically an empty object `{}` and `total_cost_usd` is `0`.

The `stop_reason` field on the result indicates how the CLI's response ended: `"end_turn"` for normal completion, `"stop_sequence"` for synthetic/error responses (e.g., invalid model). This is a CLI-level field, distinct from the `message_delta.delta.stop_reason` in the stream events.

Result subtypes and their meaning:

| `subtype` | Meaning | `finish_reason` |
|-----------|---------|-----------------|
| `success` | Normal completion | `"stop"` |
| `error_max_turns` | Hit maximum turn limit | `"stop"` |
| `error_during_execution` | Execution failure | (error) |
| `error_max_budget_usd` | Spending limit exceeded | (error) |
| `error_max_structured_output_retries` | JSON schema validation exhausted | (error) |

### Parsing Strategy

Each NDJSON line is deserialized with `serde` as a single tagged enum on `"type"`:

1. Parse as `ClaudeCliMessage` (tagged enum: `system`, `assistant`, `stream_event`, `result`, plus catch-all `Unknown` for types like `rate_limit_event`, `user`, etc.).
2. For `stream_event` variants, unwrap the inner `event` field and dispatch on `event.type`.
3. Empty lines are silently skipped. Unknown types and unparseable lines are logged at DEBUG and skipped.
4. Emit typed internal events via the mpsc channel:
```rust
enum SubprocessEvent {
	Model(String),               // Extracted from assistant or message_start
	ContentDelta(String),        // text_delta content
	Result(Box<ResultMessage>),  // Final result with usage (boxed — ResultMessage is large)
	Error(String),               // Error message
}
```
5. With `--tools ""` and process isolation, responses should always be single-turn. The parser should still handle multi-turn sequences defensively — forward `text_delta` from all turns and only treat the final `result` as the completion signal.

Use `#[serde(other)]` on the enum to handle unknown `"type"` values gracefully rather than failing deserialization.

**Internal channel buffer:** Use `mpsc::channel` with buffer size **64** for both channels (subprocess → converter, converter → SSE writer). This provides enough headroom for bursty token output without excessive memory use.

#### Reference Implementation Pitfall — Two-Layer Parsing

The `claude-max-api-proxy-rs` reference implementation uses a two-layer parsing strategy: it tries `ClaudeCliMessage` first, then falls back to parsing bare `StreamEvent` types (`content_block_delta`, etc.) at the top level. **This is incorrect for the current CLI (v2.1.81+).** Live testing confirms that ALL stream events are wrapped in a top-level `{"type": "stream_event", "event": {...}}` envelope — bare stream event types never appear at the top level. The correct approach is a single-layer parse into `ClaudeCliMessage`, with `stream_event` as a variant that contains the nested inner event.

#### Reference Implementation Pitfall — ModelUsage Field Names

The `claude-max-api-proxy-rs` reference deserializes `modelUsage` fields as snake_case (`input_tokens`, `output_tokens`). **This is incorrect.** Live testing confirms `modelUsage` uses camelCase (`inputTokens`, `outputTokens`, `cacheReadInputTokens`, `cacheCreationInputTokens`). The serde struct for per-model usage must use `#[serde(rename_all = "camelCase")]`.

#### Reference Implementation Pitfall — Non-Streaming Content

The `claude-max-api-proxy-rs` reference ignores `ContentDelta` events in non-streaming mode and uses only `result.result` for the response content. Our implementation accumulates all `text_delta` content even in non-streaming mode for correctness, falling back to `result.result` only if no deltas were collected.

---

## Response Translation

### Streaming (SSE)

The NDJSON events map to OpenAI SSE chunks:

| CLI Event | OpenAI SSE Action |
|-----------|-------------------|
| First `stream_event` with `text_delta` (across all turns) | Emit chunk with `delta.role: "assistant"` AND `delta.content: "<text>"` |
| Subsequent `stream_event` with `text_delta` (any turn) | Emit chunk with `delta.content: "<text>"` only |
| `stream_event` with `thinking_delta` | **Skip** — not forwarded to client |
| `stream_event` with other inner event types | **Skip** — internal stream lifecycle events |
| `result` with `is_error: true` | Emit error event with `result.result` as message, then `data: [DONE]` only if content was already streamed (checked first — takes priority over subtype). See Streaming Errors for details. |
| `result` with `is_error: false` and `subtype: "success"` | Emit final chunk with `finish_reason: "stop"`, usage chunk, then `data: [DONE]` |
| `result` with `is_error: false` and error subtype | Emit final chunk with `finish_reason: "stop"`, then `data: [DONE]` (log the error server-side) |
| Process error or non-zero exit | Emit error event, then close stream |

Each chunk includes:
- `id`: `chatcmpl-<8-hex-chars>` (random, stable for the request). Uses the `chatcmpl-` prefix to match OpenAI convention. Generate by taking the first 8 hex characters of a UUID v4: `format!("chatcmpl-{}", &Uuid::new_v4().to_string()[..8])`. The same ID is used for all SSE chunks in a single request and for the `x-request-id` header (without the `chatcmpl-` prefix).
- `object`: `"chat.completion.chunk"`.
- `created`: Unix timestamp (seconds) when the request started.
- `model`: The canonical model name (normalized from CLI output).
- `system_fingerprint`: Always `null`. Include in streaming chunks for consistency with the non-streaming response. Some OpenAI client libraries (e.g., the Python `openai` SDK) reference this field on chunk objects.

**Response headers:**
- All responses (streaming and non-streaming) include an `x-request-id` header set to the request's 8-hex-char ID. This aids debugging and log correlation.
- Streaming responses include `Cache-Control: no-cache` to prevent intermediary caching. Axum's `Sse` type handles `Content-Type: text/event-stream` automatically. Custom headers (like `x-request-id`) are added via tuple response: `(headers, sse).into_response()`.

**Initial SSE comment:** Before any data events, emit an SSE comment line (`: ok\n\n`) to signal the stream is established. This helps clients and load balancers detect a live connection immediately rather than waiting for the first content delta (which may take seconds due to CLI startup time). Use `Event::default().comment("ok")` as the first event in the stream. Note: Axum's `Event::comment()` produces `: ok` (with a space after the colon), matching the SSE spec's field format. The comment is emitted inside the converter task after `spawn_managed()` succeeds — if spawn fails, the client receives an HTTP error response, not an SSE stream.

**Usage in final chunk:** After the last content chunk and before `data: [DONE]`, emit a chunk with `usage` populated from the `result` message's `modelUsage` field (summing all entries if multiple models are present). This follows the OpenAI `stream_options.include_usage` convention. The usage chunk has empty `choices: []` and the `usage` object. If `extract_usage` returns `None` (both `modelUsage` and flat `usage` are absent — rare, typically only on error paths), the usage chunk is omitted entirely and the stream proceeds directly to `[DONE]`:

```json
data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1712345678,"model":"claude-sonnet-4-6","choices":[],"usage":{"prompt_tokens":25,"completion_tokens":8,"total_tokens":33}}
```

**`data: [DONE]` sentinel:** The stream terminator is literally `data: [DONE]\n\n` — it is NOT valid JSON. In Axum, emit it via `Event::default().data("[DONE]")`. This produces the correct SSE format. Do not attempt to serialize `[DONE]` as JSON.

**Keepalive:** Use `Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))` to send SSE comment lines (`: \n\n`) every 15 seconds. This is critical for keeping the connection alive through reverse proxies (nginx defaults to 60s proxy_read_timeout, Cloudflare to 100s) and HTTP clients with idle timeouts. Without keepalives, the connection may be terminated during the CLI's thinking phase (which can take 10+ seconds with `--effort max`) before any content is streamed.

**Client disconnect:** When the client disconnects, Axum drops the response stream. This drops the `mpsc::Receiver`, causing the subprocess task's `tx.send()` to return `Err`. The subprocess task then kills the child process. Flow: client disconnect → stream drop → receiver drop → sender fails → kill subprocess.

### Non-Streaming

For non-streaming requests (`stream: false`):

1. Collect all `stream_event` text deltas (from all turns) into a buffer.
2. Wait for the `result` message.
3. If `result.is_error` is `true`, return an HTTP 500 error response with `result.result` as the error message (e.g., "There's an issue with the selected model"). If `result.result` is `None`, use a generic message: `"CLI returned an error with no message"`. Do not return a `ChatCompletionResponse`.
4. Otherwise, construct a full `ChatCompletionResponse` with:
	- `choices[0].message.content` = accumulated text from all turns. Fall back to `result.result` if no deltas were collected (note: `result.result` only contains the last turn's text, so the accumulated deltas are preferred for multi-turn responses). If both are empty/None, use an empty string (some models may produce empty responses for certain prompts).
	- `choices[0].finish_reason` = mapped from result `subtype` (see table below).
	- `usage` = mapped from `modelUsage` in the result (preferred), or from the flat `usage` field as fallback.
5. Return as a single JSON response.

### Finish Reason Mapping

| Result `subtype` | OpenAI `finish_reason` |
|------------------|------------------------|
| `success` (or absent) | `"stop"` |
| `error_max_turns` | `"stop"` |
| `error_during_execution` | `"stop"` (log error server-side) |
| `error_max_budget_usd` | `"stop"` (log error server-side) |

**Important:** This table only applies when `is_error` is `false` (or absent). When `is_error` is `true`, the result is treated as an error regardless of subtype — the response is an HTTP 500 (non-streaming) or SSE error event (streaming), not a `ChatCompletionResponse` with a `finish_reason`. The `subtype` can be `"success"` while `is_error` is `true` (e.g., invalid model), so always check `is_error` first.

Note: The CLI does not currently report a distinct "max tokens reached" state, so `"length"` is not emitted. All completions use `"stop"`. If the CLI ever reports `stop_reason: "max_tokens"` (as the Anthropic API does when output is truncated), it should map to `finish_reason: "length"` — OpenAI clients like Continue.dev and Cursor use `"length"` to detect truncation and may auto-request continuations.

### Token Usage Mapping

| CLI Field (`modelUsage`, camelCase) | OpenAI Field |
|-------------------------------------|--------------|
| `inputTokens` | `prompt_tokens` |
| `outputTokens` | `completion_tokens` |
| (sum of above) | `total_tokens` |

`cacheReadInputTokens` and `cacheCreationInputTokens` from the CLI are logged for statistics but not included in the OpenAI response (no standard field exists).

Note: `modelUsage` may contain entries for multiple models (e.g., if the CLI internally used different models across turns). For the OpenAI response, sum all entries into a single `usage` object. For the stats dashboard, record usage per model separately.

If `modelUsage` is empty (e.g., on auth failure), fall back to the flat `usage` object which uses snake_case field names: `input_tokens`, `output_tokens`.

### Model Name Normalization

The CLI reports model names in several formats depending on the field:
- `system/init` → `"model": "claude-haiku-4-5-20251001"` (date-suffixed, reflects requested model)
- `assistant` and `message_start` → `"model": "claude-sonnet-4-6"` (short name, reflects actual model used — may differ from requested due to routing)
- `modelUsage` keys → `"claude-sonnet-4-6"` (short name, actual model)

The proxy normalizes all model names to our canonical names using substring matching:

- Contains `opus` → `claude-opus-4-6`
- Contains `sonnet` → `claude-sonnet-4-6`
- Contains `haiku` → `claude-haiku-4-5`
- Fallback → use the raw CLI model string as-is

The model name from the `assistant` message or `message_start` inner event (actual model used) takes precedence over the `system/init` model (requested model). This handles cases where Claude Max routes to a different model than requested. The implementation updates the model variable on every `Model` event (from both `assistant` and `message_start`). With `--tools ""` (always single-turn), these events report the same model, so first-vs-last is equivalent. If multi-turn support is added later, consider tracking only the first `Model` event.

**Fallback chain:** If no `Model` event is received (e.g., the CLI errors out before emitting an `assistant` or `message_start` event), fall back to the canonical model name resolved from the client's `model` field in the request. This ensures every response has a model name even on error paths.

---

## Error Handling

### HTTP Error Responses

All errors use OpenAI-compatible error format:

```json
{
	"error": {
		"message": "Description of what went wrong.",
		"type": "server_error",
		"code": null
	}
}
```

**Error type with IntoResponse:**
```rust
#[derive(Debug, thiserror::Error)]
enum AppError {
	#[error("{0}")]
	BadRequest(String),
	#[error("{0}")]
	NotFound(String),
	#[error("{0}")]
	ServerError(String),
	#[error("{0}")]
	Timeout(String),
	#[error("{0}")]
	ServiceUnavailable(String),
}

impl IntoResponse for AppError {
	fn into_response(self) -> Response {
		let (status, error_type) = match &self {
			AppError::BadRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error"),
			AppError::NotFound(_) => (StatusCode::NOT_FOUND, "invalid_request_error"),
			AppError::ServerError(_) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
			AppError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "server_error"),
			AppError::ServiceUnavailable(_) => (StatusCode::SERVICE_UNAVAILABLE, "server_error"),
		};
		let body = serde_json::json!({
			"error": { "message": self.to_string(), "type": error_type, "code": null }
		});
		let mut resp = (status, Json(body)).into_response();
		if matches!(self, AppError::ServiceUnavailable(_)) {
			resp.headers_mut().insert("retry-after", "30".parse().unwrap());
		}
		resp
	}
}
```

For `ServiceUnavailable`, also include a `Retry-After: 30` header in the response.

### Error Conditions

| Condition | HTTP Status | `error.type` | Action |
|-----------|-------------|--------------|--------|
| Invalid JSON body | 400 | `invalid_request_error` | Return immediately. **Implementation note:** Use `body: Bytes` as the Axum extractor (not `Json<T>`) and call `serde_json::from_slice()` manually. Axum's `Json<T>` extractor returns its own 422 error format, not our OpenAI-compatible 400 format. Manual parsing gives us control over the error response. |
| Missing or empty `model` field | 400 | `invalid_request_error` | Return immediately |
| Empty `messages` array | 400 | `invalid_request_error` | Return immediately |
| No user messages after filtering | 400 | `invalid_request_error` | Return immediately. All messages were system role or had empty content. |
| Invalid `reasoning_effort` value | 400 | `invalid_request_error` | Return immediately. Valid values: `none`, `low`, `medium`, `high`, `max`, or absent. Any other string is invalid. |
| Prompt too large (>128KB) | 400 | `invalid_request_error` | Return immediately with size information. Linux `MAX_ARG_STRLEN` is ~128KB per CLI argument. |
| System prompt too large (>128KB) | 400 | `invalid_request_error` | Return immediately with size information |
| Queue timeout (60s) | 503 | `server_error` | Include `Retry-After: 30` header |
| CLI not found in PATH | 500 | `server_error` | Log critical error; return helpful message |
| CLI subprocess spawn failure | 500 | `server_error` | Log error with spawn error details |
| CLI exited with no output | 500 | `server_error` | Process exited immediately without producing any NDJSON. Collect and include stderr output. Common cause: auth failure or invalid arguments. **(Enhancement — not yet implemented:** If stderr contains "Not logged in" or "OAuth token has expired", append a hint: "Run `claude login` to re-authenticate."**)**  |
| CLI non-zero exit without result | 500 | `server_error` | Include stderr output in message. Exit code 1 frequently indicates an auth failure. |
| `result` with `is_error: true` | 500 | `server_error` | Use `result.result` as error message. **(Enhancement — not yet implemented:** If the message contains "Not logged in", append the `claude login` hint.**)** |
| Inactivity timeout (10 min) | 504 | `server_error` | Kill subprocess, log warning. **Implementation note:** The reader task sends `SubprocessEvent::Error("Inactivity timeout".into())` — the error type information (504 vs 500) is lost through the string-typed channel. The handler recovers it by checking `if err.contains("Inactivity timeout")` and producing `AppError::Timeout` (504) instead of `AppError::ServerError` (500). This string-matching approach is intentional — adding a typed error variant to `SubprocessEvent` would complicate the enum for a single case. |
| Internal panic / unexpected error | 500 | `server_error` | Log full backtrace |

### Streaming Errors

If an error occurs after streaming has started (SSE headers already sent):

1. **Before any content was sent** (e.g., subprocess spawn failure, immediate timeout, `result` with `is_error: true` when no content deltas were emitted): Emit an error event `data: {"error": {"message": "...", "type": "server_error"}}`, then close the stream (no `data: [DONE]`).
2. **After content was already streamed** (e.g., `result` with `is_error: true` after content deltas were sent, late timeout): Emit an error event, then `data: [DONE]` to cleanly close the stream. The client already has partial content and needs a proper termination signal.

The decision between case 1 and 2 is based on whether any `ContentDelta` events were forwarded to the client (tracked via a `content_sent` boolean), not on the event type that triggered the error.

Clients should treat an abrupt stream close without `data: [DONE]` as an error.

**Retry invariant (for future retry logic):** Once any content has been committed to the client (a `ContentDelta` forwarded as an SSE data event), the response is committed and must NOT be retried. Retrying after partial content delivery would duplicate or corrupt the response. This is a hard constraint — any future retry logic (e.g., for auth token refresh or rate limit recovery) must check the `content_sent` flag and suppress retries when true. The meridian reference implementation enforces this with `didYieldContent` / `didYieldClientEvent` booleans that unconditionally re-throw errors once content has been yielded.

---

## Configuration

### CLI Arguments

```
claude-code-provider [OPTIONS]

Options:
	-p, --port <PORT>              Listen port [default: 18321]
	-H, --host <HOST>              Listen address [default: 127.0.0.1]
	-c, --max-concurrent <N>       Max concurrent subprocess [default: 5]
	-t, --timeout <SECONDS>        Subprocess inactivity timeout [default: 600]
	-q, --queue-timeout <SECONDS>  Max time a request waits in queue [default: 60]
	--claude-path <PATH>           Path to claude CLI binary [default: "claude"]
	--data-dir <PATH>              Data directory for config isolation and stats DB
	                               [default: platform data dir / claude-code-provider]
	                               Linux: ~/.local/share/claude-code-provider
	                               macOS: ~/Library/Application Support/claude-code-provider
	--working-dir <PATH>           Working directory for subprocess [default: <data-dir>/claude-config]
	                               (Note: This is OUR config flag, not a claude CLI flag. The working
	                               directory is passed to the subprocess via Command::current_dir().
	                               Defaults to the isolated config dir to prevent CLAUDE.md loading
	                               from the host filesystem. Override only if needed.)
	-v, --verbose                  Enable debug logging
	-h, --help                     Print help
	-V, --version                  Print version
```

### Environment Variables

All CLI arguments (except `--verbose`) can also be set via environment variables with `CCP_` prefix:

| Variable | CLI Equivalent |
|----------|----------------|
| `CCP_PORT` | `--port` |
| `CCP_HOST` | `--host` |
| `CCP_MAX_CONCURRENT` | `--max-concurrent` |
| `CCP_TIMEOUT` | `--timeout` |
| `CCP_QUEUE_TIMEOUT` | `--queue-timeout` |
| `CCP_CLAUDE_PATH` | `--claude-path` |
| `CCP_DATA_DIR` | `--data-dir` |
| `CCP_WORKING_DIR` | `--working-dir` (defaults to `<data-dir>/claude-config`) |

CLI arguments take precedence over environment variables.

### Config Struct

The `Config` struct is both the clap `Parser` target and the runtime configuration. Optional fields (`data_dir`, `working_dir`) use `Option<PathBuf>` with post-parse resolution methods:

```rust
#[derive(Parser, Clone, Debug)]
#[command(name = "claude-code-provider", version, about = "OpenAI-compatible API proxy backed by Claude Code CLI")]
pub struct Config {
	#[arg(short = 'p', long, default_value = "18321", env = "CCP_PORT")]
	pub port: u16,
	#[arg(short = 'H', long, default_value = "127.0.0.1", env = "CCP_HOST")]
	pub host: String,
	#[arg(short = 'c', long, default_value = "5", env = "CCP_MAX_CONCURRENT")]
	pub max_concurrent: usize,
	#[arg(short = 't', long, default_value = "600", env = "CCP_TIMEOUT")]
	pub timeout: u64,
	#[arg(short = 'q', long, default_value = "60", env = "CCP_QUEUE_TIMEOUT")]
	pub queue_timeout: u64,
	#[arg(long, default_value = "claude", env = "CCP_CLAUDE_PATH")]
	pub claude_path: String,
	#[arg(long, env = "CCP_DATA_DIR")]
	pub data_dir: Option<PathBuf>,
	#[arg(long, env = "CCP_WORKING_DIR")]
	pub working_dir: Option<PathBuf>,
	#[arg(short = 'v', long)]
	pub verbose: bool,
}

impl Config {
	pub fn resolved_data_dir(&self) -> PathBuf {
		self.data_dir.clone().unwrap_or_else(|| {
			dirs::data_dir().expect("Could not determine data directory")
				.join("claude-code-provider")
		})
	}
	pub fn isolated_config_dir(&self) -> PathBuf {
		self.resolved_data_dir().join("claude-config")
	}
	pub fn resolved_working_dir(&self) -> PathBuf {
		self.working_dir.clone().unwrap_or_else(|| self.isolated_config_dir())
	}
	pub fn stats_db_path(&self) -> PathBuf {
		self.resolved_data_dir().join("stats.redb")
	}
}
```

**Why `Option<PathBuf>` for `data_dir` and `working_dir`:** These have dynamic defaults that depend on the runtime platform (`dirs::data_dir()`) or on each other (`working_dir` defaults to `isolated_config_dir()`, which depends on `data_dir`). Clap's `default_value` only accepts static strings, so these are `None` at parse time and resolved via methods.

### Startup Validation

On startup, the server:

1. Verifies `claude` binary exists and is executable.
2. Runs `claude --version` to confirm CLI is functional and logs the version.
3. **Checks authentication status** by running `claude auth status`. The output is JSON: `{"loggedIn": true, "authMethod": "claude.ai", "subscriptionType": "max", ...}`. Parse the `loggedIn` field. If `false`, exits with: `"Claude Code is not logged in. Run 'claude login' first."`. Log the `subscriptionType` at INFO level for visibility (e.g., "Authenticated as max subscriber"). This catches auth issues early rather than failing on the first request. The check is best-effort — if parsing fails, log a warning and continue (the first request will surface any auth problems).
4. **Sets up the isolated config directory:**
   - Creates `<data-dir>/claude-config/` if it does not exist (default: `~/.local/share/claude-code-provider/claude-config/`).
   - **Cleans the config directory:** Removes all files and subdirectories (including any existing `.credentials.json` symlink). This prevents stale cached feature flags (from `.claude.json`) from enabling unexpected tools on restart.
   - Verifies the credentials source file exists at `~/.claude/.credentials.json`. If it does not exist: on **macOS**, log a warning and skip the symlink (credentials may be in the system Keychain — the `claude auth status` check in step 3 already confirmed auth works). On **Linux**, exit with: `"Claude Code credentials not found at ~/.claude/.credentials.json. Run 'claude login' first."`. This check happens before symlink creation — if the source doesn't exist on Linux, there's nothing to symlink to.
   - Creates a fresh symlink `<config-dir>/.credentials.json` → `~/.claude/.credentials.json` (skipped on macOS if the source file does not exist).
5. **Opens the stats database** at `<data-dir>/stats.redb`. Creates it if it does not exist. Initializes tables on first access.
6. **Constructs `AppState`** and wraps it in `Arc`:
```rust
struct AppState {
	config: Config,                        // Resolved configuration
	semaphore: Arc<Semaphore>,             // Concurrency limiter
	stats: Arc<Stats>,                     // Statistics (redb + in-memory)
}
```
Pass `AppState` to Axum via `.with_state(Arc::new(state))`. All route handlers receive `State(state): State<Arc<AppState>>`.
7. Binds to the configured host:port (exits with error if port is in use).
8. Logs the full configuration (port, host, concurrency, timeouts, isolated config dir path, stats DB path).

---

## Statistics Collection

### Persistent Metrics (redb)

Statistics persist across server restarts using `redb` (same pattern as `~/ckb-mcp`). The database file is stored at `<data-dir>/stats.redb` (default: `~/.local/share/claude-code-provider/stats.redb`).

**Hybrid approach:** Persistent counters (lifetime totals) are stored in redb. Volatile data (active requests, TTFT/duration samples, recent errors) is stored in-memory only since it's session-scoped.

```rust
use redb::{Database, TableDefinition};

// redb table definitions (persistent across restarts)
const TOTAL_REQUESTS: TableDefinition<&str, u64> = TableDefinition::new("total_requests");
const REQUESTS_BY_MODEL: TableDefinition<&str, u64> = TableDefinition::new("requests_by_model");
const TOTAL_ERRORS: TableDefinition<&str, u64> = TableDefinition::new("total_errors");
const TOKENS_BY_MODEL: TableDefinition<&str, &[u8]> = TableDefinition::new("tokens_by_model");
// tokens_by_model values are JSON-serialized TokenStats

struct Stats {
	db: Database,
	// In-memory only (session-scoped, not persisted)
	active_requests: AtomicU64,
	started_at: Instant,
	recent_errors: Mutex<VecDeque<ErrorRecord>>,     // last 50
	ttft_samples: Mutex<HashMap<String, VecDeque<f64>>>,     // per model, last 100
	duration_samples: Mutex<HashMap<String, VecDeque<f64>>>,  // per model, last 100
}

#[derive(Serialize, Deserialize)]
struct TokenStats {
	input_tokens: u64,
	output_tokens: u64,
	cache_read_input_tokens: u64,
	cache_creation_input_tokens: u64,
}

struct ErrorRecord {
	timestamp: String,  // ISO 8601
	model: String,
	message: String,
}
```

**Initialization (following ckb-mcp pattern):**
```rust
impl Stats {
	fn open(path: impl AsRef<Path>) -> Result<Self> {
		if let Some(parent) = path.as_ref().parent() {
			std::fs::create_dir_all(parent)?;
		}
		let db = Database::create(path)?;
		// Create tables by opening them (redb creates on first access)
		let write_txn = db.begin_write()?;
		let _ = write_txn.open_table(TOTAL_REQUESTS);
		let _ = write_txn.open_table(REQUESTS_BY_MODEL);
		let _ = write_txn.open_table(TOTAL_ERRORS);
		let _ = write_txn.open_table(TOKENS_BY_MODEL);
		write_txn.commit()?;
		Ok(Self { db, active_requests: AtomicU64::new(0), ... })
	}
}
```

**Usage:** The `Stats` struct is wrapped in `Arc<Stats>` and passed as Axum state. Increment operations use `db.begin_write()` transactions. Read operations (for stats page) use `db.begin_read()`. redb is thread-safe — no `Mutex` needed around the `Database` handle.

**Recording points:**
- **Request count** (persistent): Incremented when the request handler begins (before subprocess spawn). This ensures requests that fail to spawn are still counted.
- **Active requests** (in-memory): Tracked via an `ActiveRequestGuard` RAII type that increments on creation and decrements on drop. **Non-streaming:** Created at the start of the handler, dropped when the handler returns. **Streaming:** Created inside the spawned converter task (not the handler), so it spans the full request-to-response-completion lifecycle — the handler itself returns immediately after starting the SSE stream. Use `AtomicU64::fetch_add` / `fetch_sub`.
- **TTFT** (in-memory): Measured in the handler (non-streaming) or converter task (streaming) when the first `ContentDelta` event is received from the channel. Elapsed time is measured from a local `Instant::now()` captured after `spawn_managed()` returns. The reader task does NOT record TTFT — it only sends `SubprocessEvent::ContentDelta` through the channel. The receiving side measures wall-clock time to first delta. This approach avoids passing `Arc<Stats>` into the reader task.
- **Duration** (in-memory): Measured in the handler (non-streaming) or converter task (streaming) as total elapsed time from the same local `Instant::now()` to when the `Result` event is received or the channel closes. Not measured in the reader task.
- **Tokens** (persistent): Recorded from the `result` message's `modelUsage` field. Only recorded on successful completion (when `is_error` is false or absent). **Important:** The `modelUsage` keys are raw CLI model names (e.g., `"claude-haiku-4-5-20251001"` with date suffix). These must be normalized via `normalize_model_name()` before storing in the stats DB, so that token counts aggregate under canonical names (matching the request count keys). Without normalization, the stats page shows ghost entries with date-suffixed names that have tokens but zero requests.
- **Errors** (persistent count + in-memory recent): Recorded when a request ends in error (subprocess failure, `is_error: true`, timeout, etc.).

### Stats Dashboard (`GET /stats`)

Plain HTML page, auto-refreshes every 10 seconds via `<meta http-equiv="refresh" content="10">`. No JavaScript frameworks — plain HTML with inline `<style>` tags. Displays:

- **Header:** Server name (`Claude Code Provider`), version (from `Cargo.toml`), uptime (human-readable, e.g., "2h 15m 30s").
- **Summary cards:** Total requests, active requests, error count, error rate %.
- **Per-Model Table:** Model name, request count, avg TTFT (ms), avg duration (ms), input tokens total, output tokens total, cache read tokens, cache creation tokens.
- **Recent Errors:** Timestamp (ISO 8601), model, error message (last 10 shown, newest first).
- **Token Efficiency:** Cache read ratio = `cache_read / (cache_read + cache_creation + input)` as a percentage.

The HTML is generated as a `String` in the stats route handler using `format!()` — no template engine needed. Use a monospace font and a simple table layout. The page should be functional, not pretty.

---

## Logging

Uses the `tracing` crate with structured fields.

### Log Levels

| Level | Content |
|-------|---------|
| `ERROR` | Subprocess failures, spawn errors, unexpected panics |
| `WARN` | Inactivity timeouts, client disconnects, unknown model names, queue timeouts |
| `INFO` | Request start/complete (model, duration, tokens), server start/stop |
| `DEBUG` | NDJSON line parsing, CLI arguments, prompt construction |
| `TRACE` | Raw NDJSON lines, SSE events emitted |

### Per-Request Fields

Every log line within a request context includes:

- `request_id`: Random 8-character hex string.
- `model`: Requested model name.
- `stream`: Whether streaming is enabled.

Use a `tracing::info_span!` at the start of the request handler and `instrument` or `.enter()` to attach these fields to all log lines within the request scope:

```rust
let span = tracing::info_span!("request", request_id = %id, model = %model, stream = %stream);
let _guard = span.enter();
```

---

## Limitations

These are known limitations of the CLI subprocess approach:

1. **Latency overhead** — Each request spawns a new process (~100-500ms startup). This is unavoidable with the subprocess approach.
2. **No stop sequences** — The `claude -p` CLI does not accept custom stop sequences. The `stop` parameter is accepted but silently ignored.
3. **No `max_tokens` / `max_completion_tokens` control** — The CLI does not expose a `--max-tokens` flag. Both OpenAI parameters are accepted for client compatibility but not forwarded.
4. **No `temperature` control** — The CLI does not expose a `--temperature` flag. The parameter is accepted but not forwarded.
5. **No multimodal** — Only text content is supported. Image URLs or base64 images in messages are ignored with a warning logged.
6. **No tool use / function calling** — The proxy is text generation only. Tool use fields in the request are ignored.
7. **No `n` parameter** — Only one completion per request (OpenAI's `n` parameter is ignored).
8. **No `logprobs`** — Log probabilities are not available through the CLI.
9. **No `frequency_penalty`, `presence_penalty`, `top_p`** — These sampling parameters have no CLI equivalent. Accepted but ignored.
10. **Rate limits are Claude Max plan limits** — The proxy does not add its own rate limiting. Anthropic's server-side limits for Claude Max apply.
11. **Model may reference tools in text** — With zero tools available, the model may occasionally produce text that looks like tool invocations (e.g., XML-like `<invoke>` blocks) when asked to do something that would normally require tools (like reading a file). This is harmless generated text, not actual tool execution.
12. **Haiku model routing** — In earlier live testing with Claude Max, requesting `haiku` resulted in `modelUsage` showing `claude-sonnet-4-6` instead of haiku (server-side routing). In later testing, haiku requests were served by `claude-haiku-4-5-20251001` as expected. This routing behavior is **intermittent and server-side** — Claude Max may route haiku requests to sonnet under load. The proxy should accept and forward haiku requests regardless. The response model name reflects the actual model used (from `message_start` or `modelUsage`), not the requested one. The `modelUsage` key may include a date suffix (e.g., `"claude-haiku-4-5-20251001"`), which the proxy normalizes to `claude-haiku-4-5` via substring matching.
13. **Argument length limit** — The conversation prompt and system prompt are each passed as separate CLI arguments. Linux has a ~128KB per-argument limit (`MAX_ARG_STRLEN` = 128 * 1024 = 131,072 bytes) and a total `ARG_MAX` of ~2-3MB for all arguments combined. If an individual argument exceeds `MAX_ARG_STRLEN`, `Command::spawn()` fails with `E2BIG` ("Argument list too long"). The proxy checks both `prompt.len() > 128_000` and `system_prompt.len() > 128_000` (~128KB each) before spawning and returns a 400 error, preventing spawn failures from the per-argument OS limit. The spawn error handler in process.rs provides additional safety in case any argument unexpectedly exceeds the OS limit. In practice, conversation prompts exceeding 128KB are unusual for text-only interactions (128KB is ~30,000 words). If this becomes an issue, the prompt can be piped via stdin (future enhancement using `--input-format stream-json`).

---

## CORS

The proxy includes permissive CORS headers on all responses to support browser-based clients (custom web apps, Open WebUI, etc.):

- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type, Authorization`

All `OPTIONS` preflight requests return `204 No Content` with the above headers.

Use `tower_http::cors::CorsLayer::permissive()` as a layer on the router. This sets all the above headers automatically and handles `OPTIONS` preflight. One line: `.layer(CorsLayer::permissive())`.

This is appropriate for a local proxy. If exposed externally behind a reverse proxy, the reverse proxy should override these with stricter CORS policies.

---

## Thinking Content in Responses

When `reasoning_effort` is set to a non-`none` value, Claude may produce thinking/reasoning content alongside the text response. The CLI's NDJSON output may include `stream_event` lines whose inner `event` is a `content_block_delta` with `delta.type: "thinking_delta"` in addition to `"text_delta"`.

**Note:** In live testing (CLI v2.1.81), `--effort max` on Sonnet did NOT produce `thinking_delta` events — thinking occurred server-side but was not streamed through the CLI. This behavior may differ with Opus or future CLI versions. The proxy should still handle `thinking_delta` gracefully (skip it) in case future versions do stream it.

### Handling Strategy

**Streaming:** Thinking deltas are **excluded** from the SSE stream by default. Only `text_delta` events are forwarded as `delta.content` chunks. This is because the OpenAI Chat Completions format has no standard field for streaming thinking content.

**Non-streaming:** The `result.result` field from the CLI contains only the final text output (not thinking). Use this as the response content. If thinking content is needed in the future, it can be added as a `reasoning_content` field on the choice message (following OpenRouter's convention).

**Statistics:** Thinking tokens are counted within `outputTokens` in the CLI's `modelUsage`. The stats dashboard displays total output tokens inclusive of thinking.

---

## Security Considerations

1. **Local-only by default** — Binds to `127.0.0.1`. Users must explicitly set `--host 0.0.0.0` to expose externally.
2. **No authentication** — The proxy does not require an API key. It accepts and ignores any `Authorization` header. This is acceptable for local use; external exposure should use a reverse proxy with auth.
3. **No request body modification** — The proxy does not inject, modify, or store any credentials.
4. **Subprocess isolation** — `--tools ""` combined with `CLAUDE_CONFIG_DIR` isolation removes all tools. The CLI cannot execute tools, write files, or run commands because no tools exist in the session.
5. **No secrets in logs** — Request/response content is only logged at TRACE level.
6. **Request body size limit** — 10 MB maximum request body size. Prevents memory exhaustion from oversized requests. Use `.layer(DefaultBodyLimit::max(10 * 1024 * 1024))` on the router. Note: When this limit is exceeded, Axum returns a 413 Payload Too Large with its own plain-text error format (not our OpenAI JSON format), because the body limit middleware rejects the request before our handler runs. This is acceptable for this edge case.

---

## Project Structure

```
claude-code-provider/
├── Cargo.toml
├── src/
│   ├── main.rs              # CLI args, Axum server launch, startup validation
│   ├── config.rs            # Configuration struct, env var loading
│   ├── routes/
│   │   ├── mod.rs
│   │   ├── completions.rs   # POST /v1/chat/completions handler
│   │   ├── models.rs        # GET /v1/models handler
│   │   ├── stats.rs         # GET /stats, GET /stats/json handlers
│   │   └── health.rs        # GET /health handler
│   ├── subprocess/
│   │   ├── mod.rs
│   │   ├── manager.rs       # Semaphore-bounded process spawning
│   │   ├── process.rs       # Single subprocess lifecycle
│   │   └── ndjson.rs        # NDJSON line parsing, ClaudeCliMessage enum, stream_event unwrapping
│   ├── translate/
│   │   ├── mod.rs
│   │   ├── request.rs       # OpenAI request → CLI args + prompt
│   │   ├── response.rs      # CLI events → OpenAI response (non-streaming)
│   │   └── stream.rs        # CLI events → OpenAI SSE chunks (streaming)
│   ├── models.rs            # Model name mapping and validation
│   ├── stats.rs             # Stats collector (redb persistence)
│   └── error.rs             # Error types and OpenAI error format
├── SPEC.md                  # This file
├── README.md                # Project overview, setup instructions, API docs
├── test_openai_client.py    # OpenAI Python SDK integration tests
└── research/                # Reference implementations (not compiled)
```

---

## Runtime Note

The server uses Axum with tokio as its async runtime. All subprocess spawning uses `tokio::process::Command`. The reference implementation (`claude-max-api-proxy-rs`) uses this exact stack successfully.

**Handler return type:** The completions handler returns `Result<Response, AppError>`. Both streaming (`Sse`) and non-streaming (`Json`) responses are converted to `Response` via `.into_response()`. The handler signature:
```rust
pub async fn completions_handler(
	State(state): State<Arc<AppState>>,
	body: Bytes,
) -> Result<Response, AppError>
```
This works because `AppError` implements `IntoResponse` (Axum automatically converts `Err` variants), and both `(headers, Sse<...>).into_response()` and `(headers, Json<...>).into_response()` produce `Response`. The handler delegates to `handle_streaming()` or `handle_non_streaming()` which share the same return type.

**SSE streaming:** Axum provides `axum::response::sse::Sse` with built-in keepalive support. The architecture uses two `mpsc::channel`s (same pattern as the reference implementation):

```
Subprocess stdout ──→ [reader task] ──→ SubprocessEvent channel ──→ [converter task] ──→ SSE Event channel ──→ Client
```

```rust
// Channel 1: subprocess reader → event converter
let (sub_tx, mut sub_rx) = mpsc::channel::<SubprocessEvent>(64);

// Channel 2: event converter → SSE writer
let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, Infallible>>(64);

// Spawn subprocess (acquires semaphore, spawns reader task internally).
// Returns Err(AppError) on queue timeout or spawn failure — handler
// returns HTTP error BEFORE the SSE stream starts.
spawn_managed(config.clone(), semaphore.clone(), queue_timeout, cli_args, sub_tx).await?;

// Task 2: Convert SubprocessEvents → SSE Events
// State tracked across events:
let request_id = request_id.clone();        // stable for all chunks
let created = created;                       // unix timestamp
let mut model = requested_model.clone();     // updated from Model events
let mut is_first = true;                     // role emitted on first delta only
let mut content_sent = false;                // tracks whether any content was streamed

tokio::spawn(async move {
	let _ = sse_tx.send(Ok(Event::default().comment("ok"))).await;
	while let Some(event) = sub_rx.recv().await {
		match event {
			SubprocessEvent::Model(m) => { model = m; }
			SubprocessEvent::ContentDelta(text) => {
				content_sent = true;
				// Build ChunkDelta with role only if is_first
				// Serialize to JSON, wrap in Event::default().data(json)
				is_first = false;
			}
			SubprocessEvent::Result(result) => {
				// 1. Emit finish chunk (finish_reason: "stop")
				// 2. Emit usage chunk (choices: [], usage: {...})
				// 3. Emit Event::default().data("[DONE]")
			}
			SubprocessEvent::Error(msg) => {
				// Emit error event, optionally [DONE] if content_sent
			}
		}
	}
});

// Return SSE stream
let stream = ReceiverStream::new(sse_rx);
let sse = Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)));
let headers = [(header::HeaderName::from_static("x-request-id"), request_id)];
(headers, sse).into_response()
```

**Why two channels?** A single `SubprocessEvent::Result` produces multiple SSE events (finish chunk, usage chunk, `[DONE]` sentinel). The converter task can emit any number of SSE events per subprocess event, which a simple `stream.map()` cannot.

**Non-streaming:** Collect all subprocess events, then return a single JSON response:
```rust
let headers = [(header::HeaderName::from_static("x-request-id"), request_id)];
(headers, Json(response)).into_response()
```

**Initial SSE comment:** Before any data events, emit `: ok\n\n` via `Event::default().comment("ok")` as the first event.

**Server setup:**
```rust
let app = Router::new()
	.route("/v1/chat/completions", post(completions_handler))
	.route("/v1/models", get(models_handler))
	.route("/stats", get(stats_html_handler))
	.route("/stats/json", get(stats_json_handler))
	.route("/health", get(health_handler))
	.fallback(fallback_handler)  // Returns 404 with OpenAI error format
	.layer(CorsLayer::permissive())
	.layer(DefaultBodyLimit::max(10 * 1024 * 1024))
	.with_state(state);

// Fallback returns NotFound for any unregistered path
async fn fallback_handler() -> AppError {
	AppError::NotFound("The requested endpoint does not exist".into())
}
```

**Graceful shutdown:**
```rust
let shutdown = async {
	let ctrl_c = tokio::signal::ctrl_c();
	#[cfg(unix)]
	{
		let mut sigterm = tokio::signal::unix::signal(
			tokio::signal::unix::SignalKind::terminate()
		).expect("failed to install SIGTERM handler");
		tokio::select! {
			_ = ctrl_c => { info!("Received SIGINT, shutting down..."); }
			_ = sigterm.recv() => { info!("Received SIGTERM, shutting down..."); }
		}
	}
	#[cfg(not(unix))]
	{
		ctrl_c.await.ok();
		info!("Received SIGINT, shutting down...");
	}
};

axum::serve(listener, app)
	.with_graceful_shutdown(shutdown)
	.await?;
```

Active subprocess cleanup is handled by `kill_on_drop(true)` on each subprocess (kills child when the owning task is dropped). On graceful shutdown, Axum drops all in-flight request futures, which drops the subprocess tasks, which triggers `kill_on_drop`.

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `axum` 0.8+ | HTTP server framework |
| `tower-http` 0.6+ (with `cors` feature) | CORS middleware |
| `tokio` 1 (with `full` feature) | Async runtime |
| `tokio-stream` | `ReceiverStream` adapter for SSE |
| `serde` 1 (with `derive` feature) / `serde_json` 1 | JSON serialization/deserialization |
| `tracing` / `tracing-subscriber` (with `env-filter` feature) | Structured logging |
| `clap` 4 (with `derive`, `env` features) | CLI argument parsing |
| `uuid` 1 (with `v4` feature) | Request ID generation |
| `thiserror` 2 | Error type derivation |
| `dirs` 6 | Home directory and data directory resolution |
| `redb` 2.4+ | Persistent statistics database (embedded key-value store) |

---

## Implementation Notes

### Serde Type Hints

**ClaudeCliMessage enum** — Externally tagged on `"type"` with `#[serde(other)]` catch-all:
```rust
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ClaudeCliMessage {
	#[serde(rename = "system")]
	System { subtype: Option<String> },
	#[serde(rename = "assistant")]
	Assistant { message: Option<AssistantInner> },
	#[serde(rename = "stream_event")]
	StreamEvent { event: StreamEventInner },
	#[serde(rename = "result")]
	Result(ResultMessage),
	#[serde(other)]
	Unknown,
}
```

**StreamEventInner** — The inner event from a `stream_event` wrapper, also tagged on `"type"`:
```rust
#[derive(Deserialize)]
#[serde(tag = "type")]
enum StreamEventInner {
	#[serde(rename = "content_block_delta")]
	ContentBlockDelta { delta: Delta },
	#[serde(rename = "message_start")]
	MessageStart { message: Option<MessageStartInfo> },
	#[serde(rename = "message_delta")]
	MessageDelta {},  // Fields exist but are not used by the proxy
	#[serde(other)]
	Other,
}
```

**AssistantInner** — Nested inside the `assistant` variant's `message` field. Only `model` is extracted; the CLI also sends `content`, `id`, `role`, `stop_reason`, `stop_sequence`, `stop_details`, `usage`, and `context_management` — all silently ignored by serde. Note: the `error` field (e.g., `"invalid_request"`) appears on the top-level `assistant` object alongside `message`, NOT inside `message` — also ignored (serde skips unknown fields).
```rust
#[derive(Deserialize)]
struct AssistantInner {
	model: Option<String>,
}
```

**Delta** — The delta object inside `content_block_delta` events:
```rust
#[derive(Deserialize)]
struct Delta {
	#[serde(rename = "type")]
	delta_type: Option<String>,
	text: Option<String>,
}
```

**MessageStartInfo** — The `message` object inside `message_start` events (used for model extraction):
```rust
#[derive(Deserialize)]
struct MessageStartInfo {
	model: Option<String>,
}
```

**MessageDeltaInfo** — The `delta` object inside `message_delta` events. Not currently used by the proxy (the `MessageDelta` variant is an empty struct in the implementation), but documented for completeness:
```rust
#[derive(Deserialize)]
struct MessageDeltaInfo {
	stop_reason: Option<String>,
	// stop_sequence and stop_details are also present but always null in
	// observed testing — not needed for the proxy, so omitted.
}
```

**ResultMessage** — Top-level result fields (mostly snake_case, with `modelUsage` as the camelCase exception). Only fields used by the proxy are included; the CLI emits additional fields (`stop_reason`, `session_id`, `permission_denials`, `fast_mode_state`, `uuid`) that are silently ignored by serde:
```rust
#[derive(Deserialize)]
struct ResultMessage {
	subtype: Option<String>,
	is_error: Option<bool>,
	result: Option<String>,
	duration_ms: Option<u64>,
	duration_api_ms: Option<u64>,
	num_turns: Option<u64>,
	total_cost_usd: Option<f64>,
	usage: Option<FlatUsage>,
	#[serde(rename = "modelUsage")]
	model_usage: Option<HashMap<String, ModelUsage>>,
}
```

**FlatUsage** — Flat aggregate usage (snake_case):
```rust
#[derive(Deserialize)]
struct FlatUsage {
	input_tokens: Option<u64>,
	output_tokens: Option<u64>,
	cache_creation_input_tokens: Option<u64>,
	cache_read_input_tokens: Option<u64>,
}
```

**ModelUsage** — camelCase fields. Only fields used by the proxy are included; the CLI also emits `webSearchRequests` (always 0 with tools disabled) which is silently ignored:
```rust
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelUsage {
	input_tokens: Option<u64>,
	output_tokens: Option<u64>,
	cache_read_input_tokens: Option<u64>,
	cache_creation_input_tokens: Option<u64>,
	#[serde(rename = "costUSD")]
	cost_usd: Option<f64>,
	context_window: Option<u64>,
	max_output_tokens: Option<u64>,
}
```

Note: `costUSD` does not follow standard camelCase (it is `costUSD` not `costUsd`), so it needs `#[serde(rename = "costUSD")]` explicitly.

**ChatCompletionRequest** — The top-level request body:
```rust
#[derive(Deserialize)]
struct ChatCompletionRequest {
	model: String,
	messages: Vec<ChatMessage>,
	#[serde(default)]
	stream: bool,
	reasoning_effort: Option<String>,
	// All other fields (max_tokens, temperature, top_p, stop, etc.)
	// are accepted but ignored — no #[serde(deny_unknown_fields)].
}
```

**OpenAI ChatMessage** — A single message in the conversation:
```rust
#[derive(Deserialize)]
struct ChatMessage {
	role: String,
	#[serde(default)]
	content: Option<MessageContent>,
	// name, tool_calls, tool_call_id, etc. are accepted but ignored
}
```

**OpenAI MessageContent** — Untagged enum for string-or-array `content` field. The `content` field is `Option<MessageContent>` on the message — `null` is valid (OpenAI allows it on assistant messages):
```rust
#[derive(Deserialize)]
#[serde(untagged)]
enum MessageContent {
	Text(String),
	Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
	#[serde(rename = "type")]
	part_type: Option<String>,
	text: Option<String>,
	// image_url, etc. are silently ignored (non-text blocks)
}
```

When extracting text from a message: if `content` is `None` (null), treat it as empty string. If `content` is `Parts`, concatenate all parts where `part_type` is `"text"`. If `content` is `Text`, use it directly.

### OpenAI Response Types (Serialization)

**ChatCompletionResponse** — Non-streaming response:
```rust
#[derive(Serialize)]
struct ChatCompletionResponse {
	id: String,
	object: String,                        // "chat.completion"
	created: u64,                          // Unix timestamp
	model: String,
	system_fingerprint: Option<()>,        // Always serializes as null
	choices: Vec<Choice>,
	usage: Option<Usage>,
}

#[derive(Serialize)]
struct Choice {
	index: u32,
	message: ResponseMessage,
	finish_reason: String,
}

#[derive(Serialize)]
struct ResponseMessage {
	role: String,                          // "assistant"
	content: String,
}

#[derive(Serialize)]
struct Usage {
	prompt_tokens: u64,
	completion_tokens: u64,
	total_tokens: u64,
}
```

**ChatCompletionChunk** — Streaming SSE chunk:
```rust
#[derive(Serialize)]
struct ChatCompletionChunk {
	id: String,
	object: String,                        // "chat.completion.chunk"
	created: u64,
	model: String,
	system_fingerprint: Option<()>,        // Always null
	choices: Vec<ChunkChoice>,
	#[serde(skip_serializing_if = "Option::is_none")]
	usage: Option<Usage>,                  // Only on the final usage chunk
}

#[derive(Serialize)]
struct ChunkChoice {
	index: u32,
	delta: ChunkDelta,
	finish_reason: Option<String>,         // null during streaming, "stop" on final
	// Note: NO skip_serializing_if — None serializes as null (field present in JSON).
	// OpenAI clients expect "finish_reason": null on content chunks.
}

#[derive(Serialize)]
struct ChunkDelta {
	#[serde(skip_serializing_if = "Option::is_none")]
	role: Option<String>,                  // "assistant" on first chunk only
	#[serde(skip_serializing_if = "Option::is_none")]
	content: Option<String>,               // Text content (None on finish chunk)
}
```

### Non-Streaming Architecture

Non-streaming requests use only one channel (subprocess → handler), not the two-channel SSE pattern:

```rust
let (sub_tx, mut sub_rx) = mpsc::channel::<SubprocessEvent>(64);

// Spawn subprocess (acquires semaphore, spawns reader task internally).
spawn_managed(config.clone(), semaphore.clone(), queue_timeout, cli_args, sub_tx).await?;

// Collect all events in the handler (no converter task needed)
let mut content = String::new();
let mut model = requested_model.clone();
let mut result_msg = None;
let mut error_msg = None;

while let Some(event) = sub_rx.recv().await {
	match event {
		SubprocessEvent::Model(m) => { model = normalize_model(&m); }
		SubprocessEvent::ContentDelta(text) => { content.push_str(&text); }
		SubprocessEvent::Result(r) => { result_msg = Some(r); }
		SubprocessEvent::Error(e) => { error_msg = Some(e); }
	}
}

// After loop: check error_msg FIRST (takes priority over result_msg)
// Then check result_msg.is_error. Then build ChatCompletionResponse.
```

### Error Behavior on Empty Prompt

If the prompt argument is an empty string, the CLI exits with code 1 and emits an error to stderr: "Input must be provided either through stdin or as a prompt argument when using --print". No `result` line is emitted on stdout. The proxy prevents this via the `has_user_message` flag in `build_prompt_and_system()` — if no user message has non-empty text content, the request is rejected with 400 before prompt construction completes. This means an empty prompt string cannot reach the subprocess spawn. Note: there is no explicit `prompt.is_empty()` check after construction; the prevention is structural via the input validation. If future changes to prompt construction could produce an empty prompt, an explicit check should be added.

### Error Behavior on Invalid Model

If an invalid model name is passed via `--model`, the CLI exits with code 1. The NDJSON output includes: a `system/init` message (with the invalid model name echoed in the `model` field), a synthetic `assistant` message with `"model": "<synthetic>"` and `"error": "invalid_request"` as a top-level field on the assistant object, and a `result` message with `"subtype": "success"` AND `"is_error": true` simultaneously. No `stream_event` lines appear. The `result.result` contains a human-readable error message. The proxy should always check `is_error` on the result.

---

## Future Considerations (Post-MVP)

These are explicitly deferred but noted for future design awareness:

- **Anthropic Messages API** (`/v1/messages`) — The reference implementation provides a working Axum implementation with full Anthropic streaming format. Low effort to add later.
- **Multimodal support** — Would require encoding images and passing via `--input-format stream-json`.
- **Bidirectional streaming** — Using `--input-format stream-json` for multi-turn within a single process, reducing spawn overhead.
- **Process pooling** — Keep long-lived `claude` processes and send multiple requests via stdin.
- **Authentication** — API key validation for external exposure.
- **Rate limiting** — Server-side request throttling.
- **Docker packaging** — Containerized deployment with a fresh Claude Code install (no plugins by default). User runs `claude login` once in the container. The cleanest isolation approach — no `CLAUDE_CONFIG_DIR` setup needed.
- **TLS** — HTTPS support (or document reverse proxy setup).
- **Session management** — The reference implementation provides file-based session persistence (`~/.claude-code-cli-sessions.json`) with 24-hour TTL and hourly cleanup. This enables multi-turn conversations via `--session-id`.
- **Fallback model** — The CLI supports `--fallback-model <model>` for auto-fallback when the primary model is overloaded. Could be exposed as a proxy config option.
- **Turn limits** — `--max-turns <N>` (available in CLI v2.1.91+) limits agentic turns. Not needed with `--tools ""` (always single-turn) but could be added as defense-in-depth.
- **`--bare` mode migration** — The official documentation states `--bare` "will become the default for `-p` in a future release." When that happens, `--bare` would disable OAuth/keychain auth by default in print mode. If this change lands, the proxy would need an alternative approach (e.g., `--settings` with an `apiKeyHelper`, or the CLI may introduce a new flag for OAuth in bare mode). Monitor CLI release notes for this change.
- **OAuth token refresh** — The meridian reference implementation transparently refreshes expired OAuth tokens mid-request (reads refresh token from Keychain/file, POSTs to Anthropic's token endpoint, retries the request). Currently our proxy relies on the CLI handling its own auth; a transparent retry on "OAuth token has expired" errors would improve reliability for long-running deployments.
- **Extended context (1M) fallback** — Meridian implements a cooldown mechanism: if a `[1m]` model request fails due to Extra Usage not being enabled, it records the failure and falls back to base context (200K) for one hour before probing again. Our proxy always passes short model names (no `[1m]` suffix), so this is handled server-side by Anthropic, but explicit fallback could reduce failed-request latency.
- **Agent adapter pattern** — Meridian uses a per-agent adapter interface to handle tool blocking, session management, and system prompts differently for each client (OpenCode, Droid, Crush, Cline). If we add support for multiple client types in the future, this pattern would prevent per-client logic from sprawling across the codebase.

---

## Reference Implementation Notes

These notes capture patterns from the four reference implementations (in `research/`) that are valuable during implementation. They are implementation guidance, not spec requirements.

### claude-max-api-proxy-rs (Rust/Axum)

The closest reference to our architecture. Key patterns to study:

- **`src/subprocess.rs`** — Subprocess lifecycle with `tokio::select!` loop, inactivity timeout (30 min in their case), progress logging every 30s, TTFT tracking. Uses `BufReader::new(stdout).lines()` for line-by-line NDJSON reading. Client disconnect detected when `tx.send()` returns `Err`. Both stdout and stderr reset the inactivity timer — important for preventing false timeouts during slow model responses.
- **`src/routes.rs`** — Dual `mpsc::channel::<SubprocessEvent>(64)` pattern: subprocess → event channel → SSE writer. Non-streaming collects all events then composes response. Streaming uses `ReceiverStream::new(rx)` with `Sse::new().keep_alive(KeepAlive::default())`. The initial `:ok` SSE comment is emitted via `Event::default().comment("ok")`.
- **`src/error.rs`** — Unified `AppError` enum with `IntoResponse` impl that maps to OpenAI error JSON with `error.type`, `error.message`, and `error.code` fields. Uses `thiserror::Error` derive. Good pattern for our `error.rs`.
- **`src/adapter/openai_to_cli.rs`** — Model mapping with substring fallback (contains "opus" → "opus", etc.). Default fallback is "opus" (we use "sonnet"). Conversation prompt construction with `<previous_response>` tags for assistant messages, `<system>` XML tags for system messages (our spec uses `--append-system-prompt` instead — preferred).

**Patterns to adopt:**
- Empty content filtering: `if !text.is_empty()` before emitting `ContentDelta` — prevents empty SSE chunks.
- First-delta role emission: only the first streaming chunk includes `delta.role: "assistant"`, subsequent chunks omit it.
- Token aggregation across models: sums `input_tokens` and `output_tokens` from all entries in the `modelUsage` HashMap.
- `#[serde(skip_serializing_if = "Option::is_none")]` on `usage` (absent when `None`) and `delta.role`/`delta.content` (absent when `None`). **Not** on `finish_reason` — OpenAI clients expect `"finish_reason": null` (present in JSON), not absent.

**Pitfalls in this reference (already documented in spec):**
- Two-layer NDJSON parsing (unnecessary — all events are wrapped in `stream_event` envelope).
- `modelUsage` deserialized as snake_case (incorrect — it's camelCase).
- Non-streaming ignores `ContentDelta` events and relies solely on `result.result` (misses multi-turn text).
- No `kill_on_drop(true)` on subprocess — risk of orphans.
- Uses `--permission-mode bypassPermissions` — unnecessary and risky (we use `--tools ""` which makes permissions irrelevant).
- Uses `Stdio::null()` for stdin (correct).
- Sets `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` env var — unnecessary for this proxy (it enables multi-agent teams, which we don't use).

### meridian (TypeScript/Hono)

Most sophisticated reference — multi-agent, session lineage, telemetry. Relevant patterns:

- **`src/proxy/server.ts`** — Concurrency control via simple queue-based semaphore with `acquireSession()` / `releaseSession()`. FIFO queue of Promise resolvers. No timeout on the queue (potential issue — our spec has a 60s queue timeout, which is better). Active count tracking. `releaseSession()` always called in try-finally.
- **`src/telemetry/`** — Ring buffer for request metrics (TTFT, duration, queue wait, model, error status). HTML dashboard with client-side polling (not WebSocket). JSON API at `/telemetry/requests`, `/telemetry/summary`, `/telemetry/logs`. Close to our stats design but uses in-memory only (no persistence across restarts — our redb approach is better for lifetime totals).
- **`src/proxy/errors.ts`** — Error classification from raw error messages to structured HTTP responses. Key patterns: "oauth token has expired" → 401, "429" / "rate limit" → 429, "exited with code" → parse stderr for auth signals. Their inline token refresh + retry on 401 is elegant but out of scope for our MVP.
- **`src/proxy/models.ts`** — Model mapping with `[1m]` suffix for 1M context. Subscription-type-aware model selection. Cooldown mechanism for Extra Usage errors (1-hour backoff). Auth status caching with 60s TTL.
- **`src/proxy/openai.ts`** — Their OpenAI compatibility layer packs prior conversation turns into a `<conversation_history>` block in the system prompt, using only the last user message as the actual prompt. Each OpenAI request is a fresh session (no resume). This is because OpenAI clients replay full history themselves. Our approach is simpler — we pack all messages into the positional prompt argument.

### cliproxyapi (Go/Gin)

Enterprise-grade multi-provider proxy. Relevant patterns:

- **Hot-reload via file watching** — Not needed for MVP but shows how config can evolve.
- **Request/response translation registry** — Clean bidirectional format translation dispatch. Our `translate/` module follows a similar pattern.
- **Quota management with credential rotation** — Useful context for understanding rate limit behavior at scale.

### horselock-proxy (Node.js)

Lightweight direct-API proxy (not CLI-based, uses direct HTTP to Anthropic API). Relevant patterns:

- **Promise deduplication for token refresh** — Prevents thundering herd on concurrent auth failures. Pattern: if `refreshPromise` exists, await it instead of starting a new refresh.
- **Automatic 401 retry** — Clear cache, refresh token, retry once. Transparent to client.
- **System prompt requirement** — Their code reveals that the first system prompt block must begin with "You are Claude Code, Anthropic's official CLI for Claude." for the API to accept the request. This constraint comes from using the direct API with OAuth tokens. Our approach (using the CLI subprocess) avoids this entirely because the CLI constructs its own system prompt.
- **Debug streaming with Transform** — Non-invasive SSE monitoring that pretty-prints events. Useful for development/debugging.

