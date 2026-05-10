# Claude Code Provider

OpenAI-compatible API proxy for Claude Max accounts. Written in Rust (2024 edition).

CCP exposes the OpenAI Chat Completions API and talks directly to Anthropic's Messages API using the OAuth token from `~/.claude/.credentials.json`. It mimics Claude Code's wire fingerprint enough for Anthropic's subscription OAuth gate, but it does not invoke the Claude Code CLI per request.

## Build

```sh
cargo build --release
```

## Test

### Unit tests (Rust)

```sh
cargo test
```

### Integration tests (Python/pytest)

```sh
./tests/run.sh
```

This automatically builds the binary, starts CCP server instances on random ports, runs all tests, and tears down. Uses `uv run` with inline deps (httpx, openai, pytest, pytest-asyncio).

Useful flags:

```sh
./tests/run.sh -k "TestAuth"           # Run one test class
./tests/run.sh -k "test_health"        # Run one test
./tests/run.sh -x                      # Stop on first failure
./tests/run.sh --co                    # List tests without running
```

## Run

```sh
cargo run -- --no-auth --port 18321
```

## Architecture

- `src/main.rs` - Server setup, router, startup credential validation.
- `src/config.rs` - CLI args and env vars (all prefixed `CCP_`).
- `src/routes/completions.rs` - Main handler entrypoint for streaming and non-streaming.
- `src/routes/completions_v2.rs` - Direct Anthropic Messages request handling.
- `src/upstream/` - Credentials loading, Claude Code fingerprint headers, HTTP client, SSE parser.
- `src/translate/` - OpenAI request/response translation to and from Anthropic Messages API shapes.
- `src/models.rs` - Model definitions, alias resolution.
- `src/auth.rs` - API key middleware.
- `src/stats.rs` - Persistent stats (redb), per-model and per-key tracking.
- `src/replacements.rs` - Text replacement rules (TOML).

## Key design decisions

- CCP reads Claude OAuth credentials from `~/.claude/.credentials.json` fresh for each request and retries once after a 401 with a fresh read.
- Requests include Claude Code-compatible headers, beta flags, and session IDs so subscription OAuth calls are accepted by Anthropic.
- The canonical Claude Code system identifier is prepended by default to satisfy the OAuth gate. `--no-preamble` is for upstream debugging only.
- Tools are passed natively to Anthropic Messages API. PascalCase masking for tool names is often required via text replacement to satisfy OAuth gate fingerprinting.
- Text replacement applies outbound to prompts and tool surfaces, then inbound to assistant text and tool call names/arguments.
