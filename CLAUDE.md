# Claude Code Provider

OpenAI-compatible API proxy backed by Claude Code CLI. Written in Rust (2024 edition).

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

This automatically builds the binary, starts isolated CCP server instances on random ports, runs all tests, and tears down. No manual setup needed. Uses `uv run` with inline deps (httpx, openai, pytest, pytest-asyncio).

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

- `src/main.rs` — Server setup, router, startup validation.
- `src/config.rs` — CLI args and env vars (all prefixed `CCP_`).
- `src/routes/completions.rs` — Main handler: streaming and non-streaming.
- `src/subprocess/process.rs` — Spawns Claude CLI, pipes prompt via stdin, reads NDJSON stdout.
- `src/subprocess/manager.rs` — Concurrency control via semaphore.
- `src/translate/` — Request/response translation between OpenAI and Claude formats.
- `src/models.rs` — Model definitions, alias resolution.
- `src/auth.rs` — API key middleware.
- `src/stats.rs` — Persistent stats (redb), per-model and per-key tracking.
- `src/replacements.rs` — Text replacement rules (TOML).

## Key design decisions

- Prompt is piped via stdin to avoid the Linux MAX_ARG_STRLEN (~128KB) kernel limit.
- Each request gets an isolated config directory to prevent credential cache conflicts.
- System prompt is passed as `--append-system-prompt` CLI arg (small enough for arg limit).
- Subprocess stderr is buffered (last 50 lines) and used to enrich error diagnostics.
