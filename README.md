# Claude Code Provider

[![Version](https://img.shields.io/github/v/tag/sonami-tech/claude-code-provider?label=version)](https://github.com/sonami-tech/claude-code-provider/releases)
[![Build](https://img.shields.io/github/actions/workflow/status/sonami-tech/claude-code-provider/docker.yml?branch=master)](https://github.com/sonami-tech/claude-code-provider/actions)
[![Last Commit](https://img.shields.io/github/last-commit/sonami-tech/claude-code-provider)](https://github.com/sonami-tech/claude-code-provider/commits)
[![GHCR](https://img.shields.io/badge/ghcr.io-sonami--tech%2Fclaude--code--provider-blue?logo=docker)](https://ghcr.io/sonami-tech/claude-code-provider)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI-compatible API server that translates Chat Completions requests into Claude Code CLI subprocess calls. Drop it in front of any OpenAI SDK client, no code changes required.

## Why

An Anthropic Max subscription gives you Claude access through the Claude Code CLI. This proxy lets tools like aider, Open WebUI, OpenClaw, LiteLLM, and any OpenAI SDK client use that same access. Every request goes through the official `claude` binary, the same interface Anthropic provides and supports.

## Features

- **Model aliases** - `sonnet`, `opus`, `haiku` resolve automatically; unrecognized names fall back to Sonnet.
- **Streaming and non-streaming** - SSE and JSON responses compatible with the official Python/TypeScript SDKs.
- **Concurrency control** - bounded subprocess pool with configurable queue timeout.
- **Reasoning effort** - `low`, `medium`, `high`, `max` via the `reasoning_effort` parameter.
- **Secure by default** - auto-generates an API key on startup; explicit keys and no-auth mode also supported.
- **Persistent stats** - `/stats` dashboard and `/stats/json` endpoint with per-model and per-key metrics.
- **Isolated configuration** - subprocesses use a separate config directory, so the proxy never touches your existing Claude Code settings.
- **Single binary** - no runtime dependencies beyond the Claude CLI.

## Quick Start

### Prerequisites

1. **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** installed and authenticated:
   ```sh
   curl -fsSL https://claude.ai/install.sh | bash
   claude login
   ```
2. **[Rust toolchain](https://rustup.rs)** (1.85+ for edition 2024).

### Build and Run

```sh
git clone <repo-url> claude-code-provider
cd claude-code-provider
cargo build --release
./target/release/claude-code-provider          # port 18321, 5 concurrent
./target/release/claude-code-provider -p 8080 -c 10  # custom
```

An API key is auto-generated on startup and printed to the log. Use it with any OpenAI SDK client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18321/v1", api_key="<key from startup log>")

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Docker

Prebuilt images are published to GHCR on every push and tag.

```sh
# Stable (recommended). Latest tagged release.
docker pull ghcr.io/sonami-tech/claude-code-provider:latest

# Specific version.
docker pull ghcr.io/sonami-tech/claude-code-provider:v0.1.0

# Development. Built from every push to main.
docker pull ghcr.io/sonami-tech/claude-code-provider:dev
```

### Authentication

**Option A - Log in inside the container:**

```sh
docker run -it -p 18321:18321 ghcr.io/sonami-tech/claude-code-provider bash
claude login            # authenticate once
claude-code-provider    # start the proxy
```

**Option B - Mount credentials from host:**

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  ghcr.io/sonami-tech/claude-code-provider
```

The Docker image sets `CCP_NO_ISOLATE=true` by default since the container has no plugins or hooks to isolate from. The auto-generated API key is printed to the container log on startup.

Note: API keys passed via `-e CCP_API_KEYS=...` are visible in `docker inspect`. For production, mount a keys file instead:

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v /path/to/keys.txt:/etc/ccp-keys:ro \
  -e CCP_API_KEYS_FILE=/etc/ccp-keys \
  ghcr.io/sonami-tech/claude-code-provider
```

### Build from Source

```sh
docker build -t claude-code-provider .
```

## Configuration

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p, --port` | `CCP_PORT` | `18321` | Listen port |
| `-H, --host` | `CCP_HOST` | `127.0.0.1` | Listen address |
| `-c, --max-concurrent` | `CCP_MAX_CONCURRENT` | `5` | Max simultaneous subprocesses |
| `-t, --timeout` | `CCP_TIMEOUT` | `600` | Inactivity timeout (seconds) |
| `-q, --queue-timeout` | `CCP_QUEUE_TIMEOUT` | `60` | Queue wait timeout (seconds) |
| `--claude-path` | `CCP_CLAUDE_PATH` | `claude` | Path to Claude CLI binary |
| `--data-dir` | `CCP_DATA_DIR` | Platform default | Data directory for config and stats |
| `--working-dir` | `CCP_WORKING_DIR` | Config dir | Subprocess working directory |
| `--no-isolate` | `CCP_NO_ISOLATE` | Off | Use host Claude config directly |
| `--api-keys` | `CCP_API_KEYS` | Auto-generated | Comma-separated API keys (min 8 chars each) |
| `--api-keys-file` | `CCP_API_KEYS_FILE` | None | File with one API key per line (# comments allowed) |
| `--no-auth` | `CCP_NO_AUTH` | Off | Disable authentication entirely |
| `--replace-rules` | `CCP_REPLACE_RULES` | None | TOML file with text replacement rules (see below) |
| `--log-conversations` | `CCP_LOG_CONVERSATIONS` | Off | Log full prompts and responses |
| `--log-file` | `CCP_LOG_FILE` | None | Write conversation logs to file (implies --log-conversations) |
| `-v, --verbose` | | Off | Debug logging |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `GET /v1/models` | List available models |
| `GET /health` | Server health and active request count |
| `GET /stats` | HTML stats dashboard |
| `GET /stats/json` | Stats as JSON |

All `/v1/*` endpoints also work without the prefix (`/chat/completions`, `/models`), so both `http://host:18321` and `http://host:18321/v1` work as a base URL.

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

## Text Replacement

Automatic find-and-replace on prompts and/or responses. Create a TOML rules file:

```toml
# Inject context into prompts.
[[rule]]
scope = "prompt"
search = "COMPANY_NAME"
replace = "Acme Corp"

# Remove boilerplate from responses.
[[rule]]
scope = "response"
search = "As an AI language model, "
replace = ""

# Apply to both directions.
[[rule]]
scope = "both"
search = "http://old.internal:8080"
replace = "https://api.example.com"
```

```sh
claude-code-provider --replace-rules rules.toml
```

Rules are applied in file order. Literal string matching only. For streaming responses, replacement is per-chunk (cross-chunk matches are rare but possible).

## Limitations

- **Latency** - each request spawns a `claude -p` subprocess (~100-500ms overhead).
- **Text only** - image and audio content parts are silently ignored.
- **No tool use** - subprocesses run with `--tools ""`.
- **Ignored parameters** - `max_tokens`, `temperature`, `top_p`, `stop` are accepted but not passed through.

## License

[MIT](LICENSE)
