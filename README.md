# Claude Code Provider

An OpenAI-compatible API server that translates Chat Completions requests into [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) subprocess calls. Drop it in front of any OpenAI SDK client — no code changes required.

## Why

An Anthropic Max subscription gives you Claude access through the Claude Code CLI. This proxy lets tools like [aider](https://aider.chat), [Open WebUI](https://openwebui.com), [OpenClaw](https://openclaw.ai), [LiteLLM](https://litellm.ai), and any OpenAI SDK client use that same access — every request goes through the official `claude` binary, the same interface Anthropic provides and supports.

## Features

- **Model aliases** — `sonnet`, `opus`, `haiku` resolve automatically; unrecognized names fall back to Sonnet.
- **Streaming and non-streaming** — SSE and JSON responses compatible with the official Python/TypeScript SDKs.
- **Concurrency control** — bounded subprocess pool with configurable queue timeout.
- **Reasoning effort** — `low`, `medium`, `high`, `max` via the `reasoning_effort` parameter.
- **Persistent stats** — `/stats` dashboard and `/stats/json` endpoint with per-model metrics, latency, and error history.
- **Isolated configuration** — subprocesses use a separate config directory, so the proxy never touches your existing Claude Code settings.
- **Single binary** — no runtime dependencies beyond the Claude CLI.

## Quick Start

### Prerequisites

1. **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** installed and authenticated:
   ```sh
   curl -fsSL https://claude.ai/install.sh | bash
   claude login
   ```
2. **Rust toolchain** (1.85+ for edition 2024).

### Build and Run

```sh
git clone <repo-url> claude-code-provider
cd claude-code-provider
cargo build --release
./target/release/claude-code-provider          # port 3456, 5 concurrent
./target/release/claude-code-provider -p 8080 -c 10  # custom
```

### Try It

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:3456/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Docker

Prebuilt images are published to GHCR on every push and tag.

```sh
# Stable (recommended) — latest tagged release.
docker pull ghcr.io/sonami-tech/claude-code-provider:latest

# Specific version.
docker pull ghcr.io/sonami-tech/claude-code-provider:v0.1.0

# Development — built from every push to main.
docker pull ghcr.io/sonami-tech/claude-code-provider:dev
```

### Authentication

**Option A — Log in inside the container:**

```sh
docker run -it -p 3456:3456 ghcr.io/sonami-tech/claude-code-provider bash
claude login            # authenticate once
claude-code-provider    # start the proxy
```

**Option B — Mount credentials from host:**

```sh
docker run -p 3456:3456 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  ghcr.io/sonami-tech/claude-code-provider
```

The container is a clean environment with no plugins or hooks, so the proxy's config isolation is purely defense-in-depth. Your host Claude Code settings are never touched.

### Build from Source

```sh
docker build -t claude-code-provider .
```

## Configuration

| Flag | Env Var | Default | Description |
|------|---------|---------|-------------|
| `-p, --port` | `CCP_PORT` | `3456` | Listen port |
| `-H, --host` | `CCP_HOST` | `127.0.0.1` | Listen address |
| `-c, --max-concurrent` | `CCP_MAX_CONCURRENT` | `5` | Max simultaneous subprocesses |
| `-t, --timeout` | `CCP_TIMEOUT` | `600` | Inactivity timeout (seconds) |
| `-q, --queue-timeout` | `CCP_QUEUE_TIMEOUT` | `60` | Queue wait timeout (seconds) |
| `--claude-path` | `CCP_CLAUDE_PATH` | `claude` | Path to Claude CLI binary |
| `--data-dir` | `CCP_DATA_DIR` | Platform default | Data directory for config and stats |
| `--working-dir` | `CCP_WORKING_DIR` | Config dir | Subprocess working directory |
| `-v, --verbose` | — | Off | Debug logging |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `GET /v1/models` | List available models |
| `GET /health` | Server health and active request count |
| `GET /stats` | HTML stats dashboard |
| `GET /stats/json` | Stats as JSON |

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

## Limitations

- **Latency** — each request spawns a `claude -p` subprocess (~100-500ms overhead).
- **Text only** — image and audio content parts are silently ignored.
- **No tool use** — subprocesses run with `--tools ""`.
- **Ignored parameters** — `max_tokens`, `temperature`, `top_p`, `stop` are accepted but not passed through.

## License

[MIT](LICENSE)
