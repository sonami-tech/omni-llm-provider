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
- **Reasoning effort** - `none`, `low`, `medium`, `high`, `max` via the `reasoning_effort` parameter.
- **Tool/function calling** - OpenAI-compatible `tools` and `tool_calls` passthrough, enabled by default.
- **Secure by default** - auto-generates an API key on startup; explicit keys and no-auth mode also supported.
- **Persistent stats** - `/stats` dashboard and `/stats/json` endpoint with per-model and per-key metrics.
- **Text replacement** - TOML-based find-and-replace rules for prompts, responses, or both.
- **Conversation logging** - full prompt and response logging to stderr or a dedicated file.
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
./target/release/claude-code-provider
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

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

The auto-generated API key is printed to the container log. See [Docker documentation](docs/docker.md) for image tags, alternative auth methods, and production setup.

## API Keys

An API key is auto-generated on startup by default. To use your own:

```sh
# Single key via environment variable.
claude-code-provider --api-keys my-secret-key

# Multiple keys from a file.
claude-code-provider --api-keys-file keys.txt

# Disable auth entirely.
claude-code-provider --no-auth
```

Keys must be at least 8 characters. In Docker, avoid `-e CCP_API_KEYS=...` (visible in `docker inspect`) and mount a keys file instead. See [configuration reference](docs/configuration.md) for details.

## Text Replacement

Automatic find-and-replace on prompts and/or responses. Create a TOML rules file:

```toml
[[rule]]
scope = "prompt"
search = "COMPANY_NAME"
replace = "Acme Corp"

[[rule]]
scope = "response"
search = "As an AI language model, "
replace = ""

[[rule]]
scope = "both"
search = "http://old.internal:8080"
replace = "https://api.example.com"
```

```sh
claude-code-provider --replace-rules rules.toml
```

Rules are applied in file order. Literal string matching only. Scopes: `prompt` (before sending to Claude), `response` (before returning to client), or `both`. For streaming responses, replacement is per-chunk.

## Tool Calling

Clients that send `tools` in their requests get OpenAI-compatible `tool_calls` back. The proxy translates tool definitions into prompt instructions, parses the model's response for structured JSON, and converts it to the standard `tool_calls` format. Multi-turn conversations with `tool` role messages are supported.

```python
response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }]
)
# response.choices[0].message.tool_calls contains the function call
```

Tool passthrough is enabled by default. Use `--no-tool-passthrough` to disable. Supports `tool_choice` values: `auto`, `none`, `required`, and specific function selection. See [configuration reference](docs/configuration.md) for details.

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

Date-suffixed model names (e.g., `claude-sonnet-4-6-20260101`) are also resolved via substring matching. Unrecognized names fall back to Sonnet.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `GET /v1/models` | List available models |
| `GET /health` | Server health and active request count |
| `GET /stats` | HTML stats dashboard |
| `GET /stats/json` | Stats as JSON |

All `/v1/*` endpoints also work without the prefix (`/chat/completions`, `/models`), so both `http://host:18321` and `http://host:18321/v1` work as a base URL.

## Documentation

- [Configuration reference](docs/configuration.md) - all flags, env vars, and defaults.
- [Docker guide](docs/docker.md) - image tags, auth options, production setup.
- [Architecture](docs/architecture.md) - request flow, subprocess lifecycle, design decisions.

## Limitations

- **Latency** - each request spawns a `claude` subprocess.
- **Text only** - image and audio content parts are silently ignored.
- **Tool calling is prompt-based** - tool calls are simulated via prompt injection, not native API tool_use. Works reliably but streaming is buffered when tools are present.
- **Ignored parameters** - `max_tokens`, `temperature`, `top_p`, `stop` are accepted but not passed through.

## License

[MIT](LICENSE)
