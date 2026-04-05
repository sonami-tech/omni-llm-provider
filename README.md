# Claude Code Provider

[![Version](https://img.shields.io/github/v/tag/sonami-tech/claude-code-provider?label=version)](https://github.com/sonami-tech/claude-code-provider/releases)
[![Build](https://img.shields.io/github/actions/workflow/status/sonami-tech/claude-code-provider/docker.yml?branch=master)](https://github.com/sonami-tech/claude-code-provider/actions)
[![Last Commit](https://img.shields.io/github/last-commit/sonami-tech/claude-code-provider)](https://github.com/sonami-tech/claude-code-provider/commits)
[![GHCR](https://img.shields.io/badge/ghcr.io-sonami--tech%2Fclaude--code--provider-blue?logo=docker)](https://ghcr.io/sonami-tech/claude-code-provider)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI-compatible API server that translates Chat Completions requests into Claude Code CLI subprocess calls. Drop it in front of any OpenAI SDK client, no code changes required.

## Why

An Anthropic Max subscription gives you Claude access through the Claude Code CLI. This proxy lets any OpenAI SDK client use that same access. Every request goes through the official `claude` binary, the same interface Anthropic provides and supports.

## Features

- **Drop-in OpenAI compatibility** - streaming and non-streaming responses work with any OpenAI SDK client, no code changes required.
- **Tool/function calling** - OpenAI-compatible `tools` and `tool_calls` passthrough, enabled by default.
- **Text replacement** - TOML-based find-and-replace rules applied to prompts, responses, or both. Rewrite content on the fly without changing client code.
- **Concurrency control** - bounded subprocess pool with configurable queue timeout. Supports multiple concurrent clients safely.
- **Isolated and secure** - auto-generated API keys, separate config directory per subprocess, and no interference with your existing Claude Code settings.

## Quick Start: Docker

### 1. Create an API key file

```sh
echo "sk-my-secret-key-here" > ~/ccp-keys.txt
```

### 2. Create a text replacement rules file

Text replacement lets you rewrite prompts before they reach Claude and responses before they return to the client. A common use case is masking a product name so Claude does not develop biases about it:

```sh
cat > ~/ccp-rules.toml << 'EOF'
# Mask brand name in prompts sent to Claude.
[[rule]]
scope = "prompt"
search = "Acme"
replace = "SomeStringNobodyElseWouldChoose"

[[rule]]
scope = "prompt"
search = "acme"
replace = "somestringnobodyelsewould"

# Restore brand name in responses back to the client.
[[rule]]
scope = "response"
search = "SomeStringNobodyElseWouldChoose"
replace = "Acme"

[[rule]]
scope = "response"
search = "somestringnobodyelsewould"
replace = "acme"
EOF
```

### 3. Run the container

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v ~/ccp-rules.toml:/root/ccp-rules.toml:ro \
  -v ~/ccp-keys.txt:/root/ccp-keys.txt:ro \
  -e CCP_API_KEYS_FILE=/root/ccp-keys.txt \
  -e CCP_REPLACE_RULES=/root/ccp-rules.toml \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

The server starts on port 18321. Point any OpenAI SDK client at `http://your-host:18321/v1` with the API key from your keys file.

See [Docker documentation](docs/docker.md) for image tags, alternative auth methods, and production setup.

## Quick Start: Build from Source

### Prerequisites

1. **[Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)** installed and authenticated:
   ```sh
   curl -fsSL https://claude.ai/install.sh | bash
   claude login
   ```
2. **[Rust toolchain](https://rustup.rs)** (1.85+ for edition 2024).

### 1. Create an API key file and rules file

Follow steps 1 and 2 from the Docker section above.

### 2. Build and run

```sh
git clone <repo-url> claude-code-provider
cd claude-code-provider
cargo build --release
./target/release/claude-code-provider \
  --api-keys-file ~/ccp-keys.txt \
  --replace-rules ~/ccp-rules.toml
```

### 3. Test it

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18321/v1", api_key="sk-my-secret-key-here")

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Text Replacement

Rules are defined in a TOML file with three scopes:

- `prompt` - applied before sending to Claude.
- `response` - applied before returning to the client.
- `both` - applied in both directions.

Rules are applied in file order using literal string matching. See [configuration reference](docs/configuration.md#text-replacement) for details on streaming behavior and rule loading.

## Limitations

- **Text only** - image and audio content parts are silently ignored.

## Documentation

- [Configuration reference](docs/configuration.md) - all flags, env vars, models, endpoints, and defaults.
- [Docker guide](docs/docker.md) - image tags, auth options, production setup.
- [Architecture](docs/architecture.md) - request flow, subprocess lifecycle, design decisions.

## License

[MIT](LICENSE)
