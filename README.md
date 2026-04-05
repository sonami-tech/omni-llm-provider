# Claude Code Provider

[![Version](https://img.shields.io/github/v/tag/sonami-tech/claude-code-provider?label=version)](https://github.com/sonami-tech/claude-code-provider/releases)
[![Build](https://img.shields.io/github/actions/workflow/status/sonami-tech/claude-code-provider/docker.yml?branch=master)](https://github.com/sonami-tech/claude-code-provider/actions)
[![Last Commit](https://img.shields.io/github/last-commit/sonami-tech/claude-code-provider)](https://github.com/sonami-tech/claude-code-provider/commits)
[![GHCR](https://img.shields.io/badge/ghcr.io-sonami--tech%2Fclaude--code--provider-blue?logo=docker)](https://ghcr.io/sonami-tech/claude-code-provider)
[![Rust](https://img.shields.io/badge/rust-2024_edition-orange?logo=rust)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI-compatible API server that routes requests through the Claude Code CLI. Drop it in front of any OpenAI SDK client, no code changes required.

## How It Works

```
┌──────────────────────────┐
│  Anthropic (Claude Max)  │  Provides Claude access via your subscription.
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│     Claude Code CLI      │  Authenticates and communicates with Anthropic.
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│   Claude Code Provider   │  Translates OpenAI API requests into CLI calls.
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐
│     Your Application     │  Any OpenAI SDK client, no code changes required.
└──────────────────────────┘
```

## Quick Start

### 1. Install the [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code)

```sh
curl -fsSL https://claude.ai/install.sh | bash
claude login
```

### 2. Create an API key file

```sh
echo "sk-my-secret-key-here" > ~/ccp-keys.txt
```

### 3. Create a text replacement rules file

Text replacement prevents sensitive data from reaching the model by masking terms in prompts and restoring them in responses:

```sh
cat > ~/ccp-rules.toml << 'EOF'
# Replace sensitive terms in prompts before they reach Claude.
[[rule]]
scope = "prompt"
search = "Acme"
replace = "SomeStringNobodyElseWouldChoose"

[[rule]]
scope = "prompt"
search = "acme"
replace = "somestringnobodyelsewhouldchoose"

# Restore original terms in responses back to the client.
[[rule]]
scope = "response"
search = "SomeStringNobodyElseWouldChoose"
replace = "Acme"

[[rule]]
scope = "response"
search = "somestringnobodyelsewhouldchoose"
replace = "acme"
EOF
```

See [configuration reference](docs/configuration.md#text-replacement) for rule scopes and streaming behavior.

### 4. Start the server

**Docker:**

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v ~/ccp-rules.toml:/root/ccp-rules.toml:ro \
  -v ~/ccp-keys.txt:/root/ccp-keys.txt:ro \
  -e CCP_API_KEYS_FILE=/root/ccp-keys.txt \
  -e CCP_REPLACE_RULES=/root/ccp-rules.toml \
  ghcr.io/sonami-tech/claude-code-provider:latest
```

See [Docker documentation](docs/docker.md) for image tags, alternative auth methods, and production setup.

**Build from source** (requires [Rust](https://rustup.rs) 1.85+):

```sh
git clone https://github.com/sonami-tech/claude-code-provider.git
cd claude-code-provider
cargo build --release
./target/release/claude-code-provider \
  --api-keys-file ~/ccp-keys.txt \
  --replace-rules ~/ccp-rules.toml
```

### 5. Test it

Verify the server is running:

```sh
curl http://localhost:18321/health
```

Send a request:

```sh
curl http://localhost:18321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-secret-key-here" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Point any OpenAI SDK client at `http://your-host:18321/v1` with the API key from your keys file.

## Models

| Model | Aliases |
|-------|---------|
| `claude-opus-4-6` | `opus`, `claude-opus` |
| `claude-sonnet-4-6` | `sonnet`, `claude-sonnet` |
| `claude-haiku-4-5` | `haiku`, `claude-haiku` |

Unrecognized model names fall back to Sonnet. See [configuration reference](docs/configuration.md#models) for date-suffixed names and [reasoning effort](docs/configuration.md#reasoning-effort) levels.

## Limitations

- **Text only** — image and audio content parts are silently ignored.
- **Ignored parameters** — `max_tokens`, `temperature`, `top_p`, and `stop` are accepted for compatibility but not forwarded to the CLI.
- **Subprocess per request** — each request spawns a new `claude` process, adding startup latency.

## Documentation

- [Configuration reference](docs/configuration.md) — all flags, env vars, models, endpoints, and defaults.
- [Docker guide](docs/docker.md) — image tags, auth options, production setup.
- [Architecture](docs/architecture.md) — request flow, subprocess lifecycle, design decisions.

## License

[MIT](LICENSE)
