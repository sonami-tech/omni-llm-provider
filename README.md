# Claude Code Provider

[![Version](https://img.shields.io/github/v/tag/sonami-tech/claude-code-provider?label=version)](https://github.com/sonami-tech/claude-code-provider/releases)
[![Build](https://img.shields.io/github/actions/workflow/status/sonami-tech/claude-code-provider/docker.yml?branch=master)](https://github.com/sonami-tech/claude-code-provider/actions)
[![Last Commit](https://img.shields.io/github/last-commit/sonami-tech/claude-code-provider)](https://github.com/sonami-tech/claude-code-provider/commits)
[![GHCR](https://img.shields.io/badge/ghcr.io-sonami--tech%2Fclaude--code--provider-blue?logo=docker)](https://ghcr.io/sonami-tech/claude-code-provider)
[![Built with Claude Code](https://img.shields.io/badge/Built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.ai/code)
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

### 2. Create your project directory

```sh
mkdir ~/claude-code-provider && cd ~/claude-code-provider
```

### 3. Create an API key file

```sh
echo "sk-my-secret-key-here" > keys.txt
```

### 4. Create a text replacement rules file

Text replacement prevents sensitive data from reaching the model by masking terms in prompts and restoring them in responses:

```sh
cat > rules.toml << 'EOF'
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

### 5. Create `docker-compose.yml`

```sh
curl -fsSL https://raw.githubusercontent.com/sonami-tech/claude-code-provider/master/docker-compose.yml -o docker-compose.yml
```

Or copy the `docker-compose.yml` from the [repository root](docker-compose.yml). See the [Docker guide](docs/docker.md) for image tags, auth options, and all available settings.

### 6. Start the server

```sh
docker compose up -d
```

### 7. Test it

```sh
curl http://localhost:18321/health
```

```sh
curl http://localhost:18321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-my-secret-key-here" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Point any OpenAI SDK client at `http://your-host:18321/v1` with the API key from your keys file.

### Building from source

If you prefer to run without Docker (requires [Rust](https://rustup.rs) 1.85+):

```sh
git clone https://github.com/sonami-tech/claude-code-provider.git
cd claude-code-provider
cargo build --release
./target/release/claude-code-provider \
  --api-keys-file keys.txt \
  --replace-rules rules.toml
```

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
