# Claude Code Provider

[![Version](https://img.shields.io/github/v/tag/sonami-tech/claude-code-provider?label=version)](https://github.com/sonami-tech/claude-code-provider/releases)
[![Build](https://img.shields.io/github/actions/workflow/status/sonami-tech/claude-code-provider/docker.yml?branch=master)](https://github.com/sonami-tech/claude-code-provider/actions)
[![Last Commit](https://img.shields.io/github/last-commit/sonami-tech/claude-code-provider)](https://github.com/sonami-tech/claude-code-provider/commits)
[![GHCR](https://img.shields.io/badge/ghcr.io-sonami--tech%2Fclaude--code--provider-blue?logo=docker)](https://ghcr.io/sonami-tech/claude-code-provider)
[![built with Claude Code](https://img.shields.io/badge/built_with-Claude_Code-D97757?logo=claude&logoColor=white)](https://claude.ai/code)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An OpenAI compatibility layer for **Claude Max** accounts. CCP exposes the OpenAI Chat Completions API, accepts your existing OpenAI SDK clients with no code changes, and serves them using your Claude Max subscription — mimicking the Claude Code CLI on the wire so Anthropic accepts the calls.

> **v1 → v2.** v1 shelled out to the `claude` CLI subprocess for every request. v2 talks directly to `api.anthropic.com/v1/messages`, reusing the OAuth token and request fingerprint Claude Code itself uses. Faster, no subprocess startup, full control over streaming and parameters.

## How It Works

```
┌──────────────────────────┐
│  Anthropic (Claude Max)  │  Provides Claude access via your subscription.
└────────────▲─────────────┘
             │ Native Messages API + Claude Code OAuth token
┌────────────┴─────────────┐
│   Claude Code Provider   │  Translates OAI ⇄ Anthropic and mimics the CLI's
│                          │  request signature so the gate doesn't fire.
└────────────▲─────────────┘
             │ OpenAI /v1/chat/completions
┌────────────┴─────────────┐
│     Your Application     │  Any OpenAI SDK client, no code changes required.
└──────────────────────────┘
```

The only artifact CCP still needs from Claude Code is the OAuth token in `~/.claude/.credentials.json`, written by `claude login`. The CLI itself doesn't need to be running.

## Quick Start

### 1. Authenticate with [Claude Code](https://docs.anthropic.com/en/docs/claude-code)

CCP reads the OAuth token from `~/.claude/.credentials.json`, written by `claude login`. The CLI is only needed for this one-time login.

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
| `claude-opus-4-8` | `opus` |
| `claude-sonnet-4-6` | `sonnet` |
| `claude-haiku-4-5-20251001` | `haiku` |

Model names are tied to the active Claude Code fingerprint profile. The default
profile currently mimics Claude Code `2.1.154`; short aliases resolve to
captured Claude Code model names. Explicit `claude-*` model names are preserved
when possible to keep the outbound body fingerprint aligned with Claude Code.
Unrecognized model names fall back to the profile's default model, currently
Sonnet. See [configuration reference](docs/configuration.md#models) for
date-suffixed names and [reasoning effort](docs/configuration.md#reasoning-effort)
levels.

## Limitations

- **Text only** — `image_url` content parts are not yet translated to Anthropic image blocks; audio is unsupported.
- **Subscription-bound** — uses your Claude Max OAuth token. Per-call billing keys are not supported.
- **Tool surface fingerprinting** — Anthropic's OAuth gate flags tool names that don't look like Claude Code's. Use the text-replacement layer to PascalCase them on the way out (e.g. `memory_search` → `MemorySearch`) and reverse on the way back.

## Documentation

- [Configuration reference](docs/configuration.md) — all flags, env vars, models, endpoints, and defaults.
- [Docker guide](docs/docker.md) — image tags, auth options, production setup.
- [Architecture](docs/architecture.md) — request flow, OAuth gate handling, design decisions.

## License

[MIT](LICENSE)
