# Docker Guide

## Image Tags

Images are published to GHCR on every push to `master` and on version tags.

```sh
# Latest stable release.
docker pull ghcr.io/sonami-tech/claude-code-provider:latest

# Specific version.
docker pull ghcr.io/sonami-tech/claude-code-provider:v1.1.8

# Development. Built from every push to master.
docker pull ghcr.io/sonami-tech/claude-code-provider:dev
```

Multi-platform images are built for `linux/amd64` and `linux/arm64`.

## Setup

Create a directory to hold your configuration, then place your files in it:

```
~/claude-code-provider/
├── docker-compose.yml
├── keys.txt
├── rules.toml
└── logs/              (created automatically when logging is enabled)
```

Download `docker-compose.yml` from the repository:

```sh
mkdir -p ~/claude-code-provider && cd ~/claude-code-provider
curl -fsSL https://raw.githubusercontent.com/sonami-tech/claude-code-provider/master/docker-compose.yml -o docker-compose.yml
```

Edit it to adjust settings (enable logging, change concurrency, pin a version tag, etc.), then start:

```sh
docker compose up -d
```

## Authentication

The `docker-compose.yml` mounts the host's `~/.claude` directory into the container. This allows the CLI to read credentials and write back refreshed OAuth tokens when they expire. If you haven't logged in yet:

```sh
curl -fsSL https://claude.ai/install.sh | bash
claude login
```

After re-authenticating (`claude login`), the container picks up the new credentials automatically on the next request — no restart needed.

## Default Settings

The Docker image sets:

- `CCP_HOST=0.0.0.0` (listen on all interfaces, required for port mapping).
- `ENTRYPOINT ["claude-code-provider"]` (starts the proxy automatically).

Each request gets its own isolated config directory inside the container, preventing stale OAuth caches from blocking requests after token refresh.

The auto-generated API key is printed to the container log on startup (unless you provide your own via `keys.txt`).

## API Keys

The `docker-compose.yml` mounts `./keys.txt` into the container. Create it with one key per line (`#` comments allowed):

```
# Production keys
sk-prod-key-one
sk-prod-key-two
```

Avoid passing keys via `CCP_API_KEYS` in the environment — they are visible in `docker inspect`.

## Text Replacement and Logging

The `docker-compose.yml` mounts `./rules.toml` by default. To enable conversation logging, uncomment the log volume and environment variable:

```yaml
volumes:
  - ./logs:/var/log/ccp
environment:
  CCP_LOG_FILE: /var/log/ccp/conversations.log
```

## All Environment Variables

Uncomment or add any of these in the `environment:` section of your `docker-compose.yml`. See [configuration reference](configuration.md) for details.

| Variable | Default | Description |
|----------|---------|-------------|
| `CCP_API_KEYS_FILE` | None | Path to API keys file inside the container |
| `CCP_REPLACE_RULES` | None | Path to replacement rules TOML inside the container |
| `CCP_LOG_FILE` | None | Path to conversation log file inside the container |
| `CCP_MAX_CONCURRENT` | `5` | Max simultaneous subprocesses |
| `CCP_TIMEOUT` | `600` | Subprocess inactivity timeout (seconds) |
| `CCP_QUEUE_TIMEOUT` | `60` | Max queue wait time (seconds) |
| `CCP_MAX_TURNS` | `3` | Max agentic turns per request |
| `CCP_PORT` | `18321` | Listen port |
| `CCP_NO_AUTH` | Off | Set to `true` to disable authentication |
| `CCP_NO_TOOL_PASSTHROUGH` | Off | Set to `true` to disable tool calling |

## Verify

```sh
# Check the container is running.
docker compose logs

# Test the health endpoint.
curl http://localhost:18321/health

# Test a completion (use the API key from your keys.txt).
curl http://localhost:18321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model": "haiku", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Common Operations

```sh
# Start in the background.
docker compose up -d

# View logs.
docker compose logs -f

# Restart after changing docker-compose.yml.
docker compose up -d

# Stop.
docker compose down

# Pull a newer image.
docker compose pull && docker compose up -d
```

## Build from Source

```sh
docker build -t claude-code-provider .
```
