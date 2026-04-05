# Docker Guide

## Image Tags

Images are published to GHCR on every push to `master` and on version tags.

```sh
# Development. Built from every push to master.
docker pull ghcr.io/sonami-tech/claude-code-provider:dev

# Specific version.
docker pull ghcr.io/sonami-tech/claude-code-provider:v0.1.0
```

Multi-platform images are built for `linux/amd64` and `linux/arm64`.

## Authentication

**Option A - Mount credentials from host (recommended):**

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

**Option B - Log in inside the container:**

```sh
docker run -it --entrypoint /bin/bash -p 18321:18321 ghcr.io/sonami-tech/claude-code-provider:dev
claude login            # authenticate once
claude-code-provider    # start the proxy
```

Credentials created inside the container are lost when it stops. To persist them, add `-v claude-creds:/root/.claude` to the `docker run` command.

## Default Settings

The Docker image sets:

- `CCP_HOST=0.0.0.0` (listen on all interfaces, required for port mapping).
- `CCP_NO_ISOLATE=true` (no host config to isolate from in a container).
- `ENTRYPOINT ["claude-code-provider"]` (starts the proxy automatically).

The auto-generated API key is printed to the container log on startup.

## API Keys

API keys passed via `-e CCP_API_KEYS=...` are visible in `docker inspect`. For production, mount a keys file instead:

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v /path/to/keys.txt:/etc/ccp-keys:ro \
  -e CCP_API_KEYS_FILE=/etc/ccp-keys \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

Keys file format (one key per line, `#` comments allowed):

```
# Production keys
sk-prod-key-one
sk-prod-key-two
```

## Text Replacement and Logging

Mount a TOML rules file and a log directory:

```sh
docker run -p 18321:18321 \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v /path/to/rules.toml:/etc/ccp-rules.toml:ro \
  -v ~/ccp-logs:/var/log/ccp \
  -e CCP_REPLACE_RULES=/etc/ccp-rules.toml \
  -e CCP_LOG_FILE=/var/log/ccp/conversations.log \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

## Full Example

A production-ready example with all features:

```sh
docker run -d \
  -p 18321:18321 \
  --name ccp \
  --restart unless-stopped \
  -v ~/.claude/.credentials.json:/root/.claude/.credentials.json:ro \
  -v /path/to/keys.txt:/etc/ccp-keys:ro \
  -v /path/to/rules.toml:/etc/ccp-rules.toml:ro \
  -v ~/ccp-logs:/var/log/ccp \
  -e CCP_API_KEYS_FILE=/etc/ccp-keys \
  -e CCP_REPLACE_RULES=/etc/ccp-rules.toml \
  -e CCP_LOG_FILE=/var/log/ccp/conversations.log \
  -e CCP_MAX_CONCURRENT=10 \
  ghcr.io/sonami-tech/claude-code-provider:dev
```

## Verify

```sh
# Check the container is running.
docker logs ccp

# Test the health endpoint.
curl http://localhost:18321/health

# Test a completion (use the API key from the logs).
curl http://localhost:18321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <key>" \
  -d '{"model": "haiku", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Build from Source

```sh
docker build -t claude-code-provider .
```
