"""Per-provider capture command and proxy configuration."""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tools.capture.credentials import (
    StagedCredentials,
    codex_expected_hosts,
    codex_primary_api_host,
)


DEFAULT_PROMPT = "Say OK"


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    expected_hosts: tuple[str, ...]
    reverse_proxy: bool
    upstream_url: str | None
    filter_host: str


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "claude": ProviderSpec(
        name="claude",
        expected_hosts=("api.anthropic.com",),
        reverse_proxy=True,
        upstream_url="https://api.anthropic.com",
        filter_host="api.anthropic.com",
    ),
    "grok": ProviderSpec(
        name="grok",
        expected_hosts=(
            "cli-chat-proxy.grok.com",
            "api.x.ai",
            "auth.x.ai",
        ),
        reverse_proxy=False,
        upstream_url=None,
        filter_host="cli-chat-proxy.grok.com",
    ),
    "codex": ProviderSpec(
        name="codex",
        expected_hosts=("api.openai.com",),
        reverse_proxy=False,
        upstream_url=None,
        filter_host="api.openai.com",
    ),
}


def pick_free_port() -> int:
    if env := os.environ.get("OMNI_CAPTURE_PORT", "").strip():
        return int(env)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def mitm_ca_path() -> str | None:
    import glob

    candidates = glob.glob(os.path.expanduser("~/.mitmproxy/mitmproxy-ca-cert.pem"))
    return candidates[0] if candidates else None


def build_mitmdump_command(
    *,
    provider: str,
    flow_path: Path,
    port: int,
) -> list[str]:
    spec = PROVIDER_SPECS[provider]
    if shutil_which("mitmdump"):
        base = ["mitmdump"]
    else:
        base = ["uv", "tool", "run", "--from", "mitmproxy", "mitmdump"]

    cmd = [
        *base,
        "-w",
        str(flow_path),
        "--listen-host",
        "127.0.0.1",
        "--listen-port",
        str(port),
    ]
    if spec.reverse_proxy:
        cmd.extend(
            [
                "--mode",
                f"reverse:{spec.upstream_url}",
                "--set",
                "keep_host_header=false",
            ]
        )
    return cmd


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def build_provider_env(
    *,
    provider: str,
    staged: StagedCredentials,
    port: int,
    path_value: str,
) -> dict[str, str]:
    env: dict[str, str] = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(staged.clean_home),
    }
    env.update(staged.env_overrides)

    spec = PROVIDER_SPECS[provider]
    if spec.reverse_proxy:
        env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"
        env["CLAUDE_CODE_ENTRYPOINT"] = "sdk-cli"
        env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] = "1"
    else:
        proxy = f"http://127.0.0.1:{port}"
        env["HTTP_PROXY"] = proxy
        env["HTTPS_PROXY"] = proxy
        env["ALL_PROXY"] = proxy
        env["NO_PROXY"] = "localhost,127.0.0.1"
        ca = mitm_ca_path()
        if ca:
            env["SSL_CERT_FILE"] = ca
            env["REQUESTS_CA_BUNDLE"] = ca
            env["NODE_EXTRA_CA_CERTS"] = ca

    # Keep ambient API keys out of the isolated capture env.
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CODEX_API_KEY",
        "OPENAI_API_KEY",
        "CODEX_ACCESS_TOKEN",
        "XAI_API_KEY",
    ):
        env.pop(key, None)

    env["PATH"] = path_value
    return env


def provider_command_stdin(cmd: list[str]) -> bool:
    """True when the provider CLI reads its prompt from stdin."""
    return bool(cmd) and cmd[-1] == "-"


def build_provider_commands(
    *,
    provider: str,
    staged: StagedCredentials,
    port: int,
    prompt: str = DEFAULT_PROMPT,
    models: Sequence[str] = (),
) -> list[list[str]]:
    if provider == "claude":
        base = ["claude", "--print", "--no-session-persistence"]
        commands: list[list[str]] = [[*base, "--", prompt]]
        for model in models:
            commands.append([*base, "--model", model, "--", prompt])
        return commands

    if provider == "grok":
        return [
            [
                "grok",
                "--single",
                prompt,
                "--output-format",
                "json",
                "--always-approve",
                "--max-turns",
                "1",
                "--no-memory",
            ]
        ]

    if provider == "codex":
        return [
            [
                "codex",
                "exec",
                "-c",
                "mcp_servers={}",
                "--skip-git-repo-check",
                "--ephemeral",
                "-",
            ]
        ]

    raise ValueError(f"Unknown provider: {provider}")


def expected_hosts_for_provider(
    provider: str,
    staged: StagedCredentials,
) -> tuple[str, ...]:
    spec = PROVIDER_SPECS[provider]
    if provider == "codex" and staged.clean_codex_home is not None:
        config = staged.clean_codex_home / "config.toml"
        return tuple(codex_expected_hosts(config))
    return spec.expected_hosts


def required_hosts_for_validation(
    provider: str,
    mode: str,
    staged: StagedCredentials,
) -> tuple[str, ...]:
    spec = PROVIDER_SPECS[provider]
    if provider == "codex" and staged.clean_codex_home is not None:
        config = staged.clean_codex_home / "config.toml"
        return (codex_primary_api_host(config),)
    if provider == "grok":
        if mode == "refresh":
            return ("cli-chat-proxy.grok.com", "auth.x.ai")
        return (spec.filter_host,)
    return (spec.filter_host,)


def command_plan(
    *,
    provider: str,
    mode: str,
    staged: StagedCredentials,
    port: int,
    prompt: str = DEFAULT_PROMPT,
    models: Sequence[str] = (),
) -> dict[str, object]:
    commands = build_provider_commands(
        provider=provider,
        staged=staged,
        port=port,
        prompt=prompt,
        models=models,
    )
    env = build_provider_env(
        provider=provider,
        staged=staged,
        port=port,
        path_value=os.environ.get("PATH", ""),
    )
    return {
        "provider": provider,
        "mode": mode,
        "port": port,
        "cwd": str(staged.clean_home),
        "env": env,
        "mitmdump": build_mitmdump_command(
            provider=provider,
            flow_path=Path("<flow>"),
            port=port,
        ),
        "commands": commands,
        "expected_hosts": list(expected_hosts_for_provider(provider, staged)),
    }