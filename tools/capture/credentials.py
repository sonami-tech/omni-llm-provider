"""Provider credential copy paths and refresh metadata forcing."""

from __future__ import annotations

import json
import os
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class CredentialError(RuntimeError):
    """Raised when required provider credentials cannot be staged."""


PAST_MS = 1_000
PAST_ISO = "2000-01-01T00:00:00.000000Z"


@dataclass(frozen=True)
class StagedCredentials:
    provider: str
    clean_home: Path
    clean_codex_home: Path | None
    copied_paths: tuple[Path, ...]
    source_paths: tuple[Path, ...]
    env_overrides: dict[str, str]


_PLACEHOLDER_CLEAN_HOME = Path("<clean-home>")
_PLACEHOLDER_CLEAN_CODEX_HOME = Path("<clean-codex-home>")


def _home_dir() -> Path:
    return Path(os.environ.get("HOME", str(Path.home())))


def _codex_home_dir() -> Path:
    if env := os.environ.get("CODEX_HOME", "").strip():
        return Path(env)
    return _home_dir() / ".codex"


def codex_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    return _codex_home_dir() / "config.toml"


def claude_source_path() -> Path:
    if env := os.environ.get("CLAUDE_CREDENTIALS_PATH", "").strip():
        return Path(env)
    return _home_dir() / ".claude" / ".credentials.json"


def grok_source_paths() -> list[Path]:
    if env := os.environ.get("XAI_CREDENTIALS_PATH", "").strip():
        return [Path(env)]
    home = _home_dir()
    paths: list[Path] = []
    static_path = home / ".xai" / ".credentials.json"
    if static_path.is_file():
        paths.append(static_path)
    cli_path = home / ".grok" / "auth.json"
    if cli_path.is_file():
        paths.append(cli_path)
    return paths


def codex_source_paths() -> list[Path]:
    home = _codex_home_dir()
    paths: list[Path] = []
    for name in ("config.toml", "auth.json"):
        src = home / name
        if src.is_file():
            paths.append(src)
    return paths


def _copy_file(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    old_umask = os.umask(0o077)
    try:
        shutil.copy2(src, dest)
    finally:
        os.umask(old_umask)
    os.chmod(dest, 0o600)
    return dest


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.chmod(path, 0o600)


def force_claude_expiry_stale(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise CredentialError(f"Claude credentials malformed at {path}")
    oauth = data.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        raise CredentialError(f"Claude credentials missing claudeAiOauth at {path}")
    oauth["expiresAt"] = PAST_MS
    _write_json(path, data)
    return True


def force_grok_expiry_stale(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise CredentialError(f"Grok credentials malformed at {path}")

    changed = False
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        if entry.get("auth_mode") == "oidc":
            entry["expires_at"] = PAST_ISO
            changed = True
    if changed:
        _write_json(path, data)
        return True

    # Static-key files have no expiry metadata to force stale.
    if "apiKey" in data or "xaiApiKey" in data:
        return False
    raise CredentialError(
        f"Grok credentials at {path} had no OIDC expiry metadata to force stale"
    )


def force_codex_expiry_stale(path: Path) -> bool:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise CredentialError(f"Codex auth malformed at {path}")

    changed = False
    tokens = data.get("tokens")
    if isinstance(tokens, dict):
        if "expires_at" in tokens:
            tokens["expires_at"] = PAST_ISO
            changed = True
        if "expires_in" in tokens:
            tokens["expires_in"] = 0
            changed = True

    for key in ("expires_at", "expiresAt", "token_expires_at"):
        if key in data:
            data[key] = PAST_ISO if isinstance(data[key], str) else PAST_MS
            changed = True

    if "last_refresh" in data:
        data["last_refresh"] = PAST_ISO
        changed = True

    if not changed:
        raise CredentialError(
            f"Codex auth at {path} had no expiry metadata to force stale"
        )
    _write_json(path, data)
    return True


def plan_credentials(*, provider: str, mode: str) -> StagedCredentials:
    """Build placeholder staged paths for dry-run without copying real credentials."""
    clean_home = _PLACEHOLDER_CLEAN_HOME
    clean_codex_home = (
        _PLACEHOLDER_CLEAN_CODEX_HOME if provider == "codex" else None
    )
    env_overrides: dict[str, str] = {"HOME": str(clean_home)}

    if provider == "claude":
        dest = clean_home / ".claude" / ".credentials.json"
        env_overrides["CLAUDE_CREDENTIALS_PATH"] = str(dest)
        return StagedCredentials(
            provider=provider,
            clean_home=clean_home,
            clean_codex_home=None,
            copied_paths=(dest,),
            source_paths=(Path("<claude-credentials>"),),
            env_overrides=env_overrides,
        )

    if provider == "grok":
        if mode == "refresh":
            dests = (clean_home / ".grok" / "auth.json",)
        else:
            dests = (
                clean_home / ".grok" / "auth.json",
                clean_home / ".xai" / ".credentials.json",
            )
            env_overrides["XAI_CREDENTIALS_PATH"] = str(dests[1])
        return StagedCredentials(
            provider=provider,
            clean_home=clean_home,
            clean_codex_home=None,
            copied_paths=dests,
            source_paths=(Path("<grok-credentials>"),),
            env_overrides=env_overrides,
        )

    if provider == "codex":
        if clean_codex_home is None:
            raise CredentialError("Codex capture requires a clean CODEX_HOME")
        dests = (
            clean_codex_home / "config.toml",
            clean_codex_home / "auth.json",
        )
        env_overrides["CODEX_HOME"] = str(clean_codex_home)
        env_overrides["CODEX_API_KEY"] = ""
        env_overrides["OPENAI_API_KEY"] = ""
        env_overrides["CODEX_ACCESS_TOKEN"] = ""
        return StagedCredentials(
            provider=provider,
            clean_home=clean_home,
            clean_codex_home=clean_codex_home,
            copied_paths=dests,
            source_paths=(Path("<codex-config>"), Path("<codex-auth>")),
            env_overrides=env_overrides,
        )

    raise CredentialError(f"Unknown provider: {provider}")


def stage_credentials(
    *,
    provider: str,
    clean_home: Path,
    clean_codex_home: Path | None,
    mode: str,
    source_home: Path | None = None,
    source_codex_home: Path | None = None,
) -> StagedCredentials:
    home = source_home or _home_dir()
    codex_home = source_codex_home or _codex_home_dir()
    copied: list[Path] = []
    sources: list[Path] = []
    env_overrides: dict[str, str] = {"HOME": str(clean_home)}

    if provider == "claude":
        src = (
            source_home / ".claude" / ".credentials.json"
            if source_home is not None
            else claude_source_path()
        )
        if not src.is_file():
            raise CredentialError(f"Claude credentials not found at {src}")
        dest = clean_home / ".claude" / ".credentials.json"
        _copy_file(src, dest)
        copied.append(dest)
        sources.append(src)
        env_overrides["CLAUDE_CREDENTIALS_PATH"] = str(dest)

        if mode == "refresh":
            force_claude_expiry_stale(dest)

    elif provider == "grok":
        if source_home is not None:
            static = source_home / ".xai" / ".credentials.json"
            cli = source_home / ".grok" / "auth.json"
            if mode == "refresh":
                grok_sources = [cli] if cli.is_file() else []
            else:
                grok_sources = []
                if static.is_file():
                    grok_sources.append(static)
                if cli.is_file():
                    grok_sources.append(cli)
        else:
            if mode == "refresh":
                cli = _home_dir() / ".grok" / "auth.json"
                grok_sources = [cli] if cli.is_file() else []
            else:
                grok_sources = grok_source_paths()

        if not grok_sources:
            if mode == "refresh":
                raise CredentialError(
                    "Grok refresh requires OIDC auth.json credentials; "
                    "static apiKey/xaiApiKey credentials cannot be force-staled."
                )
            raise CredentialError(
                "Grok credentials not found (checked $XAI_CREDENTIALS_PATH, "
                "~/.xai/.credentials.json, ~/.grok/auth.json)"
            )

        refresh_mutated = False
        staged_static_dest: Path | None = None
        for src in grok_sources:
            if src.name == ".credentials.json" or src.name == "credentials.json":
                dest = clean_home / ".xai" / ".credentials.json"
                staged_static_dest = dest
            elif src.name == "auth.json":
                dest = clean_home / ".grok" / "auth.json"
            else:
                dest = clean_home / src.name
                staged_static_dest = dest
            _copy_file(src, dest)
            copied.append(dest)
            sources.append(src)
            if mode == "refresh" and dest.name == "auth.json":
                if force_grok_expiry_stale(dest):
                    refresh_mutated = True

        if mode == "refresh" and not refresh_mutated:
            raise CredentialError(
                "Grok refresh requires OIDC auth.json credentials; "
                "static apiKey/xaiApiKey credentials cannot be force-staled."
            )

        if (
            mode != "refresh"
            and staged_static_dest is not None
            and staged_static_dest.is_file()
        ):
            env_overrides["XAI_CREDENTIALS_PATH"] = str(staged_static_dest)

    elif provider == "codex":
        if clean_codex_home is None:
            raise CredentialError("Codex capture requires a clean CODEX_HOME")

        codex_sources: list[Path] = []
        for name in ("config.toml", "auth.json"):
            src = codex_home / name
            if src.is_file():
                codex_sources.append(src)

        if not codex_sources:
            raise CredentialError(
                f"Codex config/auth not found under {codex_home} "
                "(expected config.toml and/or auth.json)"
            )
        if mode == "refresh" and not any(src.name == "auth.json" for src in codex_sources):
            raise CredentialError("Codex refresh requires auth.json credentials")

        for src in codex_sources:
            dest = clean_codex_home / src.name
            _copy_file(src, dest)
            copied.append(dest)
            sources.append(src)
            if mode == "refresh" and dest.name == "auth.json":
                force_codex_expiry_stale(dest)

        env_overrides["CODEX_HOME"] = str(clean_codex_home)
        env_overrides["CODEX_API_KEY"] = ""
        env_overrides["OPENAI_API_KEY"] = ""
        env_overrides["CODEX_ACCESS_TOKEN"] = ""

    else:
        raise CredentialError(f"Unknown provider: {provider}")

    return StagedCredentials(
        provider=provider,
        clean_home=clean_home,
        clean_codex_home=clean_codex_home,
        copied_paths=tuple(copied),
        source_paths=tuple(sources),
        env_overrides=env_overrides,
    )


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_CODEX_PROVIDER_ID = "openai"


def _toml_str(value: dict[str, Any], key: str) -> str | None:
    raw = value.get(key)
    return raw if isinstance(raw, str) else None


def _find_model_provider_table(
    providers: dict[str, Any], provider_id: str
) -> dict[str, Any] | None:
    direct = providers.get(provider_id)
    if isinstance(direct, dict):
        return direct
    provider_lower = provider_id.lower()
    for key, entry in providers.items():
        if isinstance(key, str) and key.lower() == provider_lower:
            if isinstance(entry, dict):
                return entry
    return None


def _codex_base_url_from_parsed(value: dict[str, Any]) -> str | None:
    provider_id = _toml_str(value, "model_provider") or DEFAULT_CODEX_PROVIDER_ID
    providers_raw = value.get("model_providers")
    providers = providers_raw if isinstance(providers_raw, dict) else None

    if provider_id == DEFAULT_CODEX_PROVIDER_ID and providers is not None:
        openai_entry = providers.get("openai")
        if isinstance(openai_entry, dict):
            # Mirror provider-codex: reserved [model_providers.openai] uses openai_base_url.
            return _toml_str(value, "openai_base_url") or DEFAULT_OPENAI_BASE_URL

    provider = (
        _find_model_provider_table(providers, provider_id) if providers is not None else None
    )
    built_in_openai = provider_id == DEFAULT_CODEX_PROVIDER_ID and provider is None
    if built_in_openai:
        return _toml_str(value, "openai_base_url") or DEFAULT_OPENAI_BASE_URL

    if provider is None:
        return None
    base_url = provider.get("base_url")
    return base_url if isinstance(base_url, str) and base_url else None


def _strip_toml_value(raw: str) -> str:
    value = raw.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _codex_base_url_raw_fallback(text: str) -> str | None:
    provider_id = DEFAULT_CODEX_PROVIDER_ID
    openai_base_url: str | None = None
    active_section: str | None = None
    section_base_url: str | None = None
    provider_sections: dict[str, str | None] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        section_match = re.match(r"^\[([^\]]+)\]$", stripped)
        if section_match:
            if active_section and active_section.startswith("model_providers."):
                provider_sections[active_section.split(".", 1)[1]] = section_base_url
            active_section = section_match.group(1)
            section_base_url = None
            continue
        key, _, raw_value = stripped.partition("=")
        key = key.strip()
        value = _strip_toml_value(raw_value)
        if not value:
            continue
        if key == "model_provider":
            provider_id = value
        elif key == "openai_base_url":
            openai_base_url = value
        elif key == "base_url" and active_section and active_section.startswith(
            "model_providers."
        ):
            section_base_url = value

    if active_section and active_section.startswith("model_providers."):
        provider_sections[active_section.split(".", 1)[1]] = section_base_url

    if provider_id == DEFAULT_CODEX_PROVIDER_ID:
        if "openai" in provider_sections:
            return openai_base_url or DEFAULT_OPENAI_BASE_URL
        if provider_id not in provider_sections and not any(
            key.lower() == provider_id.lower() for key in provider_sections
        ):
            return openai_base_url or DEFAULT_OPENAI_BASE_URL

    for key, base_url in provider_sections.items():
        if key == provider_id or key.lower() == provider_id.lower():
            return base_url
    return None


def codex_base_url(config_path: Path) -> str | None:
    if not config_path.is_file():
        return None
    text = config_path.read_text(encoding="utf-8")
    if not text.strip():
        return DEFAULT_OPENAI_BASE_URL
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return _codex_base_url_raw_fallback(text)
    if not isinstance(parsed, dict):
        return _codex_base_url_raw_fallback(text)
    return _codex_base_url_from_parsed(parsed)


def codex_base_url_host(config_path: Path) -> str | None:
    base_url = codex_base_url(config_path)
    if not base_url:
        return None
    return urlparse(base_url).hostname


def codex_primary_api_host(config_path: Path) -> str:
    return codex_base_url_host(config_path) or "api.openai.com"


def codex_expected_hosts(config_path: Path) -> list[str]:
    hosts = ["api.openai.com"]
    base_host = codex_base_url_host(config_path)
    if base_host and base_host not in hosts:
        hosts.append(base_host)
    return hosts
