"""Redacted extraction from mitmproxy flows and JSONL captures."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO
from urllib.parse import urlparse

from tools.capture.redaction import (
    load_json_body,
    redact_billing_header,
    redact_body,
    redact_header,
    redact_text,
    redact_url,
    redact_value,
)


class ExtractionError(RuntimeError):
    """Raised when extraction produced no usable flows."""


def require_mitmproxy_flow_reader():
    """Import mitmproxy FlowReader or raise with an actionable error."""
    try:
        from mitmproxy.io import FlowReader
    except ImportError as exc:
        raise RuntimeError(
            "mitmproxy FlowReader is required for flow extraction. "
            "Install the mitmproxy Python package or use JSONL input."
        ) from exc
    return FlowReader


GENERIC_BODY_KEYS = (
    "model",
    "messages",
    "input",
    "output",
    "tool",
    "tools",
    "max_tokens",
    "temperature",
    "stream",
)

GENERIC_RESPONSE_HEADERS = (
    "content-type",
    "request-id",
    "x-request-id",
    "retry-after",
    "x-should-retry",
    "www-authenticate",
)


def print_claude_body_summary(obj: object, out: TextIO = sys.stdout) -> None:
    if not isinstance(obj, dict):
        print(f"- JSON top-level type: `{type(obj).__name__}`", file=out)
        return

    for key in ("model", "max_tokens", "temperature", "top_p", "stream"):
        if key in obj:
            print(
                f"- `{key}`: `{json.dumps(obj[key], separators=(',', ':'))}`",
                file=out,
            )

    if "thinking" in obj:
        print(
            f"- `thinking`: `{json.dumps(obj['thinking'], sort_keys=True, separators=(',', ':'))}`",
            file=out,
        )
    else:
        print("- `thinking`: absent", file=out)

    if "output_config" in obj:
        print(
            f"- `output_config`: `{json.dumps(obj['output_config'], sort_keys=True, separators=(',', ':'))}`",
            file=out,
        )
    else:
        print("- `output_config`: absent", file=out)

    if "context_management" in obj:
        print(
            f"- `context_management`: `{json.dumps(obj['context_management'], sort_keys=True, separators=(',', ':'))}`",
            file=out,
        )
    else:
        print("- `context_management`: absent", file=out)

    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        print(f"- `metadata` keys: `{','.join(sorted(metadata.keys()))}`", file=out)
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            try:
                parsed_user = json.loads(user_id)
                if isinstance(parsed_user, dict):
                    print(
                        f"- `metadata.user_id` keys: `{','.join(sorted(parsed_user.keys()))}`",
                        file=out,
                    )
            except json.JSONDecodeError:
                print("- `metadata.user_id`: non-json string", file=out)
    else:
        print("- `metadata`: absent", file=out)

    system = obj.get("system")
    if isinstance(system, list):
        print(f"- `system` blocks: `{len(system)}`", file=out)
        for idx, block in enumerate(system[:4]):
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.startswith("x-anthropic-billing-header:"):
                    print(
                        f"  - system[{idx}]: billing header `{redact_billing_header(text)}`",
                        file=out,
                    )
                else:
                    kind = block.get("type", type(block).__name__)
                    chars = len(text) if isinstance(text, str) else 0
                    cache_control = "cache_control" in block
                    print(
                        f"  - system[{idx}]: type=`{kind}`, text_chars=`{chars}`, "
                        f"cache_control=`{cache_control}`",
                        file=out,
                    )
    elif isinstance(system, str):
        print(f"- `system`: string, chars=`{len(system)}`", file=out)
    else:
        print("- `system`: absent", file=out)

    messages = obj.get("messages")
    if isinstance(messages, list):
        print(f"- `messages`: `{len(messages)}`", file=out)
        roles: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                if isinstance(role, str):
                    roles.append(role)
        if roles:
            print(f"- `message_roles`: `{','.join(roles)}`", file=out)
    else:
        print("- `messages`: absent", file=out)

    tools = obj.get("tools")
    if isinstance(tools, list):
        print(f"- `tools`: `{len(tools)}`", file=out)
    else:
        print("- `tools`: absent", file=out)


def _normalize_filter_hosts(
    *,
    provider: str | None,
    filter_host: str | None,
    filter_hosts: Sequence[str] | None,
    codex_config: Path | None = None,
    default_host: str | None = "api.anthropic.com",
) -> tuple[str, ...] | None:
    if filter_hosts:
        return tuple(filter_hosts)
    if filter_host:
        return (filter_host,)
    if provider == "codex":
        from tools.capture.credentials import codex_config_path, codex_expected_hosts

        return tuple(codex_expected_hosts(codex_config_path(codex_config)))
    if provider:
        from tools.capture.providers import PROVIDER_SPECS

        return PROVIDER_SPECS[provider].expected_hosts
    if default_host is not None:
        return (default_host,)
    return None


def _provider_uses_claude_summary(provider: str | None) -> bool:
    return provider in {None, "claude"}


def print_generic_body_summary(
    obj: object,
    out: TextIO = sys.stdout,
    *,
    body_bytes: int = 4000,
) -> None:
    if not isinstance(obj, dict):
        print(f"- JSON top-level type: `{type(obj).__name__}`", file=out)
        return

    print(f"- top-level keys: `{','.join(sorted(obj.keys()))}`", file=out)
    for key in GENERIC_BODY_KEYS:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, list):
            print(f"- `{key}`: list length `{len(value)}`", file=out)
        elif isinstance(value, dict):
            print(
                f"- `{key}`: object keys `{','.join(sorted(value.keys()))}`",
                file=out,
            )
        else:
            printed = json.dumps(value, separators=(",", ":"))
            if len(printed) > 120:
                printed = printed[:120] + "..."
            print(f"- `{key}`: `{printed}`", file=out)


def _print_response_section(
    *,
    status_code: int,
    headers: dict[str, str] | Any,
    body: bytes | str | None,
    provider: str | None,
    full_body: bool,
    body_bytes: int,
    out: TextIO,
) -> None:
    print(f"\n### Response: HTTP {status_code}", file=out)
    interesting = (
        GENERIC_RESPONSE_HEADERS
        if not _provider_uses_claude_summary(provider)
        else (
            "content-type",
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-tokens-remaining",
            "request-id",
            "anthropic-organization-id",
            "x-should-retry",
            "retry-after",
        )
    )
    for header in interesting:
        value = headers.get(header) if hasattr(headers, "get") else None
        if value:
            printed = redact_header(header, str(value))
            print(f"- `{header}: {printed}`", file=out)

    if body is None:
        return

    raw = body if isinstance(body, bytes) else str(body).encode("utf-8")
    if not raw:
        print("- response body: empty", file=out)
        return

    ct = str(headers.get("content-type", "") if hasattr(headers, "get") else "")
    print(f"\n### Response body (content-type: {ct})\n", file=out)
    if "application/json" in ct.lower():
        try:
            obj = redact_body(json.loads(raw.decode("utf-8")))
            print("```json", file=out)
            print(json.dumps(obj, indent=2)[:body_bytes], file=out)
            print("```", file=out)
        except Exception as exc:
            print(f"(json parse failed: {exc})", file=out)
            text = raw[:body_bytes].decode("utf-8", errors="replace")
            print(f"```\n{redact_text(text)}\n```", file=out)
    else:
        text = raw[:body_bytes].decode("utf-8", errors="replace")
        print(f"```\n{redact_text(text)}\n```", file=out)


def _host_matches(host: str, filter_hosts: Sequence[str]) -> bool:
    host_lower = host.lower()
    return any(host_lower == candidate.lower() for candidate in filter_hosts)


def extract_flow_markdown(
    flow_file: Path,
    *,
    provider: str | None = None,
    filter_host: str | None = None,
    filter_hosts: Sequence[str] | None = None,
    codex_config: Path | None = None,
    full_body: bool = False,
    body_bytes: int = 4000,
    out: TextIO = sys.stdout,
    require_matches: bool = False,
) -> int:
    hosts = _normalize_filter_hosts(
        provider=provider,
        filter_host=filter_host,
        filter_hosts=filter_hosts,
        codex_config=codex_config,
        default_host="api.anthropic.com",
    )
    FlowReader = require_mitmproxy_flow_reader()

    idx = 0
    with flow_file.open("rb") as handle:
        reader = FlowReader(handle)
        for flow in reader.stream():
            req = flow.request
            if hosts is not None and not _host_matches(req.host, hosts):
                continue
            idx += 1
            print(
                f"\n## Flow #{idx} - {req.method} {redact_url(req.url)}",
                file=out,
            )
            print(f"\n- HTTP version: {req.http_version}", file=out)
            print(f"- Host: `{req.host}`", file=out)
            print(f"- Path: `{redact_url(req.path)}`", file=out)
            print("\n### Request headers (in send order)\n", file=out)
            for name, value in req.headers.items(multi=True):
                printed = redact_header(name, value)
                print(f"- `{name}: {printed}`", file=out)

            ct = req.headers.get("content-type", "")
            print(f"\n### Request body (content-type: {ct})\n", file=out)
            body = req.content or b""
            if "application/json" in ct.lower():
                try:
                    obj = json.loads(body.decode("utf-8"))
                    if full_body:
                        obj = redact_body(obj)
                        print("```json", file=out)
                        print(json.dumps(obj, indent=2)[:body_bytes], file=out)
                        print("```", file=out)
                    elif _provider_uses_claude_summary(provider):
                        print_claude_body_summary(obj, out=out)
                    else:
                        print_generic_body_summary(obj, out=out, body_bytes=body_bytes)
                except Exception as exc:
                    print(f"(json parse failed: {exc})", file=out)
                    if full_body:
                        text = body[:body_bytes].decode("utf-8", errors="replace")
                        print(f"```\n{redact_text(text)}\n```", file=out)
            else:
                if full_body:
                    text = body[:body_bytes].decode("utf-8", errors="replace")
                    print(f"```\n{redact_text(text)}\n```", file=out)
                else:
                    print(f"- Body bytes: {len(body)}", file=out)

            if flow.response:
                resp = flow.response
                _print_response_section(
                    status_code=resp.status_code,
                    headers=resp.headers,
                    body=resp.content,
                    provider=provider,
                    full_body=full_body,
                    body_bytes=body_bytes,
                    out=out,
                )

    if require_matches and idx == 0:
        raise ExtractionError(
            f"Extraction produced zero flows matching hosts {list(hosts)}."
        )
    return 0


def extract_jsonl_markdown(
    capture_jsonl: Path,
    *,
    provider: str | None = None,
    filter_host: str | None = None,
    filter_hosts: Sequence[str] | None = None,
    codex_config: Path | None = None,
    body_bytes: int = 4000,
    out: TextIO = sys.stdout,
    require_matches: bool = False,
) -> int:
    hosts = _normalize_filter_hosts(
        provider=provider,
        filter_host=filter_host,
        filter_hosts=filter_hosts,
        codex_config=codex_config,
        default_host=None,
    )
    count = 0
    with capture_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            method = rec.get("method", "GET")
            url = rec.get("url", rec.get("path", ""))
            host = rec.get("host")
            if host is None and "://" in str(url):
                host = urlparse(str(url)).hostname
            if hosts is not None:
                if host is None:
                    continue
                if not _host_matches(str(host), hosts):
                    continue
            count += 1
            print(f"\n## Request #{count} - {method} {redact_url(str(url))}", file=out)

            headers = rec.get("headers") or {}
            if headers:
                print("\n### Request Headers\n", file=out)
                for name, value in headers.items():
                    print(f"- `{name}: {redact_header(name, str(value))}`", file=out)

            if "body" in rec:
                raw_body = load_json_body(rec["body"])
                print("\n### Request Body\n", file=out)
                if (
                    not _provider_uses_claude_summary(provider)
                    and isinstance(raw_body, dict)
                    and not isinstance(rec.get("body"), str)
                ):
                    print_generic_body_summary(raw_body, out=out, body_bytes=body_bytes)
                else:
                    body = redact_body(raw_body)
                    if isinstance(body, str):
                        print("```text", file=out)
                        print(redact_text(body)[:body_bytes], file=out)
                        print("```", file=out)
                    else:
                        print("```json", file=out)
                        print(json.dumps(body, indent=2)[:body_bytes], file=out)
                        print("```", file=out)

            if "status" in rec:
                print(f"\n### Response\n\n- HTTP status: `{rec['status']}`", file=out)
            response_headers = rec.get("response_headers") or {}
            for name, value in response_headers.items():
                print(f"- `{name}: {redact_header(name, str(value))}`", file=out)
            if "response_body" in rec:
                body = redact_body(load_json_body(rec["response_body"]))
                print("\n### Response Body\n", file=out)
                if isinstance(body, str):
                    print("```text", file=out)
                    print(redact_text(body)[:body_bytes], file=out)
                    print("```", file=out)
                else:
                    print("```json", file=out)
                    print(json.dumps(body, indent=2)[:body_bytes], file=out)
                    print("```", file=out)

    if require_matches and count == 0:
        raise ExtractionError(
            f"Extraction produced zero flows matching hosts {list(hosts)}."
        )
    return 0


def hosts_in_flow_file(flow_file: Path) -> set[str]:
    FlowReader = require_mitmproxy_flow_reader()

    hosts: set[str] = set()
    if not flow_file.exists() or flow_file.stat().st_size == 0:
        return hosts
    with flow_file.open("rb") as handle:
        reader = FlowReader(handle)
        for flow in reader.stream():
            hosts.add(flow.request.host)
    return hosts


def hosts_in_jsonl(jsonl_file: Path) -> set[str]:
    hosts: set[str] = set()
    if not jsonl_file.exists():
        return hosts
    with jsonl_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            url = rec.get("url", rec.get("path", ""))
            if "://" in str(url):
                from urllib.parse import urlparse

                host = urlparse(str(url)).hostname
                if host:
                    hosts.add(host)
            elif rec.get("host"):
                hosts.add(str(rec["host"]))
    return hosts


def validate_expected_hosts(
    captured_hosts: Iterable[str],
    required_hosts: Iterable[str],
) -> None:
    captured = {h.lower() for h in captured_hosts}
    required = [h.lower() for h in required_hosts]
    if not captured:
        raise RuntimeError(
            "Capture produced no flows. Refusing to claim success on empty capture."
        )
    missing = [host for host in required if host not in captured]
    if missing:
        raise RuntimeError(
            "Capture did not include required provider hosts "
            f"{list(required_hosts)}; missing {missing}; saw {sorted(captured)}."
        )
