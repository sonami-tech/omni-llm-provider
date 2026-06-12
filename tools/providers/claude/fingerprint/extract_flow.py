#!/usr/bin/env python3
"""Extract human-readable header + body summary from a mitmproxy flow file.

Usage: extract_flow.py <flow-file> [--full-body --body-bytes=N]

Writes a structured Markdown report to stdout. Used for verifying claude wire
fingerprint against Omni v2 captures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mitmproxy.io import FlowReader


def redact(value: str, length: int = 18) -> str:
    if len(value) <= length:
        return value
    return value[:length] + "...(redacted " + str(len(value) - length) + " chars)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("flow_file", type=Path)
    p.add_argument("--body-bytes", type=int, default=4000, help="max body bytes per request to print with --full-body")
    p.add_argument("--filter-host", default="api.anthropic.com")
    p.add_argument("--full-body", action="store_true", help="print redacted full JSON bodies; default is structural summary only")
    args = p.parse_args()

    with args.flow_file.open("rb") as f:
        reader = FlowReader(f)
        idx = 0
        for flow in reader.stream():
            req = flow.request
            if args.filter_host not in req.host:
                continue
            idx += 1
            print(f"\n## Flow #{idx} - {req.method} {req.url}")
            print(f"\n- HTTP version: {req.http_version}")
            print(f"- Host: `{req.host}`")
            print(f"- Path: `{req.path}`")
            print("\n### Request headers (in send order)\n")
            for name, value in req.headers.items(multi=True):
                lower = name.lower()
                if lower == "authorization":
                    printed = "<redacted>"
                elif lower == "x-api-key":
                    printed = "<redacted>"
                elif lower == "cookie":
                    printed = "<redacted>"
                else:
                    printed = value
                print(f"- `{name}: {printed}`")

            ct = req.headers.get("content-type", "")
            print(f"\n### Request body (content-type: {ct})\n")
            body = req.content or b""
            if "application/json" in ct.lower():
                try:
                    obj = json.loads(body.decode("utf-8"))
                    if args.full_body:
                        obj = redact_body(obj)
                        print("```json")
                        print(json.dumps(obj, indent=2)[: args.body_bytes])
                        print("```")
                    else:
                        print_body_summary(obj)
                except Exception as exc:
                    print(f"(json parse failed: {exc})")
                    if args.full_body:
                        print(f"```\n{body[:args.body_bytes]!r}\n```")
            else:
                if args.full_body:
                    print(f"```\n{body[:args.body_bytes]!r}\n```")
                else:
                    print(f"- Body bytes: {len(body)}")

            # Response headers (just status + content type for context)
            if flow.response:
                resp = flow.response
                print(f"\n### Response: HTTP {resp.status_code}")
                interesting = ("content-type", "anthropic-ratelimit-requests-remaining",
                               "anthropic-ratelimit-tokens-remaining", "request-id",
                               "anthropic-organization-id", "x-should-retry", "retry-after")
                for h in interesting:
                    v = resp.headers.get(h)
                    if v:
                        print(f"- `{h}: {v}`")

    return 0


def redact_body(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k.lower() in ("authorization", "api_key", "key"):
                out[k] = "<redacted>"
            else:
                out[k] = redact_body(v)
        return out
    if isinstance(obj, list):
        return [redact_body(x) for x in obj]
    return obj


def print_body_summary(obj: object) -> None:
    if not isinstance(obj, dict):
        print(f"- JSON top-level type: `{type(obj).__name__}`")
        return

    for key in ("model", "max_tokens", "temperature", "top_p", "stream"):
        if key in obj:
            print(f"- `{key}`: `{json.dumps(obj[key], separators=(',', ':'))}`")

    if "thinking" in obj:
        print(f"- `thinking`: `{json.dumps(obj['thinking'], sort_keys=True, separators=(',', ':'))}`")
    else:
        print("- `thinking`: absent")

    if "output_config" in obj:
        print(f"- `output_config`: `{json.dumps(obj['output_config'], sort_keys=True, separators=(',', ':'))}`")
    else:
        print("- `output_config`: absent")

    if "context_management" in obj:
        print(f"- `context_management`: `{json.dumps(obj['context_management'], sort_keys=True, separators=(',', ':'))}`")
    else:
        print("- `context_management`: absent")

    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        print(f"- `metadata` keys: `{','.join(sorted(metadata.keys()))}`")
        user_id = metadata.get("user_id")
        if isinstance(user_id, str):
            try:
                parsed_user = json.loads(user_id)
                if isinstance(parsed_user, dict):
                    print(f"- `metadata.user_id` keys: `{','.join(sorted(parsed_user.keys()))}`")
            except json.JSONDecodeError:
                print("- `metadata.user_id`: non-json string")
    else:
        print("- `metadata`: absent")

    system = obj.get("system")
    if isinstance(system, list):
        print(f"- `system` blocks: `{len(system)}`")
        for idx, block in enumerate(system[:4]):
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text.startswith("x-anthropic-billing-header:"):
                    print(f"  - system[{idx}]: billing header `{redact_billing_header(text)}`")
                else:
                    kind = block.get("type", type(block).__name__)
                    chars = len(text) if isinstance(text, str) else 0
                    cache_control = "cache_control" in block
                    print(f"  - system[{idx}]: type=`{kind}`, text_chars=`{chars}`, cache_control=`{cache_control}`")
    elif isinstance(system, str):
        print(f"- `system`: string, chars=`{len(system)}`")
    else:
        print("- `system`: absent")

    messages = obj.get("messages")
    if isinstance(messages, list):
        print(f"- `messages`: `{len(messages)}`")
        roles: list[str] = []
        for message in messages:
            if isinstance(message, dict):
                role = message.get("role")
                if isinstance(role, str):
                    roles.append(role)
        if roles:
            print(f"- `message_roles`: `{','.join(roles)}`")
    else:
        print("- `messages`: absent")

    tools = obj.get("tools")
    if isinstance(tools, list):
        print(f"- `tools`: `{len(tools)}`")
    else:
        print("- `tools`: absent")


def redact_billing_header(text: str) -> str:
    marker = "cch="
    if marker not in text:
        return text
    before, after = text.split(marker, 1)
    suffix = after[5:] if len(after) >= 5 else ""
    return before + marker + "<cch>" + suffix


if __name__ == "__main__":
    sys.exit(main())
