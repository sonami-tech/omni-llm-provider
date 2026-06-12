#!/usr/bin/env python3
"""Extract human-readable header + body summary from a mitmproxy flow file.

Usage: extract_flow.py <flow-file> [--body-bytes=N]

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
    p.add_argument("--body-bytes", type=int, default=4000, help="max body bytes per request to print")
    p.add_argument("--filter-host", default="api.anthropic.com")
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

            # Body
            ct = req.headers.get("content-type", "")
            print(f"\n### Request body (content-type: {ct})\n")
            body = req.content or b""
            if "application/json" in ct.lower():
                try:
                    obj = json.loads(body.decode("utf-8"))
                    obj = redact_body(obj)
                    print("```json")
                    print(json.dumps(obj, indent=2)[: args.body_bytes])
                    print("```")
                except Exception as exc:
                    print(f"(json parse failed: {exc})")
                    print(f"```\n{body[:args.body_bytes]!r}\n```")
            else:
                print(f"```\n{body[:args.body_bytes]!r}\n```")

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


if __name__ == "__main__":
    sys.exit(main())
