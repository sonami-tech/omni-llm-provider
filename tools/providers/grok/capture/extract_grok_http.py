#!/usr/bin/env python3
"""Render sanitized Grok/xAI request captures from JSONL to Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SENSITIVE_HEADERS = {"authorization", "cookie", "x-api-key"}


def redact_header(name: str, value: str) -> str:
    if name.lower() not in SENSITIVE_HEADERS:
        return value
    return "<redacted>"


def redact_body(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key.lower() in {"authorization", "api_key", "apikey", "key", "token"}:
                out[key] = "<redacted>"
            else:
                out[key] = redact_body(item)
        return out
    if isinstance(value, list):
        return [redact_body(item) for item in value]
    return value


def load_body(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_jsonl", type=Path)
    parser.add_argument("--body-bytes", type=int, default=4000)
    args = parser.parse_args()

    count = 0
    with args.capture_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            count += 1
            rec = json.loads(line)
            method = rec.get("method", "GET")
            url = rec.get("url", rec.get("path", ""))
            print(f"\n## Request #{count} - {method} {url}")

            headers = rec.get("headers") or {}
            if headers:
                print("\n### Request Headers\n")
                for name, value in headers.items():
                    print(f"- `{name}: {redact_header(name, str(value))}`")

            if "body" in rec:
                body = redact_body(load_body(rec["body"]))
                print("\n### Request Body\n")
                if isinstance(body, str):
                    print("```text")
                    print(body[: args.body_bytes])
                    print("```")
                else:
                    print("```json")
                    print(json.dumps(body, indent=2)[: args.body_bytes])
                    print("```")

            if "status" in rec:
                print(f"\n### Response\n\n- HTTP status: `{rec['status']}`")
            response_headers = rec.get("response_headers") or {}
            for name, value in response_headers.items():
                print(f"- `{name}: {redact_header(name, str(value))}`")
            if "response_body" in rec:
                body = redact_body(load_body(rec["response_body"]))
                print("\n### Response Body\n")
                if isinstance(body, str):
                    print("```text")
                    print(body[: args.body_bytes])
                    print("```")
                else:
                    print("```json")
                    print(json.dumps(body, indent=2)[: args.body_bytes])
                    print("```")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(1)
