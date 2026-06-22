"""Shared redaction helpers for capture reports."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

SENSITIVE_HEADERS = {
    "authorization",
    "cookie",
    "set-cookie",
    "proxy-authorization",
    "x-api-key",
    "x-auth-token",
    "x-access-token",
}
SENSITIVE_BODY_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "key",
    "token",
    "access_token",
    "refresh_token",
    "id_token",
    "bearer_token",
    "client_secret",
    "secret",
    "session_token",
    "jwt",
    "xaiapikey",
    "openai_api_key",
    "codex_access_token",
}
_SENSITIVE_BODY_KEYS_NORMALIZED = {
    key.lower().replace("_", "").replace("-", "") for key in SENSITIVE_BODY_KEYS
}
_SENSITIVE_TEXT_RE = re.compile(
    r"(?i)(access[_-]?token|refresh[_-]?token|id[_-]?token|bearer[_-]?token|"
    r"client[_-]?secret|session[_-]?token|api[_-]?key|xai[_-]?api[_-]?key|"
    r"openai[_-]?api[_-]?key|codex[_-]?access[_-]?token|jwt|secret)"
    r"([=:\"]+)([^&\s\",}]+)"
)


def _normalize_body_key(key: str) -> str:
    return key.lower().replace("_", "").replace("-", "")


def _is_sensitive_body_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SENSITIVE_BODY_KEYS:
        return True
    return _normalize_body_key(key) in _SENSITIVE_BODY_KEYS_NORMALIZED


_SENSITIVE_QUERY_KEYS_EXTRA = frozenset({"code"})


def _is_sensitive_query_key(key: str) -> bool:
    if key.lower() in _SENSITIVE_QUERY_KEYS_EXTRA:
        return True
    return _is_sensitive_body_key(key)


def _quote_query_value(value: str, safe: str, encoding: str, errors: str) -> str:
    if value == "<redacted>":
        return value
    return quote(value, safe=safe, encoding=encoding, errors=errors)


def _redact_query_string(query: str) -> str:
    if not query:
        return query
    pairs = parse_qsl(query, keep_blank_values=True)
    redacted = [
        (key, "<redacted>" if _is_sensitive_query_key(key) else val)
        for key, val in pairs
    ]
    return urlencode(redacted, doseq=True, quote_via=_quote_query_value)


def redact_url(value: str) -> str:
    """Redact sensitive query parameter values in a URL or path+query string."""
    if not value or "?" not in value:
        return value
    if "://" in value:
        parsed = urlparse(value)
        return urlunparse(parsed._replace(query=_redact_query_string(parsed.query)))
    path, _, query = value.partition("?")
    return f"{path}?{_redact_query_string(query)}"


def redact_value(value: str, length: int = 18) -> str:
    if len(value) <= length:
        return value
    return value[:length] + f"...(redacted {len(value) - length} chars)"


def redact_header(name: str, value: str) -> str:
    if name.lower() in SENSITIVE_HEADERS:
        return "<redacted>"
    return value


def redact_body(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if _is_sensitive_body_key(key):
                out[key] = "<redacted>"
            else:
                out[key] = redact_body(item)
        return out
    if isinstance(value, list):
        return [redact_body(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(value: str) -> str:
    return _SENSITIVE_TEXT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>",
        value,
    )


def load_json_body(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def redact_billing_header(text: str) -> str:
    marker = "cch="
    if marker not in text:
        return text
    before, after = text.split(marker, 1)
    suffix = after[5:] if len(after) >= 5 else ""
    return before + marker + "<cch>" + suffix
