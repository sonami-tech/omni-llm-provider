"""Fail-closed catalog listing for provider rebaseline.

A rebaseline overwrites the single live pin. If this pin's catalog source
cannot be read, stop. Do not keep the previous pin's model list.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


class CatalogError(RuntimeError):
    """Raised when a rebaseline cannot obtain this pin's catalog listing."""


_NO_CARRY = (
    "Do not carry a previous pin's catalog. Fix the catalog source and retry."
)


def _unique(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for model_id in ids:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        out.append(model_id)
    return out


def _fail(provider: str, detail: str) -> CatalogError:
    return CatalogError(
        f"{provider} rebaseline cannot obtain a model catalog: {detail} {_NO_CARRY}"
    )


def _claude_model(body: object) -> str | None:
    if not isinstance(body, dict):
        return None
    model = body.get("model")
    return model if isinstance(model, str) and model else None


def _require_claude_acceptance(status: int | None, model: str) -> None:
    if status is None or not 200 <= status < 300:
        detail = f"POST /v1/messages for {model} has no successful HTTP response"
        if status is not None:
            detail += f" (HTTP {status})"
        raise _fail("claude", detail + "; model acceptance is unproven")


def catalog_ids_from_jsonl(path: Path, *, provider: str) -> list[str]:
    """Pull catalog ids from a sanitized JSONL capture (no secrets required)."""
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            method = str(rec.get("method", "GET")).upper()
            url = str(rec.get("url", rec.get("path", "")))
            path_only = urlparse(url).path if "://" in url else url.split("?", 1)[0]
            if provider == "grok" and method == "GET" and path_only.rstrip("/").endswith(
                "/v1/models"
            ):
                body = rec.get("response_body", rec.get("body"))
                obj = body
                if isinstance(body, str):
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        obj = None
                if isinstance(obj, dict):
                    for model in obj.get("data") or []:
                        if isinstance(model, dict) and model.get("id"):
                            ids.append(str(model["id"]))
            if provider == "claude" and method == "POST" and "/v1/messages" in path_only:
                body = rec.get("body")
                obj = body
                if isinstance(body, str):
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        obj = None
                model = _claude_model(obj)
                if model:
                    status = rec.get("status")
                    _require_claude_acceptance(status if isinstance(status, int) else None, model)
                    ids.append(model)
    return _unique(ids)


def catalog_ids_from_flow(path: Path, *, provider: str) -> list[str]:
    """Pull catalog ids from a mitmproxy flow. Grok uses GET /v1/models."""
    from tools.capture.extract import require_mitmproxy_flow_reader

    FlowReader = require_mitmproxy_flow_reader()
    ids: list[str] = []
    with path.open("rb") as handle:
        for flow in FlowReader(handle).stream():
            req = flow.request
            path_only = req.path.split("?", 1)[0]
            if (
                provider == "grok"
                and req.method == "GET"
                and path_only.rstrip("/").endswith("/v1/models")
                and flow.response is not None
            ):
                raw = (flow.response.content or b"").replace(b"\x00", b"")
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    for model in obj.get("data") or []:
                        if isinstance(model, dict) and model.get("id"):
                            ids.append(str(model["id"]))
            if provider == "claude" and req.method == "POST" and "/v1/messages" in path_only:
                try:
                    obj = json.loads(req.content or b"")
                except json.JSONDecodeError:
                    continue
                model = _claude_model(obj)
                if model:
                    status = flow.response.status_code if flow.response is not None else None
                    _require_claude_acceptance(status, model)
                    ids.append(model)
    return _unique(ids)


def catalog_ids_from_codex_cli(*, bundled: bool = True) -> list[str]:
    """Read visibility=list slugs from `codex debug models`.

    Custom Responses `base_url` is not a reason to skip this. The bundled dump
    is the CLI's shipped catalog and does not need ChatGPT `/codex/models`.
    """
    cmd = ["codex", "debug", "models"]
    if bundled:
        cmd.append("--bundled")
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise _fail("codex", f"`codex debug models` could not run ({exc})") from exc
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()[-400:]
        raise _fail(
            "codex",
            f"`codex debug models` exited {result.returncode}"
            + (f": {err}" if err else ""),
        )
    try:
        obj = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise _fail("codex", "`codex debug models` did not emit JSON") from exc
    models = obj.get("models") if isinstance(obj, dict) else None
    if not isinstance(models, list):
        raise _fail("codex", "`codex debug models` JSON has no models list")
    ids: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        if model.get("visibility") != "list":
            continue
        slug = model.get("slug") or model.get("id")
        if slug:
            ids.append(str(slug))
    ids = _unique(ids)
    if not ids:
        raise _fail("codex", "`codex debug models` listed no visibility=list slugs")
    return ids


def require_rebaseline_catalog(
    provider: str,
    *,
    flow_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> list[str]:
    """Return this pin's catalog ids, or raise CatalogError.

    Sources:
    - grok: GET /v1/models in the capture
    - claude: model ids on captured POST /v1/messages
    - codex: `codex debug models --bundled` (not ChatGPT /codex/models)
    """
    if provider == "codex":
        return catalog_ids_from_codex_cli(bundled=True)

    if jsonl_path is not None:
        ids = catalog_ids_from_jsonl(jsonl_path, provider=provider)
    elif flow_path is not None:
        ids = catalog_ids_from_flow(flow_path, provider=provider)
    else:
        raise _fail(provider, "no capture file was provided")

    if not ids:
        if provider == "grok":
            detail = "capture has no GET /v1/models listing"
        else:
            detail = "capture has no POST /v1/messages model ids"
        raise _fail(provider, detail)
    return ids
