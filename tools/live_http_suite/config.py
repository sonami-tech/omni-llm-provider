"""Suite configuration: base URL, model pins, dual-mode skip, timeouts."""

from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "http://127.0.0.1:18321"
DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_DUAL_MODE_MODEL = "grok-4.7"
DEFAULT_TIMEOUT_S = 60.0
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_MAX_TRANSPORT_RETRIES = 3
UNKNOWN_MODEL = "no-such-model-xyz"

# Dual-mode skip: checked BEFORE resolving the dual-mode pin.
DUAL_MODE_OFF_ENV = "OMNI_TEST_DUAL_MODE_OFF"
BASE_URL_ENV = "OMNI_BASE_URL"
CLAUDE_MODEL_ENV = "OMNI_TEST_CLAUDE_MODEL"
DUAL_MODE_MODEL_ENV = "OMNI_TEST_DUAL_MODE_MODEL"
RESPONSES_MODEL_ENV = "OMNI_TEST_RESPONSES_MODEL"
TIMEOUT_ENV = "OMNI_TEST_HTTP_TIMEOUT_S"


def _truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_or(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return value.strip()


@dataclass(frozen=True)
class SuiteConfig:
    """Immutable run configuration for the live HTTP suite."""

    base_url: str = DEFAULT_BASE_URL
    claude_model: str = DEFAULT_CLAUDE_MODEL
    dual_mode_model: str = DEFAULT_DUAL_MODE_MODEL
    responses_model: str = DEFAULT_CLAUDE_MODEL
    dual_mode_off: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES

    @property
    def base_url_normalized(self) -> str:
        return self.base_url.rstrip("/")

    @classmethod
    def from_env(
        cls,
        *,
        base_url: str | None = None,
        dual_mode_off: bool | None = None,
    ) -> "SuiteConfig":
        # Dual-mode skip must be decided before pin resolution so a missing
        # Grok model never becomes an inferred skip after a 4xx.
        off_flag = (
            dual_mode_off
            if dual_mode_off is not None
            else _truthy(os.environ.get(DUAL_MODE_OFF_ENV))
        )
        claude = env_or(CLAUDE_MODEL_ENV, DEFAULT_CLAUDE_MODEL)
        responses = env_or(RESPONSES_MODEL_ENV, claude)
        dual = env_or(DUAL_MODE_MODEL_ENV, DEFAULT_DUAL_MODE_MODEL)
        timeout_raw = os.environ.get(TIMEOUT_ENV)
        timeout_s = DEFAULT_TIMEOUT_S
        if timeout_raw and timeout_raw.strip():
            try:
                timeout_s = float(timeout_raw.strip())
            except ValueError as e:
                raise ValueError(
                    f"{TIMEOUT_ENV} must be a number of seconds, got {timeout_raw!r}"
                ) from e
            if timeout_s <= 0:
                raise ValueError(f"{TIMEOUT_ENV} must be > 0, got {timeout_s}")
        url = base_url if base_url is not None else env_or(BASE_URL_ENV, DEFAULT_BASE_URL)
        return cls(
            base_url=url,
            claude_model=claude,
            dual_mode_model=dual,
            responses_model=responses,
            dual_mode_off=off_flag,
            timeout_s=timeout_s,
        )
