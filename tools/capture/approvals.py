"""Live capture and refresh approval gates."""

from __future__ import annotations

import os
from dataclasses import dataclass


LIVE_ENV = "OMNI_CAPTURE_LIVE"
REFRESH_ENV = "OMNI_CAPTURE_REFRESH"


class ApprovalError(RuntimeError):
    """Raised when a capture operation lacks required operator approval."""


@dataclass(frozen=True)
class ApprovalFlags:
    live_capture: bool = False
    refresh_capture: bool = False


def live_capture_approved(
    *,
    flag: bool = False,
    env: str = LIVE_ENV,
) -> bool:
    if flag:
        return True
    return os.environ.get(env, "").strip() in {"1", "true", "yes", "YES", "TRUE"}


def refresh_capture_approved(
    *,
    flag: bool = False,
    env: str = REFRESH_ENV,
) -> bool:
    if flag:
        return True
    return os.environ.get(env, "").strip() in {"1", "true", "yes", "YES", "TRUE"}


def require_capture_approvals(
    *,
    mode: str,
    dry_run: bool,
    live_flag: bool = False,
    refresh_flag: bool = False,
) -> ApprovalFlags:
    """Enforce approval gates unless this is a dry-run."""
    if dry_run:
        return ApprovalFlags()

    if not live_capture_approved(flag=live_flag):
        raise ApprovalError(
            f"Live capture requires --live-capture or ${LIVE_ENV}=1. "
            "This runs real provider CLIs and may spend quota."
        )

    if mode == "refresh" and not refresh_capture_approved(flag=refresh_flag):
        raise ApprovalError(
            f"Refresh capture requires --refresh-capture or ${REFRESH_ENV}=1 "
            "in addition to live-capture approval."
        )

    return ApprovalFlags(live_capture=True, refresh_capture=mode == "refresh")