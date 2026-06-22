#!/usr/bin/env python3
"""Thin compatibility wrapper around tools.capture flow extraction."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.capture.cli import main  # noqa: E402


def _compat_main() -> int:
    if len(sys.argv) < 2:
        return main(["extract", "flow", "--help"])
    flow_file = sys.argv[1]
    argv = ["extract", "flow", flow_file, *sys.argv[2:]]
    return main(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(_compat_main())
    except BrokenPipeError:
        raise SystemExit(1)