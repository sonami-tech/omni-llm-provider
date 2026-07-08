"""CLI for shared provider capture and extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.capture.approvals import ApprovalError
from tools.capture.extract import extract_flow_markdown, extract_jsonl_markdown
from tools.capture.extract import ExtractionError
from tools.capture.runner import CaptureError, run_capture
from tools.capture.workdir import TmpfsError


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omni-capture",
        description=(
            "Shared capture framework for Claude, Grok, and Codex. "
            "Live capture runs real provider CLIs and may spend quota."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Drive provider CLIs through a local MITM recorder")
    capture_sub = capture.add_subparsers(dest="capture_command", required=True)

    run = capture_sub.add_parser("run", help="Run a provider capture")
    run.add_argument("--provider", choices=["claude", "grok", "codex"], required=True)
    run.add_argument("--mode", choices=["general", "refresh"], default="general")
    run.add_argument(
        "--live-capture",
        action="store_true",
        help="Confirm live capture (or set OMNI_CAPTURE_LIVE=1)",
    )
    run.add_argument(
        "--refresh-capture",
        action="store_true",
        help="Confirm refresh capture (or set OMNI_CAPTURE_REFRESH=1)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned commands and env without running live capture",
    )
    run.add_argument(
        "--models",
        nargs="*",
        default=[],
        help=(
            "Optional model ids (Claude and Grok). Default includes one no-model "
            "capture, then one capture per listed model."
        ),
    )
    run.add_argument("--prompt", default="Say OK")
    run.add_argument(
        "--keep-flow",
        action="store_true",
        help="Retain raw flow on tmpfs only (still RAM-backed; contains live credentials)",
    )

    extract = sub.add_parser("extract", help="Render redacted Markdown from a capture file")
    extract_sub = extract.add_subparsers(dest="extract_command", required=True)

    flow = extract_sub.add_parser("flow", help="Extract from a mitmproxy .flow file")
    flow.add_argument("flow_file", type=Path)
    flow.add_argument("--provider", choices=["claude", "grok", "codex"])
    flow.add_argument("--filter-host")
    flow.add_argument(
        "--codex-config",
        type=Path,
        help="Codex config.toml for host filtering (default: $CODEX_HOME/config.toml or ~/.codex/config.toml)",
    )
    flow.add_argument("--full-body", action="store_true")
    flow.add_argument("--body-bytes", type=int, default=4000)
    flow.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero matching flows (default: fail closed)",
    )

    jsonl = extract_sub.add_parser("jsonl", help="Extract from sanitized JSONL")
    jsonl.add_argument("capture_jsonl", type=Path)
    jsonl.add_argument("--provider", choices=["claude", "grok", "codex"])
    jsonl.add_argument("--filter-host")
    jsonl.add_argument(
        "--codex-config",
        type=Path,
        help="Codex config.toml for host filtering (default: $CODEX_HOME/config.toml or ~/.codex/config.toml)",
    )
    jsonl.add_argument("--body-bytes", type=int, default=4000)
    jsonl.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow zero matching records (default: fail closed)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "capture" and args.capture_command == "run":
            result = run_capture(
                provider=args.provider,
                mode=args.mode,
                models=args.models,
                prompt=args.prompt,
                dry_run=args.dry_run,
                live_flag=args.live_capture,
                refresh_flag=args.refresh_capture,
                keep_flow=args.keep_flow,
            )
            if result.get("dry_run"):
                import json

                print(json.dumps(result["plan"], indent=2, sort_keys=True))
            else:
                extract_text = result.get("extract_text")
                if isinstance(extract_text, str) and extract_text:
                    print(extract_text, end="")
            return 0

        if args.command == "extract" and args.extract_command == "flow":
            return extract_flow_markdown(
                args.flow_file,
                provider=args.provider,
                filter_host=args.filter_host,
                codex_config=args.codex_config,
                full_body=args.full_body,
                body_bytes=args.body_bytes,
                require_matches=not args.allow_empty,
            )

        if args.command == "extract" and args.extract_command == "jsonl":
            return extract_jsonl_markdown(
                args.capture_jsonl,
                provider=args.provider,
                filter_host=args.filter_host,
                codex_config=args.codex_config,
                body_bytes=args.body_bytes,
                require_matches=not args.allow_empty,
            )
    except (ApprovalError, CaptureError, ExtractionError, TmpfsError, RuntimeError) as exc:
        print(f"[capture] FATAL: {exc}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        return 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
