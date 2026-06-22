"""Live capture orchestration."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from tools.capture.approvals import require_capture_approvals
from tools.capture.credentials import plan_credentials, stage_credentials
from tools.capture.extract import (
    ExtractionError,
    extract_flow_markdown,
    hosts_in_flow_file,
    require_mitmproxy_flow_reader,
    validate_expected_hosts,
)
from tools.capture.providers import (
    DEFAULT_PROMPT,
    build_mitmdump_command,
    build_provider_commands,
    build_provider_env,
    command_plan,
    expected_hosts_for_provider,
    pick_free_port,
    provider_command_stdin,
    required_hosts_for_validation,
)
from tools.capture.workdir import CaptureWorkdir, TmpfsError


class CaptureError(RuntimeError):
    pass


FLOW_FLUSH_GRACE_S = 0.5


def _wait_for_flow_flush(grace_s: float = FLOW_FLUSH_GRACE_S) -> None:
    """Brief pause so mitmdump can flush the flow file before validation."""
    if grace_s > 0:
        time.sleep(grace_s)


def _pgrep_available() -> bool:
    from shutil import which

    return which("pgrep") is not None


def _kill_tree(pid: int, sig: str) -> None:
    children = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    for child in children.stdout.split():
        if child.strip():
            _kill_tree(int(child), sig)
    subprocess.run(["kill", f"-{sig}", str(pid)], check=False, capture_output=True)


def _tree_pids(pid: int) -> list[int]:
    alive: list[int] = []
    probe = subprocess.run(["kill", "-0", str(pid)], check=False, capture_output=True)
    if probe.returncode == 0:
        alive.append(pid)
    children = subprocess.run(
        ["pgrep", "-P", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    for child in children.stdout.split():
        if child.strip():
            alive.extend(_tree_pids(int(child)))
    return alive


def _wait_for_port(port: int, mitm_pid: int, log_path: Path, timeout_s: float = 9.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _tree_pids(mitm_pid):
            log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
            raise CaptureError(f"mitmdump exited during startup. Log:\n{log}")
        probe = subprocess.run(
            ["bash", "-c", f"exec 3<>/dev/tcp/127.0.0.1/{port}"],
            check=False,
            capture_output=True,
        )
        if probe.returncode == 0:
            return
        time.sleep(0.3)
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    raise CaptureError(f"mitmdump did not accept connections on :{port}. Log:\n{log}")


def _stop_mitm(pid: int) -> None:
    if pid <= 0:
        return
    _kill_tree(pid, "TERM")
    for _ in range(15):
        if not _tree_pids(pid):
            return
        time.sleep(0.2)
    _kill_tree(pid, "KILL")
    for _ in range(10):
        if not _tree_pids(pid):
            return
        time.sleep(0.1)


_CAPTURE_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)


def _capture_signal_cleanup(work: CaptureWorkdir, mitm_pid: int) -> None:
    """Stop mitmdump before purging the workdir (signal-safe ordering)."""
    _stop_mitm(mitm_pid)
    work.remove_all()


def _install_capture_signal_handlers(
    work: CaptureWorkdir,
    mitm_pid: int,
) -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handler(signum: int, _frame) -> None:
        _capture_signal_cleanup(work, mitm_pid)
        signal.signal(signum, previous.get(signum, signal.SIG_DFL))
        os.kill(os.getpid(), signum)

    for sig in _CAPTURE_SIGNALS:
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_capture_signal_handlers(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def run_capture(
    *,
    provider: str,
    mode: str = "general",
    models: Sequence[str] = (),
    prompt: str = DEFAULT_PROMPT,
    dry_run: bool = False,
    live_flag: bool = False,
    refresh_flag: bool = False,
    keep_flow: bool = False,
    allow_tmpfs_fallback: bool = False,
    workdir: CaptureWorkdir | None = None,
) -> dict[str, object]:
    require_capture_approvals(
        mode=mode,
        dry_run=dry_run,
        live_flag=live_flag,
        refresh_flag=refresh_flag,
    )

    if dry_run:
        staged = plan_credentials(provider=provider, mode=mode)
        port = int(os.environ.get("OMNI_CAPTURE_PORT", "0").strip() or "0")
        return {
            "dry_run": True,
            "plan": command_plan(
                provider=provider,
                mode=mode,
                staged=staged,
                port=port,
                prompt=prompt,
                models=models,
            ),
        }

    work = workdir or CaptureWorkdir.create(
        provider=provider,
        keep_flow=keep_flow or os.environ.get("KEEP_FLOW", "").strip() in {"1", "true", "yes"},
        allow_tmpfs_fallback=allow_tmpfs_fallback,
    )

    try:
        staged = stage_credentials(
            provider=provider,
            clean_home=work.clean_home,
            clean_codex_home=work.clean_codex_home,
            mode=mode,
        )
        port = pick_free_port()

        if not _pgrep_available():
            raise CaptureError(
                "pgrep is required to reliably stop mitmdump before purging token-bearing flows."
            )

        from tools.capture.providers import mitm_ca_path

        if provider in {"grok", "codex"} and mitm_ca_path() is None:
            raise CaptureError(
                "mitmproxy CA cert not found at ~/.mitmproxy/mitmproxy-ca-cert.pem; "
                "refusing to run grok/codex capture without TLS interception."
            )

        require_mitmproxy_flow_reader()

        mitm_log = work.root / "mitm.log"
        mitm_cmd = build_mitmdump_command(
            provider=provider,
            flow_path=work.flow_path,
            port=port,
        )
        print(
            f"[capture] starting mitmdump on :{port} for provider={provider} mode={mode}",
            file=sys.stderr,
        )
        mitm_log_handle = mitm_log.open("w")
        try:
            mitm_proc = subprocess.Popen(
                mitm_cmd,
                stdout=mitm_log_handle,
                stderr=subprocess.STDOUT,
            )
        except Exception:
            mitm_log_handle.close()
            raise
        previous_handlers = _install_capture_signal_handlers(work, mitm_proc.pid)
        try:
            try:
                _wait_for_port(port, mitm_proc.pid, mitm_log)
                env = build_provider_env(
                    provider=provider,
                    staged=staged,
                    port=port,
                    path_value=os.environ.get("PATH", ""),
                )
                commands = build_provider_commands(
                    provider=provider,
                    staged=staged,
                    port=port,
                    prompt=prompt,
                    models=models,
                )
                for cmd in commands:
                    print(f"[capture] running: {' '.join(cmd)}", file=sys.stderr)
                    stdin = prompt if provider_command_stdin(cmd) else None
                    result = subprocess.run(
                        cmd,
                        cwd=str(work.clean_home),
                        env=env,
                        input=stdin,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        raise CaptureError(
                            f"Provider command failed with exit {result.returncode}: {' '.join(cmd)}"
                        )
                time.sleep(1)
            finally:
                _restore_capture_signal_handlers(previous_handlers)
                _stop_mitm(mitm_proc.pid)
        finally:
            mitm_log_handle.close()

        _wait_for_flow_flush()
        captured_hosts = hosts_in_flow_file(work.flow_path)
        expected = expected_hosts_for_provider(provider, staged)
        required = required_hosts_for_validation(provider, mode, staged)
        validate_expected_hosts(captured_hosts, required)

        with work.extract_path.open("w", encoding="utf-8") as out:
            extract_flow_markdown(
                work.flow_path,
                provider=provider,
                filter_hosts=expected,
                out=out,
                require_matches=True,
            )

        extract_text = work.extract_path.read_text(encoding="utf-8")

        result: dict[str, object] = {
            "provider": provider,
            "mode": mode,
            "extract_text": extract_text,
            "captured_hosts": sorted(captured_hosts),
            "expected_hosts": list(expected),
        }
        if work.keep_flow:
            result["workdir"] = str(work.root)
            result["flow_path"] = str(work.flow_path)
            result["extract_path"] = str(work.extract_path)

        print(
            f"[capture] done. workdir={work.root} extract={work.extract_path}",
            file=sys.stderr,
        )
        return result
    except (CaptureError, ExtractionError, TmpfsError):
        raise
    except Exception as exc:
        raise CaptureError(str(exc)) from exc
    finally:
        work.remove_all()
