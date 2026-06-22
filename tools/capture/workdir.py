"""Tmpfs-backed private workdirs and raw flow lifecycle."""

from __future__ import annotations

import atexit
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class TmpfsError(RuntimeError):
    """Raised when no RAM-backed tmpfs work base is available."""


def _fs_type(path: Path) -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["stat", "-f", "-c", "%T", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        return result.stdout.strip()
    except OSError:
        return None


def is_ramfs(path: Path) -> bool:
    fs_type = _fs_type(path)
    return fs_type in {"tmpfs", "ramfs"}


def find_tmpfs_base(*, allow_fallback: bool = False) -> Path:
    """Pick the first writable RAM-backed directory."""
    candidates: list[Path] = []
    runtime = os.environ.get("XDG_RUNTIME_DIR", "").strip()
    if runtime:
        candidates.append(Path(runtime))
    candidates.extend([Path("/dev/shm"), Path(f"/run/user/{os.getuid()}")])

    for cand in candidates:
        if not cand.is_dir():
            continue
        if not os.access(cand, os.W_OK):
            continue
        if is_ramfs(cand):
            return cand

    if allow_fallback:
        return Path(tempfile.gettempdir())

    raise TmpfsError(
        "No RAM-backed tmpfs directory is available (checked XDG_RUNTIME_DIR, "
        "/dev/shm, /run/user/UID). Refusing to write live credential-bearing flows "
        "to persistent disk."
    )


@dataclass
class CaptureWorkdir:
    base: Path
    root: Path
    flow_path: Path
    extract_path: Path
    clean_home: Path
    clean_codex_home: Path | None
    keep_flow: bool

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        keep_flow: bool = False,
        allow_tmpfs_fallback: bool = False,
    ) -> "CaptureWorkdir":
        base = find_tmpfs_base(allow_fallback=allow_tmpfs_fallback)
        prefix = f"omni-{provider}-capture."
        root = Path(tempfile.mkdtemp(prefix=prefix, dir=base))
        os.chmod(root, stat.S_IRWXU)

        clean_home = root / "clean-home"
        clean_home.mkdir(mode=0o700)
        clean_codex_home = root / "clean-codex-home"
        clean_codex_home.mkdir(mode=0o700)

        work = cls(
            base=base,
            root=root,
            flow_path=root / f"{provider}-capture.flow",
            extract_path=root / f"{provider}-capture-extract.md",
            clean_home=clean_home,
            clean_codex_home=clean_codex_home,
            keep_flow=keep_flow,
        )
        work._register_cleanup()
        return work

    def _register_cleanup(self) -> None:
        atexit.register(self.remove_all)

    def cleanup_flow(self) -> None:
        if self.keep_flow:
            self._warn_keep_flow()
            return
        self._purge_flow_file()

    def _storage_scope(self) -> str:
        if is_ramfs(self.root):
            return "tmpfs-only"
        return "persistent disk"

    def _warn_keep_flow(self) -> None:
        scope = self._storage_scope()
        if self.flow_path.exists():
            print(
                f"[capture] WARNING: KEEP_FLOW set; raw flow still at {self.flow_path} "
                f"({scope}, but contains LIVE credentials). Delete when done.",
                file=os.sys.stderr,
            )
        if self.root.exists():
            print(
                f"[capture] WARNING: KEEP_FLOW set; staged credential copies remain in "
                f"{self.root} ({scope}, but contains LIVE credentials). "
                "Delete when done.",
                file=os.sys.stderr,
            )

    def _purge_flow_file(self) -> None:
        if not self.flow_path.exists():
            return
        try:
            shutil.rmtree(self.flow_path) if self.flow_path.is_dir() else None
        except OSError:
            pass
        try:
            if shutil.which("shred"):
                import subprocess

                subprocess.run(
                    ["shred", "-u", str(self.flow_path)],
                    check=False,
                    capture_output=True,
                )
            else:
                self.flow_path.unlink(missing_ok=True)
        except OSError:
            self.flow_path.unlink(missing_ok=True)

    def remove_all(self) -> None:
        if self.keep_flow:
            self._warn_keep_flow()
            return
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


def ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.chmod(path, stat.S_IRWXU)