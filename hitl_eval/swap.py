"""Atomic symlink swap + qwen36 server reload."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx


def _kill_qwen36_process(port: int) -> bool:
    """Find and SIGTERM the qwen36 server process listening on ``port``.

    Returns True if at least one process was signalled. The plist runs
    with ``KeepAlive=true`` so launchd restarts it within seconds, which
    forces a fresh adapter load from disk (= the new symlink target).
    """
    try:
        proc = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    if proc.returncode != 0 or not proc.stdout.strip():
        return False
    pids = [int(p) for p in proc.stdout.strip().split() if p.strip().isdigit()]
    if not pids:
        return False
    for pid in pids:
        try:
            os.kill(pid, 15)  # SIGTERM
        except Exception:
            pass
    return True


def reload_qwen36_adapter(name: str = "qwen36-kicad", port: int = 9360) -> None:
    """Reload the qwen36 multi-LoRA server so it picks up the new symlink.

    Cascade:
      1. ``POST :{port}/admin/reload_adapter`` if the endpoint exists.
      2. ``launchctl kickstart`` (works only inside a GUI session — fails
         silently over plain SSH due to launchd domain restrictions).
      3. Send ``SIGTERM`` to the process listening on ``port`` so the
         launchd ``KeepAlive`` machinery restarts it.

    Step 3 is the reliable fallback when SSH-based launchctl is denied.
    """
    try:
        with httpx.Client(timeout=10) as c:
            r = c.post(
                f"http://localhost:{port}/admin/reload_adapter",
                json={"name": name},
            )
            if r.status_code == 200:
                return
    except Exception:
        pass
    try:
        rc = subprocess.run(
            [
                "launchctl",
                "kickstart",
                "-k",
                f"gui/{os.getuid()}/cc.ailiance.qwen36-{port}",
            ],
            check=False,
            timeout=10,
        )
        if rc.returncode == 0:
            return
    except Exception:
        pass
    _kill_qwen36_process(port)


def _safe_unlink_or_rmtree(p: Path) -> None:
    """Remove a path whether it's a symlink, file, or directory."""
    if p.is_symlink() or p.is_file():
        p.unlink()
    elif p.is_dir():
        import shutil
        shutil.rmtree(p)


def promote(*, new_lora_path: Path, curriculum_link: Path) -> dict:
    backup = curriculum_link.parent / (curriculum_link.name + ".prev")
    new_tmp = curriculum_link.parent / (curriculum_link.name + ".new")

    if new_tmp.is_symlink() or new_tmp.exists():
        _safe_unlink_or_rmtree(new_tmp)
    new_tmp.symlink_to(new_lora_path)

    previous_path: Path | None = None
    if curriculum_link.is_symlink() or curriculum_link.exists():
        if backup.is_symlink() or backup.exists():
            _safe_unlink_or_rmtree(backup)
        os.replace(curriculum_link, backup)
        # After the rename, `.prev` IS the previous content (dir or symlink).
        previous_path = backup

    os.replace(new_tmp, curriculum_link)
    reload_qwen36_adapter()
    return {
        "promoted_path": str(new_lora_path),
        "previous_path": str(previous_path) if previous_path else None,
    }


def rollback(*, curriculum_link: Path, previous_path: Path) -> None:
    if not previous_path.exists():
        raise RuntimeError(f"previous_path missing: {previous_path}")
    if previous_path.resolve() == curriculum_link.resolve():
        raise RuntimeError(
            f"refusing self-rollback: previous_path resolves to curriculum_link "
            f"({previous_path} -> {curriculum_link})"
        )
    new_tmp = curriculum_link.parent / (curriculum_link.name + ".new")
    if new_tmp.is_symlink() or new_tmp.exists():
        _safe_unlink_or_rmtree(new_tmp)
    new_tmp.symlink_to(previous_path)
    failed = curriculum_link.parent / (curriculum_link.name + ".failed")
    if failed.is_symlink() or failed.exists():
        _safe_unlink_or_rmtree(failed)
    os.replace(curriculum_link, failed)
    os.replace(new_tmp, curriculum_link)
    reload_qwen36_adapter()
