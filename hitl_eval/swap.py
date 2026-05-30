"""Atomic symlink swap + qwen36 server reload."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx


def reload_qwen36_adapter(name: str = "qwen36-kicad", port: int = 9360) -> None:
    """Try the admin reload endpoint, fall back to launchctl kickstart."""
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
        subprocess.run(
            [
                "launchctl",
                "kickstart",
                "-k",
                f"gui/{os.getuid()}/cc.ailiance.qwen36-{port}",
            ],
            check=False,
            timeout=10,
        )
    except Exception:
        pass


def promote(*, new_lora_path: Path, curriculum_link: Path) -> dict:
    previous = (
        curriculum_link.resolve()
        if curriculum_link.exists() or curriculum_link.is_symlink()
        else None
    )
    new_tmp = curriculum_link.parent / (curriculum_link.name + ".new")
    if new_tmp.exists() or new_tmp.is_symlink():
        new_tmp.unlink()
    new_tmp.symlink_to(new_lora_path)
    if curriculum_link.is_symlink() or curriculum_link.exists():
        backup = curriculum_link.parent / (curriculum_link.name + ".prev")
        if backup.exists() or backup.is_symlink():
            backup.unlink()
        os.replace(curriculum_link, backup)
    os.replace(new_tmp, curriculum_link)
    reload_qwen36_adapter()
    return {
        "promoted_path": str(new_lora_path),
        "previous_path": str(previous) if previous else None,
    }


def rollback(*, curriculum_link: Path, previous_path: Path) -> None:
    if not previous_path.exists():
        raise RuntimeError(f"previous_path missing: {previous_path}")
    new_tmp = curriculum_link.parent / (curriculum_link.name + ".new")
    if new_tmp.exists() or new_tmp.is_symlink():
        new_tmp.unlink()
    new_tmp.symlink_to(previous_path)
    failed = curriculum_link.parent / (curriculum_link.name + ".failed")
    if failed.exists() or failed.is_symlink():
        failed.unlink()
    os.replace(curriculum_link, failed)
    os.replace(new_tmp, curriculum_link)
    reload_qwen36_adapter()
