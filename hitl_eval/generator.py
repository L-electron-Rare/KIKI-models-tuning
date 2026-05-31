"""Subprocess wrapper around mlx_lm.generate."""
from __future__ import annotations

import subprocess
from pathlib import Path


def generate(
    adapter_path: Path,
    base: Path,
    prompt: str,
    max_tokens: int = 2048,
    bin_path: str = "mlx_lm.generate",
) -> str:
    cmd = [
        bin_path,
        "--model", str(base),
        "--adapter-path", str(adapter_path),
        "--max-tokens", str(max_tokens),
        "--prompt", prompt,
    ]
    return subprocess.check_output(
        cmd, text=True, stderr=subprocess.STDOUT, timeout=600
    )
