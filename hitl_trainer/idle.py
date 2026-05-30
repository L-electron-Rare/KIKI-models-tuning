"""Idle detection: low load avg AND within allowed hours, for N seconds continuous."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def is_idle(
    *,
    load_threshold: float,
    allowed_hours: Iterable[int],
    idle_min_seconds: int,
    state_path: Path,
) -> bool:
    now = datetime.now()
    if now.hour not in set(allowed_hours):
        return False
    load = os.getloadavg()[0]
    st = _read(state_path)
    if load < load_threshold:
        if "idle_since" not in st:
            st["idle_since"] = now.isoformat()
            _write(state_path, st)
            return False
        idle_since = datetime.fromisoformat(st["idle_since"])
        return (now - idle_since).total_seconds() >= idle_min_seconds
    st.pop("idle_since", None)
    _write(state_path, st)
    return False
