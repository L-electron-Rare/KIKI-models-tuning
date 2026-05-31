"""Daemon persisted state (last trained dataset key)."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class State:
    last_trained_key: str | None = None


def load_state(path: Path) -> State:
    try:
        data = json.loads(path.read_text())
        return State(last_trained_key=data.get("last_trained_key"))
    except FileNotFoundError:
        return State()


def save_state(state: State, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state)))
    os.replace(tmp, path)
