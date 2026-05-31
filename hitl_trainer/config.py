"""Config loaded from environment."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    minio_bucket: str
    token: str
    life_core_url: str
    shard_type: str
    shard_format: str
    base_model_path: Path
    curriculum_lora_path: Path
    output_root: Path
    target_lora: str
    epochs: int
    lr_reduced: float
    poll_seconds: int
    load_threshold: float
    idle_min_seconds: int
    state_path: Path
    idle_state_path: Path
    tmp_dir: Path
    allowed_hours: frozenset[int]


def load_config() -> Config:
    home = Path(os.path.expanduser("~"))
    return Config(
        minio_endpoint=os.environ["MINIO_ENDPOINT"],
        minio_access_key=os.environ["MINIO_ACCESS_KEY"],
        minio_secret_key=os.environ["MINIO_SECRET_KEY"],
        minio_bucket=os.environ.get("MINIO_BUCKET", "hitl-datasets"),
        token=os.environ["HITL_TRAINER_TOKEN"],
        life_core_url=os.environ.get("LIFE_CORE_URL", "https://api.saillant.cc"),
        shard_type=os.environ.get("SHARD_TYPE", "hardware"),
        shard_format=os.environ.get("SHARD_FORMAT", "dpo"),
        base_model_path=Path(os.environ["BASE_MODEL_PATH"]),
        curriculum_lora_path=Path(os.environ["CURRICULUM_LORA_PATH"]),
        output_root=Path(os.environ["OUTPUT_ROOT"]),
        target_lora=os.environ.get("TARGET_LORA", "qwen36-kicad-hitl"),
        epochs=int(os.environ.get("EPOCHS", "2")),
        lr_reduced=float(os.environ.get("LR_REDUCED", "1e-5")),
        poll_seconds=int(os.environ.get("POLL_SECONDS", "900")),
        load_threshold=float(os.environ.get("LOAD_THRESHOLD", "4.0")),
        idle_min_seconds=int(os.environ.get("IDLE_MIN_SECONDS", "1800")),
        state_path=home / ".config" / "hitl-trainer" / "state.json",
        idle_state_path=home / ".cache" / "hitl-trainer" / "idle.json",
        tmp_dir=Path("/tmp/hitl"),
        allowed_hours=frozenset({23, 0, 1, 2, 3, 4, 5, 6, 7}),
    )
