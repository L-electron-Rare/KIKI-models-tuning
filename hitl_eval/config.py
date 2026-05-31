"""Eval daemon config (env vars)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvalConfig:
    base_model_path: Path
    curriculum_link: Path
    hitl_output_root: Path
    target_lora: str
    golden_datasets_dir: Path
    domain: str
    token: str
    life_core_url: str
    poll_seconds: int
    mlx_lm_generate_bin: str
    max_tokens: int


def load_config() -> EvalConfig:
    home = Path(os.path.expanduser("~"))
    default_golden = home / "code/ailiance-models-tuning/eval/datasets"
    return EvalConfig(
        base_model_path=Path(os.environ["BASE_MODEL_PATH"]),
        curriculum_link=Path(os.environ["CURRICULUM_LORA_PATH"]),
        hitl_output_root=Path(os.environ["OUTPUT_ROOT"]),
        target_lora=os.environ.get("TARGET_LORA", "qwen36-kicad-hitl"),
        golden_datasets_dir=Path(
            os.environ.get("GOLDEN_DATASETS_DIR", str(default_golden))
        ),
        domain=os.environ.get("EVAL_DOMAIN", "kicad"),
        token=os.environ["HITL_TRAINER_TOKEN"],
        life_core_url=os.environ.get(
            "LIFE_CORE_URL", "https://api.saillant.cc"
        ),
        poll_seconds=int(os.environ.get("EVAL_POLL_SECONDS", "60")),
        mlx_lm_generate_bin=os.environ.get(
            "MLX_LM_GENERATE_BIN", "mlx_lm.generate"
        ),
        max_tokens=int(os.environ.get("EVAL_MAX_TOKENS", "2048")),
    )
